import copy
import random
import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict

from ..util.misc import NestedTensor
from ..util.box_ops import box_cxcywh_to_xyxy
from .deformable_detr import DeformableDETR
from .position_encoding import build_position_encoding, TemporalEmbeddingLearned
from .transformer import TransformerDecoderLayer, TransformerDecoder, TransformerEncoderLayer, TransformerEncoder

class DeformableDETRVsgg(DeformableDETR):
    def __init__(self, args, detr_kwargs, matcher):
        DeformableDETR.__init__(self, **detr_kwargs)
        assert args.track_query_propagation_strategy == 'consistent_pairing'

        self.args = args
        self._matcher = matcher
        self._tracking = False

        if args.tracking_token_propagation:
            self.pos_propagation_mlp = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
            )
            self.tgt_propagation_mlp = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
            )

        if args.hoi_detection:
            if args.freeze_detr:
                for p in self.parameters():
                    p.requires_grad_(False)
            else:
                for n, p in self.named_parameters():
                    if 'backbone.0.body' in n:
                        p.requires_grad_(False)

            # instance representation
            if self.args.hoi_instance_fuse_spatial_and_semantic_feat:
                self.spatial_dim, self.semantic_dim = 128, 128
                self.spatial_embed = nn.Linear(4, self.spatial_dim)
                self.semantic_embed =  nn.Embedding(args.num_classes+1, self.semantic_dim)
                self.instance_representation_fuse = nn.Sequential(
                    nn.Linear(self.args.hidden_dim+self.spatial_dim+self.semantic_dim, self.args.hidden_dim), nn.ReLU()
                )

            # for rel/interactions prediction
            rel_rep_dim = self.args.hidden_dim * 2
            self.relation_proposal_mlp = nn.Sequential(
                nn.Linear(rel_rep_dim, rel_rep_dim // 2), nn.ReLU(),
                nn.Linear(rel_rep_dim // 2, 1)
            )
            self.rel_query_pre_proj = nn.Linear(rel_rep_dim, self.args.hidden_dim)

            rel_dec_hidden_dim = self.args.hidden_dim
            self.memory_input_proj = nn.Conv2d(2048, self.args.hidden_dim, kernel_size=1)
            self.rel_memory_pos = build_position_encoding(args)

            decoder_layer = TransformerDecoderLayer(d_model=rel_dec_hidden_dim, nhead=8)
            decoder_norm = nn.LayerNorm(rel_dec_hidden_dim)
            self.interaction_decoder = TransformerDecoder(decoder_layer, None, self.args.hoi_dec_layers, decoder_norm, return_intermediate=True)
            self.relation_embed = nn.Linear(rel_dec_hidden_dim, self.args.num_relations)

            if self.args.hoi_use_temporal_dynamics:
                encoder_layer = TransformerEncoderLayer(d_model=self.args.hidden_dim, nhead=4)
                self.temporal_dynamic_encoder = TransformerEncoder(encoder_layer, num_layers=2)
                self.temporal_position_encoding = TemporalEmbeddingLearned(self.args.hoi_use_temporal_dynamics_prev_length+1)

            if self.args.hoi_oracle_mode:
                if self.args.hoi_oracle_mode_use_instant_trajectory:
                    self.traj_feat_dim = 4 * 24 * 2 # 4* frame_num * (subj+obj)
                    self.trajectory_feature_fc = nn.Linear(self.traj_feat_dim, self.traj_feat_dim)
                    self.fuse_trajectory_feature_mlp = nn.Sequential(
                        nn.Linear(self.args.hidden_dim + self.traj_feat_dim, self.args.hidden_dim), nn.ReLU(),
                        nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
                    )
                if self.args.hoi_oracle_mode_use_roialign_union_feat:
                    self.fpn = torchvision.ops.feature_pyramid_network.FeaturePyramidNetwork(in_channels_list=[256, 512, 1024, 2048], out_channels=256)
                    self.box_pooler = torchvision.ops.MultiScaleRoIAlign(['0', '1', '2', '3'], 7, sampling_ratio=2)
                    self.box_pool_fc = nn.Sequential(nn.Linear(256*7*7, self.args.hidden_dim), nn.ReLU())
                    self.union_pool_fc = nn.Linear(self.args.hidden_dim*3, self.args.hidden_dim*2)

    def train(self, mode: bool = True):
        """Sets the module in train mode."""
        self._tracking = False
        return super().train(mode)

    def tracking(self):
        """Sets the module in tracking mode."""
        self.eval()
        self._tracking = True

    # online inference
    def online_foward(self, samples: NestedTensor, targets: list = None):
        assert len(targets) == 1
        if self.args.tracking_token_propagation and 'track_query_hs_embeds' in targets[0]:
            targets[0]['track_query_hs_embeds_pos_mapping'] = self.pos_propagation_mlp(targets[0]['track_query_hs_embeds'])
            targets[0]['track_query_hs_embeds_tgt_mapping'] = self.tgt_propagation_mlp(targets[0]['track_query_hs_embeds'])

        out, out_targets, features, memory, hs = super().forward(samples, targets)
        if self.args.hoi_detection:
            if self.args.hoi_oracle_mode: # set Oracle mode (given GT objects)
                match_res = self._matcher(out, targets, only_match_by_bbox=self.args.hoi_oracle_mode_only_given_bbox)[0]
                # match_res = self._matcher(out, targets)[0]
                if not self.args.hoi_oracle_mode_only_given_bbox: # PredCls mode, otherwise SGCls
                    out['pred_logits'][0, :, -1] = 1e3 # BG class
                    out['pred_logits'][0, match_res[0], targets[0]['labels'][match_res[1]]] = 1e6
                out['pred_boxes'][0, match_res[0]] = targets[0]['boxes'][match_res[1]]
                out['match_res'] = match_res
            out = self.hoi_forward(out, features, targets[0], image_size=samples.tensors.shape[-2:])
        return out, out_targets, features, memory, hs

    def forward(self, samples: NestedTensor, targets: list = None):
        if self._tracking:
            return self.online_foward(samples, targets)

        outs = []
        for frame_id, frame_target in enumerate(targets):
            frame_image = samples.tensors[frame_id]
            out, _, features_all, _, _ = super().forward([frame_image], [frame_target])

            if self.args.hoi_detection:
                if self.args.hoi_oracle_mode: # set Oracle mode (given GT objects)
                    match_res = self._matcher(out, [frame_target], only_match_by_bbox=self.args.hoi_oracle_mode_only_given_bbox)[0]
                    # match_res = self._matcher(out, [frame_target])[0]
                    if not self.args.hoi_oracle_mode_only_given_bbox: # PredCls mode, otherwise SGCls
                        out['pred_logits'][0, :, -1] = 1e3
                        out['pred_logits'][0, match_res[0], frame_target['labels'][match_res[1]]] = 1e6
                    out['pred_boxes'][0, match_res[0]] = frame_target['boxes'][match_res[1]]
                    out['match_res'] = match_res

                    if 'aux_outputs' in outs: # aux outputs
                        for o in outs['aux_outputs']:
                            if not self.args.hoi_oracle_mode_only_given_bbox:
                                o['pred_logits'][0, :, -1] = 1e3
                                o['pred_logits'][0, match_res[0], frame_target['labels'][match_res[1]]] = 1e6
                            o['pred_boxes'][0, match_res[0]] = frame_target['boxes'][match_res[1]]
                out = self.hoi_forward(out, features_all, frame_target, image_size=frame_image.shape[-2:])
            outs.append(out)

            # propagate to the next frame
            if frame_id < len(targets)-1:
                prev_out = out
                target = targets[frame_id+1] # target for the next frame
                if 'match_res' in prev_out:
                    prev_out_ind, prev_target_ind = prev_out['match_res']
                else:
                    prev_indices = self._matcher(prev_out, [frame_target])
                    prev_out_ind, prev_target_ind = prev_indices[0]

                ## transfer matching from prev to current
                # detected prev frame tracks
                prev_track_ids = frame_target['track_ids'][prev_target_ind]

                # match track ids between frames
                target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(target['track_ids'])
                target_ind_matching = target_ind_match_matrix.any(dim=1)
                target_ind_matched_idx = target_ind_match_matrix.nonzero()[:, 1]

                # index of prev frame detection in current frame box list
                target['track_query_match_ids'] = target_ind_matched_idx
                device = target['track_query_match_ids'].device
                tracked_qids = prev_out_ind[target_ind_matching]

                # match mask to next frame
                track_queries_match_mask = torch.zeros(prev_out['hs_embed'][0].shape[0]).float()
                track_queries_match_mask[tracked_qids] = 1 # tracked in current frame
                track_queries_match_mask[prev_out_ind[~target_ind_matching]] = -1 # disappeared in current frame
                target['track_queries_match_mask'] = track_queries_match_mask.to(device)

                if self.args.tracking_token_propagation:
                    target['track_query_hs_embeds'] = prev_out['hs_embed'][0]
                    target['track_query_hs_embeds_pos_mapping'] = self.pos_propagation_mlp(prev_out['hs_embed'][0])
                    target['track_query_hs_embeds_tgt_mapping'] = self.tgt_propagation_mlp(prev_out['hs_embed'][0])
                    target['track_query_boxes'] = prev_out['pred_boxes'][0].detach()
                    if random.random() < self.args.token_propagation_sample_rate:
                        target['track_token_propagation_mask'] = (prev_out['pred_logits'][0].softmax(-1)[:,:-1].max(-1)[0].detach() > 0.7).float()
                    else:
                        target['track_token_propagation_mask'] = (target['track_queries_match_mask'] != 0).float()

                    if self.args.hoi_detection and self.args.hoi_use_temporal_dynamics:
                        if 'temporal_dynamics_feature_bank' in frame_target:
                            target['temporal_dynamics_feature_bank'] = torch.cat((prev_out['hs_embed'], frame_target['temporal_dynamics_feature_bank']), dim=0)[:self.args.hoi_use_temporal_dynamics_prev_length]
                            target['temporal_dynamics_feature_mask'] = torch.cat(((target['track_token_propagation_mask']==0).unsqueeze(0), frame_target['temporal_dynamics_feature_mask']), dim=0)[:self.args.hoi_use_temporal_dynamics_prev_length]
                        else:
                            target['temporal_dynamics_feature_bank'] = prev_out['hs_embed']
                            target['temporal_dynamics_feature_mask'] = (target['track_token_propagation_mask']==0).unsqueeze(0)

        ## compose outputs in batched format
        outputs = {key: torch.cat([o[key] for o in outs], dim=0) for key in ['pred_logits', 'pred_boxes', 'hs_embed']}
        if 'aux_outputs' in outs[0]:
            outputs['aux_outputs'] = []
            for l in range(len(outs[0]['aux_outputs'])):
                outputs['aux_outputs'].append(
                    {key: torch.cat([o['aux_outputs'][l][key] for o in outs], dim=0) for key in ['pred_logits', 'pred_boxes']}
                )

        if self.args.hoi_detection:
            outputs.update({key: [o[key][0] for o in outs] for key in ['pred_rel_pairs', 'pred_relations', 'pred_relation_exists']})
            if 'relation_aux_outputs' in outs[0]:
                outputs['relation_aux_outputs'] = []
                for l in range(len(outs[0]['relation_aux_outputs'])):
                    outputs['relation_aux_outputs'].append(
                        {key: [o['relation_aux_outputs'][l][key][0] for o in outs] for key in ['pred_relations']}
                    )

        return outputs, targets, None, None, None

    def hoi_forward(self, out, features_all, frame_target=None, image_size=None):
        assert len(features_all[-1].tensors) == 1 # frame-wise forward

        # memory input for relation transformer decoder
        memory_input_feature, memory_input_mask = features_all[-1].decompose()
        memory_pos = self.rel_memory_pos(features_all[-1])
        memory_input = self.memory_input_proj(memory_input_feature)

        # instance representations
        if self.args.hoi_use_temporal_dynamics:
            if (frame_target is not None) and 'temporal_dynamics_feature_bank' in frame_target:
                src = torch.cat((out['hs_embed'], frame_target['temporal_dynamics_feature_bank']), dim=0)
                att_mask = torch.cat((torch.zeros_like(frame_target['temporal_dynamics_feature_mask'])[:1], frame_target['temporal_dynamics_feature_mask']), dim=0).permute(1,0)
                pos_idx = torch.arange(len(src)).to(src.device).unsqueeze(-1)
            else:
                src = out['hs_embed']
                att_mask = torch.zeros((src.shape[1], 1), dtype=torch.bool, device=src.device)
                pos_idx = torch.arange(1).to(src.device).unsqueeze(-1)

            instance_representations = self.temporal_dynamic_encoder(
                src=src,
                src_key_padding_mask=att_mask,
                pos=self.temporal_position_encoding(pos_idx),
            )[0]
        else:
            instance_representations = out['hs_embed'][0]

        if self.training:
            if 'match_res' in out:
                ds, gs = out['match_res']
            else:
                ds, gs = self._matcher(out, [frame_target])[0]
            gt2det_map = torch.zeros(len(gs)).to(device=ds.device, dtype=ds.dtype)
            gt2det_map[gs] = ds
            gt_rel_pairs = gt2det_map[frame_target['relation_map'].sum(-1).nonzero(as_tuple=False)]

        imgid, num_nodes= 0, self.args.num_queries
        # >>>>>>>>>>>> relation proposal <<<<<<<<<<<<<<<
        probs = out['pred_logits'][imgid].softmax(-1)
        inst_scores, inst_labels = probs[:, :-1].max(-1)
        human_instance_ids = torch.logical_and(inst_scores>0.1, inst_labels==0).nonzero(as_tuple=False) # class0: person

        rel_mat = torch.zeros((num_nodes, num_nodes))
        rel_mat[human_instance_ids] = 1
        if self.args.hoi_oracle_mode:
            gt_mask = torch.zeros_like(rel_mat)
            gt_mask[out['match_res'][0]] += 1; gt_mask[:, out['match_res'][0]] += 1
            rel_mat[gt_mask!=2] = 0

        if self.training:
            # sampling
            if self.args.hoi_oracle_mode:
                rel_mat[gt_rel_pairs[:, :1], ds] = 1
            else:
                rel_mat[gt_rel_pairs[:, :1]] = 1
            rel_mat[gt_rel_pairs[:, 0], gt_rel_pairs[:, 1]] = 0
            rel_mat.fill_diagonal_(0)
            rel_pairs = rel_mat.nonzero(as_tuple=False) # neg pairs

            if self.args.hoi_hard_mining:
                all_pairs = torch.cat([gt_rel_pairs, rel_pairs], dim=0)
                gt_pair_count = len(gt_rel_pairs)
                all_rel_reps = self._build_relation_representations(instance_representations, out, all_pairs, imgid, features_all=features_all, image_size=image_size)
                p_relation_exist_logits = self.relation_proposal_mlp(all_rel_reps)

                gt_inds = torch.arange(gt_pair_count).to(p_relation_exist_logits.device)
                # _, sort_rel_inds = p_relation_exist_logits.sigmoid()[gt_pair_count:].squeeze(1).sort(descending=True)
                _, sort_rel_inds = torch.cat([inst_scores[all_pairs], p_relation_exist_logits.sigmoid()], dim=-1).prod(-1)[gt_pair_count:].sort(descending=True)
                sampled_rel_inds = torch.cat([gt_inds, sort_rel_inds+gt_pair_count])[:self.args.num_hoi_queries]

                sampled_rel_pairs = all_pairs[sampled_rel_inds]
                sampled_rel_reps = all_rel_reps[sampled_rel_inds]
                sampled_rel_pred_exists = p_relation_exist_logits.squeeze(1)[sampled_rel_inds]
            else:
                sampled_neg_inds = torch.randperm(len(rel_pairs)) # random sampling
                sampled_rel_pairs = torch.cat([gt_rel_pairs, rel_pairs[sampled_neg_inds]], dim=0)[:self.args.num_hoi_queries]
                sampled_rel_reps = self._build_relation_representations(instance_representations, out, sampled_rel_pairs, imgid, features_all=features_all, image_size=image_size)
                sampled_rel_pred_exists = self.relation_proposal_mlp(sampled_rel_reps).squeeze(1)
        else:
            if self.args.hoi_relation_propagation_on_inference and 'prev_top_rel_pairs' in frame_target:
                prev_rel_pairs = frame_target['prev_top_rel_pairs'].cpu()
            else:
                prev_rel_pairs = torch.zeros((0, 2)).long()
            prev_pair_count = len(prev_rel_pairs)
            prev_pair_inds = torch.arange(prev_pair_count)

            if not self.args.hoi_oracle_mode and self.args.hoi_inference_apply_nms:
                bg_inds = self.apply_nms(inst_scores, inst_labels, out['pred_boxes'][imgid])
                rel_mat[:, bg_inds] = 0
            rel_mat[prev_rel_pairs[:, 0], prev_rel_pairs[:, 1]] = 0
            rel_mat.fill_diagonal_(0)
            rel_pairs = rel_mat.nonzero(as_tuple=False)

            # predict interactiveness and sorting
            rel_pairs = torch.cat([prev_rel_pairs, rel_pairs], dim=0)
            rel_reps = self._build_relation_representations(instance_representations, out, rel_pairs, imgid, features_all=features_all, image_size=image_size)
            p_relation_exist_logits = self.relation_proposal_mlp(rel_reps)

            # _, sort_rel_inds = p_relation_exist_logits.sigmoid()[prev_pair_count:].squeeze(1).sort(descending=True)
            _, sort_rel_inds = torch.cat([inst_scores[rel_pairs], p_relation_exist_logits.sigmoid()], dim=-1).prod(-1)[prev_pair_count:].sort(descending=True)
            sampled_rel_inds = torch.cat([prev_pair_inds.to(sort_rel_inds.device),
                                          sort_rel_inds[:self.args.num_hoi_queries] + prev_pair_count])

            sampled_rel_pairs = rel_pairs[sampled_rel_inds]
            sampled_rel_reps = rel_reps[sampled_rel_inds]
            sampled_rel_pred_exists = p_relation_exist_logits.squeeze(1)[sampled_rel_inds]

        # >>>>>>>>>>>> relation classification <<<<<<<<<<<<<<<
        query_reps = self.rel_query_pre_proj(sampled_rel_reps).unsqueeze(1)
        relation_outs, _ = self.interaction_decoder(tgt=query_reps,
                                        memory=memory_input[imgid:imgid+1].flatten(2).permute(2,0,1),
                                        memory_key_padding_mask=memory_input_mask[imgid:imgid+1].flatten(1),
                                        pos=memory_pos[imgid:imgid+1].flatten(2).permute(2, 0, 1))
        if self.args.hoi_oracle_mode and self.args.hoi_oracle_mode_use_instant_trajectory: # for fair comparison with ST-HOI of fusing GT trajectory feature
            ds2gs = torch.zeros(self.args.num_queries).long() - 1
            ds2gs[out['match_res'][0]] = out['match_res'][1]
            traj_feats = frame_target['box_instant_trajectories'][ds2gs[sampled_rel_pairs]].view(-1, self.traj_feat_dim)
            relation_outs = self.fuse_trajectory_feature_mlp(
                torch.cat([relation_outs, self.trajectory_feature_fc(traj_feats.unsqueeze(0).unsqueeze(-2)).expand(len(relation_outs),-1,-1,-1)], dim=-1)
            )

        relation_logits = self.relation_embed(relation_outs)
        out.update({
            "pred_rel_pairs": [sampled_rel_pairs],
            "pred_relations": [relation_logits[-1].squeeze(1)],
            "pred_relation_exists": [sampled_rel_pred_exists],
        })

        if self.args.hoi_aux_loss:
            out['relation_aux_outputs'] = self._set_hoi_aux_loss([relation_logits])
        return out

    @torch.jit.unused
    def _set_hoi_aux_loss(self, pred_relations):
        return [{'pred_relations': [p[l].squeeze(1) for p in pred_relations]} for l in range(self.args.hoi_dec_layers - 1)]

    # merge boxes (NMS)
    def apply_nms(self, inst_scores, inst_labels, cxcywh_boxes, threshold=0.7):
        xyxy_boxes = box_cxcywh_to_xyxy(cxcywh_boxes)
        box_areas = (xyxy_boxes[:, 2:] - xyxy_boxes[:, :2]).prod(-1)
        box_area_sum = box_areas.unsqueeze(1) + box_areas.unsqueeze(0)

        union_boxes = torch.cat([torch.min(xyxy_boxes.unsqueeze(1)[:, :, :2], xyxy_boxes.unsqueeze(0)[:, :, :2]),
                                 torch.max(xyxy_boxes.unsqueeze(1)[:, :, 2:], xyxy_boxes.unsqueeze(0)[:, :, 2:])], dim=-1)
        union_area = (union_boxes[:,:,2:] - union_boxes[:,:,:2]).prod(-1)
        iou = torch.clamp(box_area_sum - union_area, min=0) / union_area
        box_match_mat = torch.logical_and(iou > threshold, inst_labels.unsqueeze(1) == inst_labels.unsqueeze(0))

        suppress_ids = []
        for box_match in box_match_mat:
            group_ids = box_match.nonzero(as_tuple=False).squeeze(1)
            if len(group_ids) > 1:
                max_score_inst_id = group_ids[inst_scores[group_ids].argmax()]
                bg_ids = group_ids[group_ids!=max_score_inst_id]
                suppress_ids.append(bg_ids)
                box_match_mat[:, bg_ids] = False
        if len(suppress_ids) > 0:
            suppress_ids = torch.cat(suppress_ids, dim=0)
        return suppress_ids

    def _build_relation_representations(self, inst_reps, outs, rel_pairs, imgid, features_all=None, image_size=None):
        if self.args.hoi_instance_fuse_spatial_and_semantic_feat:
            inst_spatial_reps = self.spatial_embed(outs['pred_boxes'][imgid])
            inst_semantic_reps = outs['pred_logits'][imgid].softmax(-1) @ self.semantic_embed.weight
            inst_reps = self.instance_representation_fuse(torch.cat([inst_reps, inst_spatial_reps, inst_semantic_reps], dim=-1))

        rel_reps = torch.cat([inst_reps[rel_pairs[:,0]], inst_reps[rel_pairs[:,1]]], dim=1)

        # fuse roi_align union feature
        if self.args.hoi_oracle_mode and self.args.hoi_oracle_mode_use_roialign_union_feat:
            feat_order_dict = OrderedDict()
            for lvl, feat in enumerate(features_all):
                feat_order_dict[str(lvl)] = feat.tensors
            fpn_feats = self.fpn(feat_order_dict)

            xyxy_boxes = box_cxcywh_to_xyxy(outs['pred_boxes'][imgid])
            img_h, img_w = image_size
            scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(xyxy_boxes.device)
            xyxy_boxes = xyxy_boxes * scale_fct[None, :]
            subj_boxes, obj_boxes = xyxy_boxes[rel_pairs[:, 0]], xyxy_boxes[rel_pairs[:, 1]]
            union_boxes = torch.cat([torch.min(subj_boxes[:, :2], obj_boxes[:, :2]), torch.max(subj_boxes[:, 2:], obj_boxes[:, 2:])], dim=-1)
            union_pool_feats = self.box_pool_fc(
                self.box_pooler(fpn_feats, [union_boxes], [image_size]).view(-1, 256*7*7)
            )
            rel_reps = self.union_pool_fc(torch.cat([rel_reps, union_pool_feats], dim=-1))

        return rel_reps
