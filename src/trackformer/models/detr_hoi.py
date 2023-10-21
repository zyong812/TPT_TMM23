from .deformable_detr import DeformableDETR
from ..util.misc import NestedTensor
from ..util import box_ops
import torch
import torch.nn as nn
from .transformer import TransformerDecoderLayer, TransformerDecoder
from .position_encoding import build_position_encoding
from collections import OrderedDict
import torchvision

class DeformableDETRHoi(DeformableDETR):
    def __init__(self, args, detr_kwargs, matcher):
        DeformableDETR.__init__(self, **detr_kwargs)

        self.args = args
        self._matcher = matcher

        if args.freeze_detr:
            for p in self.parameters():
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
        if self.args.hoi_use_interaction_decoder:
            self.interaction_decoder = TransformerDecoder(decoder_layer, None, self.args.hoi_dec_layers, decoder_norm, return_intermediate=True)
        self.relation_embed = nn.Linear(rel_dec_hidden_dim, self.args.num_relations)

        if self.args.hoi_oracle_mode and self.args.hoi_oracle_mode_use_roialign_union_feat:
            self.fpn = torchvision.ops.feature_pyramid_network.FeaturePyramidNetwork(in_channels_list=[256, 512, 1024, 2048], out_channels=256)
            self.box_pooler = torchvision.ops.MultiScaleRoIAlign(['0', '1', '2', '3'], 7, sampling_ratio=2)
            self.box_pool_fc = nn.Sequential(nn.Linear(256*7*7, self.args.hidden_dim), nn.ReLU())
            self.union_pool_fc = nn.Linear(self.args.hidden_dim*3, self.args.hidden_dim*2)

    def forward(self, samples: NestedTensor, targets: list = None):
        outs, _, features_all, _, _ = super().forward(samples, targets)

        # memory input for relation transformer decoder
        memory_input_feature, memory_input_mask = features_all[-1].decompose()
        memory_pos = self.rel_memory_pos(features_all[-1])
        memory_input = self.memory_input_proj(memory_input_feature)

        det2gt_indices = None
        if self.training or self.args.hoi_oracle_mode:
            det2gt_indices = self._matcher(outs, targets)
            gt_rel_pairs = []
            for idx, ((ds, gs), t) in enumerate(zip(det2gt_indices, targets)):
                gt2det_map = torch.zeros(len(gs)).to(device=ds.device, dtype=ds.dtype)
                gt2det_map[gs] = ds
                gt_rels = gt2det_map[t['relation_map'].sum(-1).nonzero(as_tuple=False)]
                gt_rel_pairs.append(gt_rels)

                if self.args.hoi_oracle_mode:
                    outs['pred_logits'][idx, :, -1] = 1e3 # set default class as background
                    outs['pred_logits'][idx, ds, t['labels'][gs]] = 1e6
                    outs['pred_boxes'][idx, ds] = t['boxes'][gs]
                    if 'aux_outputs' in outs:
                        for o in outs['aux_outputs']:
                            o['pred_logits'][idx, :, -1] = 1e3
                            o['pred_logits'][idx, ds, t['labels'][gs]] = 1e6
                            o['pred_boxes'][idx, ds] = t['boxes'][gs]

        pred_relation_exists, pred_rel_pairs, pred_relations = [], [], []
        bs, num_nodes = samples.tensors.shape[0], self.args.num_queries
        for imgid in range(bs):
            # >>>>>>>>>>>> relation proposal <<<<<<<<<<<<<<<
            probs = outs['pred_logits'][imgid].softmax(-1)
            inst_scores, inst_labels = probs[:, :-1].max(-1)
            human_instance_ids = torch.logical_and(inst_scores>0.1, inst_labels==0).nonzero(as_tuple=False) # class0: person

            rel_mat = torch.zeros((num_nodes, num_nodes))
            rel_mat[human_instance_ids] = 1
            if self.args.hoi_oracle_mode:
                gt_mask = torch.zeros_like(rel_mat)
                gt_mask[det2gt_indices[imgid][0]] += 1; gt_mask[:, det2gt_indices[imgid][0]] += 1
                rel_mat[gt_mask!=2] = 0

            if self.training:
                if self.args.hoi_oracle_mode:
                    rel_mat[gt_rel_pairs[imgid][:, :1], det2gt_indices[imgid][0]] = 1
                else:
                    rel_mat[gt_rel_pairs[imgid][:, :1]] = 1
                rel_mat[gt_rel_pairs[imgid][:, 0], gt_rel_pairs[imgid][:, 1]] = 0
                rel_mat.fill_diagonal_(0)
                rel_pairs = rel_mat.nonzero(as_tuple=False) # neg pairs

                if self.args.hoi_hard_mining:
                    all_pairs = torch.cat([gt_rel_pairs[imgid], rel_pairs], dim=0)
                    gt_pair_count = len(gt_rel_pairs[imgid])
                    all_rel_reps = self._build_relation_representations(outs, all_pairs, imgid, features_all=features_all, image_size=samples.tensors.shape[-2:])
                    p_relation_exist_logits = self.relation_proposal_mlp(all_rel_reps)

                    gt_inds = torch.arange(gt_pair_count).to(p_relation_exist_logits.device)
                    # _, sort_rel_inds = p_relation_exist_logits[gt_pair_count:].squeeze(1).sort(descending=True)
                    _, sort_rel_inds = torch.cat([inst_scores[all_pairs], p_relation_exist_logits.sigmoid()], dim=-1).prod(-1)[gt_pair_count:].sort(descending=True)
                    sampled_rel_inds = torch.cat([gt_inds, sort_rel_inds+gt_pair_count])[:self.args.num_hoi_queries]

                    sampled_rel_pairs = all_pairs[sampled_rel_inds]
                    sampled_rel_reps = all_rel_reps[sampled_rel_inds]
                    sampled_rel_pred_exists = p_relation_exist_logits.squeeze(1)[sampled_rel_inds]
                else:
                    sampled_neg_inds = torch.randperm(len(rel_pairs)) # random sampling
                    sampled_rel_pairs = torch.cat([gt_rel_pairs[imgid], rel_pairs[sampled_neg_inds]], dim=0)[:self.args.num_hoi_queries]
                    sampled_rel_reps = self._build_relation_representations(outs, sampled_rel_pairs, imgid, features_all=features_all, image_size=samples.tensors.shape[-2:])
                    sampled_rel_pred_exists = self.relation_proposal_mlp(sampled_rel_reps).squeeze(1)
            else:
                rel_mat.fill_diagonal_(0)
                rel_pairs = rel_mat.nonzero(as_tuple=False)
                rel_reps = self._build_relation_representations(outs, rel_pairs, imgid, features_all=features_all, image_size=samples.tensors.shape[-2:])
                p_relation_exist_logits = self.relation_proposal_mlp(rel_reps)

                # _, sort_rel_inds = p_relation_exist_logits.squeeze(1).sort(descending=True)
                _, sort_rel_inds = torch.cat([inst_scores[rel_pairs], p_relation_exist_logits.sigmoid()], dim=-1).prod(-1).sort(descending=True)
                sampled_rel_inds = sort_rel_inds[:self.args.num_hoi_queries]

                sampled_rel_pairs = rel_pairs[sampled_rel_inds]
                sampled_rel_reps = rel_reps[sampled_rel_inds]
                sampled_rel_pred_exists = p_relation_exist_logits.squeeze(1)[sampled_rel_inds]

            # >>>>>>>>>>>> relation classification <<<<<<<<<<<<<<<
            query_reps = self.rel_query_pre_proj(sampled_rel_reps).unsqueeze(1)
            if self.args.hoi_use_interaction_decoder:
                relation_outs, _ = self.interaction_decoder(tgt=query_reps,
                                                memory=memory_input[imgid:imgid+1].flatten(2).permute(2,0,1),
                                                memory_key_padding_mask=memory_input_mask[imgid:imgid+1].flatten(1),
                                                pos=memory_pos[imgid:imgid+1].flatten(2).permute(2, 0, 1))
            else:
                relation_outs = query_reps.unsqueeze(0)
            relation_logits = self.relation_embed(relation_outs)

            pred_rel_pairs.append(sampled_rel_pairs)
            pred_relations.append(relation_logits)
            pred_relation_exists.append(sampled_rel_pred_exists)

        outs.update({
            "pred_rel_pairs": pred_rel_pairs,
            "pred_relations": [p[-1].squeeze(1) for p in pred_relations],
            "pred_relation_exists": pred_relation_exists,
            "det2gt_indices": det2gt_indices,
        })

        if self.args.hoi_aux_loss:
            outs['relation_aux_outputs'] = self._set_hoi_aux_loss(pred_relations)

        return outs, targets, None, None, None

    @torch.jit.unused
    def _set_hoi_aux_loss(self, pred_relations):
        return [{'pred_relations': [p[l].squeeze(1) for p in pred_relations]} for l in range(self.args.hoi_dec_layers - 1)]

    def _build_relation_representations(self, outs, rel_pairs, imgid, features_all=None, image_size=None):
        inst_reps = outs['hs_embed'][imgid]

        if self.args.hoi_instance_fuse_spatial_and_semantic_feat:
            inst_spatial_reps = self.spatial_embed(outs['pred_boxes'][imgid])
            inst_semantic_reps = outs['pred_logits'][imgid].softmax(-1) @ self.semantic_embed.weight
            inst_reps = self.instance_representation_fuse(torch.cat([inst_reps, inst_spatial_reps, inst_semantic_reps], dim=-1))

        rel_reps = torch.cat([inst_reps[rel_pairs[:, 0]], inst_reps[rel_pairs[:, 1]]], dim=1)

        # fuse roi_align union feature
        if self.args.hoi_oracle_mode and self.args.hoi_oracle_mode_use_roialign_union_feat:
            feat_order_dict = OrderedDict()
            for lvl, feat in enumerate(features_all):
                feat_order_dict[str(lvl)] = feat.tensors
            fpn_feats = self.fpn(feat_order_dict)

            xyxy_boxes = box_ops.box_cxcywh_to_xyxy(outs['pred_boxes'][imgid])
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

    def tracking(self):
        """Compatible with vsgg eval"""
        self.eval()
