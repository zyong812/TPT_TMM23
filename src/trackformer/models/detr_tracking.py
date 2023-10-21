import random

import torch
import torch.nn as nn

from ..util import box_ops
from ..util.misc import NestedTensor
from .deformable_detr import DeformableDETR
from .detr import DETR
import numpy as np

class DETRTrackingBase(nn.Module):

    def __init__(self,
                 track_query_false_positive_prob=0.0,
                 track_query_false_negative_prob=0.0,
                 track_query_noise=0.0,
                 matcher=None,
                 track_query_propagation_strategy='trackformer',
                 tracking_token_propagation=True,
                 clip_mode=False,
                 detection_obj_score_thresh=0.9,
                 token_propagation_sample_rate=0.1,
                 tracking_match_propagation_skip_frame=False):
        self._matcher = matcher
        self._track_query_false_positive_prob = track_query_false_positive_prob
        self._track_query_false_negative_prob = track_query_false_negative_prob
        self._track_query_noise = track_query_noise

        self._tracking = False

        self.track_query_propagation_strategy = track_query_propagation_strategy
        self.tracking_token_propagation = tracking_token_propagation

        if self.track_query_propagation_strategy == 'consistent_pairing' and self.tracking_token_propagation:
            self.propagation_mlp = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
            )
        self.clip_mode = clip_mode
        self.detection_obj_score_thresh = detection_obj_score_thresh
        self.token_propagation_sample_rate = token_propagation_sample_rate
        self.tracking_match_propagation_skip_frame = tracking_match_propagation_skip_frame

    def train(self, mode: bool = True):
        """Sets the module in train mode."""
        self._tracking = False
        return super().train(mode)

    def tracking(self):
        """Sets the module in tracking mode."""
        self.eval()
        self._tracking = True

    def foward_tracking_inference(self, samples: NestedTensor, targets: list = None):
        if self.track_query_propagation_strategy == 'consistent_pairing' and self.tracking_token_propagation and targets is not None: # from the 2nd frame under consistent_pairing
            for t in targets:
                t['track_query_hs_embeds_mapping'] = self.propagation_mlp(t['track_query_hs_embeds']) # originally prev_outs

        out, targets, features, memory, hs = super().forward(samples, targets)
        return out, targets, features, memory, hs

    def foward_train_val_2frame_mode(self, samples: NestedTensor, targets: list = None):
        assert self.track_query_propagation_strategy == 'trackformer'
        if targets is not None:
            prev_out, *_ = super().forward([targets[0]['prev_image']]) ## detr detection forward

            prev_outputs_without_aux = {
                k: v for k, v in prev_out.items() if 'aux_outputs' not in k}
            prev_targets = [
                {k.replace('prev_', ''): v for k, v in target.items() if "prev" in k}
                for target in targets]
            prev_indices = self._matcher(prev_outputs_without_aux, prev_targets)

            for i, (target, prev_ind) in enumerate(zip(targets, prev_indices)):
                prev_out_ind, prev_target_ind = prev_ind

                # random subset ## 随机擦除上一帧检测到的部分目标 FN Augmentation
                if self._track_query_false_negative_prob and self.track_query_propagation_strategy == 'trackformer':
                    random_subset_mask = torch.empty(len(prev_target_ind)).uniform_()
                    random_subset_mask = random_subset_mask.ge(
                        self._track_query_false_negative_prob)

                    prev_out_ind = prev_out_ind[random_subset_mask]
                    prev_target_ind = prev_target_ind[random_subset_mask]

                ## transfer matching from prev to current
                # detected prev frame tracks
                prev_track_ids = target['prev_track_ids'][prev_target_ind]

                # match track ids between frames
                target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(target['track_ids'])
                target_ind_matching = target_ind_match_matrix.any(dim=1)
                target_ind_matched_idx = target_ind_match_matrix.nonzero()[:, 1]

                # current frame track ids detected in the prev frame
                # track_ids = target['track_ids'][target_ind_matched_idx]

                # index of prev frame detection in current frame box list
                target['track_query_match_ids'] = target_ind_matched_idx

                not_prev_out_ind = torch.arange(prev_out['pred_boxes'].shape[1])
                not_prev_out_ind = [
                    ind.item()
                    for ind in not_prev_out_ind
                    if ind not in prev_out_ind]
                random_false_out_ind = []

                # random false positives
                prev_boxes_matched = prev_out['pred_boxes'][i, prev_out_ind[target_ind_matching]]
                for prev_box_matched in prev_boxes_matched:
                    ## 从背景集合，随机挑选噪声 FP Augmentation
                    if random.uniform(0, 1) < self._track_query_false_positive_prob:
                        prev_boxes_unmatched = prev_out['pred_boxes'][i, not_prev_out_ind]

                        # only cxcy
                        # box_dists = prev_box_matched[:2].sub(prev_boxes_unmatched[:, :2]).abs()
                        # box_dists = box_dists.pow(2).sum(dim=-1).sqrt()
                        # box_weights = 1.0 / box_dists.add(1e-8)

                        prev_box_ious, _ = box_ops.box_iou(
                            box_ops.box_cxcywh_to_xyxy(prev_box_matched.unsqueeze(dim=0)),
                            box_ops.box_cxcywh_to_xyxy(prev_boxes_unmatched))
                        box_weights = prev_box_ious[0]

                        if box_weights.gt(0.0).any():
                            random_false_out_idx = not_prev_out_ind.pop(
                                torch.multinomial(box_weights.cpu(), 1).item())
                            random_false_out_ind.append(random_false_out_idx)

                prev_out_ind = torch.tensor(prev_out_ind.tolist() + random_false_out_ind).long()
                target_ind_matching = torch.tensor(
                    target_ind_matching.tolist() + [False, ] * len(random_false_out_ind)).bool()

                # matches indices with 1.0 and not matched -1.0
                track_queries_match_mask = torch.ones_like(target_ind_matching).float()
                track_queries_match_mask[~target_ind_matching] = -1.0

                # set prev frame info
                hs_embeds = prev_out['hs_embed'][i, prev_out_ind]
                if self._track_query_noise and not torch.isnan(hs_embeds.std()).any():
                    track_query_noise = torch.randn_like(hs_embeds) \
                        * hs_embeds.std(dim=1, keepdim=True)
                    hs_embeds = hs_embeds + track_query_noise * self._track_query_noise
                    # hs_embeds = track_query_noise * self._track_query_noise \
                    #     + hs_embeds * (1 - self._track_query_noise)
                target['track_query_hs_embeds'] = hs_embeds
                target['track_query_boxes'] = prev_out['pred_boxes'][i, prev_out_ind].detach()

                # add zeros for detection object queries
                device = track_queries_match_mask.device
                track_queries_match_mask = torch.tensor(
                    track_queries_match_mask.tolist() + [0, ] * self.num_queries)

                target['track_queries_match_mask'] = track_queries_match_mask.to(device)

        out, targets, features, memory, hs = super().forward(samples, targets)
        return out, targets, features, memory, hs

    def foward_train_val_clip_mode(self, samples: NestedTensor, targets: list = None):
        outs = []
        for frame_id, frame_target in enumerate(targets):
            frame_image = samples.tensors[frame_id]
            out, *_ = super().forward([frame_image], [frame_target])
            outs.append(out)

            # propagate to the next frame
            if frame_id < len(targets)-1:
                prev_out = out
                target = targets[frame_id+1] # target for the next frame
                prev_indices = self._matcher(prev_out, [frame_target])
                prev_out_ind, prev_target_ind = prev_indices[0]
                prev_tid2qid = {int(tid): int(qid) for tid, qid in zip(frame_target['track_ids'][prev_target_ind], prev_out_ind)}

                # random subset ## 随机擦除上一帧检测到的部分目标 FN Augmentation
                if self._track_query_false_negative_prob:
                    random_subset_mask = torch.empty(len(prev_target_ind)).uniform_()
                    random_subset_mask = random_subset_mask.ge(
                        self._track_query_false_negative_prob)

                    prev_out_ind = prev_out_ind[random_subset_mask]
                    prev_target_ind = prev_target_ind[random_subset_mask]

                ## transfer matching from prev to current
                # detected prev frame tracks
                prev_track_ids = frame_target['track_ids'][prev_target_ind]

                # match track ids between frames
                target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(target['track_ids'])
                target_ind_matching = target_ind_match_matrix.any(dim=1)
                target_ind_matched_idx = target_ind_match_matrix.nonzero()[:, 1]

                # current frame track ids detected in the prev frame
                # track_ids = target['track_ids'][target_ind_matched_idx]

                # index of prev frame detection in current frame box list
                target['track_query_match_ids'] = target_ind_matched_idx

                if self.track_query_propagation_strategy == 'trackformer':
                    if not self.tracking_token_propagation: continue
                    not_prev_out_ind = torch.arange(prev_out['pred_boxes'].shape[1])
                    not_prev_out_ind = [
                        ind.item()
                        for ind in not_prev_out_ind
                        if ind not in prev_out_ind]
                    random_false_out_ind = []

                    # random false positives
                    prev_boxes_matched = prev_out['pred_boxes'][0, prev_out_ind[target_ind_matching]]
                    for prev_box_matched in prev_boxes_matched:
                        ## 从背景集合，随机挑选噪声 FP Augmentation
                        if random.uniform(0, 1) < self._track_query_false_positive_prob:
                            prev_boxes_unmatched = prev_out['pred_boxes'][0, not_prev_out_ind]

                            # only cxcy
                            # box_dists = prev_box_matched[:2].sub(prev_boxes_unmatched[:, :2]).abs()
                            # box_dists = box_dists.pow(2).sum(dim=-1).sqrt()
                            # box_weights = 1.0 / box_dists.add(1e-8)

                            prev_box_ious, _ = box_ops.box_iou(
                                box_ops.box_cxcywh_to_xyxy(prev_box_matched.unsqueeze(dim=0)),
                                box_ops.box_cxcywh_to_xyxy(prev_boxes_unmatched))
                            box_weights = prev_box_ious[0]

                            if box_weights.gt(0.0).any():
                                random_false_out_idx = not_prev_out_ind.pop(
                                    torch.multinomial(box_weights.cpu(), 1).item())
                                random_false_out_ind.append(random_false_out_idx)

                    prev_out_ind = torch.tensor(prev_out_ind.tolist() + random_false_out_ind).long()
                    target_ind_matching = torch.tensor(
                        target_ind_matching.tolist() + [False, ] * len(random_false_out_ind)).bool()

                    # matches indices with 1.0 and not matched -1.0
                    track_queries_match_mask = torch.ones_like(target_ind_matching).float()
                    track_queries_match_mask[~target_ind_matching] = -1.0

                    # set prev frame info
                    hs_embeds = prev_out['hs_embed'][0, prev_out_ind]
                    if self._track_query_noise and not torch.isnan(hs_embeds.std()).any():
                        track_query_noise = torch.randn_like(hs_embeds) \
                            * hs_embeds.std(dim=1, keepdim=True)
                        hs_embeds = hs_embeds + track_query_noise * self._track_query_noise
                        # hs_embeds = track_query_noise * self._track_query_noise \
                        #     + hs_embeds * (1 - self._track_query_noise)
                    target['track_query_hs_embeds'] = hs_embeds
                    target['track_query_boxes'] = prev_out['pred_boxes'][0, prev_out_ind].detach()

                    # add zeros for detection object queries
                    device = track_queries_match_mask.device
                    track_queries_match_mask = torch.tensor(
                        track_queries_match_mask.tolist() + [0, ] * self.num_queries)

                    target['track_queries_match_mask'] = track_queries_match_mask.to(device)
                elif self.track_query_propagation_strategy == 'consistent_pairing':
                    device = target['track_query_match_ids'].device
                    tracked_qids = prev_out_ind[target_ind_matching]

                    if self.tracking_match_propagation_skip_frame: # skip frame match propagation
                        target['prev_tid2qid'] = prev_tid2qid
                        if 'prev_tid2qid' in frame_target:
                            prev_prev_tids = set(frame_target['prev_tid2qid'].keys())
                            prev_tids = set(frame_target['track_ids'].tolist())
                            cur_tids = set(target['track_ids'].tolist())
                            reappear_tids = list((cur_tids & prev_prev_tids) - prev_tids)
                            if len(reappear_tids) > 0:
                                # print(reappear_tids)
                                reappear_qids = [frame_target['prev_tid2qid'][tid] for tid in reappear_tids]
                                tracked_qids_append_reappear = tracked_qids.tolist() + reappear_qids
                                tracked_ids_append_reappear = target['track_query_match_ids'].tolist() + [target['track_ids'].tolist().index(tid) for tid in reappear_tids]

                                new_order = np.argsort(tracked_qids_append_reappear)
                                tracked_qids = torch.tensor(np.array(tracked_qids_append_reappear)[new_order])
                                target['track_query_match_ids'] = torch.tensor(np.array(tracked_ids_append_reappear)[new_order]).to(device)

                    # match mask to next frame
                    track_queries_match_mask = torch.zeros(prev_out['hs_embed'][0].shape[0]).float()
                    track_queries_match_mask[tracked_qids] = 1 # tracked in current frame
                    track_queries_match_mask[prev_out_ind[~target_ind_matching]] = -1 # disappeared in current frame
                    target['track_queries_match_mask'] = track_queries_match_mask.to(device)

                    if self.tracking_token_propagation:
                        target['track_query_hs_embeds'] = prev_out['hs_embed'][0]
                        target['track_query_hs_embeds_mapping'] = self.propagation_mlp(prev_out['hs_embed'][0])
                        target['track_query_boxes'] = prev_out['pred_boxes'][0].detach()

                        if random.random() < self.token_propagation_sample_rate:
                            target['track_token_propagation_mask'] = (prev_out['pred_logits'][0].softmax(-1)[:,:-1].max(-1)[0].detach() > self.detection_obj_score_thresh).float()
                        else:
                            target['track_token_propagation_mask'] = (target['track_queries_match_mask'] != 0).float()

        if self.track_query_propagation_strategy == 'consistent_pairing':
            ## compose outputs in batched format
            outputs = {key: torch.cat([o[key] for o in outs], dim=0) for key in ['pred_logits', 'pred_boxes', 'hs_embed']}
            if 'aux_outputs' in outs[0]:
                outputs['aux_outputs'] = []
                for l in range(len(outs[0]['aux_outputs'])):
                    outputs['aux_outputs'].append(
                        {key: torch.cat([o['aux_outputs'][l][key] for o in outs], dim=0) for key in ['pred_logits', 'pred_boxes']}
                    )
            return outputs, targets, None, None, None
        else:
            return outs, targets, None, None, None


    def forward(self, samples: NestedTensor, targets: list = None):
        if self._tracking: # tracking on inference
            return self.foward_tracking_inference(samples, targets)
        else:
            if self.clip_mode:
                return self.foward_train_val_clip_mode(samples, targets)
            else:
                return self.foward_train_val_2frame_mode(samples, targets)

# TODO: with meta classes
class DETRTracking(DETRTrackingBase, DETR):
    def __init__(self, tracking_kwargs, detr_kwargs):
        DETR.__init__(self, **detr_kwargs)
        DETRTrackingBase.__init__(self, **tracking_kwargs)


class DeformableDETRTracking(DETRTrackingBase, DeformableDETR):
    def __init__(self, tracking_kwargs, detr_kwargs):
        DeformableDETR.__init__(self, **detr_kwargs)
        DETRTrackingBase.__init__(self, **tracking_kwargs)
