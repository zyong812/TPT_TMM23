"""
Let's get the relationships yo
"""

import numpy as np
import torch
import torch.nn as nn

from .word_vectors import obj_edge_vectors
from .transformer import transformer
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.jit.annotations import Tuple, List
from collections import OrderedDict
from torchvision.models.detection.image_list import ImageList
from torchvision.ops.boxes import box_iou

def normalize_box(boxes, image_size):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    device = boxes.device
    H, W = image_size

    wh = boxes[:, 2:] - boxes[:, :2] + 1.0
    xywh = torch.cat((boxes[:, :2] + 0.5 * wh, wh), 1) / torch.tensor([[W,H,W,H]]).to(device)
    return xywh

def multilabel_focal_loss(inputs, targets, gamma=2):
    probs = inputs.sigmoid()

    # focal loss to balance positive/negative
    pos_inds = targets.eq(1).float()
    neg_inds = targets.lt(1).float()
    pos_loss = torch.log(probs) * torch.pow(1 - probs, gamma) * pos_inds
    neg_loss = torch.log(1 - probs) * torch.pow(probs, gamma) * neg_inds
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    # normalize
    num_pos = pos_inds.float().sum()
    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos

    return loss

class STTran(nn.Module):

    def __init__(self, args, frcnn, obj_classes, enc_layer_num=1, dec_layer_num=3):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        """
        super(STTran, self).__init__()
        self.args = args
        self.frcnn = frcnn
        for p in self.parameters():
            p.requires_grad_(False)

        self.obj_classes = obj_classes
        assert args.sgg_mode in ('sgdet', 'sgcls', 'predcls')
        self.mode = args.sgg_mode

        ###################################
        vis_dim, spatial_dim, hidden_dim, semantic_dim = 1024, 128, self.args.hidden_dim, 200
        self.pos_embed = nn.Sequential(nn.Linear(4, spatial_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.1))

        self.subj_fc = nn.Linear(vis_dim+spatial_dim, hidden_dim)
        self.obj_fc = nn.Linear(vis_dim+spatial_dim, hidden_dim)
        self.vr_fc = nn.Linear(256*7*7, hidden_dim)

        embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='data/glove', wv_dim=semantic_dim)
        self.obj_embed = nn.Embedding(len(obj_classes), semantic_dim)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.obj_embed2 = nn.Embedding(len(obj_classes), semantic_dim)
        self.obj_embed2.weight.data = embed_vecs.clone()

        rel_dim = hidden_dim*3 + semantic_dim*2
        self.rel_input_fc = nn.Linear(rel_dim, hidden_dim)
        self.glocal_transformer = transformer(enc_layer_num=enc_layer_num, dec_layer_num=dec_layer_num, embed_dim=hidden_dim, nhead=8,
                                              dim_feedforward=2048, dropout=0.1, mode='latter')

        self.rel_compress = nn.Linear(hidden_dim, args.num_relations)

    @torch.no_grad()
    def _get_detection_results(self, images, targets, IoU_threshold=0.5, K=16):
        # from torchvision GeneralizedRCNN forward
        self.frcnn.eval()
        device = images.device
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        image_list, _ = self.frcnn.transform(images, None)
        features = self.frcnn.backbone(image_list.tensors)

        if self.mode == 'predcls':
            detections = [{'boxes': t['boxes'],
                           'labels': t['labels'],
                           'scores': torch.ones(len(t['labels'])).to(device)} for t in targets]
            org_h, org_w = original_image_sizes[0]; det_h, det_w = image_list.image_sizes[0]
            ratios = torch.tensor([det_w/org_w, det_h/org_h, det_w/org_w, det_h/org_h]).unsqueeze(0).to(device)
            boxes_for_roi_pool = [d['boxes'] * ratios for d in detections]
        elif self.mode == 'sgdet':
            proposals, _ = self.frcnn.rpn(image_list, features, None)
            detections, _ = self.frcnn.roi_heads(features, proposals, image_list.image_sizes, None)
            boxes_for_roi_pool = [d['boxes'] for d in detections]
            detections = self.frcnn.transform.postprocess(detections, image_list.image_sizes, original_image_sizes)

        # box features
        det_nums = [len(d['boxes']) for d in detections]
        res = self.frcnn.roi_heads.box_roi_pool(features, boxes_for_roi_pool, image_list.image_sizes)
        box_features = self.frcnn.roi_heads.box_head(res).split(det_nums, dim=0)

        # relation pairs
        box_idxs, rel_pairs, rel_im_idxs, rel_union_boxes, rel_targets = [], [], [], [], []
        for image_id, (det_num, dets, tgt) in enumerate(zip(det_nums, detections, targets)):
            box_idxs.append(torch.ones(det_num).to(device) * image_id)

            # rel pairs
            rel_map = torch.zeros(det_num, det_num).to(device)
            if self.training:
                if self.mode == 'sgdet':
                    if len(tgt['labels']) < 1: # special case: no relations
                        img_rel_pairs = torch.zeros((0, 2), device=device).long()
                        rel_targets.append(torch.zeros((0, self.args.num_relations), device=device).float())
                    else:
                        # detection matching with groundtruths
                        unmatched_IDX = -1
                        label_match = (dets['labels'].unsqueeze(1) == tgt['labels'].unsqueeze(0))
                        IoUs = box_iou(dets['boxes'], tgt['boxes'])
                        IoUs[~label_match] = 0

                        overlaps, det2gt_ids = IoUs.max(dim=1)
                        det2gt_ids[overlaps < IoU_threshold] = unmatched_IDX # unmatched detections -1

                        # sampling & relation targets
                        rel_map[(dets['labels']==1) & (det2gt_ids!=unmatched_IDX)] = 1 # subject is a detected human
                        rel_map.fill_diagonal_(0)
                        img_rel_pairs = rel_map.nonzero()

                        rel_obj_matched_ids, perm = det2gt_ids[img_rel_pairs[:,1]].sort(descending=True)
                        img_rel_pairs = img_rel_pairs[perm[:K]] # sampling

                        gt_obj_ids = det2gt_ids[img_rel_pairs]
                        rel_tgt_map = tgt['relation_map'][gt_obj_ids[:, 0], gt_obj_ids[:, 1]] # set annotated relations
                        rel_tgt_map[rel_obj_matched_ids[:K]==unmatched_IDX] = 0 # negative relations
                        rel_targets.append(rel_tgt_map)
                elif self.mode == 'predcls':
                    rel_map[dets['labels']==1] = 1 # subject is human
                    rel_map.fill_diagonal_(0)
                    img_rel_pairs = rel_map.nonzero()

                    # relation targets
                    rel_tgt_map = tgt['relation_map'][img_rel_pairs[:, 0], img_rel_pairs[:, 1]]
                    rel_targets.append(rel_tgt_map)
            else:
                rel_map[(dets['labels']==1) & (dets['scores']>0.2)] = 1 # subject is human
                rel_map[:, dets['scores']<0.2] = 0
                rel_map.fill_diagonal_(0)
                img_rel_pairs = rel_map.nonzero()

            rel_pairs.append(img_rel_pairs + sum(det_nums[:image_id]))
            rel_im_idxs.append(torch.ones(len(img_rel_pairs)).to(device) * image_id)

            # union boxes
            subj_boxes, obj_boxes = boxes_for_roi_pool[image_id][img_rel_pairs[:,0]], boxes_for_roi_pool[image_id][img_rel_pairs[:,1]]
            union_boxes = torch.cat([torch.min(subj_boxes[:, :2], obj_boxes[:, :2]), torch.max(subj_boxes[:, 2:], obj_boxes[:, 2:])], dim=-1)
            rel_union_boxes.append(union_boxes)

        res = {
            'box_nums': det_nums,
            'boxes': torch.cat([torch.cat([ids.unsqueeze(-1), d['boxes']], dim=-1) for ids, d in zip(box_idxs, detections)], dim=0),
            'scores': torch.cat([d['scores'] for d in detections], dim=0),
            'labels': torch.cat([d['labels'] for d in detections], dim=0),
            'box_features': torch.cat(box_features, dim=0),
            'rel_pair_nums': [len(p) for p in rel_pairs],
            'rel_pair_idxs': torch.cat(rel_pairs, dim=0),
            'rel_im_idxs': torch.cat(rel_im_idxs, dim=0),
            'rel_union_feats': self.frcnn.roi_heads.box_roi_pool(features, rel_union_boxes, image_list.image_sizes),
            'image_org_size': torch.tensor(original_image_sizes[0]).to(device)
        }
        if self.training: res.update({'rel_targets': torch.cat(rel_targets, dim=0)})
        return res

    def forward(self, images, targets=None):
        entry = self._get_detection_results(images, targets)
        rel_pairs = entry['rel_pair_idxs']

        # visual part
        pos_features = self.pos_embed(normalize_box(entry['boxes'][:, 1:], image_size=entry['image_org_size']))
        obj_features = torch.cat([entry['box_features'], pos_features], dim=-1)
        subj_rep = self.subj_fc(obj_features[rel_pairs[:, 0]])
        obj_rep = self.obj_fc(obj_features[rel_pairs[:, 1]])
        vr = self.vr_fc(entry['rel_union_feats'].view(-1, 256*7*7))
        x_visual = torch.cat((subj_rep, obj_rep, vr), 1)

        # semantic part
        subj_emb = self.obj_embed(entry['labels'][rel_pairs[:, 0]])
        obj_emb = self.obj_embed2(entry['labels'][rel_pairs[:, 1]])
        x_semantic = torch.cat((subj_emb, obj_emb), 1)

        rel_features = self.rel_input_fc(torch.cat((x_visual, x_semantic), dim=1))
        if len(entry['rel_im_idxs']) > 0: # Spatial-Temporal Transformer
            rel_features, _, _ = \
                self.glocal_transformer(features=rel_features, im_idx=self.get_continuous_image_idxs(entry['rel_im_idxs'].clone()))
        entry["rel_logits"] = self.rel_compress(rel_features)

        if self.training:
            relation_cls_loss = multilabel_focal_loss(entry["rel_logits"], entry["rel_targets"])
            return {'relation_cls_loss': relation_cls_loss}
        else:
            return entry

    def get_continuous_image_idxs(self, org_idxes):
        sorted_unique_idxes = sorted(org_idxes.unique().tolist())
        if org_idxes[-1] != len(sorted_unique_idxes)-1:
            for new_id, org_id in enumerate(sorted_unique_idxes):
                org_idxes[org_idxes==org_id] = new_id
        return org_idxes

def build_frcnn(args):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, args.num_classes)
    return model

def build_sttran(args, obj_classes):
    frcnn = build_frcnn(args)
    sttran = STTran(args, frcnn, obj_classes)
    return sttran
