## refer https://github.com/yrcong/STTran

import torch
import torch.nn as nn
import numpy as np
from functools import reduce
from ..util import box_ops
from ..util import misc as utils

class BasicSceneGraphEvaluator:
    def __init__(self, mode, iou_threshold=0.5, constraint=False, dataset='actiongenome'):
        self.dataset = dataset
        self.result_dict = {}
        self.mode = mode
        self.result_dict[self.mode + '_recall'] = {10: [], 20: [], 50: [], 100: []}
        self.constraint = constraint
        self.iou_threshold = iou_threshold

    def reset_result(self):
        self.result_dict[self.mode + '_recall'] = {10: [], 20: [], 50: [], 100: []}

    def print_stats(self):
        print(f'======================{self.mode} (Constraint={self.constraint})============================')
        for k, v in self.result_dict[self.mode + '_recall'].items():
            print('R@%i: %f' % (k, np.mean(v)))
        print(f"#images = {len(v)}")

    def evaluate_scene_graph(self, gt, outputs, box_preds):
        '''collect the groundtruth and prediction'''

        pred_top_rel_pairs = []
        for idx, frame_gt in enumerate(gt):
            frame_box_pred = box_preds[frame_gt['image_id'].item()]

            # generate ground truth
            boxes = box_ops.box_cxcywh_to_xyxy(frame_gt['boxes'])
            img_h, img_w = frame_gt['orig_size']
            scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).unsqueeze(0).to(boxes.device)
            boxes = boxes * scale_fct

            gt_boxes = boxes.cpu().numpy().astype(float)
            gt_classes = frame_gt['labels'].cpu().numpy()
            gt_relations = frame_gt['relation_map'].nonzero().cpu().numpy()

            # relation prediction
            pred_boxes = frame_box_pred['boxes'].cpu().numpy()
            pred_classes = frame_box_pred['labels'].cpu().numpy()
            pred_obj_scores = frame_box_pred['scores'].cpu().numpy()

            rel_pairs = outputs['pred_rel_pairs'][idx].cpu().numpy()
            predicate_scores = outputs['pred_relations'][idx].sigmoid()
            triplet_scores = predicate_scores # * outputs['pred_relation_exists'][idx].sigmoid().unsqueeze(-1) # * frame_box_pred['scores'][rel_pairs[:,0]].unsqueeze(1) * frame_box_pred['scores'][rel_pairs[:,1]].unsqueeze(1)
            # triplet_scores = predicate_scores * frame_box_pred['scores'][rel_pairs[:,0]].unsqueeze(1) * frame_box_pred['scores'][rel_pairs[:,1]].unsqueeze(1)

            if self.constraint: # follow STTran, only for AG evaluation
                attention_scores, attention_rel_inds = triplet_scores[:, :3].max(-1)
                spatial_scores, spatial_rel_inds = triplet_scores[:, 3:9].max(-1); spatial_rel_inds += 3
                contacting_scores, contacting_rel_inds = triplet_scores[:, 9:].max(-1); contacting_rel_inds +=9
                all_rel_inds = torch.cat([torch.arange(len(triplet_scores))] * 3, dim=0)
                all_scores = torch.cat([attention_scores, spatial_scores, contacting_scores], dim=0)
                all_predicates = torch.cat([attention_rel_inds, spatial_rel_inds, contacting_rel_inds], dim=0)

                rel_scores, perm = all_scores.sort(descending=True)
                pred_rels = np.column_stack([rel_pairs[all_rel_inds[perm]], all_predicates[perm].cpu().numpy()])
                rel_scores = rel_scores.cpu().numpy()
            else:
                score_inds = argsort_desc(triplet_scores.cpu().numpy())[:100]
                pred_rels = np.column_stack([rel_pairs[score_inds[:, 0]], score_inds[:, 1]])
                rel_scores = triplet_scores.cpu().numpy()[score_inds[:, 0], score_inds[:, 1]]

            # # groundtruths as fake predictions
            # pred_boxes = gt_boxes
            # pred_classes = gt_classes
            # pred_obj_scores = np.ones(len(pred_classes))
            # pred_rels = gt_relations
            # rel_scores = np.ones(len(pred_rels))

            pred_top_rel_pairs.append(pred_rels)
            ################ evaluation ################
            if len(gt_relations) == 0: continue
            pred_to_gt, pred_5ples, rel_scores = evaluate_recall(
                gt_relations, gt_boxes, gt_classes,
                pred_rels, pred_boxes, pred_classes, rel_scores, pred_obj_scores,
                iou_thresh=self.iou_threshold)

            for k in self.result_dict[self.mode + '_recall']:
                match = reduce(np.union1d, pred_to_gt[:k])

                rec_i = float(len(match)) / float(gt_relations.shape[0])
                self.result_dict[self.mode + '_recall'][k].append(rec_i)
        return pred_top_rel_pairs

    def synchronize_between_processes(self):
        all_results = utils.all_gather([self.result_dict])
        metric_key = self.mode + '_recall'

        merged_result_dict = all_results[0][0]
        for p in all_results[1:]:
            for k, v in merged_result_dict[metric_key].items():
                v.extend(p[0][metric_key][k])
        self.result_dict = merged_result_dict

###########################
def evaluate_recall(gt_rels, gt_boxes, gt_classes,
                    pred_rels, pred_boxes, pred_classes, rel_scores=None, cls_scores=None,
                    iou_thresh=0.5, phrdet=False):
    """
    Evaluates the recall
    :param gt_rels: [#gt_rel, 3] array of GT relations
    :param gt_boxes: [#gt_box, 4] array of GT boxes
    :param gt_classes: [#gt_box] array of GT classes
    :param pred_rels: [#pred_rel, 3] array of pred rels. Assumed these are in sorted order
                      and refer to IDs in pred classes / pred boxes
                      (id0, id1, rel)
    :param pred_boxes:  [#pred_box, 4] array of pred boxes
    :param pred_classes: [#pred_box] array of predicted classes for these boxes
    :return: pred_to_gt: Matching from predicate to GT
             pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
             rel_scores: [cls_0score, cls1_score, relscore]
                   """
    if pred_rels.size == 0:
        return [[]], np.zeros((0,5)), np.zeros(0)

    num_gt_boxes = gt_boxes.shape[0]
    num_gt_relations = gt_rels.shape[0]
    assert num_gt_relations != 0

    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels[:, 2],
                                                gt_rels[:, :2],
                                                gt_classes,
                                                gt_boxes)
    num_boxes = pred_boxes.shape[0]
    assert pred_rels[:,:2].max() < pred_classes.shape[0]

    # Exclude self rels
    # assert np.all(pred_rels[:,0] != pred_rels[:,ĺeftright])
    #assert np.all(pred_rels[:,2] > 0)

    pred_triplets, pred_triplet_boxes, relation_scores = \
        _triplet(pred_rels[:,2], pred_rels[:,:2], pred_classes, pred_boxes,
                 rel_scores, cls_scores)

    sorted_scores = relation_scores.prod(1)
    pred_triplets = pred_triplets[sorted_scores.argsort()[::-1],:]
    pred_triplet_boxes = pred_triplet_boxes[sorted_scores.argsort()[::-1],:]
    relation_scores = relation_scores[sorted_scores.argsort()[::-1],:]
    scores_overall = relation_scores.prod(1)

    if not np.all(scores_overall[1:] <= scores_overall[:-1] + 1e-5):
        print("Somehow the relations weren't sorted properly: \n{}".format(scores_overall))
        # raise ValueError("Somehow the relations werent sorted properly")

    # Compute recall. It's most efficient to match once and then do recall after
    pred_to_gt = _compute_pred_matches(
        gt_triplets,
        pred_triplets,
        gt_triplet_boxes,
        pred_triplet_boxes,
        iou_thresh,
        phrdet=phrdet,
    )

    # Contains some extra stuff for visualization. Not needed.
    pred_5ples = np.column_stack((
        pred_rels[:,:2],
        pred_triplets[:, [0, 2, 1]],
    ))

    return pred_to_gt, pred_5ples, relation_scores


def _triplet(predicates, relations, classes, boxes,
             predicate_scores=None, class_scores=None):
    """
    format predictions into triplets
    :param predicates: A 1d numpy array of num_boxes*(num_boxes-ĺeftright) predicates, corresponding to
                       each pair of possibilities
    :param relations: A (num_boxes*(num_boxes-ĺeftright), 2.0) array, where each row represents the boxes
                      in that relation
    :param classes: A (num_boxes) array of the classes for each thing.
    :param boxes: A (num_boxes,4) array of the bounding boxes for everything.
    :param predicate_scores: A (num_boxes*(num_boxes-ĺeftright)) array of the scores for each predicate
    :param class_scores: A (num_boxes) array of the likelihood for each object.
    :return: Triplets: (num_relations, 3) array of class, relation, class
             Triplet boxes: (num_relation, 8) array of boxes for the parts
             Triplet scores: num_relation array of the scores overall for the triplets
    """
    assert (predicates.shape[0] == relations.shape[0])

    sub_ob_classes = classes[relations[:, :2]]
    triplets = np.column_stack((sub_ob_classes[:, 0], predicates, sub_ob_classes[:, 1]))
    triplet_boxes = np.column_stack((boxes[relations[:, 0]], boxes[relations[:, 1]]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[relations[:, 0]],
            class_scores[relations[:, 1]],
            predicate_scores,
        ))

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thresh, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets:
    :param pred_triplets:
    :param gt_boxes:
    :param pred_boxes:
    :param iou_thresh:
    :return:
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thresh

        else:
            sub_iou = bbox_overlaps(gt_box[None,:4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None,4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt


# From frcnn/utils/bbox.py
def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray or tensor or variable
    query_boxes: (K, 4) ndarray or tensor or variable
    Returns
    -------
    overlaps: (N, K) overlap between boxes and query_boxes
    """
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
        query_boxes = torch.from_numpy(query_boxes)
        out_fn = lambda x: x.numpy()  # If input is ndarray, turn the overlaps back to ndarray when return
    else:
        out_fn = lambda x: x

    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * (query_boxes[:, 3] - query_boxes[:, 1] + 1)

    iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t()) - torch.max(boxes[:, 0:1],
                                                                        query_boxes[:, 0:1].t()) + 1).clamp(min=0)
    ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) - torch.max(boxes[:, 1:2],
                                                                        query_boxes[:, 1:2].t()) + 1).clamp(min=0)
    ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
    overlaps = iw * ih / ua
    return out_fn(overlaps)

def intersect_2d(x1, x2):
    """
    Given two arrays [m1, n], [m2,n], returns a [m1, m2] array where each entry is True if those
    rows match.
    :param x1: [m1, n] numpy array
    :param x2: [m2, n] numpy array
    :return: [m1, m2] bool array of the intersections
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same #columns")

    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res


def argsort_desc(scores):
    """
    Returns the indices that sort scores descending in a smart way
    :param scores: Numpy array of arbitrary size
    :return: an array of size [numel(scores), dim(scores)] where each row is the index you'd
             need to get the score.
    """
    return np.column_stack(np.unravel_index(np.argsort(-scores.ravel()), scores.shape))
