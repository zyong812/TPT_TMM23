import copy
import itertools
import numpy as np
import torch
from collections import defaultdict
import json
import matplotlib.pyplot as plt
from ..util import box_ops
from ..util import misc as utils

# from ST-HOI paper/code
TEMPORAL_predicates = ['towards', 'away', 'pull', 'caress', 'push', 'press', 'wave', 'hit', 'lift', 'pat', 'grab', 'chase', 'release', 'wave_hand_to', 'squeeze', 'kick', 'shout_at', 'throw', 'smell', 'knock', 'lick', 'open', 'close', 'get_on', 'get_off']

def argsort_desc(scores):
    """
    Returns the indices that sort scores descending in a smart way
    :param scores: Numpy array of arbitrary size
    :return: an array of size [numel(scores), dim(scores)] where each row is the index you'd
             need to get the score.
    """
    return np.column_stack(np.unravel_index(np.argsort(-scores.ravel()), scores.shape))

class VidHOIEvaluator():
    def __init__(self, args):
        self.overlap_iou = 0.5
        self.max_hois = 100

        # meta_infos
        train_annotation_file = f"{args.vidhoi_path}/VidHOI_annotations/{args.train_split}_cocofmt.json"
        with open(train_annotation_file, 'r') as f:
            train_annotations = json.load(f)
        self.object_categories = [x['name'] for x in train_annotations['categories']][1:] # remove __background__
        self.predicates = [x['name'] for x in train_annotations['predicate_categories']]

        triplet_counts = np.zeros((len(self.predicates), len(self.object_categories)))
        for video_key, frame_dict in train_annotations['relation_annotations'].items():
            for frame_key, rels in frame_dict.items():
                for rel in rels:
                    triplet_counts[rel['predicate'], rel['object_class']] += 1
        self.correct_mat = (triplet_counts>0).astype("float64")

        # initialize
        self.sum_gts = {}
        self.gt_triplets = []
        self.preds = []
        self.gts = []

        self.fp = defaultdict(list)
        self.tp = defaultdict(list)
        self.score = defaultdict(list)

    def sttran_update(self, gts, outputs, box_preds):
        pred_top_rel_pairs = []
        for idx, frame_gt in enumerate(gts):
            frame_box_pred = box_preds[frame_gt['image_id'].item()]

            # relation predictions
            pred_boxes = frame_box_pred['boxes'].cpu().numpy()
            pred_classes = frame_box_pred['labels'].cpu().numpy()

            rel_pairs = outputs['pred_rel_pairs'][idx].cpu().numpy() # scores
            predicate_scores = outputs['pred_relations'][idx].sigmoid().cpu()
            triplet_scores = predicate_scores * frame_box_pred['scores'][rel_pairs[:,0]].unsqueeze(1) * frame_box_pred['scores'][rel_pairs[:,1]].unsqueeze(1)

            score_mask = self.correct_mat[:, pred_classes[rel_pairs[:,1]]].transpose() # mask unseen triplets
            triplet_scores = triplet_scores.numpy() * score_mask

            score_inds = argsort_desc(triplet_scores)[:self.max_hois] # get top100
            pred_rels = np.column_stack([rel_pairs[score_inds[:, 0]], score_inds[:, 1]]) # <sub, obj, predicate>
            rel_scores = triplet_scores[score_inds[:, 0], score_inds[:, 1]]
            pred_top_rel_pairs.append(pred_rels)

            # gt
            gt_boxes = frame_gt['boxes'].cpu().numpy()
            gt_classes = frame_gt['labels'].cpu().numpy()-1
            gt_relations = frame_gt['relation_map'].nonzero().cpu().numpy()
            if len(gt_relations) == 0: continue # skip frames with no gt relations, follow ST-HOI
            self.gts.append({
                'annotations': [{'bbox': bbox, 'category_id': label} for bbox, label in zip(gt_boxes, gt_classes)],
                'hoi_annotation': [{'subject_id': hoi[0], 'object_id': hoi[1], 'category_id': hoi[2]} for hoi in gt_relations]
            })
            for hoi in self.gts[-1]['hoi_annotation']:
                triplet = (self.gts[-1]['annotations'][hoi['subject_id']]['category_id'],
                           self.gts[-1]['annotations'][hoi['object_id']]['category_id'],
                           hoi['category_id'])
                if triplet not in self.gt_triplets: self.gt_triplets.append(triplet)
                if triplet not in self.sum_gts: self.sum_gts[triplet] = 0
                self.sum_gts[triplet] += 1

            self.preds.append({
                'predictions': [{'bbox': bbox, 'category_id': label} for bbox, label in zip(pred_boxes, pred_classes)],
                'hoi_prediction': [{'subject_id': hoi[0], 'object_id': hoi[1], 'category_id': hoi[2], 'score': score}
                                   for hoi, score in zip(pred_rels, rel_scores)]
            })

        return pred_top_rel_pairs

    def update(self, gts, outputs, box_preds):
        pred_top_rel_pairs = []
        for idx, frame_gt in enumerate(gts):
            frame_box_pred = box_preds[frame_gt['image_id'].item()]
            img_h, img_w = frame_gt['orig_size']
            # relation predictions
            pred_boxes = frame_box_pred['boxes'].cpu().numpy()
            pred_classes = frame_box_pred['labels'].cpu().numpy()

            rel_pairs = outputs['pred_rel_pairs'][idx].cpu().numpy() # scores
            predicate_scores = outputs['pred_relations'][idx].sigmoid()
            triplet_scores = predicate_scores * frame_box_pred['scores'][rel_pairs[:,0]].unsqueeze(1) * frame_box_pred['scores'][rel_pairs[:,1]].unsqueeze(1)
            # triplet_scores = predicate_scores * outputs['pred_relation_exists'][idx].sigmoid().unsqueeze(-1)

            score_mask = self.correct_mat[:, pred_classes[rel_pairs[:,1]]].transpose() # mask unseen triplets
            triplet_scores = triplet_scores.cpu().numpy() * score_mask

            score_inds = argsort_desc(triplet_scores)[:self.max_hois] # get top100
            pred_rels = np.column_stack([rel_pairs[score_inds[:, 0]], score_inds[:, 1]]) # <sub, obj, predicate>
            rel_scores = triplet_scores[score_inds[:, 0], score_inds[:, 1]]
            pred_top_rel_pairs.append(pred_rels)

            # gt
            boxes = box_ops.box_cxcywh_to_xyxy(frame_gt['boxes'])
            scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).unsqueeze(0).to(boxes.device)
            gt_boxes = (boxes * scale_fct).cpu().numpy()

            gt_classes = frame_gt['labels'].cpu().numpy()
            gt_relations = frame_gt['relation_map'].nonzero().cpu().numpy()
            if len(gt_relations) == 0: # skip frames with no gt relations, follow ST-HOI
                continue
            self.gts.append({
                'annotations': [{'bbox': bbox, 'category_id': label} for bbox, label in zip(gt_boxes, gt_classes)],
                'hoi_annotation': [{'subject_id': hoi[0], 'object_id': hoi[1], 'category_id': hoi[2]} for hoi in gt_relations]
            })
            for hoi in self.gts[-1]['hoi_annotation']:
                triplet = (self.gts[-1]['annotations'][hoi['subject_id']]['category_id'],
                           self.gts[-1]['annotations'][hoi['object_id']]['category_id'],
                           hoi['category_id'])
                if triplet not in self.gt_triplets: self.gt_triplets.append(triplet)
                if triplet not in self.sum_gts: self.sum_gts[triplet] = 0
                self.sum_gts[triplet] += 1

            self.preds.append({
                'predictions': [{'bbox': bbox, 'category_id': label} for bbox, label in zip(pred_boxes, pred_classes)],
                'hoi_prediction': [{'subject_id': hoi[0], 'object_id': hoi[1], 'category_id': hoi[2], 'score': score}
                                   for hoi, score in zip(pred_rels, rel_scores)]
            })
            # self.preds.append({
            #     'predictions': [{'bbox': bbox, 'category_id': label} for bbox, label in zip(gt_boxes, gt_classes)],
            #     'hoi_prediction': [{'subject_id': hoi[0], 'object_id': hoi[1], 'category_id': hoi[2], 'score': 1} for hoi in gt_relations]
            # }) # test evaluate with GT

        return pred_top_rel_pairs

    def synchronize_between_processes(self):
        self.gts = list(itertools.chain(*utils.all_gather(self.gts)))
        self.preds = list(itertools.chain(*utils.all_gather(self.preds)))
        self.gt_triplets = list(set(itertools.chain(*utils.all_gather(self.gt_triplets))))
        assert len(self.gts) == len(self.preds)

        all_sum_gts = utils.all_gather(self.sum_gts)
        merged_sum_gts = all_sum_gts[0].copy()
        for single_gts in all_sum_gts[1:]:
            for triplet, count in single_gts.items():
                if triplet in merged_sum_gts:
                    merged_sum_gts[triplet] += count
                else:
                    merged_sum_gts[triplet] = count
        self.sum_gts = merged_sum_gts

    def evaluate(self):
        for img_id, (img_preds, img_gts) in enumerate(zip(self.preds, self.gts)):
            print(f"Evaluating Score Matrix... : [{(img_id+1):>4}/{len(self.gts):<4}]", flush=True, end="\r")
            pred_bboxes = img_preds['predictions']
            gt_bboxes = img_gts['annotations']
            pred_hois = img_preds['hoi_prediction']
            gt_hois = img_gts['hoi_annotation']
            if len(gt_bboxes) != 0:
                if len(pred_bboxes) == 0: continue
                bbox_pairs, bbox_overlaps = self.compute_iou_mat(gt_bboxes, pred_bboxes)
                self.compute_fptp(pred_hois, gt_hois, bbox_pairs, pred_bboxes, bbox_overlaps)
            else:
                for pred_hoi in pred_hois:
                    triplet = [pred_bboxes[pred_hoi['subject_id']]['category_id'],
                               pred_bboxes[pred_hoi['object_id']]['category_id'], pred_hoi['category_id']]
                    if triplet not in self.gt_triplets:
                        continue
                    self.tp[triplet].append(0)
                    self.fp[triplet].append(1)
                    self.score[triplet].append(pred_hoi['score'])
        print(f"[stats] Score Matrix Generation completed!!          ")
        map = self.compute_map()
        return map

    # refer to: https://github.com/coldmanck/VidHOI/blob/master/vidor_eval.ipynb
    def set_rare_nonrare_triplets(self, count_threshold=25):
        rare_triplets, nonrare_triplets = [], []
        for triplet, count in self.sum_gts.items():
            if count < count_threshold:
                rare_triplets.append(triplet)
            else:
                nonrare_triplets.append(triplet)
        return rare_triplets, nonrare_triplets

    def compute_map(self):
        ap = defaultdict(lambda: 0)
        max_recall = defaultdict(lambda: 0)
        temporal_predicate_inds = [self.predicates.index(p) for p in TEMPORAL_predicates]
        temporal_ap = defaultdict(lambda: 0)
        non_temporal_ap = defaultdict(lambda: 0)
        per_predicate_stats = {}

        # rare & nonrare eval
        rare_triplets, nonrare_triplets = self.set_rare_nonrare_triplets()
        rare_ap = defaultdict(lambda: 0)
        nonrare_ap = defaultdict(lambda: 0)

        for triplet in self.gt_triplets:
            sum_gts = self.sum_gts[triplet]
            if sum_gts == 0:
                continue

            tp = np.array((self.tp[triplet]))
            fp = np.array((self.fp[triplet]))
            if len(tp) == 0:
                # ST-HOI just skip these triplets, it's a bug!! (https://github.com/coldmanck/VidHOI/blob/master/vidor_eval.ipynb)
                ap[triplet] = 0
                max_recall[triplet] = 0
                if triplet[-1] in temporal_predicate_inds:
                    temporal_ap[triplet] = 0
                else:
                    non_temporal_ap[triplet] = 0

                if triplet in rare_triplets:
                    rare_ap[triplet] = 0
                elif triplet in nonrare_triplets:
                    nonrare_ap[triplet] = 0
                continue

            score = np.array(self.score[triplet])
            sort_inds = np.argsort(-score)
            fp = fp[sort_inds]
            tp = tp[sort_inds]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / sum_gts
            prec = tp / (fp + tp)
            ap[triplet] = self.voc_ap(rec, prec)
            max_recall[triplet] = np.amax(rec)
            if triplet[-1] in temporal_predicate_inds:
                temporal_ap[triplet] = ap[triplet]
            else:
                non_temporal_ap[triplet] = ap[triplet]

            if triplet in rare_triplets:
                rare_ap[triplet] = ap[triplet]
            elif triplet in nonrare_triplets:
                nonrare_ap[triplet] = ap[triplet]

            # per predicate stats
            predicate_name = self.predicates[triplet[-1]]
            if predicate_name not in per_predicate_stats:
                per_predicate_stats[predicate_name] = {
                    'is_temporal': predicate_name in TEMPORAL_predicates,
                    'triplets': [],
                    'triplet_aps': [],
                    'triplets_gt_counts': []
                }

            per_predicate_stats[predicate_name]['triplets'].append(triplet)
            per_predicate_stats[predicate_name]['triplet_aps'].append(ap[triplet])
            per_predicate_stats[predicate_name]['triplets_gt_counts'].append(self.sum_gts[triplet])

        m_ap = np.mean(list(ap.values())) * 100 # percentage
        m_max_recall = np.mean(list(max_recall.values())) * 100
        temporal_m_ap = np.mean(list(temporal_ap.values())) * 100
        non_temporal_m_ap = np.mean(list(non_temporal_ap.values())) * 100
        # plt.hist(list(ap.values()), bins=20); plt.show()

        m_rare_ap = np.mean(list(rare_ap.values())) * 100
        m_nonrare_ap = np.mean(list(nonrare_ap.values())) * 100

        print(f'======================#total triplets={len(self.gt_triplets)} (Temporal/Spatial={len(temporal_ap)}/{len(non_temporal_ap)}, Rare/Non-rare={len(rare_ap)}/{len(nonrare_ap)}), #frames={len(self.gts)}======================')
        print(f'mAP (Full / Temporal / Spatial): {m_ap:.2f} / {temporal_m_ap:.2f} / {non_temporal_m_ap:.2f} || {m_rare_ap:.2f} / {m_nonrare_ap:.2f}, mR (Full): {m_max_recall:.2f}')

        ## per-predicate evaluation results
        print(f'======================Per-predicate======================')
        print(f"name,\tis_temporal,\tgt_counts,\tp-mAP,\tp-wmAP")
        for predicate, stat in per_predicate_stats.items():
            predicate_mAP = sum(stat['triplet_aps']) / len(stat['triplet_aps'])
            predicate_wmAP = sum(np.array(stat['triplet_aps']) * np.array(stat['triplets_gt_counts'])) / sum(stat['triplets_gt_counts']) # weighted
            print(f"{predicate},\t{stat['is_temporal']},\t{sum(stat['triplets_gt_counts'])},\t{predicate_mAP * 100 :.2f},\t{predicate_wmAP * 100 :.2f}")
            per_predicate_stats[predicate].update({'mAP': predicate_mAP, 'wmAP': predicate_wmAP})
        # pmAP_all = np.mean([x['mAP'] for x in per_predicate_stats.values()]) * 100
        # pmAP_temporal = np.mean([x['mAP'] for x in per_predicate_stats.values() if x['is_temporal']]) * 100
        # pmAP_spatial = np.mean([x['mAP'] for x in per_predicate_stats.values() if not x['is_temporal']]) * 100
        # print(f"\n Predicate-level mAP (Full / Temporal / Spatial): {pmAP_all:.2f} / {pmAP_temporal:.2f} / {pmAP_spatial:.2f}")

        return {'mAP': m_ap, 'mean max recall': m_max_recall}

    def voc_ap(self, rec, prec):
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
        return ap

    def compute_fptp(self, pred_hois, gt_hois, match_pairs, pred_bboxes, bbox_overlaps):
        pos_pred_ids = match_pairs.keys()
        vis_tag = np.zeros(len(gt_hois))
        pred_hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        if len(pred_hois) != 0:
            for pred_hoi in pred_hois:
                is_match = 0
                if len(match_pairs) != 0 and pred_hoi['subject_id'] in pos_pred_ids and pred_hoi['object_id'] in pos_pred_ids:
                    pred_sub_ids = match_pairs[pred_hoi['subject_id']]
                    pred_obj_ids = match_pairs[pred_hoi['object_id']]
                    pred_sub_overlaps = bbox_overlaps[pred_hoi['subject_id']]
                    pred_obj_overlaps = bbox_overlaps[pred_hoi['object_id']]
                    pred_category_id = pred_hoi['category_id']
                    max_overlap = 0
                    max_gt_hoi = 0
                    for gt_hoi in gt_hois:
                        if gt_hoi['subject_id'] in pred_sub_ids and gt_hoi['object_id'] in pred_obj_ids and pred_category_id == gt_hoi['category_id']:
                            is_match = 1
                            min_overlap_gt = min(pred_sub_overlaps[pred_sub_ids.index(gt_hoi['subject_id'])],
                                                 pred_obj_overlaps[pred_obj_ids.index(gt_hoi['object_id'])])
                            if min_overlap_gt > max_overlap:
                                max_overlap = min_overlap_gt
                                max_gt_hoi = gt_hoi
                triplet = (pred_bboxes[pred_hoi['subject_id']]['category_id'], pred_bboxes[pred_hoi['object_id']]['category_id'], pred_hoi['category_id'])
                if triplet not in self.gt_triplets:
                    continue
                if is_match == 1 and vis_tag[gt_hois.index(max_gt_hoi)] == 0:
                    self.fp[triplet].append(0)
                    self.tp[triplet].append(1)
                    vis_tag[gt_hois.index(max_gt_hoi)] = 1
                else:
                    self.fp[triplet].append(1)
                    self.tp[triplet].append(0)
                self.score[triplet].append(pred_hoi['score'])

    def compute_iou_mat(self, bbox_list1, bbox_list2):
        iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
        if len(bbox_list1) == 0 or len(bbox_list2) == 0:
            return {}
        for i, bbox1 in enumerate(bbox_list1):
            for j, bbox2 in enumerate(bbox_list2):
                iou_i = self.compute_IOU(bbox1, bbox2)
                iou_mat[i, j] = iou_i

        iou_mat_ov=iou_mat.copy()
        iou_mat[iou_mat>=self.overlap_iou] = 1
        iou_mat[iou_mat<self.overlap_iou] = 0

        match_pairs = np.nonzero(iou_mat)
        match_pairs_dict = {}
        match_pair_overlaps = {}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[1]):
                if pred_id not in match_pairs_dict.keys():
                    match_pairs_dict[pred_id] = []
                    match_pair_overlaps[pred_id]=[]
                match_pairs_dict[pred_id].append(match_pairs[0][i])
                match_pair_overlaps[pred_id].append(iou_mat_ov[match_pairs[0][i],pred_id])
        return match_pairs_dict, match_pair_overlaps

    def compute_IOU(self, bbox1, bbox2):
        if isinstance(bbox1['category_id'], str):
            bbox1['category_id'] = int(bbox1['category_id'].replace('\n', ''))
        if isinstance(bbox2['category_id'], str):
            bbox2['category_id'] = int(bbox2['category_id'].replace('\n', ''))
        if bbox1['category_id'] == bbox2['category_id']:
            rec1 = bbox1['bbox']
            rec2 = bbox2['bbox']
            # computing area of each rectangles
            S_rec1 = (rec1[2] - rec1[0]+1) * (rec1[3] - rec1[1]+1)
            S_rec2 = (rec2[2] - rec2[0]+1) * (rec2[3] - rec2[1]+1)

            # computing the sum_area
            sum_area = S_rec1 + S_rec2

            # find the each edge of intersect rectangle
            left_line = max(rec1[1], rec2[1])
            right_line = min(rec1[3], rec2[3])
            top_line = max(rec1[0], rec2[0])
            bottom_line = min(rec1[2], rec2[2])
            # judge if there is an intersect
            if left_line >= right_line or top_line >= bottom_line:
                return 0
            else:
                intersect = (right_line - left_line+1) * (bottom_line - top_line+1)
                return intersect / (sum_area - intersect)
        else:
            return 0
