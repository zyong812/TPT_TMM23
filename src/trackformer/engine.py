# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import logging
import math
import os
import sys
from typing import Iterable

import torch
from track import ex
import copy
from torch.utils.data import DataLoader
from tqdm import tqdm
import motmetrics as mm
import numpy as np
import time

from trackformer.datasets import get_coco_api_from_dataset
from trackformer.datasets.coco_eval import CocoEvaluator
from trackformer.datasets.panoptic_eval import PanopticEvaluator
from trackformer.datasets.actiongenome_eval import BasicSceneGraphEvaluator
from trackformer.datasets.vidhoi_eval import VidHOIEvaluator
from trackformer.models.detr_segmentation import DETRSegm
from trackformer.util import misc as utils
from trackformer.util.box_ops import box_iou, box_cxcywh_to_xywh, box_xyxy_to_xywh
from trackformer.util.track_utils import evaluate_mot_accums
from trackformer.vis import vis_results
from trackformer.util.plot_utils import check_prediction, check_annotation
from base_trackers import CentroidTracker, SORT, IOUTracker, BYTETracker

def make_results(outputs, targets, postprocessors, tracking, return_only_orig=True):
    target_sizes = torch.stack([t["size"] for t in targets], dim=0)
    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

    results = None
    if not return_only_orig:
        results = postprocessors['bbox'](outputs, target_sizes)
    results_orig = postprocessors['bbox'](outputs, orig_target_sizes)

    # # targets as predictions
    # results_orig = [
    #     {
    #         'scores': torch.ones_like(targets[0]['labels']),
    #         'scores_no_object': torch.zeros_like(targets[0]['labels']),
    #         'labels': targets[0]['labels'],
    #         'boxes': postprocessors['bbox'].process_boxes(targets[0]['boxes'], orig_target_sizes)[0],
    #     }
    # ]

    if 'segm' in postprocessors:
        results_orig = postprocessors['segm'](
            results_orig, outputs, orig_target_sizes, target_sizes)
        if not return_only_orig:
            results = postprocessors['segm'](
                results, outputs, target_sizes, target_sizes)

    if results is None:
        return results_orig, results

    for i, result in enumerate(results):
        target = targets[i]
        target_size = target_sizes[i].unsqueeze(dim=0)

        result['target'] = {}
        result['boxes'] = result['boxes'].cpu()

        # revert boxes for visualization
        for key in ['boxes', 'track_query_boxes']:
            if key in target:
                target[key] = postprocessors['bbox'].process_boxes(
                    target[key], target_size)[0].cpu()

        if 'random_boxes' in target:
            random_target_sizes = torch.stack([t["random_size"] for t in targets], dim=0)
            target['random_boxes'] = postprocessors['bbox'].process_boxes(
                target['random_boxes'], random_target_sizes[i].unsqueeze(dim=0))[0].cpu()

        if tracking and 'prev_boxes' in target:
            prev_target_sizes = torch.stack([t["prev_size"] for t in targets], dim=0)
            target['prev_boxes'] = postprocessors['bbox'].process_boxes(
                target['prev_boxes'], prev_target_sizes[i].unsqueeze(dim=0))[0].cpu()

            if len(target['track_query_match_ids']):
                track_queries_iou, _ = box_iou(
                    target['boxes'][target['track_query_match_ids']],
                    result['boxes'])
                track_queries_match_mask = target['track_queries_match_mask']

                box_ids = [box_id for box_id, mask_value in enumerate(track_queries_match_mask == 1)
                           if mask_value]

                result['track_queries_with_id_iou'] = torch.diagonal(track_queries_iou[:, box_ids])

    return results_orig, results


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, visualizers: dict, args):

    vis_iter_metrics = None
    if visualizers:
        vis_iter_metrics = visualizers['iter_metrics']

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(
        args.vis_and_log_interval,
        delimiter="  ",
        vis=vis_iter_metrics,
        debug=args.debug)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, epoch)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # in order to be able to modify targets inside the forward call we need
        # to pass it through as torch.nn.parallel.DistributedDataParallel only
        # passes copies

        outputs, targets, *_ = model(samples, targets)

        if isinstance(outputs, list): # per-frame loss computation
            loss_dict_list = [criterion(o, [t]) for o, t in zip(outputs, targets)]
            frame_num = len(loss_dict_list)
            loss_dict = {k: sum([ld[k] for ld in loss_dict_list])/frame_num for k in loss_dict_list[0]}
        else:
            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            print(targets)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value,
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"],
                             lr_backbone=optimizer.param_groups[1]["lr"])

        if visualizers and (i == 0 or not i % args.vis_and_log_interval):
            _, results = make_results(
                outputs, targets, postprocessors, args.tracking, return_only_orig=False)

            vis_results(
                visualizers['example_results'],
                samples.tensors[0],
                results[0],
                targets[0],
                args.tracking)

        # print('visualizing')
        # for j in range(len(targets)): check_annotation(samples, targets, idx=j)
        # for j in range(len(targets)): check_prediction(samples, outputs, targets=targets, idx=j)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device,
             output_dir: str, visualizers: dict, args, epoch: int = None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(
        args.vis_and_log_interval,
        delimiter="  ",
        debug=args.debug)
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    base_ds = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = tuple(k for k in ('bbox', 'segm') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    if args.hoi_detection:
        actiongenome_hoi_evaluator = BasicSceneGraphEvaluator(mode='sgdet', constraint=False)
        vidhoi_evaluator = VidHOIEvaluator(args) if args.dataset == 'vidhoi' else None

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 'Test:')):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs, targets, *_ = model(samples, targets)

        # print('check results')
        # for j in range(len(targets)): check_annotation(samples, targets, idx=j)
        # for j in range(len(targets)): check_prediction(samples, outputs, targets=targets, idx=j)

        if isinstance(outputs, list): # per-frame loss computation
            loss_dict_list = [criterion(o, [t]) for o, t in zip(outputs, targets)]
            frame_num = len(loss_dict_list)
            loss_dict = {k: sum([ld[k] for ld in loss_dict_list])/frame_num for k in loss_dict_list[0]}
        else:
            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        if visualizers and (i == 0 or not i % args.vis_and_log_interval):
            results_orig, results = make_results(
                outputs, targets, postprocessors, args.tracking, return_only_orig=False)

            vis_results(
                visualizers['example_results'],
                samples.tensors[0],
                results[0],
                targets[0],
                args.tracking)
        else:
            if isinstance(outputs, list):
                targets = [targets[-1]] # only evaluate the last frame detection performance
                results_orig, _ = make_results(outputs[-1], targets, postprocessors, args.tracking)
            else:
                results_orig, _ = make_results(outputs, targets, postprocessors, args.tracking)

        # TODO. remove cocoDts from coco eval and change example results output
        if coco_evaluator is not None:
            results_orig = {
                target['image_id'].item(): output
                for target, output in zip(targets, results_orig)}
                # for target, output in zip([targets[-1]], [results_orig[-1]])}
            coco_evaluator.update(copy.deepcopy(results_orig))

        if panoptic_evaluator is not None:
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for j, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[j]["image_id"] = image_id
                res_pano[j]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        if args.hoi_detection:
            actiongenome_hoi_evaluator.evaluate_scene_graph(targets, outputs, box_preds=results_orig)
            if vidhoi_evaluator is not None:
                vidhoi_evaluator.update(targets, outputs, box_preds=results_orig)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if args.hoi_detection:
        actiongenome_hoi_evaluator.synchronize_between_processes()
        actiongenome_hoi_evaluator.print_stats()
        if vidhoi_evaluator is not None:
            vidhoi_evaluator.synchronize_between_processes()
            vidhoi_evaluator.evaluate()
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        print(f"#images={len(coco_evaluator.coco_eval['bbox'].params.imgIds)}")
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in coco_evaluator.coco_eval:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in coco_evaluator.coco_eval:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    # stats = {}
    # TRACK EVAL
    if args.tracking and args.tracking_eval:
        stats['track_bbox'] = []

        ex.logger = logging.getLogger("submitit")

        # distribute evaluation of seqs to processes
        seqs = data_loader.dataset.sequences
        seqs_per_rank = {i: [] for i in range(utils.get_world_size())}
        for i, seq in enumerate(seqs):
            rank = i % utils.get_world_size()
            seqs_per_rank[rank].append(seq)

        # only evaluarte one seq in debug mode
        if args.debug:
            seqs_per_rank = {k: v[:1] for k, v in seqs_per_rank.items()}
            seqs = [s for ss in seqs_per_rank.values() for s in ss]

        dataset_name = seqs_per_rank[utils.get_rank()]
        if not dataset_name:
            dataset_name = seqs_per_rank[0]

        model_without_ddp = model
        if args.distributed:
            model_without_ddp = model.module

        # mask prediction is too slow and consumes a lot of memory to
        # run it during tracking training.
        if isinstance(model, DETRSegm):
            model_without_ddp = model_without_ddp.detr

        obj_detector_model = {
            'model': model_without_ddp,
            'post': postprocessors,
            'img_transform': args.img_transform}

        run = ex.run(config_updates={
            'seed': None,
            'dataset_name': dataset_name,
            'frame_range': data_loader.dataset.frame_range,
            'obj_detector_model': obj_detector_model})

        mot_accums = utils.all_gather(run.result)[:len(seqs)]
        mot_accums = [item for sublist in mot_accums for item in sublist]

        # we compute seqs results on muliple nodes but evaluate the accumulated
        # results due to seqs being weighted differently (seg length)
        eval_summary, eval_summary_str = evaluate_mot_accums(
            mot_accums, seqs)
        print(eval_summary_str)

        for metric in ['mota', 'idf1']:
            eval_m = eval_summary[metric]['OVERALL']
            stats['track_bbox'].append(eval_m)

    eval_stats = stats['coco_eval_bbox'][:3]
    if 'coco_eval_masks' in stats:
        eval_stats.extend(stats['coco_eval_masks'][:3])
    if 'track_bbox' in stats:
        eval_stats.extend(stats['track_bbox'])

    # VIS
    if visualizers:
        vis_epoch = visualizers['epoch_metrics']
        y_data = [stats[legend_name] for legend_name in vis_epoch.viz_opts['legend']]
        vis_epoch.plot(y_data, epoch)

        visualizers['epoch_eval'].plot(eval_stats, epoch)

    if args.debug:
        exit()

    return eval_stats, coco_evaluator

@torch.no_grad()
def evaluate_video_sgg(model, dataset_val, device, args, postprocessors, query_propagation_threshold=0.5, tracking_det_threshold=0.8):
    if args.distributed:
        model.module.tracking()
    else:
        model.tracking()
    print(f"Tracker: {args.sgg_postprocessing_tracker}")

    # evaluators
    base_ds = get_coco_api_from_dataset(dataset_val)
    coco_evaluator = CocoEvaluator(base_ds, ('bbox',))
    mot_accums = []
    if args.hoi_detection:
        actiongenome_hoi_evaluator = BasicSceneGraphEvaluator(mode='sgdet', constraint=False, dataset=args.dataset)
        vidhoi_evaluator = VidHOIEvaluator(args) if args.dataset == 'vidhoi' else None

    # loading val frames by video
    dataset_val.clip_mode = False # !! will affect offline evaluation
    video_names, video_startend_idxs = dataset_val.sequence_infos()
    if utils.get_world_size() > 1: # for multi-GPUs
        video_names = video_names[utils.get_rank()::utils.get_world_size()]
        video_startend_idxs = video_startend_idxs[utils.get_rank()::utils.get_world_size()]

    start_time = time.time(); model_inference_time = 0
    # video_names, video_startend_idxs = video_names[12:14], video_startend_idxs[12:14]
    for vid, (video_name, v_seg_info) in enumerate(zip(video_names, video_startend_idxs)):
        # if vid < 11: continue
        print(f'TRACK SEQ: {video_name} ({vid}/{len(video_names)})')
        video_loader = DataLoader(
            torch.utils.data.Subset(dataset_val, range(v_seg_info[0], v_seg_info[1]+1)),
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers
        )

        # trackers
        if args.sgg_postprocessing_tracker == 'IOUTracker':
            tracker = IOUTracker(iou_threshold=0.1)
        elif args.sgg_postprocessing_tracker == 'SORT':
            tracker = SORT(iou_threshold=0.1)
        elif args.sgg_postprocessing_tracker == 'BYTETracker':
            tracking_det_threshold = 1e-2
            tracker = BYTETracker(iou_threshold=0.1)
        else:
            tracker = QuerySlotTracker(num_queries=args.num_queries)

        # actiongenome_hoi_evaluator.reset_result()
        mot_accums.append(mm.MOTAccumulator(auto_id=True))
        kept_box_qids, prev_states = None, {}
        for frame_id, (samples, targets) in enumerate(tqdm(video_loader, file=sys.stdout)):
            assert len(targets) == 1
            frame_tic = time.time()

            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            targets[0].update(prev_states)

            if args.hoi_detection and args.hoi_oracle_mode and 'prev_track_ids' in prev_states:
                # match track ids between frames
                target_ind_match_matrix = prev_states['prev_track_ids'].unsqueeze(dim=1).eq(targets[0]['track_ids'])
                target_ind_matching = target_ind_match_matrix.any(dim=1)
                target_ind_matched_idx = target_ind_match_matrix.nonzero()[:, 1]

                # index of prev frame detection in current frame box list
                targets[0]['track_query_match_ids'] = target_ind_matched_idx
                tracked_qids = prev_states['prev_out_ind'][target_ind_matching]

                # match mask to next frame
                track_queries_match_mask = torch.zeros(args.num_queries).float()
                track_queries_match_mask[tracked_qids] = 1 # tracked in current frame
                track_queries_match_mask[prev_states['prev_out_ind'][~target_ind_matching]] = -1 # disappeared in current frame
                targets[0]['track_queries_match_mask'] = track_queries_match_mask.to(device)

            outputs, *_ = model(samples, targets)

            # collect frame-level evaluation
            results_post, _ = make_results(outputs, targets, postprocessors, tracking=False)
            results_orig = {target['image_id'].item(): output for target, output in zip(targets, results_post)}

            if args.hoi_detection and args.hoi_oracle_mode:
                kept_box_mask = torch.zeros_like(results_post[0]['scores']).bool()
                kept_box_mask[outputs['match_res'][0]] = True
                ## transfer matching to next frame
                prev_states.update({
                    'prev_track_ids': targets[0]['track_ids'][outputs['match_res'][1]],
                    'prev_out_ind': outputs['match_res'][0]
                })
                # print(outputs['match_res'][0], targets[0]['track_ids'][outputs['match_res'][1]])
            else:
                # NMS before propagation
                suppress_ids = apply_nms(results_post[0], kept_box_qids)
                results_post[0]['scores'][suppress_ids] = 0
                kept_box_mask = results_post[0]['scores'] > query_propagation_threshold # box kept for propagation
                kept_box_qids = kept_box_mask.nonzero()[:, 0].tolist()

            # detection evaluation
            coco_evaluator.update(copy.deepcopy(results_orig))
            if args.hoi_detection:
                top_pred_rel_pairs = actiongenome_hoi_evaluator.evaluate_scene_graph(targets, outputs, box_preds=results_orig)
                if vidhoi_evaluator is not None:
                    top_pred_rel_pairs = vidhoi_evaluator.update(targets, outputs, box_preds=results_orig)

            # check_annotation(samples, targets, idx=0)
            # check_prediction(samples, outputs, targets=targets, idx=0, threshold=0.2,
            #                  top_pred_rel_pairs=top_pred_rel_pairs, save_fig_dir=f"{args.output_dir}/demo/{video_name}")

            # mot eval
            tracking_kept_box_mask = results_post[0]['scores'] > tracking_det_threshold # output tracked boxes
            if isinstance(tracker, QuerySlotTracker):
                pred_boxes = box_cxcywh_to_xywh(outputs['pred_boxes'][0][tracking_kept_box_mask]).cpu().numpy()
                pred_labels = results_post[0]['labels'][tracking_kept_box_mask].cpu().numpy()
                pred_track_ids = tracker.update(frame_id, tracking_kept_box_mask.nonzero()[:, 0].tolist())
            else:
                tracks = tracker.update(box_cxcywh_to_xywh(outputs['pred_boxes'][0][tracking_kept_box_mask]).cpu().numpy(),
                                    results_post[0]['scores'][tracking_kept_box_mask].cpu().numpy(),
                                    results_post[0]['labels'][tracking_kept_box_mask].cpu().numpy())
                pred_boxes, pred_track_ids, pred_labels = [], [], []
                for track in tracks:
                    frame_index, track_id, bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion = track
                    pred_boxes.append([bbox_left, bbox_top, bbox_width, bbox_height])
                    pred_track_ids.append(track_id)
                    pred_labels.append(object_category)
                pred_boxes, pred_labels = np.array(pred_boxes), np.array(pred_labels)

            gt_boxes, gt_track_ids, gt_labels = box_cxcywh_to_xywh(targets[0]['boxes']).cpu().numpy(), targets[0]['track_ids'].tolist(), targets[0]['labels'].cpu().numpy()
            distance = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
            if len(distance) > 0:
                distance = np.where(gt_labels[:, None] == pred_labels[None, :], distance, np.nan)
            mot_accums[-1].update(gt_track_ids, pred_track_ids, distance)

            # temporal propagation
            prev_states.update({
                'track_query_boxes': outputs['pred_boxes'][0],
                'track_query_hs_embeds': outputs['hs_embed'][0].to(device),
                'track_token_propagation_mask': kept_box_mask.float()
            })
            if args.hoi_relation_propagation_on_inference and len(top_pred_rel_pairs[0]) > 0:
                prev_states.update({'prev_top_rel_pairs': torch.from_numpy(np.unique(top_pred_rel_pairs[0][:args.num_hoi_queries//2, :2], axis=0))})
            if args.hoi_detection and args.hoi_use_temporal_dynamics:
                if 'temporal_dynamics_feature_bank' in prev_states:
                    prev_states['temporal_dynamics_feature_bank'] = torch.cat((outputs['hs_embed'].to(device), prev_states['temporal_dynamics_feature_bank']), dim=0)[:args.hoi_use_temporal_dynamics_prev_length]
                    prev_states['temporal_dynamics_feature_mask'] = torch.cat(((kept_box_mask.float()==0).unsqueeze(0), prev_states['temporal_dynamics_feature_mask']), dim=0)[:args.hoi_use_temporal_dynamics_prev_length]
                else:
                    prev_states['temporal_dynamics_feature_bank'] = outputs['hs_embed'].to(device)
                    prev_states['temporal_dynamics_feature_mask'] = (kept_box_mask.float()==0).unsqueeze(0)
            model_inference_time += time.time() - frame_tic

    total_inference_time = time.time() - start_time
    # eval results
    mot_accums_all, video_names_all = utils.all_gather(mot_accums), utils.all_gather(video_names)
    eval_summary, eval_summary_str = evaluate_mot_accums(sum(mot_accums_all, []), sum(video_names_all, []))
    print(eval_summary_str)
    print(f'#videos={len(sum(video_names_all, []))}')

    print(f"model_inference_time={model_inference_time}s, total_inference_time={total_inference_time}s (include data loading, processing etc.)")
    if args.hoi_detection:
        actiongenome_hoi_evaluator.synchronize_between_processes()
        actiongenome_hoi_evaluator.print_stats()
        if vidhoi_evaluator is not None:
            vidhoi_evaluator.synchronize_between_processes()
            vidhoi_evaluator.evaluate()

            print(f"Model: time_per_frame={model_inference_time/len(vidhoi_evaluator.gts)*1000 :.2f}ms, frame_per_second={len(vidhoi_evaluator.gts)/model_inference_time :.2f}")
            print(f"Model(+dataLoading etc.): time_per_frame={total_inference_time/len(vidhoi_evaluator.gts)*1000 :.2f}ms, frame_per_second={len(vidhoi_evaluator.gts)/total_inference_time :.2f}")

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return

# merge boxes (NMS)
def apply_nms(res, kept_box_qids=None, threshold=0.7):
    inst_scores, inst_labels, xyxy_boxes = res['scores'].clone(), res['labels'], res['boxes']
    if kept_box_qids is not None:
        inst_scores[kept_box_qids] *= 4 # we prefer to keep tracked boxes in previous frames

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

class QuerySlotTracker:
    def __init__(self, num_queries, max_lost=30):
        self.num_queries = num_queries
        self.max_lost = max_lost

        # states
        self.query_been_activated = np.zeros(num_queries, dtype=bool)
        self.query_last_tracked_frame_id = np.zeros(num_queries, dtype=int) - 1
        self.query_assigned_track_ids = np.arange(num_queries)

    def update(self, current_frame_id, detected_query_ids):
        track_ids = []
        for qid in detected_query_ids:
            if self.query_been_activated[qid] and (current_frame_id - self.query_last_tracked_frame_id[qid] > self.max_lost):
                self.query_assigned_track_ids[qid] += self.num_queries # assign new track_id to query slot
                # print(f'{current_frame_id} - {self.query_last_tracked_frame_id[qid]} ==> {self.query_assigned_track_ids[qid]}')

            self.query_been_activated[qid] = True
            self.query_last_tracked_frame_id[qid] = current_frame_id
            track_ids.append(self.query_assigned_track_ids[qid])

        return track_ids
