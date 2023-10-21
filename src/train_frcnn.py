import sacred
import torch
import yaml
import os
import numpy as np
import random
import time
import datetime
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler
import math
import sys
import torchvision.models.detection.mask_rcnn
import torchvision
from tqdm import tqdm
import motmetrics as mm
from STTran.sttran import build_frcnn
from base_trackers import CentroidTracker, SORT, IOUTracker, BYTETracker

import trackformer.util.misc as utils
from trackformer.util.misc import nested_dict_to_namespace
from trackformer.datasets import build_dataset
from trackformer.datasets import get_coco_api_from_dataset
from trackformer.datasets.coco_eval import CocoEvaluator
from trackformer.util.box_ops import box_xyxy_to_xywh
from trackformer.util.track_utils import evaluate_mot_accums

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(print_freq, delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # todo: same data format
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        for t in targets: t['labels'] += 1 # compatibility with torchvison frcnn (set __background__=0)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, print_freq=100):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(print_freq, delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        for o in outputs: o['labels'] -= 1 # for compatibility with evaluation code
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

@torch.no_grad()
def evaludate_mot(model, dataset_val, device, chosen_tracker, det_threshold=0.8):
    print(f"Tracker: {chosen_tracker}")
    model.eval()
    mot_accums = []
    video_names, video_startend_idxs = dataset_val.sequence_infos()
    if utils.get_world_size() > 1: # for multi-GPUs
        video_names = video_names[utils.get_rank()::utils.get_world_size()]
        video_startend_idxs = video_startend_idxs[utils.get_rank()::utils.get_world_size()]

    for vid, (video_name, v_seg_info) in enumerate(zip(video_names, video_startend_idxs)):
        print(f'TRACK SEQ: {video_name} ({vid}/{len(video_names)})')
        video_loader = DataLoader(
            torch.utils.data.Subset(dataset_val, range(v_seg_info[0], v_seg_info[1]+1)),
            collate_fn=utils.frcnn_collate_fn,
            num_workers=args.num_workers
        )

        # trackers
        if chosen_tracker == 'IOUTracker':
            tracker, det_threshold = IOUTracker(iou_threshold=0.1), 0.8
        elif chosen_tracker == 'SORT':
            tracker, det_threshold = SORT(iou_threshold=0.1), 0.8
        elif chosen_tracker == 'BYTETracker':
            tracker, det_threshold = BYTETracker(iou_threshold=0.1), 0 # filter by confidence score inside

        # track one video
        mot_accums.append(mm.MOTAccumulator(auto_id=True))
        for i, (frames, targets) in enumerate(tqdm(video_loader, file=sys.stdout)):
            assert len(targets) == 1
            frames = list(img.to(device) for img in frames)
            frame_det_out = model(frames)[0]

            # get trackers
            kept_box_mask = frame_det_out['scores'] > det_threshold
            tracks = tracker.update(box_xyxy_to_xywh(frame_det_out['boxes'][kept_box_mask]).cpu().numpy(),
                                    frame_det_out['scores'][kept_box_mask].cpu().numpy(),
                                    frame_det_out['labels'][kept_box_mask].cpu().numpy()-1)
            pred_boxes, pred_track_ids, pred_labels = [], [], []
            for track in tracks:
                frame_index, track_id, bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion = track
                pred_boxes.append([bbox_left, bbox_top, bbox_width, bbox_height])
                pred_track_ids.append(track_id)
                pred_labels.append(object_category)

            # mot eval
            gt_boxes, gt_track_ids, gt_labels = box_xyxy_to_xywh(targets[0]['boxes']).cpu().numpy(), targets[0]['track_ids'].tolist(), targets[0]['labels'].cpu().numpy()
            distance = mm.distances.iou_matrix(gt_boxes, np.array(pred_boxes), max_iou=0.5)
            if len(gt_labels) == 0 or len(pred_labels) == 0:
                label_match = np.empty((0, 0))
            else:
                label_match = (gt_labels[:, None] == np.array(pred_labels)[None, :])
            distance = np.where(label_match, distance, np.nan)
            mot_accums[-1].update(gt_track_ids, pred_track_ids, distance)

    mot_accums_all, video_names_all = utils.all_gather(mot_accums), utils.all_gather(video_names)
    eval_summary, eval_summary_str = evaluate_mot_accums(sum(mot_accums_all, []), sum(video_names_all, []))
    print(eval_summary_str)
    print(f'#videos={len(sum(video_names_all, []))}')

########## configs ##########
ex = sacred.Experiment('train')
ex.add_config('cfgs/train.yaml')
ex.add_named_config('vidhoi', 'cfgs/train_vidhoi.yaml')
ex.add_named_config('frcnn', 'cfgs/train_frcnn.yaml')

@ex.main
def load_config(_config, _run):
    """ We use sacred only for config loading from YAML files. """
    sacred.commands.print_config(_run)

config = ex.run_commandline().config
args = nested_dict_to_namespace(config)
args.num_classes = 1+78 # backgrdound + objs

########## environmental settings ##########
utils.init_distributed_mode(args)
print("git:\n  {}\n".format(utils.get_sha()))

output_dir = Path(args.output_dir)
if args.output_dir:
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml.dump(vars(args), open(output_dir / 'config.yaml', 'w'), allow_unicode=True)
device = torch.device(args.device)

seed = args.seed + utils.get_rank()
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True

########## model ##########
model = build_frcnn(args)
model.to(device)

model_without_ddp = model
if args.distributed:
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu], find_unused_parameters=True)
    model_without_ddp = model.module

n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('NUM TOTAL MODEL PARAMS:', sum(p.numel() for p in model.parameters()))
print('NUM TRAINABLE MODEL PARAMS:', n_parameters)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [args.lr_drop])

if args.resume:
    print(f"Resume from model: {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu')
    resume_state_dict = checkpoint['model']
    model_without_ddp.load_state_dict(resume_state_dict)

########## dataset ##########
dataset_train = build_dataset(split='train', args=args)
dataset_val = build_dataset(split='val', args=args)

if args.distributed:
    sampler_train = utils.DistributedWeightedSampler(dataset_train)
    # sampler_train = DistributedSampler(dataset_train)
    sampler_val = DistributedSampler(dataset_val, shuffle=False)
else:
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

batch_sampler_train = torch.utils.data.BatchSampler(
    sampler_train, args.batch_size, drop_last=True)

data_loader_train = DataLoader(
    dataset_train,
    batch_sampler=batch_sampler_train,
    collate_fn=utils.frcnn_collate_fn,
    num_workers=args.num_workers)
data_loader_val = DataLoader(
    dataset_val, args.batch_size,
    sampler=sampler_val,
    drop_last=False,
    collate_fn=utils.frcnn_collate_fn,
    num_workers=args.num_workers)

########## train & eval ##########
if args.eval_only:
    evaluate(model, data_loader_val, device=device)
    # evaludate_mot(model, dataset_val, device=device, chosen_tracker=args.sgg_postprocessing_tracker)
else:
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=500)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pth'))

        # evaluate after every epoch
        evaluate(model, data_loader_val, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
