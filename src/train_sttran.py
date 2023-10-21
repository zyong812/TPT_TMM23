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
import copy

import trackformer.util.misc as utils
from trackformer.util.misc import nested_dict_to_namespace
from trackformer.datasets import build_dataset
from trackformer.datasets import get_coco_api_from_dataset
from trackformer.datasets.coco_eval import CocoEvaluator
from trackformer.datasets.vidhoi_eval import VidHOIEvaluator
from STTran.sttran import build_sttran
from trackformer.util.plot_utils import check_sttran_prediction

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(print_freq, delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # todo: same data format
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        for t in targets: t['labels'] += 1 # compatibility with torchvison frcnn (set __background__=0)

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
        losses.backward()
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
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
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(print_freq, delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    vidhoi_evaluator = VidHOIEvaluator(args)

    model_inference_time = 0
    for images, targets in metric_logger.log_every(data_loader, 10, header):
        # if targets[0]['image_id'].item() < 280: continue
        frame_tic = time.time()
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        for t in targets: t['labels'] += 1 # compatibility with torchvison frcnn (set __background__=0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        try:
            outputs = model(images, targets) # requires pytorch<=1.5
        except:
            # outputs = model(images, targets)
            continue
        model_time = time.time() - model_time

        ## evaluation
        evaluator_time = time.time()
        det_nums, det_res = outputs['box_nums'], {}
        for t, b, l, s in zip(targets, outputs['boxes'][:, 1:].to(cpu_device).split(det_nums, dim=0),
                              (outputs['labels']-1).to(cpu_device).split(det_nums, dim=0), # -1 for compatibility with evaluation code
                              outputs['scores'].to(cpu_device).split(det_nums, dim=0)):
            det_res[t['image_id'].item()] = {'boxes': b, 'labels': l, 'scores': s}
        coco_evaluator.update(copy.deepcopy(det_res)) # detection evaluation

        rel_nums, rel_outs = outputs['rel_pair_nums'], {'pred_rel_pairs': [], 'pred_relations': []}
        for idx, (rp, rl) in enumerate(zip(outputs['rel_pair_idxs'].split(rel_nums, dim=0), outputs['rel_logits'].split(rel_nums, dim=0))):
            rel_outs['pred_rel_pairs'].append(rp - sum(det_nums[:idx]))
            rel_outs['pred_relations'].append(rl)
        top_pred_rel_pairs = vidhoi_evaluator.sttran_update(targets, rel_outs, box_preds=det_res) # relation evaluation

        # for idx in range(len(targets)):
        #     check_sttran_prediction(images, rel_outs, det_res, targets, top_pred_rel_pairs, idx=idx, save_fig_dir=f"{args.output_dir}/demo")

        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        model_inference_time += time.time() - frame_tic

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # accumulate predictions from all images
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    vidhoi_evaluator.synchronize_between_processes()
    vidhoi_evaluator.evaluate()
    print(f"model_inference_time={model_inference_time}s")
    print(f"Model: time_per_frame={model_inference_time/len(vidhoi_evaluator.gts)*1000 :.2f}ms, frame_per_second={len(vidhoi_evaluator.gts)/model_inference_time :.2f}")
    return coco_evaluator

########## configs ##########
ex = sacred.Experiment('train')
ex.add_config('cfgs/train.yaml')
ex.add_named_config('vidhoi', 'cfgs/train_vidhoi.yaml')
ex.add_named_config('sttran', 'cfgs/train_sttran.yaml')

@ex.main
def load_config(_config, _run):
    """ We use sacred only for config loading from YAML files. """
    sacred.commands.print_config(_run)

config = ex.run_commandline().config
args = nested_dict_to_namespace(config)
args.num_classes = 1+78
args.object_detector = 'frcnn'

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
    collate_fn=utils.sttran_collate_fn,
    num_workers=args.num_workers)
data_loader_val = DataLoader(
    dataset_val, args.batch_size,
    sampler=sampler_val,
    drop_last=False,
    collate_fn=utils.sttran_collate_fn,
    num_workers=args.num_workers)

########## model ##########
model = build_sttran(args, obj_classes=[x['name'] for x in dataset_train.coco.dataset['categories']])
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
optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [args.lr_drop])

if args.resume:
    print(f"Resume from model: {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu')
    resume_state_dict = checkpoint['model']
    if 'frcnn' in args.resume:
        model_without_ddp.frcnn.load_state_dict(resume_state_dict, strict=False) # only load detector part
    else:
        model_without_ddp.load_state_dict(resume_state_dict, strict=False)

########## train & eval ##########
if args.eval_only:
    evaluate(model, data_loader_val, device=device)
else:
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
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
