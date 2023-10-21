# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import copy
import random
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

from . import transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):

    fields = ["labels", "area", "iscrowd", "boxes", "track_ids", "masks"]

    def __init__(self,  img_folder, ann_file, transforms, return_masks,
                 remove_no_obj_imgs=True, norm_transforms=None,
                 prev_frame=False, prev_frame_rnd_augs=0.05, clip_length=None, dataset_name=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self._norm_transforms = norm_transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, dataset_name=dataset_name)
        self.dataset_name = dataset_name

        if remove_no_obj_imgs:
            self.ids = sorted(list(set(
                [ann['image_id'] for ann in self.coco.loadAnns(self.coco.getAnnIds())])))

        self._prev_frame = prev_frame
        self._prev_frame_rnd_augs = prev_frame_rnd_augs

        self.clip_mode = isinstance(clip_length, int)
        self.clip_length = clip_length
        # self.ids = self.ids[:10] ## for debug

    def _getitem_from_id(self, image_id):
        img, target = super(CocoDetection, self).__getitem__(image_id)
        image_id = self.ids[image_id]
        target = {'image_id': image_id, 'annotations': target}
        if self.dataset_name == 'vidhoi':
            org_image_info = self.coco.imgs[image_id]
            if org_image_info['frame_key'] not in self.coco.dataset['relation_annotations'][org_image_info['video_key']]:
                relation_annotations = []
            else:
                relation_annotations = self.coco.dataset['relation_annotations'][org_image_info['video_key']][org_image_info['frame_key']]
            target.update({
                'image_key': f"{org_image_info['video_key']}/{org_image_info['frame_key']}",
                'relation_annotations': relation_annotations
            })

        img, target = self.prepare(img, target)
        # target['track_ids'] = torch.arange(len(target['labels'])) ## !!! serious bug

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # ignore
        ignore = target.pop("ignore").bool()
        for field in self.fields:
            if field in target:
                target[f"{field}_ignore"] = target[field][ignore]
                target[field] = target[field][~ignore]

        return img, target

    def __getitem__(self, idx):
        img, target = self._getitem_from_id(idx)

        target['track_ids'] = torch.arange(len(target['labels']))

        if self._prev_frame:
            prev_img = img.copy()
            prev_target = copy.deepcopy(target)

            orig_w, orig_h = img.size

            # prev img
            w, h = prev_img.size
            size = random.randint(
                int((1.0 - self._prev_frame_rnd_augs) * min(w, h)),
                int((1.0 + self._prev_frame_rnd_augs) * min(w, h)))
            prev_img, prev_target = T.RandomResize([size])(prev_img, prev_target)

            w, h = prev_img.size
            min_size = (
                int((1.0 - self._prev_frame_rnd_augs) * w),
                int((1.0 - self._prev_frame_rnd_augs) * h))
            transform = T.RandomSizeCrop(min_size=min_size)
            prev_img, prev_target = transform(prev_img, prev_target)

            w, h = prev_img.size
            if orig_w < w:
                prev_img, prev_target = T.RandomCrop((h, orig_w))(prev_img, prev_target)
            else:
                prev_img, prev_target = T.RandomPad(max_size=(orig_w, h))(prev_img, prev_target)

            w, h = prev_img.size
            if orig_h < h:
                prev_img, prev_target = T.RandomCrop((orig_h, w))(prev_img, prev_target)
            else:
                prev_img, prev_target = T.RandomPad(max_size=(w, orig_h))(prev_img, prev_target)

        img, target = self._norm_transforms(img, target)

        if self._prev_frame:
            prev_img, prev_target = self._norm_transforms(prev_img, prev_target)

            if self.clip_mode:
                img = [prev_img, img]
                target = [prev_target, target]
            else:
                target['prev_image'] = prev_img
                for k, v in prev_target.items():
                    target[f'prev_{k}'] = v

        return img, target

    def write_result_files(self, *args):
        pass


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        if isinstance(polygons, dict):
            rles = {'size': polygons['size'],
                    'counts': polygons['counts'].encode(encoding='UTF-8')}
        else:
            rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, dataset_name=None):
        self.return_masks = return_masks
        self.dataset_name = dataset_name

    def __call__(self, image, org_target):
        w, h = image.size

        image_id = org_target["image_id"]
        image_id = torch.tensor([image_id])

        anno = org_target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # x,y,w,h --> x,y,x,y
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes - 1 ## !!! 注意对标注的类别进行了移位，暂时丢掉了 __background__

        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        if anno and "track_id" in anno[0]:
            track_ids = torch.tensor([obj["track_id"] for obj in anno])
            target["track_ids"] = track_ids[keep]
        elif not len(boxes):
            target["track_ids"] = torch.empty(0)

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        ignore = torch.tensor([obj["ignore"] if "ignore" in obj else 0 for obj in anno])

        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["ignore"] = ignore[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        if self.dataset_name == 'actiongenome':
            # !!! hardcode for actiongenome relations (single human instance)
            target['relation_map'] = torch.zeros((len(boxes), len(boxes), 26))
            human_instance_id = target["labels"].tolist().index(0) # class0: person
            valid_insts = [a for a, flag in zip(org_target['annotations'], keep) if flag]
            for inst_id, info in enumerate(valid_insts):
                if 'relationships' in info:
                    predicates = info['relationships']
                    target['relation_map'][human_instance_id, inst_id, predicates] = 1
        elif self.dataset_name == 'vidhoi':
            assert len(org_target['annotations']) == len(boxes)
            tid2idx = {x['track_id']: idx for idx, x in enumerate(org_target['annotations'])}
            target['relation_map'] = torch.zeros((len(boxes), len(boxes), 50))
            for rel_inst in org_target['relation_annotations']:
                target['relation_map'][tid2idx[rel_inst['subject_tid']], tid2idx[rel_inst['object_tid']], rel_inst['predicate']] = 1

        return image, target


def make_coco_transforms(image_set, img_transform=None, no_crop=False):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # default
    max_size = 1333
    val_width = 800
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    random_resizes = [400, 500, 600]
    random_size_crop = (384, 600)

    if img_transform is not None:
        scale = img_transform.max_size / max_size
        max_size = img_transform.max_size
        val_width = img_transform.val_width

        # scale all with respect to custom max_size
        scales = [int(scale * s) for s in scales]
        random_resizes = [int(scale * s) for s in random_resizes]
        random_size_crop = [int(scale * s) for s in random_size_crop]

    if image_set == 'train':
        option2 = [
            T.RandomResize(random_resizes),
            T.RandomSizeCrop(*random_size_crop),
            T.RandomResize(scales, max_size=max_size),
        ]
        if no_crop: option2.pop(1) # avoid removing boxes by cropping

        transforms = [
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose(option2)
            ),
        ]
    elif image_set == 'val':
        transforms = [
            T.RandomResize([val_width], max_size=max_size),
        ]
    else:
        ValueError(f'unknown {image_set}')

    # transforms.append(normalize)
    return T.Compose(transforms), normalize


def build(image_set, args, mode='instances'):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'

    # image_set is 'train' or 'val'
    split = getattr(args, f"{image_set}_split")

    splits = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    transforms, norm_transforms = make_coco_transforms(image_set, args.img_transform)
    img_folder, ann_file = splits[split]
    dataset = CocoDetection(
        img_folder,
        ann_file,
        transforms=transforms,
        norm_transforms=norm_transforms,
        return_masks=args.masks,
        prev_frame=args.tracking,
        prev_frame_rnd_augs=args.coco_and_crowdhuman_prev_frame_rnd_augs)

    return dataset
