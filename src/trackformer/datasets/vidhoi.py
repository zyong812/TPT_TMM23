import json
import math
import numpy as np
from .coco import CocoDetection, make_coco_transforms
from pathlib import Path
import random
import copy
import torch
from . import transforms as T

class VidHOI(CocoDetection):

    def __init__(self, img_folder, ann_file, transforms, return_masks,
                 prev_frame=False, prev_frame_rnd_augs=0.0, norm_transform=None, clip_length=None, return_box_instant_trajectories=False):
        super(VidHOI, self).__init__(
            img_folder, ann_file, transforms, return_masks, False,
            norm_transform, prev_frame, prev_frame_rnd_augs, clip_length=clip_length, dataset_name='vidhoi')

        self.return_box_instant_trajectories = return_box_instant_trajectories

    def _add_frame_to_target(self, image_id, random_state):
        random.setstate(random_state)
        frame_img, frame_target = self._getitem_from_id(image_id)
        frame_img, frame_target = self._norm_transforms(frame_img, frame_target)

        if self.return_box_instant_trajectories:
            frame_target = self._get_box_instant_trajectories(image_id, frame_target)

        return frame_img, frame_target

    def sequence_infos(self):
        seqs = self.coco.dataset['sequences']
        startend_image_ids = self.coco.dataset['sequence_startend_image_ids']
        startend_idx = [(self.ids.index(se[0]), self.ids.index(se[1])) for se in startend_image_ids]
        return seqs, startend_idx

    def _get_box_instant_trajectories(self, idx, target):
        org_img_id = self.ids[idx]
        org_img_info = self.coco.imgs[org_img_id]
        vfolder, vkey = org_img_info['video_key'].split('/')
        if 'train' in self.root:
            vidor_anntation_file = f"data/VidHOI/VidOR/annotations/train/{vfolder}/{vkey}.json"
        elif 'validation' in self.root:
            vidor_anntation_file = f"data/VidHOI/VidOR/annotations/validation/{vfolder}/{vkey}.json"
        with open(vidor_anntation_file, 'r') as f:
            org_annotations = json.load(f)

        orig_h, orig_w = target['orig_size']
        window_size = 12
        trajectories = torch.zeros(len(target['track_ids']), window_size*2*4)
        for fid, fboxes in enumerate(org_annotations['trajectories'][int(org_img_info['frame_key'])-window_size:int(org_img_info['frame_key'])+window_size]):
            tid2info = {x['tid']: x for x in fboxes}
            for obj_id, tid in enumerate(target['track_ids']):
                if tid.item() in tid2info:
                    bbox = tid2info[tid.item()]['bbox']
                    trajectories[obj_id, fid*4:4*(fid+1)] = torch.tensor([bbox['xmin']/orig_w, bbox['ymin']/orig_h, bbox['xmax']/orig_w, bbox['ymax']/orig_h])

        target['box_instant_trajectories'] = trajectories
        return target

    def __getitem__(self, idx):
        random_state = random.getstate()

        if self.clip_mode:
            while(True): # get valid video clip
                org_img_id = self.ids[idx]
                org_img_info = self.coco.imgs[org_img_id]
                if org_img_info['frame_id'] < self.clip_length-1:
                    idx += self.clip_length
                else:
                    break

        img, target = self._getitem_from_id(idx)
        img, target = self._norm_transforms(img, target)

        if self.return_box_instant_trajectories:
            target = self._get_box_instant_trajectories(idx, target)

        if self.clip_mode:
            start_id = self.ids.index(org_img_info['first_frame_image_id'])
            prev_image_ids = np.sort((org_img_info['frame_id'] - np.arange(0, org_img_info['frame_id']+1))[1:self.clip_length]) + start_id

            prev_frame_imgs, prev_frame_targets = [], []
            for prev_image_id in prev_image_ids:
                frame_img, frame_target = self._add_frame_to_target(prev_image_id, random_state)
                prev_frame_imgs.append(frame_img)
                prev_frame_targets.append(frame_target)

            # compose clip
            append_num = self.clip_length - len(prev_frame_imgs)
            img = prev_frame_imgs + [img.clone() for _ in range(append_num)]
            target = prev_frame_targets + [copy.deepcopy(target) for _ in range(append_num)]

        return img, target


class VidHOIVideo(VidHOI):

    def __init__(self, img_folder, ann_file, transforms, return_masks=False, norm_transform=None, clip_length=3):
        super(VidHOIVideo, self).__init__(img_folder, ann_file, transforms, return_masks, norm_transform=norm_transform)

        self.clip_length = clip_length
        self.videos, self.startend_idx = self.sequence_infos()

    def sequence_infos(self):
        seqs = self.coco.dataset['sequences']
        startend_image_ids = self.coco.dataset['sequence_startend_image_ids']

        seq_clips, seq_startend_idx = [], []
        for vid, se in zip(seqs, startend_image_ids):
            vid_length = se[1] - se[0] + 1
            if vid_length < 2: continue
            seg_num = math.ceil(vid_length / self.clip_length)
            for seg_id in range(seg_num):
                if seg_id == seg_num-1:
                    seq_startend_idx.append((max(se[0], se[1]-self.clip_length+1), se[1]))
                else:
                    seq_startend_idx.append((se[0]+self.clip_length*seg_id, se[0]+self.clip_length*(seg_id+1)-1))
                seq_clips.append(f"{vid}_{seg_id}")

        return seq_clips, seq_startend_idx

    def __getitem__(self, idx):
        random_state = random.getstate()

        frames, targets = [], []
        start, end = self.startend_idx[idx]
        for fid in range(start, end+1):
            img, target = self._add_frame_to_target(fid, random_state)
            frames.append(img)
            targets.append(target)

        return frames, targets

    def __len__(self):
        return len(self.videos)

def build_vidhoi(image_set, args):
    root = Path(args.vidhoi_path)
    assert root.exists(), f'provided VidHOI path {root} does not exist'

    split = getattr(args, f"{image_set}_split")

    img_folder = f"{root}/frames/train" if image_set == 'train' else f"{root}/frames/validation"
    ann_file = root / f"VidHOI_annotations/{split}_cocofmt.json"

    if args.object_detector == 'frcnn':
        transforms = None
        norm_transforms = T.Compose([T.ToTensor()])
    else:
        transforms, norm_transforms = make_coco_transforms(
            image_set, args.img_transform, no_crop=True)

    if args.sgg_use_STTran:
        dataset = VidHOIVideo(img_folder, ann_file, transforms=transforms, norm_transform=norm_transforms, clip_length=args.clip_length)
    else:
        dataset = VidHOI(
            img_folder, ann_file,
            transforms=transforms,
            norm_transform=norm_transforms,
            return_masks=args.masks,
            prev_frame=args.tracking,
            prev_frame_rnd_augs=args.track_prev_frame_rnd_augs,
            clip_length=args.clip_length,
            return_box_instant_trajectories=(args.hoi_oracle_mode and args.hoi_oracle_mode_use_instant_trajectory)
        )

    return dataset
