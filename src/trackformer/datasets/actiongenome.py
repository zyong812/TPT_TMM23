import numpy as np
from .coco import CocoDetection, make_coco_transforms
from pathlib import Path
import random
import copy

class ActionGenome(CocoDetection):

    def __init__(self, img_folder, ann_file, transforms, return_masks,
                 prev_frame=False, prev_frame_rnd_augs=0.0, norm_transform=None, clip_length=None):
        super(ActionGenome, self).__init__(
            img_folder, ann_file, transforms, return_masks, False,
            norm_transform, prev_frame, prev_frame_rnd_augs, clip_length=clip_length, dataset_name='actiongenome')

    def _add_frame_to_target(self, image_id, random_state):
        random.setstate(random_state)
        frame_img, frame_target = self._getitem_from_id(image_id)
        frame_img, frame_target = self._norm_transforms(frame_img, frame_target)

        return frame_img, frame_target

    def sequence_infos(self):
        seqs = self.coco.dataset['sequences']
        startend_image_ids = self.coco.dataset['sequence_startend_image_ids']
        startend_idx = [(self.ids.index(se[0]), self.ids.index(se[1])) for se in startend_image_ids]
        return seqs, startend_idx

    def real_interval_to_prev_frame(self, org_img_id):
        img_info = self.coco.imgs[org_img_id]
        if img_info['id'] == img_info['first_frame_image_id']:
            return 0
        else:
            real_img_frame_idx = img_info['file_name'][-10:-4]
            prev_img_id = self.ids[self.ids.index(org_img_id)-1]
            prev_img_frame_idx = self.coco.imgs[prev_img_id]['file_name'][-10:-4]
            return int(real_img_frame_idx) - int(prev_img_frame_idx)

    def __getitem__(self, idx):
        random_state = random.getstate()

        img, target = self._getitem_from_id(idx)
        img, target = self._norm_transforms(img, target)

        if self.clip_mode:
            org_img_id = self.ids[idx]
            org_img_info = self.coco.imgs[org_img_id]
            frame_id = org_img_info['frame_id']
            start_id = self.ids.index(org_img_info['first_frame_image_id'])
            prev_image_ids = np.sort((frame_id - np.arange(0, frame_id+1))[1:self.clip_length]) + start_id

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


def build_actiongenome(image_set, args):
    root = Path(args.actiongenome_path)
    assert root.exists(), f'provided ActionGenome path {root} does not exist'

    split = getattr(args, f"{image_set}_split")

    img_folder = f"{root}/frames"
    ann_file = root / f"{split}_cocofmt.json"

    transforms, norm_transforms = make_coco_transforms(
        image_set, args.img_transform, no_crop=True)

    dataset = ActionGenome(
        img_folder, ann_file,
        transforms=transforms,
        norm_transform=norm_transforms,
        return_masks=args.masks,
        prev_frame=args.tracking,
        prev_frame_rnd_augs=args.track_prev_frame_rnd_augs,
        clip_length=args.clip_length
    )

    return dataset
