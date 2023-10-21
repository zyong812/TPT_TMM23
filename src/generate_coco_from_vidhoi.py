import json
import os
import pickle
import torch
import numpy as np
import warnings
from tqdm import tqdm
import argparse

# generate VidHOI from original VidOR annotations
def convert_vidor_to_ava_label(annot_dir):
    frame_annots = []
    used_video_dict = set()

    for folder in tqdm(os.listdir(annot_dir)):
        for video_json in os.listdir(os.path.join(annot_dir, folder)):
            with open(os.path.join(annot_dir, folder, video_json), 'r') as f:
                annot = json.load(f)

            if abs(annot['fps'] - 29.97) < 0.1:
                fps = 30
            elif annot['fps'] - 24 < 1.01: # fps 24, 25
                fps = 24
            else:
                raise(f"Invalid fps={annot['fps']}")

            for i in range(annot['frame_count']):
                if (i - (fps // 2)) % fps != 0: # 1 sample/sec
                    continue

                idx = i-1
                for rel in annot['relation_instances']:
                    if rel['begin_fid'] <= idx < rel['end_fid'] \
                        and annot['subject/objects'][rel['subject_tid']]['category'] in human_categories \
                        and rel['predicate'] in pred_categories:
                        frame_annot = annot['trajectories'][idx]

                        person_found = object_found = False
                        for ann in frame_annot:
                            if ann['tid'] == rel['subject_tid']:
                                person_annot, person_found = ann, True
                            elif ann['tid'] == rel['object_tid']:
                                object_annot, object_found = ann, True
                            if person_found and object_found:
                                break

                        frame_annots.append({
                            'video_folder': folder,
                            'video_id': annot['video_id'],
                            'frame_id': str(f'{idx+1:06d}'), # real frame index start from 1
                            'video_fps': fps, # annot['fps'],
                            'height': annot['height'],
                            'width': annot['width'],
                            # 'middle_frame_timestamp': i // fps + 1,
                            'person_box': person_annot['bbox'],
                            'object_box': object_annot['bbox'],
                            'person_id': person_annot['tid'],
                            'object_id': object_annot['tid'],
                            'object_class': obj_to_idx['person'] if annot['subject/objects'][rel['object_tid']]['category'] in human_categories else obj_to_idx[annot['subject/objects'][rel['object_tid']]['category']],
                            'action_class': pred_to_idx[rel['predicate']],
                        })

                        used_video_dict.add(folder + '/' + annot['video_id'])
    return frame_annots, used_video_dict

def dump_frames(args, relation_instance_annotations, split='train'):
    video_dir = f"{args.video_dir}/{split}"
    frame_dir = f"{args.frame_dir}/{split}"

    # Create video to frames mapping
    video2frames, video2fps = {}, {}
    for ann in relation_instance_annotations: # hoi items
        video_key = f'{ann["video_folder"]}/{ann["video_id"]}'
        frame = ann['frame_id']
        if video_key not in video2frames:
            video2frames[video_key] = set()
        video2frames[video_key].add(frame)
        video2fps[video_key] = ann['video_fps']
    print(f"Total {split} #frames (with relations): {sum([len(v) for k, v in video2frames.items()])}")

    # For each video, dump frames.
    print('Dumping video frames...')
    for v in tqdm(video2frames):
        curr_frame_dir = os.path.join(frame_dir, v)

        # keep frames with even sample rates
        keep_frames = sorted([f'{x}' for x in list(video2frames[v])])
        keep_frames = [f"{idx:06d}.png" for idx in list(range(int(keep_frames[0]), int(keep_frames[-1])+1, video2fps[v]))]

        if not os.path.exists(curr_frame_dir):
            os.makedirs(curr_frame_dir)
            # Use ffmpeg to extract frames. Different versions of ffmpeg may generate slightly different frames.
            # We used ffmpeg 2.8.15 to dump our frames.
            os.system('ffmpeg -loglevel panic -i %s/%s.mp4 %s/%%06d.png' % (video_dir, v, curr_frame_dir))

            # only keep the annotated frames included in frame_list.txt
            frames_to_delete = set(os.listdir(curr_frame_dir)) - set(keep_frames)
            for frame in frames_to_delete:
                os.remove(os.path.join(curr_frame_dir, frame))
        elif set(os.listdir(curr_frame_dir)) != set(keep_frames):
            print(f'Update dumping {curr_frame_dir}.')
            os.system('ffmpeg -loglevel panic -i %s/%s.mp4 %s/%%06d.png' % (video_dir, v, curr_frame_dir))
            frames_to_delete = set(os.listdir(curr_frame_dir)) - set(keep_frames)
            for frame in frames_to_delete:
                os.remove(os.path.join(curr_frame_dir, frame))
        else:
            print(f'Skip dumping {curr_frame_dir}.')

def convert_to_coco_annotations(args, relation_instance_annotations, split='train', video_limit=None):
    # video and key frames
    relation_annotations_dict, video2fps = {}, {}
    for rel_instance in relation_instance_annotations:
        video_key = f"{rel_instance['video_folder']}/{rel_instance['video_id']}"
        if video_key not in relation_annotations_dict:
            relation_annotations_dict[video_key] = {}
        if rel_instance['frame_id'] not in relation_annotations_dict[video_key]:
            relation_annotations_dict[video_key][rel_instance['frame_id']] = []
        relation_annotations_dict[video_key][rel_instance['frame_id']].append({
            'subject_tid': rel_instance['person_id'],
            'subject_class': 0,
            'object_tid': rel_instance['object_id'],
            'object_class': rel_instance['object_class'],
            'predicate': rel_instance['action_class'],
            'predicate_name': idx_to_pred[rel_instance['action_class']]
        })
        video2fps[video_key] = rel_instance['video_fps']

    saving_video_keys = list(relation_annotations_dict.keys())
    if video_limit is not None: saving_video_keys = sorted(saving_video_keys)[:video_limit]
    # print(saving_video_keys)

    # to coco format
    annotations_coco_format = {
        'type': 'instances',
        'images': [],
        'categories': [{'id': id, 'name': c, 'supercategory': c} for id, c in enumerate(obj_categories)],
        'annotations': [],
        'predicate_categories': [{'id': id, 'name': c, 'supercategory': c} for id, c in enumerate(pred_categories)],
        'relation_annotations': relation_annotations_dict,
        'sequences': saving_video_keys,
        'sequence_startend_image_ids': []
    }

    image_id, annotation_id = 0, 0
    for video_key in tqdm(saving_video_keys):
        frames_with_rels = sorted(list(relation_annotations_dict[video_key]))
        key_frames = [f"{idx:06d}" for idx in list(range(int(frames_with_rels[0]), int(frames_with_rels[-1])+1, video2fps[video_key]))]

        with open(f"{args.vidor_orig_annotation_dir}/{split}/{video_key}.json", 'r') as f:
            orig_vidor_anntations = json.load(f)

        track_tid2infos = {x['tid']: x for x in orig_vidor_anntations['subject/objects']}
        first_frame_image_id = image_id
        for idx, frame_key in enumerate(key_frames):
            annotations_coco_format['images'].append({
                'id': image_id,
                'file_name': f"{video_key}/{frame_key}.png",
                'frame_id': idx,
                'first_frame_image_id': first_frame_image_id,
                'video_key': video_key,
                'frame_key': frame_key
            })

            # box instances in frames
            frame_boxes = orig_vidor_anntations['trajectories'][int(frame_key)-1]
            for box in frame_boxes:
                box_cat = track_tid2infos[box['tid']]['category']
                xywh = [box['bbox']['xmin'], box['bbox']['ymin'],
                        box['bbox']['xmax']-box['bbox']['xmin']+1,
                        box['bbox']['ymax']-box['bbox']['ymin']+1]
                assert xywh[2] > 0 and xywh[3] > 0
                annotations_coco_format['annotations'].append({
                    'id': annotation_id,
                    'bbox': xywh,
                    'image_id': image_id,
                    "segmentation": [],
                    "ignore": False,
                    "visibility": True,
                    "area": xywh[2] * xywh[3],
                    "iscrowd": 0,
                    "seq": video_key,
                    "category_id": 1 if box_cat in human_categories else obj_categories.index(box_cat),
                    "track_id": box['tid'],
                })
                annotation_id += 1

            image_id += 1
        annotations_coco_format['sequence_startend_image_ids'].append((first_frame_image_id, image_id-1))

    # save annotations
    annotation_file = f'{args.annotation_dir}/{split}_cocofmt.json'
    if video_limit is not None: annotation_file = f'{args.annotation_dir}/{split}_v{video_limit}_cocofmt.json'
    with open(annotation_file, 'w') as anno_file:
        json.dump(annotations_coco_format, anno_file, indent=4)
    print(f'Saved {split} annotaions to {annotation_file}')
    print(f"{split} #keyframe (all): {len(annotations_coco_format['images'])}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dump frames")
    parser.add_argument("--task", default="convert_coco_annotations", help="dump_frames | convert_coco_annotations")
    parser.add_argument("--video_dir", default="data/VidHOI/VidOR/videos", help="Folder containing VidOR videos.")
    parser.add_argument("--frame_dir", default="data/VidHOI/frames", help="Root folder containing frames to be dumped.")
    parser.add_argument("--annotation_dir", default="data/VidHOI/VidHOI_annotations", help=("Folder containing VidHOI annotation files"))
    parser.add_argument("--vidor_orig_annotation_dir", default="data/VidHOI/VidOR/annotations", help=("Original annotations of VidOR"))
    args = parser.parse_args()

    # load meta_infos
    human_categories = ['adult', 'child', 'baby']
    with open('data/VidHOI/VidHOI_annotations/obj_categories.json', 'r') as f:
        obj_categories = json.load(f)
    with open('data/VidHOI/VidHOI_annotations/obj_to_idx.pkl', 'rb') as f:
        obj_to_idx = pickle.load(f) # exclude BG, person=0
    with open('data/VidHOI/VidHOI_annotations/idx_to_obj.pkl', 'rb') as f:
        idx_to_obj = pickle.load(f)
    print(f"#objects: {len(obj_categories)}")
    obj_categories.insert(0, '__background__')

    with open('data/VidHOI/VidHOI_annotations/pred_categories.json', 'r') as f:
        pred_categories = json.load(f)
    with open('data/VidHOI/VidHOI_annotations/pred_to_idx.pkl', 'rb') as f:
        pred_to_idx = pickle.load(f)
    with open('data/VidHOI/VidHOI_annotations/idx_to_pred.pkl', 'rb') as f:
        idx_to_pred = pickle.load(f)
    print(f"#predicates: {len(pred_categories)}")

    ###### load or generate relation instances of VidHOI from VidOR ######
    # train
    train_frame_ann_file = f'{args.annotation_dir}/train_frame_annots.json'
    if os.path.isfile(train_frame_ann_file):
        with open(train_frame_ann_file, 'r') as f:
            train_relation_annots = json.load(f)
    else:
        train_relation_annots, _ = convert_vidor_to_ava_label(f'{args.vidor_orig_annotation_dir}/train')
        with open(train_frame_ann_file, 'w') as f:
            json.dump(train_relation_annots, f)
    print(f'train hoi+hhi: {len(train_relation_annots)}')

    # val
    val_frame_ann_file = f'{args.annotation_dir}/val_frame_annots.json'
    if os.path.isfile(val_frame_ann_file):
        with open(val_frame_ann_file, 'r') as f:
            val_relation_annots = json.load(f)
    else:
        val_relation_annots, _ = convert_vidor_to_ava_label(f'{args.vidor_orig_annotation_dir}/validation')
        with open(val_frame_ann_file, 'w') as f:
            json.dump(val_relation_annots, f)
    print(f'val hoi+hhi: {len(val_relation_annots)}')

    # pre-processing tasks
    if args.task == 'dump_frames':
        dump_frames(args, train_relation_annots, split='train')
        dump_frames(args, val_relation_annots, split='validation')
    elif args.task == 'convert_coco_annotations':
        # convert_to_coco_annotations(args, train_relation_annots, split='train', video_limit=30)
        # convert_to_coco_annotations(args, val_relation_annots, split='validation', video_limit=10)
        convert_to_coco_annotations(args, train_relation_annots, split='train')
        convert_to_coco_annotations(args, val_relation_annots, split='validation')
    else:
        raise(f'Unsupported task: {args.task}')
