import json
import os
import pickle
import torch
import numpy as np

DATA_ROOT = 'data/ActionGenome'

def save_annotations(split_name, filter_nonperson_box_frame=True, video_limit=None):
    assert split_name in ['train', 'test']

    with open(DATA_ROOT + '/annotations/person_bbox.pkl', 'rb') as f:
        person_bbox = pickle.load(f)
    with open(DATA_ROOT + '/annotations/object_bbox_and_relationship_filtersmall.pkl', 'rb') as f: # follow STTran
    # with open(DATA_ROOT + '/annotations/object_bbox_and_relationship.pkl', 'rb') as f:
        object_bbox = pickle.load(f)

    # # check image exist
    # if split_name == 'train':
    #     non_exist_frames = []
    #     for frame_key in person_bbox.keys():
    #         if not os.path.isfile(f"{DATA_ROOT}/frames/{frame_key}"):
    #             print(f'{frame_key}')
    #             non_exist_frames.append(frame_key)
    #     print(non_exist_frames)
    #     assert len(non_exist_frames) == 0

    # collect valid frames
    video_dict = {}
    for frame_key in person_bbox.keys():
        if object_bbox[frame_key][0]['metadata']['set'] == split_name:
            frame_valid = False
            for j in object_bbox[frame_key]: # the frame is valid if there is visible bbox
                if j['visible']: frame_valid = True

            if frame_valid:
                video_name, frame_num = frame_key.split('/')
                if video_name in video_dict.keys():
                    video_dict[video_name].append(frame_key)
                else:
                    video_dict[video_name] = [frame_key]

    # get annotations
    video_level_annotations, video_list, video_size = {}, [], []
    non_gt_human_nums, valid_nums = 0, 0
    one_frame_video, one_frame_video, non_person_video = 0, 0, 0
    annotation_id, image_id = 0, 0
    for i in video_dict.keys():
        video = []
        gt_annotation_video = {}
        for j in sorted(video_dict[i]):
            if filter_nonperson_box_frame:
                if person_bbox[j]['bbox'].shape[0] == 0:
                    non_gt_human_nums += 1
                    continue
                else:
                    video.append(j)
                    valid_nums += 1

            # person box # 查看数据集，test split一帧最多1个human
            frame_person_box = [float(x) for x in person_bbox[j]['bbox'][0]]
            gt_annotation_frame = [
                {
                    "id": annotation_id,
                    "bbox": [frame_person_box[0], frame_person_box[1], max(frame_person_box[2]-frame_person_box[0], 0), max(frame_person_box[3]-frame_person_box[1], 0)],
                    "image_id": image_id,
                    "segmentation": [],
                    "ignore": False,
                    "visibility": True,
                    "area": (frame_person_box[2]-frame_person_box[0]) * (frame_person_box[3]-frame_person_box[1]),
                    "iscrowd": 0,
                    "seq": i,
                    "category_id": object_classes.index('person'),
                    "track_id": 0,
                }
            ]
            annotation_id += 1

            # non-human objects
            for k in object_bbox[j]:
                if k['visible']:
                    gt_annotation_frame.append({
                        "id": annotation_id,
                        "bbox": k['bbox'],
                        "image_id": image_id,
                        "segmentation": [],
                        "ignore": False,
                        "visibility": True,
                        "area": k['bbox'][2] * k['bbox'][3],
                        "iscrowd": 0,
                        "seq": i,
                        "category_id": object_classes.index(k['class']),
                        "track_id": object_classes.index(k['class']), # label as track_id, since only one intance of the same class exists in ActionGenome dataset
                        # "attention_relationship": [attention_relationships.index(r) for r in k['attention_relationship']],
                        # "spatial_relationship": [spatial_relationships.index(r) for r in k['spatial_relationship']],
                        # "contacting_relationship": [contacting_relationships.index(r) for r in k['contacting_relationship']],
                        "relationships": [relationship_classes.index(r) for r in k['attention_relationship']+k['spatial_relationship']+k['contacting_relationship']]
                    })
                    annotation_id += 1

            image_id += 1
            gt_annotation_video[j] = gt_annotation_frame

        if len(video) > 2:
            video_list.append(video)
            video_size.append(person_bbox[j]['bbox_size'])
            video_level_annotations[i]= gt_annotation_video
        elif len(video) == 1:
            one_frame_video += 1
        else:
            non_person_video += 1

    print('x'*60)
    print('There are {} videos and {} valid frames'.format(len(video_list), valid_nums))
    print('\t{} videos are invalid (no person), remove them'.format(non_person_video))
    print('\t{} videos are invalid (only one frame), remove them'.format(one_frame_video))
    print('\t{} frames have no human bbox in GT, remove them!'.format(non_gt_human_nums))
    print('x' * 60)

    # to COCO format
    seqs = sorted(list(video_level_annotations.keys()))
    if video_limit is not None: seqs=seqs[:video_limit]
    annotations_coco_format = {
        'type': 'instances',
        'categories': [{'id': id, 'name': c, 'supercategory': c} for id, c in enumerate(object_classes)],
        'images': [],
        'annotations': [],
        'sequences': seqs,
        'sequence_startend_image_ids': []
    }
    for vk in seqs:
        video_info = video_level_annotations[vk]

        vframe_image_ids = [video_info[fkey][0]['image_id'] for fkey in sorted(video_info.keys())]
        annotations_coco_format['sequence_startend_image_ids'].append((min(vframe_image_ids), max(vframe_image_ids)))

        # https://zhuanlan.zhihu.com/p/29393415
        for fid, fkey in enumerate(sorted(video_info.keys())):
            annotations_coco_format['images'].append({
                'id': video_info[fkey][0]['image_id'],
                'file_name': fkey,
                'frame_id': fid,
                'first_frame_image_id': min(vframe_image_ids)
            })
            annotations_coco_format['annotations'].extend(video_info[fkey])

    annotation_file = f'{DATA_ROOT}/{split_name}_cocofmt.json'
    if video_limit is not None: annotation_file = f'{DATA_ROOT}/{split_name}_v{video_limit}_cocofmt.json'
    with open(annotation_file, 'w') as anno_file:
        json.dump(annotations_coco_format, anno_file, indent=4)
    print(f'Saved {split_name} annotaions to {annotation_file}')

if __name__ == '__main__':
    ## save meta infos
    save_path=f'{DATA_ROOT}/meta_infos.json'

    # collect the object classes
    object_classes = []
    object_classes.append('__background__')
    with open(os.path.join(DATA_ROOT, 'annotations/object_classes.txt'), 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            object_classes.append(line)
    object_classes[9] = 'closet/cabinet'
    object_classes[11] = 'cup/glass/bottle'
    object_classes[23] = 'paper/notebook'
    object_classes[24] = 'phone/camera'
    object_classes[31] = 'sofa/couch'

    # collect relationship classes
    relationship_classes = []
    with open(os.path.join(DATA_ROOT, 'annotations/relationship_classes.txt'), 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            relationship_classes.append(line)
    relationship_classes[0] = 'looking_at'
    relationship_classes[1] = 'not_looking_at'
    relationship_classes[5] = 'in_front_of'
    relationship_classes[7] = 'on_the_side_of'
    relationship_classes[10] = 'covered_by'
    relationship_classes[11] = 'drinking_from'
    relationship_classes[13] = 'have_it_on_the_back'
    relationship_classes[15] = 'leaning_on'
    relationship_classes[16] = 'lying_on'
    relationship_classes[17] = 'not_contacting'
    relationship_classes[18] = 'other_relationship'
    relationship_classes[19] = 'sitting_on'
    relationship_classes[20] = 'standing_on'
    relationship_classes[25] = 'writing_on'

    attention_relationships = relationship_classes[0:3]
    spatial_relationships = relationship_classes[3:9]
    contacting_relationships = relationship_classes[9:]

    # save to json
    meta_infos = {
        'object_classes': object_classes,
        'relationship_classes': relationship_classes,
        'attention_relationships': attention_relationships,
        'spatial_relationships': spatial_relationships,
        'contacting_relationships': contacting_relationships
    }

    with open(save_path, 'w') as f:
        json.dump(meta_infos, f, indent=4)
    print(f'Saved meta infos to {save_path}')

    # save annotation in COCO format
    save_annotations('train')
    save_annotations('train', video_limit=1000)

    save_annotations('test')
    save_annotations('test', video_limit=500)
    save_annotations('test', video_limit=200)
