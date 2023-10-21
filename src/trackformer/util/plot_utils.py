"""
Plotting utilities to visualize training logs.
"""
import os
from pathlib import Path, PurePath
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from . import box_ops
from .misc import NestedTensor, nested_tensor_from_tensor_list

MOT_obj_label_names = ['person', '_BG_']
# # Action Genome classes
# PredicateClasses = ['looking_at', 'not_looking_at', 'unsure', 'above', 'beneath', 'in_front_of', 'behind', 'on_the_side_of', 'in', 'carrying', 'covered_by', 'drinking_from', 'eating', 'have_it_on_the_back', 'holding', 'leaning_on', 'lying_on', 'not_contacting', 'other_relationship', 'sitting_on', 'standing_on', 'touching', 'twisting', 'wearing', 'wiping', 'writing_on']
# ObjClasses = ['person', 'bag', 'bed', 'blanket', 'book', 'box', 'broom', 'chair', 'closet/cabinet', 'clothes', 'cup/glass/bottle', 'dish', 'door', 'doorknob', 'doorway', 'floor', 'food', 'groceries', 'laptop', 'light', 'medicine', 'mirror', 'paper/notebook', 'phone/camera', 'picture', 'pillow', 'refrigerator', 'sandwich', 'shelf', 'shoe', 'sofa/couch', 'table', 'television', 'towel', 'vacuum', 'window']
# VidHOI classes
PredicateClasses = ['lean_on', 'watch', 'above', 'next_to', 'behind', 'away', 'towards', 'in_front_of', 'hit', 'hold', 'wave', 'pat', 'carry', 'point_to', 'touch', 'play(instrument)', 'release', 'ride', 'grab', 'lift', 'use', 'press', 'inside', 'caress', 'pull', 'get_on', 'cut', 'hug', 'bite', 'open', 'close', 'throw', 'kick', 'drive', 'get_off', 'push', 'wave_hand_to', 'feed', 'chase', 'kiss', 'speak_to', 'beneath', 'smell', 'clean', 'lick', 'squeeze', 'shake_hand_with', 'knock', 'hold_hand_of', 'shout_at']
ObjClasses = ['person', 'car', 'guitar', 'chair', 'handbag', 'toy', 'baby_seat', 'cat', 'bottle', 'backpack', 'motorcycle', 'ball/sports_ball', 'laptop', 'table', 'surfboard', 'camera', 'sofa', 'screen/monitor', 'bicycle', 'vegetables', 'dog', 'fruits', 'cake', 'cellphone', 'cup', 'bench', 'snowboard', 'skateboard', 'bread', 'bus/truck', 'ski', 'suitcase', 'stool', 'bat', 'elephant', 'fish', 'baby_walker', 'dish', 'watercraft', 'scooter', 'pig', 'refrigerator', 'horse', 'crab', 'bird', 'piano', 'cattle/cow', 'lion', 'chicken', 'camel', 'electric_fan', 'toilet', 'sheep/goat', 'rabbit', 'train', 'penguin', 'hamster/rat', 'snake', 'frisbee', 'aircraft', 'oven', 'racket', 'faucet', 'antelope', 'duck', 'stop_sign', 'sink', 'kangaroo', 'stingray', 'turtle', 'tiger', 'crocodile', 'bear', 'microwave', 'traffic_light', 'panda', 'leopard', 'squirrel']

# COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
#           [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933],]
cmap = matplotlib.cm.get_cmap('hsv')
COLORS = [cmap(idx/300) for idx in random.sample(range(0, 300), 300)]

def check_annotation(samples, annotations, mode='train', idx=0):
    img_tensors, img_masks = samples.decompose()
    h, w = (img_masks[idx].float() < 1).nonzero(as_tuple=False).max(0)[0].cpu() + 1

    img_tensor = img_tensors[idx,:,:h,:w].cpu().permute(1,2,0)
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())

    res = annotations[idx]
    org_h, org_w = res['orig_size'].cpu().float()
    boxes = res['boxes'].cpu()
    if mode == 'train':
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes = boxes * torch.tensor([w, h, w, h]).unsqueeze(0)
    else:
        boxes = boxes * torch.tensor([w/org_w, h/org_h, w/org_w, h/org_h]).unsqueeze(0)
    obj_show_names = []
    for ind, (x, tid) in enumerate(zip(res['labels'], res['track_ids'])):
        obj_show_names.append(f"{ObjClasses[x]}_{tid}")

    # draw images
    plt.imshow(img_tensor)
    for ind, bbox in enumerate(boxes):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1,y1), x2-x1+1, y2-y1+1, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        txt = plt.text(x1, y1, obj_show_names[ind], color='black')
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
    plt.gca().yaxis.set_label_position("right")
    plt.title(f"image_id={annotations[idx]['image_id'].item()}")

    if 'relation_map' in res:
        rels = res['relation_map'].nonzero(as_tuple=False).cpu().numpy()
        rel_strs = ''
        for i, rel in enumerate(rels):
            rel_strs += f"{obj_show_names[rel[0]]} --{PredicateClasses[rel[2]]}--> {obj_show_names[rel[1]]}\n"

        print(f"image_id={annotations[idx]['image_id'].item()}:\n", rel_strs, '\n')
        plt.xlabel(rel_strs, rotation=0, fontsize=6)
        plt.text(5, 250, rel_strs, fontsize=6, color='red')

    plt.axis('off')
    plt.tight_layout()
    plt.show()

def check_prediction(samples, results, threshold=0.9, targets=None, idx=0, frame_id=None, top_pred_rel_pairs=None, save_fig_dir=None):
    if not isinstance(samples, NestedTensor):
        samples = nested_tensor_from_tensor_list(samples)
    pil_imgs, masks = samples.decompose()
    pil_img, mask= pil_imgs[idx], masks[idx]

    pil_img = ((pil_img - pil_img.min()) / (pil_img.max() - pil_img.min())).permute(1,2,0).cpu().numpy()
    h, w = (~mask).float().nonzero(as_tuple=False).max(0)[0] + 1
    pil_img = pil_img[:h, :w]

    if isinstance(results, list):
        boxes = box_ops.box_cxcywh_to_xyxy(results[idx]['pred_boxes'].detach().cpu()[0]) * torch.tensor([w,h,w,h])
        box_scores, box_labels = results[idx]['pred_logits'].softmax(-1)[..., :-1].detach().cpu()[0].max(-1)
    else:
        boxes = box_ops.box_cxcywh_to_xyxy(results['pred_boxes'][idx].detach().cpu()) * torch.tensor([w,h,w,h])
        box_scores, box_labels = results['pred_logits'][idx].softmax(-1)[..., :-1].detach().cpu().max(-1)

    # print relation predictions
    prop_obj_ids, top_rel_str = [], ''
    if 'pred_rel_pairs' in results:
        img_rel_proposal_pairs = results['pred_rel_pairs'][idx]
        img_rel_prop_scores = results['pred_relation_exists'][idx].sigmoid()
        img_rel_predicate_scores = results['pred_relations'][idx].sigmoid()
        if top_pred_rel_pairs is not None:
            print(f'Top predicted relations for image_id={targets[idx]["image_id"].item()}:')
            show_rel_triplets = top_pred_rel_pairs[idx][:10] # top 10
            prop_obj_ids = show_rel_triplets[:, :2].flatten()
            for sub, obj, predicate in show_rel_triplets:
                prop_idx = img_rel_proposal_pairs.tolist().index([sub, obj])
                predicate_score = img_rel_predicate_scores[prop_idx, predicate]
                top_rel_str += f"{ObjClasses[box_labels[sub]]}_{sub}({box_scores[sub]:.2f}) -- {PredicateClasses[predicate]}({predicate_score:.2f}) --> {ObjClasses[box_labels[obj]]}_{obj}({box_scores[obj]:.2f})\n"
            print(top_rel_str)
        else:
            print(f'Top predicted pairs for image_id={targets[idx]["image_id"].item()}:')
            prop_obj_ids = img_rel_proposal_pairs.view(-1).tolist()
            for p, ps, pred_scores in zip(img_rel_proposal_pairs, img_rel_prop_scores, img_rel_predicate_scores):
                top_predicates_scores, top_predicates = pred_scores.sort(descending=True)
                top_predicates_str = ', '.join([f"{PredicateClasses[top_predicates[k]]} ({top_predicates_scores[k]:.2f})" for k in range(3)])
                print(f'\033[94m{ObjClasses[box_labels[p[0]]]}_{p[0]}-{ObjClasses[box_labels[p[1]]]}_{p[1]} ({ps:.2f}):\t \033[92m{top_predicates_str}\033[0m')

    ######## plt detected boxes ##########
    plt.imshow(pil_img, alpha=0.5)
    for id, (sc, l, (xmin, ymin, xmax, ymax), c) in enumerate(zip(box_scores, box_labels, boxes, COLORS)):
        # if id in prop_obj_ids or sc > threshold:
        # if sc > threshold:
        if id in prop_obj_ids:
            plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c))
            # text = f'{str(id)}_l={l}({sc:0.2f})'
            text = f'{ObjClasses[l]}_{str(id)}'
            plt.text(xmin, ymin, text, fontsize=18, bbox=dict(facecolor=c, alpha=0.6))

    if len(top_rel_str) > 0: plt.text(5, 250, top_rel_str, fontsize=6, color='red')
    if targets is not None: plt.title(f"image_id={targets[idx]['image_id'].item()}")
    if frame_id is not None: plt.title(f"frame_id={frame_id}")
    plt.axis('off')
    plt.tight_layout()

    if save_fig_dir is not None:
        if not os.path.isdir(save_fig_dir):
            os.system(f'mkdir -p {save_fig_dir}')
        plt.savefig(f"{save_fig_dir}/image_id={targets[idx]['image_id'].item()}.png")
    plt.show()

def check_sttran_prediction(images, rel_outs, box_preds, targets, top_pred_rel_pairs, idx=0, save_fig_dir=None):
    pil_img = images[idx].permute(1,2,0).cpu().numpy()

    # box predictions
    real_image_id = targets[idx]['image_id'].item()
    boxes, box_scores, box_labels = box_preds[real_image_id]['boxes'], box_preds[real_image_id]['scores'], box_preds[real_image_id]['labels']

    # relation predictions
    prop_obj_ids, top_rel_str = [], ''
    if top_pred_rel_pairs is not None:
        img_rel_proposal_pairs = rel_outs['pred_rel_pairs'][idx]
        img_rel_predicate_scores = rel_outs['pred_relations'][idx].sigmoid()

        print(f'Top predicted relations for image_id={targets[idx]["image_id"].item()}:')
        show_rel_triplets = top_pred_rel_pairs[idx][:10]
        prop_obj_ids = show_rel_triplets[:, :2].flatten()
        for sub, obj, predicate in show_rel_triplets:
            prop_idx = img_rel_proposal_pairs.tolist().index([sub, obj])
            predicate_score = img_rel_predicate_scores[prop_idx, predicate]
            top_rel_str += f"{ObjClasses[box_labels[sub]]}_{sub}({box_scores[sub]:.2f}) -- {PredicateClasses[predicate]}({predicate_score:.2f}) --> {ObjClasses[box_labels[obj]]}_{obj}({box_scores[obj]:.2f})\n"

        print(top_rel_str)

    ######## plt detected boxes ##########
    plt.imshow(pil_img, alpha=0.5)
    for id, (sc, l, (xmin, ymin, xmax, ymax)) in enumerate(zip(box_scores, box_labels, boxes)):
        c = COLORS[id+idx*10] # randomize color
        if id in prop_obj_ids:
            plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c))
            # text = f'{ObjClasses[l]}_{str(id)}'
            text = f'{ObjClasses[l]}'
            plt.text(xmin, ymin, text, fontsize=18, bbox=dict(facecolor=c, alpha=0.6))

    # show image
    # if len(top_rel_str) > 0: plt.text(5, 70, top_rel_str, fontsize=6, color='red')
    # if targets is not None: plt.title(f"image_id={targets[idx]['image_id'].item()}")
    plt.axis('off')
    plt.tight_layout()
    if save_fig_dir is not None:
        if not os.path.isdir(save_fig_dir):
            os.system(f'mkdir -p {save_fig_dir}')
        plt.savefig(f"{save_fig_dir}/notext_image_id={targets[idx]['image_id'].item()}.png")
    plt.show()

def fig_to_numpy(fig):
    w, h = fig.get_size_inches() * fig.dpi
    w = int(w.item())
    h = int(h.item())
    canvas = FigureCanvas(fig)
    canvas.draw()
    numpy_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, 3)
    return np.copy(numpy_image)


def get_vis_win_names(vis_dict):
    vis_win_names = {
        outer_k: {
            inner_k: inner_v.win
            for inner_k, inner_v in outer_v.items()
        }
        for outer_k, outer_v in vis_dict.items()
    }
    return vis_win_names


def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    '''
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # verify valid dir(s) and that every item in list is Path object
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if dir.exists():
            continue
        raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_eval = pd.DataFrame(pd.np.stack(df.test_coco_eval.dropna().values)[:, 1]).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)


def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        print(f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' +
              f'score={scores.mean():0.3f}, ' +
              f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
              )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title('Precision / Recall')
    axs[0].legend(names)
    axs[1].set_title('Scores / Recall')
    axs[1].legend(names)
    return fig, axs
