cd /218019030/projects/VideoSG-on-trackformer/

# # eval
# python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=11112 --use_env \
#     src/train.py with deformable vidhoi vsgg \
#     resume='models/vidhoi/ais_ORACLE_MODE_detr+tracking+hoi_59c6a9c+union_feat+hoi_use_temporal_dynamics/checkpoint_epoch5.pth' \
#     hoi_detection=True freeze_detr=True clip_length=3 \
#     hoi_oracle_mode=True hoi_use_temporal_dynamics=True hoi_oracle_mode_use_roialign_union_feat=True \
#     eval_only=True hoi_relation_propagation_on_inference=True > 'models/vidhoi/ais_ORACLE_MODE_detr+tracking+hoi_59c6a9c+union_feat+hoi_use_temporal_dynamics/relation_prop_eval_epoch5.log.txt'


# coco detr_resnet101 train
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=11112 --use_env src/train.py with deformable \
    backbone=resnet101 batch_size=1 resume='models/coco/ais_detr-resnet101_59c6a9d/checkpoint_epoch10.pth' \
    output_dir='models/coco/ais_detr-resnet101_59c6a9d' lr_drop=20 >> 'models/coco/ais_detr-resnet101_59c6a9d/log.txt'


# # python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=11112 --use_env src/train.py with deformable actiongenome backbone=resnet101 \
# #     resume=models/actiongenome/ais_detr-resnet101_59c6a9d/checkpoint.pth eval_only=True > models/actiongenome/ais_detr-resnet101_59c6a9d/eval_all_log.txt

# python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=11112 --use_env src/train.py with deformable actiongenome \
#     resume=models/actiongenome/ais_detr_59c6a9c/checkpoint.pth eval_only=True > models/actiongenome/ais_detr_59c6a9c/eval_all_log.txt
