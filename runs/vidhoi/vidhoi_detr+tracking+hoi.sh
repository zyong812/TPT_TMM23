OUTPUT_DIR=$1

echo "OUTPUT_DIR=${OUTPUT_DIR}"
cd /218019030/projects/VideoSG-on-trackformer/
mkdir -p $OUTPUT_DIR

# DETR+Tracking+HOI
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=11112 --use_env \
    src/train.py with deformable vidhoi vsgg hoi_detection=True freeze_detr=True \
    resume='models/vidhoi/pretrained/jd42_detr+tracking_2a5d6f7+clip_length=3/checkpoint_epoch3.pth' \
    clip_length=3 hoi_use_temporal_dynamics=True hoi_relation_propagation_on_inference=True \
    output_dir=$OUTPUT_DIR >> $OUTPUT_DIR/log.txt

# # DETR+Tracking+HOI^{TDE}
# python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=11112 --use_env \
#     src/train.py with deformable vidhoi vsgg \
#     resume='models/vidhoi/pretrained/jd42_detr+tracking_2a5d6f7+clip_length=3/checkpoint_epoch3.pth' \
#     hoi_detection=True \
#     freeze_detr=True \
#     hoi_use_temporal_dynamics=True \
#     clip_length=3 \
#     output_dir=$OUTPUT_DIR >> $OUTPUT_DIR/log.txt

# # DETR+Tracking+HOI^{TDE+JFT}
# python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=11112 --use_env \
#     src/train.py with deformable vidhoi vsgg \
#     resume='models/vidhoi/ais_detr+tracking+hoi_7adaa8f+hoi_use_temporal_dynamics/checkpoint_epoch7.pth' \
#     hoi_detection=True \
#     freeze_detr=False \
#     hoi_use_temporal_dynamics=True \
#     clip_length=3 \
#     output_dir=$OUTPUT_DIR >> $OUTPUT_DIR/log.txt


# # eval
# python -u -m torch.distributed.launch --nproc_per_node=2 --master_port=11112 --use_env \
#     src/train.py with deformable vidhoi vsgg \
#     resume='models/vidhoi/ais_detr+tracking+hoi_7adaa8f+hoi_use_temporal_dynamics/jointly-tune/checkpoint_epoch4.pth' \
#     hoi_detection=True freeze_detr=True \
#     hoi_use_temporal_dynamics=True hoi_relation_propagation_on_inference=True \
#     eval_only=True \
#     output_dir=$OUTPUT_DIR >> $OUTPUT_DIR/log.txt
