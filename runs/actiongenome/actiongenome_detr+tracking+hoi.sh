OUTPUT_DIR=$1

echo "OUTPUT_DIR=${OUTPUT_DIR}"
cd /218019030/projects/VideoSG-on-trackformer/
mkdir -p $OUTPUT_DIR

# python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=11112 --use_env src/train.py \
#     with deformable actiongenome vsgg hoi_detection=True freeze_detr=False backbone=resnet101 \
#     resume=models/actiongenome/ais_r101detr+tracking+hoi_4f459af_run2/checkpoint_epoch5.pth \
#     hoi_use_temporal_dynamics=True clip_length=3 val_split=test_v500 \
#     output_dir=$OUTPUT_DIR >> $OUTPUT_DIR/log.txt

# eval
python -u src/train.py \
    with deformable actiongenome vsgg hoi_detection=True freeze_detr=True backbone=resnet101 \
    resume=models/actiongenome/ais_r101detr+tracking+hoi_4f459af_run2/jointly-tune/checkpoint_epoch4.pth \
    hoi_oracle_mode=False hoi_use_temporal_dynamics=True clip_length=3 eval_only=True hoi_relation_propagation_on_inference=False \
    > models/actiongenome/ais_r101detr+tracking+hoi_4f459af_run2/jointly-tune/checkpoint_epoch4_evallog.txt
