OUTPUT_DIR=$1

echo "OUTPUT_DIR=${OUTPUT_DIR}"
cd /218019030/projects/VideoSG-on-trackformer/
mkdir -p $OUTPUT_DIR

# eval
# python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=11112 --use_env src/train.py with deformable actiongenome vsgg train_split=train_v1000 val_split=test_v500 resume=models/actiongenome/ais_vsgg_nohoi_lr\=1e-5_65c6820/checkpoint_epoch9.pth eval_only=True output_dir=$OUTPUT_DIR >> $OUTPUT_DIR/log.txt

python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=11112 --use_env src/train.py with deformable actiongenome vsgg train_split=train_v1000 val_split=test_v200 resume=models/actiongenome/ais_detr-33d206b/checkpoint_epoch50.pth output_dir=$OUTPUT_DIR >> $OUTPUT_DIR/log.txt
