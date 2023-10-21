OUTPUT_DIR=$1

echo "OUTPUT_DIR=${OUTPUT_DIR}"
cd /218019030/projects/VideoSG-on-trackformer/
mkdir -p $OUTPUT_DIR

python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=11112 --use_env src/train.py with deformable actiongenome backbone=resnet101 \
    lr=0.0001 lr_backbone=0.00001 batch_size=1 val_split=test_v500 save_model_interval=1 \
    lr_drop=20 resume=models/pretrained/r101_deformable_detr-checkpoint.pth \
    output_dir=$OUTPUT_DIR > $OUTPUT_DIR/log.txt
