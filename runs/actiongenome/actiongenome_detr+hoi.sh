OUTPUT_DIR=$1

echo "OUTPUT_DIR=${OUTPUT_DIR}"
cd /218019030/projects/VideoSG-on-trackformer/
mkdir -p $OUTPUT_DIR

python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=11112 --use_env src/train.py \
    with deformable actiongenome hoi backbone=resnet101 \
    resume=models/actiongenome/ais_detr-resnet101_bf64ce0_run3/checkpoint_epoch20.pth \
    val_split=test_v500 \
    output_dir=$OUTPUT_DIR >> $OUTPUT_DIR/log.txt

# # eval
# python -u src/train.py \
#     with deformable actiongenome hoi backbone=resnet101 \
#     resume=models/actiongenome/ais_r101detr+hoi_4f459af_weakerdetr/checkpoint_epoch7.pth \
#     eval_only=True >> models/actiongenome/ais_r101detr+hoi_4f459af_weakerdetr/checkpoint_epoch7_evallog.txt
