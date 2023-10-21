OUTPUT_DIR=$1

echo "OUTPUT_DIR=${OUTPUT_DIR}"
cd /218019030/projects/VideoSG-on-trackformer/
mkdir -p $OUTPUT_DIR

python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=11112 --use_env src/train.py with deformable vidhoi vsgg resume=models/vidhoi/ais_detr_b82a0c8/checkpoint_best_BBOX_AP_IoU_0_50-0_95.pth output_dir=$OUTPUT_DIR >> $OUTPUT_DIR/log.txt
