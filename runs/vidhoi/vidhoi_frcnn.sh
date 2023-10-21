OUTPUT_DIR=$1

echo "OUTPUT_DIR=${OUTPUT_DIR}"
cd /218019030/projects/VideoSG-on-trackformer/
mkdir -p $OUTPUT_DIR

python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=11112 --use_env src/train_frcnn.py with frcnn vidhoi batch_size=2 output_dir=$OUTPUT_DIR >> $OUTPUT_DIR/log.txt
