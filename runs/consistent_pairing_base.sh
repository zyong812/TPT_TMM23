OUTPUT_DIR=$1

echo "OUTPUT_DIR=${OUTPUT_DIR}"
cd /218019030/projects/VideoSG-on-trackformer/
mkdir -p $OUTPUT_DIR

python -u src/train.py with deformable tracking mot17_cross_val consistent_pairing output_dir=$OUTPUT_DIR >> $OUTPUT_DIR/log.txt