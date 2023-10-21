demo_video_name=MOT17-02-FRCNN
model_folder=$1

python -u src/track.py with \
    dataset_name=$demo_video_name \
    output_dir="${model_folder}/demos" \
    frame_range.start=0.5 \
    obj_detect_checkpoint_file="${model_folder}/checkpoint.pth" \
    verbose=True \
    write_images=debug

ffmpeg -framerate 8 -start_number 301 -i "${model_folder}/demos/${demo_video_name}/${demo_video_name}/%6d.jpg" -pix_fmt yuv420p -c:v libx264 "${model_folder}/demos/out.mp4"

echo 'Done!'
