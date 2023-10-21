# End-to-End Video Scene Graph Generation with Temporal Propagation Transformer

This repository provides the official implementation of the [End-to-End Video Scene Graph Generation with Temporal Propagation Transformer](https://ieeexplore.ieee.org/abstract/document/10145598) paper.

## Installation

1. `pip3 install -r requirements.txt`
2. Install PyTorch>=1.5 and torchvision>=0.6 from [here](https://pytorch.org/get-started/previous-versions/#v150).
3. `pip install pycocotools`
4. Install MultiScaleDeformableAttention package: `python src/trackformer/models/ops/setup.py build --build-base=src/trackformer/models/ops/ install`

## Data preparation

### ActionGenome

1.Preprocess and dump frames following https://github.com/JingweiJ/ActionGenome

2.Convert to COCO annotation format using `python src/generate_coco_from_actiongenome.py`

### VidHOI

1.Download and prepare VidHOI following https://github.com/coldmanck/VidHOI

2.Dump frames
```
python src/generate_coco_from_vidhoi.py --task dump_frames
```

3.Convert to COCO annotations format
```
python src/generate_coco_from_vidhoi.py --task convert_coco_annotations
```

## Training & Evaluation

All the running scripts are in `./runs` directory.

1.Tain a Transformer-based detector for object detection in individual video frames.
```
sh ./runs/vidhoi_detr.sh your_output_dir
```

2.Fine-tune the Transformer-based detector together with the QPM module, to further build temporal associations of detected instances. 
```
sh ./runs/vidhoi_detr+tracking.sh your_output_dir
```

3.Freeze all parameters of the architecture learnt in previous step, and only optimize the modules for relation recognition.
```
sh ./runs/vidhoi_detr+tracking+hoi.sh your_output_dir
```

4.Jointly fine-tune the whole framework.
```
# Set freeze_detr=False in ./runs/vidhoi_detr+hoi.sh
sh ./runs/vidhoi_detr+hoi.sh your_output_dir
```

## Acknowledgement

The codebase builds upon [DETR](https://github.com/facebookresearch/detr), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [TrackFormer](https://github.com/timmeinhardt/trackformer), [STTran](https://github.com/timmeinhardt/trackformer) and [ByteTrack](https://github.com/ifzhang/ByteTrack). Thanks for their wonderful works.
