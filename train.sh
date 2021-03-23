#!/bin/bash

python -m torch.distributed.launch --nproc_per_node 4 train.py --resume
python -m torch.distributed.launch --nproc_per_node 4 train.py --weights yolov5m.pt --data data/droneai_c6.yaml --cfg yolov5m.yaml --epochs 300 --workers 256 --batch-size 256 --name droneai_c6_y5m
python -m torch.distributed.launch --nproc_per_node 4 train.py --weights yolov5l.pt --data data/droneai_c6.yaml --cfg yolov5l.yaml --epochs 300 --workers 256 --batch-size 256 --name droneai_c6_y5l
python -m torch.distributed.launch --nproc_per_node 4 train.py --weights yolov5x.pt --data data/droneai_c6.yaml --cfg yolov5x.yaml --epochs 300 --workers 256 --batch-size 256 --name droneai_c6_y5x
python -m torch.distributed.launch --nproc_per_node 4 train.py --weights yolov5s.pt --data data/droneai_c6.yaml --cfg yolov5s.yaml --epochs 300 --workers 256 --batch-size 256 --name droneai_c6_y5s