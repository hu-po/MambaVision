#!/bin/bash

MODEL=mamba_vision_T
DATA_DIR="/imagenette2"
DATA_PATH_TRAIN="train"
DATA_PATH_VAL="val"
BS=2
EXP=my_experiment
LR=5e-4
WD=0.05
DR=0.2

torchrun --nproc_per_node=1 mambavision/train.py \
    --dataset imagenette2 \
    --data_dir $DATA_DIR \
    --train-split $DATA_PATH_TRAIN \
    --val-split $DATA_PATH_VAL \
    --model $MODEL \
    --input-size 3 224 224 \
    --crop-pct 0.875 \
    --amp \
    --weight-decay ${WD} \
    --drop-path ${DR} \
    --batch-size $BS \
    --tag $EXP \
    --lr $LR \
    --class-map ''
