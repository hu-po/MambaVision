#!/bin/bash

MODEL=mamba_vision_T
DATA_PATH_TRAIN="/imagenette2/train"
DATA_PATH_VAL="/imagenette2/val"
BS=2
EXP=my_experiment
LR=5e-4
WD=0.05
DR=0.2

torchrun --nproc_per_node=1 mambavision/train.py --input-size 3 224 224 --crop-pct=0.875 \
--train-split=$DATA_PATH_TRAIN --val-split=$DATA_PATH_VAL --model $MODEL --amp --weight-decay ${WD} --drop-path ${DR} --batch-size $BS --tag $EXP --lr $LR
