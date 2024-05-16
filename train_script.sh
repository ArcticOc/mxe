#!/bin/zsh
torchrun --nproc_per_node=4 --master_port=35768 train.py --loss=MKLoss --output-dir=result/mini/ccos/MKP