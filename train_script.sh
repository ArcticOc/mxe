#!/bin/zsh
torchrun --nproc_per_node=4 --master_port=35768 train.py --loss=PPLoss --data-path=data/CUB_200_2011 --output-dir=result/mini/cl2n/MKP