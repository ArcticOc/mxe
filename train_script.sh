#!/bin/zsh

torchrun --nproc_per_node=3 --master_port=35757 train.py --logit=l1_dist --resume=result/PPLoss_l1_dist/checkpoints/best_shot5_model.pth --tsne --test-only


# torchrun --nproc_per_node=4 --master_port=35768 train.py --logit=l1_dist --loss=MKLoss --output-dir=result/mk --class-proxy

# torchrun --nproc_per_node=1 --master_port=35768 train.py --logit=l1_dist --loss=ProtoNet --output-dir=result/pn --class-aware-sampler='64,8' --batch-size=512

# torchrun --nproc_per_node=4 --master_port=35757 train.py --data-path=data/mini_imagenet --resume=result_comp/vit/checkpoints/best_shot5_model.pth --test-only --projection-feat-dim=384

# logit_values=("l2_dist")
# loss_values=("MKLoss" "PPLoss" "KKLoss")

# MASTER_PORT=35768
# NPROC=4

# for logit in ${logit_values[@]}; do
#   for loss in ${loss_values[@]}; do
#     echo "Running with logit=$logit and loss=$loss..."
#     torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT train.py --logit=$logit --loss=$loss --output-dir="result/${loss}_${logit}" --class-proxy
#   done
# done