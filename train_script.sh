#!/bin/zsh

# torchrun --nproc_per_node=4 --master_port=35757 train.py --logit=l2_dist --resume=result/l1wcp/KK/checkpoints/best_shot5_model.pth --tsne
# torchrun --nproc_per_node=4 --master_port=35768 train.py --logit=l2_dist --loss=MKPLoss --output-dir=result/test/MKP
# losses=('MK' 'PP' 'KK')
# logits=('l1' 'l2')
# for loss in "${losses[@]}"
# do
#     for logit in "${logits[@]}"
#     do
#         echo "Training ${logit}_cp with ${loss} loss" >> result.log
#         torchrun --nproc_per_node=4 --master_port=35757 train.py --logit=${logit}_dist --resume=result/${logit}cp/${loss}/checkpoints/best_shot5_model.pth --test-only | tail -n 16 | tee -a result.log
#         echo "Training ${logit}_wcp with ${loss} loss" >> result.log
#         torchrun --nproc_per_node=4 --master_port=35757 train.py --logit=${logit}_dist --resume=result/${logit}wcp/${loss}/checkpoints/best_shot5_model.pth --test-only | tail -n 16 | tee -a result.log
#     done
# done

# for i in {1..5}
# do
#     torchrun --nproc_per_node=4 --master_port=35768 train.py --logit=l2_dist --loss=MKLoss --class-proxy --output-dir=result/test/l2/MK_$i
#     torchrun --nproc_per_node=4 --master_port=35768 train.py --logit=l1_dist --loss=MKLoss --class-proxy --output-dir=result/test/l1/MK_$i
# done
torchrun --nproc_per_node=4 --master_port=35768 train.py --logit=l2_dist --loss=MKLoss --class-proxy
