#!/bin/zsh

# torchrun --nproc_per_node=4 --master_port=35757 train.py --logit=l2_dist --resume=result/l1wcp/KK/checkpoints/best_shot5_model.pth --tsne

# torchrun --nproc_per_node=4 --master_port=35768 train.py --logit=l2_dist --loss=MKLoss --output-dir=result/abl/PM --class-proxy --data-path=data/cifar_fs

# losses=('MK' 'PP' 'KK')
# logits=('l1' 'l2')
# feat=(160 176)
# for loss in "${losses[@]}"
# do
#     for logit in "${logits[@]}"
#     do
#         for d in "${feat[@]}"
#         do
#             echo "Training ${logit}_cp/feat_${d} with ${loss} loss" >> result.log
#             torchrun --nproc_per_node=4 --master_port=35757 train.py --projection-feat-dim=${d} --resume=result_comp/feat${d}/${logit}cp/${loss}/checkpoints/best_shot5_model.pth --test-only | tail -n 16 | tee -a result.log
#             echo "Training ${logit}_wcp/feat_${d} with ${loss} loss" >> result.log
#             torchrun --nproc_per_node=4 --master_port=35757 train.py --projection-feat-dim=${d} --resume=result_comp/feat${d}/${logit}wcp/${loss}/checkpoints/best_shot5_model.pth --test-only | tail -n 16 | tee -a result.log
#         done
#     done
# done
# for i in {1..2}
# do
    # torchrun --nproc_per_node=4 --master_port=35757 train.py --resume=result/MK/checkpoints/best_shot5_model.pth --test-only
# done

# dim=(64 96 128 160 192 224 240 256 512)
# for d in {160..224..16}
# do
    # torchrun --nproc_per_node=4 --master_port=35768 train.py --logit=l2_dist --loss=MKLoss --output-dir=result/test/MK_${d} --class-proxy --projection-feat-dim=${d}
# done
# losses=('ML')
# logits=(0.5 1.5)
# datas=('cifar_fs' 'tiered_imagenet')
# for loss in "${losses[@]}"
# do
#     for logit in "${logits[@]}"
#     do
#         for d in "${datas[@]}"
#         do

#             torchrun --nproc_per_node=4 --master_port=35757 train.py --norm=${logit} --output-dir=result_comp/${logit}cp --loss=${loss}Loss --class-proxy

#             echo "Training ${logit}_cp with ${loss} loss_1" >> result2.log
#             torchrun --nproc_per_node=4 --master_port=35787 train.py --resume=result_comp/${logit}cp/checkpoints/best_shot1_model.pth --test-only | tail -n 16 | tee -a result2.log

#             torchrun --nproc_per_node=4 --master_port=35757 train.py --norm=${logit} --output-dir=result_comp/${logit}wcp --loss=${loss}Loss

#             echo "Training ${logit}_wcp with ${loss} loss_1" >> result2.log
#             torchrun --nproc_per_node=4 --master_port=35787 train.py --resume=result_comp/${logit}wcp/checkpoints/best_shot1_model.pth --test-only | tail -n 16 | tee -a result2.log

#         done
#     done
# done
losses=('ProtoNet')
logits=('l2')
datas=('tiered_imagenet')
for loss in "${losses[@]}"
do
    for logit in "${logits[@]}"
    do
        for d in "${datas[@]}"
        do

            torchrun --nproc_per_node=1 --master_port=35757 train.py --data-path=data/${d} --logit=${logit}_dist --output-dir=result_comp/${d}/${logit}/${loss} --loss=${loss} --class-aware-sampler=64,8 --batch-size=512

            echo "Training ${logit} with ${loss} loss on ${d}" >> result.log
            torchrun --nproc_per_node=1 --master_port=35757 train.py --data-path=data/${d} --resume=result_comp/${d}/${logit}/${loss}/checkpoints/best_shot5_model.pth --test-only | tail -n 16 | tee -a result.log

            # echo "Training ${logit} with ${loss} loss on ${d}_1" >> result.log
            # torchrun --nproc_per_node=1 --master_port=35757 train.py --data-path=data/${d} --resume=result_comp/${d}/${logit}/${loss}/checkpoints/best_shot1_model.pth --test-only | tail -n 16 | tee -a result.log

        done
    done
done

sed -i '/.*time.*/d' result.log
sed -i '/Train (stats)/d' result.log