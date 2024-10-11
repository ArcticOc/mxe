#!/bin/zsh

# torchrun --nproc_per_node=4 --master_port=35757 train.py --logit=l2_dist --resume=result/l1wcp/KK/checkpoints/best_shot5_model.pth --tsne

# torchrun --nproc_per_node=4 --master_port=35768 train.py --logit=l1_dist --loss=MKLoss --output-dir=result/bs100/cf --class-proxy --batch-size=100 --data-path=data/cifar_fs

# losses=('MLLoss')
# datas=('cifar_fs' 'mini_imagenet')

# for i in {1..10}
# do
#     for loss in "${losses[@]}"
#     do
#         for d in "${datas[@]}"
#         do
#             torchrun --nproc_per_node=4 --master_port=35757 train.py --data-path=data/${d} --norm=1 --output-dir=result_comp/${d}/${loss} --loss=${loss} --class-proxy

#             echo "Run $i: Training ${loss} loss on ${d}" >> result2.log
#             torchrun --nproc_per_node=4 --master_port=35757 train.py --data-path=data/${d} --resume=result_comp/${d}/${loss}/checkpoints/best_shot5_model.pth --test-only | tail -n 16 | tee -a result2

#         done
#     done
# done
lrw=(5 10 15 20 25 30 40)
lrs=(10 15 20 25 30 35 40)
# opt=('adamw' 'adam')
# for o in "${opt[@]}"
# do
    # for lw in "${lrw[@]}"
    # do
    #     for ls in "${lrw[@]}"
    #     do
    #         torchrun --nproc_per_node=4 --master_port=35757 train.py --data-path=data/mini_imagenet --output-dir=result/vit/lrw_${lw}+lrs_${ls} --loss=MKLoss --class-proxy --logit=l1_dist --epochs=50 --projection-feat-dim=384 --lr=1e-6 --opt='adamw' --wd=3e-4 --lr-step-size=${ls} --lr-warmup-epochs=${lw}

    #         echo "RTraining lrw:${lw} and lrs: ${ls}" >> result2.log
    #         torchrun --nproc_per_node=4 --master_port=35757 train.py --data-path=data/mini_imagenet --resume=result/vit/lrw_${lw}+lrs_${ls}/checkpoints/best_shot5_model.pth --test-only --projection-feat-dim=384 | tail -n 16 | tee -a result2.log

    #     done
    # done
# done
# torchrun --nproc_per_node=4 --master_port=35757 train.py --data-path=data/mini_imagenet --output-dir=result_comp/vit --loss=MKLoss --class-proxy --logit=l1_dist --epochs=50 --projection-feat-dim=384 --lr=1e-6 --opt='adamw' --wd=3e-4 --lr-step-size=25 --lr-warmup-epochs=10

# torchrun --nproc_per_node=4 --master_port=35757 train.py --data-path=data/mini_imagenet --resume=result_comp/vit/checkpoints/best_shot5_model.pth --test-only --projection-feat-dim=384

sed -i '/.*time.*/d' result2.log
sed -i '/Train (stats)/d' result2.log
