#!/bin/zsh

# torchrun --nproc_per_node=4 --master_port=35757 train.py --logit=l2_dist --resume=result/l1wcp/KK/checkpoints/best_shot5_model.pth --tsne

torchrun --nproc_per_node=4 --master_port=35768 train.py --logit=l1_dist --loss=MKLoss --output-dir=result/mk --class-proxy

# torchrun --nproc_per_node=1 --master_port=35768 train.py --logit=l1_dist --loss=ProtoNet --output-dir=result/pn --class-aware-sampler='64,8' --batch-size=512

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
# lr=('1e-6' '3e-6' '5e-6' '7e-6' '9e-6' '1e-5' '3e-5' '5e-5' '7e-5' '9e-5' '1e-4')
# epochs=(45 50 55 40)
# opt=('adamw' 'adam')
# for o in "${opt[@]}"
# do
    # for l in "${lr[@]}"
    # do
    #     for e in "${epochs[@]}"
    #     do
    #         torchrun --nproc_per_node=4 --master_port=35757 train.py --data-path=data/mini_imagenet --output-dir=result/vit/lr_${l}+epochs_${e} --loss=MKLoss --class-proxy --logit=l1_dist --epochs=${e} --projection-feat-dim=384 --lr=${l} --opt='adamw' --wd=3e-4 --lr-step-size=30 --lr-warmup-epochs=25

    #         echo "RTraining lr:${l} and epochs:${e}" >> result3.log
    #         torchrun --nproc_per_node=4 --master_port=35757 train.py --data-path=data/mini_imagenet --resume=result/vit/lr_${l}+epochs_${e}/checkpoints/best_shot5_model.pth --test-only --projection-feat-dim=384 | tail -n 16 | tee -a result3.log

    #     done
    # done
# done
# torchrun --nproc_per_node=4 --master_port=35757 train.py --data-path=data/mini_imagenet --output-dir=result_comp/vit --loss=MKLoss --class-proxy --logit=l1_dist --epochs=60 --projection-feat-dim=384 --lr=1e-5 --opt='adamw' --wd=3e-4 --lr-step-size=30 --lr-warmup-epochs=25

# torchrun --nproc_per_node=4 --master_port=35757 train.py --data-path=data/mini_imagenet --resume=result_comp/vit/checkpoints/best_shot5_model.pth --test-only --projection-feat-dim=384

sed -i '/.*time.*/d' result3.log
sed -i '/Train (stats)/d' result3.log
