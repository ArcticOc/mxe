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

torchrun --nproc_per_node=4 --master_port=35757 train.py --data-path=data/cifar_fs --output-dir=result_comp/bce_test --loss=MLLoss --class-proxy --logit=l1_dist --norm=1

torchrun --nproc_per_node=4 --master_port=35757 train.py --data-path=data/cifar_fs --resume=result_comp/bce_test/checkpoints/best_shot5_model.pth --test-only

# sed -i '/.*time.*/d' result2.log
# sed -i '/Train (stats)/d' result2.log
