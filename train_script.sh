#!/bin/zsh
# torchrun --nproc_per_node=4 --master_port=35768 train.py --loss=PPLoss --logit=l1_dist --class-proxy --output-dir=result/l1cp/PP
# torchrun --nproc_per_node=4 --master_port=35768 train.py --loss=PPLoss --logit=l2_dist --class-proxy --output-dir=result/l2cp/PP
# torchrun --nproc_per_node=4 --master_port=35768 train.py --loss=PPLoss --logit=l1_dist --output-dir=result/l1wcp/PP
# torchrun --nproc_per_node=4 --master_port=35768 train.py --loss=PPLoss --logit=l2_dist --output-dir=result/l2wcp/PP

# torchrun --nproc_per_node=4 --master_port=35768 train.py --loss=KKLoss --logit=l1_dist --class-proxy --output-dir=result/l1cp/KK
# torchrun --nproc_per_node=4 --master_port=35768 train.py --loss=KKLoss --logit=l2_dist --class-proxy --output-dir=result/l2cp/KK
# torchrun --nproc_per_node=4 --master_port=35768 train.py --loss=KKLoss --logit=l1_dist --output-dir=result/l1wcp/KK
# torchrun --nproc_per_node=4 --master_port=35768 train.py --loss=KKLoss --logit=l2_dist --output-dir=result/l2wcp/KK

# torchrun --nproc_per_node=4 --master_port=35768 train.py --loss=MKLoss --logit=l1_dist --class-proxy --output-dir=result/l1cp/MK
# torchrun --nproc_per_node=4 --master_port=35768 train.py --loss=MKLoss --logit=l2_dist --class-proxy --output-dir=result/l2cp/MK
# torchrun --nproc_per_node=4 --master_port=35768 train.py --loss=MKLoss --logit=l1_dist --output-dir=result/l1wcp/MK
# torchrun --nproc_per_node=4 --master_port=35768 train.py --loss=MKLoss --logit=l2_dist --output-dir=result/l2wcp/MK
# torchrun --nproc_per_node=4 --master_port=35757 train.py --logit=l2_dist --resume=result/l2wcp/MK/checkpoints/best_shot5_model.pth --tsne
torchrun --nproc_per_node=4 --master_port=35768 train.py --logit=l2_dist --loss=AProxy --output-dir=result/test/AP
# torchrun --nproc_per_node=4 --master_port=35757 train.py --logit=l2_dist --resume=result/test/MKP/checkpoints/best_shot5_model.pth --test-only