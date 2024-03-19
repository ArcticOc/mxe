# Distributed training for Few-shot learning

The core of training code is based on [Pytorch classification code](https://github.com/pytorch/vision/tree/main/references/classification).

## Usage

Loss functions including the proposed one are implemented in `loss.py`.

### Training
#### Ours
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py 
--data-path <PATH_TO_miniImageNet> --output-dir ./result/resnet12/MultiXE_l1/dpp_sgd_ep120wm10/  \
--model resnet12 --projection --projection-feat-dim 128 \
--loss MultiXELoss --logit l1_dist --logit-temperature 1 --class-proxy \
--amp \
--epochs 120 --batch-size 128 --opt sgd --lr 0.1 --wd 0.0005 \
--lr-scheduler steplr --lr-gamma 0.1 --lr-step-size 84 \
--lr-warmup-method linear --lr-warmup-epochs 10 
```

#### NCA
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py 
--data-path <PATH_TO_miniImageNet> --output-dir ./result/resnet12/NCA/dpp_sgd_ep120wm10/  \
--model resnet12 --projection --projection-feat-dim 128 \
--loss FewShotNCALoss --logit l2_dist --logit-temperature 0.9 --class-proxy \
--amp \
--epochs 120 --batch-size 128 --opt sgd --lr 0.1 --wd 0.0005 \
--lr-scheduler steplr --lr-gamma 0.1 --lr-step-size 84 \
--lr-warmup-method linear --lr-warmup-epochs 10 
```

#### PN
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py 
--data-path <PATH_TO_miniImageNet> --output-dir ./result/resnet12/PN/dpp_sgd_ep120wm10/  \
--model resnet12 --projection --projection-feat-dim 128 \
--loss PNLoss --logit l2_dist --logit-temperature 1 --class-proxy \
--amp \
--epochs 120 --batch-size 128 --opt sgd --lr 0.1 --wd 0.0005 \
--lr-scheduler steplr --lr-gamma 0.1 --lr-step-size 84 \
--lr-warmup-method linear --lr-warmup-epochs 10 
```

### Evaluation
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py 
--data-path <PATH_TO_miniImageNet>  \
--resume <PATH_TO_result>/checkpoints/best_shot1_model.pth  \
--model resnet12 --projection --projection-feat-dim 128 --test-only 
```

## Results
### Computation resource
- python 3.8.18 
- pytorch 1.12.1
- cuda 11.3
- Titan V x 4 

#### mini-ImageNet

- Val-set (3000 iterations) 

| Method  | 1-shot | 5-shot |
|---|---|---|
| NCA    | 63.36 | 81.08   |
| PN     | 66.03 | 82.06   |
| Ours (MultiXE)| **68.13** | **83.14**   |

- Test-set (10000 iterations) 

| Method  | 1-shot | 5-shot |
|---|---|---|
| NCA    | 62.10 | 80.00   |
| PN     | 63.23 | 79.46   |
| Ours (MultiXE)| **65.15** | **80.40**   |

## TODO
Do ablation studies.

## Author
takumi.kobayashi (At) aist.go.jp
