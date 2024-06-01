"""
This code is based on
https://github.com/pytorch/vision/tree/main/references/classification

modified by Takumi Kobayashi
"""

import datetime
import os
import time

import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter

import loss
import models
from configuration import args
from src import utils
from src.dataloader import load_data
from src.evaluate import evaluate
from src.train_one_epoch import train_one_epoch
from src.tSNE import TSNEVisualizer


def main(args):
    writer = (
        SummaryWriter(
            # f"./runs/{args.loss}_{args.data_path.split('/')[-1]}_{datetime.now().strftime('%b%d_%H%M')}"
            f"./runs/{'_'.join(args.output_dir.split('/')[1:])}_{datetime.datetime.now().strftime('%b%d_%H%M')}"
        )
        if args.output_dir
        else None
    )

    if args.output_dir and not args.test_only:
        utils.mkdir(os.path.join(args.output_dir, "checkpoints"))

    utils.init_distributed_mode(args)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    (
        dataset,
        dataset_avg,
        dataset_val,
        dataset_test,
        train_sampler,
        train_avg_sampler,
        val_sampler,
        test_sampler,
    ) = load_data(*[os.path.join(args.data_path, x) for x in ["train", "val", "test"]], args)

    args.num_classes = len(dataset.classes)
    collate_fn = default_collate

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_avg = torch.utils.data.DataLoader(
        dataset_avg,
        batch_size=args.batch_size,
        sampler=train_avg_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    print("Creating model")
    model = getattr(models, args.model)(
        num_classes=args.num_classes,
        feature_dim=args.projection_feat_dim,
        projection=args.projection,
        use_fc=False,
        image_size=args.train_crop_size,
    )
    model.to(device)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    class_proxy_ = [args.num_classes, args.projection_feat_dim] if args.class_proxy else None

    criterion = loss.WrapperLoss(loss=getattr(loss, args.loss)(**vars(args)), class_proxy=class_proxy_)

    """ criterion_val = loss.WrapperLoss(
        loss=getattr(loss, args.loss)(**vars(args)),
        class_proxy=[args.num_classes, 640] if args.class_proxy else None,
    )  # widths=[64, 160, 320, 640] """
    criterion.to(device)
    # criterion_val.to(device)

    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
    )
    """ parameters = parameters + utils.set_weight_decay(
        criterion,
        args.weight_decay,
    ) """

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            eps=0.0316,
            alpha=0.9,
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=args.lr_warmup_decay,
                total_iters=args.lr_warmup_epochs,
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=args.lr_warmup_decay,
                total_iters=args.lr_warmup_epochs,
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[args.lr_warmup_epochs],
        )  # SequntialLR cause warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)
        if len([x for x in criterion.parameters()]) > 0:
            criterion = torch.nn.parallel.DistributedDataParallel(criterion)
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])
    if args.tsne:
        tSNE = TSNEVisualizer(data_loader_val, model, args)
        tSNE.visualize_with_tsne()

        return print("t-SNE visualization is saved.")
    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            meter = evaluate(
                model_ema,
                args.shot,
                args.val_iter,
                args,
                data_loader_avg,
                data_loader_val,
                device=device,
                header="Val (EMA)",
            )
            meter = evaluate(
                model_ema,
                args.shot,
                args.test_iter,
                args,
                data_loader_avg,
                data_loader_test,
                device=device,
                header="Test (EMA)",
            )
        else:
            meter = evaluate(
                model,
                args.shot,
                args.val_iter,
                args,
                data_loader_avg,
                data_loader_val,
                device=device,
                header="Val",
            )
            meter = evaluate(
                model,
                args.shot,
                args.test_iter,
                args,
                data_loader_avg,
                data_loader_test,
                device=device,
                header="Test",
            )
        return

    print("Start training")
    metric_logger = utils.MetricLogger(delimiter="  ")
    for s in args.shot:
        metric_logger.add_meter(f"best_shot{s}", utils.BestValue())
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # training
        if args.distributed:
            train_sampler.set_epoch(epoch)
        meter = train_one_epoch(
            model,
            criterion,
            optimizer,
            data_loader,
            device,
            epoch,
            args,
            model_ema,
            scaler,
            writer,
        )

        lr_scheduler.step()
        # evaluation
        if (epoch + 1) % args.val_freq == 0:
            meter = evaluate(
                model,
                args.shot,
                args.val_iter,
                args,
                data_loader_avg,
                data_loader_val,
                writer=writer,
                epoch=epoch,
                device=device,
                header="Val",
            )

            if model_ema:
                meter_ema = evaluate(
                    model_ema,
                    args.shot,
                    args.val_iter,
                    args,
                    data_loader_avg,
                    data_loader_val,
                    writer=writer,
                    epoch=epoch,
                    device=device,
                    header="Val (EMA)",
                )

            if args.output_dir:
                checkpoint = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                    **{k: meter.meters[k].global_avg for k in meter.meters.keys()},  # noqa: SIM118
                }
                for s in args.shot:
                    metric_logger.meters[f"best_shot{s}"].update(meter.meters[f"shot{s}_acc"].global_avg)
                if model_ema:
                    checkpoint["model_ema"] = model_ema.state_dict()
                    checkpoint.update(
                        {
                            k + "_ema": meter_ema.meters[k].global_avg
                            for k in meter_ema.meters.keys()  # noqa: SIM118
                        }
                    )
                    for s in args.shot:
                        metric_logger.meters[f"best_shot{s}"].update(meter_ema.meters[f"shot{s}_acc"].global_avg)

                if scaler:
                    checkpoint["scaler"] = scaler.state_dict()
                if args.save_all_checkpoints:
                    utils.save_on_master(
                        checkpoint,
                        os.path.join(
                            args.output_dir,
                            "checkpoints",
                            f"checkpoint_ep{epoch:02d}.pth",
                        ),
                    )
                else:
                    utils.save_on_master(
                        checkpoint,
                        os.path.join(args.output_dir, "checkpoints", "checkpoint.pth"),
                    )
                for s in args.shot:
                    if metric_logger.meters[f"best_shot{s}"].is_best:
                        utils.save_on_master(
                            checkpoint,
                            os.path.join(
                                args.output_dir,
                                "checkpoints",
                                f"best_shot{s}_model.pth",
                            ),
                        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args.shot = [int(x) for x in args.shot.split(",")]
    main(args)
