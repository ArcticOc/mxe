import time

import torch
import torch.nn as nn

from . import utils


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    device,
    epoch,
    args,
    model_ema=None,
    scaler=None,
    writer=None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.3g}"))
    metric_logger.add_meter(
        "img/s", utils.SmoothedValue(window_size=10, fmt="{value:.1f}")
    )

    header = f"Epoch: [{epoch + 1}]"
    for i, (image, target) in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output, output_fc, lamda = model(image)
            loss = criterion(output, target, lamda)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.module.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        if writer is not None:
            writer.add_scalar("Loss/train", loss.item(), epoch * len(data_loader) + i)
            if hasattr(model.module, "proj1"):
                layer = model.module.proj1
                writer.add_histogram(
                    "Gradients",
                    layer.weight.grad,
                    epoch * len(data_loader) + i,
                )

    return metric_logger
