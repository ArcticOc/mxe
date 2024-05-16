import collections

import torch

from . import utils
from .evaluation import meta_evaluate


def evaluate(
    model,
    shots,
    num_iter,
    eval_params,
    data_loader_avg,
    data_loader,
    device,
    epoch=0,
    print_freq=100,
    header="Val",
    writer=None,
    loss=None,
):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    num_processed_samples = 0
    with torch.inference_mode():
        # compute training mean
        train_avg = 0
        num = 0
        print("Train (stats)")
        for image, _ in data_loader_avg:
            image = image.to(device, non_blocking=True)
            train_avg = train_avg + model(image)[0].mean(0)
            num += 1
        train_avg = train_avg / num
        train_avg = utils.reduce_across_processes(train_avg, op="MEAN")

        # compute test-set embeddings
        test_embeddings, test_labels = [], []
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image, target = (
                image.to(device, non_blocking=True),
                target.to(device, non_blocking=True),
            )
            output = model(image)[0]
            if header in ("Val", "Val (EMA)") and eval_params.test_only is not True:
                val_loss = loss(output, target)
                writer.add_scalar(
                    f"{header}_loss", val_loss.item(), epoch
                ) if writer is not None else None
            batch_size = image.shape[0]
            num_processed_samples += batch_size

            test_embeddings.append(output)
            test_labels.append(target)

        test_embeddings = torch.cat(test_embeddings)
        test_labels = torch.cat(test_labels)
        test_embeddings = utils.gather_across_processes(test_embeddings)
        test_labels = utils.gather_across_processes(test_labels)

    # evaluation
    if utils.is_main_process():
        train_avg = train_avg.cpu().data.numpy()
        out_dict = collections.defaultdict(list)
        for out, label in zip(test_embeddings, test_labels):  # noqa: B905
            out_dict[label.item()].append(out.cpu().data.numpy())
        for s in shots:
            shot_info = meta_evaluate(out_dict, train_avg, s, num_iter, eval_params)
            metric_logger.meters[f"shot{s}_acc"].update(shot_info[0] * 100, n=1)
            metric_logger.meters[f"shot{s}_conf"].update(shot_info[1] * 100, n=1)
            accuracy = metric_logger.meters[f"shot{s}_acc"].global_avg
            writer.add_scalar(
                f"{header}/Acc/shot_{s}", accuracy, epoch
            ) if writer is not None else None
    else:
        for s in shots:  # dummy
            metric_logger.meters[f"shot{s}_acc"].update(0, n=1)

    print(f"{header}")
    for k in metric_logger.meters.keys():  # noqa: SIM118
        print("{} {:.2f}".format(k, metric_logger.meters[k].global_avg))
    return metric_logger
