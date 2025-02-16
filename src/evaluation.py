"""
This code is based on
https://github.com/fiveai/on-episodes-fsl/

modified by Takumi Kobayashi
Improved by Tong Wu
"""

import os
import random

import numpy as np
import torch
from numpy import linalg as LA
from scipy.stats import mode

from . import utils


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def metric_class_type(
    gallery,
    query,
    train_label,
    test_label,
    shot,
    args,
    train_mean=None,
    norm_type="CL2N",
):
    """
    this function performs test-time classification
    args.{num_NN, soft_assignment, median_prototype, test_way, num_classes}
    """

    # generate the classification space according to normalisation
    if norm_type == "CL2N":
        gallery = gallery - train_mean
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query - train_mean
        query = query / LA.norm(query, 2, 1)[:, None]
    elif norm_type in ("L2N", "COS"):
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query / LA.norm(query, 2, 1)[:, None]
    elif norm_type == "CCOS":
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query / LA.norm(query, 2, 1)[:, None]
        train_mean = train_mean / LA.norm(train_mean, 2, 0)
        gallery = gallery - train_mean
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query - train_mean
        query = query / LA.norm(query, 2, 1)[:, None]
    elif norm_type == "CCOS1":
        gallery = gallery / LA.norm(gallery, 1, 1)[:, None]
        query = query / LA.norm(query, 1, 1)[:, None]
        train_mean = train_mean / LA.norm(train_mean, 1, 0)
        gallery = gallery - train_mean
        gallery = gallery / LA.norm(gallery, 1, 1)[:, None]
        query = query - train_mean
        query = query / LA.norm(query, 1, 1)[:, None]
    elif norm_type == "UN":
        pass

    # Prototype classifier
    if args.classifier == "nc":
        if args.median_prototype:
            gallery = np.median(gallery.reshape(args.test_way, shot, gallery.shape[-1]), axis=1)
        else:
            gallery = gallery.reshape(args.test_way, shot, gallery.shape[-1]).mean(1)
        if norm_type == "COS":
            gallery = gallery / LA.norm(gallery, 2, 1)[:, None]

        train_label = train_label[::shot]
        subtract = gallery[:, None, :] - query
        distance = LA.norm(subtract, 2, axis=-1)
        idx = np.argpartition(distance, 1, axis=0)[:1]
        nearest_samples = np.take(train_label, idx)
        out = mode(nearest_samples, axis=0, keepdims=False)[0]
        out = out.astype(int)
        test_label = np.array(test_label)
        acc = (out == test_label).mean()
        return acc

    # Soft assignment classifier
    if args.classifier == "sa":
        subtract = gallery[:, None, :] - query
        distance = np.exp(-1 * np.sum(subtract**2, axis=2))
        norm_distance = distance / distance.sum(0)[None, :]
        # reshape the norm_distances and sum the likelihoods to get likelihood for each class
        norm_distance = norm_distance.reshape(args.test_way, shot, norm_distance.shape[-1]).sum(1)
        # get the train labels
        train_label = train_label[::shot]
        # get predictions
        prediction_idx = np.argmax(norm_distance, axis=0)
        predictions = np.take(train_label, prediction_idx)
        acc = (predictions == test_label).mean()
        return acc

    if args.classifier == "gsa":
        subtract = gallery[:, None, :] - query
        distance = np.exp(-1 * np.sum(subtract**2, axis=2))
        # subtract = np.abs(gallery[:, None, :] - query)
        # distance = np.exp(-1 * np.sum(subtract, axis=2))
        norm_distance = distance / distance.sum(0)[None, :]

        norm_distance = norm_distance.reshape(args.test_way, shot, norm_distance.shape[-1])
        soft_assignment = np.prod(norm_distance, axis=1)

        train_label = train_label[::shot]

        prediction_idx = np.argmax(soft_assignment, axis=0)
        predictions = np.take(train_label, prediction_idx)

        acc = (predictions == test_label).mean()
        return acc


def meta_evaluate(data, train_mean, shot, num_iter, args):
    """
    args.{num_NN, soft_assignment, median_prototype, test_way, num_classes,
    out_dir, test_query}
    """
    cl2n_list = []
    for _ in range(num_iter):
        # record accuracies for all three types of normalisation
        train_data, test_data, train_label, test_label = sample_case(data, shot, args)
        acc = metric_class_type(
            train_data,
            test_data,
            train_label,
            test_label,
            shot,
            args,
            train_mean=train_mean,
            norm_type=args.eval_norm_type,  # "CL2N",
        )
        cl2n_list.append(acc)

    # save
    if args.output_dir is not None:
        np.save(
            os.path.join(args.output_dir, f"meta_evaluate_shot{shot}.npy"),
            np.array(cl2n_list),
        )
    cl2n_mean, cl2n_conf = compute_confidence_interval(cl2n_list)
    return cl2n_mean, cl2n_conf


def sample_case(ld_dict, shot, args):
    """
    args.{test_way, test_query}
    """
    sample_class = random.sample(list(ld_dict.keys()), args.test_way)
    train_input = []
    test_input = []
    test_label = []
    train_label = []
    for each_class in sample_class:
        samples = random.sample(ld_dict[each_class], shot + args.test_query)
        train_label += [each_class] * len(samples[:shot])
        test_label += [each_class] * len(samples[shot:])
        train_input += samples[:shot]
        test_input += samples[shot:]
    train_input = np.array(train_input).astype(np.float32)
    test_input = np.array(test_input).astype(np.float32)
    return train_input, test_input, train_label, test_label


def validate_loss(val_loader, model, nca_f, xent_f, args):
    """
    input:
    :param: val loader. DataLoader with validation images loaded.
    :model: neural network model
    :nca_f: NCA criterion
    :xent_f: cross-entropy criterion
    args.{xent_weight}
    """
    metric_logger = utils.MetricLogger(delimiter="  ")
    model.eval()

    with torch.no_grad():
        for input, target in val_loader:
            if args.xent_weight > 0:
                features, fc_output = model(input, use_fc=True)
                xent_loss = xent_f(fc_output, target.cuda(non_blocking=True))
                metric_logger.meters["xent_loss"].update(xent_loss.item(), n=input.size(0))
            else:
                features, _ = model(input, use_fc=False)

            nca_loss = nca_f(features, target)
            metric_logger.meters["nca_loss"].update(nca_loss.item(), n=input.size(0))

    return metric_logger
