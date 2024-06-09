"""
This code is based on
https://github.com/fiveai/on-episodes-fsl/

modified by Takumi Kobayashi
"""

import collections
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
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


def get_metric(metric_type):
    """
    :param metric_type (str): choose which metric function to use (cosine, euclidean, l1, l2)
    :return metric function (callable function):
    """
    METRICS = {
        "cosine": lambda gallery, query: 1.0 - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        "euclidean": lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).sum(2),
        "l1": lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
        "l2": lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
    }
    return METRICS[metric_type]


def metric_prediction(gallery, query, train_label, metric_type):
    gallery = gallery.view(gallery.shape[0], -1)
    query = query.view(query.shape[0], -1)
    distance = get_metric(metric_type)(gallery, query)
    predict = torch.argmin(distance, dim=1)
    predict = torch.take(train_label, predict)
    return predict


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

    # if we don't do soft assignment, and no nearest neighbour (num_NN = 1), then compute
    # the prototypes
    # if (args.num_NN == 1 or shot == 1) and not args.soft_assignment:
    if not args.disable_nearest_mean_classifier and not args.soft_assignment:
        if args.median_prototype:
            gallery = np.median(gallery.reshape(args.test_way, shot, gallery.shape[-1]), axis=1)
        else:
            gallery = gallery.reshape(args.test_way, shot, gallery.shape[-1]).mean(1)
        if norm_type == "COS":
            gallery = gallery / LA.norm(gallery, 2, 1)[:, None]

        train_label = train_label[::shot]
        num_NN = 1
    else:
        num_NN = args.num_NN

    # If we do evaluation with soft assignment we evaluate differently.
    if args.soft_assignment:
        subtract = gallery[:, None, :] - query
        distance = np.exp(-1 * np.sum(subtract**2, axis=2))
        norm_distance = distance / distance.sum(0)[None, :]

        # we can do kNN if number of shots > 1
        if num_NN > 1 and shot != 1:
            # get the closest labels
            idx = np.argsort(norm_distance, axis=0)[-num_NN:]
            nearest_samples_labels = np.take(train_label, idx)
            nearest_samples_distances = np.sort(norm_distance, axis=0)[-num_NN:]

            weighted_nearest_neighbours = np.zeros((args.num_classes, nearest_samples_labels.shape[-1]))
            # sum the total contribution of each example
            for i, row in enumerate(nearest_samples_labels):
                for j, element in enumerate(row):
                    weighted_nearest_neighbours[element, j] += nearest_samples_distances[i, j]

            # predict
            predictions = np.argmax(weighted_nearest_neighbours, axis=0)
            acc = (predictions == test_label).mean()
            return acc
        else:
            # reshape the norm_distances and sum the likelihoods to get likelihood for each class
            norm_distance = norm_distance.reshape(args.test_way, shot, norm_distance.shape[-1]).sum(1)
            # get the train labels
            train_label = train_label[::shot]

            # get predictions
            prediction_idx = np.argmax(norm_distance, axis=0)
            predictions = np.take(train_label, prediction_idx)
            acc = (predictions == test_label).mean()
            return acc
    else:
        # if not doing soft assignment, just compute the nearest neighbors
        subtract = gallery[:, None, :] - query
        distance = LA.norm(subtract, 2, axis=-1)
        idx = np.argpartition(distance, num_NN, axis=0)[:num_NN]
        nearest_samples = np.take(train_label, idx)
        out = mode(nearest_samples, axis=0, keepdims=False)[0]
        out = out.astype(int)
        test_label = np.array(test_label)
        acc = (out == test_label).mean()
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


def extract_feature(train_loader, val_loader, model, args):
    """
    args.{xent_weight, multi_layer_eval}
    """

    # We use the FC layer of the model only if we are combining XENT with NCA loss
    use_fc, cat = False, False
    print("\n>> Extracting statistics from training set embeddings")
    model.eval()
    with torch.no_grad():
        # get training mean
        out_mean, fc_out_mean = [], []
        for i, (inputs, _) in enumerate(train_loader):
            outputs, fc_outputs = model(inputs, use_fc=use_fc, cat=cat)
            outputs_np = outputs.cpu().data.numpy()
            out_mean.append(outputs_np)
            if fc_outputs is not None:
                fc_out_mean.append(fc_outputs.cpu().data.numpy())
        out_mean = np.concatenate(out_mean, axis=0).mean(0)

        fc_out_mean = np.concatenate(fc_out_mean, axis=0).mean(0) if len(fc_out_mean) > 0 else -1

        output_dict = collections.defaultdict(list)
        fc_output_dict = collections.defaultdict(list)
        for i, (inputs, labels) in enumerate(val_loader):
            # compute output
            outputs, fc_outputs = model(inputs, use_fc=use_fc, cat=args.multi_layer_eval)
            outputs = outputs.cpu().data.numpy()
            fc_outputs = fc_outputs.cpu().data.numpy() if fc_outputs is not None else [None] * outputs.shape[0]
            for out, fc_out, label in zip(outputs, fc_outputs, labels):  # noqa: B905
                output_dict[label.item()].append(out)
                fc_output_dict[label.item()].append(fc_out)
        all_info = [out_mean, fc_out_mean, output_dict, fc_output_dict]
        return all_info


def extract_and_evaluate_allshots(
    model,
    train_loader_for_avg,
    eval_loader,
    split,
    args,
    model_name=None,
):
    """
    This function evaluates the last model performing model after training it.
    from 1 to k shots, where k is specified by args.evaluate_all_shots. Prints
    the scores, and saves them under args.save_id
    args.{xent_weight, multi_layer_eval, test_iter, val_iter, evaluate_all_shots, out_dir}
    args.{num_NN, soft_assignment, median_prototype, test_way, num_classes, out_dir, test_query}
    """

    num_iter = args.test_iter if split == "test" else args.val_iter

    checkpoint1 = torch.load("{}_checkpoint.pth.tar".format(model_name))
    model.load_state_dict(checkpoint1["state_dict"])
    # compute training dataset statistics
    out_mean, fc_out_mean, out_dict, fc_out_dict = extract_feature(train_loader_for_avg, eval_loader, model, args)

    save_shot_npy = np.zeros(args.evaluate_all_shots)
    for k in range(1, args.evaluate_all_shots + 1):
        shot_info = tuple([100 * x for x in meta_evaluate(out_dict, out_mean, k, num_iter, args)])
        save_shot_npy[k - 1] = shot_info[0]

        print(
            ">>>\t ### {} set:\nfeature\tUN\tL2N\tCL2N\n{}\t{:2.2f}({:2.2f})\t{:2.2f}({:2.2f})\t{:2.2f}({:2.2f})".format(
                split, "GVP " + str(k) + " Shot", *shot_info
            )
        )

    print("\n >>> Array being saved:", save_shot_npy)
    np.save(os.path.join(args.output_dir, "allshots.npy"), save_shot_npy)


def extract_and_evaluate(
    model,
    train_loader_for_avg,
    eval_loader,
    split,
    args,
    model_name=None,
    expm_id=None,
    num_iter=None,
):
    """
    This function evaluates the best and last performing model after training it.
    arguments are the model, the tensorboard writer and the expm_id.
    Prints the 1-shot and 5-shot scores for best and last model.
    args.{xent_weight, multi_layer_eval, test_iter, val_iter, out_dir, out_dir}
    args.{num_NN, soft_assignment, median_prototype, test_way, num_classes, out_dir, test_query}
    """

    if not num_iter:
        num_iter = args.test_iter if split == "test" else args.val_iter

    if model_name:
        checkpoint1 = torch.load("{}/{}_best1.pth.tar".format(os.path.join(args.output_dir, "checkpoints"), model_name))
        model.load_state_dict(checkpoint1["state_dict"])
        out_mean1, fc_out_mean1, out_dict1, fc_out_dict1 = extract_feature(
            train_loader_for_avg, eval_loader, model, args
        )
        checkpoint5 = torch.load("{}/{}_best5.pth.tar".format(os.path.join(args.output_dir, "checkpoints"), model_name))
        model.load_state_dict(checkpoint5["state_dict"])
        out_mean5, fc_out_mean5, out_dict5, fc_out_dict5 = extract_feature(
            train_loader_for_avg, eval_loader, model, args
        )
    else:
        # compute training dataset statistics
        out_mean1, fc_out_mean1, out_dict1, fc_out_dict1 = extract_feature(
            train_loader_for_avg, eval_loader, model, args
        )
        # When model_name is not passed, we are using the current model checkpoint, which is the same for 1-shot and 5-shot
        out_mean5, fc_out_mean5, out_dict5, fc_out_dict55 = (
            out_mean1,
            fc_out_mean1,
            out_dict1,
            fc_out_dict1,
        )

    shot1_info = tuple([100 * x for x in meta_evaluate(out_dict1, out_mean1, 1, num_iter, args)])
    shot5_info = tuple([100 * x for x in meta_evaluate(out_dict5, out_mean5, 5, num_iter, args)])

    print(
        ">>>\t ### {} set:\nfeature\tCL2N\n{}\t{:2.2f}({:2.2f})\n{}\t{:2.2f}({:2.2f})".format(
            split, "GVP 1Shot", *shot1_info, "GVP_5Shot", *shot5_info
        )
    )

    # if writer:
    #     writer.add_scalar(split + "/1-shot/CL2N", shot1_info[0], t)
    #     writer.add_scalar(split + "/5-shot/CL2N", shot5_info[0], t)

    return shot1_info, shot5_info


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
