import time

import torch
import torchvision
from torchvision.transforms import InterpolationMode

from . import presets, sampler, utils


def load_data(traindir, valdir, testdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    # We need a default value for the variables below because args may come
    # from train_quantization.py which doesn't define them.
    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    ra_magnitude = getattr(args, "ra_magnitude", None)
    augmix_severity = getattr(args, "augmix_severity", None)
    dataset = torchvision.datasets.ImageFolder(
        traindir,
        presets.ClassificationPresetTrain(
            crop_size=train_crop_size,
            interpolation=interpolation,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
            ra_magnitude=ra_magnitude,
            augmix_severity=augmix_severity,
            backend=args.backend,
        ),
    )
    print("Took", time.time() - st)

    print("Loading validation/test data")
    preprocessing = presets.ClassificationPresetEval(
        crop_size=val_crop_size,
        resize_size=val_resize_size,
        interpolation=interpolation,
        backend=args.backend,
    )

    dataset_avg = torchvision.datasets.ImageFolder(
        traindir,
        preprocessing,
    )
    dataset_val = torchvision.datasets.ImageFolder(
        valdir,
        preprocessing,
    )
    dataset_test = torchvision.datasets.ImageFolder(
        testdir,
        preprocessing,
    )

    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = sampler.RASampler(
                dataset, shuffle=True, repetitions=args.ra_reps
            )
        elif hasattr(args, "class_aware_sampler") and args.class_aware_sampler:
            bcls, bnum = [int(x) for x in args.class_aware_sampler.split(",")]
            assert bcls * bnum == utils.get_world_size() * args.batch_size
            train_sampler = sampler.ClassAwareDistributedSampler(
                dataset, class_per_batch=bcls, sample_per_class=bnum
            )
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        train_avg_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_avg, shuffle=False
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=False
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test, shuffle=False
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        train_avg_sampler = torch.utils.data.SequentialSampler(dataset_avg)
        val_sampler = torch.utils.data.SequentialSampler(dataset_val)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return (
        dataset,
        dataset_avg,
        dataset_val,
        dataset_test,
        train_sampler,
        train_avg_sampler,
        val_sampler,
        test_sampler,
    )
