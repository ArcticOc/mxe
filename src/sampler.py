"""
This code is based on
https://github.com/pytorch/vision/tree/main/references/classification

modified by Takumi Kobayashi
"""

import math

import torch
import torch.distributed as dist


class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU).
    Heavily based on 'torch.utils.data.DistributedSampler'.

    This is borrowed from the DeiT Repo:
    https://github.com/facebookresearch/deit/blob/main/samplers.py
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, repetitions=3):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * float(repetitions) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle
        self.seed = seed
        self.repetitions = repetitions

    def __iter__(self):
        if self.shuffle:
            # Deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(self.repetitions)]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


# NOTE: It was stated to be used in the original paper, whereas not in code.
class ClassAwareDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, class_per_batch, sample_per_class, **kwargs) -> None:
        super(ClassAwareDistributedSampler, self).__init__(dataset, **kwargs)

        self.shuffle = True
        self.y = torch.tensor([y[1] for y in dataset.samples])
        class_counts = [(self.y == c).sum().item() for c in self.y.unique()]
        max_samp_num = max(class_counts)
        num_classes = len(class_counts)
        num_samples = max_samp_num * num_classes

        if self.drop_last and num_samples % self.num_replicas != 0:
            self.num_samples = math.ceil((num_samples - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(num_samples / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.samples_in_batch = [class_per_batch, sample_per_class]
        print(len(self.dataset), self.total_size)  ##

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = self.class_aware_shuffle(g).tolist()  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def class_aware_shuffle(self, g):
        y, bc, bn = self.y, *self.samples_in_batch

        cls_to_ind = [torch.nonzero(y == c).squeeze() for c in y.unique()]
        max_samp_num = max([len(x) for x in cls_to_ind])
        cls_to_ind = torch.stack(
            [self.randshuffle(self.append(x, max_samp_num, g), g) for x in cls_to_ind]
        )  # [C x max(#sample)]

        D = self.split(cls_to_ind, bn)  # [C x bn] * r
        D = [self.randshuffle(x, g).flatten() for x in D]  # class order shuffle: [rand(C)*bn] * r

        batches = self.split(torch.cat(D), bc * bn)  # [bc*bc] * r
        batches = torch.cat([self.randshuffle(x, g) for x in batches])  # shuffle in mini-batch

        return batches

    def randshuffle(self, x, g):
        inds = torch.randperm(len(x), generator=g)
        return x[inds]

    def append(self, x, n, g):
        return torch.cat([x.repeat(n // len(x)), self.randshuffle(x, g)[: (n % len(x))]])

    def split(self, x, num, dim=-1):
        return torch.tensor_split(x, torch.arange(0, x.size(dim), num)[1:], dim)

    def histcount(self, x, C):
        return [(x == c).sum().item() for c in range(C)]
