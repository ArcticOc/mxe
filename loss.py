"""
Loss functions for few-shot learning
written by Takumi Kobayashi
"""

import torch
import torch.distributed.nn
import torch.nn as nn
from torch import distributed

import src.utils as utils

INF = 1000.0  # float('inf') # 1000000.0


def masked_logit(D, M):
    mask = torch.zeros_like(D)
    mask = mask.masked_fill(~M, -INF).masked_fill(M, float(0.0))
    return D + mask


# - logit based on pair-wise distance -#
def l2_dist(xq, xs):
    return -torch.pow(torch.cdist(xq, xs), 2).div(2)


def l1_dist(xq, xs):
    return -torch.cdist(xq, xs, p=1)


logit_funcs = {"l2_dist": l2_dist, "l1_dist": l1_dist}


# - Few-shot Loss -#
class FewShotNCALoss(nn.Module):
    def __init__(self, T=0.9, logit="l2_dist", **kwargs):
        super(FewShotNCALoss, self).__init__()

        # Logit function
        self.logit_func = logit_funcs[logit]

        self.xe = nn.CrossEntropyLoss()
        self.T = T

    def forward(self, xq, yq, xs, ys, pos):
        """
        Input args:
            xq - Feature vectors of query samples [num_q x dim]
            yq - Labels of query samples [num_q]
            xs - Feature vectors of support samples [num_s x dim]
            ys - Labels of support samples [num_s]
            pos - Index of query samples in [0,...,num_s-1]
        """
        with torch.no_grad():
            class_mask = yq.view(-1, 1) == ys.view(1, -1)  # nq x ns
            idx = (class_mask.sum(-1) > 1).cpu()  # multiple labels
            pos = torch.tensor(pos)
            ind = torch.arange(len(pos))
            yq_new = torch.zeros_like(yq)

        logit = self.logit_func(xq, xs) / self.T
        logit[ind, pos] *= 0
        logit[ind[idx], pos[idx]] -= INF
        pos_logit = torch.logsumexp(masked_logit(logit, class_mask), 1, keepdim=True)
        neg_logit = torch.logsumexp(masked_logit(logit, ~class_mask), 1, keepdim=True)

        return self.xe(torch.cat([pos_logit, neg_logit], 1), yq_new)


class PNLoss(nn.Module):
    def __init__(self, T=1.0, logit="l2_dist", **kwargs):
        super(PNLoss, self).__init__()

        # Logit function
        self.logit_func = logit_funcs[logit]

        self.xe = nn.CrossEntropyLoss()
        self.T = T

    def forward(self, xq, yq, xs, ys, pos):
        with torch.no_grad():
            classes = ys.unique()
            one_hot = ys.view(-1, 1) == classes  # ns x #class
            class_count = one_hot.sum(0, keepdim=True)  # 1 x #class
            yq_new = one_hot[pos].nonzero()[:, 1]

        mus = torch.mm(one_hot.t().float(), xs)  # #class x dim
        M = mus.unsqueeze(0).repeat(len(yq), 1, 1)  # nq x #class x dim
        M[torch.arange(len(yq)), yq_new] -= xq

        C = class_count.repeat(len(yq), 1)  # nq x #class
        C[torch.arange(len(yq)), yq_new] -= 1

        if self.logit_func == l2_dist:
            logit = -0.5 * (xq.unsqueeze(1) - M / C.unsqueeze(-1).clamp(min=0.1)).pow(
                2
            ).sum(-1)  # nq x #class
        elif self.logit_func == l1_dist:
            logit = (
                -(xq.unsqueeze(1) - M / C.unsqueeze(-1).clamp(min=0.1)).abs().sum(-1)
            )
        logit *= C > 0.1  # exclude empty class

        return self.xe(logit / self.T, yq_new)


# - Multi-label Loss -#
class MultiXELoss(nn.Module):
    def __init__(self, T=1.0, logit="l2_dist", **kwargs):
        super(MultiXELoss, self).__init__()

        # Logit function
        self.logit_func = logit_funcs[logit]

        self.T = T

    def forward(self, xq, yq, xs, ys, pos):
        # Class correspondence
        with torch.no_grad():
            class_mask = yq.view(-1, 1) == ys.view(1, -1)  # nq x ns
            idx = (class_mask.sum(-1) > 1).cpu()  # multiple labels
            pos = torch.tensor(pos)
            ind = torch.arange(len(pos))
            class_mask[ind[idx], pos[idx]] = False

        logit = self.logit_func(xq, xs) / self.T
        logit[ind, pos] *= 0
        logit[ind[idx], pos[idx]] -= INF

        loss = torch.logsumexp(logit, dim=-1) - torch.sum(
            logit * class_mask, dim=-1
        ) / class_mask.sum(-1)
        return loss.mean()


class BCELoss(nn.BCEWithLogitsLoss):
    def __init__(self, T=1.0, logit="l2_dist", **kwargs):
        super(BCELoss, self).__init__(reduction="sum")

        # Logit function
        self.logit_func = logit_funcs[logit]

        self.bias = nn.Parameter(torch.zeros(1))
        self.T = T

    def forward(self, xq, yq, xs, ys, pos):
        # Class correspondence
        with torch.no_grad():
            one_hot_yq = yq.view(-1, 1) == ys.view(1, -1)  # num_q x num_s
            pos = torch.tensor(pos)
            ind = torch.arange(len(pos))

        logit = self.logit_func(xq, xs) / self.T - self.bias
        logit[ind, pos] *= 0
        logit[ind, pos] += INF  # always correct

        loss = super(BCELoss, self).forward(logit, one_hot_yq.float())
        return loss / len(yq)


"""
Emanuel Ben-Baruch, Tal Ridnik, Nadav Zamir, Asaf Noy, Itamar Friedman, Matan Protter, and Lihi Zelnik-Manor.
Asymmetric loss for multi-label classification. In ICCV, pages 82-91, 2021
"""


class AsymmetricLoss(BCELoss):
    def __init__(
        self,
        T=1.0,
        logit="l2_dist",
        gamma_neg=4,
        gamma_pos=0,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
        **kwargs,
    ):
        super(AsymmetricLoss, self).__init__(T, logit)

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, xq, yq, xs, ys, pos):
        # Class correspondence
        with torch.no_grad():
            one_hot_yq = yq.view(-1, 1) == ys.view(1, -1)  # num_q x num_s
            pos = torch.tensor(pos)
            ind = torch.arange(len(pos))

        logit = self.logit_func(xq, xs) / self.T - self.bias
        logit[ind, pos] *= 0
        logit[ind, pos] += INF  # always correct

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(logit)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = one_hot_yq * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (~one_hot_yq) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * one_hot_yq
            pt1 = xs_neg * (~one_hot_yq)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * one_hot_yq + self.gamma_neg * (
                ~one_hot_yq
            )
            one_sided_w = torch.pow((1 - pt).clamp(min=0), one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum(dim=-1).mean()


# - Wrapper class for distributed training -#
class WrapperLoss(torch.nn.Module):
    def __init__(self, loss, class_proxy=None):
        super(WrapperLoss, self).__init__()
        if distributed.is_initialized():
            self.rank = distributed.get_rank()
        else:
            self.rank = 0

        self.loss = loss

        if class_proxy is not None:
            self.class_proxy = nn.Parameter(torch.empty(class_proxy))
            nn.init.kaiming_uniform_(self.class_proxy, a=2.23)  # \sqrt(5)
        else:
            self.class_proxy = None

    def forward(self, local_embeddings, local_labels):
        local_labels.squeeze_()
        local_labels = local_labels.long()

        batch_size = local_embeddings.size(0)

        embeddings = utils.gather_across_processes(local_embeddings)
        labels = utils.gather_across_processes(local_labels)

        if self.class_proxy is not None:
            embeddings = torch.cat([embeddings, self.class_proxy])
            class_labels = torch.arange(
                self.class_proxy.size(0), dtype=labels.dtype, device=labels.device
            )
            labels = torch.cat([labels, class_labels])

        pos = torch.arange(
            batch_size * self.rank, batch_size * (self.rank + 1)
        ).tolist()

        return self.loss(local_embeddings, local_labels, embeddings, labels, pos)
