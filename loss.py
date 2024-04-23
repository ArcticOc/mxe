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
        # total_logit = torch.logsumexp(torch.cat((pos_logit, neg_logit), dim=1), dim=1, keepdim=True)
        temp = torch.cat([pos_logit, neg_logit], 1)
        return self.xe(torch.cat([pos_logit, neg_logit], 1), yq_new)
        # return -1 * torch.sum(pos_logit - total_logit).mean()


# FIXME: There is no shot setting in this loss function
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
            ).sum(-1)
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

        denominator = torch.logsumexp(logit, dim=-1)
        temp = logit * class_mask
        numerator = torch.sum(temp, dim=-1) / class_mask.sum(-1)
        loss = denominator - numerator
        """ loss = torch.logsumexp(logit, dim=-1) - torch.sum(
            logit * class_mask, dim=-1
        ) / class_mask.sum(-1) """
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


# Ablation study
class PNCALoss(nn.Module):
    def __init__(self, T=1.0, logit_func="l2_dist", **kwargs):
        super(PNCALoss, self).__init__()
        self.T = T
        self.logit_func = logit_funcs[logit_func]
        self.xe = nn.CrossEntropyLoss()

    def forward(self, xq, yq, xs, ys, pos):
        with torch.no_grad():
            classes = ys.unique()
            one_hot = ys.view(-1, 1) == classes
            class_count = one_hot.sum(0, keepdim=True)
            yq_new = one_hot[pos].nonzero()[:, 1]

        mus = torch.mm(one_hot.t().float(), xs)
        M = mus[yq_new] - xq
        C = class_count[0, yq_new] - 1
        proto = M / C.unsqueeze(-1).clamp(min=0.1)
        # PNCALoss logic for numerator
        if self.logit_func == l2_dist:
            logit_numerator = -torch.cdist(xq, proto, p=2).diag()

        elif self.logit_func == l1_dist:
            logit_numerator = -torch.cdist(xq, proto, p=1).diag()

        pos_logit = logit_numerator.unsqueeze(-1)
        # FewShotNCALoss logic for denominator
        logit = self.logit_func(xq, xs) / self.T
        neg_logit = torch.logsumexp(logit, 1, keepdim=True)

        loss = neg_logit - pos_logit

        return loss.mean()


class NCAPLoss(nn.Module):
    def __init__(self, T=1.0, logit_func="l2_dist", **kwargs):
        super(NCAPLoss, self).__init__()
        self.T = T
        self.logit_func = logit_funcs[logit_func]
        self.xe = nn.CrossEntropyLoss()

    def forward(self, xq, yq, xs, ys, pos):
        with torch.no_grad():
            classes = ys.unique()
            one_hot = ys.view(-1, 1) == classes
            class_count = one_hot.sum(0, keepdim=True)
            yq_new = one_hot[pos].nonzero()[:, 1]
            class_mask = yq.view(-1, 1) == ys.view(1, -1)

        mus = torch.mm(one_hot.t().float(), xs)
        M = mus[yq_new] - xq
        C = class_count[0, yq_new] - 1
        proto = M / C.unsqueeze(-1).clamp(min=0.1)
        proto_n = mus / class_count.t().clamp(min=0.1)
        # PNCALoss logic for denominator
        if self.logit_func == l2_dist:
            logit_denominator_pos = -torch.cdist(xq, proto, p=2).diag()
            logit_denominator = -torch.cdist(xq, proto_n, p=2)
            logit_denominator[torch.arange(len(yq_new)), yq_new] = (
                logit_denominator_pos  # nq x class
            )

        elif self.logit_func == l1_dist:
            logit_denominator = -torch.cdist(xq, proto, p=1)

        neg_logit = torch.logsumexp(logit_denominator, 1, keepdim=True)

        # FewShotNCALoss logic for numerator
        logit = self.logit_func(xq, xs) / self.T
        pos_logit = torch.logsumexp(masked_logit(logit, class_mask), 1, keepdim=True)

        loss = neg_logit - pos_logit

        return loss.mean()


class MXEPLoss(nn.Module):
    def __init__(self, T=1.0, logit_func="l2_dist", **kwargs):
        super(MXEPLoss, self).__init__()
        self.T = T
        self.logit_func = logit_funcs[logit_func]
        self.xe = nn.CrossEntropyLoss()

    def forward(self, xq, yq, xs, ys, pos):
        with torch.no_grad():
            classes = ys.unique()
            one_hot = ys.view(-1, 1) == classes
            class_count = one_hot.sum(0, keepdim=True)
            yq_new = one_hot[pos].nonzero()[:, 1]
            class_mask = yq.view(-1, 1) == ys.view(1, -1)
            idx = (class_mask.sum(-1) > 1).cpu()  # multiple labels
            pos = torch.tensor(pos)
            ind = torch.arange(len(pos))
            class_mask[ind[idx], pos[idx]] = False

        mus = torch.mm(one_hot.t().float(), xs)
        M = mus[yq_new] - xq
        C = class_count[0, yq_new] - 1
        proto = M / C.unsqueeze(-1).clamp(min=0.1)
        proto_n = mus / class_count.t().clamp(min=0.1)
        # PNCALoss logic for denominator
        if self.logit_func == l2_dist:
            logit_denominator_pos = -torch.cdist(xq, proto, p=2).diag()
            logit_denominator = -torch.cdist(xq, proto_n, p=2)
            logit_denominator[torch.arange(len(yq_new)), yq_new] = (
                logit_denominator_pos  # nq x class
            )

        elif self.logit_func == l1_dist:
            logit_denominator = -torch.cdist(xq, proto, p=1)

        neg_logit = torch.logsumexp(logit_denominator, 1, keepdim=True)

        # FewShotNCALoss logic for numerator
        logit = self.logit_func(xq, xs) / self.T
        logit[ind, pos] *= 0
        logit[ind[idx], pos[idx]] -= INF
        pos_logit = torch.sum(logit * class_mask, dim=-1) / class_mask.sum(-1)

        loss = neg_logit - pos_logit.unsqueeze(-1)

        return loss.mean()


class PMXELoss(nn.Module):
    def __init__(self, T=1.0, logit_func="l2_dist", **kwargs):
        super(PMXELoss, self).__init__()
        self.T = T
        self.logit_func = logit_funcs[logit_func]
        self.xe = nn.CrossEntropyLoss()

    def forward(self, xq, yq, xs, ys, pos):
        with torch.no_grad():
            classes, counts = ys.unique(return_counts=True)
            one_hot = ys.view(-1, 1) == classes
            yq_new = one_hot[pos].nonzero()[:, 1]
            class_mask = yq.view(-1, 1) == ys.view(1, -1)
            idx = (class_mask.sum(-1) > 1).cpu()  # multiple labels
            pos = torch.tensor(pos)
            ind = torch.arange(len(pos))

        mus = torch.mm(one_hot.t().float(), xs)
        M = mus[yq_new] - xq
        C = counts.unsqueeze(1)[yq_new, 0] - 1
        proto = M / C.unsqueeze(-1).clamp(min=0.1)
        # PNCALoss logic for denominator
        if self.logit_func == l2_dist:
            logit_denominator = -torch.cdist(xq, proto, p=2).diag()

        elif self.logit_func == l1_dist:
            logit_denominator = -torch.cdist(xq, proto, p=1).diag()

        pos_logit = logit_denominator.unsqueeze(-1)

        # FewShotNCALoss logic for numerator
        logit = self.logit_func(xq, xs) / self.T
        logit[ind, pos] *= 0
        logit[ind[idx], pos[idx]] -= INF

        summed_distances = torch.zeros(
            xq.size(0),
            len(classes),
        ).cuda()

        for i, cls in enumerate(classes):
            class_mask_ = ys == cls

            summed_distances[:, i] = logit[:, class_mask_].sum(dim=1)

        normalized_logit = summed_distances / (counts - 1)

        neg_logit = torch.logsumexp(normalized_logit, 1, keepdim=True)

        loss = neg_logit - pos_logit

        return loss.mean()


class NCAMLoss(nn.Module):
    def __init__(self, T=1.0, logit_func="l2_dist", **kwargs):
        super(NCAMLoss, self).__init__()
        self.T = T
        self.logit_func = logit_funcs[logit_func]
        self.xe = nn.CrossEntropyLoss()

    def forward(self, xq, yq, xs, ys, pos):
        with torch.no_grad():
            classes, counts = ys.unique(return_counts=True)
            class_mask = yq.view(-1, 1) == ys.view(1, -1)
            idx = (class_mask.sum(-1) > 1).cpu()  # multiple labels
            pos = torch.tensor(pos)
            ind = torch.arange(len(pos))

        # FewShotNCALoss logic for numerator
        logit = self.logit_func(xq, xs) / self.T
        logit[ind, pos] *= 0
        logit[ind[idx], pos[idx]] -= INF

        pos_logit = torch.logsumexp(masked_logit(logit, class_mask), 1, keepdim=True)

        summed_distances = torch.zeros(
            xq.size(0),
            len(classes),
        ).cuda()

        for i, cls in enumerate(classes):
            class_mask_ = ys == cls

            summed_distances[:, i] = logit[:, class_mask_].sum(dim=1)

        normalized_logit = summed_distances / (counts - 1)

        neg_logit = torch.logsumexp(normalized_logit, 1, keepdim=True)

        loss = neg_logit - pos_logit

        return loss.mean()


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
