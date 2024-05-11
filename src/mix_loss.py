import torch
import torch.nn as nn


class MixLoss(nn.Module):
    def __init__(self, loss_a, loss_b):
        super(MixLoss, self).__init__()
        self.loss_a = loss_a
        self.loss_b = loss_b

    def forward(self, output, target, lamda=None):
        if lamda is None:
            lamda = torch.Tensor([1, 1])
        if self.loss_a == self.loss_b:
            loss = self.loss_a(output, target, lamda)

        else:
            loss = self.loss_a(output, target, lamda) + self.loss_b(
                output, target, lamda
            )

        return loss
