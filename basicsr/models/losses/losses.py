import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from models.losses.loss_util import weighted_loss
from torchvision import models

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)


class VGG(torch.nn.Module):
    def __init__(self, requires_grad=False, rgb_range=1):
        super(VGG, self).__init__()

        ### use vgg19 weights to initialize
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = requires_grad
            for param in self.slice2.parameters():
                param.requires_grad = requires_grad
            for param in self.slice3.parameters():
                param.requires_grad = requires_grad

    def forward(self, x):
        x = self.slice1(x)
        x_lv1 = x
        x = self.slice2(x)
        x_lv2 = x
        x = self.slice3(x)
        x_lv3 = x
        result = [x_lv1, x_lv2, x_lv3]
        return result



class IntraCRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(IntraCRLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.vgg = VGG()
        self.l1 = nn.L1Loss()
        self.weights = [1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, pred, target, neg, weight=None, lamda1=1, lamda2=0.95, lamda3=0.90, **kwargs):
        neg1 = lamda1 * neg + (1 - lamda1) * target
        neg2 = lamda2 * neg + (1 - lamda2) * target
        neg3 = lamda3 * neg + (1 - lamda3) * target
        pred_vgg, pos_vgg, neg1_vgg, neg2_vgg, neg3_vgg = self.vgg(pred), self.vgg(target), self.vgg(neg1), self.vgg(
            neg2), self.vgg(neg3)
        loss = 0
        d_ap, d_an = 0, 0
        for i in range(len(pred_vgg)):
            d1 = self.l1(pred_vgg[i], pos_vgg[i].detach())
            d2 = self.l1(pred_vgg[i], neg1_vgg[i].detach())
            d3 = self.l1(pred_vgg[i], neg2_vgg[i].detach())
            d4 = self.l1(pred_vgg[i], neg3_vgg[i].detach())
            contrastive_l = d1 / (d2 + lamda2 * d3 + lamda3 * d4 + 1e-7)

            loss += self.weights[i] * contrastive_l
        return self.loss_weight * l1_loss(pred, target, weight,
                                          reduction=self.reduction) + self.loss_weight * loss * 0.05


