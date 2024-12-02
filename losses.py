import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cross_entropy
from torch.nn.modules.loss import _WeightedLoss

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass
#尝试从LovaszSoftmax.pytorch.lovasz_losses 模块导入 lovasz_hinge函数。
#自己改的
__all__ = ['BCEDiceLoss', 'LovaszHingeLoss','SoftDiceLoss']

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        #使用 F.binary_cross_entropy_with_logits 函数计算二元交叉熵损失，
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice
#这个损失函数结合了两个组件：二元交叉熵（binary cross-entropy）和 Dice 损失。这个损失函数是混合的。
#计算平均 Dice 系数的相反数，然后将 BCE 损失和 Dice 损失相加，乘以一个系数 0.5，返回最终的损失值。

class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        num = target.size(0)

        probs = torch.sigmoid(input)
        m1 = probs.view(num, -1)
        m2 = target.view(num, -1)
        intersection = m1 * m2

        score = (
            2.0
            * (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) + self.smooth)
        )
        score = 1 - score.sum() / num
        return score
