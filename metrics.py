import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.distance import cdist

from scipy.spatial.distance import pdist, squareform
import medpy.metric.binary as medpy_metric




def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)
    output1 = output_
    target1 = target_
    # acc = (output1 == target1).sum() / target1.size

    output2=output_
    target2=target_
    # 计算Recall指标
    true_positives = (output2 & target2).sum()
    false_negatives = (target2 & ~output2).sum()
    recall = true_positives / (true_positives + false_negatives)
    pre= true_positives / (true_positives + (output_ & ~target_).sum())





    return iou, dice ,pre ,recall


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
