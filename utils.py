import os
import shutil
import numpy as np
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_FR(pred, labels, threshold=0.5):

    '''
    Class 0: the same indentity
    Class 1: Difference indentity
    (sim > threshold and same indentity ) or (sim < threshold and difference indentity)
    '''
    # print(pred, labels)
    condition = ((pred > threshold) & (labels == 0)) | ((pred <= threshold) & (labels == 1))
    accuracy = condition.sum().item() / len(labels)
    return accuracy

def get_predict(pred, threshold=0.5):
    if isinstance(pred, torch.Tensor):
        pred = pred.item()
    if (pred[0] > threshold):
        return 1
    return 0
