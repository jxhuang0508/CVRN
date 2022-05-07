import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn


def cross_entropy_2d(predict, target):
    """
    Args:
        predict:(n, c, h, w)
        target:(n, h, w)
    """
    assert not target.requires_grad
    assert predict.dim() == 4
    assert target.dim() == 3
    assert predict.size(0) == target.size(0), f"{predict.size(0)} vs {target.size(0)}"
    assert predict.size(2) == target.size(1), f"{predict.size(2)} vs {target.size(1)}"
    assert predict.size(3) == target.size(2), f"{predict.size(3)} vs {target.size(3)}"
    n, c, h, w = predict.size()
    target_mask = (target >= 0) * (target != 255)
    target = target[target_mask]
    if not target.data.dim():
        return Variable(torch.zeros(1))
    predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
    predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    loss = F.cross_entropy(predict, target, size_average=True)
    return loss

def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))

def entropy_loss_regularized(v, regularized_weight):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    entropy_map2 = torch.sum(torch.mul(v, torch.log2(v + 1e-30)), 1)
    entropy_map2 = entropy_map2 - ((torch.sum(entropy_map2) / (n * h * w)) * regularized_weight)
    return -torch.sum(entropy_map2) / (n * h * w * np.log2(c))

def entropy_loss_regularized_max(v, regularized_weight, gpu_id):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    entropy_map2 = torch.sum(torch.mul(v, torch.log2(v + 1e-30)), 1)
    entropy_map2 = torch.min((entropy_map2 - ((torch.sum(entropy_map2) / (n * h * w)) * regularized_weight)), torch.zeros(entropy_map2.size()).pin_memory().to(gpu_id))
    return -torch.sum(entropy_map2) / (n * h * w * np.log2(c))

def sigmoid_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x c x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30)) + torch.mul((1-v), torch.log2((1-v) + 1e-30))) / (n * h * w * c * np.log2(2))

def sigmoid_loss_regularized(v, regularized_weight):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x c x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    # calculate entropy
    entropy_map = torch.mul(v, torch.log2(v + 1e-30)) + torch.mul((1 - v), torch.log2((1 - v) + 1e-30))
    # regularized
    entropy_map = entropy_map - ((torch.sum(entropy_map) / (n * h * w * c)) * regularized_weight)
    return -torch.sum(entropy_map) / (n * h * w * c * np.log2(2))

def sigmoid_loss_regularized_max(v, regularized_weight, gpu_id):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x c x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    # calculate entropy
    entropy_map = torch.mul(v, torch.log2(v + 1e-30)) + torch.mul((1 - v), torch.log2((1 - v) + 1e-30))
    # regularized
    entropy_map = torch.min((entropy_map - ((torch.sum(entropy_map) / (n * h * w * c)) * regularized_weight)), torch.zeros(entropy_map.size()).pin_memory().to(gpu_id))

    return -torch.sum(entropy_map) / (n * h * w * c * np.log2(2))

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)

def sigmoid_2_entropy(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x c x h x w
        output: batch_size x 1 x h x w
    """
    return -(torch.mul(v, torch.log2(v + 1e-30)) + torch.mul((1-v), torch.log2((1-v) + 1e-30))) / np.log2(2)


def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)


