# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn


def cosine_similarity(x, y, eps=1e-8):
    return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + eps)


def pearson_correlation(x, y, eps=1e-8):
    return cosine_similarity(x - x.mean(1).unsqueeze(1), y - y.mean(1).unsqueeze(1), eps)


def logit_relation(y_s, y_t):
    p = pearson_correlation(y_s, y_t).mean()
    loss = 1 - p
    return loss


def class_relation(y_s, y_t):
    return logit_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DIST(nn.Module):
    def __init__(self, beta=1., gamma=1., use_sigmoid=False, loss_weight=1.0):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight

    def forward(self, y_s, y_t):
        assert y_s.ndim in (2, 4)
        if y_s.ndim == 4:
            num_classes = y_s.shape[1]
            y_s = y_s.transpose(1, 3).reshape(-1, num_classes)
            y_t = y_t.transpose(1, 3).reshape(-1, num_classes)
        if self.use_sigmoid:
            y_s = y_s.sigmoid()
            y_t = y_t.sigmoid()
        else:
            y_s = y_s.softmax(dim=1)
            y_t = y_t.softmax(dim=1)
        l_loss = logit_relation(y_s, y_t)
        c_loss = class_relation(y_s, y_t)
        loss = self.beta * l_loss + self.gamma * c_loss
        return self.loss_weight * loss



