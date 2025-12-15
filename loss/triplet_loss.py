from turtle import pd
import torch
from torch import nn


def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    return dist.clamp(min=1e-12).sqrt()


def hard_example_mining(dist_mat, labels, return_inds=False):
    """Modified to handle imbalanced classes."""
    N = dist_mat.size(0)

    # 生成掩码并排除自身（i==j）
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t()) & ~torch.eye(N, dtype=torch.bool, device=labels.device)
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    dist_ap = torch.zeros(N, device=dist_mat.device)  # 存储每个样本的最难正样本距离
    dist_an = torch.zeros(N, device=dist_mat.device)  # 存储每个样本的最难负样本距离

    for i in range(N):
        # 处理正样本
        pos_mask = is_pos[i]
        if pos_mask.any():
            dist_ap[i] = dist_mat[i][pos_mask].max()  # 最远正样本
        else:
            dist_ap[i] = 0.0  # 无正样本时的默认值

        # 处理负样本
        neg_mask = is_neg[i]
        if neg_mask.any():
            dist_an[i] = dist_mat[i][neg_mask].min()  # 最近负样本
        else:
            dist_an[i] = 0.0  # 无负样本时的默认值

    return dist_ap, dist_an


class TripletLoss(object):
    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        if self.margin is not None:
            # 使用 MarginRankingLoss，传入三个参数
            y = torch.ones_like(dist_an)
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            # 使用 SoftMarginLoss，传入两个参数: (dist_an - dist_ap, target)
            y = torch.ones_like(dist_an)
            loss = self.ranking_loss(dist_an - dist_ap, y)

        return loss, dist_ap, dist_an