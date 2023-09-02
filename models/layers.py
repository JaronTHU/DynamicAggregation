#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================
@author: Jaron
@time: 2023/06/19 09:18:08
@email: fjjth98@163.com
@description: 
================================================
"""

import torch
import torch.nn as nn

from typing import List


class Transpose(nn.Module):

    def __init__(self, dim0, dim1, contiguous=False) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        self.contiguous = contiguous

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x.contiguous() if self.contiguous else x
  

def build_conv1d(in_dim: int, out_dim: int):
    return nn.Sequential(
        nn.Conv1d(in_dim, out_dim, 1),
        nn.BatchNorm1d(out_dim),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
    )


def build_linear(in_dim: int, out_dim: int):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim, 1),
        nn.BatchNorm1d(out_dim),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
    )


def build_mlp(dims: List[int], final_bn: bool = False, backend: str = 'conv1d'):
    mlp = []
    if backend == 'conv1d':
        for i in range(1, len(dims)):
            mlp += [*build_conv1d(dims[i-1], dims[i])]
    elif backend == 'linear':
        for i in range(1, len(dims)):
            mlp += [*build_linear(dims[i-1], dims[i])]
    else:
        raise NotImplementedError(f'Backend {backend} not supported!')
    mlp = mlp[:-1] if final_bn else mlp[:-2]
    return nn.Sequential(*mlp)


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature
