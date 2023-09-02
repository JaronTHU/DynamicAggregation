#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================
@author: Jaron
@time: 2023/08/21 10:10:50
@email: fjjth98@163.com
@description: dynamic aggregation
this is the final version, including the following models (PoolPT)
DA-Naive: PoolPointTransformer
DA-Light: DMPPointTransformer and PoolPT
DA-Heavy: IDPT after the final transformer encoder layer
================================================
"""


import torch
import torch.nn as nn

from typing import List

from .build import MODELS
from .layers import build_mlp, get_graph_feature
from .PFT import HeadOnlyPointTransformer, HeadOnlyPointTransformerPartSeg


class NaiveDA(nn.Module):

    def __init__(self, in_dim: int, hidden_dims: List[int], pool_type: List[str] = ['mean', 'max']) -> None:
        super().__init__()
        self.mlp = build_mlp([in_dim] + hidden_dims, final_bn=False)
        self.pool_type = pool_type
        self.out_dim = hidden_dims[-1] * len(pool_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): (B, N, C1)

        Returns:
            torch.Tensor: (B, C2)
        """
        x, y = self.mlp(x.transpose(1, 2).contiguous()), []    # (B, C3, N)
        if 'mean' in self.pool_type:
            y.append(x.mean(dim=-1))
        if 'max' in self.pool_type:
            y.append(x.max(dim=-1)[0])
        return torch.cat(y, dim=-1)
    

class LightDA(nn.Module):

    def __init__(self, in_dim: int, hidden_dims_1: List[int], hidden_dims_2: List[int] = None, softmax: bool = True) -> None:
        super().__init__()
        self.mlp1 = build_mlp([in_dim] + hidden_dims_1, final_bn=True)
        if softmax:
            self.mlp1 = nn.Sequential(*self.mlp1, nn.Softmax(dim=-1))
        if hidden_dims_2 is None:
            hidden_dims_2 = hidden_dims_1
        self.mlp2 = build_mlp([in_dim] + hidden_dims_2, final_bn=True)
        self.out_dim = hidden_dims_1[-1] * hidden_dims_2[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): (B, N, C1)

        Returns:
            torch.Tensor: (B, C2)
        """
        x = x.transpose(1, 2).contiguous()
        x1, x2 = self.mlp1(x), self.mlp2(x)
        x = torch.bmm(x1, x2.transpose(1, 2)).flatten(1, 2)
        return x


class HeavyDA(nn.Module):

    def __init__(self, in_dim: int, out_dim: int = None, k: int = 20, num_layers: int = 3):
        super().__init__()
        self.k = k
        self.out_dim = in_dim if out_dim is None else out_dim
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_dim * 2, in_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ))
        self.final_conv = nn.Sequential(
            nn.Conv1d(in_dim * num_layers, self.out_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.out_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        x = [x.transpose(1, 2).contiguous()]
        for conv in self.convs:
            x.append(conv(get_graph_feature(x[-1], k=self.k)).max(dim=-1, keepdim=False)[0])
        x = self.final_conv(torch.cat(x[1:], dim=1)).max(dim=-1, keepdim=False)[0]
        return x


@MODELS.register_module()
class DAPointTransformer(HeadOnlyPointTransformer):
    """_summary_

    Args:
        HeadOnlyPointTransformer (_type_): _description_
    """
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        del self.pool_type
        if config.variant == 'naive':
            self.da = NaiveDA(in_dim=self.trans_dim, **config.da_config)
        elif config.variant == 'light':
            self.da = LightDA(in_dim=self.trans_dim, **config.da_config)
        elif config.variant == 'heavy':
            self.da = HeavyDA(in_dim=self.trans_dim, **config.da_config)
        else:
            raise NotImplementedError(f'{config.variant} not support for DAPointTransformer!')
        self.cls_head_finetune[0] = nn.Linear(self.trans_dim + self.da.out_dim, 256)

    def train(self, mode=True):
        # default only cls_head_finetune are effected by train(mode)
        self.training = mode
        self.da.train(mode)
        self.cls_head_finetune.train(mode)

    def pool_features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x[:, 0], self.da(x[:, 1:])], dim=1)


@MODELS.register_module()
class DAPointTransformerPartSeg(HeadOnlyPointTransformerPartSeg):
    """_summary_

    Args:
        HeadOnlyPointTransformer (_type_): _description_
    """
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        del self.pool_type
        if config.variant == 'naive':
            self.da = NaiveDA(in_dim=self.trans_dim * len(config.fetch_idx), **config.da_config)
        elif config.variant == 'light':
            self.da = LightDA(in_dim=self.trans_dim * len(config.fetch_idx), **config.da_config)
        elif config.variant == 'heavy':
            self.da = HeavyDA(in_dim=self.trans_dim * len(config.fetch_idx), **config.da_config)
        else:
            raise NotImplementedError(f'{config.variant} not support for DAPointTransformer!')
        self.cls_head_finetune[0] = nn.Conv1d(self.da.out_dim + 1088, 512, 1)

    def train(self, mode=True):
        # default only cls_head_finetune are effected by train(mode)
        self.training = mode
        self.da.train(mode)
        self.fp.train(mode)
        self.label_conv.train(mode)
        self.cls_head_finetune.train(mode)

    def pool_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.da(x)
