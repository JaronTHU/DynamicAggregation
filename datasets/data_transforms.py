#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================
@author: Jaron
@time: 2023/04/24 11:38:55
@email: fjjth98@163.com
@description: Replace numpy and for iteration
PointcloudRandomInputDropout abandon, because not all point-based models are PointNet

ScanObjectNN gravity axis is 1 (y) not 2 (z)
================================================
"""

import torch

from math import ceil
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation


class PointcloudResample(object):
    """
    np.random.choice pytorch version
    references: https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/19
    """
    def __init__(self, npoints) -> None:
        self.npoints = npoints
        self.npoints_all = ceil(npoints / 1024) * 1200

    def __call__(self, pc):
        b, n, _ = pc.size()
        if n > self.npoints:
            if n > self.npoints_all:
                idx = furthest_point_sample(pc, self.npoints_all)   # (B, npoints_all)
                idx = idx[:, torch.randperm(self.npoints_all, device=pc.device)[:self.npoints]]
            else:
                idx = torch.randperm(n, dtype=torch.int, device=pc.device)[:self.npoints].unsqueeze(0).repeat(b, 1)
            pc = gather_operation(pc.transpose(1, 2).contiguous(), idx).transpose(1, 2).contiguous()
        return pc


class PointcloudNaiveSample(object):
    def __init__(self, npoints) -> None:
        self.npoints = npoints

    def __call__(self, pc):
        # avoid inplace replacement
        return pc[:, :self.npoints].clone()


class PointcloudRotate(object):
    def __init__(self, rotate_range: int = torch.pi, rotate_axis: str = 'y') -> None:        
        assert rotate_axis in ['x', 'y', 'z'], 'Rotation axis must be one of [x,y,z]!'

        self.rotate_range = rotate_range
        self.rotate_axis = rotate_axis

    def __call__(self, pc):
        b = pc.size(0)
        angles = torch.empty(b, dtype=torch.float, device=pc.device).uniform_(-self.rotate_range, self.rotate_range)
        cosval = angles.cos()
        sinval = angles.sin()
        ones = torch.ones(b, dtype=torch.float, device=pc.device)
        zeros = torch.zeros(b, dtype=torch.float, device=pc.device)
        if self.rotate_axis == 'x':
            rotations = [ones, zeros, zeros, zeros, cosval, -sinval, zeros, sinval, cosval]
        elif self.rotate_axis == 'y':
            rotations = [cosval, zeros, sinval, zeros, ones, zeros, -sinval, zeros, cosval]
        else:
            rotations = [cosval, -sinval, zeros, sinval, cosval, zeros, zeros, zeros, ones]
        rotations = torch.stack(rotations, dim=1).view(b, 3, 3)
        return torch.bmm(pc, rotations)


class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        b = pc.size(0)
        scales = torch.empty(b, 1, 3, dtype=torch.float, device=pc.device).uniform_(self.scale_low, self.scale_high)
        translations = torch.empty(b, 1, 3, dtype=torch.float, device=pc.device).uniform_(-self.translate_range, self.translate_range)
        return pc * scales + translations


class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, pc):
        jitters = torch.empty_like(pc).normal_(0., self.std).clamp_(-self.clip, self.clip)
        return pc + jitters


class PointcloudScale(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        b = pc.size(0)
        scales = torch.empty(b, 1, 3, dtype=torch.float, device=pc.device).uniform_(self.scale_low, self.scale_high)
        return pc * scales


class PointcloudTranslate(object):
    def __init__(self, translate_range=0.2):
        self.translate_range = translate_range

    def __call__(self, pc):
        b = pc.size(0)
        translations = torch.empty(b, 1, 3, dtype=torch.float, device=pc.device).uniform_(-self.translate_range, self.translate_range)
        return pc + translations


class RandomHorizontalFlip(object):
    def __init__(self, upright_axis='y', aug_prob=0.95, flip_prob=0.5):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        # Use the rest of axes for flipping.
        self.upright_axis = ord(upright_axis) - ord('x')
        self.aug_prob = aug_prob
        self.flip_prob = flip_prob

    def __call__(self, pc):
        b = pc.size(0)
        pc = pc.clone()
        aug_mask = torch.rand(b, device=pc.device) < self.aug_prob
        for curr_ax in range(pc.size(2)):
            if curr_ax == self.upright_axis:
                continue
            flip_mask = (torch.rand(b, device=pc.device) < self.flip_prob) & aug_mask
            pc[flip_mask, :, curr_ax] = pc[flip_mask, :, curr_ax].max(dim=1, keepdim=True)[0] - pc[flip_mask, :, curr_ax]
        return pc
