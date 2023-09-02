#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================
@author: Jaron
@time: 2023/09/01 09:04:47
@email: fjjth98@163.com
@description: 
================================================
"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from .build import MODELS
from .PT import TransformerEncoder
from .PFT import HeadOnlyPointTransformer, HeadOnlyPointTransformerPartSeg


class VPTTransformerEncoder(TransformerEncoder):
    def __init__(self, num_prompts: int, vpt_shallow: bool, prompt_dropout: float = 0., embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__(embed_dim, depth, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate)

        self.num_prompts = num_prompts
        self.prompt_depth = 1 if vpt_shallow else depth

        self.prompts = nn.Parameter(torch.empty(self.prompt_depth, num_prompts, embed_dim))
        self.pos_prompts = nn.Parameter(torch.empty(1, num_prompts, embed_dim))
        self.prompt_dropout = nn.Dropout(prompt_dropout)

        trunc_normal_(self.prompts, std=.02)
        trunc_normal_(self.pos_prompts, std=.02)

    def forward(self, x, pos):
        b = x.size(0)
        x = torch.cat((
            x[:, :1],
            self.prompt_dropout(self.prompts[0, None].expand(b, -1, -1)),
            x[:, 1:]
        ), dim=1)
        pos = torch.cat((
            pos[:, :1],
            self.prompt_dropout(self.pos_prompts.expand(b, -1, -1)),
            pos[:, 1:]
        ), dim=1)

        x = self.blocks[0](x + pos)
        for i in range(1, len(self.blocks)):
            if self.prompt_depth != 1:
                x[:, 1:self.num_prompts+1] = self.prompt_dropout(self.prompts[i, None].expand(b, -1, -1))
            x = self.blocks[i](x + pos)

        return x


# reference: https://github.com/KMnP/vpt/blob/main/src/models/vit_prompt/vit.py
@MODELS.register_module()
class VPTPointTransformer(HeadOnlyPointTransformer):

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = VPTTransformerEncoder(
            num_prompts=config.num_prompts,
            vpt_shallow=config.vpt_shallow,
            prompt_dropout=config.prompt_dropout,
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )
        self.blocks.requires_grad_(False)
        self.blocks.eval()
        self.blocks.prompts.requires_grad_(True)
        self.blocks.pos_prompts.requires_grad_(True)

    def train(self, mode=True):
        self.training = mode
        self.cls_head_finetune.train(mode)
        self.blocks.prompt_dropout.train(mode)

    def pool_features(self, x: torch.Tensor) -> torch.Tensor:
        """ get pooled features before classification

        Args:
            x (torch.Tensor): (B, N, C)

        Returns:
            torch.Tensor: (B, kC)
        """
        pooled = []
        if 'cls' in self.pool_type:
            pooled.append(x[:, 0])
        if 'mean' in self.pool_type:
            pooled.append(x[:, 1:].mean(dim=1))
        if 'max' in self.pool_type:
            pooled.append(x[:, 1:].max(dim=1)[0])
        if 'prompt_mean' in self.pool_type:
            pooled.append(x[:, 1:self.num_prompts+1].mean(dim=1))
        if 'prompt_max' in self.pool_type:
            pooled.append(x[:, 1:self.num_prompts+1].max(dim=1)[0])
        if 'patch_mean' in self.pool_type:
            pooled.append(x[:, self.num_prompts+1:].mean(dim=1))
        if 'patch_max' in self.pool_type:
            pooled.append(x[:, self.num_prompts+1:].max(dim=1)[0])
        return torch.cat(pooled, dim=1)
    
    def efficient_state_dict(self):
        """Only return changed parameters (leave the base model alone)
        """
        # only the following parameters are conserved 
        state_dict = dict()
        for k, v in self.state_dict().items():
            # support dp and ddp models
            if k.startswith('module.'):
                k = k[7:]
            add_item = True
            if 'prompts' not in k:
                for i in self.fixed_names:
                    if k.startswith(i):
                        add_item = False
                        break
            if add_item:
                state_dict[k] = v
        return state_dict


class VPTTransformerEncoderPartSeg(VPTTransformerEncoder):
    def __init__(self, fetch_idx: list, num_prompts: int, vpt_shallow: bool, prompt_dropout: float = 0., embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__(num_prompts, vpt_shallow, prompt_dropout, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate)
        self.fetch_idx = fetch_idx

    def forward(self, x, pos):
        b = x.size(0)
        x = torch.cat((
            self.prompt_dropout(self.prompts[0, None].expand(b, -1, -1)),
            x
        ), dim=1)
        pos = torch.cat((
            self.prompt_dropout(self.pos_prompts.expand(b, -1, -1)),
            pos
        ), dim=1)

        feature_list = []
        x = self.blocks[0](x + pos)
        if 0 in self.fetch_idx:
            feature_list.append(x)
        for i in range(1, len(self.blocks)):
            if self.prompt_depth != 1:
                x[:, :self.num_prompts] = self.prompt_dropout(self.prompts[i, None].expand(b, -1, -1))
            x = self.blocks[i](x + pos)
            if i in self.fetch_idx:
                feature_list.append(x)

        return torch.stack(feature_list, dim=2)



@MODELS.register_module()
class VPTPointTransformerPartSeg(HeadOnlyPointTransformerPartSeg):

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = VPTTransformerEncoderPartSeg(
            fetch_idx=config.fetch_idx,
            num_prompts=config.num_prompts,
            vpt_shallow=config.vpt_shallow,
            prompt_dropout=config.prompt_dropout,
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )
        self.blocks.requires_grad_(False)
        self.blocks.eval()
        self.blocks.prompts.requires_grad_(True)
        self.blocks.pos_prompts.requires_grad_(True)

    def train(self, mode=True):
        self.training = mode
        self.fp.train(mode)
        self.label_conv.train(mode)
        self.cls_head_finetune.train(mode)
        self.blocks.prompt_dropout.train(mode)

    def pool_features(self, x: torch.Tensor) -> torch.Tensor:
        """ get pooled features before classification

        Args:
            x (torch.Tensor): (B, N, C)

        Returns:
            torch.Tensor: (B, kC)
        """
        pooled = []
        if 'mean' in self.pool_type:
            pooled.append(x.mean(dim=1))
        if 'max' in self.pool_type:
            pooled.append(x.max(dim=1)[0])
        if 'prompt_mean' in self.pool_type:
            pooled.append(x[:, :self.num_prompts].mean(dim=1))
        if 'prompt_max' in self.pool_type:
            pooled.append(x[:, :self.num_prompts].max(dim=1)[0])
        if 'patch_mean' in self.pool_type:
            pooled.append(x[:, self.num_prompts:].mean(dim=1))
        if 'patch_max' in self.pool_type:
            pooled.append(x[:, self.num_prompts:].max(dim=1)[0])
        return torch.cat(pooled, dim=1)
    
    def efficient_state_dict(self):
        """Only return changed parameters (leave the base model alone)
        """
        # only the following parameters are conserved 
        state_dict = dict()
        for k, v in self.state_dict().items():
            # support dp and ddp models
            if k.startswith('module.'):
                k = k[7:]
            add_item = True
            if 'prompts' not in k:
                for i in self.fixed_names:
                    if k.startswith(i):
                        add_item = False
                        break
            if add_item:
                state_dict[k] = v
        return state_dict