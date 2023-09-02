#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================
@author: Jaron
@time: 2023/04/27 17:48:55
@email: fjjth98@163.com
@description: Inserting adpater modules between Transformers
================================================
"""

import torch
import torch.nn as nn

from .build import MODELS
from .PT import Block
from .layers import Transpose
from .PFT import HeadOnlyPointTransformer, HeadOnlyPointTransformerPartSeg


class AdapterBlock(Block):
    """

    References: https://github.com/KMnP/vpt/blob/main/src/models/vit_adapter/adapter_block.py

    Args:
        Block (_type_): _description_
    """

    def __init__(self, adapter_config, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0, attn_drop=0, drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer, norm_layer)
        self.adapter_type = adapter_config.type
        if self.adapter_type == 'Pfeiffer':
            self.adapter = self._make_adapter(adapter_config, dim)
        elif self.adapter_type == 'Houlsby':
            self.adapter = nn.ModuleList([
                self._make_adapter(adapter_config, dim),
                self._make_adapter(adapter_config, dim)
            ])
        else:
            raise NotImplementedError(f'{adapter_config.NAME} adapter not support!')
        
    def forward(self, x):
        if self.adapter_type == 'Pfeiffer':
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x1 = self.mlp(self.norm2(x))
            x = x + self.drop_path(x1 + self.adapter(x1))
        elif self.adapter_type == 'Houlsby':
            x1 = self.attn(self.norm1(x))
            x = x + self.drop_path(x1 + self.adapter[0](x1))
            x1 = self.mlp(self.norm2(x))
            x = x + self.drop_path(x1 + self.adapter[1](x1))
        return x

    @staticmethod
    def _make_adapter(config, dim):
        hidden_dim = int(config.ratio * dim)
        if config.NAME == 'mlp':
            adapter = nn.Sequential(
                Transpose(1, 2),
                nn.Conv1d(dim, hidden_dim, 1),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv1d(hidden_dim, dim, 1), 
                Transpose(1, 2)
            )
        else:
            raise NotImplementedError(f'{config.NAME} adapter not support!')
        return adapter


class AdapterTransformerEncoder(nn.Module):
    def __init__(self, adapter_config, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., prefix_adapter=True, num_adapters=1):
        super().__init__()

        adapter_blocks = [
            AdapterBlock(
                adapter_config=adapter_config, 
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            ) for i in range(num_adapters)
        ]
        blocks = [
            Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            ) for i in range(depth-num_adapters)
        ]

        self.blocks = nn.ModuleList(adapter_blocks + blocks) if prefix_adapter else nn.ModuleList(blocks + adapter_blocks)

    def forward(self, x, pos):
        for block in self.blocks:
            x = block(x + pos)
        return x
    

@MODELS.register_module()
class AdapterPointTransformer(HeadOnlyPointTransformer):
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # use AdapterTransformerEncoder instead
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = AdapterTransformerEncoder(
            adapter_config=config.adapter_config,
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
            prefix_adapter=config.prefix_adapter,
            num_adapters=config.num_adapters
        )
        # new blocks must re- no_grad and eval
        self.blocks.requires_grad_(False)
        self.blocks.eval()
        for block in self.blocks.blocks:
            if hasattr(block, 'adapter'):
                block.adapter.requires_grad_(True)

    def train(self, mode=True):
        # the classification head is effected
        self.training = mode
        self.cls_head_finetune.train(mode)
        # adapter does not use norm or dropout, but drop_path should be included or not?
        for block in self.blocks.blocks:
            block.drop_path.train(mode)

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
            if 'adapter' not in k:
                for i in self.fixed_names:
                    if k.startswith(i):
                        add_item = False
                        break
            if add_item:
                state_dict[k] = v
        return state_dict


class AdapterTransformerEncoderPartSeg(AdapterTransformerEncoder):
    def __init__(self, fetch_idx: list, adapter_config, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., prefix_adapter=True, num_adapters=1):
        super().__init__(adapter_config, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, prefix_adapter, num_adapters)
        self.fetch_idx = fetch_idx

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): (B, N, C)
            pos (torch.Tensor): (B, N, C)

        Returns:
            torch.Tensor: (B, N, k, C)
        """
        feature_list = []
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in self.fetch_idx:
                feature_list.append(x)
        return torch.stack(feature_list, dim=2)
    

@MODELS.register_module()
class AdapterPointTransformerPartSeg(HeadOnlyPointTransformerPartSeg):
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # use AdapterTransformerEncoder instead
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = AdapterTransformerEncoderPartSeg(
            fetch_idx=config.fetch_idx,
            adapter_config=config.adapter_config,
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
            prefix_adapter=config.prefix_adapter,
            num_adapters=config.num_adapters
        )
        self.blocks.requires_grad_(False)
        self.blocks.eval()
        for block in self.blocks.blocks:
            if hasattr(block, 'adapter'):
                block.adapter.requires_grad_(True)

    def train(self, mode=True):
        # the classification head is effected
        self.training = mode
        self.fp.train(mode)
        self.label_conv.train(mode)
        self.cls_head_finetune.train(mode)
        for block in self.blocks.blocks:
            block.drop_path.train(mode)

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
            if 'adapter' not in k:
                for i in self.fixed_names:
                    if k.startswith(i):
                        add_item = False
                        break
            if add_item:
                state_dict[k] = v
        return state_dict
