#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================
@author: Jaron
@time: 2023/04/23 09:42:10
@email: fjjth98@163.com
@description: Partially Finetune (PFT) Models
================================================
"""

import torch

from .build import MODELS
from .PT import PointTransformer, PointTransformerPartSeg


@MODELS.register_module()
class HeadOnlyPointTransformer(PointTransformer):

    fixed_names = ['encoder', 'pos_embed', 'blocks', 'norm']

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        for i in self.fixed_names:
            eval(f'self.{i}.requires_grad_(False)')
            eval(f'self.{i}.eval()')

    def train(self, mode=True):
        # default only cls_head_finetune are effected by train(mode)
        self.training = mode
        self.cls_head_finetune.train(mode)

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        if 'backbone_path' in ckpt:
            super().load_model_from_ckpt(ckpt['backbone_path'])
        super().load_model_from_ckpt(bert_ckpt_path)

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
            for i in self.fixed_names:
                if k.startswith(i):
                    add_item = False
                    break
            if add_item:
                state_dict[k] = v
        return state_dict


@MODELS.register_module()
class HeadOnlyPointTransformerPartSeg(PointTransformerPartSeg):

    fixed_names = ['encoder', 'pos_embed', 'blocks', 'norm']

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        for i in self.fixed_names:
            eval(f'self.{i}.requires_grad_(False)')
            eval(f'self.{i}.eval()')

    def train(self, mode=True):
        # default only cls_head_finetune are effected by train(mode)
        self.training = mode
        self.fp.train(mode)
        self.label_conv.train(mode)
        self.cls_head_finetune.train(mode)

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        if 'backbone_path' in ckpt:
            super().load_model_from_ckpt(ckpt['backbone_path'])
        super().load_model_from_ckpt(bert_ckpt_path)

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
            for i in self.fixed_names:
                if k.startswith(i):
                    add_item = False
                    break
            if add_item:
                state_dict[k] = v
        return state_dict
 

@MODELS.register_module()
class BiasPointTransformer(HeadOnlyPointTransformer):

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # bias term are reserved
        for name, p in self.named_parameters():
            if self._trainable_bias(name):
                p.requires_grad_(True)

    @staticmethod
    def _trainable_bias(name):
        # filter all bias terms in the blocks
        return name.startswith('blocks.') and name.endswith('.bias')

    def efficient_state_dict(self):
        """Only return changed parameters (leave the base model alone)
        """
        state_dict = dict()
        for k, v in self.state_dict().items():
            # support dp and ddp models
            if k.startswith('module.'):
                k = k[7:]
            add_item = True
            if not self._trainable_bias(k):
                for i in self.fixed_names:
                    if k.startswith(i):
                        add_item = False
                        break
            if add_item:
                state_dict[k] = v
        return state_dict


@MODELS.register_module()
class BiasPointTransformerPartSeg(HeadOnlyPointTransformerPartSeg):

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        for name, p in self.named_parameters():
            if self._trainable_bias(name):
                p.requires_grad_(True)

    @staticmethod
    def _trainable_bias(name):
        return name.startswith('blocks.') and name.endswith('.bias')

    def efficient_state_dict(self):
        """Only return changed parameters (leave the base model alone)
        """
        state_dict = dict()
        for k, v in self.state_dict().items():
            # support dp and ddp models
            if k.startswith('module.'):
                k = k[7:]
            add_item = True
            if not self._trainable_bias(k):
                for i in self.fixed_names:
                    if k.startswith(i):
                        add_item = False
                        break
            if add_item:
                state_dict[k] = v
        return state_dict
