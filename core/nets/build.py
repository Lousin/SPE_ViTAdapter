'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import *
from .heads     import *
from core.utils.utils import num_total_parameters

logger = logging.getLogger(__name__)


def _build_backbone(cfg):
    backbone = RWKV(cfg.MODEL.BACKBONE.RWKV.encoder, cfg.MODEL.BACKBONE.RWKV.feature_layers, cfg.MODEL.BACKBONE.RWKV.feat_size)
    logger.info(f'   - Backbone: {cfg.MODEL.BACKBONE.NAME} (# param: {num_total_parameters(backbone):,d})')

    if cfg.MODEL.USE_GROUPNORM_BACKBONE:
        logger.info(f'   - GroupNorm built for backbone')
        logger.info(f'   - Pretrained model loaded from {cfg.MODEL.BACKBONE.PRETRAINED}')

    return backbone

def _build_heads(cfg):
    """ Build head networks based on the configuration. Individual head networks
        take input from the output of the backbone network.

    Args:
        cfg ([type]): Object containing configuration parameters

    Returns:
        heads: nn.ModuleList of head networks
    """
    heads   = []

    # Create heads
    for i, name in enumerate(cfg.MODEL.HEAD.NAMES):
        if name == 'heatmap':
            head = HeatmapHead(in_channels= cfg.MODEL.BACKBONE.RWKV.encoder.hidden_size,
                               depth = cfg.MODEL.HEAD.DEPTH,
                               num_keypoints = cfg.DATASET.NUM_KEYPOINTS,
                               use_group_norm = cfg.MODEL.USE_GROUPNORM_HEADS,
                               group_norm_size = cfg.MODEL.GROUPNORM_SIZE)

        elif name == 'efficientpose':
            head = SPEPoseHead(cfg,
                               in_channels=cfg.MODEL.BACKBONE.RWKV.encoder.hidden_size,
                               depth = cfg.MODEL.HEAD.DEPTH,
                               num_iter = cfg.MODEL.HEAD.NUM_ITER,
                               use_group_norm = cfg.MODEL.USE_GROUPNORM_HEADS,
                               group_norm_size = cfg.MODEL.GROUPNORM_SIZE)

        elif name == 'segmentation':
            head = SegmentationHead(in_channels=cfg.MODEL.BACKBONE.RWKV.encoder.hidden_size,
                                    depth = cfg.MODEL.HEAD.DEPTH,
                                    use_group_norm = cfg.MODEL.USE_GROUPNORM_HEADS,
                                    group_norm_size = cfg.MODEL.GROUPNORM_SIZE)

        else:
            logger.error(f'{name}-type head is not defined or imported')

        logger.info(f'   - Head #{i+1}: {name} (# param: {num_total_parameters(head):,d})')

        heads.append(head)

    if cfg.MODEL.USE_GROUPNORM_HEADS:
        logger.info(f'   - GroupNorm built for prediction heads')

    return nn.ModuleList(heads)

def _shannon_entropy(x):
    """ Shannon entropy of pixel-wise logits """
    b = torch.sigmoid(x) * F.logsigmoid(x)
    b = -1.0 * b.mean()
    return b

class SPNv2(nn.Module):
    ''' Generic ConvNet consisting of a backbone and (possibly multiple) heads
        for different tasks
    '''
    def __init__(self, cfg):
        super().__init__()
        logger.info('Creating SPNv2 ...')

        # Build backbone
        self.backbone = _build_backbone(cfg)

        # Build task-specific heads
        self.heads      = _build_heads(cfg)
        self.head_names = cfg.MODEL.HEAD.NAMES

        # Which heads to compute loss?
        self.loss_h_idx = [self.head_names.index(h) for h in cfg.MODEL.HEAD.LOSS_HEADS]

        # Loss factors
        self.loss_factors = cfg.MODEL.HEAD.LOSS_FACTORS

        # Which head for inference?
        self.test_h_idx = [self.head_names.index(h) for h in cfg.TEST.HEAD]

        # Entropy minimization for segmentation?
        self.min_entropy = cfg.ODR.MIN_ENTROPY

    def forward(self, x, is_train=False, gpu=torch.device('cpu'), **targets):

        # Backbone forward pass
        x = self.backbone(x.to(gpu, non_blocking=True))

        if is_train:
            # Training - prediction heads
            loss = 0
            losses = {}
            for i, head in enumerate(self.heads):
                # --- Supervised loss if specified --- #
                if i in self.loss_h_idx:
                    if self.head_names[i] == 'efficientpose':
                        head_targets = {
                            k: v.to(gpu, non_blocking=True) for k, v in targets.items() \
                                if k in ['boundingbox', 'rotationmatrix', 'translation']
                        }
                    elif self.head_names[i] == 'heatmap':
                        head_targets = {
                            k: v.to(gpu, non_blocking=True) for k, v in targets.items() \
                                if k in ['heatmap']
                        }
                    elif self.head_names[i] == 'segmentation':
                        head_targets = {
                            k: v.to(gpu, non_blocking=True) for k, v in targets.items() \
                                if k in ['mask']
                        }
                    else:
                        raise NotImplementedError(f'{self.head_names[i]} is not implemented')

                    # Through i-th head
                    loss_i, loss_items = head(x, **head_targets)

                    # Append individual loss
                    loss   = loss + self.loss_factors[i] * loss_i
                    losses = {**losses, **loss_items}

            # --- Unsupervised loss --- #
            # Min entropy via segmentation
            if self.min_entropy and 'segmentation' in self.head_names:
                i = self.head_names.index('segmentation')
                logit  = self.heads[i](x) # [B, 1, H, W]
                loss_i = _shannon_entropy(logit)
                loss_items = {'ent': loss_i.detach()}

                loss   = loss + 1.0 * loss_i
                losses = {**losses, **loss_items}

            return loss, losses
        else:
            out = []
            for i in self.test_h_idx:
                out.append(self.heads[i](x))
            return out

def _check_bn_exists(module, module_name):
    """ Check if BN layers exist in a module """
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.BatchNorm2d):
            warnings.warn(f'GroupNorm is activated for {module_name} but found a BatchNorm layer at {name}!')

def build_spnv2(cfg):
    net = SPNv2(cfg)

    # if using group_norm, make sure there's no BN layers
    if cfg.MODEL.USE_GROUPNORM_BACKBONE:
        _check_bn_exists(net.backbone, 'backbone')
    if cfg.MODEL.USE_GROUPNORM_HEADS:
        _check_bn_exists(net.heads, 'heads')

    if cfg.MODEL.PRETRAIN_FILE:
        load_dict = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location='cpu')
        net.load_state_dict(load_dict, strict=True)

        logger.info(f'   - Pretrained model loaded from {cfg.MODEL.PRETRAIN_FILE}')

    return net