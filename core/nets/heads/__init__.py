'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from .spe_pose      import SPEPoseHead
from .heatmap       import HeatmapHead
from .segmentation  import SegmentationHead

__all__ = [
    'SPEPoseHead', 'HeatmapHead', 'SegmentationHead'
]