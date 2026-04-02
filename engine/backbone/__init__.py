"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from .common import (
    get_activation,
    FrozenBatchNorm2d,
    freeze_batch_norm2d,
)
from .presnet import PResNet
from .test_resnet import MResNet

from .timm_model import TimmModel
from .torchvision_model import TorchVisionModel

from .csp_resnet import CSPResNet
from .csp_darknet import CSPDarkNet, CSPPAN

from .hgnetv2 import HGNetv2  #这个是我改进的主干
# from .hgnetv2_org import HGNetv2  #如果使用跑baseline想用预训练权重，那么就启动这个命令，关闭上一行命令

# from .hgnetv2_EBlock import HGNetv2_EBlock