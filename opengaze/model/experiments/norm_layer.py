from mmengine.model import BaseModule

from opengaze.registry import MODELS
from opengaze.model.wrapper import DataFnMixin

import torch as torch
import torch.nn as nn
import torchvision as tv


class LayerNorm2d(nn.GroupNorm):
  '''Mimic the interface of `nn.BatchNorm2d` for torchvision's ResNet.'''

  def __init__(self, num_features, **kwargs):
    super(LayerNorm2d, self).__init__(num_groups=1, num_channels=num_features, **kwargs)


@MODELS.register_module()
class ExpNormLayerResNet(DataFnMixin, BaseModule):
  '''
  Description:
    ResNet50 with normalization layers replaced by the specified type.

  Input:
    - normalized face patch, shape: (B, 3, 224, 224)

  Output:
    - gaze vector (pitch, yaw), shape: (B, 2)
  '''

  def __init__(self, norm_layer: str, init_cfg: dict = None):
    super(ExpNormLayerResNet, self).__init__(init_cfg=init_cfg)

    assert norm_layer in ['BatchNorm', 'LayerNorm']
    norm_layer = LayerNorm2d if norm_layer == 'LayerNorm' else nn.BatchNorm2d

    backbone = tv.models.resnet50(norm_layer=norm_layer, num_classes=2)
    self.conv = nn.Sequential(*[
      module
      for name, module in backbone.named_children()
      if not name in ['fc']
    ])
    self.fc = backbone.fc

  def data_fn(self, data_dict: dict):
    return dict(face=data_dict['face'])

  def forward(self, face: torch.Tensor):
    feat = self.conv(face).flatten(start_dim=1)
    gaze = self.fc(feat)

    return gaze
