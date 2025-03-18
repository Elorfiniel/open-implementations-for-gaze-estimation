from mmengine.model import BaseModel

from opengaze.registry import MODELS, LOSSES

import torch as torch
import torch.nn as nn
import torchvision as tv


class LayerNorm2d(nn.GroupNorm):
  '''Mimic the interface of `nn.BatchNorm2d` for torchvision's ResNet.'''

  def __init__(self, num_features, **kwargs):
    super(LayerNorm2d, self).__init__(num_groups=1, num_channels=num_features, **kwargs)


@MODELS.register_module()
class ExpNormLayerResNet(BaseModel):
  '''
  Description:
    ResNet50 with normalization layers replaced by the specified type.

  Input:
    - normalized face patch, shape: (B, 3, 224, 224)

  Output:
    - gaze vector (pitch, yaw), shape: (B, 2)
  '''

  def __init__(self, norm_layer: str, init_cfg=None, loss_cfg=dict(type='L1Loss')):
    super(ExpNormLayerResNet, self).__init__(init_cfg=init_cfg)

    self.loss_fn = LOSSES.build(loss_cfg)

    assert norm_layer in ['BatchNorm', 'LayerNorm']
    norm_layer = LayerNorm2d if norm_layer == 'LayerNorm' else nn.BatchNorm2d

    backbone = tv.models.resnet50(norm_layer=norm_layer, num_classes=2)
    self.conv = nn.Sequential(*[
      module
      for name, module in backbone.named_children()
      if not name in ['fc']
    ])
    self.fc = backbone.fc

  def forward(self, mode='tensor', **data_dict):
    feats = self.conv(data_dict['face'])
    feats = torch.flatten(feats, start_dim=1)
    feats = self.fc(feats)

    if mode == 'loss':
      loss = self.loss_fn(feats, data_dict['gaze'])
      return dict(loss=loss)

    if mode == 'predict':
      return feats, data_dict['gaze']

    return feats
