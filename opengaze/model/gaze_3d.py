from mmengine.model import BaseModel

from opengaze.registry import MODELS, LOSSES

import torch as torch
import torch.nn as nn
import torchvision as tv


@MODELS.register_module()
class LeNet(BaseModel):
  '''
  Bibliography:
    Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling.
    "Appearance-Based Gaze Estimation in the Wild."

  ArXiv:
    https://arxiv.org/abs/1504.02863

  Input:
    - normalized eye patch, shape: (B, 1, 36, 60)
    - normalized head pose, shape: (B, 2)

  Output:
    - gaze vector (pitch, yaw), shape: (B, 2)
  '''

  def __init__(self, init_cfg=None, loss_cfg=dict(type='MSELoss')):
    super(LeNet, self).__init__(init_cfg=init_cfg)

    self.conv = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )

    self.fc_1 = nn.Sequential(
      nn.Linear(in_features=50*6*12, out_features=500),
      nn.ReLU(inplace=True),
    )

    self.fc_2 = nn.Linear(in_features=500+2, out_features=2)

    self.loss_fn = LOSSES.build(loss_cfg)

  def forward(self, mode='tensor', **data_dict):
    feats = self.conv(data_dict['eyes'])
    feats = self.fc_1(torch.flatten(feats, start_dim=1))

    gazes = self.fc_2(torch.cat([feats, data_dict['pose']], dim=1))

    if mode == 'loss':
      loss = self.loss_fn(gazes, data_dict['gaze'])
      return dict(loss=loss)

    if mode == 'predict':
      return gazes, data_dict['gaze']

    return gazes


@MODELS.register_module()
class GazeNet(BaseModel):
  '''
  Bibliography:
    Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling.
    “MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation.”

  ArXiv:
    https://arxiv.org/abs/1711.09017

  Input:
    - normalized eye patch, shape: (B, 3, 36, 60)
    - normalized head pose, shape: (B, 2)

  Ouptput:
    - gaze vector (pitch, yaw), shape: (B, 2)
  '''

  def __init__(self, init_cfg=None, loss_cfg=dict(type='MSELoss')):
    super(GazeNet, self).__init__(init_cfg=init_cfg)

    pretrained_vgg16 = tv.models.vgg16(
      weights=tv.models.VGG16_Weights.DEFAULT,
    )
    self.conv = pretrained_vgg16.features
    pooling_layers = [
      n for n, c in self.conv.named_children()
      if isinstance(c, nn.MaxPool2d)
    ]
    for layer in pooling_layers[:2]:
      setattr(self.conv, layer, nn.MaxPool2d(kernel_size=2, stride=1))

    self.fc_1 = nn.Sequential(
      nn.Linear(in_features=512*4*7, out_features=4096),
      nn.ReLU(inplace=True),
    )

    self.fc_2 = nn.Sequential(
      nn.Linear(in_features=4096+2, out_features=4096),
      nn.ReLU(inplace=True),
      nn.Linear(in_features=4096, out_features=2),
    )

    self.loss_fn = LOSSES.build(loss_cfg)

  def forward(self, mode='tensor', **data_dict):
    feats = self.conv(data_dict['eyes'])
    feats = self.fc_1(torch.flatten(feats, start_dim=1))

    gazes = self.fc_2(torch.cat([feats, data_dict['pose']], dim=1))

    if mode == 'loss':
      loss = self.loss_fn(gazes, data_dict['gaze'])
      return dict(loss=loss)

    if mode == 'predict':
      return gazes, data_dict['gaze']

    return gazes


@MODELS.register_module()
class FullFace(BaseModel):
  '''
  Bibliography:
    Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling.
    “It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation.”

  Arxiv:
    https://arxiv.org/abs/1611.08860

  Input:
    - normalized face patch, shape: (B, 3, 224, 224)

  Output:
    - gaze vector, shape: (B, 2)
  '''

  def __init__(self, init_cfg=None, loss_cfg=dict(type='L1Loss')):
    super(FullFace, self).__init__(init_cfg=init_cfg)

    pretrained_alexnet = tv.models.alexnet(
      weights=tv.models.AlexNet_Weights.DEFAULT,
    )
    self.conv = pretrained_alexnet.features

    self.sw = nn.Sequential(
      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1),
      nn.ReLU(inplace=True),
    )

    self.fc = nn.Sequential(
      nn.Linear(in_features=256*13*13, out_features=4096),
      nn.ReLU(inplace=True),
      nn.Linear(in_features=4096, out_features=4096),
      nn.ReLU(inplace=True),
      nn.Linear(in_features=4096, out_features=2),
    )

    self.loss_fn = LOSSES.build(loss_cfg)

  def forward(self, mode='tensor', **data_dict):
    feats = self.conv(data_dict['face'])
    feats = self.sw(feats) * feats

    gazes = self.fc(torch.flatten(feats, start_dim=1))

    if mode == 'loss':
      loss = self.loss_fn(gazes, data_dict['gaze'])
      return dict(loss=loss)

    if mode == 'predict':
      return gazes, data_dict['gaze']

    return gazes


@MODELS.register_module()
class XGaze224(BaseModel):
  '''
  Bibliography:
    Zhang, Xucong, Seonwook Park, Thabo Beeler, Derek Bradley, Siyu Tang, and Otmar Hilliges.
    “ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation.”

  Arxiv:
    https://arxiv.org/abs/2007.15837

  Input:
    - normalized face patch, shape: (B, 3, 224, 224)

  Output:
    - gaze vector (pitch, yaw), shape: (B, 2)
  '''

  def __init__(self, init_cfg=None, loss_cfg=dict(type='L1Loss')):
    super(XGaze224, self).__init__(init_cfg=init_cfg)

    self.loss_fn = LOSSES.build(loss_cfg)

    pretrained = tv.models.resnet50(weights=tv.models.ResNet50_Weights.DEFAULT)
    self.conv = nn.Sequential(*[
      module
      for name, module in pretrained.named_children()
      if not name in ['fc']
    ])

    self.fc = nn.Linear(in_features=2048, out_features=2)

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
