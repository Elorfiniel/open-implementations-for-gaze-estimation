from mmengine.model import BaseModule

from opengaze.registry import MODELS
from opengaze.model.wrapper import DataFnMixin

import torch as torch
import torch.nn as nn
import torchvision as tv


@MODELS.register_module()
class LeNet(DataFnMixin, BaseModule):
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

  def __init__(self, init_cfg: dict = None):
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

  def data_fn(self, data_dict: dict):
    return dict(eyes=data_dict['eyes'], pose=data_dict['pose'])

  def forward(self, eyes: torch.Tensor, pose: torch.Tensor):
    feat = self.conv(eyes).flatten(start_dim=1)
    feat = self.fc_1(feat)

    feat = torch.cat([feat, pose], dim=1)
    gaze = self.fc_2(feat)

    return gaze


@MODELS.register_module()
class GazeNet(DataFnMixin, BaseModule):
  '''
  Bibliography:
    Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling.
    "MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation."

  ArXiv:
    https://arxiv.org/abs/1711.09017

  Input:
    - normalized eye patch, shape: (B, 3, 36, 60)
    - normalized head pose, shape: (B, 2)

  Ouptput:
    - gaze vector (pitch, yaw), shape: (B, 2)
  '''

  def __init__(self, init_cfg: dict = None):
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

  def data_fn(self, data_dict: dict):
    return dict(eyes=data_dict['eyes'], pose=data_dict['pose'])

  def forward(self, eyes: torch.Tensor, pose: torch.Tensor):
    feat = self.conv(eyes).flatten(start_dim=1)
    feat = self.fc_1(feat)

    feat = torch.cat([feat, pose], dim=1)
    gaze = self.fc_2(feat)

    return gaze


@MODELS.register_module()
class DilatedNet(DataFnMixin, BaseModule):
  '''
  Bibliography:
    Chen, Zhaokang, and Bertram E. Shi.
    "Appearance-Based Gaze Estimation Using Dilated-Convolutions."

  ArXiv:
    https://doi.org/10.48550/arXiv.1903.07296

  Input:
    - normalized face patch, shape: (B, 3, 96, 96)
    - normalized reye patch, shape: (B, 3, 64, 96)
    - normalized leye patch, shape: (B, 3, 64, 96)

  Output:
    - gaze vector (pitch, yaw), shape: (B, 2)
  '''

  def __init__(self, p_dropout: float = 0.1, init_cfg: dict = None):
    super(DilatedNet, self).__init__(init_cfg=init_cfg)

    pretrained_vgg16_features = tv.models.vgg16(
      weights=tv.models.VGG16_Weights.DEFAULT,
    ).features
    self.face_conv_1 = nn.Sequential(*pretrained_vgg16_features[:16])
    self.face_conv_2 = nn.Sequential(
      nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1),
      nn.BatchNorm2d(num_features=64),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
      nn.BatchNorm2d(num_features=64),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
      nn.BatchNorm2d(num_features=64),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.face_fc = nn.Sequential(
      nn.Linear(in_features=64*4*4, out_features=256),
      nn.BatchNorm1d(num_features=256),
      nn.ReLU(inplace=True),
      nn.Dropout(p=p_dropout),
      nn.Linear(in_features=256, out_features=32),
      nn.BatchNorm1d(num_features=32),
      nn.ReLU(inplace=True),
      nn.Dropout(p=p_dropout),
    )

    pretrained_vgg16_features = tv.models.vgg16(
      weights=tv.models.VGG16_Weights.DEFAULT,
    ).features
    self.eyes_conv_1 = nn.Sequential(*pretrained_vgg16_features[:9])
    self.eyes_conv_2 = nn.Sequential(
      nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1),
      nn.BatchNorm2d(num_features=64),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, dilation=(2, 2)),
      nn.BatchNorm2d(num_features=64),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, dilation=(3, 3)),
      nn.BatchNorm2d(num_features=64),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, dilation=(4, 5)),
      nn.BatchNorm2d(num_features=128),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, dilation=(5, 11)),
      nn.BatchNorm2d(num_features=128),
      nn.ReLU(inplace=True),
    )
    self.eyes_fc = nn.Sequential(
      nn.Linear(in_features=128*4*6, out_features=256),
      nn.BatchNorm1d(num_features=256),
      nn.ReLU(inplace=True),
      nn.Dropout(p=p_dropout),
    )

    self.fc = nn.Sequential(
      nn.Linear(in_features=32+256+256, out_features=544),
      nn.BatchNorm1d(num_features=544),
      nn.ReLU(inplace=True),
      nn.Dropout(p=p_dropout),
      nn.Linear(in_features=544, out_features=256),
      nn.BatchNorm1d(num_features=256),
      nn.ReLU(inplace=True),
      nn.Dropout(p=p_dropout),
      nn.Linear(in_features=256, out_features=2),
    )

  def data_fn(self, data_dict: dict):
    return dict(face=data_dict['face'], reye=data_dict['reye'], leye=data_dict['leye'])

  def forward(self, face: torch.Tensor, reye: torch.Tensor, leye: torch.Tensor):
    feat_face = self.face_conv_2(self.face_conv_1(face))
    feat_face = torch.flatten(feat_face, start_dim=1)
    feat_face = self.face_fc(feat_face)

    feat_reye = self.eyes_conv_2(self.eyes_conv_1(reye))
    feat_reye = torch.flatten(feat_reye, start_dim=1)
    feat_reye = self.eyes_fc(feat_reye)
    feat_leye = self.eyes_conv_2(self.eyes_conv_1(leye))
    feat_leye = torch.flatten(feat_leye, start_dim=1)
    feat_leye = self.eyes_fc(feat_leye)

    feat = torch.cat([feat_face, feat_reye, feat_leye], dim=1)
    gaze = self.fc(feat)

    return gaze


@MODELS.register_module()
class FullFace(DataFnMixin, BaseModule):
  '''
  Bibliography:
    Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling.
    "It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation."

  Arxiv:
    https://arxiv.org/abs/1611.08860

  Input:
    - normalized face patch, shape: (B, 3, 224, 224)

  Output:
    - gaze vector, shape: (B, 2)
  '''

  def __init__(self, init_cfg: dict = None):
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

  def data_fn(self, data_dict: dict):
    return dict(face=data_dict['face'])

  def forward(self, face: torch.Tensor):
    feat = self.conv(face)
    feat = self.sw(feat) * feat
    feat = feat.flatten(start_dim=1)

    gaze = self.fc(feat)

    return gaze


@MODELS.register_module()
class XGaze224(DataFnMixin, BaseModule):
  '''
  Bibliography:
    Zhang, Xucong, Seonwook Park, Thabo Beeler, Derek Bradley, Siyu Tang, and Otmar Hilliges.
    "ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation."

  Arxiv:
    https://arxiv.org/abs/2007.15837

  Input:
    - normalized face patch, shape: (B, 3, 224, 224)

  Output:
    - gaze vector (pitch, yaw), shape: (B, 2)
  '''

  def __init__(self, init_cfg: dict = None):
    super(XGaze224, self).__init__(init_cfg=init_cfg)

    pretrained = tv.models.resnet50(weights=tv.models.ResNet50_Weights.DEFAULT)
    self.conv = nn.Sequential(*[
      module
      for name, module in pretrained.named_children()
      if not name in ['fc']
    ])

    self.fc = nn.Linear(in_features=2048, out_features=2)

  def data_fn(self, data_dict: dict):
    return dict(face=data_dict['face'])

  def forward(self, face: torch.Tensor):
    feat = self.conv(face).flatten(start_dim=1)
    gaze = self.fc(feat)

    return gaze
