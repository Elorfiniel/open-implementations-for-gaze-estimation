from mmengine.model import BaseModel

from opengaze.registry import MODELS, LOSSES

import torch as torch
import torch.nn as nn
import torchvision as tv


@MODELS.register_module()
class ITracker(BaseModel):
  '''
  Bibliography:
    Krafka, Kyle, Aditya Khosla, Petr Kellnhofer, Harini Kannan,
    Suchendra Bhandarkar, Wojciech Matusik, and Antonio Torralba.
    "Eye Tracking for Everyone."

  ArXiv:
    https://arxiv.org/abs/1606.05814

  Input:
    - face crop, shape: (B, 3, 224, 224)
    - reye crop, shape: (B, 3, 224, 224)
    - leye crop, shape: (B, 3, 224, 224)
    - face grid, shape: (B, 625)

  Output:
    - point of gaze (gx, gy), shape: (B, 2)
  '''

  def __init__(self, init_cfg=None, loss_cfg=dict(type='MSELoss')):
    super(ITracker, self).__init__(init_cfg=init_cfg)

    self.face_conv = nn.Sequential(
      nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
      nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
      nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
    )

    self.face_fc = nn.Sequential(
      nn.Linear(64*12*12, 128),
      nn.ReLU(inplace=True),
      nn.Linear(128, 64),
      nn.ReLU(inplace=True),
    )

    self.eyes_conv = nn.Sequential(
      nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
      nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
      nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
    )

    self.eyes_fc = nn.Sequential(
      nn.Linear(2*64*12*12, 128),
      nn.ReLU(inplace=True),
    )

    self.grid_fc = nn.Sequential(
      nn.Linear(25*25, 256),
      nn.ReLU(inplace=True),
      nn.Linear(256, 128),
      nn.ReLU(inplace=True),
    )

    self.fc = nn.Sequential(
      nn.Linear(64+128+128, 128),
      nn.ReLU(inplace=True),
      nn.Linear(128, 2),
    )

    self.loss_fn = LOSSES.build(loss_cfg)

  def forward(self, mode='tensor', **data_dict):
    feats_face = self.face_conv(data_dict['face'])
    feats_face = torch.flatten(feats_face, start_dim=1)
    feats_face = self.face_fc(feats_face)

    feats_reye = self.eyes_conv(data_dict['reye'])
    feats_reye = torch.flatten(feats_reye, start_dim=1)
    feats_leye = self.eyes_conv(data_dict['leye'])
    feats_leye = torch.flatten(feats_leye, start_dim=1)
    feats_eyes = torch.cat([feats_reye, feats_leye], dim=1)
    feats_eyes = self.eyes_fc(feats_eyes)

    feats_grid = self.grid_fc(data_dict['grid'])

    feats = torch.cat([feats_face, feats_eyes, feats_grid], dim=1)
    gazes = self.fc(feats)

    if mode == 'loss':
      loss = self.loss_fn(gazes, data_dict['gaze'])
      return dict(loss=loss)

    if mode == 'predict':
      return gazes, data_dict['gaze']

    return gazes


class _AffNetSELayer(nn.Module):
  def __init__(self, n_channels: int, reduction: int):
    super(_AffNetSELayer, self).__init__()

    self.gap = nn.AdaptiveAvgPool2d(1)
    self.se = nn.Sequential(
      nn.Linear(n_channels, n_channels // reduction),
      nn.ReLU(inplace=True),
      nn.Linear(n_channels // reduction, n_channels),
      nn.Sigmoid(),
    )

  def forward(self, feats: torch.Tensor):
    n, c, h, w = feats.size()

    squeeze = self.gap(feats).view(n, c)
    excite = self.se(squeeze).view(n, c, 1, 1)
    se_feats = torch.mul(feats, excite)

    return se_feats

class _AffNetAdaGN(nn.Module):
  def __init__(self, in_features: int, n_groups: int, n_channels: int):
    super(_AffNetAdaGN, self).__init__()

    self.n_groups = n_groups
    self.fc = nn.Sequential(
      nn.Linear(in_features, 2*n_channels),
      nn.LeakyReLU(inplace=True),
    )
    self.gn = nn.GroupNorm(n_groups, n_channels)

  def forward(self, feats: torch.Tensor, factor: torch.Tensor):
    n, c, h, w = feats.size()

    style = self.fc(factor).view(n, 2, c, 1, 1)

    feats = self.gn(feats)
    trans = style[:, 0, :] * feats + style[:, 1, :]
    feats = feats + trans

    return feats

class _AffNetFace(nn.Module):
  def __init__(self):
    super(_AffNetFace, self).__init__()

    self.conv = nn.Sequential(
      nn.Conv2d(3, 48, kernel_size=5, stride=2, padding=0),
      nn.GroupNorm(6, 48),
      nn.ReLU(inplace=True),
      nn.Conv2d(48, 96, kernel_size=5, stride=1, padding=0),
      nn.GroupNorm(12, 96),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(96, 128, kernel_size=5, stride=1, padding=2),
      nn.GroupNorm(16, 128),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
      nn.GroupNorm(16, 192),
      nn.ReLU(inplace=True),
      _AffNetSELayer(192, 16),
      nn.Conv2d(192, 128, kernel_size=3, stride=2, padding=0),
      nn.GroupNorm(16, 128),
      nn.ReLU(inplace=True),
      _AffNetSELayer(128, 16),
      nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=0),
      nn.GroupNorm(8, 64),
      nn.ReLU(inplace=True),
      _AffNetSELayer(64, 16),
    )
    self.fc = nn.Sequential(
      nn.Linear(64*5*5, 128),
      nn.LeakyReLU(inplace=True),
      nn.Linear(128, 64),
      nn.LeakyReLU(inplace=True),
    )

  def forward(self, feats: torch.Tensor):
    n, c, h, w = feats.size()

    feats = self.conv(feats).view(n, -1)
    feats = self.fc(feats)

    return feats

class _AffNetEyes(nn.Module):
  def __init__(self):
    super(_AffNetEyes, self).__init__()

    self.feats_1_1 = nn.Sequential(
      nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0),
      nn.GroupNorm(3, 24),
      nn.ReLU(inplace=True),
      nn.Conv2d(24, 48, kernel_size=5, stride=1, padding=0),
    )
    self.feats_1_2 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      _AffNetSELayer(48, 16),
      nn.Conv2d(48, 64, kernel_size=5, stride=1, padding=1),
    )
    self.feats_1_3 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
    )

    self.feats_2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.feats_2_2 = nn.Sequential(
      nn.ReLU(inplace=True),
      _AffNetSELayer(128, 16),
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
    )
    self.feats_2_3 = nn.ReLU(inplace=True)

    self.adagn_1_1 = _AffNetAdaGN(128, 8, 48)
    self.adagn_1_2 = _AffNetAdaGN(128, 8, 64)
    self.adagn_2_1 = _AffNetAdaGN(128, 8, 128)
    self.adagn_2_2 = _AffNetAdaGN(128, 8, 64)

  def forward(self, feats: torch.Tensor, factor: torch.Tensor):
    feats = self.adagn_1_1(self.feats_1_1(feats), factor)
    feats = self.adagn_1_2(self.feats_1_2(feats), factor)
    feats_1 = feats = self.feats_1_3(feats)

    feats = self.adagn_2_1(self.feats_2_1(feats), factor)
    feats = self.adagn_2_2(self.feats_2_2(feats), factor)
    feats_2 = feats = self.feats_2_3(feats)

    feats = torch.cat([feats_1, feats_2], dim=1)

    return feats

@MODELS.register_module()
class AffNet(BaseModel):
  '''
  Bibliography:
    Bao, Yiwei, Yihua Cheng, Yunfei Liu, and Feng Lu.
    "Adaptive Feature Fusion Network for Gaze Tracking in Mobile Tablets."

  ArXiv:
    https://arxiv.org/abs/2103.11119

  Input:
    - face crop, shape: (B, 3, 224, 224)
    - reye crop, shape: (B, 3, 112, 112), flip: true
    - leye crop, shape: (B, 3, 112, 112), flip: none
    - crop rect, shape: (B, 12)

  Output:
    - point of gaze (gx, gy), shape: (B, 2)
  '''

  def __init__(self, init_cfg=None, loss_cfg=dict(type='SmoothL1Loss')):
    super(AffNet, self).__init__(init_cfg=init_cfg)

    self.face = _AffNetFace()

    self.rect = nn.Sequential(
      nn.Linear(12, 64),
      nn.LeakyReLU(inplace=True),
      nn.Linear(64, 96),
      nn.LeakyReLU(inplace=True),
      nn.Linear(96, 128),
      nn.LeakyReLU(inplace=True),
      nn.Linear(128, 64),
      nn.LeakyReLU(inplace=True),
    )

    self.eyes = _AffNetEyes()

    self.eyes_m_1 = nn.Sequential(
      _AffNetSELayer(256, 16),
      nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1),
    )
    self.eyes_m_2 = nn.Sequential(
      nn.ReLU(inplace=True),
      _AffNetSELayer(64, 16),
    )
    self.eyes_ada = _AffNetAdaGN(128, 8, 64)

    self.eyes_fc = nn.Sequential(
      nn.Linear(64*5*5, 128),
      nn.LeakyReLU(inplace=True),
    )

    self.fc = nn.Sequential(
      nn.Linear(64+64+128, 128),
      nn.LeakyReLU(inplace=True),
      nn.Linear(128, 2),
    )

    self.loss_fn = LOSSES.build(loss_cfg)

  def forward(self, mode='tensor', **data_dict):
    feats_face = self.face(data_dict['face'])
    feats_rect = self.rect(data_dict['rect'])

    factor = torch.cat([feats_face, feats_rect], dim=1)

    feats_reye = self.eyes(data_dict['reye'], factor)
    feats_leye = self.eyes(data_dict['leye'], factor)

    feats_eyes = torch.cat([feats_reye, feats_leye], dim=1)
    feats_eyes = self.eyes_m_1(feats_eyes)
    feats_eyes = self.eyes_ada(feats_eyes, factor)
    feats_eyes = self.eyes_m_2(feats_eyes)
    feats_eyes = torch.flatten(feats_eyes, start_dim=1)
    feats_eyes = self.eyes_fc(feats_eyes)

    feats = torch.cat([feats_face, feats_eyes, feats_rect], dim=1)
    gazes = self.fc(feats)

    if mode == 'loss':
      loss = self.loss_fn(gazes, data_dict['gaze'])
      return dict(loss=loss)

    if mode == 'predict':
      return gazes, data_dict['gaze']

    return gazes
