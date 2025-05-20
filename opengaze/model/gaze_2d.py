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
