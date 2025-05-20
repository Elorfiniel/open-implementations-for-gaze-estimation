from mmengine.model import BaseModel

from opengaze.registry import MODELS, LOSSES

import torch as torch
import torch.nn as nn
import torchvision as tv


@MODELS.register_module()
class ITrackerPlus(BaseModel):
  '''Replace conv backbone in iTracker with MobileNet-v2.

  Reference:
    - repo: https://gitee.com/elorfiniel/gaze-point-estimation-2023
    - file: source/models/modules/itrackers.py

  Input:
    - face crop, shape: (B, 3, 224, 224)
    - reye crop, shape: (B, 3, 224, 224)
    - leye crop, shape: (B, 3, 224, 224)

  Output:
    - point of gaze (gx, gy), shape: (B, 2)
  '''

  def __init__(self, init_cfg=None, loss_cfg=dict(type='MSELoss')):
    super(ITrackerPlus, self).__init__(init_cfg=init_cfg)

    self.face_conv = tv.models.mobilenet_v2(
      weights=tv.models.MobileNet_V2_Weights.DEFAULT,
    )

    self.face_fc = nn.Sequential(
      nn.Linear(1000, 256),
      nn.ReLU(inplace=True),
      nn.Linear(256, 128),
      nn.ReLU(inplace=True),
    )

    self.eyes_conv = tv.models.mobilenet_v2(
      weights=tv.models.MobileNet_V2_Weights.DEFAULT,
    )

    self.eyes_fc = nn.Sequential(
      nn.Linear(2*1000, 128),
      nn.ReLU(inplace=True),
    )

    self.kpts_fc = nn.Sequential(
      nn.Linear(8, 64),
      nn.ReLU(inplace=True),
      nn.Linear(64, 128),
      nn.ReLU(inplace=True),
    )

    self.fc = nn.Sequential(
      nn.Linear(128+128+128, 128),
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

    feats_kpts = self.kpts_fc(data_dict['kpts'])

    feats = torch.cat([feats_face, feats_eyes, feats_kpts], dim=1)
    gazes = self.fc(feats)

    if mode == 'loss':
      loss = self.loss_fn(gazes, data_dict['gaze'])
      return dict(loss=loss)

    if mode == 'predict':
      return gazes, data_dict['gaze']

    return gazes
