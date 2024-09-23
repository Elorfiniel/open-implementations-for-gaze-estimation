from mmengine.model import BaseModel

from template.registry import MODELS

import torch as torch
import torch.nn as nn


@MODELS.register_module()
class MPIIGaze_LeNet(BaseModel):
  '''
  Bibliography:
    Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling.
    "Appearance-Based Gaze Estimation in the Wild."
    In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 4511-20, 2015.
    https://doi.org/10.1109/CVPR.2015.7299081.

  ArXiv:
    https://arxiv.org/abs/1504.02863
  '''

  def __init__(self):
    super(MPIIGaze_LeNet, self).__init__()
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
    self.fc_2 = nn.Linear(in_features=502, out_features=2)

  def forward(self, ipts, tgts, mode='tensor'):
    feats = self.conv(ipts['eyes'])
    feats = self.fc_1(torch.flatten(feats, start_dim=1))

    gazes = self.fc_2(torch.cat([feats, ipts['pose']], dim=1))

    if mode == 'loss':
      loss = nn.functional.smooth_l1_loss(gazes, tgts['gaze'])
      return dict(loss=loss)

    if mode == 'predict':
      return gazes, tgts['gaze']

    return gazes
