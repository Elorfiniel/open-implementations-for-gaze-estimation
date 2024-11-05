from mmengine.model import BaseModel

from template.registry import MODELS, LOSSES

import torch as torch
import torch.nn as nn
import torchvision as tv


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

  def __init__(self, init_cfg=None, loss_cfg=dict(type='MSELoss')):
    super(MPIIGaze_LeNet, self).__init__(init_cfg=init_cfg)

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

  def forward(self, ipts, tgts, mode='tensor'):
    feats = self.conv(ipts['eyes'])
    feats = self.fc_1(torch.flatten(feats, start_dim=1))

    gazes = self.fc_2(torch.cat([feats, ipts['pose']], dim=1))

    if mode == 'loss':
      loss = self.loss_fn(gazes, tgts['gaze'])
      return dict(loss=loss)

    if mode == 'predict':
      return gazes, tgts['gaze']

    return gazes


@MODELS.register_module()
class MPIIGaze_GazeNet(BaseModel):
  '''
  Bibliography:
    Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling.
    “MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation.”
    IEEE Transactions on Pattern Analysis and Machine Intelligence 41, no. 1 (January 2019): 162-75.
    https://doi.org/10.1109/TPAMI.2017.2778103.

  ArXiv:
    https://arxiv.org/abs/1711.09017
  '''

  def __init__(self, init_cfg=None, loss_cfg=dict(type='MSELoss')):
    super(MPIIGaze_GazeNet, self).__init__(init_cfg=init_cfg)

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

  def forward(self, ipts, tgts, mode='tensor'):
    feats = self.conv(ipts['eyes'])
    feats = self.fc_1(torch.flatten(feats, start_dim=1))

    gazes = self.fc_2(torch.cat([feats, ipts['pose']], dim=1))

    if mode == 'loss':
      loss = self.loss_fn(gazes, tgts['gaze'])
      return dict(loss=loss)

    if mode == 'predict':
      return gazes, tgts['gaze']

    return gazes


@MODELS.register_module()
class MPIIFaceGaze_FullFace(BaseModel):
  '''
  Bibliography:
    Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling.
    “It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation.”
    In 2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2299-2308, 2017.
    https://doi.org/10.1109/CVPRW.2017.284.

  Arxiv:
    https://arxiv.org/abs/1611.08860
  '''

  def __init__(self, init_cfg=None, loss_cfg=dict(type='L1Loss')):
    super(MPIIFaceGaze_FullFace, self).__init__(init_cfg=init_cfg)

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

  def forward(self, ipts, tgts, mode='tensor'):
    feats = self.conv(ipts['face'])
    feats = self.sw(feats) * feats

    gazes = self.fc(torch.flatten(feats, start_dim=1))

    if mode == 'loss':
      loss = self.loss_fn(gazes, tgts['gaze'])
      return dict(loss=loss)

    if mode == 'predict':
      return gazes, tgts['gaze']

    return gazes
