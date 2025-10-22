from mmengine.model import BaseModule
from timm.layers import ConvNormAct

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


class _CANetFace(nn.Module):
  def __init__(self, out_features: int = 256):
    super(_CANetFace, self).__init__()

    conv_kwargs = dict(
      kernel_size=3, stride=1, padding=1,
      norm_layer=nn.BatchNorm2d,
      act_layer=nn.ReLU,
      act_kwargs=dict(inplace=True),
    )
    self.conv = nn.Sequential(
      ConvNormAct(in_channels=3, out_channels=64, **conv_kwargs),
      ConvNormAct(in_channels=64, out_channels=64, **conv_kwargs),
      nn.MaxPool2d(kernel_size=2, stride=2),
      ConvNormAct(in_channels=64, out_channels=128, **conv_kwargs),
      ConvNormAct(in_channels=128, out_channels=128, **conv_kwargs),
      nn.MaxPool2d(kernel_size=2, stride=2),
      ConvNormAct(in_channels=128, out_channels=256, **conv_kwargs),
      ConvNormAct(in_channels=256, out_channels=256, **conv_kwargs),
      ConvNormAct(in_channels=256, out_channels=256, **conv_kwargs),
      ConvNormAct(in_channels=256, out_channels=256, **conv_kwargs),
      ConvNormAct(in_channels=256, out_channels=256, **conv_kwargs),
      ConvNormAct(in_channels=256, out_channels=256, **conv_kwargs),
      nn.MaxPool2d(kernel_size=2, stride=2),
      ConvNormAct(in_channels=256, out_channels=512, **conv_kwargs),
      ConvNormAct(in_channels=512, out_channels=512, **conv_kwargs),
      nn.MaxPool2d(kernel_size=2, stride=2),
      ConvNormAct(in_channels=512, out_channels=1024, **conv_kwargs),
      nn.AdaptiveAvgPool2d(output_size=(1, 1)),
    )
    self.fc = nn.Sequential(
      nn.Linear(in_features=1024, out_features=out_features),
      nn.ReLU(inplace=True),
    )

  def forward(self, feats: torch.Tensor):
    n, c, h, w = feats.size()

    feats = self.conv(feats).view(n, -1)
    feats = self.fc(feats)

    return feats

class _CANetEyes(nn.Module):
  def __init__(self, out_features: int = 256):
    super(_CANetEyes, self).__init__()

    conv_kwargs = dict(
      kernel_size=3, stride=1, padding=1,
      norm_layer=nn.BatchNorm2d,
      act_layer=nn.ReLU,
      act_kwargs=dict(inplace=True),
    )
    self.conv = nn.Sequential(
      ConvNormAct(in_channels=3, out_channels=64, **conv_kwargs),
      ConvNormAct(in_channels=64, out_channels=64, **conv_kwargs),
      nn.MaxPool2d(kernel_size=2, stride=2),
      ConvNormAct(in_channels=64, out_channels=128, **conv_kwargs),
      ConvNormAct(in_channels=128, out_channels=128, **conv_kwargs),
      ConvNormAct(in_channels=128, out_channels=128, **conv_kwargs),
      nn.MaxPool2d(kernel_size=2, stride=2),
      ConvNormAct(in_channels=128, out_channels=256, **conv_kwargs),
      ConvNormAct(in_channels=256, out_channels=256, **conv_kwargs),
      ConvNormAct(in_channels=256, out_channels=256, **conv_kwargs),
      nn.MaxPool2d(kernel_size=2, stride=2),
      ConvNormAct(in_channels=256, out_channels=512, **conv_kwargs),
      nn.MaxPool2d(kernel_size=2, stride=2),
      ConvNormAct(in_channels=512, out_channels=1024, **conv_kwargs),
      nn.AdaptiveAvgPool2d(output_size=(1, 1)),
    )
    self.fc = nn.Sequential(
      nn.Linear(in_features=1024, out_features=out_features),
      nn.ReLU(inplace=True),
    )

  def forward(self, feats: torch.Tensor):
    n, c, h, w = feats.size()

    feats = self.conv(feats).view(n, -1)
    feats = self.fc(feats)

    return feats

class _CANetHead(nn.Module):
  def __init__(self, num_features: int = 256):
    super(_CANetHead, self).__init__()

    self.gate = nn.GRU(
      input_size=num_features,
      hidden_size=num_features,
      num_layers=1,
      batch_first=True,
    )
    self.fc = nn.Linear(in_features=num_features, out_features=2)

  def forward(self, feat: torch.Tensor, feat_hidden: torch.Tensor = None):
    _, hidden = self.gate(feat, feat_hidden)
    _, n, c = hidden.size()
    gaze = self.fc(hidden.view(n, c))

    return gaze, hidden

class _CANetAttention(nn.Module):
  def __init__(self, num_features: int = 256):
    super(_CANetAttention, self).__init__()

    self.face_fc = nn.Linear(in_features=num_features, out_features=num_features)
    self.eyes_fc = nn.Linear(in_features=num_features, out_features=num_features)

    self.score_fc = nn.Linear(in_features=num_features, out_features=1)

  def forward(self, feat_face: torch.Tensor, feat_reye: torch.Tensor, feat_leye: torch.Tensor):
    feat_f = self.face_fc(feat_face)
    feat_r = self.eyes_fc(feat_reye)
    feat_l = self.eyes_fc(feat_leye)

    feat_r = nn.functional.tanh(feat_f + feat_r)
    mr = self.score_fc(feat_r)
    feat_l = nn.functional.tanh(feat_f + feat_l)
    ml = self.score_fc(feat_l)

    scores = torch.cat([mr, ml], dim=1)
    wr, wl = torch.split(scores, 1, dim=1)
    feat_eyes = wr * feat_reye + wl * feat_leye

    return feat_eyes

@MODELS.register_module()
class CANet(DataFnMixin, BaseModule):
  '''
  Bibliography:
    Cheng, Yihua, Shiyao Huang, Fei Wang, Chen Qian, and Feng Lu.
    "A Coarse-to-Fine Adaptive Network for Appearance-Based Gaze Estimation."

  Arxiv:
    https://arxiv.org/abs/2001.00187

  Input:
    - normalized face patch, shape: (B, 3, 224, 224)
    - normalized reye patch, shape: (B, 3, 36, 60)
    - normalized leye patch, shape: (B, 3, 36, 60)

  Output:
    - gaze vector (pitch, yaw), shape: (B, 2)
  '''

  def __init__(self, init_cfg: dict = None):
    super(CANet, self).__init__(init_cfg=init_cfg)

    self.face = _CANetFace(out_features=256)
    self.face_head = _CANetHead(num_features=256)

    self.reye = _CANetEyes(out_features=256)
    self.leye = _CANetEyes(out_features=256)
    self.eyes_head = _CANetHead(num_features=256)

    self.att = _CANetAttention(num_features=256)

  def data_fn(self, data_dict: dict):
    return dict(face=data_dict['face'], reye=data_dict['reye'], leye=data_dict['leye'])

  def forward(self, face: torch.Tensor, reye: torch.Tensor, leye: torch.Tensor):
    feat_face = self.face(face).unsqueeze(dim=1)
    feat_reye = self.reye(reye).unsqueeze(dim=1)
    feat_leye = self.leye(leye).unsqueeze(dim=1)

    gaze_b, h1 = self.face_head(feat_face)
    feat_face_hidden = h1.squeeze(dim=0).unsqueeze(dim=1)
    feat_eyes = self.att(feat_face_hidden, feat_reye, feat_leye)
    gaze_r, h2 = self.eyes_head(feat_eyes, h1)
    gaze = gaze_b + gaze_r

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


@MODELS.register_module()
class GazeTR(DataFnMixin, BaseModule):
  '''
  Bibliography:
    Cheng, Yihua, and Feng Lu.
    "Gaze Estimation Using Transformer."

  Arxiv:
    https://arxiv.org/abs/2105.14424

  Input:
    - normalized face patch, shape: (B, 3, 224, 224)

  Output:
    - gaze vector (pitch, yaw), shape: (B, 2)
  '''

  def __init__(self, n_encoders: int = 6, d_model: int = 32, n_head: int = 8,
               d_ffn: int = 512, p_dropout: float = 0.1, init_cfg: dict = None):
    super(GazeTR, self).__init__(init_cfg=init_cfg)

    resnet18 = tv.models.resnet18()
    conv_layers = [
      module
      for name, module in resnet18.named_children()
      if not name in ['avgpool', 'fc']
    ]
    self.conv = nn.Sequential(
      *conv_layers,
      nn.Conv2d(in_channels=512, out_channels=d_model, kernel_size=1),
      nn.BatchNorm2d(num_features=d_model),
      nn.ReLU(inplace=True),
    )

    self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
    self.pos_embed = nn.Embedding(num_embeddings=50, embedding_dim=d_model)

    encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model,
      nhead=n_head,
      dim_feedforward=d_ffn,
      dropout=p_dropout,
      batch_first=True,
    )
    self.encoder = nn.TransformerEncoder(
      encoder_layer=encoder_layer,
      num_layers=n_encoders,
      norm=nn.LayerNorm(normalized_shape=d_model),
    )

    self.fc = nn.Linear(in_features=d_model, out_features=2)

  def data_fn(self, data_dict: dict):
    return dict(face=data_dict['face'])

  def forward(self, face: torch.Tensor):
    n, c, h, w = face.size()

    patch_token = self.conv(face).flatten(start_dim=2)
    patch_token = patch_token.permute(0, 2, 1)

    cls_token = self.cls_token.repeat(n, 1, 1)
    patch_token = torch.cat([cls_token, patch_token], dim=1)

    pos = torch.arange(0, 50).to(
      device=self.pos_embed.weight.device,
    )
    pos_token = self.pos_embed(pos).unsqueeze(dim=0).repeat(n, 1, 1)

    feat = self.encoder(patch_token + pos_token)
    feat_cls = feat[:, 0, :]
    gaze = self.fc(feat_cls)

    return gaze
