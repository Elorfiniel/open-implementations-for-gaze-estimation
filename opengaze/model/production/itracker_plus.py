from mmengine.model import BaseModule
from torch.ao.quantization import QuantStub, DeQuantStub
from torchvision.models.mobilenetv2 import MobileNetV2
from torchvision.models.quantization.mobilenetv2 import QuantizableInvertedResidual
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.models.quantization.utils import _replace_relu, _fuse_modules

from opengaze.registry import MODELS
from opengaze.model.wrapper import DataFnMixin

import torch as torch
import torch.nn as nn
import torchvision as tv


@MODELS.register_module()
class ITrackerPlus(DataFnMixin, BaseModule):
  '''Replace conv backbone in iTracker with MobileNet-v2.

  Reference:
    - repo: https://gitee.com/elorfiniel/gaze-point-estimation-2023
    - file: source/models/modules/itrackers.py

  Input:
    - face crop, shape: (B, 3, 224, 224)
    - reye crop, shape: (B, 3, 224, 224)
    - leye crop, shape: (B, 3, 224, 224)
    - face kpts, shape: (B, 8)

  Output:
    - point of gaze (gx, gy), shape: (B, 2)
  '''

  def __init__(self, init_cfg: dict = None):
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
      nn.Linear(4+2*2, 64),
      nn.ReLU(inplace=True),
      nn.Linear(64, 128),
      nn.ReLU(inplace=True),
    )

    self.fc = nn.Sequential(
      nn.Linear(128+128+128, 128),
      nn.ReLU(inplace=True),
      nn.Linear(128, 2),
    )

  def data_fn(self, data_dict: dict):
    return dict(
      face=data_dict['face'], reye=data_dict['reye'],
      leye=data_dict['leye'], kpts=data_dict['kpts'],
    )

  def forward(self, face: torch.Tensor, reye: torch.Tensor,
              leye: torch.Tensor, kpts: torch.Tensor):
    feat_face = self.face_conv(face).flatten(start_dim=1)
    feat_face = self.face_fc(feat_face)

    feat_reye = self.eyes_conv(reye).flatten(start_dim=1)
    feat_leye = self.eyes_conv(leye).flatten(start_dim=1)
    feat_eyes = torch.cat([feat_reye, feat_leye], dim=1)
    feat_eyes = self.eyes_fc(feat_eyes)

    feat_kpts = self.kpts_fc(kpts)

    feat = torch.cat([feat_face, feat_eyes, feat_kpts], dim=1)
    gaze = self.fc(feat)

    return gaze


# class QuantITrackerPlus(ITrackerPlus):
#   '''Quantized version of ITrackerPlus model, used for static PTQ or QAT.'''

#   def __init__(self, init_cfg: dict = None):
#     super(QuantITrackerPlus, self).__init__(init_cfg=init_cfg)

#     self.quant_face = QuantStub()
#     self.quant_reye = QuantStub()
#     self.quant_leye = QuantStub()
#     self.quant_kpts = QuantStub()
#     self.dequant = DeQuantStub()

#     self._prepare_quant_modules()

#   def _prepare_quant_modules(self):
#     reassign = {} # For reassigning quantized modules

#     for name, module in self.named_children():
#       if type(module) is MobileNetV2:
#         new_module = MobileNetV2(block=QuantizableInvertedResidual)
#         new_module.load_state_dict(module.state_dict())
#         reassign[name] = new_module
#     for name, module in reassign.items():
#       self._modules[name] = module

#     _replace_relu(self)

#   def forward(self, face: torch.Tensor, reye: torch.Tensor,
#               leye: torch.Tensor, kpts: torch.Tensor):
#     gaze = super(QuantITrackerPlus, self).forward(
#       face=self.quant_face(face),
#       reye=self.quant_reye(reye),
#       leye=self.quant_leye(leye),
#       kpts=self.quant_kpts(kpts),
#     )
#     gaze = self.dequant(gaze)

#     return gaze

#   def fuse_model(self, is_qat: bool = None):
#     for module in self.modules():
#       if type(module) is Conv2dNormActivation:
#         _fuse_modules(module, ['0', '1', '2'], is_qat, inplace=True)
#       if type(module) is QuantizableInvertedResidual:
#         module.fuse_model(is_qat)


@MODELS.register_module(name='QuantITrackerPlus')
def build_quant_itracker_plus(*args, **kwargs):
  raise NotImplementedError(f'QuantITrackerPlus is not implemented yet.')
