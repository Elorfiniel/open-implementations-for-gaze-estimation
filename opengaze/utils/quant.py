from opengaze.registry import MODELS

from typing import Union, Tuple, Dict
from mmengine.config import Config
from torch.ao.quantization import prepare_qat
from torch.quantization import (
  PerChannelMinMaxObserver,
  MovingAverageMinMaxObserver,
  QConfig, FakeQuantize,
)

import os.path as osp
import torch
import torch.nn as nn


def load_fused(model_cfg: dict, ckpt_path: str = ''):
  '''Load and fuse fp32 model from checkpoint, if provided.

  Args:
    `model_cfg`: model configuration in the registry.
    `ckpt_path`: path to the checkpoint file.
  '''

  model = MODELS.build(model_cfg)
  if ckpt_path:
    ckpt = torch.load(osp.abspath(ckpt_path), map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=True)

  if not hasattr(model, 'fuse_model'):
    raise RuntimeError(f'Model {model_cfg["type"]} does not support fusion protocol.')

  model.train()
  model.fuse_model()

  return prepare_qat(model, inplace=True)

def save_traced(model: nn.Module, model_path: str, inputs: Union[Tuple, Dict]):
  '''Save traced model to the specified file.

  Args:
    `model`: model to be traced and saved.
    `model_path`: path to the model file.
    `inputs`: example inputs, either a tuple or a dict.
  '''

  if isinstance(inputs, (tuple, list)):
    script_module = torch.jit.trace(model, example_inputs=inputs)
  elif isinstance(inputs, dict):
    script_module = torch.jit.trace(model, example_kwarg_inputs=inputs)
  else:
    raise TypeError(f'Expecting a tuple or a dict, but got {type(inputs)}.')

  torch.jit.save(script_module, osp.abspath(model_path))


class QuantConfigAdapter:
  def __init__(self, model_cfg: dict, ckpt_path: str = ''):
    '''Config adapter for quantization-aware training.

    Args:
      `model_cfg`: model configuration in the registry.
      `ckpt_path`: path to the checkpoint file.
    '''

    self.model_cfg = model_cfg
    self.ckpt_path = ckpt_path

  def model_qconfig(self):
    '''Quantization settings for activations and weights.'''

    wt_qconfig = FakeQuantize.with_args(
      observer=PerChannelMinMaxObserver,
      dtype=torch.qint8,
      quant_min=-64, quant_max=63,
      qscheme=torch.per_channel_symmetric,
    )
    act_qconfig = FakeQuantize.with_args(
      observer=MovingAverageMinMaxObserver,
      dtype=torch.quint8,
      quant_min=0, quant_max=127,
      qscheme=torch.per_tensor_affine,
    )

    return QConfig(activation=act_qconfig, weight=wt_qconfig)

  def adapt(self, cfg: Config) -> Config:
    '''Adapt runner config for quantization-aware training.

    Args:
      `cfg`: config to be adapted.
    '''

    cfg.model = load_fused(self.model_cfg, self.ckpt_path)
    cfg.model.qconfig = self.model_qconfig()

    return cfg
