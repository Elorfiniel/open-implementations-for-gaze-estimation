from torch.quantization import (
  PerChannelMinMaxObserver,
  MovingAverageMinMaxObserver,
  QConfig, FakeQuantize,
)

from opengaze.registry import QCONFIGS

import torch as torch
import torch.nn as nn


@QCONFIGS.register_module()
class X86BasicQConfig:
  '''See also: https://docs.pytorch.org/docs/stable/quantization.html#best-practices'''

  def attach_qconfig(self, model: nn.Module):
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
    model.qconfig = QConfig(activation=act_qconfig, weight=wt_qconfig)

    return model
