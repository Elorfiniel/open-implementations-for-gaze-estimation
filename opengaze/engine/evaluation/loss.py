from opengaze.registry import LOSSES

import torch.nn as nn


LOSSES.register_module(name='L1Loss', module=nn.L1Loss)
LOSSES.register_module(name='MSELoss', module=nn.MSELoss)
LOSSES.register_module(name='SmoothL1Loss', module=nn.SmoothL1Loss)
