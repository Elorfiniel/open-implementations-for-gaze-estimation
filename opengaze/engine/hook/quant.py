from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from typing import Union, Tuple, Dict
from torch.ao.quantization import disable_observer
from torch.nn.intrinsic.qat import freeze_bn_stats

from opengaze.registry import HOOKS
from opengaze.utils.quant import save_traced

import torch


@HOOKS.register_module()
class FreezeQuantParamsHook(Hook):
  '''Training a quantized model with high accuracy requires accurate modeling
  of the numerics at inference. This hook modifies the training loop to:

    1. switch batch norm to use running mean and variance towards the end of training
    2. freeze the quantizer parameters (scale and zero point) to finetune weights
  '''

  priority = 'NORMAL'

  def __init__(self, freeze_bn: int = -1, freeze_qt: int = -1):
    '''Accurately model the numerics of the quantized model at inference.

    Args:
      `freeze_bn`: epoch/iter to freeze batch norm parameters.
      `freeze_qt`: epoch/iter to freeze quantizer parameters.
    '''

    self.freeze_bn = freeze_bn
    self.freeze_qt = freeze_qt

  def after_train_epoch(self, runner: Runner):
    if runner.epoch == self.freeze_bn:
      runner.model.apply(freeze_bn_stats)

    if runner.epoch == self.freeze_qt:
      runner.model.apply(disable_observer)


@HOOKS.register_module()
class SaveQuantModuleHook(Hook):

  priority = 'VERY_LOW'

  def __init__(self, model_path: str, input_shapes: Union[Tuple, Dict]):
    '''Save the quantized model after quantization-aware training.

    Args:
      `model_path`: path to the model file.
      `input_shapes`: input shapes of the model.
    '''

    self.model_path = model_path
    self.input_shapes = input_shapes

  def _fetch_model(self, runner: Runner):
    if is_model_wrapper(runner.model):
      model = runner.model.module
    else:
      model = runner.model

    return model

  def _tensor_mode_inputs(self):
    if isinstance(self.input_shapes, (tuple, list)):
      example_inputs = ['tensor']
      for shape in self.input_shapes:
        example_inputs.append(torch.randn(shape))

    elif isinstance(self.input_shapes, dict):
      example_inputs = dict(mode='tensor')
      for name, shape in self.input_shapes.items():
        example_inputs[name] = torch.randn(shape)

    else:
      raise TypeError(f'Expecting a tuple or a dict, but got {type(self.input_shapes)}.')

    return example_inputs

  def after_run(self, runner: Runner):
    model = self._fetch_model(runner).cpu().eval()
    inputs = self._tensor_mode_inputs()
    save_traced(model, self.model_path, inputs)
