# Tool: trace model after quantization-aware training
#
# This tool assumes that quantization-aware training
# has been performed on a model wrapped inside the
# base model wrapper, see "opengaze/model/wrapper.py"
#
# Usage: python trace-quantized.py \
#   --config <qat-script-config-file> \
#   --qat-fp32 <fp32-checkpoint-file> \
#   --qat-int8 <int8-quantized-model> \
#   --input-shapes input_name=input_shape
#
# Use '--help' to see option descriptions.

from opengaze.runtime.scripts import ScriptEnv, ScriptOptions

from mmengine.config import Config, DictAction
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from torch.ao.quantization import convert
from typing import Dict, List, Tuple, Union

import argparse
import torch
import torch.nn as nn


def convert_wrapped_model(runner: Runner):
  '''Convert the actual wrapped model from mmengine runner:

  Runner wrapper -> Base wrapper -> Actual model

  See also base wrapper in "opengaze/model/wrapper.py".
  '''

  # Unwrap base model wrapper from mmengine runner
  if is_model_wrapper(runner.model):
    model = runner.model.module
  else:
    model = runner.model

  # Convert the actual model in base model wrapper
  model_int8 = convert(model.model.cpu().eval())

  return model_int8


def prepare_example_inputs(input_shapes: Union[Dict, List, Tuple]):
  if isinstance(input_shapes, (list, tuple)):
    example_inputs = list()
    for shape in input_shapes:
      example_inputs.append(torch.randn(shape))

  elif isinstance(input_shapes, dict):
    example_inputs = dict()
    for name, shape in input_shapes.items():
      example_inputs[name] = torch.randn(shape)

  else:
    raise TypeError(f'Expecting a list, a tuple or a dict, but got {type(input_shapes)}.')

  return example_inputs


def trace_quant_model(model_int8: nn.Module, example_inputs: Union[Dict, List]):
  if isinstance(example_inputs, (list, tuple)):
    script_module = torch.jit.trace(model_int8, example_inputs=example_inputs)
  elif isinstance(example_inputs, dict):
    script_module = torch.jit.trace(model_int8, example_kwarg_inputs=example_inputs)
  else:
    raise TypeError(f'Expecting a list, a tuple or a dict, but got {type(example_inputs)}.')

  return script_module


def build_config(opts: argparse.Namespace):
  # Default runtime config
  config = ScriptEnv.load_config_dict('configs/default-runtime.py')

  # Config from quantization-aware training script
  train_config = ScriptEnv.load_config_dict(opts.config)

  # Only model config and checkpoint info are needed
  config['model'] = train_config['model']
  config['load_from'] = opts.qat_fp32

  return Config(config)


def main_procedure(opts: argparse.Namespace):
  ScriptEnv.unified_runtime_environment()

  config = build_config(opts)
  ScriptEnv.merge_config(config, opts)

  runner = Runner.from_cfg(config)
  runner.load_or_resume()

  model_int8 = convert_wrapped_model(runner)
  example_inputs = prepare_example_inputs(opts.input_shapes)
  script_module = trace_quant_model(model_int8, example_inputs)
  torch.jit.save(script_module, opts.qat_int8)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='trace model after quantization-aware training.')

  parser.add_argument(
    '--config', type=str, required=True,
    help='config file produced by quantization-aware training script.',
  )
  parser.add_argument(
    '--qat-fp32', type=str, required=True,
    help='(qat) load from this fp32 model as source.'
  )
  parser.add_argument(
    '--qat-int8', type=str, required=True,
    help='(qat) save into this int8 model as target.'
  )

  parser.add_argument(
    '--input-shapes', nargs='+', action=DictAction,
    help='mapping from input names to shapes, eg. "name=[N,C,H,W]".'
  )

  opts, _ = ScriptOptions(parser).parse_args()

  main_procedure(opts)
