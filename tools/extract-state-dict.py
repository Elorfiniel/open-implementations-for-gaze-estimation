# Tool: extract state dict from mmengine checkpoint
#
# This tool assumes that the checkpoint file produced by `mmengine`
# contains a `state_dict` field, which contains the model weights
#
# Usage: python extract-state-dict.py mmengine-ckpt state-dict-file
#
# Use '--help' to see option descriptions


from opengaze.runtime.scripts import ScriptEnv

import argparse
import os
import os.path as osp
import torch


def main_procedure(opts: argparse.Namespace):
  ScriptEnv.unified_runtime_environment()

  if not osp.isfile(opts.mmengine_ckpt):
    raise FileNotFoundError(f'No checkpoint file found at "{opts.mmengine_ckpt}".')

  load_kwargs = dict(weights_only=False)  # See torch 2.6.0 updates
  if opts.map_location:
    load_kwargs['map_location'] = opts.map_location
  mmengine_ckpt = torch.load(opts.mmengine_ckpt, **load_kwargs)

  if not opts.key in mmengine_ckpt:
    raise KeyError(f'No key "{opts.key}" found in checkpoint file.')
  state_dict = mmengine_ckpt[opts.key]

  os.makedirs(osp.dirname(opts.state_dict_file), exist_ok=True)
  torch.save(state_dict, opts.state_dict_file)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='extract state dict from mmengine checkpoint.')

  parser.add_argument(
    'mmengine-ckpt', type=str,
    help='checkpoint file saved via mmengine.',
  )
  parser.add_argument(
    'state-dict-file', type=str,
    help='state dict file to save extracted parameters.',
  )

  parser.add_argument(
    '--key', type=str, default='state_dict',
    help='key of the state dict in mmengine checkpoint.',
  )
  parser.add_argument(
    '--map-location', type=str, default='',
    help='map location of the state dict, if specified.',
  )

  main_procedure(parser.parse_args())
