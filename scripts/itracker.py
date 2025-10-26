from opengaze.engine.transform import BaseTransform
from opengaze.registry import TRANSFORMS
from opengaze.runtime.scripts import ScriptEnv, ScriptOptions

from mmengine.config import Config
from mmengine.runner import Runner

import argparse
import numpy as np
import torch


@TRANSFORMS.register_module()
class ITrackerDataAdapter(BaseTransform):
  def __init__(self, grid_size: int = 25):
    self.grid_size = grid_size

  def transform(self, results):
    '''Make face grid from face bbox, see official implementation of iTracker:
    https://github.com/CSAILVision/GazeCapture/blob/master/pytorch/ITrackerData.py
    '''

    x, y, w, h = results['grid'].tolist()

    grid_length = self.grid_size * self.grid_size
    grid = np.zeros(grid_length, dtype=np.float32)

    ind_x = np.array([i % self.grid_size for i in range(grid_length)])
    ind_y = np.array([i // self.grid_size for i in range(grid_length)])

    cond_x = np.logical_and(ind_x >= x, ind_x < x + w)
    cond_y = np.logical_and(ind_y >= y, ind_y < y + h)

    grid[np.logical_and(cond_x, cond_y)] = 1.0
    results['grid'] = torch.tensor(grid, dtype=torch.float32)

    return results


def build_config(opts: argparse.Namespace):
  # Default runtime config
  config = ScriptEnv.load_config_dict('configs/default_runtime.py')

  # Model config
  model_cfgs = ScriptEnv.load_config_dict('configs/model/gaze_2d.py')
  config['model'] = model_cfgs['itracker']

  # Dataset config
  dataset_cfgs = ScriptEnv.load_config_dict('configs/dataset/mit_gaze_capture.py')

  pipeline = [dict(type='ITrackerDataAdapter', grid_size=25)]
  for cfg_name in ['train', 'valid', 'test']:
    dataset_cfgs[cfg_name].update(pipeline=pipeline)

  config['train_dataloader'] = dict(
    dataset=dataset_cfgs['train'],
    num_workers=opts.num_workers,
    batch_size=opts.batch_size,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
  )
  config['val_dataloader'] = dict(
    dataset=dataset_cfgs['valid'],
    num_workers=opts.num_workers,
    batch_size=opts.batch_size,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
  )
  config['test_dataloader'] = dict(
    dataset=dataset_cfgs['test'],
    num_workers=opts.num_workers,
    batch_size=opts.batch_size,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
  )

  # Evaluator config
  config['val_evaluator'] = dict(type='DistanceError')
  config['test_evaluator'] = dict(type='DistanceError')

  # Loop config
  config['train_cfg'] = dict(
    by_epoch=True, max_epochs=opts.max_epochs,
    val_begin=1, val_interval=1,
  )
  config['val_cfg'] = dict(type='ValLoop')
  config['test_cfg'] = dict(type='TestLoop')

  # Optimizer config
  optimizer = dict(type='Adam', lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-4)
  config['optim_wrapper'] = dict(type='OptimWrapper', optimizer=optimizer)

  # Scheduler config
  config['param_scheduler'] = [
    dict(
      type='StepLR', by_epoch=True, begin=0,
      step_size=opts.step_size, gamma=opts.gamma,
    ),
  ]

  # Hook config
  config['custom_hooks'] = [
    dict(
      type='CheckpointHook',
      save_best='mae',
      rule='less',
      save_last=False,
    ),
  ]
  if opts.ema_epoch in range(opts.max_epochs):
    ema_hook = dict(type='EMAHook', begin_epoch=opts.ema_epoch)
    config['custom_hooks'].append(ema_hook)

  # Enable automatic scaling of learning rate
  config['auto_scale_lr'] = dict(enable=True, base_batch_size=opts.batch_size)

  return Config(config)


def main_procedure(opts: argparse.Namespace):
  ScriptEnv.unified_runtime_environment()

  config = build_config(opts)
  ScriptEnv.merge_config(config, opts)

  runner = Runner.from_cfg(config)
  if opts.mode == 'train':
    runner.train()
  if opts.mode == 'test':
    runner.test()



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='run script for itracker baseline.')

  parser.add_argument(
    '--mode', choices=['train', 'test'], default='train',
    help='select mode for script, train or test.',
  )

  config_group = parser.add_argument_group(
    title='config options',
    description='config options for script.',
  )

  config_group.add_argument(
    '--num-workers', type=int, default=4,
    help='number of workers for pytorch dataloader.',
  )
  config_group.add_argument(
    '--batch-size', type=int, default=100,
    help='batch size for pytorch dataloader.',
  )
  config_group.add_argument(
    '--max-epochs', type=int, default=30,
    help='max number of epochs for training.',
  )
  config_group.add_argument(
    '--step-size', type=int, default=10,
    help='step size for learning rate scheduler.',
  )
  config_group.add_argument(
    '--gamma', type=float, default=0.1,
    help='gamma for learning rate scheduler.',
  )

  config_group.add_argument(
    '--ema-epoch', type=int, default=-1,
    help='begin epoch of exponential moving average.',
  )

  opts, _ = ScriptOptions(parser).parse_args()

  main_procedure(opts)
