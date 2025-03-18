# This experiment compares the performance of CNN backbones (eg. ResNet, in the
# "feature extraction + gaze regression" fashion) with different normalization
# methods. Specifically, we compare the performance of:
#   - ResNet-50 + BatchNorm
#   - ResNet-50 + LayerNorm
#
# The experiment is conducted on the XGaze224 dataset.
#   (Pitch, Yaw) = Model(Image)


from opengaze.runtime.scripts import ScriptEnv, ScriptOptions

from mmengine.config import Config
from mmengine.runner import Runner

import argparse


def build_config(opts: argparse.Namespace):
  # Default runtime config
  config = ScriptEnv.load_config_dict('configs/default_runtime.py')

  # Model config
  model_cfgs = ScriptEnv.load_config_dict('configs/model/experiments/norm_layer.py')
  model_dict = { 'BatchNorm': 'resnet50_bn', 'LayerNorm': 'resnet50_ln' }
  config['model'] = model_cfgs[model_dict[opts.norm]]

  # Dataset config
  dataset_cfgs = ScriptEnv.load_config_dict('configs/dataset/xgaze224.py')
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
  config['test_dataloader'] = config['val_dataloader']

  # Evaluator config
  config['val_evaluator'] = dict(type='AngularError')
  config['test_evaluator'] = dict(type='AngularError')

  # Loop config
  config['train_cfg'] = dict(
    by_epoch=True, max_epochs=opts.max_epochs,
    val_begin=1, val_interval=1,
  )
  config['val_cfg'] = dict(type='ValLoop')
  config['test_cfg'] = dict(type='TestLoop')

  # Optimizer config
  optimizer = dict(type='Adam', lr=1e-4)
  config['optim_wrapper'] = dict(type='OptimWrapper', optimizer=optimizer)

  # Scheduler config
  config['param_scheduler'] = [
    dict(
      type='LinearLR', by_epoch=False,
      start_factor=opts.warm_up_start,
      begin=0, end=opts.warn_up_step,
    ),
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
  parser = argparse.ArgumentParser(description='run script for normalization experiments.')

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
    '--max-epochs', type=int, default=50,
    help='max number of epochs for training.',
  )
  config_group.add_argument(
    '--warm-up-start', type=float, default=0.01,
    help='warm up start for learning rate scheduler.',
  )
  config_group.add_argument(
    '--warn-up-step', type=int, default=100,
    help='warn up step for learning rate scheduler.',
  )
  config_group.add_argument(
    '--step-size', type=int, default=20,
    help='step size for learning rate scheduler.',
  )
  config_group.add_argument(
    '--gamma', type=float, default=0.1,
    help='gamma for learning rate scheduler.',
  )

  config_group.add_argument(
    '--norm', type=str, required=True,
    choices=['BatchNorm', 'LayerNorm'],
    help='select normalization method for backbone.',
  )

  config_group.add_argument(
    '--ema-epoch', type=int, default=-1,
    help='begin epoch of exponential moving average.',
  )

  opts, _ = ScriptOptions(parser).parse_args()

  main_procedure(opts)
