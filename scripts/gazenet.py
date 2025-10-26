from opengaze.runtime.scripts import ScriptEnv, ScriptOptions
from opengaze.runtime.cmdargs import str2bool

from mmengine.config import Config
from mmengine.runner import Runner

import argparse


def build_config(opts: argparse.Namespace):
  # Default runtime config
  config = ScriptEnv.load_config_dict('configs/default_runtime.py')

  # Model config
  model_cfgs = ScriptEnv.load_config_dict('configs/model/gaze_3d.py')
  config['model'] = model_cfgs['gazenet']

  # Dataset config
  dataset_cfgs = ScriptEnv.load_config_dict('configs/dataset/mpiigaze.py')
  dataset_cfgs['train']['test_pp'] = f'p{opts.test_pp:02d}'
  dataset_cfgs['valid']['test_pp'] = f'p{opts.test_pp:02d}'
  dataset_cfgs['train']['eval_subset'] = opts.eval_subset
  dataset_cfgs['valid']['eval_subset'] = opts.eval_subset
  transform = [
    dict(type='Grayscale', num_output_channels=3),
    dict(type='ToTensor'),
  ]
  dataset_cfgs['train']['transform'] = transform
  dataset_cfgs['valid']['transform'] = transform
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
  optimizer = dict(type='Adam', lr=1e-5, betas=(0.90, 0.95))
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
  parser = argparse.ArgumentParser(description='run script for gazenet baseline.')

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
    '--max-epochs', type=int, default=60,
    help='max number of epochs for training.',
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
    '--test-pp', type=int, default=0,
    choices=list(range(0, 15)),
    help='person id for leave-one-out test',
  )
  config_group.add_argument(
    '--eval-subset', type=str2bool, default='true',
    help='whether to use evaluation subset',
  )

  config_group.add_argument(
    '--ema-epoch', type=int, default=-1,
    help='begin epoch of exponential moving average.',
  )

  opts, _ = ScriptOptions(parser).parse_args()

  main_procedure(opts)
