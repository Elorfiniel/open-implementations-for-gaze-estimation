from opengaze.engine.transform import BaseTransform
from opengaze.registry import TRANSFORMS
from opengaze.runtime.scripts import ScriptEnv, ScriptOptions

from mmengine.config import Config
from mmengine.runner import Runner
from torchvision.transforms import functional, ColorJitter

import argparse
import numpy as np
import torch


@TRANSFORMS.register_module()
class ITrackerPlusDataAdapter(BaseTransform):
  def transform(self, results):
    results['kpts'] = torch.cat([
      results['bbox'][:4],  # Face bbox
      results['ldmk'][468], # Reye center
      results['ldmk'][473], # Leye center
    ], dim=0)
    return results

@TRANSFORMS.register_module()
class ITrackerPlusColorJitter(BaseTransform):
  def __init__(self, b: float = 0.0, c: float = 0.0, s: float = 0.0, h: float = 0.0):
    color_jitter = ColorJitter(brightness=b, contrast=c, saturation=s, hue=h)

    self.b = color_jitter.brightness
    self.c = color_jitter.contrast
    self.s = color_jitter.saturation
    self.h = color_jitter.hue

  def _apply_color_jitter(self, image, params):
    fn_idx, b, c, s, h = params

    for idx in fn_idx:
      if idx == 0 and b is not None:
        image = functional.adjust_brightness(image, b)
      elif idx == 1 and c is not None:
        image = functional.adjust_contrast(image, c)
      elif idx == 2 and s is not None:
        image = functional.adjust_saturation(image, s)
      elif idx == 3 and h is not None:
        image = functional.adjust_hue(image, h)

    return image

  def transform(self, results):
    params = ColorJitter.get_params(self.b, self.c, self.s, self.h)

    for image_name in ['face', 'reye', 'leye']:
      results[image_name] = self._apply_color_jitter(results[image_name], params)

    return results

@TRANSFORMS.register_module()
class ITrackerPlusRandomHFlip(BaseTransform):
  def __init__(self, p_hflip: float = 0.5):
    self.p_hflip = p_hflip

  def transform(self, results):
    if np.random.random() < self.p_hflip:
      results['face'] = functional.hflip(results['face'])
      results['reye'] = functional.hflip(results['reye'])
      results['leye'] = functional.hflip(results['leye'])
      results['kpts'][[0, 4, 6]] = -results['kpts'][[0, 4, 6]]
    return results


def build_config(opts: argparse.Namespace):
  # Default runtime config
  config = ScriptEnv.load_config_dict('configs/default_runtime.py')

  # Model config
  model_cfgs = ScriptEnv.load_config_dict('configs/model/production/itracker_plus.py')
  config['model'] = model_cfgs['itracker_plus']

  # Dataset config
  dataset_cfgs = ScriptEnv.load_config_dict('configs/dataset/gazecapture.py')

  pipeline = [
    dict(type='ITrackerPlusDataAdapter'),
    dict(type='ITrackerPlusColorJitter', b=0.6, c=0.4, s=0.4, h=0.4),
    dict(type='ITrackerPlusRandomHFlip', p_hflip=0.5),
  ]
  dataset_cfgs['train'].update(pipeline=pipeline)

  pipeline = [dict(type='ITrackerPlusDataAdapter')]
  for cfg_name in ['valid', 'test']:
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
  optimizer = dict(type='Adam', lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-3)
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

def adapt_config(cfg: Config, opts: argparse.Namespace):
  # Existing config
  config = cfg.to_dict()

  # Model config
  model_cfgs = ScriptEnv.load_config_dict('configs/model/production/itracker_plus.py')
  config['model'] = model_cfgs['quant_itracker_plus']

  # Hook config
  config['custom_hooks'].extend([
    dict(
      type='FreezeQuantParamsHook',
      freeze_bn=opts.qat_freeze_bn,
      freeze_qt=opts.qat_freeze_qt,
    ),
  ])

  return Config(config)


def main_procedure(opts: argparse.Namespace):
  ScriptEnv.unified_runtime_environment()

  config = build_config(opts)
  if opts.quant:
    config = adapt_config(config, opts)
  ScriptEnv.merge_config(config, opts)

  runner = Runner.from_cfg(config)
  if opts.mode == 'train':
    runner.train()
  if opts.mode == 'test':
    runner.test()



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='run script for itracker plus model.')

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
    '--batch-size', type=int, default=60,
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
    '--ema-epoch', type=int, default=24,
    help='begin epoch of exponential moving average.',
  )

  quant_group = parser.add_argument_group(
    title='quant options',
    description='config options for quantization-aware training.',
  )

  quant_group.add_argument(
    '--quant', action='store_true', default=False,
    help='enable quantization aware training.',
  )
  quant_group.add_argument(
    '--qat-freeze-bn', type=int, default=-1,
    help='(qat) epoch to freeze batch norm layers.'
  )
  quant_group.add_argument(
    '--qat-freeze-qt', type=int, default=-1,
    help='(qat) epoch to freeze quantizer params.'
  )

  opts, _ = ScriptOptions(parser).parse_args()

  main_procedure(opts)
