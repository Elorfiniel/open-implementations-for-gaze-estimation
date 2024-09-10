from argparse import ArgumentParser

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from template.registry import RUNNERS

import logging
import os
import os.path as osp
import pprint as pp


def work_dir_from_config(config):
  project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
  work_dirname = osp.splitext(osp.basename(config))[0].replace('.', '_')
  work_dir = osp.join(project_root, 'work_dir', work_dirname)
  return work_dir


def parse_cmdargs():
  parser = ArgumentParser(description='initialize training from config.')

  parser.add_argument('config', type=str, help='config file for training.')

  parser.add_argument('--work-dir', type=str, help='directory for logs and checkpoints.')
  parser.add_argument('--amp', action='store_true', default=False,
                      help='enable automatic mixed-precision training.')
  parser.add_argument('--resume', action='store_true', default=False,
                      help='resume from the latest checkpoint the the work directory.')
  parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                      help='training launcher, select none for non-distributed envs.')
  parser.add_argument('--local-rank', type=int, default=0)

  parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                      help='override settings in the used config file.')

  cmdargs = parser.parse_args()
  if 'LOCAL_RANK' not in os.environ:
    os.environ['LOCAL_RANK'] = str(cmdargs.local_rank)

  return cmdargs


def main_procedure():
  cmdargs = parse_cmdargs()

  cfg = Config.fromfile(cmdargs.config)
  cfg.launcher = cmdargs.launcher
  if cmdargs.cfg_options is not None:
    cfg.merge_from_dict(cmdargs.cfg_options)

  if cmdargs.work_dir is not None:
    cfg.work_dir = cmdargs.work_dir
  elif cfg.get('work_dir', None) is None:
    cfg.work_dir = work_dir_from_config(cmdargs.config)

  if cmdargs.amp is True:
    optim_wrapper = cfg.optim_wrapper.type
    if optim_wrapper == 'AmpOptimWrapper':
      print_log('automatic mixed-precision training is already enabled.',
                logger='current', level=logging.WARNING)
    else:
      assert optim_wrapper == 'OptimWrapper', (
        '`--amp` is only supported when the optimizer wrapper type is `OptimWrapper`'
      )
      cfg.optim_wrapper.type = 'AmpOptimWrapper'
      cfg.optim_wrapper.loss_scale = 'dynamic'

  cfg.resume = cmdargs.resume

  if 'runner_type' not in cfg:
    runner = Runner.from_cfg(cfg)
  else:
    runner = RUNNERS.build(cfg)

  runner.train()


if __name__ == '__main__':
  main_procedure()
