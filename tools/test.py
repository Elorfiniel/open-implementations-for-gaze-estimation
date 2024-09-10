from argparse import ArgumentParser

from mmengine.config import Config, DictAction
from mmengine.evaluator import DumpResults
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


def dump_file_from_cmdarg(cmdarg):
  dirname, basename = osp.split(osp.abspath(cmdarg))
  if not basename.endswith(('.pkl', '.pickle')):
    basename = f'{osp.splitext(basename)[0]}.pkl'

  dump_file = osp.join(dirname, basename)
  return dump_file


def parse_cmdargs():
  parser = ArgumentParser(description='initialize testing from config.')

  parser.add_argument('config', type=str, help='config file for testing.')
  parser.add_argument('checkpoint', type=str, help='checkpoint for the model.')

  parser.add_argument('--work-dir', type=str, help='directory for logs and results.')
  parser.add_argument('--dump-pkl', type=str, help='pickle dump predictions for offline evaluation.')
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

  cfg.load_from = cmdargs.checkpoint

  if 'runner_type' not in cfg:
    runner = Runner.from_cfg(cfg)
  else:
    runner = RUNNERS.build(cfg)

  if cmdargs.dump_pkl is not None:
    dump_file = dump_file_from_cmdarg(cmdargs.dump_pkl)
    runner.test_evaluator.metrics.append(DumpResults(dump_file))

  runner.test()


if __name__ == '__main__':
  main_procedure()
