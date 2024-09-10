from argparse import ArgumentParser

from mmengine.config import Config, DictAction
from mmengine.logging import print_log

import logging
import os.path as osp
import pprint as pp


def dump_file_from_cmdarg(cmdarg):
  dirname, basename = osp.split(osp.abspath(cmdarg))
  if not basename.endswith(('.py')):
    basename = f'{osp.splitext(basename)[0]}.py'

  dump_file = osp.join(dirname, basename)
  return dump_file


def parse_cmdargs():
  parser = ArgumentParser(description='print the merged config.')

  parser.add_argument('config', type=str, help='config file for training.')

  parser.add_argument('--dump-cfg', type=str, help='dump the merged config for edition.')

  parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                      help='override settings in the used config file.')

  cmdargs = parser.parse_args()

  return cmdargs


def main_procedure():
  cmdargs = parse_cmdargs()

  cfg = Config.fromfile(cmdargs.config)
  if cmdargs.cfg_options is not None:
    cfg.merge_from_dict(cmdargs.cfg_options)

  print_log(pp.pformat(cfg.to_dict()), logger='current', level=logging.INFO)

  if cmdargs.dump_cfg is not None:
    dump_file = dump_file_from_cmdarg(cmdargs.dump_cfg)
    cfg.dump(dump_file)


if __name__ == '__main__':
  main_procedure()
