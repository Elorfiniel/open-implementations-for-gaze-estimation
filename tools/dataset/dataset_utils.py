import datetime
import logging
import os
import os.path as osp


__all__ = ['fetch_dataset_logger', 'resource_path', 'create_data_folder']


_stream_handler = logging.StreamHandler()
_stream_handler.setLevel(logging.INFO)

_formatter = logging.Formatter(
  '[ %(asctime)s ] [ %(name)s ] process %(process)d - %(levelname)s: %(message)s',
  datefmt='%m-%d %H:%M:%S',
)
_stream_handler.setFormatter(_formatter)

_module_logger = logging.getLogger('dataset')
_module_logger.addHandler(_stream_handler)
_module_logger.setLevel(logging.INFO)


def fetch_dataset_logger(dataset: str = ''):
  '''Fetch the logger for dataset, creating one if not already exists.'''
  return _module_logger.getChild(dataset) if dataset else _module_logger


def resource_path(resource: str):
  '''Build resource path by prepending the resource folder.'''
  workspace = osp.dirname(osp.dirname(osp.dirname(__file__)))
  return osp.join(workspace, 'resource', resource)


def create_data_folder(dataset: str = ''):
  '''Create a data folder for the specified dataset.'''

  if not dataset:
    timestamp = datetime.datetime.now().strftime('%m-%d-%Y-%H-%M-%S')
    dataset = f'unknown-{timestamp}'

  workspace = osp.dirname(osp.dirname(osp.dirname(__file__)))
  data_folder = osp.join(osp.abspath(workspace), 'data', dataset)

  os.makedirs(data_folder, exist_ok=True)

  return data_folder
