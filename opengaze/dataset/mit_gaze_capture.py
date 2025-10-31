from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

from opengaze.registry import DATASETS
from opengaze.utils.dataset import build_image_transform, build_data_pipeline

import cv2
import h5py
import json
import numpy as np
import os
import os.path as osp
import torch


@DATASETS.register_module()
class GazeCapture(Dataset):
  def __init__(self, root: str, split: str, devices: list,
               transform=None, pipeline=None):
    '''GazeCapture dataset.

    Args:
      `root`: root directory of the dataset where prepared data for each
      subject is stored, eg. 'data/gazecapture'.

      `split`: dataset split, select from `['train', 'val', 'test']`.

      `devices`: list of capture devices.

      `transform`: image transformation.

      `pipeline`: data processing pipeline.

    Devices used to collect GazeCapture dataset include:
      - 'iPad 2', 'iPad 3', 'iPad 4', 'iPad Air', 'iPad Air 2', 'iPad Mini', 'iPad Pro'
      - 'iPhone 4S', 'iPhone 5', 'iPhone 5C', 'iPhone 5S'
      - 'iPhone 6', 'iPhone 6 Plus', 'iPhone 6s', 'iPhone 6s Plus'
    '''

    super(GazeCapture, self).__init__()

    subject_folders = [f for f in os.listdir(root) if osp.isdir(osp.join(root, f))]

    subjects = [] # Load subjects captured by devices from dataset split
    for subject_folder in subject_folders:
      metadata_file = osp.join(root, subject_folder, 'meta.json')
      with open(metadata_file, 'r') as file:
        metadata = json.load(file)
      if metadata['split'] == split and metadata['device'] in devices:
        subjects.append(subject_folder)

    self.data = ConcatDataset([
      _GazeCaptureSubject(root, subject, transform)
      for subject in subjects
    ])
    self.pipeline = build_data_pipeline(pipeline)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.pipeline(self.data[idx])


class _GazeCaptureSubject(Dataset):
  def __init__(self, root: str, subject: str, transform=None):
    '''GazeCapture dataset for each subject.

    Args:
      `root`: root directory of the dataset where prepared data for each
      subject is stored, eg. 'data/gazecapture'.

      `subject`: subject ID (eg. '00002') to load.

      `transform`: image transformation.
    '''

    super(_GazeCaptureSubject, self).__init__()

    self.root = root
    self.subject = subject

    self.hdf_path = osp.join(root, subject, 'annot.h5')
    with h5py.File(self.hdf_path, 'r', swmr=True) as hdf_file:
      self.ds = [d for d in hdf_file.keys() if d != 'name']
      self.n_samples = hdf_file['name'].len()
    self.hdf = None

    self.transform = build_image_transform(transform)

  def _load_hdf_file(self, hdf_path):
    if self.hdf is None:
      self.hdf = h5py.File(hdf_path, 'r', swmr=True)

  def _load_crop(self, image_path):
    crop = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = Image.fromarray(crop, mode='RGB')
    if self.transform:
      crop = self.transform(crop)
    return crop

  def __len__(self):
    return self.n_samples

  def __getitem__(self, idx):
    self._load_hdf_file(self.hdf_path)

    image_name = self.hdf['name'].asstr()[idx]

    face = self._load_crop(osp.join(self.root, self.subject, 'face', image_name))
    reye = self._load_crop(osp.join(self.root, self.subject, 'reye', image_name))
    leye = self._load_crop(osp.join(self.root, self.subject, 'leye', image_name))

    data_dict = dict(face=face, reye=reye, leye=leye)
    for dname in self.ds:
      data = np.array(self.hdf[dname][idx])
      data = torch.tensor(data, dtype=torch.float32)
      data_dict[dname] = data

    return data_dict
