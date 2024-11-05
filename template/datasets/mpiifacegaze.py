from mmengine.dataset import Compose
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

from template.registry import DATASETS

import bisect
import cv2
import numpy as np
import os
import os.path as osp
import torch as torch

import template.datasets.utils as utils


@DATASETS.register_module()
class MPIIFaceGaze(Dataset):
  def __init__(self, root, train=True, test_pp='p00', transform=None):
    '''MPIIFaceGaze Dataset.

    `root`: root directory of dataset where prepared data for each person
    is stored, eg. 'data/mpiifacegaze/normalized-gen'.

    `train`: load data for training, otherwise for testing.

    `test_pp`: person ID for Leave-One-Out test, eg. 'p00'.

    `transform`: image transformation.
    '''

    person_indices = [f'p{i:02d}' for i in range(15)]
    if not test_pp in person_indices:
      raise RuntimeError(f'Person ID {test_pp} not in range "p00" - "p14".')
    person_indices.remove(test_pp)

    if train:
      self.data = ConcatDataset([
        _MPIIFaceGaze_PP(
          root=osp.join(root, train_pp),
          transform=transform,
        ) for train_pp in person_indices
      ])
    else:
      self.data = _MPIIFaceGaze_PP(
        root=osp.join(root, test_pp),
        transform=transform,
      )

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]


class _MPIIFaceGaze_PP(Dataset):
  def __init__(self, root, transform=None):
    '''Load data for one person in MPIIFaceGaze dataset.

    `root`: root directory of data for one person, eg. 'data/mpiifacegaze/normalized-gen/p00'.

    `transform`: image transformation.
    '''

    self.n_samples = self._load_data(root)
    self.transform = self._build_transform(transform)

  def _load_data(self, root):
    self.dates = sorted(os.listdir(root))

    self.f_gaze = [np.load(osp.join(root, dd, 'f_gaze.npy')) for dd in self.dates]
    self.f_pose = [np.load(osp.join(root, dd, 'f_pose.npy')) for dd in self.dates]
    for dd, f_gaze, f_pose in zip(self.dates, self.f_gaze, self.f_pose):
      if not len(f_gaze) == len(f_pose):
        raise RuntimeError(f'Inconsistent number of labels for {dd} in "{root}".')

    n_samples_dd = [len(x) for x in self.f_gaze]
    self.cumulative_sizes = np.cumsum(n_samples_dd).tolist()

    for dd in self.dates:
      if not osp.isdir(osp.join(root, dd, 'f_img')):
        raise RuntimeError(f'Face image folder for {dd} does not exist in "{root}".')

    self.root = root

    return self.cumulative_sizes[-1]

  def _build_transform(self, transform):
    if transform is None:
      transform = dict(type='ToTensor')

    if isinstance(transform, dict):
      transform = [transform]
    if isinstance(transform, (list, tuple)):
      transform = Compose(transform)

    return transform

  def __len__(self):
    return self.n_samples

  def __getitem__(self, idx):
    date_idx = bisect.bisect_right(self.cumulative_sizes, idx)
    if date_idx == 0:
      sample_idx = idx
    else:
      sample_idx = idx - self.cumulative_sizes[date_idx - 1]

    gaze = self.f_gaze[date_idx][sample_idx]
    img = cv2.imread(
      osp.join(self.root, self.dates[date_idx], 'f_img', f'{sample_idx:04d}.jpg'),
      cv2.IMREAD_UNCHANGED,
    )
    pose = self.f_pose[date_idx][sample_idx]

    gaze = utils.gaze_3d_2d_v(gaze)
    pose = utils.pose_3d_2d_v(pose)

    gaze = torch.tensor(gaze, dtype=torch.float32)
    pose = torch.tensor(pose, dtype=torch.float32)

    img = Image.fromarray(img, mode='RGB')
    if self.transform:
      img = self.transform(img)

    return dict(face=img, pose=pose), dict(gaze=gaze)
