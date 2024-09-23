from torch.utils.data import Dataset, ConcatDataset

from template.registry import DATASETS

import cv2
import numpy as np
import os
import os.path as osp
import torch as torch

import template.datasets.utils as utils


@DATASETS.register_module()
class MPIIGaze(Dataset):
  def __init__(self, root, train, test_pp, transform=None):
    '''MPIIGaze Dataset.

    `root`: Root directory of dataset where prepared data for each person
    is stored, eg. 'data/mpiigaze/normalized-ext'.

    `train`: load data for training, otherwise for testing.

    `test_pp`: Person ID for Leave-One-Out test, eg. 'p00'.

    `transform`: Image transformation.
    '''

    person_indices = [f'p{i:02d}' for i in range(15)]
    if not test_pp in person_indices:
      raise RuntimeError(f'Person ID {test_pp} not in range "p00" - "p14".')
    person_indices.remove(test_pp)

    if train:
      self.data = ConcatDataset([
        _MPIIGaze_PP(
          root=osp.join(root, train_pp),
          transform=transform,
        ) for train_pp in person_indices
      ])
    else:
      self.data = _MPIIGaze_PP(
        root=osp.join(root, test_pp),
        transform=transform,
      )

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]


class _MPIIGaze_PP(Dataset):
  def __init__(self, root, transform=None):
    '''Load data for one person in MPIIGaze dataset.

    `root`: Root directory of data for one person, eg. 'data/mpiigaze/normalized-ext/p00'.

    `transform`: Image transformation.
    '''

    self.n_samples = self._load_data(root)
    self.transform = transform

  def _load_data(self, root):
    dates = sorted(os.listdir(root))

    attrs = ['l_gaze', 'l_img', 'l_pose', 'r_gaze', 'r_img', 'r_pose']
    for attr in attrs:
      attr_value = np.concatenate([
        np.load(osp.join(root, dd, f'{attr}.npy'))
        for dd in dates
      ], axis=0)
      setattr(self, attr, attr_value)

    return len(self.l_img) + len(self.r_img)

  def __len__(self):
    return self.n_samples

  def __getitem__(self, idx):
    is_left, real_idx = not bool(idx & 1), idx // 2

    gaze = (self.l_gaze if is_left else self.r_gaze)[real_idx]
    img = (self.l_img if is_left else self.r_img)[real_idx]
    pose = (self.l_pose if is_left else self.r_pose)[real_idx]

    gaze = utils.gaze_3d_2d_v(gaze)
    pose = utils.pose_3d_2d_v(pose)

    if not is_left:
      # mirror reflection: w.r.t. the XoZ plane (or y axis)
      #   for eye image, it's equivalent to horizontal flip
      #   for gaze and pose, it's equivalent to negate yaw
      img = cv2.flip(img, flipCode=1)
      gaze[1], pose[1] = -gaze[1], -pose[1]

    gaze = torch.tensor(gaze, dtype=torch.float32)
    pose = torch.tensor(pose, dtype=torch.float32)

    img = torch.tensor(img, dtype=torch.float32)
    if self.transform: img = self.transform(img)

    return dict(eyes=img, pose=pose), dict(gaze=gaze)
