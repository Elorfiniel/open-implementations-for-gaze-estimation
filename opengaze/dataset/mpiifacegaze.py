from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

from opengaze.registry import DATASETS
from opengaze.utils.dataset import build_image_transform, build_data_pipeline
from opengaze.utils.euler import gaze_3d_2d_a, pose_3d_2d_a

import bisect
import cv2
import h5py
import numpy as np
import os
import os.path as osp
import torch as torch


@DATASETS.register_module()
class MPIIFaceGaze(Dataset):
  def __init__(self, root, train=True, test_pp='p00',
               transform=None, pipeline=None):
    '''MPIIFaceGaze Dataset.

    Args:
      `root`: root directory of dataset where prepared data for each person
      is stored, eg. 'data/mpiifacegaze'.

      `train`: load data for training, otherwise for testing.

      `test_pp`: person ID for Leave-One-Out test, eg. 'p00'.

      `transform`: image transformation.

      `pipeline`: data processing pipeline.
    '''

    person_indices = [f'p{i:02d}' for i in range(15)]
    if not test_pp in person_indices:
      raise RuntimeError(f'Person ID {test_pp} not in range "p00" - "p14".')
    person_indices.remove(test_pp)

    if train:
      self.data = ConcatDataset([
        _MPIIFaceGazePP(
          root=osp.join(root, train_pp),
          transform=transform,
        ) for train_pp in person_indices
      ])
    else:
      self.data = _MPIIFaceGazePP(
        root=osp.join(root, test_pp),
        transform=transform,
      )

    self.pipeline = build_data_pipeline(pipeline)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.pipeline(self.data[idx])


class _MPIIFaceGazePP(Dataset):
  def __init__(self, root, transform=None):
    '''Load data for one person in MPIIFaceGaze dataset.

    `root`: root directory of data for one person, eg. 'data/mpiifacegaze/normalize/p00'.

    `transform`: image transformation.
    '''

    self.n_samples = self._load_data(root)
    self.transform = build_image_transform(transform)

  def _load_data(self, root):
    self.dates = sorted([
      item
      for item in os.listdir(root)
      if osp.isdir(osp.join(root, item))
    ])

    self.data = {
      'face-gaze': [],
      'face-pose': [],
    } # Load all data except images in memory

    for dd in self.dates:
      with h5py.File(osp.join(root, f'{dd}.h5'), 'r', swmr=True) as hdf_file:
        for key in self.data.keys():
          value = np.array(hdf_file[key])
          self.data[key].append(value)

    n_samples_dd = [len(x) for x in self.data['face-gaze']]
    self.cumulative_sizes = np.cumsum(n_samples_dd).tolist()

    self.root = root

    return self.cumulative_sizes[-1]

  def _load_crop(self, image_path):
    crop = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    crop = Image.fromarray(crop, mode='RGB')
    if self.transform:
      crop = self.transform(crop)
    return crop

  def __len__(self):
    return self.n_samples

  def __getitem__(self, idx):
    date_idx = bisect.bisect_right(self.cumulative_sizes, idx)
    if date_idx == 0:
      sample_idx = idx
    else:
      sample_idx = idx - self.cumulative_sizes[date_idx - 1]

    face = self._load_crop(
      osp.join(self.root, self.dates[date_idx], f'{sample_idx:04d}.jpg'),
    )
    gaze = self.data['face-gaze'][date_idx][sample_idx]
    pose = self.data['face-pose'][date_idx][sample_idx]

    gp, gy = gaze_3d_2d_a(gaze[0], gaze[1], gaze[2])
    gaze = np.array([gp, gy], dtype=np.float32)
    pp, py = pose_3d_2d_a(pose[0], pose[1], pose[2])
    pose = np.array([pp, py], dtype=np.float32)

    gaze = torch.tensor(gaze, dtype=torch.float32)
    pose = torch.tensor(pose, dtype=torch.float32)

    return dict(face=face, pose=pose, gaze=gaze)
