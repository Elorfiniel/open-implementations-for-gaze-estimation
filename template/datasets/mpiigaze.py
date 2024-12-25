from mmengine.dataset import Compose
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

from template.registry import DATASETS
from template.datasets import utils

import cv2
import numpy as np
import os
import os.path as osp
import torch as torch


@DATASETS.register_module()
class MPIIGaze(Dataset):
  def __init__(self, root, train=True, test_pp='p00', eval_subset=False, transform=None):
    '''MPIIGaze Dataset.

    `root`: root directory of dataset where prepared data for each person
    is stored, eg. 'data/mpiigaze/normalized-ext'.

    `train`: load data for training, otherwise for testing.

    `test_pp`: person ID for Leave-One-Out test, eg. 'p00'.

    `eval_subset`: only use data from eval subset, which contains 3000 samples
    for each person in an accompanying folder of the root directory.

    `transform`: image transformation.
    '''

    person_indices = [f'p{i:02d}' for i in range(15)]
    if not test_pp in person_indices:
      raise RuntimeError(f'Person ID {test_pp} not in range "p00" - "p14".')
    person_indices.remove(test_pp)

    if train:
      self.data = ConcatDataset([
        _MPIIGaze_PP(
          root=osp.join(root, train_pp),
          eval_subset=eval_subset,
          transform=transform,
        ) for train_pp in person_indices
      ])
    else:
      self.data = _MPIIGaze_PP(
        root=osp.join(root, test_pp),
        eval_subset=eval_subset,
        transform=transform,
      )

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]


class _MPIIGaze_PP(Dataset):
  def __init__(self, root, eval_subset=False, transform=None):
    '''Load data for one person in MPIIGaze dataset.

    `root`: root directory of data for one person, eg. 'data/mpiigaze/normalized-ext/p00'.

    `eval_subset`: only use data from eval subset. See also `MPIIGaze`.

    `transform`: image transformation.
    '''

    self.n_samples = self._load_data(root, eval_subset)
    self.transform = self._build_transform(transform)

  def _parse_eval_lines(self, lines):
    l_eval, r_eval = dict(), dict() # left vs. right

    for line in lines:
      file_path, side = line.split(' ')
      dd, img_path = osp.split(file_path)
      cnt = int(osp.splitext(img_path)[0])

      x_eval = l_eval if 'l' in side else r_eval
      if dd not in x_eval:
        x_eval[dd] = [cnt - 1]
      else:
        x_eval[dd].append(cnt - 1)

    return l_eval, r_eval

  def _load_data(self, root, eval_subset):
    if eval_subset:
      eval_root = osp.join(osp.dirname(osp.dirname(root)), 'evaluation')
      if not osp.exists(eval_root):
        raise RuntimeError(f'No evaluation subset found in "{root}".')

      eval_file = osp.join(eval_root, f'{osp.basename(root)}.txt')
      with open(eval_file, 'r') as fp:
        lines = [line.strip() for line in fp]
        l_eval, r_eval = self._parse_eval_lines(lines)

    dates = sorted(os.listdir(root))

    attrs = ['l_gaze', 'l_img', 'l_pose', 'r_gaze', 'r_img', 'r_pose']
    for attr in attrs:
      attr_value = [] # a list of loaded ndarrays
      for dd in dates:
        data = np.load(osp.join(root, dd, f'{attr}.npy'))
        if eval_subset:
          x_eval = l_eval if 'l' in attr else r_eval
          data = data[x_eval.get(dd, [])]
        if data.size > 0:
          attr_value.append(data)

      attr_value = np.concatenate(attr_value, axis=0)
      setattr(self, attr, attr_value)

    return len(self.l_img) + len(self.r_img)

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

    img = Image.fromarray(img, mode='L')
    if self.transform:
      img = self.transform(img)

    return dict(eyes=img, pose=pose), dict(gaze=gaze)
