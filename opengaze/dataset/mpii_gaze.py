from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

from opengaze.registry import DATASETS
from opengaze.utils.dataset import build_image_transform, build_data_pipeline
from opengaze.utils.euler import gaze_3d_2d_a, pose_3d_2d_a

import cv2
import h5py
import numpy as np
import os
import os.path as osp
import torch as torch


@DATASETS.register_module()
class MPIIGaze(Dataset):
  def __init__(self, root, train=True, test_pp='p00', eval_subset=False,
               transform=None, pipeline=None):
    '''MPIIGaze Dataset.

    Args:
      `root`: root directory of dataset where prepared data for each person
      is stored, eg. 'data/mpiigaze'.

      `train`: load data for training, otherwise for testing.

      `test_pp`: person ID for Leave-One-Out test, eg. 'p00'.

      `eval_subset`: only use data from eval subset, which contains 3000 samples
      for each person in an accompanying folder of the root directory.

      `transform`: image transformation.

      `pipeline`: data processing pipeline.
    '''

    person_indices = [f'p{i:02d}' for i in range(15)]
    if not test_pp in person_indices:
      raise RuntimeError(f'Person ID {test_pp} not in range "p00" - "p14".')
    person_indices.remove(test_pp)

    if train:
      self.data = ConcatDataset([
        _MPIIGazePP(
          root=osp.join(root, 'normalize', train_pp),
          eval_subset=eval_subset,
          transform=transform,
        ) for train_pp in person_indices
      ])
    else:
      self.data = _MPIIGazePP(
        root=osp.join(root, 'normalize', test_pp),
        eval_subset=eval_subset,
        transform=transform,
      )

    self.pipeline = build_data_pipeline(pipeline)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.pipeline(self.data[idx])


class _MPIIGazePP(Dataset):
  def __init__(self, root, eval_subset=False, transform=None):
    '''Load data for one person in MPIIGaze dataset.

    `root`: root directory of data for one person, eg. 'data/mpiigaze/normalize/p00'.

    `eval_subset`: only use data from eval subset. See also `MPIIGaze`.

    `transform`: image transformation.
    '''

    self.n_samples = self._load_data(root, eval_subset)
    self.transform = build_image_transform(transform)

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

    self.data = {
      'leye-img': [],
      'leye-gaze': [],
      'leye-pose': [],
      'reye-img': [],
      'reye-gaze': [],
      'reye-pose': [],
    } # Load all data in memory (~1GB)

    for dd in dates:
      with h5py.File(osp.join(root, dd, 'annot.h5'), 'r', swmr=True) as hdf_file:
        for key in self.data.keys():
          value = np.array(hdf_file[key])
          if eval_subset:
            x_eval = l_eval if 'l' in key else r_eval
            value = value[x_eval.get(dd, [])]
          if value.size > 0:
            self.data[key].append(value)

    for key, value in self.data.items():
      self.data[key] = np.concatenate(value, axis=0)

    return len(self.data['leye-img']) + len(self.data['reye-img'])

  def __len__(self):
    return self.n_samples

  def __getitem__(self, idx):
    is_left, real_idx = not bool(idx & 1), idx // 2

    prefix = 'leye' if is_left else 'reye'

    eyes = self.data[f'{prefix}-img'][real_idx]
    gaze = self.data[f'{prefix}-gaze'][real_idx]
    pose = self.data[f'{prefix}-pose'][real_idx]

    gp, gy = gaze_3d_2d_a(gaze[0], gaze[1], gaze[2])
    gaze = np.array([gp, gy], dtype=np.float32)
    pp, py = pose_3d_2d_a(pose[0], pose[1], pose[2])
    pose = np.array([pp, py], dtype=np.float32)

    if not is_left:
      # Mirror reflection: w.r.t. the XoZ plane (or y axis)
      #   For eye image, it's equivalent to horizontal flip
      #   For gaze and pose, it's equivalent to negate yaw
      eyes = cv2.flip(eyes, flipCode=1)
      gaze[1], pose[1] = -gaze[1], -pose[1]

    gaze = torch.tensor(gaze, dtype=torch.float32)
    pose = torch.tensor(pose, dtype=torch.float32)

    eyes = Image.fromarray(eyes, mode='L')
    if self.transform:
      eyes = self.transform(eyes)

    return dict(eyes=eyes, pose=pose, gaze=gaze)
