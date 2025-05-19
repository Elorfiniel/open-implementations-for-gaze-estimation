from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

from opengaze.registry import DATASETS
from opengaze.utils.dataset import build_image_transform

import h5py
import numpy as np
import os.path as osp
import torch


@DATASETS.register_module()
class XGaze224(Dataset):
  def __init__(self, root: str, subjects: list, label=False, transform=None):
    '''XGaze dataset, with face patches preprocessed to 224x224.

    Args:
      `root`: root directory of the dataset, where hdf files
      (eg. subjectXXXX.h5) for each subject are stored.

      `subjects`: list of subject IDs (eg. '0000') to load.

      `label`: whether gaze labels should be loaded.

      `transform`: image transformation.
    '''

    super(XGaze224, self).__init__()
    self.data = ConcatDataset([
      XGaze224Subject(root, subject, label, transform)
      for subject in subjects
    ])

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]


@DATASETS.register_module()
class XGaze224Subject(Dataset):
  def __init__(self, root: str, subject: str, label=False, transform=None):
    '''XGaze dataset for each subject, with face patches preprocessed to 224x224.

    Args:
      `root`: root directory of the dataset, where hdf files
      (eg. subjectXXXX.h5) for each subject are stored.

      `subject`: subject ID (eg. '0000') to load.

      `label`: whether gaze labels should be loaded.

      `transform`: image transformation.
    '''

    super(XGaze224Subject, self).__init__()

    self.root = root
    self.subject = subject
    self.label = label

    hdf_path = osp.join(root, f'subject{subject}.h5')
    self.hdf = h5py.File(hdf_path, 'r', swmr=True)

    self.n_samples = self.hdf['frame_index'].size
    self.transform = build_image_transform(transform)

  def __len__(self):
    return self.n_samples

  def __getitem__(self, idx):
    fid = torch.tensor(self.hdf['frame_index'][idx], dtype=torch.int32)
    cid = torch.tensor(self.hdf['cam_index'][idx], dtype=torch.int32)

    rot = torch.tensor(self.hdf['face_mat_norm'][idx], dtype=torch.float32)

    pose_py = np.array(self.hdf['face_head_pose'][idx])
    pose = torch.tensor(pose_py, dtype=torch.float32)

    face_img = np.array(self.hdf['face_patch'][idx][:, :, ::-1])
    face = self.transform(Image.fromarray(face_img, 'RGB'))

    data_dict = dict(fid=fid, cid=cid, rot=rot, pose=pose, face=face)
    if self.label:
      gaze_py = np.array(self.hdf['face_gaze'][idx])
      gaze = torch.tensor(gaze_py, dtype=torch.float32)
      data_dict.update(gaze=gaze)

    return data_dict
