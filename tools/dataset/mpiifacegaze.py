from template.runtime.files import ProjectTree
from template.runtime.log import runtime_logger

from mpiigaze import mpii_data_normalization

import argparse
import cv2
import numpy as np
import os
import os.path as osp
import scipy.io as sio
import shutil


_logger = runtime_logger('mpiifacegaze')


def load_mpii_face_model(mpii_model_path):
  face_model = sio.loadmat(mpii_model_path)['model'].T
  face_model = face_model.reshape((6, 3)).astype(np.float32)
  return face_model

def estimate_head_pose(mpii_model, landmarks, camera_mat, camera_dist):
  _, rvec, tvec = cv2.solvePnP(mpii_model, landmarks, camera_mat, camera_dist, flags=cv2.SOLVEPNP_EPNP)
  _, rvec, tvec = cv2.solvePnP(mpii_model, landmarks, camera_mat, camera_dist, rvec, tvec, True)
  return rvec, tvec


def _load_annot_pp_dd(date_folder, **kwargs):
  label_file_all_dates = np.loadtxt(
    fname=osp.join(osp.dirname(date_folder), f'{kwargs["pp"]}.txt'),
    dtype=dict(
      names=(
        'img_path',
        'gx_px', 'gy_px',
        'lm01_x', 'lm01_y', # reye, outer
        'lm02_x', 'lm02_y', # reye, inner
        'lm03_x', 'lm03_y', # leye, inner
        'lm04_x', 'lm04_y', # leye, outer
        'lm05_x', 'lm05_y', # mouth, rcorner
        'lm06_x', 'lm06_y', # mouth, lcorner
        'rvec_x', 'rvec_y', 'rvec_z',
        'tvec_x', 'tvec_y', 'tvec_z',
        'fc_x', 'fc_y', 'fc_z',
        'gx_mm', 'gy_mm', 'gz_mm',
        'eval',
      ),
      formats=(
        'U32',
        'i4', 'i4',
        'i4', 'i4',
        'i4', 'i4',
        'i4', 'i4',
        'i4', 'i4',
        'i4', 'i4',
        'i4', 'i4',
        'f4', 'f4', 'f4',
        'f4', 'f4', 'f4',
        'f4', 'f4', 'f4',
        'f4', 'f4', 'f4',
        'U8',
      ),
    ),
    delimiter=' ',
    ndmin=1,
  )
  label_mask = list(map(lambda x: kwargs['dd'] in x[0], label_file_all_dates))

  label_file = label_file_all_dates[label_mask]
  for label in label_file:
    label['img_path'] = osp.basename(label['img_path'])

  return label_file

def _gen_normalized_pp_dd(cam_data, scn_pose, scn_size, date_folder, label_file, pp_dd_folder):
  os.makedirs(pp_dd_folder, exist_ok=True)

  cam_mat, cam_dist = cam_data['cameraMatrix'], cam_data['distCoeffs']
  mr, mt = scn_pose['rvects'], scn_pose['tvecs']
  screen_h_mm = scn_size['height_mm'][0, 0]
  screen_h_px = scn_size['height_pixel'][0, 0]
  screen_w_mm = scn_size['width_mm'][0, 0]
  screen_w_px = scn_size['width_pixel'][0, 0]

  mpii_model_path = ProjectTree.resource_path('face-models/mpiigaze-generic.mat')
  face_model = load_mpii_face_model(mpii_model_path)

  l_gaze, l_img, l_pose = [], [], []
  r_gaze, r_img, r_pose = [], [], []

  f_gaze, f_pose = [], []
  f_img_folder = osp.join(pp_dd_folder, 'f_img')
  os.makedirs(f_img_folder, exist_ok=True)

  for sample_idx in range(len(label_file)):
    sample = label_file[sample_idx] # a numpy structured array

    img = cv2.imread(osp.join(date_folder, sample['img_path']), cv2.IMREAD_UNCHANGED)
    pog = np.array(sample[['gx_px', 'gy_px']].tolist(), dtype=np.float32)

    tgt_scn = np.array([
      pog[0] / screen_w_px * screen_w_mm,
      pog[1] / screen_h_px * screen_h_mm,
      0.0,
    ], dtype=np.float32)
    mR = cv2.Rodrigues(mr)[0]
    tgt_cam = np.dot(mR, tgt_scn) + mt.reshape((3, ))

    landmarks_2d = np.array(sample[[
      'lm01_x', 'lm01_y', # reye, outer
      'lm02_x', 'lm02_y', # reye, inner
      'lm03_x', 'lm03_y', # leye, inner
      'lm04_x', 'lm04_y', # leye, outer
      'lm05_x', 'lm05_y', # mouth, rcorner
      'lm06_x', 'lm06_y', # mouth, lcorner
    ]].tolist(), dtype=np.float32).reshape((6, 2))
    hr, ht = estimate_head_pose(face_model, landmarks_2d, cam_mat, cam_dist)
    hR = cv2.Rodrigues(hr)[0]

    landmarks_3d = np.dot(hR, face_model.T) + ht
    re = 0.5 * (landmarks_3d[:, 0] + landmarks_3d[:, 1])
    le = 0.5 * (landmarks_3d[:, 2] + landmarks_3d[:, 3])
    fe = np.average(landmarks_3d, axis=1)

    leye_result = mpii_data_normalization(img, le, 960, 600, (60, 36), hR, cam_mat)
    warp_l, Kv_l, S_l, R2_l, W_l = leye_result
    reye_result = mpii_data_normalization(img, re, 960, 600, (60, 36), hR, cam_mat)
    warp_r, Kv_r, S_r, R2_r, W_r = reye_result

    face_result = mpii_data_normalization(img, fe, 960, 300, (448, 448), hR, cam_mat)
    warp_f, Kv_f, S_f, R2_f, W_f = face_result

    # [Revisiting Data Normalization]
    #   Discard the scaling component `S_x` when calculating gaze and pose vector
    leye_gaze = np.dot(np.dot(S_l, R2_l), tgt_cam - le)
    leye_gaze = leye_gaze / np.linalg.norm(leye_gaze)
    reye_gaze = np.dot(np.dot(S_r, R2_r), tgt_cam - re)
    reye_gaze = reye_gaze / np.linalg.norm(reye_gaze)

    face_gaze = np.dot(np.dot(S_f, R2_f), tgt_cam - fe)
    face_gaze = face_gaze / np.linalg.norm(face_gaze)

    leye_img = cv2.equalizeHist(cv2.cvtColor(warp_l, cv2.COLOR_BGR2GRAY))
    reye_img = cv2.equalizeHist(cv2.cvtColor(warp_r, cv2.COLOR_BGR2GRAY))

    cv2.imwrite(
      osp.join(f_img_folder, f'{sample_idx:04d}.jpg'),
      warp_f,
      [cv2.IMWRITE_JPEG_QUALITY, 100],
    )

    leye_pose = np.dot(np.dot(S_l, R2_l), hR)
    leye_pose = cv2.Rodrigues(leye_pose)[0].reshape((3, ))
    reye_pose = np.dot(np.dot(S_r, R2_r), hR)
    reye_pose = cv2.Rodrigues(reye_pose)[0].reshape((3, ))

    face_pose = np.dot(np.dot(S_f, R2_f), hR)
    face_pose = cv2.Rodrigues(face_pose)[0].reshape((3, ))

    l_gaze.append(leye_gaze), l_img.append(leye_img), l_pose.append(leye_pose)
    r_gaze.append(reye_gaze), r_img.append(reye_img), r_pose.append(reye_pose)

    f_gaze.append(face_gaze), f_pose.append(face_pose)

  l_gaze = np.stack(l_gaze, axis=0).astype(np.float32)
  l_img = np.stack(l_img, axis=0).astype(np.uint8)
  l_pose = np.stack(l_pose, axis=0).astype(np.float32)
  r_gaze = np.stack(r_gaze, axis=0).astype(np.float32)
  r_img = np.stack(r_img, axis=0).astype(np.uint8)
  r_pose = np.stack(r_pose, axis=0).astype(np.float32)

  f_gaze = np.stack(f_gaze, axis=0).astype(np.float32)
  f_pose = np.stack(f_pose, axis=0).astype(np.float32)

  for filename, ndarray in zip(
    ['l_gaze.npy', 'l_img.npy', 'l_pose.npy', 'r_gaze.npy', 'r_img.npy', 'r_pose.npy'],
    [l_gaze, l_img, l_pose, r_gaze, r_img, r_pose],
    # [(N, 3), (N, 36, 60), (N, 3), (N, 3), (N, 36, 60), (N, 3)]
  ): np.save(osp.join(pp_dd_folder, filename), ndarray)

  for filename, ndarray in zip(
    ['f_gaze.npy', 'f_pose.npy'],
    [f_gaze, f_pose],
    # [(N, 3), (N, 3)]
  ): np.save(osp.join(pp_dd_folder, filename), ndarray)

def gen_normalized_data(dataset_path: str, opt_folder: str):
  _logger.info(f'generate normalized data for MPIIFaceGaze dataset')

  persons_folder = dataset_path
  persons = [
    f for f in os.listdir(persons_folder)
    if osp.isdir(osp.join(persons_folder, f))
  ]

  for pid, person in enumerate(persons):
    person_folder = osp.join(persons_folder, person)
    calib_folder = osp.join(person_folder, 'Calibration')
    dates_folders = os.listdir(person_folder)
    for filename in ['Calibration', f'{person}.txt']:
      dates_folders.remove(filename)

    cam_data = sio.loadmat(osp.join(calib_folder, 'Camera.mat'))
    scn_pose = sio.loadmat(osp.join(calib_folder, 'monitorPose.mat'))
    scn_size = sio.loadmat(osp.join(calib_folder, 'screenSize.mat'))

    _logger.info(f'process data in "{person_folder}"')
    for df in dates_folders:
      _logger.info(f'generate normalized data for ({person}, {df})')
      _gen_normalized_pp_dd(
        cam_data, scn_pose, scn_size,
        date_folder=osp.join(person_folder, df),
        label_file=_load_annot_pp_dd(
          osp.join(person_folder, df), pp=person, dd=df,
        ),
        pp_dd_folder=osp.join(opt_folder, person, df),
      )

  _logger.info(f'generate normalized data: done')


def main_procedure(cmdargs: argparse.Namespace):
  dataset_path = osp.abspath(cmdargs.dataset_path)
  _logger.info(f'prepare data for MPIIFaceGaze dataset at "{dataset_path}"')

  data_folder = ProjectTree.data_path('mpiifacegaze')
  os.makedirs(data_folder, exist_ok=True)
  _logger.info(f'data will be stored in "{data_folder}"')

  try:
    # Note: `ext_normalized_data` not implemented due to the huge size of its output
    #
    # You can directly download the normalized data from the dataset website:
    #   http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze_normalized.zip
    #
    # The generated folder structure does not match that of the downloaded one
    # If you want to extract the downloaded data, please do manual conversion
    gen_normalized_data(dataset_path, osp.join(data_folder, 'normalized-gen'))
    # Note: no evaluation samples available
  except Exception as ex:
    _logger.error(f'failed to prepare data for MPIIFaceGaze dataset')
    _logger.error(ex)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Prepare data for MPIIFaceGaze dataset.')

  parser.add_argument(
    '--dataset-path', type=str, required=True,
    help='Path to the extracted MPIIFaceGaze dataset.',
  )

  main_procedure(parser.parse_args())
