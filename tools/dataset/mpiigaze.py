from template.runtime.files import ProjectTree
from template.runtime.log import runtime_logger

import argparse
import cv2
import numpy as np
import os
import os.path as osp
import scipy.io as sio
import shutil


_logger = runtime_logger('mpiigaze')


def mpii_data_normalization(image, center, focal_norm, dist_norm, crop_norm, R1, Kc):
  # Configure normalized camera coordinate frame
  distance = np.linalg.norm(center)
  scaling = dist_norm / distance
  Kv = np.array([
    [focal_norm, 0.0, crop_norm[0] / 2],
    [0.0, focal_norm, crop_norm[1] / 2],
    [0.0, 0.0, 1.0],
  ])
  S = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, scaling],
  ])

  # Find axes of normalized camera coordinate frame in camera coordinate frame
  z_axis = center / np.linalg.norm(center)
  x_axis_head = R1[:, 0]
  y_axis = np.cross(z_axis, x_axis_head)
  y_axis = y_axis / np.linalg.norm(y_axis)
  x_axis = np.cross(y_axis, z_axis)
  x_axis = x_axis / np.linalg.norm(x_axis)
  R2 = np.vstack([x_axis, y_axis, z_axis])

  # Calculate perspective transformation (for image warpping)
  W = np.dot(np.dot(Kv, S), np.dot(R2, np.linalg.inv(Kc)))
  warp = cv2.warpPerspective(image, W, crop_norm)

  return warp, Kv, S, R2, W


def _ext_normalized_pp_dd(mat, pp_dd_folder):
  os.makedirs(pp_dd_folder, exist_ok=True)

  mat_data, mat_file = mat['data'], mat['filenames']

  l_gaze = mat_data['left'][0, 0]['gaze'][0, 0].astype(np.float32)
  l_img = mat_data['left'][0, 0]['image'][0, 0].astype(np.uint8)
  l_pose = mat_data['left'][0, 0]['pose'][0, 0].astype(np.float32)
  r_gaze = mat_data['right'][0, 0]['gaze'][0, 0].astype(np.float32)
  r_img = mat_data['right'][0, 0]['image'][0, 0].astype(np.uint8)
  r_pose = mat_data['right'][0, 0]['pose'][0, 0].astype(np.float32)

  for filename, ndarray in zip(
    ['l_gaze.npy', 'l_img.npy', 'l_pose.npy', 'r_gaze.npy', 'r_img.npy', 'r_pose.npy'],
    [l_gaze, l_img, l_pose, r_gaze, r_img, r_pose],
    # [(N, 3), (N, 36, 60), (N, 3), (N, 3), (N, 36, 60), (N, 3)]
  ): np.save(osp.join(pp_dd_folder, filename), ndarray)

def ext_normalized_data(dataset_path: str, opt_folder: str):
  _logger.info(f'extract normalized data from MPIIGaze dataset')

  persons_folder = osp.join(dataset_path, 'Data', 'Normalized')
  persons = os.listdir(persons_folder)

  for pid, person in enumerate(persons):
    person_folder = osp.join(persons_folder, person)
    normalized_files = os.listdir(person_folder)

    _logger.info(f'process data in "{person_folder}"')
    for nf in normalized_files:
      _logger.info(f'extract normalized data for ({person}, {osp.splitext(nf)[0]})')
      _ext_normalized_pp_dd(
        mat=sio.loadmat(osp.join(person_folder, nf)),
        pp_dd_folder=osp.join(opt_folder, person, osp.splitext(nf)[0]),
      )

  _logger.info(f'extract normalized data: done')


def _load_annot_pp_dd(date_folder, **kwargs):
  label_file = np.loadtxt(
    fname=osp.join(date_folder, 'annotation.txt'),
    dtype=dict(
      names=(
        'lm01_x', 'lm01_y',
        'lm02_x', 'lm02_y',
        'lm03_x', 'lm03_y',
        'lm04_x', 'lm04_y',
        'lm05_x', 'lm05_y',
        'lm06_x', 'lm06_y',
        'lm07_x', 'lm07_y',
        'lm08_x', 'lm08_y',
        'lm09_x', 'lm09_y',
        'lm10_x', 'lm10_y',
        'lm11_x', 'lm11_y',
        'lm12_x', 'lm12_y',
        'gx_px', 'gy_px',
        'gx_mm', 'gy_mm', 'gz_mm',
        'rvec_x', 'rvec_y', 'rvec_z',
        'tvec_x', 'tvec_y', 'tvec_z',
        're_x', 're_y', 're_z',
        'le_x', 'le_y', 'le_z',
      ),
      formats=(
        'i4', 'i4',
        'i4', 'i4',
        'i4', 'i4',
        'i4', 'i4',
        'i4', 'i4',
        'i4', 'i4',
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
        'f4', 'f4', 'f4',
      ),
    ),
    delimiter=' ',
    ndmin=1,
  )

  return label_file

def _gen_normalized_pp_dd(cam_data, scn_pose, scn_size, date_folder, label_file, pp_dd_folder):
  os.makedirs(pp_dd_folder, exist_ok=True)

  cam_mat, cam_dist = cam_data['cameraMatrix'], cam_data['distCoeffs']
  mr, mt = scn_pose['rvects'], scn_pose['tvecs']
  screen_h_mm = scn_size['height_mm'][0, 0]
  screen_h_px = scn_size['height_pixel'][0, 0]
  screen_w_mm = scn_size['width_mm'][0, 0]
  screen_w_px = scn_size['width_pixel'][0, 0]

  l_gaze, l_img, l_pose = [], [], []
  r_gaze, r_img, r_pose = [], [], []

  for sample_idx in range(len(label_file)):
    sample = label_file[sample_idx] # a numpy structured array

    img = cv2.imread(osp.join(date_folder, f'{sample_idx+1:04d}.jpg'), cv2.IMREAD_UNCHANGED)
    pog = np.array(sample[['gx_px', 'gy_px']].tolist(), dtype=np.float32)

    tgt_scn = np.array([
      pog[0] / screen_w_px * screen_w_mm,
      pog[1] / screen_h_px * screen_h_mm,
      0.0,
    ], dtype=np.float32)
    mR = cv2.Rodrigues(mr)[0]
    tgt_cam = np.dot(mR, tgt_scn) + mt.reshape((3, ))

    hr = np.array(sample[['rvec_x', 'rvec_y', 'rvec_z']].tolist(), dtype=np.float32)
    hR = cv2.Rodrigues(hr.reshape((3, 1)))[0]
    le = np.array(sample[['le_x', 'le_y', 'le_z']].tolist(), dtype=np.float32)
    re = np.array(sample[['re_x', 're_y', 're_z']].tolist(), dtype=np.float32)

    leye_result = mpii_data_normalization(img, le, 960, 600, (60, 36), hR, cam_mat)
    warp_l, Kv_l, S_l, R2_l, W_l = leye_result
    reye_result = mpii_data_normalization(img, re, 960, 600, (60, 36), hR, cam_mat)
    warp_r, Kv_r, S_r, R2_r, W_r = reye_result

    # [Revisiting Data Normalization]
    #   Discard the scaling component `S_x` when calculating gaze and pose vector
    leye_gaze = np.dot(np.dot(S_l, R2_l), tgt_cam - le)
    leye_gaze = leye_gaze / np.linalg.norm(leye_gaze)
    reye_gaze = np.dot(np.dot(S_r, R2_r), tgt_cam - re)
    reye_gaze = reye_gaze / np.linalg.norm(reye_gaze)

    leye_img = cv2.equalizeHist(cv2.cvtColor(warp_l, cv2.COLOR_BGR2GRAY))
    reye_img = cv2.equalizeHist(cv2.cvtColor(warp_r, cv2.COLOR_BGR2GRAY))

    leye_pose = np.dot(np.dot(S_l, R2_l), hR)
    leye_pose = cv2.Rodrigues(leye_pose)[0].reshape((3, ))
    reye_pose = np.dot(np.dot(S_r, R2_r), hR)
    reye_pose = cv2.Rodrigues(reye_pose)[0].reshape((3, ))

    l_gaze.append(leye_gaze), l_img.append(leye_img), l_pose.append(leye_pose)
    r_gaze.append(reye_gaze), r_img.append(reye_img), r_pose.append(reye_pose)

  l_gaze = np.stack(l_gaze, axis=0).astype(np.float32)
  l_img = np.stack(l_img, axis=0).astype(np.uint8)
  l_pose = np.stack(l_pose, axis=0).astype(np.float32)
  r_gaze = np.stack(r_gaze, axis=0).astype(np.float32)
  r_img = np.stack(r_img, axis=0).astype(np.uint8)
  r_pose = np.stack(r_pose, axis=0).astype(np.float32)

  for filename, ndarray in zip(
    ['l_gaze.npy', 'l_img.npy', 'l_pose.npy', 'r_gaze.npy', 'r_img.npy', 'r_pose.npy'],
    [l_gaze, l_img, l_pose, r_gaze, r_img, r_pose],
    # [(N, 3), (N, 36, 60), (N, 3), (N, 3), (N, 36, 60), (N, 3)]
  ): np.save(osp.join(pp_dd_folder, filename), ndarray)

def gen_normalized_data(dataset_path: str, opt_folder: str):
  _logger.info(f'generate normalized data for MPIIGaze dataset')

  persons_folder = osp.join(dataset_path, 'Data', 'Original')
  persons = os.listdir(persons_folder)

  for pid, person in enumerate(persons):
    person_folder = osp.join(persons_folder, person)
    calib_folder = osp.join(person_folder, 'Calibration')
    dates_folders = os.listdir(person_folder)
    for filename in ['Calibration']:
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


def copy_evaluation_samples(dataset_path, opt_folder):
  _logger.info(f'copy annotations of 3000 evaluation samples')

  eval_folder = osp.join(dataset_path, 'Evaluation Subset', 'sample list for eye image')
  shutil.copytree(eval_folder, opt_folder, dirs_exist_ok=True)

  _logger.info(f'copy annotations: done')


def main_procedure(cmdargs: argparse.Namespace):
  dataset_path = osp.abspath(cmdargs.dataset_path)
  _logger.info(f'prepare data for MPIIGaze dataset at "{dataset_path}"')

  data_folder = ProjectTree.data_path('mpiigaze')
  os.makedirs(data_folder, exist_ok=True)
  _logger.info(f'data will be stored in "{data_folder}"')

  try:
    ext_normalized_data(dataset_path, osp.join(data_folder, 'normalized-ext'))
    gen_normalized_data(dataset_path, osp.join(data_folder, 'normalized-gen'))
    copy_evaluation_samples(dataset_path, osp.join(data_folder, 'evaluation'))
  except Exception as ex:
    _logger.error(f'failed to prepare data for MPIIGaze dataset')
    _logger.error(ex)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Prepare data for MPIIGaze dataset.')

  parser.add_argument(
    '--dataset-path', type=str, required=True,
    help='Path to the extracted MPIIGaze dataset.',
  )

  main_procedure(parser.parse_args())
