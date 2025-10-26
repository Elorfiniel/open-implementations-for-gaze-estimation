from opengaze.utils import MpiiDataNormalizer
from opengaze.runtime.scripts import ScriptEnv
from opengaze.runtime.log import runtime_logger
from opengaze.runtime.parallel import FunctionalTask, run_parallel

import argparse
import concurrent.futures as futures
import cv2
import h5py
import numpy as np
import os
import os.path as osp
import scipy.io as sio
import shutil


rt_logger = runtime_logger(
  name='mpii-gaze',
  log_file=ScriptEnv.log_path('prepare-mpii-gaze.log'),
)


class Screen2Camera:
  def __init__(self, scn_pose: dict, scn_size: dict):
    self.mr = scn_pose['rvects']
    self.mt = scn_pose['tvecs']

    self.screen_h_mm = scn_size['height_mm'][0, 0]
    self.screen_h_px = scn_size['height_pixel'][0, 0]
    self.screen_w_mm = scn_size['width_mm'][0, 0]
    self.screen_w_px = scn_size['width_pixel'][0, 0]

  def load_target(self, pog: np.ndarray):
    tgt_scn = np.array([
      pog[0] / self.screen_w_px * self.screen_w_mm,
      pog[1] / self.screen_h_px * self.screen_h_mm,
      0.0,
    ], dtype=np.float32)
    mR = cv2.Rodrigues(self.mr)[0]
    tgt_cam = np.dot(mR, tgt_scn) + self.mt.reshape((3, ))

    return tgt_cam

class Camera2Normal:
  def __init__(self, cam_data: dict, normalizer: MpiiDataNormalizer):
    self.cam_mat = cam_data['cameraMatrix']
    self.normalizer = normalizer

  def _unit_vector(self, v: np.ndarray):
    return v / np.linalg.norm(v)

  def normalize_data(self, look_at, R1, image, tgt):
    Kv, S, R2, W = self.normalizer.normalize_matrices(look_at, R1, self.cam_mat)
    warp = self.normalizer.warp_image(image, W)

    # Revisiting Data Normalization: discard the scaling component `S_x`
    gaze = self._unit_vector(np.dot(R2, tgt - look_at))
    pose = self._unit_vector(cv2.Rodrigues(np.dot(R2, R1))[0].reshape((3, )))

    return warp, gaze, pose


def load_mpii_annot_pp_dd(date_folder, **kwargs):
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

def process_pp_dd(persons_folder, pp, dd, opt_folder):
  # Load calibration data, such as camera matrix, screen size, etc
  calib_folder = osp.join(persons_folder, pp, 'Calibration')
  cam_data = sio.loadmat(osp.join(calib_folder, 'Camera.mat'))
  scn_pose = sio.loadmat(osp.join(calib_folder, 'monitorPose.mat'))
  scn_size = sio.loadmat(osp.join(calib_folder, 'screenSize.mat'))

  # Load annotations for current person and date
  date_folder = osp.join(persons_folder, pp, dd)
  label_file = load_mpii_annot_pp_dd(date_folder)

  # Create output folder for current person
  person_opt_folder = osp.join(opt_folder, pp)
  os.makedirs(person_opt_folder, exist_ok=True)

  # Create hdf datasets
  hdf_file = h5py.File(osp.join(person_opt_folder, f'{dd}.h5'), 'w')
  leye_img = hdf_file.create_dataset(
    'leye-img', shape=(len(label_file), 36, 60),
    dtype=np.uint8, chunks=(1, 36, 60),
  )
  leye_gaze = hdf_file.create_dataset(
    'leye-gaze', shape=(len(label_file), 3),
    dtype=np.float32, chunks=(1, 3),
  )
  leye_pose = hdf_file.create_dataset(
    'leye-pose', shape=(len(label_file), 3),
    dtype=np.float32, chunks=(1, 3),
  )
  reye_img = hdf_file.create_dataset(
    'reye-img', shape=(len(label_file), 36, 60),
    dtype=np.uint8, chunks=(1, 36, 60),
  )
  reye_gaze = hdf_file.create_dataset(
    'reye-gaze', shape=(len(label_file), 3),
    dtype=np.float32, chunks=(1, 3),
  )
  reye_pose = hdf_file.create_dataset(
    'reye-pose', shape=(len(label_file), 3),
    dtype=np.float32, chunks=(1, 3),
  )

  # Process gaze data for current person and date
  normalizer = MpiiDataNormalizer(960, (60, 36), distance=600)
  scn2cam = Screen2Camera(scn_pose, scn_size)
  cam2nor = Camera2Normal(cam_data, normalizer)

  for idx in range(len(label_file)):
    sample = label_file[idx] # Numpy structured array

    img = cv2.imread(osp.join(date_folder, f'{idx+1:04d}.jpg'), cv2.IMREAD_UNCHANGED)

    pog = np.array(sample[['gx_px', 'gy_px']].tolist(), dtype=np.float32)
    tgt = scn2cam.load_target(pog)

    hr = np.array(sample[['rvec_x', 'rvec_y', 'rvec_z']].tolist(), dtype=np.float32)
    hR = cv2.Rodrigues(hr.reshape((3, 1)))[0]
    le = np.array(sample[['le_x', 'le_y', 'le_z']].tolist(), dtype=np.float32)
    re = np.array(sample[['re_x', 're_y', 're_z']].tolist(), dtype=np.float32)

    img_l, gaze_l, pose_l = cam2nor.normalize_data(le, hR, img, tgt)
    img_l = cv2.equalizeHist(cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY))
    leye_img[idx] = img_l; leye_gaze[idx] = gaze_l; leye_pose[idx] = pose_l

    img_r, gaze_r, pose_r = cam2nor.normalize_data(re, hR, img, tgt)
    img_r = cv2.equalizeHist(cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY))
    reye_img[idx] = img_r; reye_gaze[idx] = gaze_r; reye_pose[idx] = pose_r

  # Close hdf file
  hdf_file.close()
  # Log processing result
  rt_logger.info(f'processed: {pp}, {dd}, {len(label_file)} samples')

def process_tasks(dataset_path, opt_folder):
  persons_folder = osp.join(dataset_path, 'Data', 'Original')
  persons = os.listdir(persons_folder)

  functional_tasks = []

  for person in persons:
    person_folder = osp.join(persons_folder, person)
    dates_folders = os.listdir(person_folder)
    for filename in ['Calibration']:
      dates_folders.remove(filename)

    for date in dates_folders:
      args = (persons_folder, person, date, opt_folder)
      task = FunctionalTask(process_pp_dd, *args)
      functional_tasks.append(task)

  return functional_tasks

def copy_evaluation_samples(dataset_path, opt_folder):
  eval_folder = osp.join(dataset_path, 'Evaluation Subset', 'sample list for eye image')
  shutil.copytree(eval_folder, opt_folder, dirs_exist_ok=True)


def main_procedure(cmdargs: argparse.Namespace):
  dataset_path = osp.abspath(cmdargs.dataset_path)
  rt_logger.info(f'mpii-gaze dataset: "{dataset_path}"')

  data_folder = ScriptEnv.data_path('mpii-gaze')
  os.makedirs(data_folder, exist_ok=True)
  rt_logger.info(f'processed data: "{data_folder}"')

  tasks = process_tasks(dataset_path, osp.join(data_folder, 'normalize'))
  executor = futures.ProcessPoolExecutor(cmdargs.max_workers)
  run_parallel(executor, tasks, rt_logger)

  copy_evaluation_samples(dataset_path, osp.join(data_folder, 'evaluation'))



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='prepare data for MPIIGaze dataset.')

  parser.add_argument(
    '--dataset-path', type=str, required=True,
    help='Path to the extracted MPIIGaze dataset.',
  )
  parser.add_argument(
    '--max-workers', type=int, default=None,
    help='Maximum number of processes in the process pool.',
  )

  main_procedure(parser.parse_args())
