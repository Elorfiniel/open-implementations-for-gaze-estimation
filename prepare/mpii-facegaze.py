from opengaze.utils import MpiiDataNormalizer
from opengaze.utils.geom import PoseEstimator
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


rt_logger = runtime_logger(
  name='mpii-facegaze',
  log_file=ScriptEnv.log_path('prepare-mpii-facegaze.log'),
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
    pose = cv2.Rodrigues(np.dot(R2, R1))[0].reshape((3, ))

    return warp, gaze, pose

class HeadPoseEstimator(PoseEstimator):
  def __init__(self, cam_data: dict):
    super().__init__(cam_data['cameraMatrix'], cam_data['distCoeffs'])

    mpii_model_path = ScriptEnv.resource_path('face-models/mpiigaze-generic.mat')
    face_model = sio.loadmat(mpii_model_path)['model'].T
    self.face_model = face_model.reshape((6, 3)).astype(np.float32)

  def estimate(self, landmarks_2d: np.ndarray):
    return super().estimate(self.face_model, landmarks_2d)


def load_mpii_annot_pp_dd(date_folder, **kwargs):
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

def calculate_face_center(landmarks_3d: np.ndarray):
  # Use convention of ETH-XGaze data normalization

  eyes_center = np.mean(landmarks_3d[:, 0:4], axis=1)
  mouth_center = np.mean(landmarks_3d[:, 4:6], axis=1)
  face_center = 0.5 * (eyes_center + mouth_center)

  return face_center

def process_pp_dd(persons_folder, pp, dd, opt_folder):
  # Load calibration data, such as camera matrix, screen size, etc
  calib_folder = osp.join(persons_folder, pp, 'Calibration')
  cam_data = sio.loadmat(osp.join(calib_folder, 'Camera.mat'))
  scn_pose = sio.loadmat(osp.join(calib_folder, 'monitorPose.mat'))
  scn_size = sio.loadmat(osp.join(calib_folder, 'screenSize.mat'))

  # Load annotations for current person and date
  date_folder = osp.join(persons_folder, pp, dd)
  label_file = load_mpii_annot_pp_dd(date_folder, pp=pp, dd=dd)

  # Create output folder for current person and date
  pp_dd_opt_folder = osp.join(opt_folder, pp, dd)
  os.makedirs(pp_dd_opt_folder, exist_ok=True)

  # Create hdf datasets
  hdf_file = h5py.File(osp.join(pp_dd_opt_folder, f'annot.h5'), 'w')
  face_gaze = hdf_file.create_dataset(
    'face-gaze', shape=(len(label_file), 3),
    dtype=np.float32, chunks=(1, 3),
  )
  face_pose = hdf_file.create_dataset(
    'face-pose', shape=(len(label_file), 3),
    dtype=np.float32, chunks=(1, 3),
  )
  # Create face patch folders
  face_folder = osp.join(pp_dd_opt_folder, 'face')
  os.makedirs(face_folder, exist_ok=True)

  # Process gaze data for current person and date
  normalizer = MpiiDataNormalizer(960, (448, 448), distance=300)
  scn2cam = Screen2Camera(scn_pose, scn_size)
  cam2nor = Camera2Normal(cam_data, normalizer)
  pose_estim = HeadPoseEstimator(cam_data)

  for idx in range(len(label_file)):
    sample = label_file[idx] # Numpy structured array

    img = cv2.imread(osp.join(date_folder, sample['img_path']), cv2.IMREAD_UNCHANGED)

    pog = np.array(sample[['gx_px', 'gy_px']].tolist(), dtype=np.float32)
    tgt = scn2cam.load_target(pog)

    landmarks_2d = np.array(sample[[
      'lm01_x', 'lm01_y', # reye, outer
      'lm02_x', 'lm02_y', # reye, inner
      'lm03_x', 'lm03_y', # leye, inner
      'lm04_x', 'lm04_y', # leye, outer
      'lm05_x', 'lm05_y', # mouth, rc
      'lm06_x', 'lm06_y', # mouth, lc
    ]].tolist(), dtype=np.float32).reshape((6, 2))
    hr, ht = pose_estim.estimate(landmarks_2d)
    hR = cv2.Rodrigues(hr)[0]
    landmarks_3d = np.dot(hR, pose_estim.face_model.T) + ht
    fe = calculate_face_center(landmarks_3d)

    img_f, gaze_f, pose_f = cam2nor.normalize_data(fe, hR, img, tgt)
    cv2.imwrite(
      osp.join(face_folder, f'{idx:04d}.jpg'),
      cv2.resize(img_f, (224, 224), interpolation=cv2.INTER_CUBIC),
      [cv2.IMWRITE_JPEG_QUALITY, 100],
    )
    face_gaze[idx] = gaze_f; face_pose[idx] = pose_f

  # Close hdf file
  hdf_file.close()
  # Log processing result
  rt_logger.info(f'processed: {pp}, {dd}, {len(label_file)} samples')

def process_tasks(dataset_path, opt_folder):
  persons_folder = osp.abspath(dataset_path)
  persons = [
    f for f in os.listdir(persons_folder)
    if osp.isdir(osp.join(persons_folder, f))
  ]

  functional_tasks = []

  for person in persons:
    person_folder = osp.join(persons_folder, person)
    dates_folders = os.listdir(person_folder)
    for filename in ['Calibration', f'{person}.txt']:
      dates_folders.remove(filename)

    for date in dates_folders:
      args = (persons_folder, person, date, opt_folder)
      task = FunctionalTask(process_pp_dd, *args)
      functional_tasks.append(task)

  return functional_tasks


def main_procedure(cmdargs: argparse.Namespace):
  dataset_path = osp.abspath(cmdargs.dataset_path)
  rt_logger.info(f'mpii-facegaze dataset: "{dataset_path}"')

  data_folder = ScriptEnv.data_path('mpii-facegaze')
  os.makedirs(data_folder, exist_ok=True)
  rt_logger.info(f'processed data: "{data_folder}"')

  tasks = process_tasks(dataset_path, data_folder)
  executor = futures.ProcessPoolExecutor(cmdargs.max_workers)
  run_parallel(executor, tasks, rt_logger)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='prepare data for MPIIFaceGaze dataset.')

  parser.add_argument(
    '--dataset-path', type=str, required=True,
    help='Path to the extracted MPIIFaceGaze dataset.',
  )
  parser.add_argument(
    '--max-workers', type=int, default=None,
    help='Maximum number of processes in the process pool.',
  )

  main_procedure(parser.parse_args())
