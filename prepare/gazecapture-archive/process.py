# This script follows the pipeline from the `gaze-point-estimation-2023`
# project, using data from the archived GazeCapture dataset

from opengaze.utils import FaceLandmarks, FaceAlignment
from opengaze.runtime.scripts import ScriptEnv
from opengaze.runtime.log import runtime_logger
from opengaze.runtime.parallel import FunctionalTask, run_parallel

import argparse
import concurrent.futures as futures
import cv2
import h5py
import json
import numpy as np
import os
import os.path as osp
import shutil
import tarfile


rt_logger = runtime_logger(
  name='gazecapture',
  log_file=ScriptEnv.log_path('prepare-gazecapture.log'),
)


def load_json_data(json_file: str):
  with open(json_file, 'r') as file:
    return json.load(file)

def save_json_data(json_data: dict, json_file: str):
  with open(json_file, 'w') as file:
    json.dump(json_data, file, indent=2)

def load_gc_annot_subject(subject_folder):
  metadata = load_json_data(osp.join(subject_folder, 'metadata.json'))

  with h5py.File(osp.join(subject_folder, 'labels.h5'), 'r') as hdf_file:
    image_names = hdf_file['labels']['image_names']
    gaze_labels_x = hdf_file['labels']['dot_x_cam']
    gaze_labels_y = hdf_file['labels']['dot_y_cam']

    face_valid = np.asarray(hdf_file['labels']['face_valid'], dtype=bool)
    reye_valid = np.asarray(hdf_file['labels']['reye_valid'], dtype=bool)
    leye_valid = np.asarray(hdf_file['labels']['leye_valid'], dtype=bool)

    label_file = np.array([
      z for z in zip(image_names, gaze_labels_x, gaze_labels_y)
    ], dtype=[('image', 'U32'), ('gx_cm', 'f4'), ('gy_cm', 'f4')])
    image_valid = face_valid & reye_valid & leye_valid

  valid_labels = label_file[image_valid]

  return metadata, valid_labels

def rotate_vector(vector: np.ndarray, angle: float):
  radian = np.deg2rad(angle, dtype=np.float32)

  cos, sin = np.cos(radian), np.sin(radian)
  mat = np.array([[cos, -sin], [sin, cos]])

  return np.dot(mat, vector)

def process_subject(subjects_folder, subject, opt_folder):
  # Load annotations for current subject
  subject_folder = osp.join(subjects_folder, subject)
  metadata, valid_labels = load_gc_annot_subject(subject_folder)

  # Create output folder for current subject
  subject_opt_folder = osp.join(opt_folder, subject)
  os.makedirs(subject_opt_folder, exist_ok=True)

  # Create hdf datasets
  hdf_file = h5py.File(osp.join(subject_opt_folder, f'annot.h5'), 'w')
  name = hdf_file.create_dataset(
    'name', shape=len(valid_labels),
    dtype=h5py.string_dtype(length=32),
    chunks=1, maxshape=len(valid_labels),
  )
  gaze = hdf_file.create_dataset(
    'gaze', shape=(len(valid_labels), 2),
    dtype=np.float32, chunks=(1, 2),
    maxshape=(len(valid_labels), 2),
  )
  bbox = hdf_file.create_dataset(
    'bbox', shape=(len(valid_labels), 12),
    dtype=np.float32, chunks=(1, 12),
    maxshape=(len(valid_labels), 12),
  )
  ldmk = hdf_file.create_dataset(
    'ldmk', shape=(len(valid_labels), 478, 2),
    dtype=np.float32, chunks=(1, 478, 2),
    maxshape=(len(valid_labels), 478, 2),
  )

  # Create image patch folders
  face_folder = osp.join(subject_opt_folder, 'face')
  os.makedirs(face_folder, exist_ok=True)
  reye_folder = osp.join(subject_opt_folder, 'reye')
  os.makedirs(reye_folder, exist_ok=True)
  leye_folder = osp.join(subject_opt_folder, 'leye')
  os.makedirs(leye_folder, exist_ok=True)

  # Process gaze data for current subject
  alignment = FaceAlignment(width_expand=1.6, hw_ratio=1.0)
  n_samples = 0 # Number of final samples

  landmarker = FaceLandmarks(p_detection=0.6, p_presence=0.6)
  tarball = tarfile.open(osp.join(subject_folder, 'images.tar.gz'), 'r:gz')
  with landmarker, tarball:
    for idx in range(len(valid_labels)):
      sample = valid_labels[idx]  # Numpy structured array

      try:  # Check if the image exists
        tarball.getmember(sample['image'])
      except KeyError: continue

      img_raw_data = tarball.extractfile(sample['image']).read()
      img = cv2.imdecode(
        np.frombuffer(img_raw_data, np.uint8),
        cv2.IMREAD_UNCHANGED,
      )
      pog = np.array(sample[['gx_cm', 'gy_cm']].tolist(), dtype=np.float32)

      landmarks = landmarker.process(img, bgr2rgb=True)
      if landmarks is None: continue
      align_dict = alignment.align(img, landmarks)

      cv2.imwrite(
        osp.join(face_folder, sample['image']),
        align_dict['face_crop'],
        [cv2.IMWRITE_JPEG_QUALITY, 100],
      )
      cv2.imwrite(
        osp.join(reye_folder, sample['image']),
        align_dict['reye_crop'],
        [cv2.IMWRITE_JPEG_QUALITY, 100],
      )
      cv2.imwrite(
        osp.join(leye_folder, sample['image']),
        align_dict['leye_crop'],
        [cv2.IMWRITE_JPEG_QUALITY, 100],
      )
      name[idx] = sample['image']
      gaze[idx] = rotate_vector(pog, -align_dict['theta'])
      bbox[idx] = np.concatenate([
        align_dict['face_bbox'],
        align_dict['reye_bbox'],
        align_dict['leye_bbox'],
      ], axis=0)
      ldmk[idx] = align_dict['ldmks']

      n_samples = n_samples + 1

  gaze.resize(n_samples, axis=0)
  bbox.resize(n_samples, axis=0)
  ldmk.resize(n_samples, axis=0)

  # Close hdf file
  hdf_file.close()

  # Save metadata about subject
  save_json_data(dict(
    device=metadata['device'],
    split=metadata['split'],
    counts=n_samples,
  ), osp.join(subject_opt_folder, 'meta.json'))

  # Log processing result
  if n_samples == 0:
    rt_logger.warning(f'skipped: subject {subject}, no valid samples')
    shutil.rmtree(subject_opt_folder, ignore_errors=True)
  else:
    rt_logger.info(f'processed: subject {subject}, {n_samples} samples')

def process_tasks(archive_path, opt_folder):
  subjects_folder = osp.abspath(archive_path)
  subjects = [
    f for f in os.listdir(subjects_folder)
    if osp.isdir(osp.join(subjects_folder, f))
  ]

  functional_tasks = []

  for subject in subjects:
    args = (subjects_folder, subject, opt_folder)
    task = FunctionalTask(process_subject, *args)
    functional_tasks.append(task)

  return functional_tasks


def main_procedure(cmdargs: argparse.Namespace):
  archive_path = osp.abspath(cmdargs.archive_path)
  rt_logger.info(f'gazecapture archive: "{archive_path}"')

  data_folder = ScriptEnv.data_path('gazecapture')
  os.makedirs(data_folder, exist_ok=True)
  rt_logger.info(f'processed data: "{data_folder}"')

  tasks = process_tasks(archive_path, data_folder)
  executor = futures.ProcessPoolExecutor(cmdargs.max_workers)
  run_parallel(executor, tasks)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='prepare data for GazeCapture dataset.')

  parser.add_argument(
    '--archive-path', type=str, required=True,
    help='Path to the archived GazeCapture dataset.',
  )
  parser.add_argument(
    '--max-workers', type=int, default=None,
    help='Maximum number of processes in the process pool.',
  )

  main_procedure(parser.parse_args())
