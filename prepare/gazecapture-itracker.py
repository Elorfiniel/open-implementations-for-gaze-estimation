# This script follows the pipeline from the "Eye Tracking for Everyone" paper

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
  metadata = load_json_data(osp.join(subject_folder, 'info.json'))

  image_names = load_json_data(osp.join(subject_folder, 'frames.json'))
  gaze_labels = load_json_data(osp.join(subject_folder, 'dotInfo.json'))

  apple_face = load_json_data(osp.join(subject_folder, 'appleFace.json'))
  apple_reye = load_json_data(osp.join(subject_folder, 'appleRightEye.json'))
  apple_leye = load_json_data(osp.join(subject_folder, 'appleLeftEye.json'))
  apple_grid = load_json_data(osp.join(subject_folder, 'faceGrid.json'))

  face_valid = np.asarray(apple_face['IsValid'], dtype=bool)
  reye_valid = np.asarray(apple_reye['IsValid'], dtype=bool)
  leye_valid = np.asarray(apple_leye['IsValid'], dtype=bool)

  fetch_bbox = lambda d: np.stack([d['X'], d['Y'], d['W'], d['H']], axis=1).astype(int)

  face_bbox = fetch_bbox(apple_face) + np.array([-1, -1, 1, 1])
  reye_bbox = fetch_bbox(apple_reye) + np.array([0, -1, 0, 0])
  reye_bbox[:, :2] += face_bbox[:, :2]
  leye_bbox = fetch_bbox(apple_leye) + np.array([0, -1, 0, 0])
  leye_bbox[:, :2] += face_bbox[:, :2]
  grid_bbox = fetch_bbox(apple_grid)

  label_file = np.array([
    z for z in zip(image_names, gaze_labels['XCam'], gaze_labels['YCam'])
  ], dtype=[('image', 'U32'), ('gx_cm', 'f4'), ('gy_cm', 'f4')])
  image_valid = face_valid & reye_valid & leye_valid

  valid_labels = label_file[image_valid]
  valid_bboxes = dict(
    face=face_bbox[image_valid],
    reye=reye_bbox[image_valid],
    leye=leye_bbox[image_valid],
    grid=grid_bbox[image_valid],
  )

  return metadata, valid_labels, valid_bboxes

def image_crop(image: np.ndarray, bbox: np.ndarray):
  a_src = np.maximum(bbox[:2], 0)
  b_src = np.minimum(bbox[:2] + bbox[2:], (image.shape[1], image.shape[0]))

  a_dst = a_src - bbox[:2]
  b_dst = a_dst + b_src - a_src

  crop = np.zeros((bbox[3], bbox[2], 3), dtype=np.uint8)
  crop[a_dst[1]:b_dst[1], a_dst[0]:b_dst[0]] = image[a_src[1]:b_src[1], a_src[0]:b_src[0]]

  return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)

def process_subject(subjects_folder, subject, opt_folder):
  # Load annotations for current subject
  subject_folder = osp.join(subjects_folder, subject)
  metadata, valid_labels, valid_bboxes = load_gc_annot_subject(subject_folder)

  # Create output folder for current subject
  subject_opt_folder = osp.join(opt_folder, subject)
  os.makedirs(subject_opt_folder, exist_ok=True)

  # Create hdf datasets
  hdf_file = h5py.File(osp.join(subject_opt_folder, f'annot.h5'), 'w')
  max_n_samples = max(len(valid_labels), 1)
  name = hdf_file.create_dataset(
    'name', shape=max_n_samples,
    dtype=h5py.string_dtype(length=32),
    chunks=1, maxshape=max_n_samples,
  )
  gaze = hdf_file.create_dataset(
    'gaze', shape=(max_n_samples, 2),
    dtype=np.float32, chunks=(1, 2),
    maxshape=(max_n_samples, 2),
  )
  grid = hdf_file.create_dataset(
    'grid', shape=(max_n_samples, 4),
    dtype=int, chunks=(1, 4),
    maxshape=(max_n_samples, 4),
  )

  # Create image patch folders
  face_folder = osp.join(subject_opt_folder, 'face')
  os.makedirs(face_folder, exist_ok=True)
  reye_folder = osp.join(subject_opt_folder, 'reye')
  os.makedirs(reye_folder, exist_ok=True)
  leye_folder = osp.join(subject_opt_folder, 'leye')
  os.makedirs(leye_folder, exist_ok=True)

  # Process gaze data for current subject
  n_samples = 0 # Number of final samples

  for idx in range(len(valid_labels)):
    sample = valid_labels[idx]  # Numpy structured array

    img = cv2.imread(
      osp.join(subject_folder, 'frames', sample['image']),
      flags=cv2.IMREAD_UNCHANGED,
    )
    pog = np.array(sample[['gx_cm', 'gy_cm']].tolist(), dtype=np.float32)

    if img is None: continue

    cv2.imwrite(
      osp.join(face_folder, sample['image']),
      image_crop(img, valid_bboxes['face'][idx]),
      [cv2.IMWRITE_JPEG_QUALITY, 100],
    )
    cv2.imwrite(
      osp.join(reye_folder, sample['image']),
      image_crop(img, valid_bboxes['reye'][idx]),
      [cv2.IMWRITE_JPEG_QUALITY, 100],
    )
    cv2.imwrite(
      osp.join(leye_folder, sample['image']),
      image_crop(img, valid_bboxes['leye'][idx]),
      [cv2.IMWRITE_JPEG_QUALITY, 100],
    )
    name[idx] = sample['image']
    gaze[idx] = pog
    grid[idx] = valid_bboxes['grid'][idx]

    n_samples = n_samples + 1

  name.resize(n_samples, axis=0)
  gaze.resize(n_samples, axis=0)
  grid.resize(n_samples, axis=0)

  # Close hdf file
  hdf_file.close()

  # Save metadata about subject
  save_json_data(dict(
    device=metadata['DeviceName'],
    split=metadata['Dataset'],
    counts=n_samples,
  ), osp.join(subject_opt_folder, 'meta.json'))

  # Log processing result
  if n_samples == 0:
    rt_logger.warning(f'skipped: subject {subject}, no valid samples')
    shutil.rmtree(subject_opt_folder, ignore_errors=True)
  else:
    rt_logger.info(f'processed: subject {subject}, {n_samples} samples')

def process_tasks(dataset_path, opt_folder):
  subjects_folder = osp.abspath(dataset_path)
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
  dataset_path = osp.abspath(cmdargs.dataset_path)
  rt_logger.info(f'gazecapture dataset: "{dataset_path}"')

  data_folder = ScriptEnv.data_path('gazecapture')
  os.makedirs(data_folder, exist_ok=True)
  rt_logger.info(f'processed data: "{data_folder}"')

  tasks = process_tasks(dataset_path, data_folder)
  executor = futures.ProcessPoolExecutor(cmdargs.max_workers)
  run_parallel(executor, tasks, rt_logger)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='prepare data for GazeCapture dataset.')

  parser.add_argument(
    '--dataset-path', type=str, required=True,
    help='Path to the extracted GazeCapture dataset.',
  )
  parser.add_argument(
    '--max-workers', type=int, default=None,
    help='Maximum number of processes in the process pool.',
  )

  main_procedure(parser.parse_args())
