# This script converts the folder structure of the GazeCapture dataset into
# a more compact format for archive purpose, which leads to faster access

from opengaze.runtime.log import runtime_logger
from opengaze.runtime.parallel import FunctionalTask, run_parallel

from numpy.lib import recfunctions as rfn

import argparse
import concurrent.futures as futures
import h5py
import json
import numpy as np
import os
import os.path as osp
import pandas as pd
import tarfile


rt_logger = runtime_logger('gazecapture')


def load_json_data(json_file: str):
  with open(json_file, 'r') as file:
    return json.load(file)

def save_json_data(json_data: dict, json_file: str):
  with open(json_file, 'w') as file:
    json.dump(json_data, file, indent=2)

def load_gc_subject_labels(subject_folder):
  image_names = load_json_data(osp.join(subject_folder, 'frames.json'))

  apple_face = load_json_data(osp.join(subject_folder, 'appleFace.json'))
  apple_reye = load_json_data(osp.join(subject_folder, 'appleRightEye.json'))
  apple_leye = load_json_data(osp.join(subject_folder, 'appleLeftEye.json'))
  apple_grid = load_json_data(osp.join(subject_folder, 'faceGrid.json'))

  gaze_labels = load_json_data(osp.join(subject_folder, 'dotInfo.json'))
  screen_whr = load_json_data(osp.join(subject_folder, 'screen.json'))

  images = np.array(image_names, dtype=[('image_names', 'S32'), ])
  labels = pd.DataFrame(data=dict(
    face_x=apple_face["X"],
    face_y=apple_face["Y"],
    face_w=apple_face["W"],
    face_h=apple_face["H"],
    face_valid=apple_face["IsValid"],
    reye_x=apple_reye["X"],
    reye_y=apple_reye["Y"],
    reye_w=apple_reye["W"],
    reye_h=apple_reye["H"],
    reye_valid=apple_reye["IsValid"],
    leye_x=apple_leye["X"],
    leye_y=apple_leye["Y"],
    leye_w=apple_leye["W"],
    leye_h=apple_leye["H"],
    leye_valid=apple_leye["IsValid"],
    dot_n=gaze_labels["DotNum"],
    dot_x_pts=gaze_labels["XPts"],
    dot_y_pts=gaze_labels["YPts"],
    dot_x_cam=gaze_labels["XCam"],
    dot_y_cam=gaze_labels["YCam"],
    dot_time=gaze_labels["Time"],
    grid_x=apple_grid["X"],
    grid_y=apple_grid["Y"],
    grid_w=apple_grid["W"],
    grid_h=apple_grid["H"],
    grid_valid=apple_grid["IsValid"],
    screen_w=screen_whr["W"],
    screen_h=screen_whr["H"],
    screen_r=screen_whr["Orientation"],
  )).to_records(index=False)

  return rfn.merge_arrays([images, labels], flatten=True, usemask=False)

def compress_gc_subject_images(subject_folder, tarball):
  images_folder = osp.join(subject_folder, 'frames')
  images = os.listdir(images_folder)
  for image_name in images:
    image_path = osp.join(images_folder, image_name)
    tarball.add(image_path, arcname=image_name)

def load_gc_subject_metadata(subject_folder):
  metadata = load_json_data(osp.join(subject_folder, 'info.json'))
  return dict(
    total=metadata["TotalFrames"],
    device=metadata['DeviceName'],
    split=metadata['Dataset'],
  )

def process_subject(subjects_folder, subject, archive_path):
  # Create output folder for current subject
  archive_folder = osp.join(archive_path, subject)
  os.makedirs(archive_folder, exist_ok=True)

  # Load annotations for current subject
  subject_folder = osp.join(subjects_folder, subject)
  # Create hdf dataset (motion data is discarded)
  labels = load_gc_subject_labels(subject_folder)
  with h5py.File(
    osp.join(archive_folder, 'labels.h5'), 'w',
  ) as hdf_file:
    dataset_kwargs = dict(compression="gzip", compression_opts=5)
    hdf_file.create_dataset('labels', data=labels, **dataset_kwargs)
  # Compress images (saved as tarball)
  with tarfile.open(
    osp.join(archive_folder, 'images.tar.gz'),
    mode='w:gz', compresslevel=5,
  ) as tarball:
    compress_gc_subject_images(subject_folder, tarball)
  # Load metadata for current subject
  save_json_data(
    load_gc_subject_metadata(subject_folder),
    osp.join(archive_folder, 'metadata.json'),
  )

def process_tasks(dataset_path, archive_path):
  subjects_folder = osp.abspath(dataset_path)
  subjects = [
    f for f in os.listdir(subjects_folder)
    if osp.isdir(osp.join(subjects_folder, f))
  ]

  functional_tasks = []

  for subject in subjects:
    args = (subjects_folder, subject, archive_path)
    task = FunctionalTask(process_subject, *args)
    functional_tasks.append(task)

  return functional_tasks


def main_procedure(cmdargs: argparse.Namespace):
  dataset_path = osp.abspath(cmdargs.dataset_path)
  rt_logger.info(f'gazecapture dataset: "{dataset_path}"')

  archive_path = osp.abspath(cmdargs.archive_path)
  os.makedirs(archive_path, exist_ok=True)
  rt_logger.info(f'gazecapture archive: "{archive_path}"')

  tasks = process_tasks(dataset_path, archive_path)
  executor = futures.ProcessPoolExecutor(cmdargs.max_workers)
  run_parallel(executor, tasks)



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='archive data for GazeCapture dataset.')

  parser.add_argument(
    '--dataset-path', type=str, required=True,
    help='Path to the extracted GazeCapture dataset.',
  )
  parser.add_argument(
    '--archive-path', type=str, required=True,
    help='Path to the archived GazeCapture dataset.',
  )
  parser.add_argument(
    '--max-workers', type=int, default=None,
    help='Maximum number of processes in the process pool.',
  )

  main_procedure(parser.parse_args())
