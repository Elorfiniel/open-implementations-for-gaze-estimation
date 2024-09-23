import cv2
import numpy as np


__all__ = [
  'gaze_2d_3d_a', 'gaze_3d_2d_a',
  'gaze_2d_3d_v', 'gaze_3d_2d_v',
  'pose_3d_2d_a',
  'pose_3d_2d_v',
]


def gaze_2d_3d_a(pitch: float, yaw: float):
  x = -np.cos(pitch) * np.sin(yaw)
  y = -np.sin(pitch)
  z = -np.cos(pitch) * np.cos(yaw)
  return x, y, z

def gaze_3d_2d_a(x: float, y: float, z: float):
  pitch = np.arcsin(-y)
  yaw = np.arctan2(-x, -z)
  return pitch, yaw

def gaze_2d_3d_v(gaze_2d: np.ndarray):
  '''Convert 2D gaze angle to 3D gaze vector.

  `gaze_2d`: (pitch, yaw), ndarray of shape `(2, )`.
  '''

  x, y, z = gaze_2d_3d_a(gaze_2d[0], gaze_2d[1])
  return np.array([x, y, z], dtype=np.float32)

def gaze_3d_2d_v(gaze_3d: np.ndarray):
  '''Convert 3D gaze vector to 2D gaze angle.

  `gaze_3d`: (x, y, z), ndarray of shape `(3, )`.
  '''

  pitch, yaw = gaze_3d_2d_a(gaze_3d[0], gaze_3d[1], gaze_3d[2])
  return np.array([pitch, yaw], dtype=np.float32)


def _pose_2d_3d_a(pitch: float, yaw: float):
  raise NotImplementedError(f'Not implemented due to loss of information.')

def pose_3d_2d_a(x: float, y: float, z: float):
  r = np.array([x, y, z], dtype=np.float32)
  R = cv2.Rodrigues(r.reshape((3, 1)))[0]
  pitch = np.arcsin(R[1, 2])
  yaw = np.arctan2(R[0, 2], R[2, 2])
  return pitch, yaw

def _pose_2d_3d_v(pose_2d: np.ndarray):
  '''Convert 2D pose angle to 3D pose vector.

  `pose_2d`: (pitch, yaw), ndarray of shape `(2, )`.

  Warn:
    This function is a only placeholder for inverse conversion.
    The conversion cannot be done due to the loss of information.
    See function `pose_3d_2d_a`, where x and y are discarded.
  '''

  x, y, z = _pose_2d_3d_a(pose_2d[0], pose_2d[1])
  return np.array([x, y, z], dtype=np.float32)

def pose_3d_2d_v(pose_3d: np.ndarray):
  '''Convert 3D pose vector to 2D pose angle.

  `gaze_3d`: (x, y, z), ndarray of shape `(3, )`.
  '''

  pitch, yaw = pose_3d_2d_a(pose_3d[0], pose_3d[1], pose_3d[2])
  return np.array([pitch, yaw], dtype=np.float32)
