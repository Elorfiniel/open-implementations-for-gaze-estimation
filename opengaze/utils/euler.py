# Notes on the coordinate systems.
#
# When the viewer is captured in a frontal view (i.e. the camera is placed
# directly in front of the viewer, targeting the face), the head coordinate
# frame is defined as follows:
#   - x axis: points to the right (from right eye to left eye)
#   - y axis: points downwards (from forehead to chin)
#   - z axis: points towards the viewer (from camera to viewer)

import cv2
import numpy as np
import torch


def gaze_2d_3d_a(pitch: float, yaw: float):
  '''Convert 2D gaze (pitch, yaw) to 3D gaze (x, y, z).

  Args:
    `pitch`: vertical angle in radians.
    `yaw`: horizontal angle in radians.

  Note that when pitch is positive, the viewer is looking upward. And
  when the yaw is positive, the viewer is looking to his/her right.
  '''

  x = -np.cos(pitch) * np.sin(yaw)
  y = -np.sin(pitch)
  z = -np.cos(pitch) * np.cos(yaw)

  return x, y, z

def gaze_3d_2d_a(x: float, y: float, z: float):
  '''Convert 3D gaze (x, y, z) to 2D gaze (pitch, yaw).

  Args:
    `x`: x coordinate in head coordinate frame.
    `y`: y coordinate in head coordinate frame.
    `z`: z coordinate in head coordinate frame.

  Note that 3D gaze should be of unit length, eg. `||g|| = 1`.
  '''

  pitch = np.arcsin(-y)
  yaw = np.arctan2(-x, -z)

  return pitch, yaw


def gaze_2d_3d_n(pitch: np.ndarray, yaw: np.ndarray):
  assert pitch.ndim == 1 and pitch.shape == yaw.shape

  x = -np.cos(pitch) * np.sin(yaw)
  y = -np.sin(pitch)
  z = -np.cos(pitch) * np.cos(yaw)

  return x, y, z

def gaze_3d_2d_n(x: np.ndarray, y: np.ndarray, z: np.ndarray):
  assert x.ndim == 1 and x.shape == y.shape and x.shape == z.shape

  pitch = np.arcsin(-y)
  yaw = np.arctan2(-x, -z)

  return pitch, yaw


def gaze_2d_3d_t(pitch: torch.Tensor, yaw: torch.Tensor):
  assert pitch.ndim == 1 and pitch.shape == yaw.shape

  x = -torch.cos(pitch) * torch.sin(yaw)
  y = -torch.sin(pitch)
  z = -torch.cos(pitch) * torch.cos(yaw)

  return x, y, z

def gaze_3d_2d_t(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
  assert x.ndim == 1 and x.shape == y.shape and x.shape == z.shape

  pitch = torch.arcsin(-y)
  yaw = torch.arctan2(-x, -z)

  return pitch, yaw


def pose_3d_2d_a(x: float, y: float, z: float):
  '''Convert 3D pose (x, y, z) to 2D pose (pitch, yaw).

  Args:
    `x`: x coordinate in reference frame.
    `y`: y coordinate in reference frame.
    `z`: z coordinate in reference frame.
  '''

  r = np.array([x, y, z], dtype=np.float32)
  R = cv2.Rodrigues(r.reshape((3, 1)))[0]
  pitch = np.arcsin(R[1, 2])
  yaw = np.arctan2(R[0, 2], R[2, 2])

  return pitch, yaw
