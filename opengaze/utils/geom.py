import cv2
import numpy as np


class PoseEstimator:
  def __init__(self, cam_mat: np.ndarray, cam_dist: np.ndarray = None):
    '''Generic pose estimator for 3D objects.

    Args:
      `cam_mat`: camera intrinsic parameters of shape `(3, 3)`.
      `cam_dist`: optional camera distortion parameters.
    '''

    self.cam_mat = cam_mat
    self.cam_dist = cam_dist

  def estimate(self, ldmks_3d: np.ndarray, ldmks_2d: np.ndarray):
    '''Estimate pose from 3D and 2D face landmarks.

    Args:
      `ldmks_3d`: 3D landmarks of shape `(N, 3)`.
      `ldmks_2d`: 2D landmarks of shape `(N, 2)`.
    '''

    _, rvec, tvec = cv2.solvePnP(
      ldmks_3d, ldmks_2d,
      self.cam_mat, self.cam_dist,
      flags=cv2.SOLVEPNP_EPNP,
    )
    _, rvec, tvec = cv2.solvePnP(
      ldmks_3d, ldmks_2d,
      self.cam_mat, self.cam_dist,
      rvec, tvec, True,
      flags=cv2.SOLVEPNP_ITERATIVE,
    )

    return rvec, tvec # Shape: (3, 1), (3, 1)
