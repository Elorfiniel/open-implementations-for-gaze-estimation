import cv2
import numpy as np


class MpiiDataNormalizer:
  def __init__(self, focal_length=960.0, image_size=(60.0, 36.0), distance=600.0):
    '''Perform data normalization used in MPIIGaze dataset.

    Args:
      `focal_length`: focal length of the normalized camera.
      `image_size`: size (w, h) of the normalized image.
      `distance`: distance from the looked-at object to the normalized camera.
    '''

    self.f = focal_length
    self.c = image_size
    self.d = distance

  def normalize_matrices(self, look_at, R1, Kc):
    '''Calculate matrices used in data normalization.

    Args:
      `look_at`: 3d location of the looked-at object (eg. eye center).
      `R1`: transistion matrix (original camera -> world), such that the three columns
      correspond to the x, y, z axes of the world in the camera coordinate frame.
      `Kc`: intrinsic parameters of the original camera.

    Return:
      `Kv`: intrinsic parameters of the normalized camera.
      `S`: scaling matrix for image warping.
      `R2`: transistion matrix (normalized camera -> original camera).
      `W`: perspective transformation for image warping.
    '''

    distance = np.linalg.norm(look_at)
    scaling = self.d / distance

    Kv = np.array([
      [self.f, 0.0, self.c[0] / 2],
      [0.0, self.f, self.c[1] / 2],
      [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    S = np.array([
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, scaling],
    ], dtype=np.float32)

    # Find axes of normalized camera coordinate frame in camera coordinate frame
    z_axis = look_at / distance
    x_axis_head = R1[:, 0]
    y_axis = np.cross(z_axis, x_axis_head)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    R2 = np.vstack([x_axis, y_axis, z_axis])

    # Calculate perspective transformation for image warping
    W = np.dot(np.dot(Kv, S), np.dot(R2, np.linalg.inv(Kc)))

    return Kv, S, R2, W

  def warp_image(self, image, W):
    '''Warp to normalized image with perspective transformation.

    Args:
      `image`: image taken by the original camera.
      `W`: perspective transformation for image warping.
    '''

    return cv2.warpPerspective(image, W, self.c)
