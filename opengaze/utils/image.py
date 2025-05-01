import cv2
import numpy as np


def scaled_crop(image: np.ndarray, bbox: tuple, size: tuple):
  '''Crop image region from the bounding box, then resize to the given size.

  Args:
    `image`: input image of shape `(h, w, c)`.
    `bbox`: region `(x_min, y_min, x_max, y_max)` to be cropped.
    `size`: size `(w, h)` of the output image.
  '''

  x_min, y_min, x_max, y_max = bbox
  crop_w, crop_h = size

  src_pts = np.array([(x_min, y_min), (x_max, y_min), (x_min, y_max)], dtype=np.float32)
  tgt_pts = np.array([(0, 0), (crop_w, 0), (0, crop_h)], dtype=np.float32)

  M = cv2.getAffineTransform(src_pts, tgt_pts)

  return cv2.warpAffine(image, M, (crop_w, crop_h), flags=cv2.INTER_CUBIC)
