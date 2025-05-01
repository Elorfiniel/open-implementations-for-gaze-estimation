from opengaze.runtime.scripts import ScriptEnv
from opengaze.utils.image import scaled_crop

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import pickle


def load_pickle_file(file_path: str):
  with open(file_path, 'rb') as file:
    return pickle.load(file)

def load_bfm_noneck(file_path: str):
  '''Load the mean shape and BFM basis for shape and expression.'''

  bfm_params = load_pickle_file(file_path)

  kpts = bfm_params['keypoints'].astype(int)

  mean = bfm_params['u'].astype(np.float32)
  w_sh = bfm_params['w_shp'].astype(np.float32)
  w_ex = bfm_params['w_exp'].astype(np.float32)

  return dict(
    mean=mean[kpts],  # Shape: (68*3, 1)
    w_sh=w_sh[kpts],  # Shape: (68*3, 40)
    w_ex=w_ex[kpts],  # Shape: (68*3, 10)
  )

def load_param_norm(file_path: str):
  param_norm = load_pickle_file(file_path)
  return dict(
    mean=param_norm['mean'],  # Shape: (12 + 40 + 10, )
    std=param_norm['std'],    # Shape: (12 + 40 + 10, )
  )

def load_tddfa_onnx(onnx_file: str):
  return ort.InferenceSession(onnx_file)


class FaceBoundingBox:
  def __init__(self, p_detection: float = 0.8, p_suppression: float = 0.2):
    '''Perform face detection using MediaPipe Face Detection Task.

    Args:
      `p_detection`: min confidence score for face detection.
      `p_suppression`: min threshold for non-maximum-suppression.
    '''

    BaseOptions = mp.tasks.BaseOptions
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    model_asset_path = ScriptEnv.resource_path('mediapipe/face_detection.tflite')
    self.options = FaceDetectorOptions(
      base_options=BaseOptions(model_asset_path=model_asset_path),
      running_mode=VisionRunningMode.IMAGE,
      min_detection_confidence=p_detection,
      min_suppression_threshold=p_suppression,
    )
    self._detector = None

  def create(self):
    '''Create face detector instance.'''
    if self._detector is None:
      FaceDetector = mp.tasks.vision.FaceDetector
      self._detector = FaceDetector.create_from_options(self.options)

  def destroy(self):
    '''Destroy face detector instance.'''
    if self._detector is not None:
      self._detector.close()
      self._detector = None

  def __enter__(self):
    self.create()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.destroy()

  def process(self, image: np.ndarray, bgr2rgb: bool = False):
    '''Takes as input an RGB image of shape `(h, w, c)`, then
    returns the detected face bounding boxes of shape `(K, 4)`,
    where each bounding box is represented as a row vector of
    `[x_min, y_min, x_max, y_max]`.

    Args:
      `image`: input face image of shape `(h, w, c)`.
      `bgr2rgb`: convert image from BGR to RGB.
    '''

    image_h, image_w, _ = image.shape

    if bgr2rgb: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(data=image, image_format=mp.ImageFormat.SRGB)
    results = self._detector.detect(mp_image)

    if len(results.detections) > 0:
      bboxes = [d.bounding_box for d in results.detections]
      return np.array([
        [b.origin_x, b.origin_y, b.origin_x + b.width, b.origin_y + b.height]
        for b in bboxes
      ], dtype=np.float32)

    return None


class SparseFaceLandmarks:
  def __init__(self, width_expand: float = 1.6, image_size: int = 120):
    '''Perform sparse face landmark detection using 3DDFA-v2 model.

    Args:
      `width_expand`: width expansion factor for face bounding box.
      `image_size`: input image size for 3DDFA-v2 model.
    '''

    self.width_expand = width_expand
    self.image_size = image_size

    self.bfm_noneck = load_bfm_noneck(ScriptEnv.resource_path('3ddfa-v2/bfm-noneck-v3.pkl'))
    self.param_norm = load_param_norm(ScriptEnv.resource_path('3ddfa-v2/param-mean-std.pkl'))
    self.tddfa_onnx = load_tddfa_onnx(ScriptEnv.resource_path('3ddfa-v2/tddfa-v2-mb1.onnx'))

  def _face_from_bbox(self, image: np.ndarray, bbox: np.ndarray):
    x_min, y_min, x_max, y_max = bbox

    face_center_x = (x_min + x_max) / 2
    face_center_y = (y_min + y_max) / 2

    crop_a = self.width_expand * (x_max - x_min + y_max - y_min) / 2

    x_min = face_center_x - crop_a / 2
    x_max = face_center_x + crop_a / 2
    y_min = face_center_y - crop_a / 2
    y_max = face_center_y + crop_a / 2

    face_crop = scaled_crop(
      image, (x_min, y_min, x_max, y_max),
      (self.image_size, self.image_size),
    )
    face_bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

    return face_crop, face_bbox

  def _run_tddfa_session(self, face_crop: np.ndarray):
    face_crop = np.array((face_crop - 127.5) / 128.0, dtype=np.float32)
    face_crop = np.expand_dims(face_crop.transpose(2, 0, 1), axis=0)

    param = self.tddfa_onnx.run(None, {'input': face_crop})[0]
    param = param.flatten().astype(np.float32)
    param = self.param_norm['mean'] + param * self.param_norm['std']

    return param

  def _face_landmarks(self, face_bbox: np.ndarray, param: np.ndarray):
    proj = param[:12].reshape(3, 4)

    a_sh = param[12:52].reshape(40, 1)
    a_ex = param[52:62].reshape(10, 1)

    mean = self.bfm_noneck['mean']
    shape = self.bfm_noneck['w_sh'] @ a_sh
    exp = self.bfm_noneck['w_ex'] @ a_ex

    ldmks_3d = np.reshape(mean + shape + exp, (68, 3)).transpose(1, 0)

    x_min, y_min, x_max, y_max = face_bbox
    scale_x = (x_max - x_min) / self.image_size
    scale_y = (y_max - y_min) / self.image_size

    wproj_3d = proj[:, :3] @ ldmks_3d + proj[:, 3:]
    ldmks_2d = np.array([
      x_min + scale_x * (wproj_3d[0, :] - 1.0),
      y_min + scale_y * (self.image_size - wproj_3d[1, :]),
    ], dtype=np.float32)

    # Convert 3D landmarks to a canonical coordinate frame, where the origin locates
    # at the center of the 3D bounding box around the head, the x-axis points to the
    # right, the y-axis points downwards, and the z-axis points to the face
    #
    # Note: similar to the convention used in OpenCV's pinhole camera model
    x_min, y_min, z_min = np.min(mean.reshape((68, 3)), axis=0)
    x_max, y_max, z_max = np.max(mean.reshape((68, 3)), axis=0)

    ldmks_3d[0, :] = +1e-3 * (ldmks_3d[0, :] - (x_min + x_max) / 2)
    ldmks_3d[1, :] = -1e-3 * (ldmks_3d[1, :] - (y_min + y_max) / 2)
    ldmks_3d[2, :] = -1e-3 * (ldmks_3d[2, :] - (z_min + z_max) / 2)

    return ldmks_3d.T, ldmks_2d.T

  def process(self, image: np.ndarray, bbox: np.ndarray, bgr2rgb: bool = False):
    '''Takes as input an RGB image of shape `(h, w, c)`, then
    returns the detected 68 face landmarks in both 3D and 2D
    coordinate frames, where the 3D coordinate frame is in
    a canonical space and the 2D coordinate frame is in the
    original image space.

    Args:
      `image`: input face image of shape `(h, w, c)`.
      `bbox`: detected face bounding box of shape `(4, )`.
      `bgr2rgb`: convert image from BGR to RGB.

    Note that 3DDFA-v2 model takes as input a preprocessed BGR
    image. In case of a BGR image, set `bgr2rgb` to `True`,
    even though no channel order conversion is performed.
    '''

    if not bgr2rgb: image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    face_crop, face_bbox = self._face_from_bbox(image, bbox)
    param = self._run_tddfa_session(face_crop)
    ldmks_3d, ldmks_2d = self._face_landmarks(face_bbox, param)

    return ldmks_3d, ldmks_2d
