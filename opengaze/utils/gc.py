from opengaze.runtime.scripts import ScriptEnv
from opengaze.utils.image import scaled_crop

import cv2
import mediapipe as mp
import numpy as np


class FaceLandmarks:
  def __init__(self, mode: str = 'IMAGE', p_detection: float = 0.8,
               p_presence: float = 0.8, p_tracking: float = 0.8):
    '''Perform face landmark detection using MediaPipe Face Landmark Task.

    Args:
      `mode`: running mode for face mesh task, either `IMAGE` or `VIDEO`.
      `p_detection`: min confidence score for face detection.
      `p_presence`: min confidence score for face presence.
      `p_tracking`: min confidence score for face tracking.
    '''

    assert mode in ['IMAGE', 'VIDEO']

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    model_asset_path = ScriptEnv.resource_path('mediapipe/face_landmarker.task')
    self.options = FaceLandmarkerOptions(
      base_options=BaseOptions(model_asset_path=model_asset_path),
      running_mode=getattr(VisionRunningMode, mode),
      num_faces=1,
      min_face_detection_confidence=p_detection,
      min_face_presence_confidence=p_presence,
      min_tracking_confidence=p_tracking,
    )
    self._landmarker = None

  def create(self):
    '''Create face landmarker instance.'''
    if self._landmarker is None:
      FaceLandmarker = mp.tasks.vision.FaceLandmarker
      self._landmarker = FaceLandmarker.create_from_options(self.options)

  def destroy(self):
    '''Destroy face landmarker instance.'''
    if self._landmarker is not None:
      self._landmarker.close()
      self._landmarker = None

  def __enter__(self):
    self.create()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.destroy()

  def process(self, image: np.ndarray, bgr2rgb: bool = False):
    '''Takes as input an RGB image of shape `(h, w, c)`, then
    returns the detected face landmarks of shape `(478, 2)`.

    Args:
      `image`: input face image of shape `(h, w, c)`.
      `bgr2rgb`: convert image from BGR to RGB.
    '''

    image_h, image_w, _ = image.shape

    if bgr2rgb: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(data=image, image_format=mp.ImageFormat.SRGB)
    results = self._landmarker.detect(mp_image)

    if len(results.face_landmarks) > 0:
      landmarks = [[l.x, l.y] for l in results.face_landmarks[0]]
      landmarks = np.array(landmarks) * np.array([image_w, image_h])
      return landmarks.astype(np.float32)

    return None


class FaceAlignment:
  def __init__(self, width_expand: float = 1.6, hw_ratio: float = 1.0):
    '''Perform face alignment following the `gaze-point-estimation-2023` project, see also:
    https://gitee.com/elorfiniel/gaze-point-estimation-2023/blob/master/source/utils/common/facealign.py

    Args:
      `width_expand`: width expansion factor for eye crops.
      `hw_ratio`: the ratio of height over width for eye crops.
    '''

    self.width_expand = width_expand
    self.hw_ratio = hw_ratio

  def _bbox_from_ldmk(self, landmarks: np.ndarray):
    x_min, y_min = np.min(landmarks, axis=0)
    x_max, y_max = np.max(landmarks, axis=0)
    return np.asarray([x_min, y_min, x_max, y_max], dtype=int)

  def _align_angle(self, landmarks: np.ndarray):
    # Warn: Possible inconsistency between frontal-profile and profile-profile view
    #
    #   Data normalization was proposed to cancal out the geometric variability
    #   brought by head pose and user-camera distance, see also:
    #     "Revisiting Data Normalization for Appearance-Based Gaze Estimation" by Zhang et al.
    #
    #   3D Gaze Estimation (incorporated by most articles about gaze estimation):
    #     1. Normalized camera looks at the origin of head coordinate frame, ie. face center
    #     2. X-axis of head and camera coordinate frames are parallel, ie. horizontal eyes
    #     3. Normalized camera is placed at a fixed distance from the origin mentioned above
    #
    #   2D Gaze Estimation (implemented in the `gaze-estimation-2023` project):
    #     1. Similar idea as above, to cancel out the variability brought by face rotation
    #     2. Inner eye corners are used to align the face with the camera
    #     3. Gaze labels (PoG) are correctly converted in line with the transformation
    #
    #   Note that the pinhole camera model is based on perspective projection, which means two
    #   objects with the same Y_cam coordinates but different Z_cam coordinates will be projected to
    #   locations with different Y_img coordinates on the image plane
    #
    #   In light of this, data normalization may introduce inconsistency among people with
    #   different head poses, eg. frontal-profile and profile-profile view
    ldmk_reye, ldmk_leye = landmarks[133], landmarks[362]

    norm = np.linalg.norm(ldmk_reye - ldmk_leye, ord=2)
    sin = (ldmk_reye[1] - ldmk_leye[1]) / norm
    theta = -np.rad2deg(np.arcsin(sin))

    return theta

  def _align_rotate(self, image: np.ndarray, landmarks: np.ndarray, theta: float):
    image_h, image_w, _ = image.shape

    image_l = 2 * max(image_h, image_w)
    image_a = int((image_l - image_w) / 2)
    image_b = int((image_l - image_h) / 2)

    image_pad = cv2.copyMakeBorder(
      image, image_b, image_b, image_a, image_a,
      cv2.BORDER_CONSTANT, None, value=(0, 0, 0),
    )
    M = cv2.getRotationMatrix2D(
      center=(image_w / 2 + image_a, image_h / 2 + image_b),
      angle=theta, scale=1.0,
    )
    image_rot = cv2.warpAffine(
      image_pad, M, (image_w + 2 * image_a, image_h + 2 * image_b),
      flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
    )
    ldmks_rot = np.dot(
      np.concatenate([
        landmarks + np.array([image_a, image_b]),
        np.ones(shape=(len(landmarks), 1)),
      ], axis=1),
      M.T,
    )

    return image_rot, ldmks_rot

  def _align_face(self, image_rot: np.ndarray, ldmks_rot: np.ndarray):
    image_h, image_w, _ = image_rot.shape

    center = np.array([image_w / 2, image_h / 2])
    metric = max(image_h, image_w) / 2

    x_min, y_min, x_max, y_max = self._bbox_from_ldmk(ldmks_rot)
    y_max = y_min + x_max - x_min

    face_crop = scaled_crop(image_rot, (x_min, y_min, x_max, y_max), (224, 224))
    face_bbox = np.concatenate([
      np.array([(x_min + x_max) / 2, (y_min + y_max) / 2]) - center,
      np.array([x_max - x_min, y_max - y_min]),
    ], axis=0) / metric

    return face_crop, face_bbox

  def _align_reye(self, image_rot: np.ndarray, ldmks_rot: np.ndarray):
    image_h, image_w, _ = image_rot.shape

    center = np.array([image_w / 2, image_h / 2])
    metric = max(image_h, image_w) / 2

    ldmk_indices = [
      160, 33, 161, 163, 133, 7, 173, 144,
      145, 246, 153, 154, 155, 157, 158, 159,
    ]
    x_min, y_min, x_max, y_max = self._bbox_from_ldmk(ldmks_rot[ldmk_indices])
    eye_center_x, eye_center_y = ldmks_rot[468]

    crop_w = self.width_expand * (x_max - x_min)
    crop_h = self.hw_ratio * crop_w

    x_min = eye_center_x - crop_w / 2
    x_max = eye_center_x + crop_w / 2
    y_min = eye_center_y - crop_h / 2
    y_max = eye_center_y + crop_h / 2

    reye_crop = scaled_crop(image_rot, (x_min, y_min, x_max, y_max), (224, 224))
    reye_bbox = np.concatenate([
      np.array([(x_min + x_max) / 2, (y_min + y_max) / 2]) - center,
      np.array([x_max - x_min, y_max - y_min]),
    ], axis=0) / metric

    return reye_crop, reye_bbox

  def _align_leye(self, image_rot: np.ndarray, ldmks_rot: np.ndarray):
    image_h, image_w, _ = image_rot.shape

    center = np.array([image_w / 2, image_h / 2])
    metric = max(image_h, image_w) / 2

    ldmk_indices = [
      384, 385, 386, 387, 388, 390, 263, 362,
      398, 466, 373, 374, 249, 380, 381, 382,
    ]
    x_min, y_min, x_max, y_max = self._bbox_from_ldmk(ldmks_rot[ldmk_indices])
    eye_center_x, eye_center_y = ldmks_rot[473]

    crop_w = self.width_expand * (x_max - x_min)
    crop_h = self.hw_ratio * crop_w

    x_min = eye_center_x - crop_w / 2
    x_max = eye_center_x + crop_w / 2
    y_min = eye_center_y - crop_h / 2
    y_max = eye_center_y + crop_h / 2

    leye_crop = scaled_crop(image_rot, (x_min, y_min, x_max, y_max), (224, 224))
    leye_bbox = np.concatenate([
      np.array([(x_min + x_max) / 2, (y_min + y_max) / 2]) - center,
      np.array([x_max - x_min, y_max - y_min]),
    ], axis=0) / metric

    return leye_crop, leye_bbox

  def _align_ldmks(self, image_rot: np.ndarray, ldmks_rot: np.ndarray):
    image_h, image_w, _ = image_rot.shape

    center = np.array([image_w / 2, image_h / 2])
    metric = max(image_h, image_w) / 2

    return (ldmks_rot - center) / metric

  def align(self, image: np.ndarray, landmarks: np.ndarray):
    '''Align face image using the detected landmarks.

    Args:
      `image`: input face image of shape `(h, w, c)`.
      `landmarks`: detected face landmarks of shape `(478, 2)`.
    '''

    # Calculate alignment angle from the inner eye cornors
    theta = self._align_angle(landmarks)
    # Rotate the image and landmarks by the calculated angle
    image_rot, ldmks_rot = self._align_rotate(image, landmarks, theta)

    # Crop face and eye crops from the rotated image
    face_crop, face_bbox = self._align_face(image_rot, ldmks_rot)
    reye_crop, reye_bbox = self._align_reye(image_rot, ldmks_rot)
    leye_crop, leye_bbox = self._align_leye(image_rot, ldmks_rot)

    # Normalize landmarks wrt the alignment for training
    ldmks = self._align_ldmks(image_rot, ldmks_rot)

    return dict(
      theta=theta, ldmks=ldmks,
      face_crop=face_crop, face_bbox=face_bbox,
      reye_crop=reye_crop, reye_bbox=reye_bbox,
      leye_crop=leye_crop, leye_bbox=leye_bbox,
    )
