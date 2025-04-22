from opengaze.runtime.scripts import ScriptEnv

import cv2
import mediapipe as mp
import numpy as np


class FaceLandmarks:
  def __init__(self, mode: str = 'IMAGE', p_detection: float = 0.8,
               p_presence: float = 0.8, p_tracking: float = 0.8):
    '''Perform face landmark detection using MediaPipe FaceMesh Solution.

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

    # Note: face_mesh = general_mesh + identity + expression
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

  def process(self, image: np.ndarray, bgr2rgb: bool = True):
    '''Takes as input an image of shape `(h, w, c)`, then returns
    as output the detected face landmarks of shape `(478, 3)`.

    Args:
      `image`: input face image of shape `(h, w, c)`.
      `bgr2rgb`: convert image from BGR to RGB.
    '''

    image_h, image_w, _ = image.shape

    if bgr2rgb: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(data=image, image_format=mp.ImageFormat.SRGB)
    detections = self._landmarker.detect(mp_image)

    if len(detections.face_landmarks) > 0:
      landmarks = [[l.x, l.y] for l in detections.face_landmarks[0]]
      landmarks = np.array(landmarks) * np.array([image_w, image_h])
      return landmarks.astype(np.float32)

    return None
