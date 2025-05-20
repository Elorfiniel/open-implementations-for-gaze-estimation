from torchvision import transforms as tvt

from opengaze.registry import TRANSFORMS

from .base import BaseTransform
from .gaze_2d import ITrackerMakeGrid


TRANSFORMS.register_module(name='Grayscale', module=tvt.Grayscale)
TRANSFORMS.register_module(name='Normalize', module=tvt.Normalize)
TRANSFORMS.register_module(name='ToTensor', module=tvt.ToTensor)
