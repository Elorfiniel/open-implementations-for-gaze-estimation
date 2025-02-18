from opengaze.registry import TRANSFORMS

import torchvision.transforms as tvt


TRANSFORMS.register_module(name='Grayscale', module=tvt.Grayscale)
TRANSFORMS.register_module(name='Normalize', module=tvt.Normalize)
TRANSFORMS.register_module(name='ToTensor', module=tvt.ToTensor)
