from template.registry import TRANSFORMS

import torchvision.transforms as tvt


TRANSFORMS.register_module(name='RandomCrop', module=tvt.RandomCrop)
TRANSFORMS.register_module(name='RandomResizedCrop', module=tvt.RandomResizedCrop)
TRANSFORMS.register_module(name='RandomRotation', module=tvt.RandomRotation)
TRANSFORMS.register_module(name='Resize', module=tvt.Resize)
TRANSFORMS.register_module(name='Normalize', module=tvt.Normalize)
TRANSFORMS.register_module(name='ToTensor', module=tvt.ToTensor)
