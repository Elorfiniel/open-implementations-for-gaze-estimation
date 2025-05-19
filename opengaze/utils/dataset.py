from mmengine.dataset import Compose


def build_image_transform(transform=None):
  '''Build image transformation pipeline.

  Args:
    `transform`: image transformation, which can take the following types:
      - None: use default transform pipeline (i.e. ToTensor).
      - callable: returned as-is.
      - dict: a config dict supported by mmengine.
      - list: a list of aforementioned dict.
  '''

  if transform is None:
    transform = dict(type='ToTensor')

  if isinstance(transform, dict):
    transform = [transform]
  if isinstance(transform, (list, tuple)):
    transform = Compose(transform)

  return transform
