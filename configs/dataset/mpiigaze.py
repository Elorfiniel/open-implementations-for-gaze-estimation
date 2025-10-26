# image transformation
transform = [
  dict(type='ToTensor'),
]

# dataset settings
train = dict(
  type='MPIIGaze',
  root='data/mpii-gaze',
  train=True,
  test_pp='p00',
  eval_subset=True,
  transform=transform,
)

valid = dict(
  type='MPIIGaze',
  root='data/mpii-gaze',
  train=False,
  test_pp='p00',
  eval_subset=True,
  transform=transform,
)
