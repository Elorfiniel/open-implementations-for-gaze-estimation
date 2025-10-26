# image transformation
transform = [
  dict(type='ToTensor'),
  dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]

# dataset settings
train = dict(
  type='MPIIFaceGaze',
  root='data/mpii-facegaze',
  train=True,
  test_pp='p00',
  transform=transform,
)

valid = dict(
  type='MPIIFaceGaze',
  root='data/mpii-facegaze',
  train=False,
  test_pp='p00',
  transform=transform,
)
