# model settings
model = dict(
  type='MPIIGaze_GazeNet',
  init_cfg=[
    dict(type='Kaiming', layer='Linear'),
  ],
)
