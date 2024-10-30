# model settings
model = dict(
  type='MPIIGaze_GazeNet',
  init_cfg=[
    dict(type='Kaiming', mode='fan_in', layer='Linear'),
  ],
)
