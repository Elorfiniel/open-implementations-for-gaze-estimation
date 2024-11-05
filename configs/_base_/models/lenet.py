# model settings
model = dict(
  type='MPIIGaze_LeNet',
  init_cfg=[
    dict(type='Kaiming', mode='fan_in'),
  ],
)
