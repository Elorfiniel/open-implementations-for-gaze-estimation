# model settings
model = dict(
  type='MPIIGaze_GazeNet',
  init_cfg=[
    dict(
      type='Kaiming', mode='fan_in', layer=None,
      override=[
        dict(name='fc_1'),
        dict(name='fc_2'),
      ],
    ),
  ],
)
