# model settings
lenet = dict(
  type='LeNet',
  init_cfg=[
    dict(type='Kaiming', mode='fan_in'),
  ],
)

gazenet = dict(
  type='GazeNet',
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

fullface = dict(
  type='FullFace',
  init_cfg=[
    dict(
      type='Kaiming', mode='fan_in', layer=None,
      override=[
        dict(name='sw'),
        dict(name='fc'),
      ],
    ),
  ],
)

xgaze224 = dict(
  type='XGaze224',
  init_cfg=[
    dict(
      type='Kaiming', mode='fan_in', layer=None,
      override=[dict(name='fc')],
    ),
  ],
)
