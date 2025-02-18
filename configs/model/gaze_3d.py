# model settings
xgaze224 = dict(
  type='XGaze224',
  init_cfg=[
    dict(
      type='Kaiming', mode='fan_in', layer=None,
      override=[dict(name='fc')],
    ),
  ],
)
