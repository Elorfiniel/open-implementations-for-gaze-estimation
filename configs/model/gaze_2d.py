# model settings
itracker = dict(
  type='ITracker',
  init_cfg=[
    dict(type='Kaiming', mode='fan_in'),
  ],
)

affnet = dict(
  type='AffNet',
  init_cfg=[
    dict(type='Kaiming', mode='fan_in'),
  ],
)
