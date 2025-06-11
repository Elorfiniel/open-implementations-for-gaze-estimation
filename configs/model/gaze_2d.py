# model settings
itracker = dict(
  type='BackboneHead',
  model_cfg=dict(
    type='ITracker',
    init_cfg=[
      dict(
        type='Kaiming', mode='fan_in',
        layer=['Conv2d', 'Linear'],
      ),
    ],
  ),
  loss_cfg=dict(type='MSELoss'),
)

affnet = dict(
  type='BackboneHead',
  model_cfg=dict(
    type='AffNet',
    init_cfg=[
      dict(
        type='Kaiming', mode='fan_in',
        layer=['Conv2d', 'Linear'],
      ),
    ],
  ),
  loss_cfg=dict(type='SmoothL1Loss')
)
