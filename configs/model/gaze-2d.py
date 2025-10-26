# model settings
ITracker = dict(
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

AFFNet = dict(
  type='BackboneHead',
  model_cfg=dict(
    type='AFFNet',
    init_cfg=[
      dict(
        type='Kaiming', mode='fan_in',
        layer=['Conv2d', 'Linear'],
      ),
    ],
  ),
  loss_cfg=dict(type='SmoothL1Loss'),
)
