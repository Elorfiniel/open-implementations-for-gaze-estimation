# model settings
itracker_plus = dict(
  type='BackboneHead',
  model_cfg=dict(
    type='ITrackerPlus',
    init_cfg=[
      dict(
        type='Kaiming', mode='fan_in', layer=None,
        override=[
          dict(name='face_fc'),
          dict(name='eyes_fc'),
          dict(name='kpts_fc'),
          dict(name='fc'),
        ],
      ),
    ],
  ),
  loss_cfg=dict(type='MSELoss'),
)

quant_itracker_plus = dict(
  type='BackboneHead',
  model_cfg=dict(type='QuantITrackerPlus', init_cfg=None),
  loss_cfg=dict(type='MSELoss'),
)
