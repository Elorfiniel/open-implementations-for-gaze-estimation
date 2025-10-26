# model settings
LeNet = dict(
  type='BackboneHead',
  model_cfg=dict(
    type='LeNet',
    init_cfg=[
      dict(
        type='Kaiming', mode='fan_in',
        layer=['Conv2d', 'Linear'],
      ),
    ],
  ),
  loss_cfg=dict(type='MSELoss'),
)

GazeNet = dict(
  type='BackboneHead',
  model_cfg=dict(
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
  ),
  loss_cfg=dict(type='MSELoss'),
)

DilatedNet = dict(
  type='BackboneHead',
  model_cfg=dict(
    type='DilatedNet',
    init_cfg=[
      dict(
        type='Kaiming', mode='fan_in', layer=None,
        override=[
          dict(name='face_conv_2'),
          dict(name='face_fc'),
          dict(name='eyes_conv_2'),
          dict(name='eyes_fc'),
          dict(name='fc'),
        ],
      ),
    ],
    p_dropout=0.1,
  ),
  loss_cfg=dict(type='MSELoss'),
)

FullFace = dict(
  type='BackboneHead',
  model_cfg=dict(
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
  ),
  loss_cfg=dict(type='L1Loss'),
)

CANet = dict(
  type='BackboneHead',
  model_cfg=dict(
    type='CANet',
    init_cfg=[
      dict(
        type='Kaiming', mode='fan_in',
        layer=['Conv2d', 'Linear'],
      ),
    ],
  ),
  loss_cfg=dict(type='L1Loss'),
)

XGaze224 = dict(
  type='BackboneHead',
  model_cfg=dict(
    type='XGaze224',
    init_cfg=[
      dict(
        type='Kaiming', mode='fan_in', layer=None,
        override=[dict(name='fc')],
      ),
    ],
  ),
  loss_cfg=dict(type='L1Loss'),
)

GazeTR = dict(
  type='BackboneHead',
  model_cfg=dict(
    type='GazeTR',
    init_cfg=[
      dict(
        type='Kaiming', mode='fan_in',
        layer=['Conv2d', 'Linear'],
      ),
    ],
  ),
  loss_cfg=dict(type='L1Loss'),
)
