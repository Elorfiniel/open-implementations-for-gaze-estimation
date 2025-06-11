# model settings
resnet50_bn = dict(
  type='BackboneHead',
  model_cfg=dict(
    type='ExpNormLayerResNet',
    norm_layer='BatchNorm',
    init_cfg=[
      dict(
        type='Kaiming', mode='fan_in',
        layer=['Conv2d', 'Linear'],
      ),
    ],
  ),
  loss_cfg=dict(type='L1Loss'),
)

resnet50_ln = dict(
  type='BackboneHead',
  model_cfg=dict(
    type='ExpNormLayerResNet',
    norm_layer='LayerNorm',
    init_cfg=[
      dict(
        type='Kaiming', mode='fan_in',
        layer=['Conv2d', 'Linear'],
      ),
    ],
  ),
  loss_cfg=dict(type='L1Loss'),
)
