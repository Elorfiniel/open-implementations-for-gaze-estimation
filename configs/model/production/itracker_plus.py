# model settings
itracker_plus = dict(
  type='ITrackerPlus',
  init_cfg=dict(
    dict(
      type='Kaiming', mode='fan_in', layer=None,
      override=[
        dict(name='face_fc'),
        dict(name='eyes_fc'),
        dict(name='kpts_fc'),
        dict(name='fc'),
      ],
    ),
  ),
)
