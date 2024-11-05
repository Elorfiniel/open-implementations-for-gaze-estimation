# model settings
model = dict(
  type='MPIIFaceGaze_FullFace',
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
