_base_ = [
  '../_base_/datasets/mpiigaze.py',
  '../_base_/models/gazenet.py',
  '../_base_/schedules/schedule_20e-adam-warmup-coslr-restart.py',
  '../_base_/default_runtime.py',
]

custom_hooks = [
  dict(
    type='CheckpointHook',
    save_best='mae',
    rule='less',
    save_last=False,
  ),
]


# override values inherited from base configs
train_dataloader = dict(
  dataset=dict(
    transform=[
      dict(type='Grayscale', num_output_channels=3),
      dict(type='ToTensor'),
    ],
  ),
)

val_dataloader = dict(
  dataset=dict(
    transform=[
      dict(type='Grayscale', num_output_channels=3),
      dict(type='ToTensor'),
    ],
  ),
)

test_dataloader = dict(
  dataset=dict(
    transform=[
      dict(type='Grayscale', num_output_channels=3),
      dict(type='ToTensor'),
    ],
  ),
)
