# inheritted settings
_base_ = [
  '../_base_/datasets/mpiigaze.py',
  '../_base_/models/gazenet.py',
  '../_base_/default_runtime.py',
]

# custom hook settings
custom_hooks = [
  dict(
    type='CheckpointHook',
    save_best='mae',
    rule='less',
    save_last=False,
  ),
  dict(type='EMAHook'),
]

# shared settings
base_batch_size = 256
dataloader = dict(
  dataset=dict(
    root='data/mpiigaze/normalized-gen',
    transform = [
      dict(type='Grayscale', num_output_channels=3),
      dict(type='ToTensor'),
    ],
  ),
  batch_size=base_batch_size,
)

# dataloader settings
train_dataloader = dataloader
val_dataloader = dataloader
test_dataloader = dataloader

auto_scale_lr = dict(base_batch_size=base_batch_size, enable=True)

# optimizer settings
optimizer = dict(type='Adam', lr=1e-5, betas=(0.90, 0.95))
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# scheduler settings
param_scheduler = [
  dict(
    type='StepLR',
    by_epoch=False,
    begin=0,
    step_size=5000,
    gamma=0.1,
  ),
]

train_cfg = dict(by_epoch=True, max_epochs=10, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
