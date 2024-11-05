# inheritted settings
_base_ = [
  '../_base_/datasets/mpiifacegaze.py',
  '../_base_/models/fullface.py',
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
]

# shared settings
base_batch_size = 256
dataloader = dict(
  dataset=dict(
    transform = [
      # Note: in reality, you might want to consider data augmentation because
      # MPIIFaceGaze is a small dataset (~37.6k samples) for gaze estimation
      #   - For training (eg. RandomAutoContrast, RandomEqualize, ColorJitter)
      #   - For testing (eg. AutoContrast, Equalize)
      # Tips: use dataset wrapper to transform ipts and tgts at the same time
      dict(type='ToTensor'),
      dict(
        type='Normalize',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
      )
    ],
  ),
  batch_size=base_batch_size,
)

# dataloader settings
train_dataloader = dataloader
val_dataloader = dataloader
test_dataloader = dataloader

auto_scale_lr = dict(base_batch_size=base_batch_size, enable=True)

# model settings
model = dict(loss_cfg=dict(type='L1Loss'))

# optimizer settings
optimizer = dict(type='Adam', lr=1e-5, betas=(0.90, 0.95))
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# scheduler settings
param_scheduler = [
  dict(
    type='StepLR',
    by_epoch=False,
    begin=0,
    step_size=2500,
    gamma=0.1,
  ),
]

train_cfg = dict(by_epoch=True, max_epochs=20, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
