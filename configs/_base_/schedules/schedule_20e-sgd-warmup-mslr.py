# optimizer settings
optimizer = dict(
  type='SGD',
  lr=1e-3,
  momentum=0.90,
  weight_decay=0.002,
  nesterov=True,
)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# scheduler settings
param_scheduler = [
  dict(
    type='LinearLR',
    start_factor=0.1,
    begin=0,
    end=5,
    by_epoch=True,
    convert_to_iter_based=True,
  ),
  dict(
    type='MultiStepLR',
    gamma=0.316,
    milestones=[12, 16],
    begin=5,
    end=20,
    by_epoch=True,
  ),
]

# runner schedule settings
train_cfg = dict(by_epoch=True, max_epochs=20, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# default hooks
default_hooks = dict(
  timer=dict(type='IterTimerHook'),
  logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
  param_scheduler=dict(type='ParamSchedulerHook'),
  checkpoint=dict(type='CheckpointHook', interval=5, by_epoch=True),
)
