# optimizer settings
optimizer = dict(
  type='Adam',
  lr=1e-3,
  weight_decay=0.005,
  betas=(0.90, 0.95),
)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# scheduler settings
param_scheduler = [
  dict(
    type='LinearLR',
    start_factor=0.1,
    begin=0,
    end=2,
    by_epoch=True,
    convert_to_iter_based=True,
  ),
  dict(
    type='CosineRestartLR',
    periods=[2 for _ in range(2, 20, 2)],
    restart_weights=[1.0 for _ in range(2, 20, 2)],
    eta_min_ratio=0.1,
    begin=2,
    by_epoch=True,
    convert_to_iter_based=True,
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
