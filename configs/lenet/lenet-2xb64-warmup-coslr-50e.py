_base_ = [
  '../_base_/datasets/mpiigaze.py',
  '../_base_/models/lenet.py',
  '../_base_/schedules/schedule_50e-sgd-warmup-coslr.py',
  '../_base_/default_runtime.py',
]

custom_hooks = [
  dict(
    type='CheckpointHook',
    interval=1,
    by_epoch=True,
    save_best='mae',
    rule='less',
    save_last=False,
  ),
]
