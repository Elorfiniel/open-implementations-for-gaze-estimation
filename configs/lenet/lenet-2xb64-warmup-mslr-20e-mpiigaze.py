_base_ = [
  '../_base_/datasets/mpiigaze.py',
  '../_base_/models/lenet.py',
  '../_base_/schedules/schedule_20e-sgd-warmup-mslr.py',
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
