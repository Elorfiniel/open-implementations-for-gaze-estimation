default_scope = 'template'

env_cfg = dict(
  cudnn_benchmark=False,
  mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
  dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]

log_processor = dict(type='LogProcessor', by_epoch=True)
log_level = 'INFO'

load_from = None
resume = False
