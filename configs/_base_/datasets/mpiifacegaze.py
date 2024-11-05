# dataloader settings
train_dataloader = dict(
  dataset=dict(
    type='MPIIFaceGaze',
    root='data/mpiifacegaze/normalized-gen',
    train=True,
    test_pp='p00',
    transform=dict(type='ToTensor'),
  ),
  num_workers=4,
  batch_size=64,
  sampler=dict(
    type='DefaultSampler',
    shuffle=True,
  ),
  collate_fn=dict(type='default_collate'),
)

val_dataloader = dict(
  dataset=dict(
    type='MPIIFaceGaze',
    root='data/mpiifacegaze/normalized-gen',
    train=False,
    test_pp='p00',
    transform=dict(type='ToTensor'),
  ),
  num_workers=4,
  batch_size=64,
  sampler=dict(
    type='DefaultSampler',
    shuffle=False,
  ),
  collate_fn=dict(type='default_collate'),
)

test_dataloader = val_dataloader

# evaluator settings
val_evaluator = dict(type='AngularError_PitchYaw')
test_evaluator = dict(type='AngularError_PitchYaw')
