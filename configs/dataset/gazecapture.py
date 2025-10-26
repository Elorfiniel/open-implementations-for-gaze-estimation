# image transformation
transform = [
  dict(type='ToTensor'),
  dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]

# dataset settings
train = dict(
  type='GazeCapture',
  root='data/mit-gaze-capture',
  split='train',
  devices=[
    'iPad 2', 'iPad 3', 'iPad 4', 'iPad Air', 'iPad Air 2', 'iPad Mini', 'iPad Pro',
    'iPhone 4S', 'iPhone 5', 'iPhone 5C', 'iPhone 5S',
    'iPhone 6', 'iPhone 6 Plus', 'iPhone 6s', 'iPhone 6s Plus',
  ],
  transform=transform,
)

valid = dict(
  type='GazeCapture',
  root='data/mit-gaze-capture',
  split='val',
  devices=[
    'iPad 2', 'iPad 3', 'iPad 4', 'iPad Air', 'iPad Air 2', 'iPad Mini', 'iPad Pro',
    'iPhone 4S', 'iPhone 5', 'iPhone 5C', 'iPhone 5S',
    'iPhone 6', 'iPhone 6 Plus', 'iPhone 6s', 'iPhone 6s Plus',
  ],
  transform=transform,
)

test = dict(
  type='GazeCapture',
  root='data/mit-gaze-capture',
  split='test',
  devices=[
    'iPad 2', 'iPad 3', 'iPad 4', 'iPad Air', 'iPad Air 2', 'iPad Mini', 'iPad Pro',
    'iPhone 4S', 'iPhone 5', 'iPhone 5C', 'iPhone 5S',
    'iPhone 6', 'iPhone 6 Plus', 'iPhone 6s', 'iPhone 6s Plus',
  ],
  transform=transform,
)
