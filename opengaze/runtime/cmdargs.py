import argparse


def str2bool(value: str):
  '''Convert string to equivalent boolean value.

  Args:
    `value`: string to be converted.
  '''

  if isinstance(value, bool): return value

  value = value.strip().lower()
  if value in ['true', 't', 'yes', 'y', '1']:
    return True
  elif value in ['false', 'f', 'no', 'n', '0']:
    return False
  else:
    raise argparse.ArgumentTypeError(f'Not a valid boolean value: {value}.')
