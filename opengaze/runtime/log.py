# TODO: Replace this trivial logger with a better one (e.g. loguru)

import logging
import logging.config


__all__ = ['runtime_logger']


LOGGING_CONFIG = {
  'version': 1,
  'disable_existing_loggers': False,
  'formatters': {
    'simple': {
      'format': '[ %(asctime)s ] [ %(name)s ] process %(process)d - %(levelname)s: %(message)s',
      'datefmt': '%Y-%m-%dT%H:%M:%S',
    },
  },
  'handlers': {
    'console': {
      'class': 'logging.StreamHandler',
      'level': 'INFO',
      'formatter': 'simple',
    },
  },
  'loggers': {},  # Runtime loggers
}


def runtime_logger(name: str, level: int = logging.INFO):
  '''Create a runtime logger with given name and level.'''

  if not name in LOGGING_CONFIG['loggers']:
    LOGGING_CONFIG['loggers'][name] = dict(
      level=level,
      handlers=['console'],
      propagate=False,
    )
    logging.config.dictConfig(LOGGING_CONFIG)

  return logging.getLogger(name)
