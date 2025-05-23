from opengaze.runtime.scripts import ScriptEnv
from opengaze.runtime.log import runtime_logger

import concurrent.futures as futures


__all__ = ['FunctionalTask', 'run_parallel', 'submit_functional_task']


class FunctionalTask:
  def __init__(self, task_fn, *args, done_fn=None, **kwargs):
    self.task_fn = task_fn
    self.args = args
    self.kwargs = kwargs
    self.done_fn = done_fn


def _wrap_done_fn(task: FunctionalTask, logger=None):
  if logger is None:
    logger = runtime_logger(
      name='parallel',
      log_file=ScriptEnv.log_path('parallel.log'),
    )

  if task.done_fn is None:
    done_fn = lambda f: f.result()
  else:
    done_fn = task.done_fn

  def wrapper(future: futures.Future):
    try:
      done_fn(future) # Anticipating potential exceptions
    except Exception:
      err_msg = f'task {task.task_fn}: args {task.args}, kwargs {task.kwargs}'
      logger.error(err_msg, exc_info=True)

  return wrapper


def submit_functional_task(task: FunctionalTask, executor, logger=None):
  '''Submit a functional task to the executor.

  Args:
    `task`: an instance of `FunctionalTask`.
    `executor`: a process pool executor.
    `logger`: an instance of `logging.Logger`.
  '''

  future = executor.submit(task.task_fn, *task.args, **task.kwargs)
  done_fn = _wrap_done_fn(task, logger=logger)
  future.add_done_callback(done_fn)


def run_parallel(executor, tasks, logger=None):
  '''Run tasks in parallel, scheduled in order on the executor.

  Args:
    `executor`: process pool executor used for parallel tasks.
    `tasks`: an iterable of `FunctionalTask`.
    `logger`: an instance of `logging.Logger` for logging.
  '''

  with executor:
    for task in tasks:
      submit_functional_task(task, executor, logger)
