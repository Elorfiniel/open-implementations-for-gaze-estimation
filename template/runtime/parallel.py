import concurrent.futures as futures
import logging


__all__ = ['FunctionalTask', 'run_parallel', 'submit_functional_task']


def _wrap_done_fn(done_fn, _task_fn, _args, _kwargs):
  if done_fn is None:
    done_fn = lambda f: f.result()

  def wrapper(future: futures.Future):
    try:
      done_fn(future) # Anticipating potential exceptions
    except Exception:
      err_msg = f'task {_task_fn}: args {_args}, kwargs {_kwargs}'
      logging.error(err_msg, exc_info=True)

  return wrapper


class FunctionalTask:
  def __init__(self, task_fn, *args, done_fn=None, **kwargs):
    self.task_fn = task_fn
    self.args = args
    self.kwargs = kwargs
    self.done_fn = done_fn


def submit_functional_task(task: FunctionalTask, executor: futures.ProcessPoolExecutor):
  '''Submit a functional task to the executor.'''

  future = executor.submit(task.task_fn, *task.args, **task.kwargs)
  done_fn = _wrap_done_fn(task.done_fn, task.task_fn, task.args, task.kwargs)
  future.add_done_callback(done_fn)

def run_parallel(executor: futures.ProcessPoolExecutor, tasks):
  '''Run tasks in parallel, scheduled in order on the executor.

  Args:
    `executor`: process pool executor used for parallel tasks
    `tasks`: a generator of tasks (FunctionalTask)
  '''

  with executor:
    for task in tasks:
      submit_functional_task(task, executor)
