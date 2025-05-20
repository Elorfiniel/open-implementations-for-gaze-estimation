from opengaze.engine.transform.base import BaseTransform
from opengaze.registry import TRANSFORMS

import numpy as np
import torch


@TRANSFORMS.register_module()
class ITrackerMakeGrid(BaseTransform):
  def __init__(self, grid_size: int = 25):
    super(ITrackerMakeGrid, self).__init__()
    self.grid_size = grid_size

  def transform(self, results):
    '''Make face grid from face bbox, see official implementation of iTracker:
    https://github.com/CSAILVision/GazeCapture/blob/master/pytorch/ITrackerData.py
    '''

    x, y, w, h = results['grid'].tolist()

    grid_length = self.grid_size * self.grid_size
    grid = np.zeros(grid_length, dtype=np.float32)

    ind_x = np.array([i % self.grid_size for i in range(grid_length)])
    ind_y = np.array([i // self.grid_size for i in range(grid_length)])

    cond_x = np.logical_and(ind_x >= x, ind_x < x + w)
    cond_y = np.logical_and(ind_y >= y, ind_y < y + h)

    grid[np.logical_and(cond_x, cond_y)] = 1.0
    results['grid'] = torch.tensor(grid, dtype=torch.float32)

    return results
