from mmengine.evaluator import BaseMetric

from opengaze.registry import METRICS
from opengaze.utils.euler import gaze_2d_3d_t

import torch


@METRICS.register_module()
class AngularError(BaseMetric):
  def _gaze_2d_to_3d(self, gaze_2d: torch.Tensor):
    x, y, z = gaze_2d_3d_t(gaze_2d[:, 0], gaze_2d[:, 1])
    return torch.stack([x, y, z], dim=1)

  def process(self, data_batch, data_samples):
    preds_2d, gazes_2d = data_samples
    preds_3d = self._gaze_2d_to_3d(preds_2d)
    gazes_3d = self._gaze_2d_to_3d(gazes_2d)

    dot = torch.sum(preds_3d * gazes_3d, dim=1)
    m_p = torch.linalg.vector_norm(preds_3d, ord=2, dim=1)
    m_g = torch.linalg.vector_norm(gazes_3d, ord=2, dim=1)

    # Clamp cosine similarity to [-1, 1], to avoid NaNs
    # caused by numeric error, eg. 1.0000001
    sims = torch.clamp(dot / (m_p * m_g), min=-1.0, max=1.0)

    degs = torch.rad2deg(torch.acos(sims))
    errs = torch.mean(degs).cpu()

    self.results.append(dict(errs=errs))

  def compute_metrics(self, results):
    return dict(mae=sum([r['errs'] for r in results]) / len(results))


@METRICS.register_module()
class DistanceError(BaseMetric):
  def process(self, data_batch, data_samples):
    preds_2d, gazes_2d = data_samples

    dists = torch.linalg.vector_norm(preds_2d - gazes_2d, dim=1)
    errs = torch.mean(dists).cpu()

    self.results.append(dict(errs=errs))

  def compute_metrics(self, results):
    return dict(mae=sum([r['errs'] for r in results]) / len(results))
