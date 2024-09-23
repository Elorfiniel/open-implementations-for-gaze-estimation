from mmengine.evaluator import BaseMetric

from template.registry import METRICS

import torch as torch


@METRICS.register_module()
class AngularError_PitchYaw(BaseMetric):
  def _gaze_2d_to_3d(self, gaze_2d: torch.Tensor):
    x = -torch.cos(gaze_2d[:, 0]) * torch.sin(gaze_2d[:, 1])
    y = -torch.sin(gaze_2d[:, 0])
    z = -torch.cos(gaze_2d[:, 0]) * torch.cos(gaze_2d[:, 1])
    return torch.stack([x, y, z], dim=1)

  def process(self, data_batch, data_samples):
    preds_2d, gazes_2d = data_samples
    preds_3d = self._gaze_2d_to_3d(preds_2d)
    gazes_3d = self._gaze_2d_to_3d(gazes_2d)

    dot = torch.sum(preds_3d * gazes_3d, dim=1)
    m_p = torch.linalg.vector_norm(preds_3d, ord=2, dim=1)
    m_g = torch.linalg.vector_norm(gazes_3d, ord=2, dim=1)

    degs = torch.rad2deg(torch.acos(dot / (m_p * m_g)))
    errs = torch.mean(degs).cpu()

    self.results.append(dict(errs=errs))

  def compute_metrics(self, results):
    return dict(mae=sum([r['errs'] for r in results]) / len(results))
