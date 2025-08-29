from opengaze.registry import LOSSES

from typing import Callable

import numpy as np
import sklearn.mixture as skm
import torch
import torch.nn as nn
import torch.nn.functional as F


LOSSES.register_module(name='L1Loss', module=nn.L1Loss)
LOSSES.register_module(name='MSELoss', module=nn.MSELoss)
LOSSES.register_module(name='SmoothL1Loss', module=nn.SmoothL1Loss)


class _GMMWeightedLoss(nn.Module):
  def __init__(self, weights: list,
               max_loss_clip: float = 20.0,
               bin_ema_alpha: float = 0.1,
               n_hist_bins: int = 50,
               n_resampled: int = 1000,
               n_update_steps: int = 10,
               n_warmup_steps: int = 1000,
               *,
               _loss_fn: Callable = None):
    '''Initialize the GMM-weighted loss module.

    This loss function uses a Gaussian Mixture Model (GMM) to dynamically adjust
    sample weights based on the loss distribution, allowing the model to focus
    differently on samples with different loss values.

    Args:
      `weights`: weight values for different GMM components with increasing mean.

      `max_loss_clip`: maximum loss clipping value.

      `bin_ema_alpha`: exponential moving average, where `X_ema = alpha * X_ema + (1 - alpha) * X`.

      `n_hist_bins`: number of bins in the loss histogram.

      `n_resampled`: number of samples to resample when fitting GMM.

      `n_update_steps`: number of steps/iters between GMM updates.

      `n_warmup_steps`: number of warmup steps/iters before enabling weighted loss.
    '''

    super(_GMMWeightedLoss, self).__init__()

    _weights = torch.tensor(weights, dtype=torch.float32).flatten()
    self.register_buffer('weights', _weights)

    self.loss_fn = _loss_fn or F.mse_loss

    self.max_components = _weights.size(0)

    self.max_loss_clip = max_loss_clip
    self.bin_ema_alpha = bin_ema_alpha
    self.n_hist_bins = n_hist_bins
    self.n_resampled = n_resampled

    _bins = torch.zeros(n_hist_bins, dtype=torch.float32)
    self.register_buffer('bins', _bins)
    _edges = torch.linspace(0, max_loss_clip, n_hist_bins + 1, dtype=torch.float32)
    self.register_buffer('edges', _edges)

    self.n_actual_steps = 0
    self.n_update_steps = n_update_steps
    self.n_warmup_steps = n_warmup_steps

    self.loss_enabled = False
    self.n_components = 0
    self.gmm_components = None

  def _device(self):
    return self.weights.device

  def set_weighted_loss(self, enabled: bool):
    self.loss_enabled = enabled

  def update_steps(self):
    warmup_stage = self.n_actual_steps < self.n_warmup_steps

    update_stage = not warmup_stage and (
      self.n_actual_steps - self.n_warmup_steps
    ) % self.n_update_steps == 0
    self.n_actual_steps = self.n_actual_steps + 1

    return warmup_stage, update_stage

  def update_loss_distribution(self, sample_loss: torch.Tensor):
    sample_loss = torch.clamp(sample_loss, min=0.0, max=self.max_loss_clip)
    bin_index = torch.bucketize(sample_loss, self.edges, right=True) - 1
    bin_index = torch.clamp(bin_index, min=0, max=self.n_hist_bins - 1)
    bin_counts = torch.bincount(bin_index, minlength=self.n_hist_bins)

    if not self.loss_enabled:
      w1, w2 = 1 / self.n_actual_steps, (self.n_actual_steps - 1) / self.n_actual_steps
    else:
      w1, w2 = self.bin_ema_alpha, 1.0 - self.bin_ema_alpha

    self.bins = w1 * bin_counts + w2 * self.bins

  def update_gmm_components(self):
    probs = (self.bins / self.bins.sum()).cpu().numpy()
    bin_edges = self.edges.cpu().numpy()

    bin_indices = np.random.choice(self.n_hist_bins, self.n_resampled, p=probs)
    loss_np = np.zeros(self.n_resampled, dtype=np.float32)
    for it, bin_index in enumerate(bin_indices):
      lhs, rhs = bin_edges[bin_index], bin_edges[bin_index + 1]
      loss_np[it] = np.random.uniform(lhs, rhs)
    loss_np = loss_np.reshape(-1, 1)

    gmms, bics = [], []
    for n_comp in range(1, self.max_components + 1):
      gmm = skm.GaussianMixture(n_components=n_comp)
      gmm.fit(loss_np)
      gmms.append(gmm)
      bics.append(gmm.bic(loss_np))

    best_index = np.argmin(bics)
    self.gmm_components = gmms[best_index]
    self.n_components = best_index + 1

  def generate_sample_weights(self, sample_loss: torch.Tensor):
    if self.gmm_components is None:
      return torch.ones(sample_loss.size(0), dtype=torch.float32)

    loss_np = sample_loss.detach().cpu().numpy().reshape(-1, 1)
    post_probs = self.gmm_components.predict_proba(loss_np)
    means = self.gmm_components.means_.flatten()
    sorted_indices = np.argsort(means)

    sorted_weights = self.weights[sorted_indices].cpu().numpy()
    weights = np.sum(sorted_weights * post_probs, axis=1)

    return torch.tensor(weights, device=self._device())

  def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
    sample_loss = self.loss_fn(y_pred, y_true, reduction='none').sum(dim=1)
    if not self.loss_enabled:
      return sample_loss.mean()

    warmup_stage, update_stage = self.update_steps()
    self.update_loss_distribution(sample_loss)
    if warmup_stage:
      return sample_loss.mean()

    if self.training and update_stage:
      with torch.no_grad():
        self.update_gmm_components()

    weights = self.generate_sample_weights(sample_loss)
    return (weights * sample_loss).mean()


@LOSSES.register_module()
class GMMWeightedL1Loss(_GMMWeightedLoss):
  def __init__(self, *args, **kwargs):
    super(GMMWeightedL1Loss, self).__init__(*args, _loss_fn=F.l1_loss, **kwargs)


@LOSSES.register_module()
class GMMWeightedMSELoss(_GMMWeightedLoss):
  def __init__(self, *args, **kwargs):
    super(GMMWeightedMSELoss, self).__init__(*args, _loss_fn=F.mse_loss, **kwargs)


@LOSSES.register_module()
class GMMWeightedSmoothL1Loss(_GMMWeightedLoss):
  def __init__(self, *args, **kwargs):
    super(GMMWeightedSmoothL1Loss, self).__init__(*args, _loss_fn=F.smooth_l1_loss, **kwargs)
