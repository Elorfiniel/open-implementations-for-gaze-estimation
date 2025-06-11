from mmengine.model import BaseModel

from opengaze.registry import MODELS, LOSSES

import torch as torch


class DataFnMixin:
  def data_fn(self, data_dict: dict):
    '''Takes as input the data dict from mmengine, and returns the actual
    data dict that the wrapped model expects.

    Args:
      `data_dict`: a dictionary of data for batch samples.
    '''

    return data_dict


@MODELS.register_module()
class BackboneHead(BaseModel):
  '''Model wrapper for Backbone-Head architecture, which takes many input streams,
  ie. face image, face bbox, and outputs the gaze prediction for each sample.
  '''

  def __init__(self, model_cfg: dict, loss_cfg: dict):
    '''Model wrapper for Backbone-Head architecture.

    Args:
      `model_cfg`: configuration dict for registered models of type `BaseModel`.
      `loss_cfg`: configuration dict for registered loss functions.

    Note that `data_fn` processes the input data dict from mmengine, then passes
    the actual data dict to the wrapped model. When wrapping a model, remember
    to provide the actual implementation of `data_fn`. The default `data_fn` simply
    returns the input data dict (no-op transformation).
    '''

    super(BackboneHead, self).__init__()

    self.model: DataFnMixin = MODELS.build(model_cfg)
    self.loss_fn = LOSSES.build(loss_cfg)

  def forward(self, mode='tensor', **data_dict):
    '''Parse actual data dict from the input data dict provided by mmengine,
    then runs the forward pass of the wrapped model. This method bridges the
    difference between mmengine and the wrapped model, however, it requires
    that the wrapped model specifys its inputs in a kwargs style.

    Args:
      `mode`: mode of forward pass, see `BaseModel.forward` for more details.
      `data_dict`: input data dict provided by mmengine.

    For convenience, `data_dict['gaze']` provides the ground-truth gaze label,
    see the implementation of gaze datasets for more details.
    '''

    actual_data_dict = self.model.data_fn(data_dict)
    gaze = self.model(**actual_data_dict)

    if mode == 'loss':
      loss = self.loss_fn(gaze, data_dict['gaze'])
      return dict(loss=loss)

    if mode == 'predict':
      return gaze, data_dict['gaze']

    return gaze
