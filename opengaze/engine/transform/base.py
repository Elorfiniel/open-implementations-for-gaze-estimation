from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple, Union


class BaseTransform(metaclass=ABCMeta):
  """Base class for all transformations (using API from `mmcv`)."""

  def __call__(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
    return self.transform(results)

  @abstractmethod
  def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
    """The transform function. Subclass should override this method.

    See also: https://mmcv.readthedocs.io/zh-cn/2.x/api/transforms.html
    """
