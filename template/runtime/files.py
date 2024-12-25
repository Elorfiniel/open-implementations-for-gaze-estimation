import os.path as osp


class ProjectTree:

  ROOT = osp.dirname(osp.dirname(osp.dirname(__file__)))

  @staticmethod
  def resource_path(resource: str):
    '''Build resource path by prepending the resource folder.'''
    return osp.join(ProjectTree.ROOT, 'resource', resource)

  @staticmethod
  def data_path(dataset: str):
    '''Build data path by prepending the data folder.'''
    return osp.join(ProjectTree.ROOT, 'data', dataset)

  @staticmethod
  def logfile(filename: str):
    '''Build log file path by prepending the logs folder.'''
    return osp.join(ProjectTree.ROOT, 'logs', filename)
