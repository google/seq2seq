"""
All graph components that create Variables should inherit from this
base class defined in this file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class GraphModule(object):
  """
  C onvenience class that makes it easy to share variables.
  Each insance of this class creates its own set of variables, but
  each subsequent execution of an instance will re-use its variables.

  Graph components that define variables should inherit from this class
  and implement their logic in the `_build` method.
  """

  def __init__(self, name):
    """
    Initialize the module. Each subclass must call this constructor with a name.

    Args:
      name: Name of this module. Used for `tf.make_template`.
    """
    self.name = name
    self._template = tf.make_template(name, self._build, create_scope_now_=True)
    # Docstrings for the class should be the docstring for the _build method
    self.__doc__ = self._build.__doc__
    # pylint: disable=E1101
    self.__call__.__func__.__doc__ = self._build.__doc__

  def _build(self, *args, **kwargs):
    """Subclasses should implement their logic here.
    """
    raise NotImplementedError

  def __call__(self, *args, **kwargs):
    # pylint: disable=missing-docstring
    return self._template(*args, **kwargs)

  def variable_scope(self):
    """Returns the proper variable scope for this module.
    """
    return tf.variable_scope(self._template.variable_scope)
