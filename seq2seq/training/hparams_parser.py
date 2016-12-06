import argparse

class HParamsParser(object):
  """Pases a comma-separated string of hyperaprameters
  """

  def __init__(self, default_params):
    self.default_params = default_params
    self.parser = self._create_parser()

  def _create_parser(self):
    parser = argparse.ArgumentParser()
    for key, value in self.default_params.items():
      parser.add_argument("--{}".format(key), type=type(value), default=value)
    return parser

  def parse(self, params_str):
    """Passes a parameter strng and returns a dictionary of values"""
    params = params_str.split(",")
    params = ["--" + param for param in params]
    return vars(self.parser.parse_args(params))
