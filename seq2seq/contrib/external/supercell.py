'''

supercell

https://github.com/hardmaru/supercell/

inspired by http://supercell.jp/

'''

#pylint: skip-file

import tensorflow as tf
import numpy as np

from tensorflow.python.framework import function

# Orthogonal Initializer from
# https://github.com/OlavHN/bnlstm
def orthogonal(shape):
  flat_shape = (shape[0], np.prod(shape[1:]))
  a = np.random.normal(0.0, 1.0, flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  q = u if u.shape == flat_shape else v
  return q.reshape(shape)

def lstm_ortho_initializer(scale=1.0):
  def _initializer(shape, dtype=tf.float32, partition_info=None):
    size_x = shape[0]
    size_h = shape[1]/4 # assumes lstm.
    t = np.zeros(shape)
    t[:, :size_h] = orthogonal([size_x, size_h])*scale
    t[:, size_h:size_h*2] = orthogonal([size_x, size_h])*scale
    t[:, size_h*2:size_h*3] = orthogonal([size_x, size_h])*scale
    t[:, size_h*3:] = orthogonal([size_x, size_h])*scale
    return tf.constant(t, dtype)
  return _initializer

def layer_norm_all_op_get_var(num_units, base, scope='ln_gamma'):
  return tf.get_variable(scope, [base*num_units], initializer=tf.constant_initializer(1.0))

@function.Defun(*[tf.float32] * 2 + [tf.int32]*3, func_name='layer_norm_all_op')
def layer_norm_all_op(h, gamma, batch_size, base, num_units):
  # Layer Norm (faster version)
  #
  # Performas layer norm on multiple base at once (ie, i, g, j, o for lstm)
  #
  # Reshapes h in to perform layer norm in parallel
  h_reshape = tf.reshape(h, [batch_size, base, num_units])
  mean = tf.reduce_mean(h_reshape, [2], keep_dims=True)
  var = tf.reduce_mean(tf.square(h_reshape - mean), [2], keep_dims=True)
  epsilon = tf.constant(1e-3)
  rstd = tf.rsqrt(var + epsilon)
  h_reshape = (h_reshape - mean) * rstd
  # reshape back to original
  h = tf.reshape(h_reshape, [batch_size, base * num_units])
  return gamma * h

def layer_norm_all(h, batch_size, base, num_units, scope="layer_norm", reuse=False, gamma_start=1.0, epsilon = 1e-3):
  # Layer Norm (faster version, but not using defun)
  #
  # Performas layer norm on multiple base at once (ie, i, g, j, o for lstm)
  #
  # Reshapes h in to perform layer norm in parallel
  h_reshape = tf.reshape(h, [batch_size, base, num_units])
  mean = tf.reduce_mean(h_reshape, [2], keep_dims=True)
  var = tf.reduce_mean(tf.square(h_reshape - mean), [2], keep_dims=True)
  epsilon = tf.constant(epsilon)
  rstd = tf.rsqrt(var + epsilon)
  h_reshape = (h_reshape - mean) * rstd
  # reshape back to original
  h = tf.reshape(h_reshape, [batch_size, base * num_units])
  with tf.variable_scope(scope):
    if reuse == True:
      tf.get_variable_scope().reuse_variables()
    gamma = tf.get_variable('ln_gamma', [4*num_units], initializer=tf.constant_initializer(gamma_start))
  return gamma * h

def layer_norm(x, num_units, scope="layer_norm", reuse=False, gamma_start=1.0, epsilon = 1e-3):
  axes = [1]
  mean = tf.reduce_mean(x, axes, keep_dims=True)
  x_shifted = x-mean
  var = tf.reduce_mean(tf.square(x_shifted), axes, keep_dims=True)
  inv_std = tf.rsqrt(var + epsilon)
  with tf.variable_scope(scope):
    if reuse == True:
      tf.get_variable_scope().reuse_variables()
    gamma = tf.get_variable('ln_gamma', [num_units], initializer=tf.constant_initializer(gamma_start))
  output = gamma*(x_shifted)*inv_std
  return output

def super_linear(x, output_size, scope=None, reuse=False,
  init_w="ortho", weight_start=0.0, use_bias=True, bias_start=0.0, input_size=None):
  # support function doing linear operation.  uses ortho initializer defined earlier.
  shape = x.get_shape().as_list()
  with tf.variable_scope(scope or "linear"):
    if reuse == True:
      tf.get_variable_scope().reuse_variables()

    w_init = None # uniform
    if input_size == None:
      x_size = shape[1]
    else:
      x_size = input_size
    h_size = output_size
    if init_w == "zeros":
      w_init=tf.constant_initializer(0.0)
    elif init_w == "constant":
      w_init=tf.constant_initializer(weight_start)
    elif init_w == "gaussian":
      w_init=tf.random_normal_initializer(stddev=weight_start)
    elif init_w == "ortho":
      w_init=lstm_ortho_initializer(1.0)

    w = tf.get_variable("super_linear_w",
      [x_size, output_size], tf.float32, initializer=w_init)
    if use_bias:
      b = tf.get_variable("super_linear_b", [output_size], tf.float32,
        initializer=tf.constant_initializer(bias_start))
      return tf.matmul(x, w) + b
    return tf.matmul(x, w)

def hyper_norm(layer, hyper_output, embedding_size, num_units,
               scope="hyper", use_bias=True):
  '''
  HyperNetwork norm operator
  
  provides context-dependent weights

  layer: layer to apply operation on
  hyper_output: output of the hypernetwork cell at time t
  embedding_size: embedding size of the output vector (see paper)
  num_units: number of hidden units in main rnn
  '''
  # recurrent batch norm init trick (https://arxiv.org/abs/1603.09025).
  init_gamma = 0.10 # cooijmans' da man.
  with tf.variable_scope(scope):
    zw = super_linear(hyper_output, embedding_size, init_w="constant",
      weight_start=0.00, use_bias=True, bias_start=1.0, scope="zw")
    alpha = super_linear(zw, num_units, init_w="constant",
      weight_start=init_gamma / embedding_size, use_bias=False, scope="alpha")
    result = tf.mul(alpha, layer)
  return result

def hyper_bias(layer, hyper_output, embedding_size, num_units,
               scope="hyper"):
  '''
  HyperNetwork norm operator
  
  provides context-dependent bias

  layer: layer to apply operation on
  hyper_output: output of the hypernetwork cell at time t
  embedding_size: embedding size of the output vector (see paper)
  num_units: number of hidden units in main rnn
  '''

  with tf.variable_scope(scope):
    zb = super_linear(hyper_output, embedding_size, init_w="gaussian",
      weight_start=0.01, use_bias=False, bias_start=0.0, scope="zb")
    beta = super_linear(zb, num_units, init_w="constant",
      weight_start=0.00, use_bias=False, scope="beta")
  return layer + beta
  
class LSTMCell(tf.contrib.rnn.RNNCell):
  """
  Layer-Norm, with Ortho Initialization and
  Recurrent Dropout without Memory Loss.
  https://arxiv.org/abs/1607.06450 - Layer Norm
  https://arxiv.org/abs/1603.05118 - Recurrent Dropout without Memory Loss
  derived from
  https://github.com/OlavHN/bnlstm
  https://github.com/LeavesBreathe/tensorflow_with_latest_papers
  """

  def __init__(self, num_units, forget_bias=1.0, use_layer_norm=False,
    use_recurrent_dropout=False, dropout_keep_prob=0.90):
    """Initialize the Layer Norm LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (default 1.0).
      use_recurrent_dropout: float, Whether to use Recurrent Dropout (default False)
      dropout_keep_prob: float, dropout keep probability (default 0.90)
    """
    self.num_units = num_units
    self.forget_bias = forget_bias
    self.use_layer_norm = use_layer_norm
    self.use_recurrent_dropout = use_recurrent_dropout
    self.dropout_keep_prob = dropout_keep_prob

  # @property
  # def input_size(self):
  #   return self._input_size

  @property
  def output_size(self):
    return self.num_units

  @property
  def state_size(self):
    return tf.contrib.rnn.LSTMStateTuple(self.num_units, self.num_units)

  def __call__(self, x, state, scope=None):
    with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
      c, h = state

      h_size = self.num_units
      
      batch_size =  x.get_shape().as_list()[0]
      x_size = x.get_shape().as_list()[-1]
      
      w_init=None # uniform

      h_init=lstm_ortho_initializer()

      W_xh = tf.get_variable('W_xh',
        [x_size, 4 * self.num_units], initializer=w_init)

      W_hh = tf.get_variable('W_hh_i',
        [self.num_units, 4*self.num_units], initializer=h_init)

      W_full = tf.concat_v2([W_xh, W_hh], 0)

      bias = tf.get_variable('bias',
        [4 * self.num_units], initializer=tf.constant_initializer(0.0))

      concat = tf.concat_v2([x, h], 1) # concat for speed.
      concat = tf.matmul(concat, W_full) + bias
      
      # new way of doing layer norm (faster)
      if self.use_layer_norm:
        concat = layer_norm_all(concat, batch_size, 4, self.num_units, 'ln')

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = tf.split(concat, 4, 1)

      if self.use_recurrent_dropout:
        g = tf.nn.dropout(tf.tanh(j), self.dropout_keep_prob)
      else:
        g = tf.tanh(j) 

      new_c = c*tf.sigmoid(f+self.forget_bias) + tf.sigmoid(i)*g
      if self.use_layer_norm:
        new_h = tf.tanh(layer_norm(new_c, self.num_units, 'ln_c')) * tf.sigmoid(o)
      else:
        new_h = tf.tanh(new_c) * tf.sigmoid(o)
    
    return new_h, tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

class HyperLSTMCell(tf.contrib.rnn.RNNCell):
  '''
  HyperLSTM, with Ortho Initialization,
  Layer Norm and Recurrent Dropout without Memory Loss.
  
  https://arxiv.org/abs/1609.09106
  '''

  def __init__(self, num_units, forget_bias=1.0,
    use_recurrent_dropout=False, dropout_keep_prob=0.90, use_layer_norm=True,
    hyper_num_units=128, hyper_embedding_size=16,
    hyper_use_recurrent_dropout=False):
    '''Initialize the Layer Norm HyperLSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (default 1.0).
      use_recurrent_dropout: float, Whether to use Recurrent Dropout (default False)
      dropout_keep_prob: float, dropout keep probability (default 0.90)
      use_layer_norm: boolean. (default True)
        Controls whether we use LayerNorm layers in main LSTM and HyperLSTM cell.
      hyper_num_units: int, number of units in HyperLSTM cell.
        (default is 128, recommend experimenting with 256 for larger tasks)
      hyper_embedding_size: int, size of signals emitted from HyperLSTM cell.
        (default is 4, recommend trying larger values but larger is not always better)
      hyper_use_recurrent_dropout: boolean. (default False)
        Controls whether HyperLSTM cell also uses recurrent dropout. (Not in Paper.)
        Recommend turning this on only if hyper_num_units becomes very large (>= 512)
    '''
    self.num_units = num_units
    self.forget_bias = forget_bias
    self.use_recurrent_dropout = use_recurrent_dropout
    self.dropout_keep_prob = dropout_keep_prob
    self.use_layer_norm = use_layer_norm
    self.hyper_num_units = hyper_num_units
    self.hyper_embedding_size = hyper_embedding_size
    self.hyper_use_recurrent_dropout = hyper_use_recurrent_dropout

    self.total_num_units = self.num_units + self.hyper_num_units

    self.hyper_cell=LSTMCell(hyper_num_units,
                             use_recurrent_dropout=hyper_use_recurrent_dropout,
                             use_layer_norm=use_layer_norm,
                             dropout_keep_prob=dropout_keep_prob)

  @property
  def output_size(self):
    return self.num_units

  @property
  def state_size(self):
    return tf.contrib.rnn.LSTMStateTuple(self.num_units+self.hyper_num_units,
                                         self.num_units+self.hyper_num_units)

  def __call__(self, x, state, timestep = 0, scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      total_c, total_h = state
      c = total_c[:, 0:self.num_units]
      h = total_h[:, 0:self.num_units]
      hyper_state = tf.contrib.rnn.LSTMStateTuple(total_c[:,self.num_units:],
                                                  total_h[:,self.num_units:])

      w_init=None # uniform

      h_init=lstm_ortho_initializer(1.0)
      
      x_size = x.get_shape().as_list()[-1]
      embedding_size = self.hyper_embedding_size
      num_units = self.num_units
      batch_size = x.get_shape().as_list()[0]

      W_xh = tf.get_variable('W_xh',
        [x_size, 4*num_units], initializer=w_init)
      W_hh = tf.get_variable('W_hh',
        [num_units, 4*num_units], initializer=h_init)
      bias = tf.get_variable('bias',
        [4*num_units], initializer=tf.constant_initializer(0.0))

      # concatenate the input and hidden states for hyperlstm input
      hyper_input = tf.concat_v2([x, h], 1)
      hyper_output, hyper_new_state = self.hyper_cell(hyper_input, hyper_state)

      xh = tf.matmul(x, W_xh)
      hh = tf.matmul(h, W_hh)

      # split Wxh contributions
      ix, jx, fx, ox = tf.split(xh, 4, 1)
      ix = hyper_norm(ix, hyper_output, embedding_size, num_units, 'hyper_ix')
      jx = hyper_norm(jx, hyper_output, embedding_size, num_units, 'hyper_jx')
      fx = hyper_norm(fx, hyper_output, embedding_size, num_units, 'hyper_fx')
      ox = hyper_norm(ox, hyper_output, embedding_size, num_units, 'hyper_ox')

      # split Whh contributions
      ih, jh, fh, oh = tf.split(hh, 4, 1)
      ih = hyper_norm(ih, hyper_output, embedding_size, num_units, 'hyper_ih')
      jh = hyper_norm(jh, hyper_output, embedding_size, num_units, 'hyper_jh')
      fh = hyper_norm(fh, hyper_output, embedding_size, num_units, 'hyper_fh')
      oh = hyper_norm(oh, hyper_output, embedding_size, num_units, 'hyper_oh')

      # split bias
      ib, jb, fb, ob = tf.split(bias, 4, 0) # bias is to be broadcasted.
      ib = hyper_bias(ib, hyper_output, embedding_size, num_units, 'hyper_ib')
      jb = hyper_bias(jb, hyper_output, embedding_size, num_units, 'hyper_jb')
      fb = hyper_bias(fb, hyper_output, embedding_size, num_units, 'hyper_fb')
      ob = hyper_bias(ob, hyper_output, embedding_size, num_units, 'hyper_ob')

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i = ix + ih + ib
      j = jx + jh + jb
      f = fx + fh + fb
      o = ox + oh + ob

      if self.use_layer_norm:
        concat = tf.concat_v2([i, j, f, o], 1)
        concat = layer_norm_all(concat, batch_size, 4, num_units, 'ln_all')
        i, j, f, o = tf.split(concat, 4, 1)

      if self.use_recurrent_dropout:
        g = tf.nn.dropout(tf.tanh(j), self.dropout_keep_prob)
      else:
        g = tf.tanh(j) 

      new_c = c*tf.sigmoid(f+self.forget_bias) + tf.sigmoid(i)*g
      if self.use_layer_norm:
        new_h = tf.tanh(layer_norm(new_c, num_units, 'ln_c')) * tf.sigmoid(o)
      else:
        new_h = tf.tanh(new_c) * tf.sigmoid(o)
    
      hyper_c, hyper_h = hyper_new_state
      new_total_c = tf.concat_v2([new_c, hyper_c], 1)
      new_total_h = tf.concat_v2([new_h, hyper_h], 1)

    return new_h, tf.contrib.rnn.LSTMStateTuple(new_total_c, new_total_h)