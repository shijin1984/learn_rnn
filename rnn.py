#!/usr/bin/python

import math
import numpy as np

class RnnCell:

  class Params:
    '''The wrapper of RNN cell params.
    '''
    def __init__(self, Wh, Uh, bh, Wy, by):
      '''Constructs the cell. The dimensions are (given the dimension of h is H):
         Wh: 28*H  Uh: H*H,  bh: H*1
         Wy: 10*H,  by: 10*1

      Args:
        The matrix and bias as in the Elman network.
      '''
      _array = lambda x: x if isinstance(x, np.ndarray) else np.array(x)
      dim_h = len(bh)
      assert Wh.shape == (dim_h, 28)
      self.Wh = _array(Wh)
      assert Uh.shape == (dim_h, dim_h)
      self.Uh = _array(Uh)
      self.bh = _array(bh)
      assert Wy.shape == (10, dim_h)
      self.Wy = _array(Wy)
      assert len(by) == 10
      self.by = _array(by)

    def zeros(self):
      '''Creates another Params instance in the same size and all zeros.
      '''
      return self.__class__(
          Wh=np.zeros(self.Wh.shape),
          Uh=np.zeros(self.Uh.shape),
          bh=np.zeros(self.bh.shape),
          Wy=np.zeros(self.Wy.shape),
          by=np.zeros(self.by.shape))

    def __iadd__(self, other):
      '''The += operator.
      '''
      self.Wh += other.Wh
      self.Uh += other.Uh
      self.bh += other.bh
      self.Wy += other.Wy
      self.by += other.by
      return self

    def __imul__(self, rhs):
      '''The *= operator.
      '''
      self.Wh *= rhs
      self.Uh *= rhs
      self.bh *= rhs
      self.Wy *= rhs
      self.by *= rhs
      return self

    def __iter__(self):
      return iter([self.Wh, self.Uh, self.bh, self.Wy, self.by])

    def inf_norm(self):
      a = max(abs(np.max(x)) for x in self)
      b = max(abs(np.min(x)) for x in self)
      return max(a, b)


  def __init__(self, Wh, Uh, bh, Wy, by):
    self._params = self.Params(Wh, Uh, bh, Wy, by)


  def forward(self, x, h):
    '''Calculates the y(t) and h(t) from x=x(t) and h=h(t-1).
    '''
    z_h = np.matmul(self._params.Wh, x) + np.matmul(self._params.Uh, h) + self._params.bh
    hh = self._relu(z_h)
    y = self._relu(np.matmul(self._params.Wy, hh) + self._params.by)
    return y, hh


  def backward(self, x, h, y, hh, grad_y, grad_hh):
    '''Calculates the gradients.

    Args:
      x, h: the input of the cell x(t) and h(t-1).
      y, hh: the output of the cell y(t) and h(t)
      grad_y, grad_hh: the gradients on the output y(t) and h(t).

    Returns:
      grad_params, grad_h: the gradients of parameters of (Wh, Uh, bh, Wy, by)
        and grad_h to propagate to the previous cell.
    '''
    # For y = ReLU (W_y h(t) + b_y)
    grad_y = self._grad_relu(y, grad_y)
    grad_by = grad_y
    grad_Wy, grad_y2hh = self._grad_mv(self._params.Wy, hh, grad_y)

    grad_hh = grad_hh + grad_y2hh
    # For h = ReLU(W_h x + U_h h(t-1) + b_h)
    grad_hh = self._grad_relu(hh, grad_hh)
    grad_bh = grad_hh
    grad_Wh, _ = self._grad_mv(self._params.Wh, x, grad_hh)
    grad_Uh, grad_h = self._grad_mv(self._params.Uh, h, grad_hh)

    grad_params = self.Params(grad_Wh, grad_Uh, grad_bh, grad_Wy, grad_by)
    return grad_params, grad_h


  def _relu(self, x):
    '''The ReLU activation function.
    '''
    return np.array([e if e > 0 else 0.0 for e in x ])


  def _grad_relu(self, x, grad):
    '''The gradients of "grad" through ReLU backwards, given the input is x.
    '''
    m = np.array([1.0 if e > 0 else 0.0 for e in x])
    return m * grad


  def _grad_mv(self, M, v, grad):
    '''The gradients back propagate of matmul z = Mv.

    z_i = (m_i, v), m_i is the row vector shaped (1, n).
    On v:
    The gradient of z_i over v is m_i^T.
    The result shall be
    \sum grad_z_i m_i^T = (m1^T, m2^T, ...) grad_z = M^T grad_z

    On M:
    The gradient of z_i over m_i is v^T, and 0 for all the other rows.
    The result shall be (grad_z_1*v^T, ..., grad_z_i*v^T, ...)T = grad_z v^T,
    in which grad_z is n*1 and v^T is 1*n.


    Args:
      m, v: The matmul of matrix * vector.
      grad: The gradient of z = mv.

    Returns:
      grad_m, grad_v: the gradients propagated to m and v.
    '''
    grad_v = np.matmul(M.T, grad)
    grad_M = np.matmul(grad.reshape(len(grad), 1),
                       v.reshape(1, len(v)))
    return grad_M, grad_v


  def dim_h(self):
    return len(self._params.bh)


class Rnn:
  def __init__(self):
    '''Initialize the network with random values.
    '''
    H = 20
    self._cell = RnnCell(Wh=np.random.rand(H, 28),
                         Uh=np.random.rand(H, H),
                         bh=np.random.rand(H) - 0.5,
                         Wy=np.random.rand(10, H),
                         by=np.random.rand(10) - 0.5)
    self._y = []
    self._h = []
    self._x = []


  def predict(self, input):
    '''Predict on the input sized 28*28, and returns the softmax hypothesis.
    '''
    self._reset(input)
    for i in range(28):
      x = self._x[i]
      h = self._h[i-1] if i > 0 else np.zeros(self._cell.dim_h())
      self._y[i], self._h[i] = self._cell.forward(x, h)

    return self._softmax(self._y[-1])


  def back_propagate(self, input, label, p=None):
    '''Calculates the weight updates based on input and label.

    If the loss function is -\sum l_i log p_i, where l is one-hot label, the
    loss gradient on the last activation is p_i - l_i.
    '''
    p = self.predict(input) if p is None else p
    grad_y = np.array([p[i] - (1 if i == label else 0) for i in range(len(p))])
    grad_h = np.zeros(self._cell.dim_h())

    # The gradients on parameters.
    grad_params = self._cell._params.zeros()

    for i in range(28 - 1, -1, -1):
      _grad_params, _grad_h = (
          self._cell.backward(x=self._x[i],
                              h=self._h[i-1] if i > 0 else np.zeros(self._cell.dim_h()),
                              y=self._y[i],
                              hh=self._h[i],
                              grad_y=grad_y,
                              grad_hh=grad_h))

      grad_params += _grad_params
      grad_y = np.zeros(grad_y.shape)
      grad_h = _grad_h

    return grad_params


  def _softmax(self, x):
    # To avoid overflow.
    max_x = np.max(x)
    if max_x > 30:
      x =  x / max_x
    exp = np.exp(x)
    s = np.sum(exp)
    return exp / s


  def _reset(self, x):
    assert x.shape == (28, 28)
    self._y = [None]*28
    self._h = [None]*28
    self._x = x


  def run_epoch(self, xy, is_train=False, learning_rate=0.05):
    samples = 0
    errors = 0
    loss = 0.0
    grad = self._cell._params.zeros() if is_train else None
    for x,y in xy:
      samples += 1
      p = self.predict(x)
      loss += -math.log(p[y])
      if np.argmax(p) != y:
        errors += 1

      if is_train:
        grad += self.back_propagate(x, y, p)

    print('In %s mode:' % ('TRAIN' if is_train else 'TEST'))
    print('Error rate is %.1f%%, the loss is %.1f' % (100.0*errors/samples, loss))

    if is_train:
      scale = grad.inf_norm() / self._cell._params.inf_norm()
      grad *= -learning_rate / scale
      print('The inf-norm of param and grad are: ', self._cell._params.inf_norm(), grad.inf_norm())
      self._cell._params += grad


def load_data(path):
  '''Load the MINIST dataset.

  Format:
    x_*: (n_samples, 28, 28), in row-first order.
    y_*: (n_samples,) of integer.
  '''
  f = np.load(path)
  x_train, y_train = f['x_train'], f['y_train']
  x_test, y_test = f['x_test'], f['y_test']
  x_train =  x_train / 255.0
  x_test =  x_test / 255.0
  f.close()
  return (x_train, y_train), (x_test, y_test)



if __name__ == '__main__':
  np.seterr(all='raise')
  rnn = Rnn()
  # Downloaded from https://s3.amazonaws.com/img-datasets/mnist.npz
  train, test = load_data('./mnist.npz')
  n_train = len(train[1])
  n_test = len(test[1])
  print('Dataset loaded: %d training and %d test samples.' % ( n_train, n_test))

  for epoch in range(100):
    print('------- Running Epoch %d --------' % (epoch))
    if epoch % 5 == 0:
      rnn.run_epoch(zip(test[0], test[1]), False)
    rnn.run_epoch(zip(train[0], train[1]), True, 0.05)
