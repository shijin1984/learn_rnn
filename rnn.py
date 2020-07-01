#!/usr/bin/python

import numpy as np

class RnnCell:

  def __init__(self, Wh, Uh, bh, Wy, by):
    '''Constructs the cell. The dimensions are (given the dimension of h is H):
       Wh: 28*H  Uh: H*H,  bh: H*1
       Wy: 10*H,  by: 10*1

    Args:
      The matrix and bias as in the Elman network.
    '''
    dim_h = len(bh)
    assert Wh.shape == (dim_h, 28)
    self._Wh = np.array(Wh)
    assert Uh.shape == (dim_h, dim_h)
    self._Uh = np.array(Uh)
    self._bh = np.array(bh)
    assert Wy.shape == (10, dim_h)
    self._Wy = np.array(Wy)
    assert len(by) == 10
    self._by = np.array(by)


  def forward(self, x, h):
    '''Calculates the y(t) and h(t) from x=x(t) and h=h(t-1).
    '''
    z_h = np.matmul(self._Wh, x) + np.matmul(self._Uh, h) + self._bh
    hh = self._relu(z_h)
    y = self._relu(np.matmul(self._Wy, hh) + self._by)
    return y, hh


  def _relu(self, x):
    '''The ReLU activation function.
    '''
    return np.array([e if e > 0 else 0.0 for e in x ])


  def dim_h(self):
    return len(self._bh)


class Rnn:
  def __init__(self):
    '''Initialize the network with random values.
    '''
    H = 20
    self._cell = RnnCell(Wh=np.random.rand(H, 28) - 0.5,
                         Uh=np.random.rand(H, H) - 0.5,
                         bh=np.random.rand(H) - 0.5,
                         Wy=np.random.rand(10, H) - 0.5,
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


  def _softmax(self, x):
    exp = np.exp(x)
    s = np.sum(exp)
    return exp / s


  def _reset(self, x):
    assert x.shape == (28, 28)
    self._y = [None]*28
    self._h = [None]*28
    self._x = x


if __name__ == '__main__':
  rnn = Rnn()
  x = np.random.rand(28, 28)
  print(rnn.predict(x))
