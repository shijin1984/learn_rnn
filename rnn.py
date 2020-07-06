#!/usr/bin/python

from elman import RnnCell
import math
import numpy as np

class Rnn:
  def __init__(self):
    '''Initialize the network with random values.
    '''
    H = 20

    def _glorot(row, col):
      limit = math.sqrt(6.0 / (row * col))
      m = np.random.rand(row, col)
      # Scale from (0,1) to (-limit, limit)
      return (m * (2*limit)) - limit

    self._cell = RnnCell(Wh=_glorot(H, 28),
                         Uh=_glorot(H, H),
                         bh=np.zeros(H),
                         Wy=_glorot(10, H),
                         by=np.zeros(10))
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
      # scale = grad.inf_norm() / self._cell._params.inf_norm()
      scale = 1.0
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

  # train = (train[0][:5000], train[1][:5000])
  for epoch in range(100):
    print('------- Running Epoch %d --------' % (epoch))
    if epoch % 5 == 0:
      rnn.run_epoch(zip(test[0], test[1]), False)
    rnn.run_epoch(zip(train[0], train[1]), True, 5e-4)
