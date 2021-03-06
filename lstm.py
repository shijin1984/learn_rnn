import math
import numpy as np

class LSTMCell:
  '''The LSTM cell.

  The LSTM takes input of x(t), h(t-1) and C(t-1), outputs y(t)=h(t).
  Also, it updates the internal state of C(t).

  First of all, it calculates the vectors:
    zf = Wf x(t) + Uf h(t-1) + bf
    zi = Wi x(t) + Ui h(t-1) + bi
    zo = Wo x(t) + Uo h(t-1) + bo
    zc = Wc x(t) + Uc h(t-1) + bc
  and also the gates and states:
    f = sigmoid(zf)
    i = sigmoid(zi)
    o = sigmoid(zo)
    c = tanh(zc)

  Then updates the cell state:
    C(t) = f*C(t-1) + i*c

  Finally,
    updates: C(t) = c
    outputs: y(t) = h(t) = o * tanh(c)
  '''

  class Params:
    '''The wrapper of RNN cell params.
    '''
    def __init__(self, Wf, Uf, bf, Wi, Ui, bi, Wo, Uo, bo,
                 Wc, Uc, bc):
      '''Constructs the cell. The dimensions are (given the dimension of h is H):
         Wf, Wi, Wc: H * 28
         Uf, Ui, Uc: H * 10
         bf, bi, bc: H

         Wo: 10*28, Uo: 10*10, bo: 10

      Args:
        The matrix and bias as in the LSTM cell.
      '''
      _array = lambda x: x if isinstance(x, np.ndarray) else np.array(x)
      def _matrix(m, shape):
        assert isinstance(m, np.ndarray)
        assert m.shape == shape
        return m

      def _vector(x, length):
        assert len(x) == length
        return x if isinstance(x, np.ndarray) else np.array(x)

      H = len(bf)
      self.H = H
      self.Wf = _matrix(Wf, (H, 28))
      self.Wi = _matrix(Wi, (H, 28))
      self.Wc = _matrix(Wc, (H, 28))

      self.Uf = _matrix(Uf, (H, 10))
      self.Ui = _matrix(Ui, (H, 10))
      self.Uc = _matrix(Uc, (H, 10))

      self.bf = _vector(bf, H)
      self.bi = _vector(bi, H)
      self.bc = _vector(bc, H)

      self.Wo = _matrix(Wo, (10, 28))
      self.Uo = _matrix(Uo, (10, 10))
      self.bo = _vector(bo, 10)


    def zeros(self):
      '''Creates another Params instance in the same size and all zeros.
      '''
      return self.__class__(
          Wf=np.zeros(self.Wf.shape),
          Wi=np.zeros(self.Wi.shape),
          Wo=np.zeros(self.Wo.shape),
          Wc=np.zeros(self.Wc.shape),
          Uf=np.zeros(self.Uf.shape),
          Ui=np.zeros(self.Ui.shape),
          Uo=np.zeros(self.Uo.shape),
          Uc=np.zeros(self.Uc.shape),
          bf=np.zeros(self.bf.shape),
          bi=np.zeros(self.bi.shape),
          bo=np.zeros(self.bo.shape),
          bc=np.zeros(self.bc.shape))

    def __iadd__(self, other):
      '''The += operator.
      '''
      self.Wf += other.Wf
      self.Wi += other.Wi
      self.Wo += other.Wo
      self.Wc += other.Wc
      self.Uf += other.Uf
      self.Ui += other.Ui
      self.Uo += other.Uo
      self.Uc += other.Uc
      self.bf += other.bf
      self.bi += other.bi
      self.bo += other.bo
      self.bc += other.bc
      return self

    def __imul__(self, rhs):
      '''The *= operator.
      '''
      self.Wf *= rhs
      self.Wi *= rhs
      self.Wo *= rhs
      self.Wc *= rhs
      self.Uf *= rhs
      self.Ui *= rhs
      self.Uo *= rhs
      self.Uc *= rhs
      self.bf *= rhs
      self.bi *= rhs
      self.bo *= rhs
      self.bc *= rhs
      return self


  def __init__(self):
    def _glorot(row, col):
      limit = math.sqrt(6.0 / (row * col))
      m = np.random.rand(row, col)
      # Scale from (0,1) to (-limit, limit)
      return (m * (2*limit)) - limit

    H = 10
    self._params = self.Params(
        Wf=_glorot(H, 28),
        Wi=_glorot(H, 28),
        Wc=_glorot(H, 28),
        Uf=_glorot(H, 10),
        Ui=_glorot(H, 10),
        Uc=_glorot(H, 10),
        bf=np.zeros(H),
        bi=np.zeros(H),
        bc=np.zeros(H),
        Wo=_glorot(10, 28),
        Uo=_glorot(10, 10),
        bo=np.zeros(10))

  def forward(self, x, h):
    '''Calculates the y(t) and h(t) from x=x(t) and h=concat(h(t-1), C(t-1))
    '''
    # h(t-1) and C(t-1) are concat into `h', break it here.
    h, C = h[:10], h[10:]

    self._zf = (
        np.matmul(self._params.Wf, x)  + np.matmul(self._params.Uf, h) + self._params.bf)
    self._zi = (
        np.matmul(self._params.Wi, x)  + np.matmul(self._params.Ui, h) + self._params.bi)
    self._zo = (
        np.matmul(self._params.Wo, x)  + np.matmul(self._params.Uo, h) + self._params.bo)
    self._zc = (
        np.matmul(self._params.Wc, x)  + np.matmul(self._params.Uc, h) + self._params.bc)

    sigmoid = lambda z: 1.0 / (1 + np.exp(-z))
    self._f = sigmoid(self._zf)
    self._i = sigmoid(self._zi)
    self._o = sigmoid(self._zo)
    self._c = np.tanh(self._zc)

    self._C = self._f*C + self._i*self._c
    y = self._o * np.tanh(self._C)

    return y, np.concatenate((y, self._C))

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
    # Call forward() to update the intermediate results.
    y, hh = self.forward(x, h)
    # Break the combined h of h, C
    h, C = h[:10], h[10:]
    grad_h, grad_C = grad_hh[:10], grad_hh[10:]
    # h(t) = y(t) so to merge the gradients.
    grad_h += grad_y

    grad_o = grad_h * np.tanh(self._C)
    grad_C = grad_h * self._o * self._d_tanh(self._C)

    grad_f = grad_C * C
    grad_C_prev = grad_C * self._f
    grad_i = grad_C * self._c
    grad_c = grad_C * self._i

    grad_zf = grad_f * self._d_sigmoid(self._zf)
    grad_zi = grad_i * self._d_sigmoid(self._zi)
    grad_zo = grad_o * self._d_sigmoid(self._zo)
    grad_zc = grad_c * self._d_tanh(self._zc)

    grad_Wf, _, grad_Uf, grad_h_zf, grad_bf = self._grad_WUb(
        self._params.Wf, x, self._params.Uf, h, grad_zf)
    grad_Wi, _, grad_Ui, grad_h_zi, grad_bi = self._grad_WUb(
        self._params.Wi, x, self._params.Ui, h, grad_zi)
    grad_Wo, _, grad_Uo, grad_h_zo, grad_bo = self._grad_WUb(
        self._params.Wo, x, self._params.Uo, h, grad_zo)
    grad_Wc, _, grad_Uc, grad_h_zc, grad_bc = self._grad_WUb(
        self._params.Wc, x, self._params.Uc, h, grad_zc)

    grad_params = self.Params(
        Wf=grad_Wf, Wi=grad_Wi, Wo=grad_Wo, Wc=grad_Wc,
        Uf=grad_Uf, Ui=grad_Ui, Uo=grad_Uo, Uc=grad_Uc,
        bf=grad_bf, bi=grad_bi, bo=grad_bo, bc=grad_bc)

    grad_h_prev = (
        grad_h_zf + grad_h_zi + grad_h_zo + grad_h_zc)
    grad_prev = np.concatenate((grad_h_prev, grad_C_prev))

    return grad_params, grad_prev


  def _relu(self, x):
    '''The ReLU activation function.
    '''
    return np.array([e if e > 0 else 0.0 for e in x ])


  def _d_tanh(self, x):
    '''Derivative of tanh(x): 1 - tanh(x)**2.
    '''
    tanh = np.tanh(x)
    tanh = -tanh * tanh
    return tanh + 1.0

  def _d_sigmoid(self, x):
    '''Derivative of s(x): s(x)*(1-s(x)).
    '''
    sigmoid = lambda z: 1.0 / (1 + np.exp(-z))
    s = sigmoid(x)
    return s * (-s + 1.0)


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

  def _grad_WUb(self, W, x, U, h, grad):
    '''z = W*x + U*h + b, and grad = grad_z.
    Returns the gradients of W, x, U, h, b.
    '''
    grad_W, grad_x = self._grad_mv(W, x, grad)
    grad_U, grad_h = self._grad_mv(U, h, grad)
    return grad_W, grad_x, grad_U, grad_h, grad


if __name__ == '__main__':
  cell = LSTMCell()
  x = np.random.rand(28)
  h = np.random.rand(20)
  print(cell.backward(x, h, None, None, np.random.rand(10), np.random.rand(20)))
