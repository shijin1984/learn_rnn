import tensorflow as  tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, RNN, Softmax

class ElmanCell(Layer):

  def __init__(self, units, state_size, **kwargs):
    self.units = units
    self.state_size = units
    super(ElmanCell, self).__init__(**kwargs)

  def build(self, input_shape):
    # h(t) = activation(
    #     Wh x(t) + Uh h(t-1) + bh)
    self.Wh = self.add_weight(shape=(input_shape[-1], self.state_size),
                              name='Wh')
    self.Uh = self.add_weight(shape=(self.state_size, self.state_size),
                              name='Uh')
    self.bh = self.add_weight(shape=(self.state_size,),
                              name='bh')
    # y(t) = activate(Wy h(t) + by)
    self.Wy = self.add_weight(shape=(self.state_size, self.units),
                              name='Wy')
    self.by = self.add_weight(shape=(self.units,),
                              name='by')
    self.built = True

  def call(self, inputs, states):
    # h(t) = activation(
    #     Wh x(t) + Uh h(t-1) + bh)
    h = states[-1]
    hh = K.dot(inputs, self.Wh) + K.dot(h, self.Uh)
    hh = activations.relu(K.bias_add(hh, self.bh))
    # y(t) = activate(Wy h(t) + by)
    y = K.dot(hh, self.Wy)
    y = K.bias_add(y, self.by)
    return activations.relu(y), [hh]


def build_model(input_shape):
  model = Sequential()
  cell = ElmanCell(10, 20)
  model.add(RNN(cell, input_shape=input_shape[1:]))
  model.add(Softmax())

  model.build()

  return model


if __name__ == '__main__':
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train / 255.0
  x_test = x_test / 255.0

  print(x_train.shape[1:])


  model = build_model(x_train.shape)
  model.summary()
  opt = tf.keras.optimizers.SGD(learning_rate=1e-2)
  model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,
                metrics=['accuracy'])

  model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
