import tensorflow as  tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

def build_model():
  model = Sequential()
  model.add(LSTM(10, input_shape=(x_train.shape[1:]), activation='relu',
                 return_sequences=False))

  model.add(Dense(10, activation='softmax'))

  return model


if __name__ == '__main__':
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train / 255.0
  x_test = x_test / 255.0


  model = build_model()
  model.summary()
  opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
  model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,
                metrics=['accuracy'])

  model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
