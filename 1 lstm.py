import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 784, 1).astype('float32') / 255.0  
x_test = x_test.reshape(-1, 784, 1).astype('float32') / 255.0     

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()
model.add(LSTM(1, input_shape=(784, 1)))  
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(x_train.reshape(-1, 784, 1), y_train,  
                    epochs=10,
                    batch_size=128,
                    validation_data=(x_test.reshape(-1, 784, 1), y_test))


model.save('1_lstm.h5')

test_loss, test_acc = model.evaluate(x_test.reshape(-1, 784, 1), y_test)
print(f'Test accuracy: {test_acc:.4f}')