import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM

np.random.seed=42

inputs = np.random.normal(size=(1, 3, 5))  #samples, timesteps, features
print(f"Входы:\n{inputs}")

model = tf.keras.Sequential([
    LSTM(4)
])
model.build(input_shape=(None, 3, 5))

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

model.summary()

weights = model.get_weights()
W, R, b = weights  # W-веса для входных данных, R-рекуретные веса, biases
outputs = model.predict(inputs)
print(f"Выходы модели:\n{outputs}")

def lstm_function(inputs,weights):
    batch_size, timesteps,inputdim = inputs.shape
    units = R.shape[0]
    outputs = np.zeros((batch_size, units))

    for sample_idx in range(batch_size):
        h = np.zeros(units)
        c = np.zeros(units)

        for t in range(timesteps):
            x = inputs[sample_idx, t, :]

            gates = np.dot(x, W) + np.dot(h, R) + b
            i = tf.sigmoid(gates[:units])          
            f = tf.sigmoid(gates[units:2*units])   
            c_t = np.tanh(gates[2*units:3*units]) 
            o = tf.sigmoid(gates[3*units:])        
            
            c = f * c + i * c_t
            h = o * np.tanh(c)
        outputs[sample_idx] = h

    return outputs

outputs_function = lstm_function(inputs,weights)
print(f"Выходы модели с помощью функции:\n{outputs_function}")



