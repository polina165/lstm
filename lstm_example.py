import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM

np.random.seed=42

inputs = np.random.normal(size=(1, 2, 5))  
print(f"Входы:\n{inputs}")

model = tf.keras.Sequential([
    LSTM(2)
])
model.build(input_shape=(None, 2, 5))
model.summary()

weights = model.get_weights()
W, R, b = weights  
outputs = model.predict(inputs)
print(f"Выходы модели:\n{outputs}")

def lstm_function(inputs, weights, return_sequences=False):   
    W, R, b = weights
    batch_size, timesteps, input_dim = inputs.shape
    neurons= R.shape[0]     
    
    if return_sequences:
        outputs = np.zeros((batch_size, timesteps, neurons))
    else:
        outputs = np.zeros((batch_size, neurons))

    for batch_item in range(batch_size):
        h = np.zeros(neurons)
        c = np.zeros(neurons)

        for t in range(timesteps):
            x = inputs[batch_item, t, :]
            
            gate_size = neurons * 4
            gates = np.zeros(gate_size)
            
            for k in range(gate_size):
    
                for i in range(input_dim):
                    gates[k] += x[i] * W[i, k]
                
                for j in range(neurons):
                    gates[k] += h[j] * R[j, k]
                
                gates[k] += b[k]
            
            
            i = tf.sigmoid(gates[:neurons])          # input gate
            f = tf.sigmoid(gates[neurons:2*neurons])   # forget gate
            c_t = np.tanh(gates[2*neurons:3*neurons])  # candidate cell state
            o = tf.sigmoid(gates[3*neurons:])        # output gate
            
            # Обновляем состояния
            c = f * c + i * c_t
            h = o * np.tanh(c)
            
            # Сохраняем выход для каждого шага, если нужно
            if return_sequences:
                outputs[batch_item, t] = h
            
        # Сохраняем только последний выход, если не нужны все последовательности
        if not return_sequences:
            outputs[batch_item] = h
    
    return outputs

outputs_function =lstm_function(inputs, weights,return_sequences=False)
print(f"Выходы модели с помощью функции:\n{outputs_function}")