import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

def relu(x):
    return np.maximum(0, x)  

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def dense_function(inputs, weights, biases, activation=None):
    inputs = np.array(inputs)
    if inputs.ndim == 1:
        inputs = inputs.reshape(1, -1) 
    
    weights = np.array(weights)
    biases = np.array(biases)
    
    n_neurons = weights.shape[1]  
    n_inputs = weights.shape[0]  
    
    output = []
    
    for i in range(n_neurons):  
        weighted_sum = 0
        for j in range(n_inputs): 
            weighted_sum += inputs[0][j] * weights[j][i]  
        
        weighted_sum += biases[i]  
        
        if activation == 'relu':
            output.append(relu(weighted_sum))
        elif activation == 'softmax':
            output.append(weighted_sum)
        else:
            output.append(weighted_sum)
    
    output = np.array(output)
    
    if activation == 'softmax':
        output = softmax(output)
    
    return output

model = load_model('1_lstm_n.h5')

weights = model.get_weights()

(_, _), (x_test, _) = mnist.load_data()
x_test = x_test.astype('float32') / 255.0
sample_input = x_test[:1].reshape(1, 784, 1)  

def lstm_function(inputs, weights, return_sequences=False):
    W, R, b = weights
    batch_size, timesteps, inputdim = inputs.shape
    units = R.shape[0] 
    
    gate_size = units * 4
    
    if return_sequences:
        outputs = np.zeros((batch_size, timesteps, units))
    else:
        outputs = np.zeros((batch_size, units))

    for sample_idx in range(batch_size):
        h = np.zeros(units)
        c = np.zeros(units)

        for t in range(timesteps):
            x = inputs[sample_idx, t, :]
            
            gates = np.zeros(gate_size)
            
            for k in range(gate_size):
    
                for i in range(inputdim):
                    gates[k] += x[i] * W[i, k]
                
                for j in range(units):
                    gates[k] += h[j] * R[j, k]
                
                gates[k] += b[k]
            
            i = tf.sigmoid(gates[:units])          
            f = tf.sigmoid(gates[units:2*units])   
            c_t = np.tanh(gates[2*units:3*units])  
            o = tf.sigmoid(gates[3*units:])        
            
            c = f * c + i * c_t
            h = o * np.tanh(c)
            
            if return_sequences:
                outputs[sample_idx, t] = h
        
        if not return_sequences:
            outputs[sample_idx] = h

    return outputs


lstm_layer_model = tf.keras.models.Model(inputs=model.inputs, 
                                        outputs=model.layers[0].output)

lstm_layer_output = lstm_layer_model.predict(sample_input)


function_output_lstm = lstm_function(sample_input,  weights[:3])

print("Сравнение выходов LSTM слоя:")
print(f"Выход Keras LSTM слоя:\n{np.round(lstm_layer_output, 4)}")
print(f"\nВыход LSTM с помощью функции:\n{np.round(function_output_lstm,4)}")
print(f"Совпадают:{np.allclose(lstm_layer_output, function_output_lstm, atol=1e-4)}")

dense1_output = tf.keras.models.Model(inputs=model.inputs, 
                                     outputs=model.layers[1].output).predict(sample_input)
dense2_output = tf.keras.models.Model(inputs=model.inputs, 
                                     outputs=model.layers[2].output).predict(sample_input)
dense3_output = tf.keras.models.Model(inputs=model.inputs, 
                                     outputs=model.layers[3].output).predict(sample_input)


function_output_dense1 = dense_function(function_output_lstm , weights[3], weights[4], 'relu')
print("\nСравнение первого Dense слоя:")
print(f"Keras:\n{np.round(dense1_output,4)}")
print(f"С помощью функции:\n{np.round(function_output_dense1,4)}")
print(f"Совпадают:{np.allclose(dense1_output, function_output_dense1, atol=1e-4)}")

function_output_dense2= dense_function(function_output_dense1, weights[5], weights[6], 'relu')
print("\nСравнение второго Dense слоя:")
print(f"Keras:\n{np.round(dense2_output,4)}")
print(f"С помощью функции:\n{np.round(function_output_dense2,4)}")
print(f"Совпадают:{np.allclose(dense2_output, function_output_dense2, atol=1e-4)}")

function_output_dense3 = dense_function(function_output_dense2, weights[7], weights[8], 'softmax')
print("\nСравнение итогового выхода модели:")
print(f"Keras:\n{np.round(dense3_output,4)}")
print(f"С помощью функции:\n{np.round(function_output_dense3,4)}")
print(f"Совпадают:{np.allclose(dense3_output, function_output_dense3, atol=1e-4)}")
