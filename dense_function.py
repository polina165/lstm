
import tensorflow as tf
import numpy as np

'''Описание переменных:
inputs-входы, weights-веса, biases-смещения, activation-функция активации,
neurons_count-количество нейронов, inputs_count-количество входов, weighted_sum-сумма весов, outputs-выходы
'''

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
    
    neurons_count = weights.shape[1]  
    inputs_count = weights.shape[0]  
    
    outputs = []
    
    for i in range(neurons_count):  
        weighted_sum = 0
        for j in range(inputs_count): 
            weighted_sum += inputs[0][j] * weights[j][i]  
        
        weighted_sum += biases[i]  
        
        if activation == 'relu':
            outputs.append(relu(weighted_sum))
        elif activation == 'softmax':
            outputs.append(weighted_sum)
        else:
            outputs.append(weighted_sum)
    
    outputs = np.array(outputs)
    
    if activation == 'softmax':
        outputs = softmax(outputs)
    
    return outputs 

