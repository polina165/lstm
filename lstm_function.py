import numpy as np
import tensorflow as tf

'''Описание переменных:
inputs-входы, weights-веса, return_sequences-последовательность, reccurent-рекуретные веса,
biases-смещения, batch_size-размер батча, batch_item-элемент батча, timesteps-временные шаги,
input_step-размерность одного входного вектора, neurons_count-количество нейронов, h-скрытое состояние,
c-состояние ячейки, x-входной вектор на текущем временном шаге, gate_size-размер ворот, 
gates-значения всех ворот, i-input gate, f-forget gate, c_t-candidate cell state, 
o-output gate, outputs-выходы
'''

def lstm_function(inputs, weights, return_sequences=False):   
    weight, recurrent, biases = weights
    batch_size, timesteps, input_step = inputs.shape
    neurons_count= recurrent.shape[0]     
    
    if return_sequences:
        outputs = np.zeros((batch_size, timesteps, neurons_count))
    else:
        outputs = np.zeros((batch_size, neurons_count))

    for batch_item in range(batch_size):
        h = np.zeros(neurons_count)
        c = np.zeros(neurons_count)

        for t in range(timesteps):
            x = inputs[batch_item, t, :]
            
            gate_size = neurons_count * 4
            gates = np.zeros(gate_size)
            
            for k in range(gate_size):
    
                for i in range(input_step):
                    gates[k] += x[i] * weight[i, k]
                
                for j in range(neurons_count):
                    gates[k] += h[j] * recurrent[j, k]
                
                gates[k] += biases[k]
            
            
            i = tf.sigmoid(gates[:neurons_count])         
            f = tf.sigmoid(gates[neurons_count:2*neurons_count])   
            c_t = np.tanh(gates[2*neurons_count:3*neurons_count])  
            o = tf.sigmoid(gates[3*neurons_count:])        
            
            
            c = f * c + i * c_t
            h = o * np.tanh(c)
            
            if return_sequences:
                outputs[batch_item, t] = h
            
        
        if not return_sequences:
            outputs[batch_item] = h
    
    return outputs

