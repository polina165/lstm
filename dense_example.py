# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np

np.random.seed = 5

inputs = np.random.normal(size=(1,5)) # (номер семпла, семпл) (batch_size, input_dim)

#inputs = np.array([[1, 0]])

print(f"Входы:{inputs}")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5,)),
    tf.keras.layers.Dense(5, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
              
weight = model.get_weights()
weights,biases = weight
print(f"Веса:\n{weights}")
print(f"Смещения:\n{biases}")

#activation_function = model.layers[0].activation
#print(f"Функция активация: {activation_function([-1, 1])}") # пример использования функции активации из слоя

etalon_output = model.predict(inputs)
print(f"Выходы модели:\n{etalon_output}")

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

function_output = dense_function(inputs, weights, biases, activation='softmax')
print(f"Выходы с помощью функции:\n{function_output}")