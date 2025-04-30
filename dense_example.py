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
    tf.keras.layers.Dense(5, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
              
weight = model.get_weights()
weights,biases = weight
print(f"Веса:\n{weights}")
print(f"Смещения:\n{biases}")

activation_function = model.layers[0].activation
#print(f"Функция активация: {activation_function([-1, 1])}") # пример использования функции активации из слоя

etalon_output = model.predict(inputs)
print(f"Выходы модели:\n{etalon_output}")

def dense_output(weights, inputs, biases):
    output = []  
    
    for i in range(len(inputs[0])):
        weighted_sum = 0
        for j in range(len(biases)):
            weighted_sum += inputs[0][j] * weights[j][i]
        weighted_sum += biases[i]
        output.append(activation_function(weighted_sum)) 

    return np.array(output)  

function_output = dense_output(weights, inputs, biases)
print(f"Выходы с помощью функции:\n{function_output}")