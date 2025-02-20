# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np

np.random.seed = 5

inputs = np.random.normal(size=(1,2)) # (номер семпла, семпл) (batch_size, input_dim)

#inputs = np.array([[1, 0]])

print(inputs)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(2, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

weights = model.get_weights()
print(weights)

activation_function = model.layers[0].activation
print(activation_function([-1, 1])) # пример использования функции активации из слоя

etalon_output = model.predict(inputs)
print(etalon_output)

