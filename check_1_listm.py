import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from dense_function import dense_function
from lstm_function import lstm_function


model = load_model('1_lstm.h5')

weights = model.get_weights()


(_, _), (x_test, _) = mnist.load_data()
x_test = x_test.astype('float32') / 255.0
sample_input = x_test[:1].reshape(1, 784, 1)

lstm_layer_model = tf.keras.models.Model(inputs=model.inputs, 
                                        outputs=model.layers[0].output)

lstm_layer_output = lstm_layer_model.predict(sample_input)


function_output_lstm = lstm_function(sample_input, weights[:3])

print("Сравнение выходов LSTM слоя:")
print(f"Выход Keras LSTM слоя:\n{np.round(lstm_layer_output, 4)}")
print(f"Выход LSTM с помощью функции:\n{np.round(function_output_lstm,4)}")
print(f"Совпадают:{np.allclose(lstm_layer_output, function_output_lstm, atol=1e-4)}")

dense1_output = tf.keras.models.Model(inputs=model.inputs, 
                                     outputs=model.layers[1].output).predict(sample_input)
dense2_output = tf.keras.models.Model(inputs=model.inputs, 
                                     outputs=model.layers[2].output).predict(sample_input)
dense3_output = tf.keras.models.Model(inputs=model.inputs, 
                                     outputs=model.layers[3].output).predict(sample_input)


function_output_dense1 = dense_function(function_output_lstm , weights[3], weights[4], 'relu')
print("Сравнение первого Dense слоя:")
print(f"Keras:\n{np.round(dense1_output,4)}")
print(f"С помощью функции:\n{np.round(function_output_dense1,4)}")
print(f"Совпадают:{np.allclose(dense1_output, function_output_dense1, atol=1e-4)}")

function_output_dense2= dense_function(function_output_dense1, weights[5], weights[6], 'relu')
print("Сравнение второго Dense слоя:")
print(f"Keras:\n{np.round(dense2_output,4)}")
print(f"С помощью функции:\n{np.round(function_output_dense2,4)}")
print(f"Совпадают:{np.allclose(dense2_output, function_output_dense2, atol=1e-4)}")

function_output_dense3 = dense_function(function_output_dense2, weights[7], weights[8], 'softmax')
print("Сравнение итогового выхода модели:")
print(f"Keras:\n{np.round(dense3_output,4)}")
print(f"С помощью функции:\n{np.round(function_output_dense3,4)}")
print(f"Совпадают:{np.allclose(dense3_output, function_output_dense3, atol=1e-4)}")