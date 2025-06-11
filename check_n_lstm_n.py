import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from dense_function import dense_function
from lstm_function import lstm_function


model = load_model('n_lstm_n.h5')

weights = model.get_weights()

(_, _), (x_test, _) = mnist.load_data()
x_test = x_test.astype('float32') / 255.0
sample_input = x_test[:1].reshape(1, 784, 1)  


lstm1_output_func = lstm_function(sample_input, weights[:3], return_sequences=True)

lstm1_layer_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[0].output)
lstm1_output_keras = lstm1_layer_model.predict(sample_input)

print("Сравнение выходов LSTM слоя:")
print(f"Выход Keras LSTM слоя:\n{np.round(lstm1_output_keras[0, :5, :], 4)}")
print(f"\nВыход LSTM с помощью функции:\n{np.round(lstm1_output_func[0, :5, :],4)}")
print(f"Совпадают:{np.allclose(lstm1_output_keras, lstm1_output_func, atol=1e-4)}")
lstm2_weights = weights[3:6]  
lstm2_output_func = lstm_function(lstm1_output_func, lstm2_weights, return_sequences=True)

lstm2_layer_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[1].output)
lstm2_output_keras = lstm2_layer_model.predict(sample_input)

print(f"Выход Keras LSTM слоя:\n{np.round(lstm2_output_keras[0, :5, :], 4)}")
print(f"\nВыход LSTM с помощью функции:\n{np.round(lstm2_output_func[0, :5, :],4)}")
print(f"Совпадают:{np.allclose(lstm2_output_keras, lstm2_output_func, atol=1e-4)}")
lstm3_weights = weights[6:9] 
lstm3_output_func = lstm_function(lstm2_output_func, lstm3_weights, return_sequences=False)


lstm3_layer_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[2].output)
lstm3_output_keras = lstm3_layer_model.predict(sample_input)
print(f"Выход Keras LSTM слоя:\n{np.round(lstm3_output_keras, 4)}")
print(f"\nВыход LSTM с помощью функции:\n{np.round(lstm3_output_func,4)}")
print(f"Совпадают:{np.allclose(lstm3_output_keras, lstm3_output_func, atol=1e-4)}")

dense1_output = tf.keras.models.Model(inputs=model.inputs, 
                                     outputs=model.layers[3].output).predict(sample_input)
dense2_output = tf.keras.models.Model(inputs=model.inputs, 
                                     outputs=model.layers[4].output).predict(sample_input)
dense3_output = tf.keras.models.Model(inputs=model.inputs, 
                                     outputs=model.layers[5].output).predict(sample_input)


function_output_dense1 = dense_function(lstm3_output_func , weights[9], weights[10], 'relu')
print("\nСравнение первого Dense слоя:")
print(f"Keras:\n{np.round(dense1_output,4)}")
print(f"С помощью функции:\n{np.round(function_output_dense1,4)}")
print(f"Совпадают:{np.allclose(dense1_output, function_output_dense1, atol=1e-4)}")

function_output_dense2= dense_function(function_output_dense1, weights[11], weights[12], 'relu')
print("\nСравнение второго Dense слоя:")
print(f"Keras:\n{np.round(dense2_output,4)}")
print(f"С помощью функции:\n{np.round(function_output_dense2,4)}")
print(f"Совпадают:{np.allclose(dense2_output, function_output_dense2, atol=1e-4)}")

function_output_dense3 = dense_function(function_output_dense2, weights[13], weights[14], 'softmax')
print("\nСравнение итогового выхода модели:")
print(f"Keras:\n{np.round(dense3_output,4)}")
print(f"С помощью функции:\n{np.round(function_output_dense3,4)}")
print(f"Совпадают:{np.allclose(dense3_output, function_output_dense3, atol=1e-4)}")
