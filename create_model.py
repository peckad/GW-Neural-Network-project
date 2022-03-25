#define the function that builds the neural network

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def make_model(learning_rate, feature_col, metrics): #define the function
	model = tf.keras.models.Sequential([
	layers.DenseFeatures(feature_col),  #first layer takes feature columns as arg, this defines the columns from the dataset that will be taken as features
	layers.Dense(100, activation='sigmoid'), #1st hidden - set activation function
	#layers.Dropout(rate=0.2),  #optional dropout layer
	layers.Dense(80),    #more hidden layers
	layers.Dense(40),
	layers.Dense(20),
	layers.Dense(10),
	layers.Dense(1, activation='sigmoid')  #output layer - 1 neuron - sigmoid activation function gives value between 0 and 1
	])
  
  #compile the model
	model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate), loss=tf.keras.losses.BinaryCrossentropy(), metrics=metrics)

	return model
