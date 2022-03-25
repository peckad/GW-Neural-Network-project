#The main script for the neural network

#all the imports necessary
import tensorflow as tf
import pandas as pd
import numpy as np
from features import feature_columns   #the features columns to feed to the input layer
from create_model_2 import make_model   #the function that builds the model
from train_model import train_model   #the function that trains the model
from plot_metrics import plot_metrics   #the function that plots the metrics during training
from evaluate_model import model_evaluate  #the function to evaluate the model on the test set
from roc_curve_2 import plot_roc_curve   #the function that plots the ROC curves
from data_import import test_eval, test_roc, train_dfn  #data is imported from another script

#set the hyperparameters
learning_rate = 0.01
epochs = 1000
batch_size = 800
label_name ='Label'
validation_split = 0.10

#define the metrics
metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
	tf.keras.metrics.Precision(name='precision'),
	tf.keras.metrics.Recall(name='recall'),
	tf.keras.metrics.TruePositives(name='tp'),
	tf.keras.metrics.FalsePositives(name='fp'),
	tf.keras.metrics.TrueNegatives(name='tn'),
	tf.keras.metrics.FalseNegatives(name='fn')
]

#build the model
deap_nn = make_model(learning_rate, feature_columns, metrics)
print('model created')

#train the model
epochs, hist = train_model(deap_nn, train_df, epochs, label_name, batch_size, validation_split)
print('model trained')

#the list of metrics to plot
list_of_metrics = ['accuracy','precision', 'recall','loss',
	'val_accuracy','val_precision','val_recall','val_loss']

#plot those metrics
plot_metrics(epochs, hist, list_of_metrics)
print('Curve printed')

#evaluate the model against the test data
model_evaluate(deap_nn, label_name, test_eval, batch_size)

#plot the ROC curves
plot_roc_curve(test_roc, label_name, deap_nn)

print('Done')
