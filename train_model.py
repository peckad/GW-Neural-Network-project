#define the function that trains the neural network

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from z_score import z_score  #import normalising function

def train_model(model, dataset, epochs, label_name, batch_size, validation_split):
 
	dataset_norm = z_score(dataset)  #normalise the dataset
	features = {name:np.array(value) for name, value in dataset_norm.items()}  #extract the features from the normalied dataset
	label = np.array(features_label.pop(label_name))  #extract the labels from the non-normalised dataset
	
	#train the model
	history = model.fit(features, label, epochs=epochs, batch_size=batch_size, validation_split=validation_split, shuffle=True)

	epochs = history.epoch
	hist = pd.DataFrame(history.history)

	return epochs, hist
