#define the function that evaluates the model against the test set

import pandas as pd
import numpy as np
from z_score import z_score #the function to nornalise the data

def model_evaluate(model, label_name, test, batch_size):
  
	x_test = test.drop(labels=label_name, axis=1) #select the features
	x_test_norm = z_score(x_test) #normalise them
  
  #extract the columns containing the features
	test_features = {name:np.array(value) for name, value in x_test_norm.items()}
	y_test = test.pop(label_name) #extract the label
  
  #the .evaluate method evaluates the mdoel against the test set
	evaluation = model.evaluate(x=test_features, y=y_test, batch_size=batch_size)
  
	return evaluation
