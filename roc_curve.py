#this cript plots the roc curves

import tensorflow
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve #the methods that give the roc and auc
from z_score import z_score  # the function to normalise the data
from roc_obstime import plot_roc  #the function that plots the roc with far per year

def plot_roc_curve(test, label_name, model):
	x_test = test.drop(labels=label_name, axis=1)  #take the features
	x_test_norm = z_score(x_test)  #normalise them
	test_features = {name:np.array(value) for name, value in x_test_norm.items()}  #exatract the columns of features
	y_test = test.pop(label_name)  #take the label  
	print('features generated')

	rdm_probs = [0 for i in range(len(y_test))]  #generate random predictions
	y_pred = model.predict(test_features)   #generate the model predictions for the test data
	print('predictions generated')

  #compute the roc curve auc for the model from the test features and preditions
	model_auc = roc_auc_score(y_test, y_pred)
	print('Model: ROC AUC = %.3f' %(model_auc))

  #compute the true positive and false posituve rates for the model and the ransom classifier
	rdm_fpr, rdm_tpr, _ = roc_curve(y_test, rdm_probs)
	model_fpr, model_tpr, _ = roc_curve(y_test, y_pred)

	#plt.style.use('bmh')
	fig = plt.figure(figsize=(14,20))

  #the 1st subplot has normal axes
	ax1 = plt.subplot(211)
	ax1.plot(rdm_fpr, rdm_tpr, linestyle='--', label='Random')
	ax1.plot(model_fpr, model_tpr, label='Model')
	ax1.set_xlabel('False positive rate')
	ax1.set_ylabel('True positive rate')
	ax1.grid(True, which='both')
	ax1.legend()

  #the 2nd sublop has log axes
	ax3 = plt.subplot(212)
	ax3.loglog(model_fpr, model_tpr, color='orange', label='Model')
	ax3.set_xlabel('False positive rate')
	ax3.set_ylabel('True positive rate')
	ax3.grid(True, which='both')

  #save the figure
	plt.savefig('ROC.png')
	print('ROC plotted')

  #call another function to plot the ROC curves with FAR per year
	plot_roc(rdm_fpr, rdm_tpr, model_fpr, model_tpr)
