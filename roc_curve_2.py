#this function plots the roc curve with the false alarm rate per year
#it is called in the main roc curve script

import tensorflow
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from z_score import z_score

def plot_roc(rdm_fpr,rdm_tpr, model_fpr, model_tpr):

	t = 166   #the total observation time in days

	fig = plt.figure(figsize=(14,20))
  
  	#the 1st subplot has normal axes
	ax1 = plt.subplot(211)   
	ax1.plot(rdm_fpr/t*365, rdm_tpr, linestyle='--', label='Random') #plot the false alarm rate per year and true positive rate with varying thresholds for a random classifier
	ax1.plot(model_fpr/t*365, model_tpr, label='Model')   #for the model
	ax1.set_xlabel('False alarm rate')
	ax1.set_ylabel('True positive rate')
	ax1.grid(True, which='both')
	ax1.legend()

 	#the 2nd subplot as log axes - the random classifier isn't plotted, it gives a vertical line at the right end of the graph
	ax3 = plt.subplot(212)
	ax3.loglog(model_fpr/t*365, model_tpr, color='orange', label='Model')
	ax3.set_xlabel('False alarm rate')
	ax3.set_ylabel('True positive rate')
	ax3.grid(True, which='both')

	#save the figure
	plt.savefig('ROC_yr.png')
	print('ROC with rate per year plotted')
