#this function plots 3 graphs of the evolution of the metrics during training

import matplotlib.pyplot as plt

def plot_metrics(epochs, hist, metrics):

	fig = plt.figure(figsize=(12,25))
  #the 1st subplot is for the metrics on the test set
	ax1 = plt.subplot(311)
  #the 2nd subplot is for the metrics on the validation set
	ax2 = plt.subplot(312)
  #the 3rd subplpt is for the loss on the test and validation sets
	ax3 = plt.subplot(313)

  #set the axis labels
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Value')
	ax2.set_xlabel('Epoch')
	ax2.set_ylabel('Value')
	ax3.set_xlabel('Epoch')
	ax3.set_ylabel('Value')
  
  #set the grids 
	ax1.grid()
	ax2.grid()
	ax3.grid()

  #this loop plots the graphs
	for m in metrics:
		x = hist[m] #extract the evolution of the metrics
		if m=='accuracy' or m=='precision' or m=='recall': #the first graph plots accuracy, precision, and recall
			ax1.plot(x, label=m)
		elif m=='val_accuracy' or m=='val_precision' or m=='val_recall': #the 2nd graph plots accuracy, precision, and recall on the validation set
			ax2.plot(x, label=m)
		else:   #the 3rd graph plots loss - the only metric now left in the list
			ax3.plot(x, label=m)

  #set the legend
	ax1.legend()
	ax2.legend()
	ax3.legend()

  #save the figure
	plt.savefig('metrics.png')
