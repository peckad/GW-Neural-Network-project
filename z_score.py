#define the function that normalises the data from the pandas dataframe
import pandas as pd

def z_score(data):
	mean = data.mean()  #take the mean of each column
	std = data.std()  #the standard deviation
	data = (data-mean)/std   #calculate the normalised data

	return data
