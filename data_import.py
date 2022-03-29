#load the data into dataframes
import pandas as pd

#load the training set from a csv into a pandas dataframe - only load the columns needed
train = pd.read_csv('train_set.csv', usecols=['Label','penalty','norm','netcc0','netcc2','rho0','neted0','neted1','ecor','Qveto0','Qveto1','Lveto2'])

#do the same for the test set - use different copies to evaluate the model and build the ROC
test_eval = pd.read_csv('test_set.csv', usecols=['Label','penalty','norm','netcc0','netcc2','rho0','neted0','neted1','ecor','Qveto0','Qveto1','Lveto2'])

test_eval = pd.read_csv('test_set.csv', usecols=['Label','penalty','norm','netcc0','netcc2','rho0','neted0','neted1','ecor','Qveto0','Qveto1','Lveto2'])

#shuffle the training set
train_df = train.sample(frac=1, ranodm_state=12)
