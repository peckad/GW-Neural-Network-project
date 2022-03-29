#this script creates teh featuure_columns object used to build the input layer of the network
#it creates 11 neurons and selects the right column from the dataframe to feed to each
#the object is imported to the main script

import tensorflow as tf
from tensorflow.keras import layers

feature_columns = []

#a feature column is created for all 11 features used
penalty = tf.feature_column.numeric_column('penalty')
feature_columns.append(penalty)

norm = tf.feature_column.numeric_column('norm')
feature_columns.append(norm)

netcc0 = tf.feature_column.numeric_column('netcc0')
feature_columns.append(netcc0)

netcc2 = tf.feature_column.numeric_column('netcc2')
feature_columns.append(netcc2)

rho0 = tf.feature_column.numeric_column('rho0')
feature_columns.append(rho0)

neted0 = tf.feature_column.numeric_column('neted0')
feature_columns.append(neted0)

neted1 = tf.feature_column.numeric_column('neted1')
feature_columns.append(neted1)

ecor = tf.feature_column.numeric_column('ecor')
feature_columns.append(ecor)

Qveto0 = tf.feature_column.numeric_column('Qveto0')
feature_columns.append(Qveto0)

Qveto1 = tf.feature_column.numeric_column('Qveto1')
feature_columns.append(Qveto1)

Lveto2 = tf.feature_column.numeric_column('Lveto2')
feature_columns.append(Lveto2)
