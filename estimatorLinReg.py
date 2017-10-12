import tensorflow as tf
import numpy as np

# Declare list of features. We only have one numeric feature. There are many other types
# of columns that are more complicated and useful.
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# An estimator is the front end to invoke training (fitting) and evaluation (inference).
# There are many predefined types like linear regression, linear classification, and many
# neural network classifiers and regressors. The following code provides an estimator
# that does linear regression.
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# TensorFlow provides many helper methods to read and set up data sets. Here was use two
# data sets: one for training and one for evaluation. We have to tell the function how
# many batches of data (num_epochs) we want and how big each batch should be.

x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
