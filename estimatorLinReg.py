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