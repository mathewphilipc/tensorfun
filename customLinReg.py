import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Declare a list of features (we only have one real-valued feature)
def model_fn(features, labels, mode):
	# Build a lienar model and predict values
	W = tf.get_variable("W", [1], dtype=tf.float64)
	b = tf.get_variable("b", [1], dtype=tf.float64)
	y = W  * features['x'] + b
	# Loss sub-graph
	loss = tf.reduce_sum(tf.square(y - labels))
	# Training sub-graph
	global_step = tf.train.get_global_step()
	optimizer = tf.group(optimizer.minimize(loss), tf.assign_add(global_set, 1))

	# EstimatorSpec connects subgraphs we built to the appropriate functionality
	return tf.estimator.EstimatorSpec(
		mode=mode,
		predictions=y,
		loss=loss,
		train_op=train)