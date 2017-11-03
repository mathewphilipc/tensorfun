# cf https://www.tensorflow.org/api_docs/python/tf/feature_column/numeric_column

# various imports
import numpy as np
import tensorflow as tf
import os

# Shut off some annoying warning messages
# (Not as ominous as it sounds)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

XOR_TRAINING = "XORdata.csv"

# important note, observed from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/base.py
# about how load_csv_from_header works: In the top row, entry 1 (i.e., header[0])
# gives the number of samples. Entry 2 (i.e., header[1]) gives the number of features
# Output values are given in last col

print("Let's learn XOR")
def main():
	training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
	filename=XOR_TRAINING,
	target_dtype=np.int,
	features_dtype=np.float32)
	print("...\nTraining data imported succesfully")

	# For simplicity we'll use the same set for testing at the moment

	test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
	filename=XOR_TRAINING,
	target_dtype=np.int,
	features_dtype=np.float32)
	print("...\nTest data imported successfully")

	# Each data point has four input variables
	feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]
	print("...\nDefined feature columns")

	# Build 3 layer DNN with 10, 20, 10 units respectively
	classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
		hidden_units=[10,20,10],
		n_classes=2,
		model_dir="/home/mathew/Desktop/CNAM/TensorFun/xor_model")

	# Note that model_dir is persistent after training
	# In order to re-run training with, e.g., different hidden unit numbers,
	# you must delete model_dir. To make this easier, I changed the path
	# and stored it in the project directory (TensorFun)
	# Maybe there's a code way to delete it from in here?
	# Look into this tomorrow

	print("...\nDefined classifier")

	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": np.array(training_set.data)},
		y=np.array(training_set.target),
		num_epochs=None,
		shuffle=True)

	print("...\nTraining inputs defined")

	classifier.train(input_fn=train_input_fn, steps=100)


if __name__ == "__main__":
    main()