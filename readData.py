# How do you import nicely data from csv to tf?
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
filename_queue = tf.train.string_input_producer(["XORdata.csv"])
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
record_defaults = [[1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3, col4])
