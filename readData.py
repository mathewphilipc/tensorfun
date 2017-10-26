# How do you import nicely data from csv to tf?
# SOF source -> https://stackoverflow.com/questions/37091899/how-to-actually-read-csv-data-in-tensorflow
from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import math as math
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#filename_queue = tf.train.string_input_producer(["XORdata.csv"])
#reader = tf.TextLineReader()
#key, value = reader.read(filename_queue)
#record_defaults = [[1], [1], [1], [1], [1]]
#col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=record_defaults)
#features = tf.stack([col1, col2, col3, col4])

def file_len(fname):
	with open(fname) as f:
		for i, l in enumerate(f):
			pass
	return i+1

filename = "XORdata.csv"

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
# args = parser.parse_args()


#setup text reader
file_length = file_len(filename)
filename_queue = tf.train.string_input_producer([filename])
reader = tf.TextLineReader(skip_header_lines=1)
_, csv_row = reader.read(filename_queue)

# setup CSV decoding
record_defaults = [[0], [0], [0], [0], [0]]
col1,col2,col3,col4,col5 = tf.decode_csv(csv_row, record_defaults=record_defaults)

# turn features back into a tensor
# features = tf.pack([col1,col2,col3,col4,col5])