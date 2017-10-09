# Getting the hang of tensorflow
# Source: https://www.tensorflow.org/get_started/get_started

# Basic imports

import numpy as np
import tensorflow as tf

# From https://github.com/tensorflow/tensorflow/issues/7778
# "Those are simply warnings. They are just informing you if
# you build TensorFlow from source it can be faster on your
# machine. Those instructions are not enabled by default on
# the builds available I think to be compatible with more
# CPUs as possible... To deactive these warnings... do the 
# following: "

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'	


#x = 1
#if x == 1:
#   p rint("x is 1.")
#print("hello world")
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
#print(node1)
#print(node2)
print(node1, node2)

# To actually extract values, run session
sess = tf.Session()
print(sess.run([node1,node2]))

# Let's do some computation with our computational graph
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

# Abstact into placeholders
# "A placeholder is a promise to private a value later"

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
# + provides a shortcut for tf.add(a,b)

print(sess.run(adder_node, {a:3, b:4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2,4]}))