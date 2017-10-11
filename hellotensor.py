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
#print(node1, node2)

# To actually extract values, run session
sess = tf.Session()
#print(sess.run([node1,node2]))

# Let's do some computation with our computational graph
node3 = tf.add(node1, node2)
#print("node3:", node3)
#print("sess.run(node3):", sess.run(node3))

# Abstact into placeholders
# "A placeholder is a promise to private a value later"

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
# + provides a shortcut for tf.add(a,b)

#print(sess.run(adder_node, {a: 3, b: 4.5}))
#print(sess.run(adder_node, {a: [1, 3], b: [2,4]}))

add_and_triple = adder_node * 3.
#print(sess.run(add_and_triple, {a: 3, b: 4.5}))

# Throw in some variables and build a linear model
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# Constants are initialized when you call tf.constant, and their value can never change.
# By contrast, variables are not initialized when you call tf.Variable. To initialize all
# the variables in a TensorFlow program, you must explicitly call a special operation as
# follows:

init = tf.global_variables_initializer()
sess.run(init)
# It is import to realize init is a handle to the TensorFlow sub-graph that initializes
# all the global variables. Until we call sess.run, that variables are uninitialized.
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

# From linear model to some real linear analysis

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# On inspection, the ideal parameters are W = -1 and b = 1
# A variable is initialized to the value provided to tf.Variable but can be changed using
# operations like tf.assign.

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))