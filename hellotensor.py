import numpy as np
import tensorflow as tf
x = 1
if x == 1:
    print("x is 1.")
print("hello world")
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1)
print(node2)
print(node1, node2)
