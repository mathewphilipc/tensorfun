# adapted from http://adventuresinmachinelearning.com/neural-networks-tutorial/

import matplotlib.pylab as plt
import numpy as np
x = np.arange(-8, 8, 0.1)
f = 1 / (1 + np.exp(-x))

w1 = 0.5
w2 = 1.0
w3 = 2.0
l1 = 'w = 0.5'
l2 = 'w = 1.0'
l3 = 'w = 2.0'
for w, l in [(w1, l1), (w2, l2), (w3, l3)]:
    f = 1 / (1 + np.exp(-x*w))
    plt.plot(x, f, label=l)
plt.xlabel('x')
plt.ylabel('h_w(x)')
plt.legend(loc=2)
# plt.show()

# matrix of weights connecting input layer to hidden layer
# initialized to example starting weights
w1 = np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6]])

# matrix of weights connecting hidden layer to output layer
# formalized as 1x3 matrix
w2 = np.zeros((1,3))
w2[0,:] = np.array([0.5, 0.5, 0.5]);
#print('%g', w2[0])

# biases
b1 = np.array([0.8, 0.8, 0.8])
b2 = np.array([0.2])

def f(x):
	return 1 / (1 + np.exp(-x))

def simple_looped_nn_calc(n_layers, x, w, b):
	for l in range(n_layers - 1):
		# Set up input array
		# If it's the first layer, this will just be the x input vector
		# Otherwise it will be the output of the previous layer
		if l==0:
			node_in = x
		else:
			node_in = h
		h = np.zeros((w[l].shape[0],))

		for i in range(w[l].shape[0]):
			f_sum = 0
			for j in range(w[1].shape[1]):
				f_sum += w[l][i][j] * node_in[j]
			f_sum += b[l][i]
			h[i] = f(f_sum)
	return h

w = [w1, w2]
b = [b1, b2]
x = [1.5, 2.0, 3]

#  simple_looped_nn_calc(3, x, w, b)


# better ff using matrix notation


def matrix_feed_forward_calc(n_layers, x, w, b):
	for l in range(n_layers - 1):
		if l == 0:
			node_in = x
		else:
			node_in = z
		z = w[l].dot(node_in) + b[l]
		h = f(z)
	return h

# Fiddling with multiple hidden layers before we abstract further

newX = x;
newW = [w1, w1, w2]
newB = [b1, b1, b2]
# print matrix_feed_forward_calc(4, newX, newW, newB)