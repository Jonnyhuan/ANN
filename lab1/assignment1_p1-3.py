# coding: UTF-8 
"""
network.py
~~~~~~~~~~
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt

USE_TEST_SET = False
USE_VALID_SET = False
BATCH_NORM = False
TRAIN_DATA_NUM = 8	#训练集大小
VALID_DATA_NUM = 0	#训练集大小
TEST_DATA_NUM = 0		#测试集大小	

# 2-layer neural network configuration
N2_INODE_NUM = 8			#输入层节点数
N2_HNODE_NUM = 3			#中间层节点数
N2_ONODE_NUM = 8			#输出层节点数

# 3-layer neural network configuration
N3_INODE_NUM = 5			#输入层节点数
N3_H1NODE_NUM = 4			#中间层节点数
N3_H2NODE_NUM = 4			#中间层节点数
N3_ONODE_NUM = 1			#输出层节点数


class Network(object):

	def __init__(self, sizes):
		"""The list ``sizes`` contains the number of neurons in the
		respective layers of the network.  For example, if the list
		was [2, 3, 1] then it would be a three-layer network, with the
		first layer containing 2 neurons, the second layer 3 neurons,
		and the third layer 1 neuron.  The biases and weights for the
		network are initialized randomly, using a Gaussian
		distribution with mean 0, and variance 1.  Note that the first
		layer is assumed to be an input layer, and by convention we
		won't set any biases for those neurons, since biases are only
		ever used in computing the outputs from later layers."""
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x)
						for x, y in zip(sizes[:-1], sizes[1:])]

	def feedforward(self, a):
		"""Return the output of the network if ``a`` is input."""
		for b, w in zip(self.biases, self.weights):
			a = tanh(np.dot(w, a)+b)
		return a

	def SGD(self, training_data, epochs, mini_batch_size, eta, weight_decay = 0, validation_data=None, test_data=None):
		"""Train the neural network using mini-batch stochastic
		gradient descent.  The ``training_data`` is a list of tuples
		``(x, y)`` representing the training inputs and the desired
		outputs.  The other non-optional parameters are
		self-explanatory.  If ``test_data`` is provided then the
		network will be evaluated against the test data after each
		epoch, and partial progress printed out.  This is useful for
		tracking progress, but slows things down substantially."""

		if USE_TEST_SET: 
			n_test = test_data.shape[1]
			test_error = []
			test_accuracy = []
		if USE_VALID_SET: 
			n_valid = validation_data.shape[1]
			validation_error = []
			validation_accuracy = []

		n = training_data.shape[1]
		train_error = []
		train_accuracy = []
		stop_epoch = epochs
		for j in range(epochs):
			stop_epoch = j
			# random.shuffle(training_data)
			for k in range(0, n, mini_batch_size):
				mini_batch = training_data[:, k:k+mini_batch_size]
				# print(mini_batch)
				self.update_mini_batch(mini_batch, eta, weight_decay)

			mse, train_correct_num = self.evaluate(training_data)
			train_error.append(mse)
			train_accuracy.append(train_correct_num/n)

		plt.subplot(1,2,1)
		plt.plot([x for x in range(len(train_accuracy))], train_accuracy, color="blue", label = "training accuracy", linewidth=2.0, linestyle="-")
		plt.axis([0, stop_epoch+1, 0, 1.2]) 
		plt.legend()

		plt.subplot(1,2,2)
		plt.plot([x for x in range(len(train_error))], train_error, color="blue", label = "training error", linewidth=2.0, linestyle="-")
		plt.xlim(0, stop_epoch+1) 
		plt.legend()

		x = training_data[0:8, :]
		mse, train_correct_num = self.evaluate(training_data)
		# print("mse is:{}, accuracy: {}/{}".format(mse,train_correct_num,n))
		# print("output is:{}".format(self.feedforward(x)))
		

	def update_mini_batch(self, mini_batch, eta, weight_decay=0):
		"""Update the network's weights and biases by applying
		gradient descent using backpropagation to a single mini batch.
		The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
		is the learning rate."""

		if BATCH_NORM == True:
			mini_batch_length = mini_batch.shape[1]
		else:
			mini_batch_length = 1

		x = mini_batch[0:8, :]
		y = mini_batch[8:, :]
		# print(x)
		nabla_b, nabla_w = self.backprop(x, y)
		self.weights = [(1-eta*weight_decay/mini_batch_length)*w-(eta/len(mini_batch))*nw
						for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(eta*weight_decay/mini_batch_length)*nb
					   for b, nb in zip(self.biases, nabla_b)]

	def backprop(self, x, y):
		"""Return a tuple ``(nabla_b, nabla_w)`` representing the
		gradient for the cost function C_x.  ``nabla_b`` and
		``nabla_w`` are layer-by-layer lists of numpy arrays, similar
		to ``self.biases`` and ``self.weights``."""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		# feedforward
		activation = x
		activations = [x] # list to store all the activations, layer by layer
		zs = [] # list to store all the z vectors, layer by layer
		for b, w in zip(self.biases, self.weights):
			temp = np.dot(b, np.ones((1,activation.shape[1])))
			# print(w.shape, activation.shape, temp.shape)
			z = np.dot(w, activation) + temp
			zs.append(z)
			activation = tanh(z)
			activations.append(activation)
		# backward pass
		delta = self.cost_derivative(activations[-1], y) * \
			tanh_prime(zs[-1])
		nabla_b[-1] = np.sum(delta,axis=1).reshape((delta.shape[0],1))
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		# Note that the variable l in the loop below is used a little
		# differently to the notation in Chapter 2 of the book.  Here,
		# l = 1 means the last layer of neurons, l = 2 is the
		# second-last layer, and so on.  It's a renumbering of the
		# scheme in the book, used here to take advantage of the fact
		# that Python can use negative indices in lists.
		for l in range(2, self.num_layers):
			z = zs[-l]
			sp = tanh_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = np.sum(delta,axis=1).reshape((delta.shape[0],1))
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nabla_b, nabla_w)

	def evaluate(self, test_data):	#网络准确率测试函数
		"""Return the number of test inputs for which the neural
		network outputs the correct result. Note that the neural
		network's output is assumed to be the index of whichever
		neuron in the final layer has the highest activation."""
		# test_results = [(np.argmax(self.feedforward(x)), y)
		# 				for (x, y) in test_data]
		
		x = test_data[0:8, :]
		y = test_data[8:, :]
		# print("x:")
		# print(x)
		output = self.feedforward(x)
		# print("output:")
		# print(output)
		error = output-y
		# print(np.argmax(y,axis=0))
		# print(np.argmax(output,axis=0))
		mse = np.sum(np.power(error,2)) / 2
		# print(output.shape, y.shape, error.shape, mse.shape, np.abs(error))
		test_results = [(np.argmax(x), np.argmax(y)) for (x,y) in zip(output, y)]
		correct_num = sum(int(x==y) for (x,y) in test_results)
		# print("correct number is:{}".format(correct_num))
		return mse, correct_num

	def cost_derivative(self, output_activations, y):
		"""Return the vector of partial derivatives \partial C_x /
		\partial a for the output activations."""
		return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
	"""The sigmoid function."""
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	"""Derivative of the sigmoid function."""
	return sigmoid(z)*(1-sigmoid(z))

def tanh(z):
	return (2.0/(1+np.exp(-z)))-1

def tanh_prime(z):
	return ((1+tanh(z))*(1-tanh(z)))/2

# def threshold(x):
# 	if x > 0.5:
# 		return 1
# 	else:
# 		return 0

# AutoEncoder 数据集
def data_set_generator():

	x = np.eye(8)
	x[x==0] = -1

	training_data = np.concatenate((x,x),axis = 1)

#	print(training_data.T)

	return training_data.T


# 主函数
def main():
	sizes = [N2_INODE_NUM, N2_HNODE_NUM, N2_ONODE_NUM] # 2-layer neural network
	nn = Network(sizes)
	
	# # learning相关参数
	eta = 0.5
	epochs = 20000
	mini_batch_size = TRAIN_DATA_NUM
	weight_decay_lamda = 0

	training_data = data_set_generator()
	nn.SGD(training_data, epochs, mini_batch_size, eta, weight_decay_lamda, training_data, training_data)

	plt.show()

if __name__ == "__main__":
	main()








