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

USE_TEST_SET = True
USE_VALID_SET = True
TRAIN_DATA_NUM = 800	#训练集大小
VALID_DATA_NUM = 200	#训练集大小
TEST_DATA_NUM = 200		#测试集大小	
INODE_NUM = 5			#输入层节点数
HNODE_NUM = 8			#中间层节点数
ONODE_NUM = 1			#输出层节点数

START = 301
TEST_START = 1301
END = 1500

INDEX_START = -100
INDEX_END = 2000

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
			#print(w.shape, a.shape, b.shape)
			a = sigmoid(np.dot(w, a)+b)
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
		for j in range(epochs):
			# random.shuffle(training_data)
			for k in range(0, n, mini_batch_size):
				mini_batch = training_data[:, k:k+mini_batch_size]
				self.update_mini_batch(mini_batch, eta, weight_decay)

			# error, train_correct = self.evaluate(training_data)
			# train_error.append(error)
			# train_accuracy.append(train_correct/n)

			if USE_VALID_SET:
				mse, validation_correct_num = self.evaluate(validation_data)
				validation_error.append(mse)
				validation_accuracy.append(validation_correct_num/n_valid)
				# print("Epoch {0}, validation set: {1} / {2}".format(j, validation_correct, n_valid))
			# else:
				# print("Epoch {0}, validation set complete".format(j))

			if USE_TEST_SET:
				mse, test_correct_num = self.evaluate(test_data)
				test_error.append(mse)
				test_accuracy.append(test_correct_num/n_test)
				# print("Epoch {0}, test set: {1} / {2}".format(j, test_correct, n_test))
			# else:
				# print("Epoch {0}, test set complete".format(j))

		test_range = [x for x in range(TEST_START,END+1)]
		x = test_data[0:-1, :]
		output = self.feedforward(x)
		plt.subplot(2,1,1)
		plt.plot(test_range,output.flatten(),'g--')
		plt.legend()

		plt.subplot(2,2,3)
		plt.plot([x for x in range(epochs)], validation_accuracy, color="blue", label = "validation accuracy", linewidth=2.0, linestyle="-")
		plt.plot([x for x in range(epochs)], test_accuracy, color="red", label = "test accuracy", linewidth=2.0, linestyle="-")
		plt.axis([0, epochs, 0, 1.2]) 
		plt.legend()
		
		plt.subplot(2,2,4)
		plt.plot([x for x in range(epochs)], validation_error, color="blue", label = "validation error", linewidth=2.0, linestyle="-")
		plt.plot([x for x in range(epochs)], test_error, color="red", label = "test error", linewidth=2.0, linestyle="-")
		plt.legend()

	def update_mini_batch(self, mini_batch, eta, weight_decay=0):
		"""Update the network's weights and biases by applying
		gradient descent using backpropagation to a single mini batch.
		The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
		is the learning rate."""

		# nabla_b = [np.zeros(b.shape) for b in self.biases]
		# nabla_w = [np.zeros(w.shape) for w in self.weights]

		mini_batch_length = mini_batch.shape[1]
		x = mini_batch[0:-1, :]
		y = mini_batch[-1, :]
		# print(x.shape)
		# print(y.shape)
		# delta_nabla_b, delta_nabla_w = self.backprop(x, y)
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
			# print("activation shape is: {}".format(activation.shape))
			# print("w shape is: {}".format(w.shape))
			# print(b.shape)
			temp = np.dot(b, np.ones((1,activation.shape[1])))
			z = np.dot(w, activation) + temp
			# print(w.shape, activation.shape, b.shape, temp.shape, z.shape)
			# print("z shape is: {}".format(z.shape))
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		# backward pass
		delta = self.cost_derivative(activations[-1], y) * \
			sigmoid_prime(zs[-1])
		# print(delta.shape)
		# print(activations[-2].shape)
		nabla_b[-1] = np.sum(delta,axis=1)
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		# print(delta.shape, activations[-2].transpose().shape, nabla_b[-1].shape, nabla_w[-1].shape)
		# Note that the variable l in the loop below is used a little
		# differently to the notation in Chapter 2 of the book.  Here,
		# l = 1 means the last layer of neurons, l = 2 is the
		# second-last layer, and so on.  It's a renumbering of the
		# scheme in the book, used here to take advantage of the fact
		# that Python can use negative indices in lists.
		for l in range(2, self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = np.sum(delta,axis=1).reshape((delta.shape[0],1))
			# print(sp.shape, nabla_b[-l].shape)
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nabla_b, nabla_w)

	def evaluate(self, test_data):	#网络准确率测试函数
		"""Return the number of test inputs for which the neural
		network outputs the correct result. Note that the neural
		network's output is assumed to be the index of whichever
		neuron in the final layer has the highest activation."""
		# test_results = [(np.argmax(self.feedforward(x)), y)
		# 				for (x, y) in test_data]
		
		x = test_data[0:-1, :]
		y = test_data[-1, :]
		output = self.feedforward(x)
		error = output-y
		mse = np.sum(np.power(error,2)) / 2
		correct_num = np.sum(error<0.05)
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

# def threshold(x):
# 	if x > 0.5:
# 		return 1
# 	else:
# 		return 0

# 数据集产生函数
def data_set_generator():

	y = []
	t = [x for x in range(INDEX_START,INDEX_END)]
	offset = INDEX_START
	for i in t:
		if i < 0:
			y.append(0)
		elif i == 0:
			y.append(0.9*1.5)
		elif i < 25:
			y.append(0.9*y[-1])
		else:
			y.append(y[-1] + 0.2*y[i-26]/(1+pow(y[i-26],10)) - 0.1*y[-1])

	plt.subplot(2,1,1)
	plt.plot(t,y,'r-')
	# plt.legend()

	training_data = []
	validation_data = []
	test_data = []
	training_range = [x for x in range(START-offset,TEST_START-offset)]
	random.shuffle(training_range)
	j = 0
	for i in training_range:
		if j < TRAIN_DATA_NUM:
			# training_data.append([np.array([[y[i-20]], [y[i-15]], [y[i-10]],[y[i-5]], [y[i]]]), y[i+5]])
			training_data.append([y[i-20], y[i-15], y[i-10],y[i-5], y[i], y[i+5]])
		else:
			# validation_data.append([np.array([[y[i-20]], [y[i-15]], [y[i-10]],[y[i-5]], [y[i]]]), y[i+5]])
			validation_data.append([y[i-20], y[i-15], y[i-10],y[i-5], y[i], y[i+5]])
		j += 1

	for i in range(TEST_START-offset,END-offset+1):
		# test_data.append([np.array([[y[i-20]], [y[i-15]], [y[i-10]],[y[i-5]], [y[i]]]), y[i+5]])
		test_data.append([y[i-20], y[i-15], y[i-10],y[i-5], y[i], y[i+5]])

	return np.array(training_data).T, np.array(validation_data).T, np.array(test_data).T

# 主函数
def main():
	sizes = [INODE_NUM, HNODE_NUM, ONODE_NUM]
	nn = Network(sizes)
	
	# # learning相关参数
	eta = 0.3
	epochs = 1000
	mini_batch_size = 50
	weight_decay_lamda = 0.01

	training_data, validation_data, test_data = data_set_generator()
	nn.SGD(training_data, epochs, mini_batch_size, eta, weight_decay_lamda, validation_data, test_data)

	plt.show()

if __name__ == "__main__":
	main()