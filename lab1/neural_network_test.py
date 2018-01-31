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

TRAIN_DATA_NUM = 100	#训练集大小
TEST_DATA_NUM = 50		#测试集大小	
INODE_NUM = 2			#输入层节点数
HNODE_NUM = 60			#中间层节点数
ONODE_NUM = 1			#输出层节点数


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
			a = sigmoid(np.dot(w, a)+b)
		return a

	def SGD(self, training_data, epochs, mini_batch_size, eta,
			test_data=None):
		"""Train the neural network using mini-batch stochastic
		gradient descent.  The ``training_data`` is a list of tuples
		``(x, y)`` representing the training inputs and the desired
		outputs.  The other non-optional parameters are
		self-explanatory.  If ``test_data`` is provided then the
		network will be evaluated against the test data after each
		epoch, and partial progress printed out.  This is useful for
		tracking progress, but slows things down substantially."""
		if test_data: 
			n_test = len(test_data)
			test_error = []
			test_accuracy = []
		n = len(training_data)
		train_error = []
		train_accuracy = []
		for j in range(epochs):
			random.shuffle(training_data)
			mini_batches = [
				training_data[k:k+mini_batch_size]
				for k in range(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta)

			error, train_correct = self.evaluate(training_data)
			train_error.append(error)
			train_accuracy.append(train_correct/n)

			if test_data:
				error, test_correct = self.evaluate(test_data)
				test_error.append(error)
				test_accuracy.append(test_correct/n_test)
				print("Epoch {0}: {1} / {2}".format(j, test_correct, n_test))
			else:
				print("Epoch {0} complete".format(j))

		plt.subplot(1,3,2)
		plt.plot([x for x in range(epochs)], train_accuracy, color="blue", label = "training accuracy", linewidth=2.0, linestyle="-")
		plt.plot([x for x in range(epochs)], test_accuracy, color="red", label = "test accuracy", linewidth=2.0, linestyle="-")
		plt.axis([0, epochs, 0, 1.2]) 
		plt.legend()
		
		plt.subplot(1,3,3)
		plt.plot([x for x in range(epochs)], train_error, color="blue", label = "training error", linewidth=2.0, linestyle="-")
		plt.plot([x for x in range(epochs)], test_error, color="red", label = "test error", linewidth=2.0, linestyle="-")
		plt.legend()

	def update_mini_batch(self, mini_batch, eta):
		"""Update the network's weights and biases by applying
		gradient descent using backpropagation to a single mini batch.
		The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
		is the learning rate."""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.weights = [w-(eta/len(mini_batch))*nw
						for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(eta/len(mini_batch))*nb
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
			z = np.dot(w, activation)+b
			# print("z shape is: {}".format(z.shape))
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		# backward pass
		delta = self.cost_derivative(activations[-1], y) * \
			sigmoid_prime(zs[-1])
		# print(delta.shape)
		# print(activations[-2].shape)
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
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
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nabla_b, nabla_w)

	def evaluate(self, test_data):	#网络准确率测试函数
		"""Return the number of test inputs for which the neural
		network outputs the correct result. Note that the neural
		network's output is assumed to be the index of whichever
		neuron in the final layer has the highest activation."""
		# test_results = [(np.argmax(self.feedforward(x)), y)
		# 				for (x, y) in test_data]
		for (x, y) in test_data:
			output = [(float(self.feedforward(x)), y) for (x, y) in test_data]
			error = [(x-y)*(x-y) for (x, y) in output]
			error_total = sum(error)/2
			test_results = [(threshold(x), y) for (x, y) in output]
			correct = sum(int(x == y) for (x, y) in test_results)
		return error_total, correct

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

def threshold(x):
	if x > 0.5:
		return 1
	else:
		return 0

# 数据集产生函数
def data_set_generator(mu1, mu2, sigma1, sigma2):
	test_data = []
	for i in range(TEST_DATA_NUM):
		if random.random() < 0.5:
			x = random.normalvariate(mu1,sigma1)
			y = random.normalvariate(mu1,sigma1)
			# print(np.array([x,y]).shape)
			test_data.append([np.array([[x],[y]]), 0])
		else:
			x = random.normalvariate(mu2,sigma2)
			y = random.normalvariate(mu2,sigma2)
			test_data.append([np.array([[x],[y]]), 1])

	train_data = []
	data_minus_one = []
	data_one = []
	for i in range(TRAIN_DATA_NUM):
		if random.random() < 0.5:
			x = random.normalvariate(mu1,sigma1)
			y = random.normalvariate(mu1,sigma1)
			train_data.append([np.array([[x],[y]]), 0])
			data_minus_one.append([x,y])
		else:
			x = random.normalvariate(mu2,sigma2)
			y = random.normalvariate(mu2,sigma2)
			train_data.append([np.array([[x],[y]]), 1])
			data_one.append([x,y])
	
	data_minus_one = np.array(data_minus_one)
	data_one = np.array(data_one)
	# for i in range(train_labels):
	# 	if train_labels[:,i] == 1:
	# 		data_one[:,i] = train_data[:,i]
	# 	else:
	# 		data_minus_one[:,i] = train
	# print(np.shape(data_one))
	plt.figure()
	plt.subplot(1,3,1)
	plt.scatter(data_minus_one.T[0],data_minus_one.T[1],label='class 0', color='k', s=25, marker="o")
	plt.scatter(data_one.T[0],data_one.T[1],label='class 1', color='r', s=25, marker="x")
	plt.legend()
	# plt.show()
	return train_data, test_data

# 主函数
def main():
	sizes = [INODE_NUM, HNODE_NUM, ONODE_NUM]
	nn = Network(sizes)

	# 数据集分布参数
	mu1 = -2
	mu2 = 2
	sigma1 = 2
	sigma2 = 2
	
	# learning相关参数
	eta = 0.1
	epochs = 100
	mini_batch_size = 20

	train_data, test_data = data_set_generator(mu1, mu2, sigma1, sigma2)
	nn.SGD(train_data, epochs, mini_batch_size, eta, test_data)

	plt.show()

if __name__ == "__main__":
	main()