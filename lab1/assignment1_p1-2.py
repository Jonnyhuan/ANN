import numpy as np
import random
import matplotlib.pyplot as plt


TRAIN_DATA_NUM = 100
VALID_DATA_NUM = 50
INODE_NUM = 2
HNODE_NUM = 50
ONODE_NUM = 1

# generating training data, linear seperatable
def data_set_generator(mu1, mu2, sigma1, sigma2):
	valid_data = np.zeros((INODE_NUM,VALID_DATA_NUM))
	valid_labels = np.ones((ONODE_NUM,VALID_DATA_NUM))

	for i in range(VALID_DATA_NUM):
		if random.random() < 0.5:
			x = random.normalvariate(mu1,sigma1)
			y = random.normalvariate(mu1,sigma1)
			valid_labels[:,i] = 0
		else:
			x = random.normalvariate(mu2,sigma2)
			y = random.normalvariate(mu2,sigma2)
			valid_labels[:,i] = 1
		valid_data[:,i] = np.array([x,y])

	train_data = np.zeros((INODE_NUM,TRAIN_DATA_NUM))
	train_labels = np.ones((ONODE_NUM,TRAIN_DATA_NUM))
	
	data_minus_one = []
	data_one = []
	for i in range(TRAIN_DATA_NUM):
		if random.random() < 0.5:
			x = random.normalvariate(mu1,sigma1)
			y = random.normalvariate(mu1,sigma1)
			train_labels[:,i] = 0
			data_minus_one.append([x,y])
		else:
			x = random.normalvariate(mu2,sigma2)
			y = random.normalvariate(mu2,sigma2)
			train_labels[:,i] = 1
			data_one.append([x,y])
		train_data[:,i] = np.array([x,y])
	
	data_minus_one = np.array(data_minus_one)
	data_one = np.array(data_one)
	# for i in range(train_labels):
	# 	if train_labels[:,i] == 1:
	# 		data_one[:,i] = train_data[:,i]
	# 	else:
	# 		data_minus_one[:,i] = train
	plt.subplot(1,2,1)
	plt.scatter(data_minus_one.T[0],data_minus_one.T[1],label='minus_one', color='k', s=25, marker="o")
	plt.scatter(data_one.T[0],data_one.T[1],label='one', color='r', s=25, marker="x")
	plt.legend()
	# plt.show()
	return train_data, train_labels, valid_data, valid_labels

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_prime(x):
	return sigmoid(x)*(1-sigmoid(x))

def tanh(z):
	return (2.0/(1+np.exp(-z)))-1

def tanh_prime(z):
	return ((1+tanh(z))*(1-tanh(z)))/2

def threshold(x):
	th = np.zeros(x.shape)
	th[x>0.5] = 1
	th[x<=0.5] = 0
	return th

# forwardpass
def forwardpass(x, hx_w, oh_w):
	hnode_in = np.dot(hx_w, x)
	hnode = tanh(hnode_in)
	hnode = np.row_stack((hnode, np.ones((1, x.shape[1]))))

	onode_in = np.dot(oh_w, hnode)
	onode = tanh(onode_in)

	return hnode_in, hnode, onode_in, onode


def eval(x, hx_w, oh_w, labels, epoch):
	x = np.row_stack((x, np.ones((1, x.shape[1]))))
	hnode_in, hnode, onode_in, onode = forwardpass(x, hx_w, oh_w)
	y = threshold(onode)
	error = np.sum(np.power((onode-labels),2))/2
	accuracy = np.sum(y == labels) / x.shape[1]
	# print("epoch:{}, accuracy is:{}, error is:{} ...\n".format(epoch, accuracy, error))
	# if accuracy == 0:
	# 	print("y:{}, train labels are:{}".format(np.sum(Y), np.sum(train_labels)))
	# 	print(output)
	# 	print(train_labels)	
	return accuracy


def train(train_data, train_labels, valid_data, valid_labels, epochs, eta):
	x = np.row_stack((train_data, np.ones((1, train_data.shape[1]))))
	hx_w = np.random.randn(HNODE_NUM, INODE_NUM+1)
	oh_w = np.random.randn(ONODE_NUM, HNODE_NUM+1)
	i = 0
	accuracy = []
	while i < epochs:
		# forward pass
		hnode_in, hnode, onode_in, onode = forwardpass(x, hx_w, oh_w)

		# BP
		delta_o = (onode - train_labels) * tanh_prime(onode_in)
		delta_h = np.dot(oh_w[:,:HNODE_NUM].T, delta_o) * tanh_prime(hnode_in)
		# delta_h = np.delete(delta_h, -1, axis = 1)

		# update weights
		d_hx_w = -eta * np.dot(delta_h, x.T)
		d_oh_w = -eta * np.dot(delta_o, hnode.T)
		hx_w += d_hx_w
		oh_w += d_oh_w

		accuracy.append(eval(valid_data, hx_w, oh_w, valid_labels, i))
		i += 1

	plt.subplot(1,2,2)
	ep = [ep for ep in range(epochs)]
	plt.plot(ep, accuracy)
	plt.show()

def main():
	mu1 = -2
	mu2 = 2
	sigma1 = 2
	sigma2 = 2
	eta = 0.0001
	epochs = 200
	train_data, train_labels, valid_data, valid_labels = data_set_generator(mu1, mu2, sigma1, sigma2)
	train(train_data, train_labels, valid_data, valid_labels, epochs, eta)

if __name__ == "__main__":
	main()