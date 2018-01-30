import numpy as np
import random
import matplotlib.pyplot as plt


TRAIN_DATA_NUM = 100
INODE_NUM = 2
HNODE_NUM = 4
ONODE_NUM = 1

# generating training data, linear seperatable
def data_set_generator(mu1, mu2, sigma1, sigma2):
	train_data = np.zeros((INODE_NUM,TRAIN_DATA_NUM))
	train_labels = np.ones((ONODE_NUM,TRAIN_DATA_NUM))
	
	data_minus_one = []
	data_one = []
	for i in range(TRAIN_DATA_NUM):
		if random.random() < 0.5:
			x = random.normalvariate(mu1,sigma1)
			y = random.normalvariate(mu1,sigma1)
			train_labels[:,i] = -1
			data_minus_one.append([x,y])
			#j += 1 
		else:
			x = random.normalvariate(mu2,sigma2)
			y = random.normalvariate(mu2,sigma2)
			train_labels[:,i] = 1
			data_one.append([x,y])
			#k += 1
		train_data[:,i] = np.array([x,y])
	
	data_minus_one = np.array(data_minus_one)
	data_one = np.array(data_one)
	# for i in range(train_labels):
	# 	if train_labels[:,i] == 1:
	# 		data_one[:,i] = train_data[:,i]
	# 	else:
	# 		data_minus_one[:,i] = train
	print(np.shape(data_one))
	plt.scatter(data_minus_one.T[0],data_minus_one.T[1],label='minus_one', color='k', s=25, marker="o")
	plt.scatter(data_one.T[0],data_one.T[1],label='one', color='r', s=25, marker="x")
	# plt.show()
	return train_data, train_labels

# one layer perceptron
def perceptron(X, W, T, eta):
	X = np.row_stack((X,np.ones((1, TRAIN_DATA_NUM))))
	Y = np.dot(W,X) 
	# print((Y-T))
	delta_W = -eta*np.dot((Y-T), X.T)
	W = W + delta_W
	return W, Y

def sigmoid(X):
	return 1/(1+np.exp(-X))

def threshold(X):
	th = np.zeros(X.shape)
	th[X>0] = 1
	th[X<=0] = -1
	return th

# def eval(Y, T):
# 	output = threshold(Y)
# 	accuracy = np.sum(output == T) / TRAIN_DATA_NUM
# 	speratingLine = 

# def feedforward(X, W):
# 	return sigmoid(np.dot(X,W))

def data_set_shuffle(train_data, train_labels):
	data_set = np.row_stack((train_data,train_labels))
	np.random.shuffle(data_set)
	train_data = np.array(data_set[0:INODE_NUM,:])
	train_labels = np.array(data_set[INODE_NUM:INODE_NUM+ONODE_NUM,:])
	return train_data, train_labels


def train(train_data, train_labels, epochs, eta):
	W = np.random.randn(ONODE_NUM, INODE_NUM+1)
	i = 0
	while i < epochs:
		# train_data, train_labels = data_set_shuffle(train_data, train_labels)
		# delta rule updating W
		W, Y = perceptron(train_data, W, train_labels, eta)
	
		# print((Y-train_labels))
		# evaluating results
		# print(Y)
		output = threshold(Y)
		error = np.sum(np.power((Y-train_labels),2))/2
		accuracy = np.sum(output == train_labels) / TRAIN_DATA_NUM
		print("epoch:{}, accuracy is:{}, error is:{} ...\n".format(i, accuracy, error))
		if accuracy == 0:
			print("Y:{}, train labels are:{}".format(np.sum(Y), np.sum(train_labels)))
			print(output)
			print(train_labels)

		x = np.linspace(-4, 4, 100)
		speratingLine = -W[0,0]/W[0,1]*x - W[0,2]/W[0,1]
		plt.plot(x,speratingLine)
		plt.pause(0.2)
		# eta *= 0.9
		i += 1
	# while True:
	# 	plt.pause(0.1)
	plt.show()

def main():
	mu1 = -2
	mu2 = 2
	sigma1 = 0.5
	sigma2 = 0.5
	eta = 0.001
	epochs = 20
	train_data, train_labels = data_set_generator(mu1, mu2, sigma1, sigma2)
	train(train_data, train_labels, epochs, eta)

if __name__ == "__main__":
	main()

