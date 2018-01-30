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