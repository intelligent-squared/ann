'''
- one input, one neuron, one output, and one sample data
- weigh, bias, activation function
- feed forward
- loss function
- training process
	- the whole purpose of training process is findout value of weigh and bias 
	which minimize loss function
- gradient descend

'''

import numpy as np
import matplotlib.pyplot as plt

class Neuron:

	def __init__(self):
		self.weight = 0.7
		self.bias = 0.1
	
	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))
		
	def loss(self):
		'''
			use mean square error for loss function
		'''
		return np.sum(np.square(self.error()))/2

	def error(self):
		return self.feed_forward(self.input) - self.output
	
	def calculate_derivative(self):
		self.derivative_loss_bias = self.error()*self.sigmoid(self.z(self.input))*(1-self.sigmoid(self.z(self.input)))
		self.derivative_loss_weigh = self.derivative_loss_bias * self.input
	
	def update_weigh_bias(self):
		self.weight = self.weight - self.learing_rate*self.derivative_loss_weigh
		self.bias = self.bias - self.learing_rate*self.derivative_loss_bias
	
	def z(self, x):
		return self.weight*x + self.bias
	
	def feed_forward(self, x):
		return self.sigmoid(self.z(x))

	def train(self, 
			train_data, 
			learing_rate, 
			epoch):

		self.input = train_data[0]
		self.output = train_data[1]
		self.learing_rate = learing_rate
		self.epoch = epoch

		loss = []
		for i in range(self.epoch):
			self.calculate_derivative()
			self.update_weigh_bias()

			# check lost after update
			loss.append(self.loss())
			print(self.loss())


		self.plot_loss_on_epoch(epoch, loss)

	def plot_loss_on_epoch(self, epoch, loss):	
		plt.xticks(np.arange(0, epoch, 1))
		plt.plot(range(epoch), loss)
		plt.show()



# train data
input = np.array([0.2])
output = np.array([0.4])
train_data = (input, output)

# train parameter
learing_rate = 0.2
epoch = 1000

n = Neuron()
n.train(train_data, learing_rate, epoch)