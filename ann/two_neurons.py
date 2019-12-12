'''
- two input, two neural, two output
- rebuild every thing for neural net
'''

import numpy as np

class NeuralNetwork():

	def __init__(self):
		self.weigh = np.random.randn(2,2)
		print(self.weigh, self.weigh)

		self.bias = np.random.randn(2,1)
		print(self.bias)

	def feed_forward(self):
		pass

	def z(self):
		# z0 = self.weigh[0]

	def train(self, train_data):
		self.input = train_data[0]
		print(self.input)
		print(np.transpose(self.input))

		self.output = train_data[1]


input = np.array([[0.1, 0.3]])
target = np.array([[0.4, 0.9]])
train_data = (input, target)

nn = NeuralNetwork()

nn.train(train_data)

