"""
	- init image array
	- init filter array
	- calculate convolution of 2 matrix 
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import misc


class CNN():

	def __init__(self, image):
		# init layer and filter
		# image_layer = np.ones((5,5))
		# filter = np.ones((3,3)) * 2

		# print(image_layer)
		# print(filter)

		# self.conv(image_layer, filter, 1)

		# edge detection
		filter = np.array([[-1., -1., -1.],
						[-1., 8., -1.],
						[-1., -1., -1.]])

		# the sobel edge operator, horizontal
		filter = np.array([[-1., -2., -1.],
						[0., 0., 0.],
						[1., 2., 1.]])

		# the sobel edge operator, vertical
		filter = np.array([[-1., 0., 1.],
						[-2., 0., 2.],
						[-1., 0., 1.]])

		print('filter \n' ,filter)

		image = self.read_img(image)
		out = self.conv(image, filter, 1)
		

	def read_img(self, img):
		image = misc.imread(img, flatten=0)
		print(image.shape)
		plt.imshow(image)
		plt.show()
		return image


	def conv(self, image, filter, stride):

		# image
		image_row = image.shape[0]
		print('image row', image_row)
		image_col = image.shape[1]
		print('image col', image_col)

		# filter
		filter_row = filter.shape[0]
		print('filter_row', filter_row)
		filter_col = filter.shape[1]
		print('filter_col', filter_col)

		# out
		out_row = math.floor((image_row - filter_row) / stride) + 1
		print('out_row', out_row)

		out_col = math.floor((image_col - filter_col) / stride) + 1
		print('out_col', out_col)

		out_channel = image.shape[2]


		# convolute
		out = np.random.rand(out_row, out_col, out_channel)

		# calculate each output point
		for channel in range(out_channel):
			for col in range(out_col):
				for row in range(out_row):
					image_part = image[row:row+filter_row, col:col+filter_col, channel]
					# print(image_part)
					out[row,col,channel] = np.sum(np.multiply(image_part, filter))
					# print(out[row,col])

		print(out)
		plt.imshow(out)
		plt.show()
		return out


# c = CNN('lena_gray.bmp')
c = CNN('lena_color.tiff')



