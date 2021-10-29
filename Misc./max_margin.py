#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@version: Python 3.7
@author: Huihui Xu
@email: huihui.xu@pitt.edu
"""
import numpy as np
import matplotlib.pyplot as plt


def counting_thetas(data, labels, mistakes):
	if len(data)!=len(labels) or len(data) != len(mistakes):
		return "Check the inputs! "
	labels = np.array(labels)
	data = np.array(data)
	mistakes = np.array(mistakes)
	offset = labels.dot(mistakes)

	thetas = data * labels[:, np.newaxis]* mistakes[:, np.newaxis]
	thetas = np.sum(thetas, axis = 0)
	return offset, thetas




if __name__ == "__main__":
	data = np.array([[0,0],[2,0],[3,0],[0,2],[2,2],[5,1],[5,2],[2,4],[4,4],[5,5]])
	labels = np.array([-1, -1,-1,-1,-1,1,1,1,1,1])
	# plot data
	colors = ['b' if label == -1 else 'r' for label in labels]
	plt.figure()
	plt.scatter(data[:,0], data[:,1], s=40, c=colors)
	plt.show()

	mistakes = [1,9,10,5,9,11,0, 3,1,1]
	offset, thetas = counting_thetas(data, labels, mistakes)
	print(offset)
	print(thetas)