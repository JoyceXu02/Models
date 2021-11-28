#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@version: Python 3.7
@author: Huihui Xu
@email: huihui.xu@pitt.edu
"""

import numpy as np

def clusters(data, inits, norm = 'l1'):
	dists = []
	cluster1, cluster2 = [],[]
	for init in inits:
		if norm == 'l1':
			dist = np.sum(np.abs(np.array(init) - data), axis = 1)
		else:
			dist = np.sqrt(np.sum((np.array(init) - data)**2, axis =1))
		dists.append(dist)
	dists = np.column_stack(dists)
	min_idx = list(np.argmin(dists, axis = 1)) # clustered data
	for i, idx in enumerate(min_idx):
		if idx == 0:
			cluster1.append(data[i])
		else:
			cluster2.append(data[i])
	return cluster1, cluster2

def min_cost(cluster, init, norm = 'l1'):

	cluster = np.array(cluster)
	new_init = list(np.median(cluster, axis = 0))
	return new_init

def kmeans(data, inits, T, norm = 'l1'):
	cluster1, cluster2 = clusters(data, inits, norm)

	for t in range(T):
		new_inits = []		
		new_init1=min_cost(cluster1, inits[0], norm)
		new_inits.append(list(new_init1))
		new_init2 = min_cost(cluster2, inits[1], norm)
		new_inits.append(list(new_init2))

	return new_inits, cluster1, cluster2


if __name__ == "__main__":
	data = np.array([[0,-6],[4,4], [0,0],[-5,2]])
	inits = [[-5,2],[0,-6]]
	new_inits, cluster1, cluster2 = kmeans(data, inits, 10)
	print(new_inits)
	print(cluster1)
	print(cluster2)