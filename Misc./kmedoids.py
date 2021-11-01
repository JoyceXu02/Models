#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@version: Python 3.7
@author: Huihui Xu
@email: huihui.xu@pitt.edu
"""
import numpy as np
import matplotlib.pyplot as plt

def clusters(data, inits, norm = 'l1'):
	dists = []
	cluster1, cluster2 = [],[]
	for init in inits:
		if norm == 'l1':
			dist = np.sum(np.abs(init - np.array(data)), axis = 1)
		else:
			dist = np.sqrt(np.sum((init -np.array(data))**2, axis =1))
		dists.append(dist)	
	# compare and sign to corresponding clusters
	dists = np.column_stack(dists) # dim: n_data * n_clusters
	min_idx = list(np.argmin(dists, axis = 1)) # clustered data
	for i, idx in enumerate(min_idx):
		if idx ==0:
			cluster1.append(data[i])
		else:
			cluster2.append(data[i])
	return cluster1, cluster2

def min_cost(cluster, init, norm = 'l1'):
	if norm == 'l1':
		init_cost = np.sum(np.abs(init - np.array(cluster)))
	else:
		init_cost = np.sum(np.sqrt(np.sum((init - np.array(cluster))**2, axis =1)))
	new_init = None
	dists = []
	for point in cluster:
		if norm == 'l1':
			dist = np.sum(np.sum(np.abs(point - np.array(cluster)), axis = 1))		
		else:
			dist = np.sum(np.sqrt(np.sum((point -np.array(cluster))**2, axis = 1)))
		if dist < init_cost:
			new_init = point
		else:
			new_init = init
		
	return new_init



def kmedoids(data, inits, T, norm = 'l1'):
	cluster1, cluster2 = clusters(data, inits, norm )
	
	for t in range(T):
		new_inits = []		
		new_init1=min_cost(cluster1, inits[0], norm)
		new_inits.append(list(new_init1))
		new_init2 = min_cost(cluster2, inits[1], norm)
		new_inits.append(list(new_init2))

	return new_inits, cluster1, cluster2


if __name__ == "__main__":
	data = np.array([[0,-6],[4,4], [0,0],[-5,2]])
	# plot data
	# plt.figure()
	# plt.scatter(data[:,0], data[:,1], s=40)
	# plt.show()
	
	new_inits, cluster1, cluster2= kmedoids(data, [[-5,2],[0,-6]], 10, 'l2')
	print(new_inits)
	print(cluster1)
	print(cluster2)
	



