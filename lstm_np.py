#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@version: Python 3.7
@author: Huihui Xu
@email: huihui.xu@pitt.edu
@date: 10/28/2021
"""

import numpy as np

def sigmoid(x):
	return 1/(1+np.exp(-x))

def tanh(x):
	return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

# input gate: control new information that enter in the cell
def input(w_ih, h, w_ix, x, b_i):
	"""
	paras:
	w_ih: weight matrix of input gate and hidden state h
	h: hidden state h at time {t-1} from the last hidden state
	w_ix: weight matrix of input gate and new information x
	x: new information at time t
	b_i: bias of input gate

	"""
	return sigmoid(np.matmul(w_ih, h) + np.matmul(w_ix, x) + b_i)

# forget gate: 
def forget(w_fh, h, w_fx, x, b_f):
	"""
	paras:
	w_fh: weight matrix of forget gate and hidden state h
	h: hidden state h at time {t-1} from the last hidden state
	w_fx: weight matrix of forget gate and new information x
	x: new information at time t
	b_f: bias of forget gate

	"""
	return sigmoid(np.matmul(w_fh, h) + np.matmul(w_fx, x) + b_f)

# output gate
def output(w_oh, h, w_ox, x, b_o):
	return sigmoid(np.matmul(w_oh, h) + np.matmul(w_ox, x) + b_o)


# memory cell
def memory(f, c, i, w_ch, h, w_cx, x, b_c):
	"""
	f: output of forget gate
	c: cell state from time {t-1}
	i: output of input gate
	w_ch: weight matrix of cell and hidden state
	h: hidden state from time {t-1}
	w_cx: weight matrix of cell and new information
	x: new information at time x
	b_c: bias of cell state

	"""
	curr_cell = tanh(np.matmul(w_ch, h) + np.matmul(w_cx, x) + b_c)
	cell = f.dot(c) + i.dot(curr_cell)
	return cell

# hidden state
def hidden(o, c):
	return o.dot(tanh(c))