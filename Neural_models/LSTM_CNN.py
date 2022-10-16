#!/usr/local/bin/env python3
# encoding: utf-8


"""
@version: Python3.7
@author: Huihui Xu
@email: huihui.xu@pitt.edu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.Embedding as Embedding

# Put LSTM before CNN

class LSTM_CNN(nn.Module):
	def __init__(self, embed_dim, vocab, n_filters, filter_sizes, output_dim, dropout,
				 hidden_dropout, n_layers, hidden_dim, target_size, bidirectional= False):
		super(LSTM_CNN,self).__init__()

		
		# embedding layer
		self.embedding = nn.Embedding(vocab, embed_dim)
		self.hidden_dim = hidden_dim
		# dropout layer
		self.droput = nn.Dropout(dropout)

		# list of conv layers
		self.convs = nn.MuduleList(
			[nn.Conv2d(1, n_filters, (filter_size, embed_dim))for filter_size in filter_sizes])


		self.lstm = nn.LSTM(self.embedding, 
							self.hidden_dim,
							num_layers = n_layers,
							bidrectional = bidrectional,
							dropout = hidden_dropout)

		# if bidireciotn
		if bidirectional:
			self.fc= nn.Linear(self.hidden_dim*2, target_size)
		else:
			self.lstm = nn.Linear(self.hidden_dim, target_size)

		self.fc = nn.Linear(len(filter_sizes)*n_filters, output_dim)


	def forward(self, text):





