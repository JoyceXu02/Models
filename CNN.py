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

class CNN(nn.Module):
	def __init__(self, vocab, embed_dim, n_filters, filter_sizes, output_dim, dropout):
		super(self).__init__()
		"""
		Parameters
		vocab: vocabulary size
		embed_dim: embedding dimensionality 
		n_filters: number of filters(for each size of filter)
		filter_sizes: different sizes of filters(eg: 2, 3, 4)
		output_dim: if classification, then it's number of labels
		dropout: dropout to prevent overfitting

		"""


		# embedding layer
		self.embedding = nn.Embedding(vocab, embed_dim)
		# list of conv layers
		self.convs = nn.MuduleList(
			[nn.Conv2d(1, n_filters, (filter_size, embed_dim))for filter_size in filter_sizes])
		self.droput = nn.Dropout(dropout)
		self.fc = nn.Linear(len(filter_sizes)*n_filters, output_dim)

	def forward(self, text):

		# [batch_size, sent_len, embed_dim]
		embed = self.embedding(text)

		# [batch_size, 1, sent_len, embed_dim]
		# unsqueeze: match the input shape requirement of Con2d
		embed = embed.unsqueeze(1)

		# [batch_size, n_filters, sent_len-filter_sizes[i]+1] * len(filter_sizes)
		convd = [F.relu(conv(embed)) for conv in self.convs]

		# [batch_size, n_filters] * len(convd)
		pooled = [F.max_pool1d(convdd, kernel_size = convdd.shape[2]).squeeze(2) for convdd in convd]

		# [batch_size, n_filters*len(filter_sizes)]
		# concatenate max pooled list to a fully connected layer.
		cat = torch.cat(pooled, dim = 1)

		# [batch_size, output_dim]
		fc = self.fc(self.dropout(cat))
		return fc

