#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@version: Python 3.7
@author: Huihui Xu
@email: huihui.xu@pitt.edu
@date: 10/29/2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):

	def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, target_size, dropout, 
				bidrectional = False):
		super().__init__()
		self.hidden_dim = hidden_dim
		# embedding layer
		self.embedding = nn.Embedding(vocab, embed_dim)
		self.bidrectional = bidrectional
		self.lstm = nn.LSTM(self.embedding, 
							self.hidden_dim,
							num_layers = n_layers,
							bidrectional = bidrectional,
							dropout = dropout)
		if bidrectional:
			self.fc = nn.Linear(self.hidden_dim*2, target_size)
		else:
			self.fc = nn.Linear(self.hidden_dim, target_size)
		self.droput = nn.Dropout(dropout)

	def forward(self, text, max_length):
		embed = self.embedding(text)

		padded_embed = nn.utils.rnn.pack_padded_sequence(embed, max_length.to('cpu'))

		packed_output, (hidden, cell) = self.lstm(padded_embed)

		output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

		if bidrectional:
			hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))

		fc = self.fc(hidden)
		return fc



