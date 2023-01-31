#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:05:15 2018

@author: edward
"""
import torch
from torch import nn
from .access import Access
import numpy


class DNC(nn.Module):
	def __init__(self, input_size: int, hidden_nodes: int, output_nodes: int,
	                   mem_size: 16, mem_n: 32, batch_size: 16, num_read_heads: 3):
		super(DNC, self).__init__()

		self.input_size = input_size
		self.hidden_nodes = hidden_nodes
		self.output_nodes = output_nodes

		self.mem_size = mem_size
		self.mem_n = mem_n
		self.num_read_heads = num_read_heads

		self.batch_size = batch_size

		self.controller_input_size = self.input_size + self.num_read_heads * self.mem_size
		self.linear_input_size = self.hidden_nodes + self.num_read_heads * self.mem_size

		self._controller = nn.LSTMCell(self.controller_input_size, self.hidden_nodes)
		self._access = Access(self.mem_size, self.num_read_heads, self.hidden_nodes)
		self._linear = nn.Linear(self.linear_input_size, self.output_nodes)

		nn.init.orthogonal_(self._controller.weight_hh)
		nn.init.orthogonal_(self._controller.weight_ih)
		nn.init.orthogonal_(self._linear.weight)

		self._linear.weight.data.fill_(0.0)
		

	def forward(self, x, state):
		read = state['read']
		c_state = state['c_state']
		memory = state['memory']
		a_state = state['a_state']
		link_matrix = state['link_matrix']

		# get the controller output
		h, c = c_state
		control_out, cell_state = self._controller(torch.cat([x, read], 1), c_state)

		c_state = ( control_out, cell_state)

		# split the controller output into interface and control vectors
		read, memory, a_state, link_matrix  = self._access(control_out, memory, a_state, link_matrix)

		output = self._linear(torch.cat([control_out, read], 1))

		state = {
			'read': read,
			'c_state': c_state,
			'a_state': a_state,
			'memory': memory,
			'link_matrix': link_matrix,
		}

		return output, state


	def reset(self):
		device = torch.device('cpu')
		#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		read = torch.zeros(self.batch_size, self.num_read_heads * self.mem_size).to(device)

		hidden = torch.zeros(self.batch_size, self.hidden_nodes).to(device)
		cells  = torch.zeros(self.batch_size, self.hidden_nodes).to(device)
		c_state = (hidden, cells)

		memory = torch.zeros(self.batch_size, self.mem_n, self.mem_size).to(device)
		link_matrix = torch.zeros(self.batch_size, self.mem_n, self.mem_n).to(device)

		r_weights = torch.zeros(self.batch_size, self.num_read_heads, self.mem_n).to(device)
		w_weights = torch.zeros(self.batch_size, self.mem_n).to(device)

		usage = torch.zeros(self.batch_size, self.mem_n).to(device)
		precedence = torch.zeros(self.batch_size, self.mem_n).to(device)

		a_state = r_weights, w_weights, usage, precedence

		state = {
			'read': read,
			'c_state': c_state,
			'a_state': a_state,
			'memory': memory,
			'link_matrix': link_matrix,
		}

		return state