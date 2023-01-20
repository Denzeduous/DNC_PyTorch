#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:05:15 2018

@author: edward
"""
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class DNC(nn.Module):
	def __init__(self, controller: nn.LSTMCell, output_layer: nn.Linear):
		super(DNC, self).__init__()

		self.controller = controller
		self.access = Access(params)
		self.linear = output
		
		nn.init.orthogonal_(self.controller.weight_hh)
		nn.init.orthogonal_(self.controller.weight_ih)
		nn.init.orthogonal_(self.linear.weight)
		self.linear.weight.data.fill_(0.0)


	def forward(self, x, state):
		read = state['read']
		c_state = state['c_state']
		memory = state['memory']
		a_state = state['a_state']
		link_matrix = state['link_matrix']

		# get the controller output
		h, c = c_state
		control_out, cell_state = self.controller(torch.cat([x, read], 1), c_state)

		c_state = ( control_out, cell_state)

		# split the controller output into interface and control vectors
		read, memory, a_state, link_matrix  = self.access(control_out, memory, a_state, link_matrix)

		output = self.linear(torch.cat([control_out, read], 1))

		state = {'read': read,
				'c_state': c_state,
				'a_state': a_state,
				'memory': memory,
				'link_matrix': link_matrix}

		return output, state


	def reset(self):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		read = torch.zeros(self.params.batch_size, 
		                   self.params.num_read_heads * self.params.mem_size)
		            .to(device)
		
		hidden = torch.zeros(self.params.batch_size, self.params.c_out_size).to(device)
		cells = torch.zeros(self.params.batch_size, self.params.c_out_size) .to(device)
		c_state = (hidden,cells)
		
		memory = torch.zeros(self.params.batch_size, self.params.memory_n, self.params.mem_size).to(device)
		link_matrix = torch.zeros(self.params.batch_size, self.params.memory_n, self.params.memory_n).to(device)
		
		r_weights = torch.zeros(self.params.batch_size, self.params.num_read_heads, self.params.memory_n).to(device)
		w_weights = torch.zeros(self.params.batch_size, self.params.memory_n).to(device)
		
		usage = torch.zeros(self.params.batch_size, self.params.memory_n).to(device)
		precedence = torch.zeros(self.params.batch_size, self.params.memory_n).to(device)
		
		a_state = r_weights, w_weights, usage, precedence
		
		state = {'read': read,
				'c_state': c_state,
				'a_state': a_state,
				'memory': memory,
				'link_matrix': link_matrix}       
		
		return state


class Params:
	def __init__(self):
		self.input_size = 8
		self.c_out_size = 64
		self.l_out_size = self.input_size - 1
		self.mem_size = 16
		self.memory_n = 32    
		self.batch_size = 16
		self.num_read_heads = 3
		self.seq_length = 20
		
		self.l_in_size = self.c_out_size + self.num_read_heads * self.mem_size
		self.c_in_size = self.input_size + self.num_read_heads * self.mem_size