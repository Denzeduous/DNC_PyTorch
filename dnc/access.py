import torch
from torch import nn
import torch.nn.functional as tfunc
from .heads import ReadHead, WriteHead


def index_slice(x, sizes):
	tot = 0
	indices = []
	for s in sizes:
		indices.append(s+tot)
		tot += s

	h,w = x.size()
	outputs = []
	cur = 0
	for ind in indices:
		assert ind <= w
		cur_slice = x[:, cur:ind]
		outputs.append(cur_slice)

		cur = ind
	return outputs


class Access(nn.Module):
	def __init__(self, mem_size: int, num_read_heads: int, controller_output_size: int):
		super(Access, self).__init__()

		self.mem_size = mem_size
		self.num_read_heads = num_read_heads
		self.controller_output_size = controller_output_size

		self.read_heads = [ReadHead() for _ in range(self.num_read_heads)]
		self.write_head = WriteHead()

		self.interface_indices = [self.mem_size] * self.num_read_heads + [1] * self.num_read_heads \
		                       + [self.mem_size] + [1]   \
		                       + [self.mem_size] * 2     \
		                       + [1]*self.num_read_heads \
		                       + [1] + [1] + [3] * self.num_read_heads # 3 read modes

		interface_size = sum(self.interface_indices)
		self.interface_linear = nn.Linear(self.controller_output_size, interface_size)

		nn.init.orthogonal_(self.interface_linear.weight)

		self.interface_linear.bias.data.fill_(0.0)


	def split(self, i_vec):
		i_vec_split = index_slice(i_vec, self.interface_indices)
		cur = 0
		r_keys = []
		r_betas = []
		# w_key = []
		# w_beta = []
		# e_vector = []
		# w_vector = []
		fgates  = []
		# a_gate = []
		# w_gate = []
		r_modes = []

		for i in range(self.num_read_heads):
			r_keys.append(i_vec_split[cur])
			cur += 1

		for i in range(self.num_read_heads):
			r_betas.append(i_vec_split[cur])
			cur += 1

		w_key = i_vec_split[cur]
		cur += 1

		w_beta = i_vec_split[cur]
		cur += 1

		e_vector = i_vec_split[cur]
		cur += 1

		w_vector = i_vec_split[cur]
		cur += 1

		for i in range(self.num_read_heads):
			fgates.append(i_vec_split[cur])
			cur += 1       

		a_gate = i_vec_split[cur]
		cur += 1

		w_gate = i_vec_split[cur]
		cur += 1       

		for i in range(self.num_read_heads):
			r_modes.append(i_vec_split[cur])
			cur += 1        

		return r_keys, r_betas, w_key, w_beta, e_vector, w_vector, fgates, a_gate, w_gate, r_modes
 

	@staticmethod
	def get_allocations(r_weights, w_weights, usage, f_gates):
		retention = torch.prod(1 - tfunc.sigmoid(torch.cat(f_gates, 1)).unsqueeze(2)*r_weights, 1)
		usage = (usage + w_weights - usage * w_weights) * retention

		sorted_usage, inds  = usage.sort(1)
		usage_cumprod = torch.cumprod(sorted_usage, 1)

		before_scat = (1-sorted_usage)*usage_cumprod
		allocations = torch.zeros_like(usage).scatter_(1, inds, before_scat)        

		return allocations, usage


	def forward(self, x, memory, a_state, link_matrix):
		r_weights, w_weights, usage, precedence = a_state
		interface_vector = self.interface_linear(x)

		split = self.split(interface_vector)
		r_keys, r_betas, w_key, w_beta, e_vector, w_vector, f_gates, a_gate, w_gate, r_modes = split

		allocations, usage = self.get_allocations(r_weights, w_weights, usage, f_gates)

		# write to the momory
		w_weights, memory, link_matrix, precedence = self.write_head(w_key, w_beta, e_vector, w_vector, a_gate, w_gate, allocations, memory, link_matrix, precedence)    

		# read the memory
		reads = []
		r_weights_out = []

		for i in range(self.num_read_heads):
			read, r_weight = self.read_heads[i](r_keys[i], r_betas[i], r_modes[i], r_weights[:, i], memory, link_matrix)
			reads.append(read.squeeze(1))
			r_weights_out.append(r_weight)

		return torch.cat(reads, 1), memory, (torch.stack(r_weights_out, 1), w_weights, usage, precedence), link_matrix