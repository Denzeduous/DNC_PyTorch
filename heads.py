import torch
from torch import nn
import torch.nn.functional as tfunc


def one_plus(x):
	return 1 + (1+x.exp()).log()


class HeadFuncs:
	@staticmethod
	def query(key, beta, memory):
		if len(key.size()) < len(memory.size()):
			h, w = key.size()
			key = key.view(h,1,w)
			
		beta = one_plus(beta)

		# normalize the key and memory
		m = memory / (memory.norm(2, dim=2, keepdim=True) + 1e-8 )
		k = key / (key.norm(2, dim=1, keepdim=True) + 1e-8)
		
		weights = (m * k).sum(-1)
		
		return tfunc.softmax(weights * beta, dim=1)


class ReadHead(nn.Module):
	def __init__(self):
		super(ReadHead, self).__init__()


	def forward(self, r_key, r_beta, r_mode, r_weights, memory, link_matrix):
		r_mode = tfunc.softmax(r_mode, 1)
		
		c_weights = HeadFuncs.query(r_key, r_beta, memory)
		f_weights = torch.bmm(link_matrix, r_weights.unsqueeze(2)).squeeze(2)
		b_weights = torch.bmm(link_matrix.transpose(1,2), r_weights.unsqueeze(2)).squeeze(2)
		
		# slice to retrain original dims
		weights = r_mode[:,0:1]*b_weights + r_mode[:,1:2]*c_weights + r_mode[:,2:3] * f_weights
		
		return torch.bmm(weights.unsqueeze(1), memory), weights


class WriteHead(nn.Module):
	def __init__(self):
		super(WriteHead, self).__init__()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	@staticmethod
	def update_link(link, weights, precedence):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		b, h, w = link.size()

		assert h == w
		assert weights.size() == precedence.size()

		w_i = weights.unsqueeze(2).repeat(1,1,w)
		w_j = weights.unsqueeze(1).repeat(1,h,1)
		p_j = precedence.unsqueeze(1).repeat(1,h,1)
		
		link = (1 - w_i - w_j) * link + w_i * p_j
		
		mask = 1 -torch.eye(h).unsqueeze(0).repeat(b,1,1).to(device)
		link = link*mask        
		
		return link

	
	def forward(self, w_key, w_beta, e_vector, w_vector, a_gate, w_gate, allocations, memory, link_matrix, precedence):
		c_weights = HeadFuncs.query(w_key, w_beta, memory)
		
		b, h, w = memory.size()

		assert (c_weights.size() == (b,h))
		assert (w_vector.size() == (b,w))
		assert (w_key.size() == (b,w))
		
		assert (c_weights.size() == allocations.size() == precedence.size())
		assert (w_key.size() == e_vector.size() == w_vector.size())
		
		w_gate = tfunc.sigmoid(w_gate)
		a_gate = tfunc.sigmoid(a_gate)
		e_vector = tfunc.sigmoid(e_vector)
		
		w_weights = w_gate*(a_gate*allocations + (1-a_gate)*c_weights)
		link_matrix = self.update_link(link_matrix, w_weights, precedence)
		
		memory = memory*(torch.ones_like(memory) \
		       - torch.bmm(w_weights.unsqueeze(2), e_vector.unsqueeze(1))) \
		       + torch.bmm(w_weights.unsqueeze(2), w_vector.unsqueeze(1))
		
		precedence = (1 - w_weights.sum(1, keepdim=True)) * precedence + w_weights
		
		return w_weights, memory, link_matrix, precedence