# just a practice implementation for attention
# not used in miniGPT

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):

	def __init__(self, d, row=0, col=1):
		self.row = row
		self.col = col
		
		super().__init__()	

		self.W_q = nn.Linear(in_features=d, out_features=d, bias=False)
		self.W_k = nn.Linear(in_features=d, out_features=d, bias=False)
		self.W_v = nn.Linear(in_features=d, out_features=d, bias=False)

	def forward(self, q_embed, k_embed, v_embed, mask=None):
		q = self.W_q(q_embed)
		k = self.W_k(k_embed)
		v = self.W_v(v_embed)

		QK = torch.matmul(q, k.transpose(self.row, self.col))
		scaled_QK = QK / torch.tensor(k.size(self.col)**0.5)

		if mask not None:
			scaled_QK = scaled_QK.masked_fill(mask=mask, value=-1e9)

		attention_per = F.softmax(QK_dim, self-col)
		attention_val = torch.matmul(attention_per, v)
		
		return attention_val


class MultiHeaddedAttention(nn.Module):

	def __init__(self, d=2, row=0, col=1, heads=1):
		self.row = row
		self.col = col

		super().__init__()

		self.heads = nn.ModuleList(
			SelfAttention(d, row, col) for _ in range(heads)
		)


	def foward(self, q_embed, k_embed, v_embed):
		return torch.cat(
			[head(q_embed, k_embed, v_embed) for head in self.heads]
				, dim=self.col)

