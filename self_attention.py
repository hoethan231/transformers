import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):

	def __init__(self, d, row=0, col=1):
		self.d = d
		self.row = row
		self.col = col

		self.W_q = nn.Linear(d, d)
		self.W_k = nn.Linear(d, d)
		self.W_v = nn.Linear(d, d)

	def forward(self, embeddings):
		QK = torch.matmul(q, torch.transpose(k))
		QK_dim = QK / self.W_k.shape[1]
		attention_per = F.softmax(QK_dim)
		attention_val = toch.matmul(attention_per, torch.transpose(self.W_v))
		
		return attention_val

