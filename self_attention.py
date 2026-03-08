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

	def forward(self, embeddings):
		q = self.W_q(embeddings)
		k = self.W_k(embeddings)
		v = self.W_v(embeddings)

		QK = torch.matmul(q, k.transpose(self.row, self.col))
		scaled_QK = QK / torch.tensor(k.size(self.col)**0.5)
		attention_per = F.softmax(QK_dim, self-col)
		attention_val = torch.matmul(attention_per, v)
		
		return attention_val

