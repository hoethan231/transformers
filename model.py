import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GBTConfig:
	block_size:	int = 1024
	vocab_size:	int = 50304
	n_layers:	int = 12
	n_heads:	int = 12
	n_embd:		int = 768
	dropout:	float = 0.0
	bias:		bool = True

def LayerNorm(nn.Module):
	
	def __init__(self, ndims, bias):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(ndims))
		self.bias = nn.Parameter(torch.zeros(ndims)) if bias else None

	def forward(self, input)
		return F.layer_norm(input, self.weight,shape, self.weight, self.bias, 1e-5)


def MLP(nn.Module):

	def __init__(selif, config):
		super().__init__()
		self.sequence = nn.Sequence([
			nn.Linear(config.n_embd, 4*config.n_embd, config.bias),
			nn.GELU(),
			nn.Linear(4*config.n_embd, config-n_embd, config.bias)
			nn.Dropout(config.dropout)
			])

	def forward(self, x):
		x = self.sequence(x)
		return x








