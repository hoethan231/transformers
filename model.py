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

class LayerNorm(nn.Module):
	
	def __init__(self, ndims, bias):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(ndims))
		self.bias = nn.Parameter(torch.zeros(ndims)) if bias else None

	def forward(self, input)
		return F.layer_norm(input, self.weight,shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):

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


class CausalSelfAttention(nn.Module):

	def __init__(self, config):
		super().__init__()
		
		self.n_embd		= config.n_embd
		self.n_heads		= config.n_heads
		self.c_attn		= nn.Linear(config.n_embd, 3*config.n_embd, config.bias)
		self.c_proj		= nn.Linear(config.n_emdb, config.n_embd, config.bias)
		self.attn_dropout	= nn.Dropout(config.dropout)
		self.resid_dropout	= nn.Dropout(config.dropout)
		self.flash		= hasattr(torch.nn.functional, 'scaled_dot_product_attention')
		if not self.flash:
			self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

	def forward(self, x):
		B, T, C = x.split()

		q, k, v = self.c_attn(x).split(config.n_embd, dim=2)
		q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
		k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
		v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

		if self.flash:
			y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
		else:
			attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
			attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
			attn = F.softmax(attn, dim=-1)
			attn = self.attn_dropout(attn)
			y = attn @ v
		
		y = y.transpose(1, 2).contiguous().view(B, T, C)
		y = self.resid_dropout(self.c_proj(y))
		return y
			

class Block(nn.module):

	def __init__(self, config):
		self.norm1 = LayerNorm(config.n_embd, config.bias)
		self.attn = CausalSelfAttention(config)
		self.norm2 = LayerNorm(config.n_embd, config.bias)
		self.mlp = MLP(config)

	def forward(self, x):
		x = self.attn(self.norm1(x))
		x = self.mlp(self.norm2(x))
		return x








