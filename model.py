

@dataclass
class GBTConfig:
	block_size:	int = 1024
	vocab_size:	int = 50304
	n_layers:	int = 12
	n_heads:	int = 12
	n_embd:		int = 768
	dropout:	float = 0.0
	bias:		bool = True


