import torch
from torch.optim import AdamW
import numpy as np
from model import GPT, GBTConfig

batch_size = 12
block_size = 1024
max_iters = 5000
eval_interval = 500
eval_iters = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

training_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('val.bin', dtype=np.uint16, mode='r')


def get_batch(split):
	data = training_data if split == 'training' else val_data
	batch_idxs = torch.randint(len(data)-block_size, (batch_size,))
	x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in batch_idxs])
	y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in batch_idxs])
	return x.to(device), y.to(device)

config = GBTConfig()
model = GPT(config).to(device)
model = torch.compile(model)


optimizer = AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):

	if step % eval_interval == 0:
		model.eval()
		with torch.no_grad():
			losses = torch.zeros(eval_iters)
			for i in range(eval_iters):
				x_v, y_v = get_batch('val')
				_, loss = model(x_v, y_v)
				losses[i] = loss.item()
			print(f'Step {step}: val loss {losses.mean():.4f}')	
			model.train()
		
		torch.save({
			'model': model.state_dict(),
			'optimizer': optimizer.state_dict(),
			'step': step,
			'config': config,
		}, 'ckpt.pt')

	x_t, y_t = get_batch('training')
	with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
		logits, loss = model(x_t, y_t)

	if step % 100 == 0:
		print(f'Step {step}: train loss {loss.item():.4f}')

	loss.backward()
	torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
	optimizer.step()
	optimizer.zero_grad(set_to_none=True)