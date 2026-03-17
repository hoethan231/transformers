# miniGPT
 
A from-scratch implementation of GPT-2 in PyTorch, built as a learning exercise following Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).
 
## What this is
 
I wanted to understand how LLMs work at the code level after learning the conceptual. So I read through nanoGPT and implemented it myself, writing each component from scratch, but the model itself is the bare minimum.
 
The architecture is GPT-2 small (124M parameters): token and positional embeddings, causal self-attention with multiple heads, feedforward MLP blocks, residual connections, and layer normalization throughout.
 
## Setup
 
```bash
uv init
uv python pin 3.11
uv add torch --index-url https://download.pytorch.org/whl/cu124
uv add tiktoken numpy
```
 
## Training
 
Prepare the data first:
 
```bash
python prepare.py
```
 
Then train:
 
```bash
python train.py
```
 
Trained on an NVIDIA A40 (48GB) via HPC. Checkpoints are saved to `ckpt.pt` every 500 steps. An example SLURM script is included as well.
 
## References
 
- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
