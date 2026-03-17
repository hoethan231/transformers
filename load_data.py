import os
import requests
import tiktoken
import numpy as np

data_path = os.path.join(os.path.dirname(__file__), "input.txt")

if not os.path.exists(data_path):
	data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

	with open(data_path, "w", encoding="utf-8") as f:
		f.write(requests.get(data_url).text)

with open(data_path, "r", encoding="utf-8") as f:
	data = f.read()
n = len(data)
training = data[:int(n*0.9)]
val	 = data[int(n*0.9):]

enc = tiktoken.get_encoding('gpt2')
train_enc = enc.encode(training)
val_enc = enc.encode(val)

train_out = np.array(train_enc, dtype=np.uint16)
val_out = np.array(val_enc, dtype=np.uint16)
train_out.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_out.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))
	
print("Done!")

