import numpy as np
import torch



data = [i for i in range(10)]
asc = torch.tile(torch.Tensor(np.arange(10)), (8, 1))
index = np.arange(10, dtype=np.int32)

for step in range(5):
	cont = list(zip(data,index))
	np.random.shuffle(cont)
	data, index = list(zip(*cont))
	index = list(index)
	data = list(data)

	asc = asc[:,index]
	print(data)
	print(asc)
