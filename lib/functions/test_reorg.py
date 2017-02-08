from reorg import *
import numpy as np
from chainer import Variable

x_data = np.random.randn(100, 3, 32, 32).astype(np.float32)
x = Variable(x_data)

y = reorg(x)
print(x.shape, y.shape)
