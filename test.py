import numpy as np


a = np.matrix([1,1,1,1])

b = np.matrix([1,1,1,1])

c = float(np.matmul(a,b.transpose()))
print c