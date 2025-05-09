
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib import cm
import datetime
import time 

from scipy.stats import norm

from scipy.sparse import lil_array
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm
from numpy.random import rand

n = 30000

x = np.random.rand(n)
M = np.random.rand(n, n)

t1 = time.time()
M1 = M @ x
t2 = time.time()
print("Time taken for dense matrix multiplication: ", t2 - t1)



vec = np.arange(n)
m = 4

A = np.zeros((n, n))

for i in range(n):
    cols = np.random.choice(n, m, replace=False)
    A[i, cols] = np.random.normal(size=m)

# Make the matrix sparse
A_spc = lil_array(A)
A_spc = A_spc.tocsr()


t1 = time.time()
M1 = A @ vec
t2 = time.time()

print("Time taken for dense matrix multiplication: ", t2 - t1)

t1 = time.time()
M2 = A_spc @ vec
t2 = time.time()
print("Time taken for sparse matrix multiplication: ", t2 - t1)