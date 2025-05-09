import numpy as np
import random
import os 
import matplotlib.pyplot as plt
import matplotlib as mpl

### Test parallelization
import multiprocessing 
import time

from utilis.Grid import Grid


# Test sparce matrix multiplication
from scipy.sparse import csr_matrix
from scipy import sparse

from scipy.sparse import lil_array
from scipy.sparse.linalg import spsolve



function_name = "test_function"
class_name = "test_class"
# Get the longest class name and function name
longest_class_name = max(len(class_name), len("class_name"))
longest_function_name = max(len(function_name), len("function_name"))

print(len(class_name), len("class_name"))
print(len(function_name), len("function_name"))

str_d = class_name + " " * (longest_class_name - len(class_name)) + " "  + " " * (longest_function_name - len(function_name)) + function_name



def make_G_matrix(observation_index, n_total_points, make_sparse=True):
    """
    Create the G matrix for the Kalman filter.
    :param observation_index: index of the observation
    :param n_total_points: total number of points in the state
    :param make_sparse: whether to make the matrix sparse or not
    :return: G matrix
    """
    
    # Create a sparse matrix with zeros and a 1 at the observation index

    G = np.zeros((len(observation_index), n_total_points))
    G[np.arange(len(observation_index)), observation_index] = 1

    if make_sparse:
        G = lil_array(G)
        G = G.tocsr()

    return G


def G_at_vector(observation_index, vec):
    """
    G is a indexing matrix with shape (n_obs, n_total_points) 
    along each row is a vector of zeros with a 1 at the observation index
    
    """
    return vec[observation_index]

def G_at_matrix(observation_index, matrix):
    """
    G is a indexing matrix with shape (n_obs, n_total_points) 
    along each row is a vector of zeros with a 1 at the observation index
    
    """
    return matrix[observation_index,:]

def G_at_matrix_at_G_transpose(observation_index, matrix):
    """
    G is a indexing matrix with shape (n_obs, n_total_points) 
    along each row is a vector of zeros with a 1 at the observation index
    
    """
    return matrix[observation_index,:][:,observation_index]


n = 10
vec = np.arange(n)
m = 3
observation_index = np.random.choice(n, m, replace=False)
observation_index = np.sort(observation_index)
print("observation_index", observation_index)


G = make_G_matrix(observation_index, n, make_sparse=False)

M = np.arange(n * n).reshape(n, n)  

print("vec.shape", vec.shape)
print("M.shape", M.shape)
print("G.shape", G.shape)
print("observation_index.shape", observation_index.shape)

print("G", G)
print("M"  , M)
print("G at vector", G_at_vector(observation_index, vec))
print("G @ vector", G.dot(vec.T))

G_at_vec_m1 = G_at_vector(observation_index, vec)
G_at_vec_m2 = G.dot(vec.T)
tot_diff = np.sum(np.abs(G_at_vec_m1 - G_at_vec_m2))
print("G at vector - G @ vector", tot_diff)

G_at_matrix_m1 = G_at_matrix(observation_index, M)
G_at_matrix_m2 = G.dot(M)
print("G_at_matrix_m1.sheape", G_at_matrix_m1.shape)
print("G_at_matrix_m2.shape", G_at_matrix_m2.shape)
print("G_at_matrix_m1", G_at_matrix_m1)
print("G_at_matrix_m2", G_at_matrix_m2)
tot_diff = np.sum(np.abs(G_at_matrix_m1 - G_at_matrix_m2))
print("G at matrix - G @ matrix", tot_diff)

G_at_matrix_at_G_transpose_m1 = G_at_matrix_at_G_transpose(observation_index, M)
G_at_matrix_at_G_transpose_m2 = G.dot(M).dot(G.T)
print("G_at_matrix_at_G_transpose_m1.shape", G_at_matrix_at_G_transpose_m1.shape)
print("G_at_matrix_at_G_transpose_m2.shape", G_at_matrix_at_G_transpose_m2.shape)
print("G_at_matrix_at_G_transpose_m1", G_at_matrix_at_G_transpose_m1)
print("G_at_matrix_at_G_transpose_m2", G_at_matrix_at_G_transpose_m2)
tot_diff = np.sum(np.abs(G_at_matrix_at_G_transpose_m1 - G_at_matrix_at_G_transpose_m2))
print("G at matrix at G transpose - G @ matrix @ G transpose", tot_diff)



# Testing the speed 
for n in [1000,10000, 34000]:
    vec = np.arange(n)
    m = random.randint(20, 50)
    observation_index = np.random.choice(n, m, replace=False)
    observation_index = np.sort(observation_index)
    print("n", n)
    start_time = time.time()
    G = make_G_matrix(observation_index, n, make_sparse=False)
    end_time = time.time()
    dur_1 = end_time - start_time
    print(f"G matrix time {end_time - start_time:.4f} seconds")

    M = np.arange(n * n).reshape(n, n)  

    print("vec.shape", vec.shape)
    print("M.shape", M.shape)
    print("G.shape", G.shape)
    print("G.T.shape", G.T.shape)
    print("observation_index.shape", observation_index.shape)

    start_time = time.time()
    G_at_vec_m1 = G_at_vector(observation_index, vec)
    end_time = time.time()
    dur_2 = end_time - start_time
    print("G_at_vec_m1 time", end_time - start_time)

    start_time = time.time()
    G_at_vec_m2 = G.dot(vec)
    end_time = time.time()
    dur_3 = end_time - start_time
    print("G @ vec time", end_time - start_time)

    

    if np.max(np.abs(G_at_vec_m1 - G_at_vec_m2)) > 1e-10:
        print("Something is wrong with G @ vec")


    start_time = time.time()
    G_at_matrix_m1 = G_at_matrix(observation_index, M)
    end_time = time.time()
    dur_3 = end_time - start_time

    print("G_at_matrix_m1 time", end_time - start_time)
    start_time = time.time()
    G_at_matrix_m2 = G.dot(M)
    end_time = time.time()
    dur_5 = end_time - start_time
    print("G @ M time", end_time - start_time)

    if np.max(np.abs(G_at_matrix_m1 - G_at_matrix_m2)) > 1e-10:
        print("Something is wrong with G @ M")

    start_time = time.time()
    G_at_matrix_at_G_transpose_m1 = G_at_matrix_at_G_transpose(observation_index, M)
    end_time = time.time()
    dur_6 = end_time - start_time
    print("G_at_matrix_at_G_transpose_m1 time", end_time - start_time)

    start_time = time.time()
    G_at_matrix_at_G_transpose_m2 = G.dot(M).dot(G.T)
    end_time = time.time()
    dur_7 = end_time - start_time
    print("G @ M @ G.T time", end_time - start_time, "shape", G_at_matrix_at_G_transpose_m2.shape)


    if np.max(np.abs(G_at_matrix_at_G_transpose_m1 - G_at_matrix_at_G_transpose_m2)) > 1e-10:
        print("Something is wrong with G @ M @ G.T")


    print("time G @ vec / G_at_vec_m1", dur_3 / dur_2)
    print("time G @ M / G_at_matrix_m1", dur_5 / dur_3)
    print("time G @ M @ G.T / G_at_matrix_at_G_transpose_m1", dur_7 / dur_6)
    


