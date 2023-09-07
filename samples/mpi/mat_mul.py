import numpy as np
import dace
from dace.sdfg import utils
import dace.dtypes as dtypes
import time

dim_1 = 1024
dim_2 = 1024

tile = 128

a = np.arange(dim_1 * dim_2).reshape(dim_1, dim_2)
b = np.arange(dim_1 * dim_2).reshape(dim_2, dim_1)

def matrix_mul(a, b):
  a_mat = np.array(a, dtype=np.int64)
  b_mat = np.array(b, dtype=np.int64)
  c_mat = np.zeros((a_mat.shape[0], b_mat.shape[1]), dtype=np.int64)

  if a_mat.shape[1] != b_mat.shape[0]:
    raise ValueError("A, B matrix dimension mismatched!")

  # more or less like C stationary
  # for i in range(a_mat.shape[0]):
  #   for j in range(b_mat.shape[1]):
  #     for k in range(a_mat.shape[1]):
  #       c_mat[i][j] += a_mat[i][k] * b_mat[k][j]
  
  @dace.program
  def mpi4py_passive_rma_put(a_mat: dace.int64[dim_1,dim_2], b_mat: dace.int64[dim_1,dim_2], c_mat: dace.int64[dim_1,dim_2], tile: dace.int64):
    for i_tile in range(a_mat.shape[0] // tile):
      for j_tile in range(b_mat.shape[1] // tile):
        for k_tile in range(a_mat.shape[1] // tile):
          for i in range(i_tile * tile, min((i_tile + 1) * tile, a_mat.shape[0])):
            for j in range(j_tile * tile, min((j_tile + 1) * tile, b_mat.shape[1])):
              for k in range(k_tile * tile, min((k_tile + 1) * tile, a_mat.shape[1])):
                c_mat[i][j] += a_mat[i][k] * b_mat[k][j]

  sdfg = None
  sdfg = mpi4py_passive_rma_put.to_sdfg()
  sdfg.openmp_sections = False
  func = sdfg.compile()
  
  start = time.time()
  func(a_mat, b_mat, c_mat, tile)
  time_con = time.time() - start

  return c_mat, time_con

c_mat, time_con = matrix_mul(a,b)
print(c_mat, time_con)

start = time.time()
c_np = np.matmul(a,b)
time_con = time.time() - start
print(c_np, time_con)

print("Result correctness:", np.allclose(c_mat, c_np))
