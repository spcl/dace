import numpy as np
import dace
from dace.sdfg import utils
import dace.dtypes as dtypes
from mpi4py import MPI
import time

dim_1 = 128
dim_2 = 128

a = np.arange(dim_1 * dim_2).reshape(dim_1, dim_2)
b = np.arange(dim_1 * dim_2).reshape(dim_2, dim_1)

# to check if this process owns this chunk of data
# compare given i and j with grid_i and grid_j
@dace.program
def owner(i, j, grid_i, grid_j):
  if i == grid_i and j == grid_j:
    return True
  else:
    return False

# get matrix form remote rank
@dace.program
def get_mat(win: dace.RMA_window, buffer: dace.int32[dim_1,dim_2], dim_0: dace.int32, dim_1: dace.int32, grid_dim: dace.int32):
  rank = dim_0 * grid_dim + dim_1
  win.Lock(rank)
  win.Get(buffer, target_rank=rank)
  win.Flush(rank)
  win.Unlock(rank)

def matrix_mul(a, b):
  # check if matrix multiplication is valid
  if a.shape[1] != b.shape[0]:
    raise ValueError("A, B matrix dimension mismatched!")
  
  # comm init
  comm_world = MPI.COMM_WORLD
  comm_rank = comm_world.Get_rank()
  comm_size = comm_world.Get_size()

  grid_dim = 2
  grid_i = comm_rank // grid_dim
  grid_j = comm_rank % grid_dim

  if comm_size != 2:
      raise ValueError("Please run this test with two processes.")

  a_mat = np.array(a + comm_rank, dtype=np.int64)
  b_mat = np.array(b + comm_rank, dtype=np.int64)
  foreign_a_mat = np.zeros(a.shape, dtype=np.int64)
  foreign_b_mat = np.zeros(b.shape, dtype=np.int64)
  c_mat = np.zeros((a_mat.shape[0], b_mat.shape[1]), dtype=np.int64)


  # more or less like C stationary
  # for i in range(a_mat.shape[0]):
  #   for j in range(b_mat.shape[1]):
  #     for k in range(a_mat.shape[1]):
  #       c_mat[i][j] += a_mat[i][k] * b_mat[k][j]

  
  @dace.program
  def mpi4py_send_recv(comm_rank: dace.int32, a_mat: dace.int32[dim_1,dim_2], foreign_a_mat: dace.int32[dim_1,dim_2], grid_dim: dace.int32):
    a_win = MPI.Win.Create(a_mat, comm=comm_world)
    if comm_rank == 0:
      get_mat(a_win, foreign_a_mat, 0, 1, grid_dim)
    else:
      get_mat(a_win, foreign_a_mat, 0, 0, grid_dim)
    return foreign_a_mat

  sdfg = None
  if comm_rank == 0:
      sdfg = mpi4py_send_recv.to_sdfg(simplify=True)
  func = utils.distributed_compile(sdfg, comm_world)


  start = time.time()

  foreign_a_mat = func(comm_rank=comm_rank, a_mat=a_mat, foreign_a_mat=foreign_a_mat, grid_dim=grid_dim)
  if comm_rank == 0:
    if(np.allclose(a_mat+1, foreign_a_mat)):
      print("Good")
  else:
    if(np.allclose(a_mat-1, foreign_a_mat)):
      print("Good")

  time_con = time.time() - start


  # to ensure every process completed the calculation
  comm_world.Barrier()

matrix_mul(a,b)

  # more or less like C stationary
  # for i in range(a_mat.shape[0]):
  #   for j in range(b_mat.shape[1]):
  #     for k in range(a_mat.shape[1]):
  #       c_mat[i][j] += a_mat[i][k] * b_mat[k][j]
  
  # @dace.program
  # def mpi4py_passive_rma_put(a_mat: dace.int32[dim_1,dim_2], b_mat: dace.int32[dim_1,dim_2], c_mat: dace.int32[dim_1,dim_2], tile: dace.int32):
  #   for i_tile in range(a_mat.shape[0] // tile):
  #     for j_tile in range(b_mat.shape[1] // tile):
  #       for k_tile in range(a_mat.shape[1] // tile):
  #         for i in range(i_tile * tile, min((i_tile + 1) * tile, a_mat.shape[0])):
  #           for j in range(j_tile * tile, min((j_tile + 1) * tile, b_mat.shape[1])):
  #             for k in range(k_tile * tile, min((k_tile + 1) * tile, a_mat.shape[1])):
  #               c_mat[i][j] += a_mat[i][k] * b_mat[k][j]

  # sdfg = None
  # sdfg = mpi4py_passive_rma_put.to_sdfg()
  # sdfg.openmp_sections = False
  # func = sdfg.compile()
  
  # func(a_mat, b_mat, c_mat, tile)