import numpy as np
import dace
from dace.sdfg import utils
import dace.dtypes as dtypes
from mpi4py import MPI
import time


# to check if this process owns this chunk of data
# compare given i and j with grid_i and grid_j
def owner(i, j, grid_i, grid_j):
  if i == grid_i and j == grid_j:
    return True
  else:
    return False


# get matrix form remote rank
def get_mat(win, buffer, dim_0, dim_1, grid_dim):
  rank = dim_0 * grid_dim + dim_1
  win.Lock(rank)
  win.Get(buffer, target_rank=rank)
  win.Flush(rank)
  win.Unlock(rank)


def matrix_mul(comm_world, a, b):
  # check if matrix multiplication is valid
  if a.shape[1] != b.shape[0]:
    raise ValueError("A, B matrix dimension mismatched!")

  # comm init
  comm_rank = comm_world.Get_rank()
  comm_size = comm_world.Get_size()

  grid_dim = int(np.floor(np.sqrt(comm_size)))
  grid_i = comm_rank // grid_dim
  grid_j = comm_rank % grid_dim

  local_i_dim = a.shape[0]
  local_j_dim = b.shape[1]
  local_k_dim = a.shape[1]

  whole_i_dim = grid_dim * a.shape[0]
  whole_j_dim = grid_dim * b.shape[1]
  whole_k_dim = grid_dim * a.shape[1]

  a_mat = np.array(a + comm_rank, dtype=np.int32)
  b_mat = np.array(b + comm_rank, dtype=np.int32)
  c_mat = np.zeros((a_mat.shape[0], b_mat.shape[1]), dtype=np.int32)

  # local buffers for remote fetching
  foreign_a_mat = np.zeros(a.shape, dtype=np.int32)
  foreign_b_mat = np.zeros(b.shape, dtype=np.int32)

  # RMA windows
  a_win = MPI.Win.Create(a_mat, comm=comm_world)
  b_win = MPI.Win.Create(b_mat, comm=comm_world)

  start = time.time()
  for i in range(whole_i_dim // local_i_dim):
    for j in range(whole_j_dim // local_j_dim):
      for k in range(whole_k_dim // local_k_dim):
        if owner(i, j, grid_i, grid_j):
          get_mat(a_win, foreign_a_mat, i, k, grid_dim)
          get_mat(b_win, foreign_b_mat, k, j, grid_dim)

          c_mat += np.matmul(foreign_a_mat, foreign_b_mat)
  time_con = time.time() - start

  # to ensure every process completed the calculation
  comm_world.Barrier()

  return c_mat, time_con


if __name__ == "__main__":
  comm_world = MPI.COMM_WORLD
  comm_rank = comm_world.Get_rank()
  comm_size = comm_world.Get_size()

  grid_dim = int(np.floor(np.sqrt(comm_size)))
  grid_i = comm_rank // grid_dim
  grid_j = comm_rank % grid_dim

  dim_1 = 256
  dim_2 = 256

  a = np.ones((dim_1, dim_2), dtype=np.int32)
  b = np.ones((dim_2, dim_1), dtype=np.int32)

  c_mat, time_con = matrix_mul(comm_world, a, b)
  # print(comm_rank, c_mat)
  # print(comm_rank, "matrix_mul time:", time_con)

  whole_a = np.ones((dim_1 * grid_dim, dim_2 * grid_dim), dtype=np.int32)
  for i in range(grid_dim):
    for j in range(grid_dim):
      whole_a[i * dim_1:(i+1) * dim_1, j * dim_2:(j+1) * dim_2] += i * grid_dim + j

  whole_b = np.ones((dim_2 * grid_dim, dim_1 * grid_dim), dtype=np.int32)
  for i in range(grid_dim):
    for j in range(grid_dim):
      whole_b[i * dim_2:(i+1) * dim_2, j * dim_1:(j+1) * dim_1] += i * grid_dim + j

  start = time.time()
  c_np = np.matmul(whole_a, whole_b)
  time_con = time.time() - start

  # print(comm_rank, c_np[grid_i * dim_1:(grid_i+1) * dim_1, grid_j* dim_2:(grid_j+1) * dim_2])
  # print(comm_rank, "np.matmul time:", time_con)

  # print("Result correctness:", np.allclose(c_mat, c_np[grid_i * dim_1:(grid_i+1) * dim_1, grid_j* dim_2:(grid_j+1) * dim_2]))
  assert(np.allclose(c_mat, c_np[grid_i * dim_1:(grid_i+1) * dim_1, grid_j* dim_2:(grid_j+1) * dim_2]))
