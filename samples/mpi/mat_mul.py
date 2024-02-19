import numpy as np
import dace
from dace.sdfg import utils
import dace.dtypes as dtypes
from mpi4py import MPI
import time


def matrix_mul(comm_world, dim_1, dim_2):
  # comm init
  comm_rank = comm_world.Get_rank()
  comm_size = comm_world.Get_size()

  a_mat = np.full((dim_1, dim_2), 1 + comm_rank, dtype=np.float32)
  b_mat = np.full((dim_2, dim_1), 1 + comm_rank, dtype=np.float32)
  c_mat = np.zeros((dim_1, dim_1), dtype=np.float32)

  @dace.program
  def dist_mat_mult(a_mat: dace.float32[a_mat.shape[0], a_mat.shape[1]],
                    b_mat: dace.float32[b_mat.shape[0], b_mat.shape[1]],
                    c_mat: dace.float32[a_mat.shape[0], b_mat.shape[1]],
                    comm_rank: dace.int32,
                    comm_size: dace.int32):
    grid_dim = int(np.floor(np.sqrt(comm_size)))
    grid_i = comm_rank // grid_dim
    grid_j = comm_rank % grid_dim

    local_i_dim = a_mat.shape[0]
    local_j_dim = b_mat.shape[1]
    local_k_dim = a_mat.shape[1]

    whole_i_dim = grid_dim * a_mat.shape[0]
    whole_j_dim = grid_dim * b_mat.shape[1]
    whole_k_dim = grid_dim * a_mat.shape[1]

    # local buffers for remote fetching
    foreign_a_mat = np.zeros(a_mat.shape, dtype=np.float32)
    foreign_b_mat = np.zeros(b_mat.shape, dtype=np.float32)

    # RMA windows
    a_win = MPI.Win.Create(a_mat, comm=comm_world)
    b_win = MPI.Win.Create(b_mat, comm=comm_world)
    for i in range(whole_i_dim // local_i_dim):
      for j in range(whole_j_dim // local_j_dim):
        for k in range(whole_k_dim // local_k_dim):
          # check if this process owns this chunk of data
          if i == grid_i and j == grid_j:
            target_rank_a = i * grid_dim + k
            target_rank_b = k * grid_dim + j
            a_win.Lock(target_rank_a)
            a_win.Get(foreign_a_mat, target_rank=target_rank_a)
            a_win.Flush(target_rank_a)
            a_win.Unlock(target_rank_a)

            b_win.Lock(target_rank_b)
            b_win.Get(foreign_b_mat, target_rank=target_rank_b)
            b_win.Flush(target_rank_b)
            b_win.Unlock(target_rank_b)

            c_mat += foreign_a_mat @ foreign_b_mat

    # as MPI barrier
    # to ensure every process completed the calculation
    a_win.Fence(0)
    a_win.Fence(0)

  sdfg = None
  if comm_rank == 0:
    # ValueError: Node type "Win_lock" not supported for promotion
    sdfg = dist_mat_mult.to_sdfg(simplify=False)
  func = utils.distributed_compile(sdfg, comm_world)

  start = time.time()

  func(a_mat=a_mat, b_mat=b_mat, c_mat=c_mat, comm_rank=comm_rank, comm_size=comm_size)

  time_con = time.time() - start

  return c_mat, time_con


def weak_scaling(comm_world, comm_rank, comm_size):
  grid_dim = int(np.floor(np.sqrt(comm_size)))
  grid_i = comm_rank // grid_dim
  grid_j = comm_rank % grid_dim

  dim_1 = 1024
  dim_2 = 1024

  c_mat, time_con = matrix_mul(comm_world, dim_1, dim_2)
  # print(comm_rank, c_mat)
  # print(comm_rank, "matrix_mul time:", time_con)

  whole_a = np.ones((dim_1 * grid_dim, dim_2 * grid_dim), dtype=np.float32)
  for i in range(grid_dim):
    for j in range(grid_dim):
      whole_a[i * dim_1:(i+1) * dim_1, j * dim_2:(j+1) * dim_2] += i * grid_dim + j

  whole_b = np.ones((dim_2 * grid_dim, dim_1 * grid_dim), dtype=np.float32)
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


def strong_scaling(comm_world, comm_rank, comm_size):
  grid_dim = int(np.floor(np.sqrt(comm_size)))
  grid_i = comm_rank // grid_dim
  grid_j = comm_rank % grid_dim

  total_dim = 8192
  dim_1 = total_dim
  dim_2 = total_dim
  if total_dim % comm_size > 0:
    dim_1 += comm_size - total_dim % comm_size
    dim_2 += comm_size - total_dim % comm_size

  local_dim_1 = dim_1 // grid_dim
  local_dim_2 = dim_2 // grid_dim

  a = np.ones((local_dim_1, local_dim_2), dtype=np.float32)
  b = np.ones((local_dim_2, local_dim_1), dtype=np.float32)

  c_mat, time_con = matrix_mul(comm_world, local_dim_1, local_dim_2)
  # print(comm_rank, c_mat)
  # print(comm_rank, "matrix_mul time:", time_con)

  # validation, since it will compute the whole matrix in the edge
  # whole_a = np.ones((local_dim_1 * grid_dim, local_dim_2 * grid_dim), dtype=np.float32)
  # for i in range(grid_dim):
  #   for j in range(grid_dim):
  #     whole_a[i * local_dim_1:(i+1) * local_dim_1, j * local_dim_2:(j+1) * local_dim_2] += i * grid_dim + j

  # whole_b = np.ones((local_dim_2 * grid_dim, local_dim_1 * grid_dim), dtype=np.float32)
  # for i in range(grid_dim):
  #   for j in range(grid_dim):
  #     whole_b[i * local_dim_2:(i+1) * local_dim_2, j * local_dim_1:(j+1) * local_dim_1] += i * grid_dim + j

  # start = time.time()
  # c_np = np.matmul(whole_a, whole_b)
  # time_con = time.time() - start
  # # print("Result correctness:", np.allclose(c_mat, c_np[grid_i * local_dim_1:(grid_i+1) * local_dim_1, grid_j* local_dim_2:(grid_j+1) * local_dim_2]))
  # assert(np.allclose(c_mat, c_np[grid_i * local_dim_1:(grid_i+1) * local_dim_1, grid_j* local_dim_2:(grid_j+1) * local_dim_2]))

if __name__ == "__main__":
  comm_world = MPI.COMM_WORLD
  comm_rank = comm_world.Get_rank()
  comm_size = comm_world.Get_size()

  grid_dim = int(np.floor(np.sqrt(comm_size)))

  if comm_size != grid_dim ** 2:
      raise ValueError("Please run this test with a square number of processes.")

  # weak_scaling(comm_world, comm_rank, comm_size)
  strong_scaling(comm_world, comm_rank, comm_size)
