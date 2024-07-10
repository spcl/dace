import dace
from dace.transformation.auto import auto_optimize as aopt
import numpy as np
import cupy as cp
from dace.sdfg import utils as sdutil
from dace.sdfg import nodes
from dace.transformation.dataflow.tiling import MapTiling
from dace.transformation.dataflow.change_thread_block_map import ChangeThreadBlockMap
from dace.transformation.dataflow.block_coarsening import BlockCoarsening
from dace.transformation.dataflow.add_thread_block_map import AddThreadBlockMap
from dace.transformation.dataflow.thread_coarsening import ThreadCoarsening
from dace.transformation.dataflow.explicit_memory_move import ExplicitMemoryMove
from dace import dtypes


N = dace.symbol('N')
M = dace.symbol('M')

def apply_add_thread_block_schedule(sdfg):
  for state in sdfg.states():
      outer = None
      for node in sdutil.dfs_topological_sort(state):
          if isinstance(node, nodes.MapEntry):
              if outer == None and node.map.schedule == dtypes.ScheduleType.GPU_Device:
                  outer = node
          if outer != None:
              AddThreadBlockMap.apply_to(sdfg=sdfg, verify=False, map_entry=outer)
              sdfg.save("added_thread_block_map.sdfg")
              outer = None

def apply_change_thread_block_schedule(sdfg):
    for state in sdfg.states():
        outer = None
        inner = None
        for node in sdutil.dfs_topological_sort(state):
            if isinstance(node, nodes.MapEntry):
                if outer == None and node.map.schedule == dtypes.ScheduleType.GPU_Device:
                    outer = node
                elif inner == None and node.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
                    inner = node
            if outer and inner:
                ChangeThreadBlockMap.apply_to(sdfg=sdfg, verify=False, device_scheduled_map_entry = outer, 
                                                    thread_block_scheduled_map_entry = inner, 
                                                    options={"dim_size_x":32,"dim_size_y":2,"dim_size_z":2})
                sdfg.save("changed_thread_block_map.sdfg")
                outer = None
                inner = None

def apply_thread_coarsening(sdfg):
    for state in sdfg.states():
        outer = None
        inner = None
        for node in sdutil.dfs_topological_sort(state):
            if isinstance(node, nodes.MapEntry):
                if outer == None and node.map.schedule == dtypes.ScheduleType.GPU_Device:
                    outer = node
                elif inner == None and node.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
                    inner = node
            if outer and inner:
                ThreadCoarsening.apply_to(sdfg=sdfg, verify=False, device_map_entry=outer, thread_block_map_entry=inner, options={"tile_size_x":4})
                sdfg.save("thread_coarsened.sdfg")
                outer = None
                inner = None

def apply_block_coarsening(sdfg):
    for state in sdfg.states():
        outer = None
        inner = None
        seq = None
        for node in sdutil.dfs_topological_sort(state):
            if isinstance(node, nodes.MapEntry):
                if outer == None and node.map.schedule == dtypes.ScheduleType.GPU_Device:
                    outer = node
                elif inner == None and node.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
                    inner = node
                elif seq == None and node.map.schedule == dtypes.ScheduleType.Sequential:
                    seq = node
            if outer and inner and seq:
                BlockCoarsening.apply_to(sdfg=sdfg, verify=False, device_map_entry=outer, thread_block_map_entry=inner, sequential_map_entry=seq, options={"block_iter_x":4, "block_iter_y":2, "block_iter_z":2})
                sdfg.save("block_coarsened.sdfg")
                outer = None
                inner = None
                seq = None

def apply_mem_move(sdfg):
    sdfg.save("base.sdfg")
    try:
        for state in sdfg.states():
            dev = None
            outer = None
            inner = None
            for node in sdutil.dfs_topological_sort(state):
                if isinstance(node, nodes.MapEntry):
                    if dev == None and node.map.schedule == dtypes.ScheduleType.GPU_Device:
                        dev = node
                    elif outer == None and node.map.schedule == dtypes.ScheduleType.Sequential:
                        outer = node
                    elif inner == None and node.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
                        inner = node
                if outer and inner and dev:
                    ExplicitMemoryMove.apply_to(sdfg=sdfg, verify=False, device_map_entry=dev, grid_strided_map_entry=outer, thread_block_map_entry=inner, options={"memory_location":dtypes.StorageType.GPU_Shared})
                    sdfg.save("memory_moved.sdfg")
                    outer = None
                    inner = None
                    dev = None
    except Exception as e:
        sdfg.save("failed_transformed.sdfg")
        print(e)
        raise Exception(e)

def _tensor_add_kernel(A, B):
   A += 0.5 * B

def _test_transformations(opt_sdfg, A, B, A2, B2, _N):
  _tensor_add_kernel(A, B)
  apply_add_thread_block_schedule(opt_sdfg)
  opt_sdfg(A=A2, B=B2, N=_N)
  assert(np.allclose(A, A2.get()))
  _tensor_add_kernel(A, B)
  apply_change_thread_block_schedule(opt_sdfg)
  opt_sdfg(A=A2, B=B2, N=_N)
  assert(np.allclose(A, A2.get()))
  _tensor_add_kernel(A, B)
  apply_thread_coarsening(opt_sdfg)
  opt_sdfg(A=A2, B=B2, N=_N)
  assert(np.allclose(A, A2.get()))
  _tensor_add_kernel(A, B)
  apply_block_coarsening(opt_sdfg)
  opt_sdfg(A=A2, B=B2, N=_N)
  assert(np.allclose(A, A2.get()))

def test_tensor_add_1d():
  @dace.program
  def dace_kernel(A: dace.float32[N] @ dtypes.StorageType.GPU_Global,
                  B: dace.float32[N] @ dtypes.StorageType.GPU_Global):
      A[0:N] += 0.5 * B[0:N]
  sdfg = dace_kernel.to_sdfg()
  opt_sdfg = aopt.auto_optimize(sdfg, dace.DeviceType.GPU)
  _N = 4096
  A = np.random.rand(_N).astype(np.float32)
  B = np.random.rand(_N).astype(np.float32)
  A2 = cp.asarray(A, cp.float32)
  B2 = cp.asarray(B, cp.float32)

  _test_transformations(opt_sdfg, A, B, A2, B2, _N)

def test_tensor_add_2d():
  @dace.program
  def dace_kernel(A: dace.float32[N, N] @ dtypes.StorageType.GPU_Global, 
                  B: dace.float32[N, N] @ dtypes.StorageType.GPU_Global):
      A[0:N, 0:N] += 0.5 * B[0:N, 0:N]
  sdfg = dace_kernel.to_sdfg()
  opt_sdfg = aopt.auto_optimize(sdfg, dace.DeviceType.GPU)
  _N = 128
  A = np.random.rand(_N, _N).astype(np.float32)
  B = np.random.rand(_N, _N).astype(np.float32)
  A2 = cp.asarray(A, cp.float32)
  B2 = cp.asarray(B, cp.float32)

  _test_transformations(opt_sdfg, A, B, A2, B2, _N)


def test_tensor_add_3d():
  @dace.program
  def dace_kernel(A: dace.float32[N, N, N] @ dtypes.StorageType.GPU_Global, 
                  B: dace.float32[N, N, N] @ dtypes.StorageType.GPU_Global):
      A[0:N, 0:N, 0:N] += 0.5 * B[0:N, 0:N, 0:N]
  sdfg = dace_kernel.to_sdfg()
  opt_sdfg = aopt.auto_optimize(sdfg, dace.DeviceType.GPU)
  _N = 24
  A = np.random.rand(_N, _N, _N).astype(np.float32)
  B = np.random.rand(_N, _N, _N).astype(np.float32)
  A2 = cp.asarray(A, cp.float32)
  B2 = cp.asarray(B, cp.float32)

  _test_transformations(opt_sdfg, A, B, A2, B2, _N)



def test_tensor_add_4d():
  @dace.program
  def dace_kernel(A: dace.float32[N, N, N, N] @ dtypes.StorageType.GPU_Global, 
                  B: dace.float32[N, N, N, N] @ dtypes.StorageType.GPU_Global):
      A[0:N, 0:N, 0:N, 0:N] += 0.5 * B[0:N, 0:N, 0:N, 0:N]
  sdfg = dace_kernel.to_sdfg()
  opt_sdfg = aopt.auto_optimize(sdfg, dace.DeviceType.GPU)
  _N = 16
  A = np.random.rand(_N, _N, _N, _N).astype(np.float32)
  B = np.random.rand(_N, _N, _N, _N).astype(np.float32)
  A2 = cp.asarray(A, cp.float32)
  B2 = cp.asarray(B, cp.float32)

  _test_transformations(opt_sdfg, A, B, A2, B2, _N)

def test_jacobi_2d():
  def jacobi_kernel(TSTEPS, A, B):
      for _ in range(0, TSTEPS):
          B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] +
                              A[2:, 1:-1] + A[:-2, 1:-1])
          A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] +
                              B[2:, 1:-1] + B[:-2, 1:-1])


  @dace.program
  def dace_jacobi_kernel(TSTEPS: dace.int32, 
                  A: dace.float32[N, N] @ dtypes.StorageType.GPU_Global, 
                  B: dace.float32[N, N] @ dtypes.StorageType.GPU_Global):
      for _ in range(0, TSTEPS):
          B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] +
                              A[2:, 1:-1] + A[:-2, 1:-1])
          A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] +
                              B[2:, 1:-1] + B[:-2, 1:-1])

  sdfg = dace_jacobi_kernel.to_sdfg()
  opt_sdfg = aopt.auto_optimize(sdfg, dace.DeviceType.GPU)
  apply_add_thread_block_schedule(opt_sdfg)
  apply_change_thread_block_schedule(opt_sdfg)
  apply_thread_coarsening(opt_sdfg)
  apply_block_coarsening(opt_sdfg)
  _N = 1024
  steps = 10
  A = np.random.rand(_N, _N).astype(np.float32)
  B = np.random.rand(_N, _N).astype(np.float32)
  A2 = cp.asarray(A, cp.float32)
  B2 = cp.asarray(B, cp.float32)
  jacobi_kernel(TSTEPS=steps, A=A, B=B)
  opt_sdfg(TSTEPS=steps, A=A2, B=B2, N=_N)
  assert(np.allclose(A, A2.get()))
  assert(np.allclose(B, B2.get()))

def test_mem_move_1d():
  @dace.program
  def dace_kernel_1d(A: dace.float32[M] @ dtypes.StorageType.GPU_Global, 
                     B: dace.float32[M] @ dtypes.StorageType.GPU_Global):
    for i0 in dace.map[0:M:256] @ dtypes.ScheduleType.GPU_Device:
      for g0 in dace.map[0:2:1] @ dtypes.ScheduleType.Sequential:
        for j0 in dace.map[0:128:1] @ dtypes.ScheduleType.GPU_ThreadBlock:
            A[i0 + g0*128 + j0] = A[i0 + g0*128 + j0] +  np.float32(0.5) * B[i0 + g0*128 + j0]

  sdfg = dace_kernel_1d.to_sdfg()
  sdfg.validate()
  apply_mem_move(sdfg)
  sdfg.validate()

  _M = 1024
  A = np.random.rand(_M).astype(np.float32)
  B = np.random.rand(_M).astype(np.float32)
  A2 = cp.asarray(A, cp.float32)
  B2 = cp.asarray(B, cp.float32)
  _tensor_add_kernel(A, B)
  sdfg(A=A2, B=B2, M=_M)
  sdfg.validate()
  AH = A2.get()
  assert(np.allclose(A, AH))

def test_mem_move_1d_type_2():
  @dace.program
  def dace_kernel_1d_type_2(A: dace.float32[M] @ dtypes.StorageType.GPU_Global, 
                            B: dace.float32[M] @ dtypes.StorageType.GPU_Global):
    for i0 in dace.map[0:M:256] @ dtypes.ScheduleType.GPU_Device:
      for g0 in dace.map[i0:i0+256:128] @ dtypes.ScheduleType.Sequential:
        for j0 in dace.map[g0:g0+128:1] @ dtypes.ScheduleType.GPU_ThreadBlock:
            A[j0] = A[j0] +  np.float32(0.5) * B[j0]

  sdfg = dace_kernel_1d_type_2.to_sdfg()
  sdfg.validate()
  apply_mem_move(sdfg)
  sdfg.validate()

  _M = 1024
  A = np.random.rand(_M).astype(np.float32)
  B = np.random.rand(_M).astype(np.float32)
  A2 = cp.asarray(A, cp.float32)
  B2 = cp.asarray(B, cp.float32)
  _tensor_add_kernel(A, B)
  sdfg(A=A2, B=B2, M=_M)
  sdfg.validate()
  AH = A2.get()
  assert(np.allclose(A, AH))


def test_mem_move_2d():
  @dace.program
  def dace_kernel_2d(A: dace.float32[M, M] @ dtypes.StorageType.GPU_Global, 
                     B: dace.float32[M, M] @ dtypes.StorageType.GPU_Global):
    for i0, i1 in dace.map[0:M:32, 0:M:32] @ dtypes.ScheduleType.GPU_Device:
      for g0, g1 in dace.map[0:2:1, 0:2:1] @ dtypes.ScheduleType.Sequential:
        for j0, j1 in dace.map[0:16:1, 0:16:1] @ dtypes.ScheduleType.GPU_ThreadBlock:
            A[i0 + g0*16 + j0, i1 + g1*16 + j1] = A[i0 + g0*16 + j0, i1 + g1*16 + j1] + np.float32(0.5) * B[i0 + g0*16 + j0, i1 + g1*16 + j1]

  sdfg = dace_kernel_2d.to_sdfg()
  sdfg.validate()
  apply_mem_move(sdfg)
  sdfg.validate()

  _M = 256
  A = np.random.rand(_M, _M).astype(np.float32)
  B = np.random.rand(_M, _M).astype(np.float32)
  A2 = cp.asarray(A, cp.float32)
  B2 = cp.asarray(B, cp.float32)
  _tensor_add_kernel(A, B)
  sdfg(A=A2, B=B2, M=_M)
  sdfg.validate()
  AH = A2.get()
  assert(np.allclose(A, AH))


def test_mem_move_2d_type_2():
  @dace.program
  def dace_kernel_2d_type_2(A: dace.float32[M, M] @ dtypes.StorageType.GPU_Global,
                            B: dace.float32[M, M] @ dtypes.StorageType.GPU_Global):
    for i0, i1 in dace.map[0:M:32, 0:M:32] @ dtypes.ScheduleType.GPU_Device:
      for g0, g1 in dace.map[i0:i0+32:16, i1:i1+32:16] @ dtypes.ScheduleType.Sequential:
        for j0, j1 in dace.map[g0:g0+16:1, g1:g1+16:1] @ dtypes.ScheduleType.GPU_ThreadBlock:
            A[j0, j1] = A[j0, j1] +  np.float32(0.5) * B[j0, j1]

  sdfg = dace_kernel_2d_type_2.to_sdfg()
  sdfg.validate()
  apply_mem_move(sdfg)
  sdfg.validate()

  _M = 256
  A = np.random.rand(_M, _M).astype(np.float32)
  B = np.random.rand(_M, _M).astype(np.float32)
  A2 = cp.asarray(A, cp.float32)
  B2 = cp.asarray(B, cp.float32)
  _tensor_add_kernel(A, B)
  sdfg(A=A2, B=B2, M=_M)
  sdfg.validate()
  AH = A2.get()
  assert(np.allclose(A, AH))

def test_mem_move_3d():
  @dace.program
  def dace_kernel_3d(A: dace.float32[M, N, N] @ dtypes.StorageType.GPU_Global, 
                     B: dace.float32[M, N, N] @ dtypes.StorageType.GPU_Global):
    for i0, i1, i2 in dace.map[0:M:32, 0:N:8, 0:N:8] @ dtypes.ScheduleType.GPU_Device:
      for g0, g1, g2 in dace.map[0:2:1, 0:2:1, 0:2:1] @ dtypes.ScheduleType.Sequential:
        for j0, j1, j2 in dace.map[0:16:1, 0:4:1, 0:4:1] @ dtypes.ScheduleType.GPU_ThreadBlock:
            A[i0 + g0*16 + j0, i1 + g1*4 + j1, i2 + g2*4 + j2] = \
              A[i0 + g0*16 + j0, i1 + g1*4 + j1, i2 + g2*4 + j2] + \
              np.float32(0.5) * \
              B[i0 + g0*16 + j0, i1 + g1*4 + j1, i2 + g2*4 + j2]

  sdfg = dace_kernel_3d.to_sdfg()
  sdfg.validate()
  apply_mem_move(sdfg)
  sdfg.validate()


  _M = 128
  _N = 8
  A = np.random.rand(_M, _N, _N).astype(np.float32)
  B = np.random.rand(_M, _N, _N).astype(np.float32)
  A2 = cp.asarray(A, cp.float32)
  B2 = cp.asarray(B, cp.float32)
  _tensor_add_kernel(A, B)
  sdfg(A=A2, B=B2, M=_M, N=_N)
  sdfg.validate()
  AH = A2.get()
  assert(np.allclose(A, AH))


def test_mem_move_3d_type_2():
  @dace.program
  def dace_kernel_3d_type_2(A: dace.float32[M, N, N] @ dtypes.StorageType.GPU_Global, 
                            B: dace.float32[M, N, N] @ dtypes.StorageType.GPU_Global):
    for i0, i1, i2 in dace.map[0:M:32, 0:N:8, 0:N:8] @ dtypes.ScheduleType.GPU_Device:
      for g0, g1, g2 in dace.map[i0:i0+32:16, i1:i1+8:4, i2:i2+8:4] @ dtypes.ScheduleType.Sequential:
        for j0, j1, j2 in dace.map[g0:g0+16:1, g1:g1+4:1, g2:g2+4:1] @ dtypes.ScheduleType.GPU_ThreadBlock:
            A[j0, j1, j2] = A[j0, j1, j2] +  np.float32(0.5) * B[j0, j1, j2]

  sdfg = dace_kernel_3d_type_2.to_sdfg()
  sdfg.validate()
  apply_mem_move(sdfg)
  sdfg.validate()

  _M = 128
  _N = 8
  A = np.random.rand(_M, _N, _N).astype(np.float32)
  B = np.random.rand(_M, _N, _N).astype(np.float32)
  A2 = cp.asarray(A, cp.float32)
  B2 = cp.asarray(B, cp.float32)
  _tensor_add_kernel(A, B)
  sdfg(A=A2, B=B2, M=_M, N=_N)
  sdfg.validate()
  AH = A2.get()
  assert(np.allclose(A, AH))


def test_mem_move_4d_type_2():
  @dace.program
  def dace_kernel_4d_type_2(A: dace.float32[M, N, N, N] @ dtypes.StorageType.GPU_Global, 
                            B: dace.float32[M, N, N, N] @ dtypes.StorageType.GPU_Global):
    for i0, i1, i2, i3 in dace.map[0:M:32, 0:N:8, 0:N:8, 0:N:1] @ dtypes.ScheduleType.GPU_Device:
      for g0, g1, g2, g3 in dace.map[i0:i0+32:16, i1:i1+8:4, i2:i2+8:4, i3:i3+1:1] @ dtypes.ScheduleType.Sequential:
        for j0, j1, j2, j3 in dace.map[g0:g0+16:1, g1:g1+4:1, g2:g2+4:1, g3:g3+1:1] @ dtypes.ScheduleType.GPU_ThreadBlock:
            A[j0, j1, j2, j3] = A[j0, j1, j2, j3] +  np.float32(0.5) * B[j0, j1, j2, j3]

  sdfg = dace_kernel_4d_type_2.to_sdfg()
  sdfg.validate()
  apply_mem_move(sdfg)
  sdfg.validate()

  _M = 64
  _N = 8
  A = np.random.rand(_M, _N, _N, _N).astype(np.float32)
  B = np.random.rand(_M, _N, _N, _N).astype(np.float32)
  A2 = cp.asarray(A, cp.float32)
  B2 = cp.asarray(B, cp.float32)
  _tensor_add_kernel(A, B)
  sdfg(A=A2, B=B2, M=_M, N=_N)
  sdfg.validate()
  AH = A2.get()
  assert(np.allclose(A, AH))

def _tensor_add_kernel_transposed_1d(A, B, N):
  for i in range(N):
    A[i] += 0.5 * B[N - (i+1)]

def test_mem_move_transposed_1d_type_2():
  @dace.program
  def dace_kernel_1d_transposed_type_2(A: dace.float32[M] @ dtypes.StorageType.GPU_Global, 
                            B: dace.float32[M] @ dtypes.StorageType.GPU_Global):
    for i0 in dace.map[0:M:256] @ dtypes.ScheduleType.GPU_Device:
      for g0 in dace.map[i0:i0+256:128] @ dtypes.ScheduleType.Sequential:
        for j0 in dace.map[g0:g0+128:1] @ dtypes.ScheduleType.GPU_ThreadBlock:
            A[j0] = A[j0] +  np.float32(0.5) * B[M - (j0 + 1)]

  sdfg = dace_kernel_1d_transposed_type_2.to_sdfg()
  sdfg.validate()
  apply_mem_move(sdfg)
  sdfg.validate()

  _M = 1024
  A = np.random.rand(_M).astype(np.float32)
  B = np.random.rand(_M).astype(np.float32)
  A2 = cp.asarray(A, cp.float32)
  B2 = cp.asarray(B, cp.float32)
  _tensor_add_kernel_transposed_1d(A, B, _M)
  sdfg(A=A2, B=B2, M=_M)
  sdfg.validate()
  AH = A2.get()
  assert(np.allclose(A, AH))

def _tensor_add_kernel_transposed_2d(A, B, N):
  for i in range(N):
    for j in range(N):
      A[i, j] += 0.5 * B[j, i]

def test_mem_move_2d_transposed_type_2():
  @dace.program
  def dace_kernel_2d_transposed_type_2(A: dace.float32[M, M] @ dtypes.StorageType.GPU_Global,
                                       B: dace.float32[M, M] @ dtypes.StorageType.GPU_Global):
    for i0, i1 in dace.map[0:M:32, 0:M:32] @ dtypes.ScheduleType.GPU_Device:
      for g0, g1 in dace.map[i0:i0+32:16, i1:i1+32:16] @ dtypes.ScheduleType.Sequential:
        for j0, j1 in dace.map[g0:g0+16:1, g1:g1+16:1] @ dtypes.ScheduleType.GPU_ThreadBlock:
            A[j0, j1] = A[j0, j1] +  np.float32(0.5) * B[j1, j0]

  sdfg = dace_kernel_2d_transposed_type_2.to_sdfg()
  sdfg.validate()
  apply_mem_move(sdfg)
  sdfg.validate()

  _M = 256
  A = np.random.rand(_M, _M).astype(np.float32)
  B = np.random.rand(_M, _M).astype(np.float32)
  A2 = cp.asarray(A, cp.float32)
  B2 = cp.asarray(B, cp.float32)
  _tensor_add_kernel_transposed_2d(A, B, _M)
  sdfg(A=A2, B=B2, M=_M)
  sdfg.validate()
  AH = A2.get()
  assert(np.allclose(A, AH))

if __name__ == '__main__':
  """
  test_tensor_add_1d()
  test_tensor_add_2d()
  test_tensor_add_3d()
  test_tensor_add_4d()
  test_jacobi_2d()
  test_mem_move_1d()
  test_mem_move_1d_type_2()
  test_mem_move_2d()
  test_mem_move_2d_type_2()
  test_mem_move_3d()
  test_mem_move_3d_type_2()
  test_mem_move_4d_type_2()
  """
  test_mem_move_transposed_1d_type_2()
  test_mem_move_2d_transposed_type_2()