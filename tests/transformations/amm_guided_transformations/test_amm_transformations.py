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
from dace import dtypes

N = dace.symbol('N')

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

def apply_load_compute(sdfg):
  pass

def tensor_add_kernel(A, B):
   A += 0.5 * B

def _test_transformations(opt_sdfg, A, B, A2, B2, _N):
  tensor_add_kernel(A, B)
  apply_add_thread_block_schedule(opt_sdfg)
  opt_sdfg(A=A2, B=B2, N=_N)
  assert(np.allclose(A, A2.get()))
  tensor_add_kernel(A, B)
  apply_change_thread_block_schedule(opt_sdfg)
  opt_sdfg(A=A2, B=B2, N=_N)
  assert(np.allclose(A, A2.get()))
  tensor_add_kernel(A, B)
  apply_thread_coarsening(opt_sdfg)
  opt_sdfg(A=A2, B=B2, N=_N)
  assert(np.allclose(A, A2.get()))
  tensor_add_kernel(A, B)
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

if __name__ == '__main__':
  test_tensor_add_1d()
  test_tensor_add_2d()
  test_tensor_add_3d()
  test_tensor_add_4d()
  test_jacobi_2d()
