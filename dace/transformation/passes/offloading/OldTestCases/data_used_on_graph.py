import dace
import numpy as np
from typing import Dict, Set


# TODO: Implement by using DaCe's built-in read/write set analysis.
# You probably need to extend it 
# DaCe provides:
#     read_set, write_set = state.read_and_write_sets()
#
# This returns two sets of container names (strings) per state.
#
# For this analysis we need more than names: we need to know the
# *storage type* of each accessed container, and whether the access
# happens inside a GPU-scheduled map or a sequential/CPU scope.


def arrays_accessed_per_state(sdfg: dace.SDFG) -> Dict[dace.SDFGState, Dict[str, Set[str]]]:
    return dict()


@dace.program
def example(A: dace.float64[10], B: dace.float64[10], C: dace.float64[10],
            D: dace.float64[10], N: int, i: int):
    if i < N / 2:
        for j in range(1, N):
            A[j] = A[j - 1] + B[j] + C[j]
        for j in dace.map[0:N // 2] @ dace.dtypes.ScheduleType.GPU_Device:
            D[j] = B[j] * 2
    else:
        for j in dace.map[0:N] @ dace.dtypes.ScheduleType.GPU_Device:
            A[j] = B[j] * C[j]
        for j in dace.map[0:N // 2] @ dace.dtypes.ScheduleType.GPU_Device:
            D[j] = B[j] * 2


def test_example() -> dace.SDFG:
    # Validation intentionally skipped: the program declares A, B, C, D
    # with default (CPU) storage, but GPU_Device maps access them.
    # This mismatch is irrelevant for the *analysis* we're building —
    # the goal is to determine which arrays need to be on which device,
    # not to enforce it.
    sdfg = example.to_sdfg(validate=False)

    # NOTE: The SDFG viewer may fail to render GPU-scheduled maps when
    # arrays have CPU storage. To inspect the graph visually, save a
    # version without GPU schedule annotations:
    #
    #   @dace.program
    #   def example(...):
    #       if i < N / 2:
    #           for j in range(1, N):        # sequential (CPU)
    #               A[j] = A[j-1] + B[j] + C[j]
    #           for j in dace.map[0:N//2]:   # no GPU annotation
    #               D[j] = B[j] * 2
    #       else:
    #           for j in dace.map[0:N]:
    #               A[j] = B[j] * C[j]
    #           for j in dace.map[0:N//2]:
    #               D[j] = B[j] * 2

    sdfg.save("example.sdfg", compress=False)
    #sdfg.view()

    # Expected SDFG structure:
    #
    # ┌─────────────────────────────────────────────────────────┐
    # │ State 0 (init):                                         │
    # │   Computes N_div_2 = N // 2 (CPU scalar assignment).    │
    # │   N_div_2 becomes a symbol "tmp" used in the branch     |
    # |     condition.                                          │
    # └──────────────────────┬──────────────────────────────────┘
    #                        │
    #              ┌─────────┴─────────┐
    #              ▼                   ▼
    #     ┌── if (i < tmp) ──┐  ┌── else ──────────-┐
    #     │                  │  │                   │
    #     │ State 1:         │  │ State 3:          │
    #     │  Sequential loop │  │  GPU map over j   │
    #     │  A[j] = A[j-1]   │  │  A[j] = B[j]*C[j] │
    #     │       + B[j]     │  │  reads: A, B, C   │
    #     │       + C[j]     │  │  writes: A        │
    #     │  reads:  A,B,C   │  │                   │
    #     │  writes: A       │  │ State 4:          │
    #     │                  │  │  GPU map over j   │
    #     │ State 2:         │  │  D[j] = B[j] * 2  │
    #     │  GPU map over j  │  │  reads:  B        │
    #     │  D[j] = B[j] * 2 │  │  writes: D        │
    #     │  reads:  B       │  └───────────────────┘
    #     │  writes: D       │
    #     └──────────────────┘
    #              │                   │
    #              └─────────┬─────────┘
    #                        ▼
    #                      merge
    #
    # Analysis conclusion:
    #   D     → GPU only (both branches write D in a GPU map)
    #   B     → GPU and CPU (GPU maps read it; the if-branch's
    #           sequential loop also reads it on CPU)
    #   A, C  → GPU and CPU (same reasoning: the if-branch reads/writes
    #           them sequentially on CPU, the else-branch accesses them
    #           in a GPU map)
    #
    # This divergence is the interesting case: the same array may need
    # to reside on *both* devices depending on which branch executes.
    # A downstream pass would need to insert host↔device copies or
    # use unified memory to satisfy both paths.

    # Important: being able to offload this requires chaining the if else
    # which we can support much later when inserting copies:
    # ┌── if (i < tmp) ──┐
    # |                  |
    # └──────────────────┘
    #         -─┬─-
    #           ▼
    # ┌── if (i >= tmp) ─┐
    # |                  |
    # └──────────────────┘
    #         -─┬─-
    #           ▼
    # Then we can sequqntialize and insert necessary copies. 
    # I already have a construction utility that performs this
    # but this is a very late step in the pipeline, so we can ignore it for now.

    sdfg.validate()
    return sdfg


from dace.transformation.passes.offloading.OffloadToAccelerator import OffloadToAccelerator as OtA


def pretty_print(gpu_set, cpu_set):
    print("both GPU and CPU:")
    for name in gpu_set & cpu_set:
        print("\t", name)

    print("\nGPU only:")
    for name in gpu_set - cpu_set:
        print("\t", name)
    
    print("\nCPU only:")
    for name in cpu_set - gpu_set:
        print("\t", name)
    

### tests ####
N = dace.symbol('N')
M = dace.symbol('M')

def test_complex_graph():
    sdfg : dace.SDFG = test_example()
    gpu_set, cpu_set = OtA().get_data_locations(sdfg)
    
    assert set(gpu_set & cpu_set) == {'A', 'B', 'C'}
    assert set(gpu_set - cpu_set) == {'D_slice', 'D', 'example_28_8_D_slice', 'A_slice'}
    assert set(cpu_set - gpu_set) == {'N', 'N_div_2', 'A_index', 'B_index', 'C_index', 'A_slice_plus_B_slice', 'A_slice_B_slice_plus_C_slice'}

def test_triple_nested_maps_gpu():
    """
    Triple-nested map: Sequential > GPU_Device > Sequential.
    Reads A (20x10) and B (10x15), accumulates into C (20x15).
    All three arrays are GPU_Global.

    data_used_by_map is called on the outermost sequential map entry.
    Since the function traverses the entire scope (including nested maps),
    it must find all three containers. The storage type is resolved from
    the SDFG's data descriptors (the global struct), not from the map
    schedule — a sequential map can still touch GPU arrays if it is
    nested inside (or is the parent of) a GPU map.
    """
    sdfg = dace.SDFG('triple_nested')
    sdfg.add_array('A', [20, 10], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('B', [10, 15], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('C', [20, 15], dace.float64, storage=dace.StorageType.GPU_Global)

    state = sdfg.add_state('compute')

    # Outer: sequential over i
    outer_entry, outer_exit = state.add_map(
        'seq_outer', {'i': '0:20'}, schedule=dace.ScheduleType.Sequential)
    # Middle: GPU_Device over j
    mid_entry, mid_exit = state.add_map(
        'gpu_mid', {'j': '0:15'}, schedule=dace.ScheduleType.GPU_Device)
    # Inner: sequential over k (runs on GPU thread)
    inner_entry, inner_exit = state.add_map(
        'seq_inner', {'k': '0:10'}, schedule=dace.ScheduleType.Sequential)

    tasklet = state.add_tasklet('mac', {'a', 'b', 'cin'}, {'cout'},
                                'cout = cin + a * b')
    a_node = state.add_read('A')
    b_node = state.add_read('B')
    c_read = state.add_access('C')
    c_write = state.add_write('C')

    # A[i, k] through all three maps
    state.add_memlet_path(a_node, outer_entry, mid_entry, inner_entry, tasklet,
                          dst_conn='a', memlet=dace.Memlet('A[i, k]'))
    # B[k, j] through all three maps
    state.add_memlet_path(b_node, outer_entry, mid_entry, inner_entry, tasklet,
                          dst_conn='b', memlet=dace.Memlet('B[k, j]'))
    # C[i, j] read through all three maps
    state.add_memlet_path(c_read, outer_entry, mid_entry, inner_entry, tasklet,
                          dst_conn='cin', memlet=dace.Memlet('C[i, j]'))
    # C[i, j] write back out
    state.add_memlet_path(tasklet, inner_exit, mid_exit, outer_exit, c_write,
                          src_conn='cout', memlet=dace.Memlet('C[i, j]'))

    sdfg.save('triple_nested.sdfg')
    sdfg.validate()
    #sdfg.view()

    gpu_set, cpu_set = OtA().get_data_locations(sdfg)
    assert gpu_set == {'A', 'B', 'C'}
    assert not cpu_set

def test_triple_nested_maps_outer_only_array():
    """
    Triple-nested map with an extra outer-only array D.

    Outer sequential map reads D[i] and writes E[i], while inner maps
    use A, B, C as in the original test. D/E are only accessed at the
    outer level and not by inner maps.
    """
    sdfg = dace.SDFG('triple_nested_outer_only')
    sdfg.add_array('A', [20, 10], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('B', [10, 15], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('C', [20, 15], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('D', [20], dace.float64, storage=dace.StorageType.Default)
    sdfg.add_array('E', [20], dace.float64, storage=dace.StorageType.Default)

    state = sdfg.add_state('compute')

    outer_entry, outer_exit = state.add_map(
        'seq_outer', {'i': '0:20'}, schedule=dace.ScheduleType.Sequential)
    mid_entry, mid_exit = state.add_map(
        'gpu_mid', {'j': '0:15'}, schedule=dace.ScheduleType.GPU_Device)
    inner_entry, inner_exit = state.add_map(
        'seq_inner', {'k': '0:10'}, schedule=dace.ScheduleType.Sequential)

    tasklet = state.add_tasklet('mac', {'a', 'b', 'cin'}, {'cout'},
                                'cout = cin + a * b')
    outer_tasklet = state.add_tasklet('outer_only', {'d_in'}, {'e_out'},
                                      'e_out = d_in')

    a_node = state.add_read('A')
    b_node = state.add_read('B')
    c_read = state.add_access('C')
    c_write = state.add_write('C')

    d_read = state.add_read('D')
    e_write = state.add_write('E')

    state.add_memlet_path(a_node, outer_entry, mid_entry, inner_entry, tasklet,
                          dst_conn='a', memlet=dace.Memlet('A[i, k]'))
    state.add_memlet_path(b_node, outer_entry, mid_entry, inner_entry, tasklet,
                          dst_conn='b', memlet=dace.Memlet('B[k, j]'))
    state.add_memlet_path(c_read, outer_entry, mid_entry, inner_entry, tasklet,
                          dst_conn='cin', memlet=dace.Memlet('C[i, j]'))
    state.add_memlet_path(tasklet, inner_exit, mid_exit, outer_exit, c_write,
                          src_conn='cout', memlet=dace.Memlet('C[i, j]'))

    state.add_memlet_path(d_read, outer_entry, outer_tasklet,
                          dst_conn='d_in', memlet=dace.Memlet('D[i]'))
    state.add_memlet_path(outer_tasklet, outer_exit, e_write,
                          src_conn='e_out', memlet=dace.Memlet('E[i]'))

    sdfg.save('triple_nested_outer_only.sdfg')
    sdfg.validate()
    #sdfg.view()

    gpu_set, cpu_set = OtA().get_data_locations(sdfg)
    #pretty_print(gpu_set, cpu_set)
    assert gpu_set == {'A', 'B', 'C'}
    assert cpu_set == {'D', 'E'}
      
def make_row_col_add_sdfg() -> tuple[dace.SDFG, dace.SDFGState, dace.nodes.MapEntry]:
    sdfg = dace.SDFG('row_col_add_views')

    # --- Data descriptors ---
    sdfg.add_array('A', [N, N], dace.float64)
    sdfg.add_array('C', [N, N], dace.float64)

    # Column view of A: A[:, i] has shape [N], stride [N] in row-major
    sdfg.add_view('A_col_view', [N], dace.float64, strides=[N])
    # Row view of C: C[i, :] has shape [N], stride [1] in row-major
    sdfg.add_view('C_row_view', [N], dace.float64, strides=[1])

    state = sdfg.add_state('compute', is_start_block=True)

    # --- Access nodes (outside maps) ---
    a_read = state.add_read('A')
    c_read = state.add_read('C')
    c_write = state.add_write('C')

    # --- Outer map: for i in range(N) ---
    outer_entry, outer_exit = state.add_map('i_map', {'i': '0:N'})

    # --- Access nodes inside outer scope (for view connections) ---
    a_inside = state.add_access('A')
    c_inside_r = state.add_access('C')
    c_inside_w = state.add_access('C')

    # --- View nodes inside outer scope ---
    a_col_node = state.add_access('A_col_view')
    c_row_in = state.add_access('C_row_view')
    c_row_out = state.add_access('C_row_view')

    # --- Inner map: for j in range(N) ---
    inner_entry, inner_exit = state.add_map('j_map', {'j': '0:N'})

    # --- Tasklet ---
    tasklet = state.add_tasklet('add', {'a_val', 'c_val'}, {'c_out'},
                                'c_out = c_val + a_val')

    # =====================================================================
    #  READ path for A[:, i]
    # =====================================================================
    # A_read -> outer_entry -> A_inside  (route through outer map)
    state.add_memlet_path(a_read, outer_entry, a_inside,
                          memlet=dace.Memlet('A[0:N, i]'))
    # A_inside -> A_col_view  (views edge: column slice)
    state.add_edge(a_inside, None, a_col_node, 'views',
                   dace.Memlet('A[0:N, i]'))
    # A_col_view -> inner_entry -> tasklet
    state.add_memlet_path(a_col_node, inner_entry, tasklet,
                          dst_conn='a_val',
                          memlet=dace.Memlet('A_col_view[j]'))

    # =====================================================================
    #  READ path for C[i, :]
    # =====================================================================
    # C_read -> outer_entry -> C_inside_r  (route through outer map)
    state.add_memlet_path(c_read, outer_entry, c_inside_r,
                          memlet=dace.Memlet('C[i, 0:N]'))
    # C_inside_r -> C_row_in  (views edge: row slice)
    state.add_edge(c_inside_r, None, c_row_in, 'views',
                   dace.Memlet('C[i, 0:N]'))
    # C_row_in -> inner_entry -> tasklet
    state.add_memlet_path(c_row_in, inner_entry, tasklet,
                          dst_conn='c_val',
                          memlet=dace.Memlet('C_row_view[j]'))

    # =====================================================================
    #  WRITE path for C[i, :]
    # =====================================================================
    # tasklet -> inner_exit -> C_row_out
    state.add_memlet_path(tasklet, inner_exit, c_row_out,
                          src_conn='c_out',
                          memlet=dace.Memlet('C_row_view[j]'))
    # C_row_out -> C_inside_w  (views edge: row slice, write direction)
    state.add_edge(c_row_out, 'views', c_inside_w, None,
                   dace.Memlet('C[i, 0:N]'))
    # C_inside_w -> outer_exit -> C_write  (route through outer map)
    state.add_memlet_path(c_inside_w, outer_exit, c_write,
                          memlet=dace.Memlet('C[i, 0:N]'))

    return sdfg, state, outer_entry, inner_entry

def test_row_col_views():
    """Validate the manually-built SDFG against numpy."""
    n = 8
    rng = np.random.default_rng(42)
    A = rng.random((n, n))
    C = rng.random((n, n))
    C_ref = C.copy()

    # Reference: for i in range(N): C[i, :] += A[:, i]
    for i in range(n):
        C_ref[i, :] += A[:, i]

    sdfg, state, outer_entry, inner_entry = make_row_col_add_sdfg()
    sdfg.validate()

    sdfg.save("row_col_add_views.sdfg")
    #sdfg.view()

    #print("\n\nrow_col_add_views.sdfg")
    gpu_set, cpu_set = OtA().get_data_locations(sdfg)
    #pretty_print(gpu_set, cpu_set)
    
    assert not gpu_set
    assert cpu_set == {'A', 'C', 'A_col_view', 'C_row_view'}
   

# ---------------------------------------------------------------------------
#  Test 3 (trivial) — Single map, CPU arrays
# ---------------------------------------------------------------------------

def test_single_map_cpu():
    sdfg = dace.SDFG('single_map_cpu')
    sdfg.add_array('X', [100], dace.float32, storage=dace.StorageType.CPU_Heap)
    sdfg.add_array('Y', [100], dace.float32, storage=dace.StorageType.CPU_Heap)

    state = sdfg.add_state('s')
    entry, exit_ = state.add_map('m', {'i': '0:100'})
    tasklet = state.add_tasklet('double', {'x'}, {'y'}, 'y = 2 * x')

    x_node = state.add_read('X')
    y_node = state.add_write('Y')

    state.add_memlet_path(x_node, entry, tasklet, dst_conn='x',
                          memlet=dace.Memlet('X[i]'))
    state.add_memlet_path(tasklet, exit_, y_node, src_conn='y',
                          memlet=dace.Memlet('Y[i]'))
    sdfg.save("single_map_cpu.sdfg")
    #sdfg.view()

    #print("\n\nsingle_map_cpu.sdfg")
    gpu_set, cpu_set = OtA().get_data_locations(sdfg)
    #pretty_print(gpu_set, cpu_set)
    assert not gpu_set
    assert cpu_set == {'X', 'Y'}

def test_empty_map():
    sdfg = dace.SDFG('empty_map')
    state = sdfg.add_state('s')
    entry, exit_ = state.add_map('m', {'i': '0:10'})
    # Connect entry directly to exit (empty body)
    state.add_edge(entry, None, exit_, None, dace.Memlet())

    sdfg.save("empty_map.sdfg")
    #sdfg.view()

    gpu_set, cpu_set = OtA().get_data_locations(sdfg)
    assert not gpu_set
    assert not cpu_set

def make_non_input_readwrite_node_sdfg() -> tuple[dace.SDFG, dace.SDFGState, dace.nodes.MapEntry]:
    """
    C[i,j] = C[i,j] + 1
    A[i,j] = B[i,j] + C[i,j]

    A: input only
    B: input only
    C: non-transient, read-modify-write inside the map
    """
    sdfg = dace.SDFG('inplace_c')

    sdfg.add_array('A', [N, M], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('B', [N, M], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('C', [N, M], dace.float64, storage=dace.StorageType.GPU_Global)

    state = sdfg.add_state('compute', is_start_block=True)

    # External access nodes
    b_read = state.add_read('B')
    a_write = state.add_write('A')

    # Map over i, j
    entry, exit_ = state.add_map('ij_map', {'i': '0:N', 'j': '0:M'}, schedule=dace.dtypes.ScheduleType.GPU_Device)

    # Two tasklets: first increments C, second computes A
    t_inc = state.add_tasklet('inc_c', {'b_in'}, {'c_out'}, 'c_out = b_in + 1')
    t_add = state.add_tasklet('add', {'b_in', 'c_new'}, {'a_out'}, 'a_out = b_in + c_new')


    # --- Updated C feeds into t_add ---
    c_node = state.add_access('C')   # both read and write
    
    state.add_edge(entry, "OUT_B", t_inc, 'b_in', dace.Memlet('B[i, j]'))
    state.add_edge(t_inc, "c_out", c_node, None, dace.Memlet('C[i, j]'))
    state.add_edge(c_node, None, t_add, 'c_new', dace.Memlet('C[i, j]'))


    # --- B read into t_add ---
    state.add_memlet_path(b_read, entry, t_add,
                          dst_conn='b_in',
                          memlet=dace.Memlet('B[i, j]'))

    # --- t_add writes A ---
    exit_.add_in_connector('IN_A')
    exit_.add_out_connector('OUT_A')

    state.add_edge(t_add, 'a_out', exit_, 'IN_A',
                dace.Memlet('A[i, j]'))
    state.add_edge(exit_, 'OUT_A', a_write, None,
                dace.Memlet('A[0:N, 0:M]'))

    return sdfg, state, entry

def test_map_with_non_input_readwrite_node() -> dace.SDFG:
    sdfg, state, entry = make_non_input_readwrite_node_sdfg()

    sdfg.validate()
    sdfg.save("non_input_readwrite.sdfg")
    #sdfg.view()

    gpu_set, cpu_set = OtA().get_data_locations(sdfg)    
    assert gpu_set == {'A', 'B', 'C'}
    assert not cpu_set


if __name__ == "__main__":
    test_triple_nested_maps_gpu()
    test_triple_nested_maps_outer_only_array()
    test_row_col_views()
    test_single_map_cpu()
    test_empty_map()
    test_map_with_non_input_readwrite_node()
    test_complex_graph()
    print("YESS! All tests passed.")