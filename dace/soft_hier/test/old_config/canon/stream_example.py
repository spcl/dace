import dace
import numpy as np

from dace.sdfg.state import LoopRegion

# Define symbols
N = dace.symbol("N")
NPE = dace.symbol("NPE")

# Create a new SDFG
sdfg = dace.SDFG("streaming_sdfg")

# Add arrays A and B, C and tmp accumulator
# Everything has square size, NPE = Number of PEs in 1 Dim (NPE x NPE PEs)
sdfg.add_array("A", shape=[N * NPE, N * NPE], dtype=dace.float32)
sdfg.add_array("B", shape=[N * NPE, N * NPE], dtype=dace.float32)
sdfg.add_array("C", shape=[N * NPE, N * NPE], dtype=dace.float32)
sdfg.add_array("tmp", shape=[N, N], dtype=dace.float32, transient=True)

state = sdfg.add_state("streaming_state")

# Single iteration device map for simplicity
device_entry, device_exit = state.add_map(
    "device_map", {"i": "0:N:N", "j": "0:N:N"}, schedule=dace.ScheduleType.GPU_Device
)

# Add threadblock map of dimensions 0:NPE, 0:NPE inside the device map
# ThreadBlock / Core group map is always necessary
tb_entry, tb_exit = state.add_map(
    "threadblock_map",
    {"ti": "0:NPE", "tj": "0:NPE"},
    schedule=dace.ScheduleType.GPU_ThreadBlock,
)


# To reuse the symbols later on
i = dace.symbol("i")
j = dace.symbol("j")
ti = dace.symbol("ti")
tj = dace.symbol("tj")


# Add access nodes for A and B, C, tmp
A = state.add_read("A")
B = state.add_read("B")
C = state.add_access("C")
tmp = state.add_access("tmp")

# Memlet path A -> Dev -> Tblock
state.add_memlet_path(
    A,
    device_entry,
    tb_entry,
    src_conn="OUT_A",
    dst_conn="IN_A",
    memlet=dace.Memlet("A[0:N*NPE, 0:N*NPE]"),
)

# Same for B
state.add_memlet_path(
    B,
    device_entry,
    tb_entry,
    src_conn="OUT_B",
    dst_conn="IN_B",
    memlet=dace.Memlet("B[0:N*NPE, 0:N*NPE]"),
)


state.add_edge(tb_entry, None, tmp, None, dace.Memlet(None))

# Create BSP nested SDFG of canon
nsdfg = dace.SDFG("nested_canon_sdfg")

# s is the stream prefix
# Load A, B at init step to sA, sB

# At compute (a CFG region), compute localA, localB, tmp = localA * localB
# localA is loaded from sA, localB is loaded from sB

# At communication, write current localA to the correct stream dimension

# Declare all data used
nsdfg.add_stream("sA", dtype=dace.float32, buffer_size=N*N, shape=(NPE, NPE), transient=True)
nsdfg.add_stream("sB", dtype=dace.float32, buffer_size=N*N, shape=(NPE, NPE), transient=True)
nsdfg.add_array("tmp", shape=[N, N], dtype=dace.float32, transient=False)
nsdfg.add_array("localA", shape=[N, N], dtype=dace.float32, transient=True)
nsdfg.add_array("localB", shape=[N, N], dtype=dace.float32, transient=True)
nsdfg.add_array("A", shape=[N, N], dtype=dace.float32, transient=False)
nsdfg.add_array("B", shape=[N, N], dtype=dace.float32, transient=False)

nsdfg_node = dace.nodes.NestedSDFG(
    label="nested_canon_node", sdfg=nsdfg, inputs={"A", "B", "tmp"}, outputs={"tmp"},
    symbol_mapping={"N": N, "NPE": NPE, "ti":ti, "tj":tj}
)

# Define inputs and outputs to the nested SDFG (acc)
state.add_edge(tmp, None, nsdfg_node, "tmp", dace.Memlet("tmp[0:N,0:N]"))
tmp2 = state.add_access("tmp")
state.add_edge(nsdfg_node, "tmp", tmp2, None, dace.Memlet("tmp[0:N,0:N]"))

state.add_memlet_path(
    tmp2,
    tb_exit,
    device_exit,
    C,
    dst_conn="IN_C",
    memlet=dace.Memlet("tmp[i * N: (i+1)*N, j * N: (j+1)*N]"),
)

# Move the accessed tiles to the nested SDFG
state.add_memlet_path(
    tb_entry,
    nsdfg_node,
    src_conn="OUT_A",
    dst_conn="A",
    memlet=dace.Memlet(
        "A[(i+ti) * N:((i+ti+1))*N, ((ti+tj) % NPE) * N:(((ti+tj) % NPE) +1) * N]",
    ),
)
state.add_memlet_path(
    tb_entry,
    nsdfg_node,
    src_conn="OUT_B",
    dst_conn="B",
    memlet=dace.Memlet(
        "B[((ti+tj) % NPE) * N:(((ti+tj) % NPE) +1) * N, (j+tj) * N:(j+tj+1)*N]",
    ),
)

# Create init state, first tiles are loaded to streams
init_state = nsdfg.add_state("canon_init")
A_init = init_state.add_read("A")
B_init = init_state.add_read("B")
sA_init = init_state.add_write("sA")
sB_init = init_state.add_write("sB")

init_state.add_edge(A_init, None, sA_init, None,
    memlet=dace.Memlet(
        data="A",
        subset=dace.subsets.Range([(0,N-1,1),(0,N-1,1)]),
        other_subset=dace.subsets.Range([((ti, ti, 1), (ti, ti, 1), 1), (((tj + ti) % NPE), ((tj + ti) % NPE), 1)])
    ),
)
init_state.add_edge(B_init, None, sB_init, None,
        memlet=dace.Memlet(
        data="B",
        subset=dace.subsets.Range([(0,N-1,1),(0,N-1,1)]),
        other_subset=dace.subsets.Range([(((ti + tj) % NPE), ((ti + tj) % NPE), 1), ((tj, tj, 1), (tj, tj, 1), 1)])
    ),
)

# Define the loop region (compute + communicate steps)
lr = LoopRegion(
    label="canon",
    condition_expr="_c < NPE",
    loop_var="_c",
    initialize_expr="_c = 0",
    update_expr="_c += 1",
    sdfg=sdfg,
)

nsdfg.add_node(lr)
nsdfg.add_edge(init_state, lr, dace.InterstateEdge(None, None))

lr_s1 : dace.SDFGState = lr.add_state("canon_compute")

# Load tiles from the streams, compute, store to temp
sA_compute = lr_s1.add_access("sA")
sB_compute = lr_s1.add_access("sB")
localA_compute = lr_s1.add_access("localA")
localB_compute = lr_s1.add_access("localB")
tmp_compute = lr_s1.add_access("tmp")
tasklet_GEMM = lr_s1.add_tasklet("GEMM", {"_B", "_A"}, {"_tmp"}, "OUT_tmp = _A * _B")

lr_s1.add_edge(sA_compute, None, localA_compute, None, dace.Memlet("sA[ti,tj]"))
lr_s1.add_edge(sB_compute, None, localB_compute, None, dace.Memlet("sB[ti,tj]"))
lr_s1.add_edge(localA_compute, None, tasklet_GEMM, "_A", dace.Memlet("localA[0:N, 0:N]"))
lr_s1.add_edge(localB_compute, None, tasklet_GEMM, "_B", dace.Memlet("localB[0:N, 0:N]"))
lr_s1.add_edge(tasklet_GEMM, "_tmp", tmp_compute, None, dace.Memlet("tmp[0:N, 0:N]"))

# Add communicate steps local data -> next streams
lr_s2 : dace.SDFGState = lr.add_state("canon_communicate")
lr.add_edge(lr_s1, lr_s2, dace.InterstateEdge(None, None))

localA_comm = lr_s2.add_access("localA")
localB_comm = lr_s2.add_access("localB")
sA_comm = lr_s2.add_access("sA")
sB_comm = lr_s2.add_access("sB")

lr_s2.add_edge(localA_comm, None, sA_comm, None,
    memlet=dace.Memlet(
        data="localA",
        subset=dace.subsets.Range([(0,N-1,1),(0,N-1,1)]),
        other_subset=dace.subsets.Range([((ti, ti, 1), (ti, ti, 1), 1), (((tj + NPE - 1) % NPE), ((tj + NPE -1) % NPE), 1)])
    ),
)
lr_s2.add_edge(localB_comm, None, sB_comm, None,
        memlet=dace.Memlet(
        data="localB",
        subset=dace.subsets.Range([(0,N-1,1),(0,N-1,1)]),
        other_subset=dace.subsets.Range([(((ti + NPE - 1) % NPE), ((ti + NPE -1) % NPE), 1), ((tj, tj, 1), (tj, tj, 1), 1)])
    ),
)


# Validate and save the SDFG

sdfg.save("streaming_sdfg.sdfgz")
sdfg.validate()