import argparse
import dace
import numpy as np
from dace.transformation.interstate import LoopToMap

def make_sdfg():

    sdfg = dace.SDFG("loop_to_map_test")

    init = sdfg.add_state("init")
    guard = sdfg.add_state("guard")
    body = sdfg.add_state("body")
    after = sdfg.add_state("after")

    N = dace.symbol("N", dace.uint64)

    sdfg.add_edge(init, guard, dace.InterstateEdge(assignments={"i": "0"}))
    sdfg.add_edge(guard, body, dace.InterstateEdge(condition="i < N"))
    sdfg.add_edge(guard, after, dace.InterstateEdge(condition="i >= N"))
    sdfg.add_edge(body, guard, dace.InterstateEdge(assignments={"i": "i + 1"}))

    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    sdfg.add_array("C", [N], dace.float64)
    sdfg.add_array("D", [N], dace.float64)

    a = body.add_read("A")
    b = body.add_read("B")
    c = body.add_write("C")
    d = body.add_write("D")

    tasklet0 = body.add_tasklet("tasklet0", {"a"}, {"c"}, "c = 1/a")
    tasklet1 = body.add_tasklet("tasklet1", {"a", "b"}, {"d"},
                                "d = sqrt(a**2 + b**2)")

    body.add_memlet_path(a, tasklet0, dst_conn="a", memlet=dace.Memlet("A[i]"))
    body.add_memlet_path(tasklet0, c, src_conn="c", memlet=dace.Memlet("C[i]"))

    body.add_memlet_path(a, tasklet1, dst_conn="a", memlet=dace.Memlet("A[i]"))
    body.add_memlet_path(b, tasklet1, dst_conn="b", memlet=dace.Memlet("B[i]"))
    body.add_memlet_path(tasklet1, d, src_conn="d", memlet=dace.Memlet("D[i]"))

    return sdfg

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", default=64, type=int)
    args = parser.parse_args()

    sdfg = make_sdfg()

    sdfg.save("loop_to_map_before.sdfg")

    sdfg.apply_transformations(LoopToMap)

    sdfg.save("loop_to_map_after.sdfg")

    a = 4 * np.ones((args.N, ), dtype=np.float64)
    b = 3 * np.ones((args.N, ), dtype=np.float64)
    c = np.empty((args.N, ), dtype=np.float64)
    d = np.empty((args.N, ), dtype=np.float64)

    sdfg(A=a, B=b, C=c, D=d, N=args.N)

    if not all(c[:] == 0.25) or not all(d[:] == 5):
        raise ValueError("Unexpected value returned.")
