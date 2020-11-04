# Copyright 2020-2020 ETH Zurich and the DaCe authors. All rights reserved.
import argparse
import dace
import numpy as np
import os
from dace.transformation.interstate import LoopToMap


def make_sdfg(with_wcr, map_in_guard, reverse_loop, use_variable, assign_after):

    sdfg = dace.SDFG(f"loop_to_map_test_{with_wcr}_{map_in_guard}_"
                     f"{reverse_loop}_{use_variable}_{assign_after}")
    sdfg.set_global_code("#include <fstream>\n#include <mutex>")

    init = sdfg.add_state("init")
    guard = sdfg.add_state("guard")
    body = sdfg.add_state("body")
    after = sdfg.add_state("after")
    post = sdfg.add_state("post")

    N = dace.symbol("N", dace.int32)

    if not reverse_loop:
        sdfg.add_edge(init, guard, dace.InterstateEdge(assignments={"i": "0"}))
        sdfg.add_edge(guard, body, dace.InterstateEdge(condition="i < N"))
        sdfg.add_edge(guard, after, dace.InterstateEdge(condition="i >= N"))
        sdfg.add_edge(body, guard,
                      dace.InterstateEdge(assignments={"i": "i + 1"}))
    else:
        sdfg.add_edge(init, guard,
                      dace.InterstateEdge(assignments={"i": "N - 1"}))
        sdfg.add_edge(guard, body, dace.InterstateEdge(condition="i >= 0"))
        sdfg.add_edge(guard, after, dace.InterstateEdge(condition="i < 0"))
        sdfg.add_edge(body, guard,
                      dace.InterstateEdge(assignments={"i": "i - 1"}))
    sdfg.add_edge(
        after, post,
        dace.InterstateEdge(assignments={"i": "N"} if assign_after else None))

    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    sdfg.add_array("C", [N], dace.float64)
    sdfg.add_array("D", [N], dace.float64)
    sdfg.add_array("E", [1], dace.uint16)

    a = body.add_read("A")
    b = body.add_read("B")
    c = body.add_write("C")
    d = body.add_write("D")

    if map_in_guard:
        guard_read = guard.add_read("C")
        guard_write = guard.add_write("C")
        guard.add_mapped_tasklet("write_self", {"i": "0:N"},
                                 {"c_in": dace.Memlet("C[i]")},
                                 "c_out = c_in", {"c_out": dace.Memlet("C[i]")},
                                 external_edges=True,
                                 input_nodes={"C": guard_read},
                                 output_nodes={"C": guard_write})

    tasklet0 = body.add_tasklet("tasklet0", {"a"}, {"c"}, "c = 1/a")
    tasklet1 = body.add_tasklet("tasklet1", {"a", "b"}, {"d"},
                                "d = sqrt(a**2 + b**2)")

    tasklet2 = body.add_tasklet("tasklet2", {}, {},
                                """\
static std::mutex mutex;
std::unique_lock<std::mutex> lock(mutex);
std::ofstream of("loop_to_map_test.txt", std::ofstream::app);
of << i << "\\n";""",
                                language=dace.Language.CPP)

    body.add_memlet_path(a, tasklet0, dst_conn="a", memlet=dace.Memlet("A[i]"))
    body.add_memlet_path(tasklet0,
                         c,
                         src_conn="c",
                         memlet=dace.Memlet(
                             "C[i]",
                             wcr="lambda a, b: a + b" if with_wcr else None))

    body.add_memlet_path(a, tasklet1, dst_conn="a", memlet=dace.Memlet("A[i]"))
    body.add_memlet_path(b, tasklet1, dst_conn="b", memlet=dace.Memlet("B[i]"))
    body.add_memlet_path(tasklet1,
                         d,
                         src_conn="d",
                         memlet=dace.Memlet(
                             "D[i]",
                             wcr="lambda a, b: a + b" if with_wcr else None))

    e = post.add_write("E")
    post_tasklet = post.add_tasklet("post", {}, {"e"},
                                    "e = i" if use_variable else "e = N")
    post.add_memlet_path(post_tasklet,
                         e,
                         src_conn="e",
                         memlet=dace.Memlet("E[0]"))

    return sdfg


def apply_and_verify(sdfg, n):

    if n is None:
        n = dace.int32(16)

    a = 4 * np.ones((n, ), dtype=np.float64)
    b = 3 * np.ones((n, ), dtype=np.float64)
    c = np.zeros((n, ), dtype=np.float64)
    d = np.zeros((n, ), dtype=np.float64)
    e = np.empty((1, ), dtype=np.uint16)

    num_transformations = sdfg.apply_transformations(LoopToMap)

    try:
        os.remove("loop_to_map_test.txt")
    except FileNotFoundError:
        pass

    sdfg(A=a, B=b, C=c, D=d, E=e, N=n)

    if not all(c[:] == 0.25) or not all(d[:] == 5):
        raise ValueError("Validation failed.")

    if e[0] != n:
        raise ValueError("Validation failed.")

    numbers_written = []
    with open("loop_to_map_test.txt", "r") as f:
        for line in f:
            numbers_written.append(int(line.strip()))
    if not all(sorted(numbers_written) == np.arange(n)):
        raise ValueError("Validation failed.")

    os.remove("loop_to_map_test.txt")

    return num_transformations


def test_loop_to_map(n=None):
    # Case 0: no wcr, no dataflow in guard. Transformation should apply
    if apply_and_verify(make_sdfg(False, False, False, False, False), n) != 1:
        raise RuntimeError("LoopToMap was not applied.")


def test_loop_to_map_wcr(n=None):
    # Case 1: WCR on the edge. Transformation should apply
    if apply_and_verify(make_sdfg(True, False, False, False, False), n) != 1:
        raise RuntimeError("LoopToMap was not applied.")


def test_loop_to_map_dataflow_on_guard(n=None):
    # Case 2: there is dataflow on the guard state. Should not apply
    if apply_and_verify(make_sdfg(False, True, False, False, False), n) != 0:
        raise RuntimeError("LoopToMap should not have been applied.")


def test_loop_to_map_negative_step(n=None):
    # Case 3: loop order reversed. Transformation should still apply
    if apply_and_verify(make_sdfg(False, False, True, False, False), n) != 1:
        raise RuntimeError("LoopToMap was not applied.")


def test_loop_to_map_variable_used(n=None):
    # Case 4: the loop variable is used in a later state: should not apply
    if apply_and_verify(make_sdfg(False, False, False, True, False), n) != 0:
        raise RuntimeError("LoopToMap should not have been applied.")


def test_loop_to_map_variable_reassigned(n=None):
    # Case 5: the loop variable is used in a later state, but reassigned first:
    # should apply
    if apply_and_verify(make_sdfg(False, False, False, True, True), n) != 1:
        raise RuntimeError("LoopToMap was not applied.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", default=16, type=int)
    args = parser.parse_args()

    n = np.int32(args.N)

    test_loop_to_map(n)
    test_loop_to_map_wcr(n)
    test_loop_to_map_dataflow_on_guard(n)
    test_loop_to_map_negative_step(n)
    test_loop_to_map_variable_used(n)
    test_loop_to_map_variable_reassigned(n)
