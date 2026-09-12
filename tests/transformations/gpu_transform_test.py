# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Unit tests for the GPU to-device transformation. """

import dace
import numpy as np
import pytest


@pytest.mark.gpu
def test_scalar_to_symbol_in_nested_sdfg():
    """
    Offloading automatically creates copy-out states for GPU scalars that are used in host-side interstate
    edges. However, this process may only be applied in top-level SDFGs and not in NestedSDFGs that have GPU-device
    schedule but are not part of a single GPU kernel, leading to illegal memory accesses.
    """

    @dace.program
    def nested_program(a: dace.int32, out: dace.int32[10]):
        for i in range(10):
            if a < 5:
                out[i] = 0
                a *= 2
            else:
                out[i] = 10
                a /= 2

    @dace.program
    def main_program(a: dace.int32):
        out = np.ndarray((10, ), dtype=np.int32)
        nested_program(a, out)
        return out

    sdfg = main_program.to_sdfg(simplify=False)
    sdfg.apply_gpu_transformations(simplify=False)
    out = sdfg(a=4)
    assert np.array_equal(out, np.array([0, 10] * 5, dtype=np.int32))


@pytest.mark.gpu
def test_write_subset():

    @dace.program
    def write_subset(A: dace.int32[20, 20]):
        for i, j in dace.map[2:18, 2:18]:
            A[i, j] = i + j

    sdfg = write_subset.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations(simplify=False)

    ref = np.ones((20, 20), dtype=np.int32)
    val = np.copy(ref)

    write_subset.f(ref)
    sdfg(A=val)

    assert np.array_equal(ref, val)


def test_a_fully_overwritten_array_is_not_staged_down_first():
    """An array nothing reads, that a map overwrites entirely, needs no host-to-device copy.

    Its entry value cannot be observed, so staging it down transfers the whole array on every call
    and then discards it. The copy-out still has to happen, which is what makes the elision safe
    only when the device writes ALL of it.
    """

    M, N = dace.symbol('M'), dace.symbol('N')

    @dace.program
    def write_full(A: dace.int32[M, N]):
        for i, j in dace.map[0:M, 0:N]:
            A[i, j] = i + j

    sdfg = write_full.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations(simplify=False)

    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.AccessNode) and node.data == 'A':
                assert state.out_degree(node) == 0, (f'"A" is read in {state.label!r}: the host array is '
                                                     'still staged down before the map overwrites it')
    assert any(
        isinstance(node, dace.nodes.AccessNode) and node.data == 'A' and state.in_degree(node) > 0
        for state in sdfg.states() for node in state.nodes()), 'the result never reaches the caller\'s array'


def test_a_partially_written_array_keeps_its_copy_in():
    """The control for the elision above: a map covering only part of the array must still be staged
    down, because the copy-out sends the whole device buffer back and the untouched elements have to
    be the ones the caller passed in, not whatever the allocation held."""

    M = dace.symbol('M')

    @dace.program
    def write_interior(A: dace.int32[M]):
        for i in dace.map[1:M - 1]:
            A[i] = i

    sdfg = write_interior.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations(simplify=False)

    assert any(
        isinstance(node, dace.nodes.AccessNode) and node.data == 'A' and state.out_degree(node) > 0
        for state in sdfg.states() for node in state.nodes()), ('a partially written array lost its copy-in; the '
                                                                'elements the map skips would come back as garbage')


def test_an_indirect_write_keeps_its_copy_in():
    """``A[x[i], y[j]]`` carries the WHOLE array as its subset -- that is where it might land -- while
    writing 256 of the 400 elements. A covering subset is therefore not proof the array is fully
    written, and the volume is what says so; without that second test the copy-in is dropped and the
    144 elements the scatter misses come back as whatever the allocation held."""

    @dace.program
    def write_subset_dynamic(A: dace.int32[20, 20], x: dace.int32[20], y: dace.int32[20]):
        for i, j in dace.map[2:18, 2:18]:
            A[x[i], y[j]] = i + j

    sdfg = write_subset_dynamic.to_sdfg(simplify=True)
    writes = [(e.data.subset, e.data.volume) for state in sdfg.states() for node in state.data_nodes()
              if node.data == 'A' for e in state.in_edges(node)]
    assert writes, 'no write to "A" to inspect'
    assert any(str(subset) == '0:20, 0:20' and volume != 400
               for subset, volume in writes), (f'the indirect write no longer over-approximates its subset: {writes}')

    sdfg.apply_gpu_transformations(simplify=False)
    assert any(
        isinstance(node, dace.nodes.AccessNode) and node.data == 'A' and state.out_degree(node) > 0
        for state in sdfg.states() for node in state.nodes()), 'the indirect write lost its copy-in'


@pytest.mark.gpu
def test_write_subset_dynamic():

    @dace.program
    def write_subset_dynamic(A: dace.int32[20, 20], x: dace.int32[20], y: dace.int32[20]):
        for i, j in dace.map[2:18, 2:18]:
            A[x[i], y[j]] = i + j

    sdfg = write_subset_dynamic.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations(simplify=False)

    ref = np.ones((20, 20), dtype=np.int32)
    val = np.copy(ref)

    x = np.random.permutation(20).astype(np.int32)
    y = np.random.permutation(20).astype(np.int32)

    write_subset_dynamic.f(ref, x, y)
    sdfg(A=val, x=x, y=y)

    assert np.array_equal(ref, val)


@pytest.mark.parametrize(["transient", "scalar"], [[False, False], [False, True], [True, False], [True, True]])
def test_free_tasklet(transient, scalar):
    sdfg = dace.SDFG("assign")

    state = sdfg.add_state("main")
    if scalar:
        arr_name, arr = sdfg.add_scalar("A", dace.float32, transient=transient)
    else:
        arr_name, arr = sdfg.add_array("A", (4, ), dace.float32, transient=transient)

    an = state.add_access(arr_name)

    t = state.add_tasklet("assign", {}, {"_out"}, "_out = 2.0")
    state.add_edge(t, "_out", an, None, dace.memlet.Memlet("A" if scalar else "A[0]"))

    sdfg.validate()

    sdfg.apply_gpu_transformations(validate=True, validate_all=True, simplify=False)

    sdfg.validate()


def test_free_tasklet_connectorless_dependency_edge():
    """A global-code tasklet with a connector-less (empty-memlet) dependency in-edge --
    e.g. an edge sequencing a reduction-init tasklet -- must wrap in the GPU_Device map
    without crashing. Pre-fix the connector rebuild did ``'IN_' + e.dst_conn`` and threw
    ``TypeError`` when ``dst_conn`` is None; the edge is now threaded through the map as a
    dependency edge with no IN_/OUT_ connector."""
    sdfg = dace.SDFG("gcode_depedge")
    arr_name, _ = sdfg.add_array("A", (4, ), dace.float32, transient=False)
    state = sdfg.add_state("main")

    seed = state.add_tasklet("seed", {}, {"_o"}, "_o = 0.0")
    state.add_edge(seed, "_o", state.add_access(arr_name), None, dace.memlet.Memlet("A[0]"))

    follow = state.add_tasklet("follow", {}, {"_o2"}, "_o2 = 1.0")
    state.add_edge(follow, "_o2", state.add_access(arr_name), None, dace.memlet.Memlet("A[1]"))

    # Connector-less empty-memlet dependency edge: seed must run before follow.
    state.add_nedge(seed, follow, dace.memlet.Memlet())

    sdfg.validate()
    sdfg.apply_gpu_transformations(validate=True, validate_all=True, simplify=False)
    sdfg.validate()


if __name__ == '__main__':
    test_toplevel_transient_lifetime()
    test_scalar_to_symbol_in_nested_sdfg()
    test_write_subset()
    test_a_fully_overwritten_array_is_not_staged_down_first()
    test_a_partially_written_array_keeps_its_copy_in()
    test_an_indirect_write_keeps_its_copy_in()
    test_write_subset_dynamic()
    test_free_tasklet_connectorless_dependency_edge()
    for scalar in [False, True]:
        for transient in [False, True]:
            test_free_tasklet(transient, scalar)
