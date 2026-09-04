# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the multi-output extension of the ``SplitTasklets`` pass.

A tasklet whose body is a list of straight-line ``target = expr`` assignments -- one per
output connector -- is split into one single-output tasklet per output connector, so
downstream loop-fission / lift passes can treat each write independently. A cross-statement
read of an earlier output connector (a RAW) is materialised through that output's array via
an intermediate access node, preserving the producer-before-consumer order as a real data
dependence.

The split is refused (the tasklet left intact) when it cannot be proven sound:

- an **in-place read-modify-write** -- an input connector reads an array that an output
  connector also writes (polybench covariance's finalize normalises ``cov[i, j]`` in place
  and mirrors it to ``cov[j, i]``, with ``cov[i, j]`` fed by an upstream WCR reduction).
  Splitting that shape corrupts the read / normalize / mirror ordering and NaNs, so it is
  left intact -- the pre-existing behaviour that keeps covariance correct;
- a shared local temp target, an unordered same-array WAW, or a non-straight-line body.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import networkx as nx
import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.split_tasklets import SplitTasklets

M = dace.symbol('M')
N = dace.symbol('N')


def _tasklets(sdfg):
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)]


def _multi_output_tasklets(sdfg):
    return [t for t in _tasklets(sdfg) if len(t.out_connectors) > 1]


def _reads_array(sdfg, out_conn, array):
    """True iff the single-output tasklet producing ``out_conn`` has an input edge sourced
    from an access node of ``array`` -- i.e. the cross-statement read is materialised
    through that array rather than a private scalar transient."""
    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.Tasklet) and out_conn in n.out_connectors:
            for ie in g.in_edges(n):
                if isinstance(ie.src, dace.nodes.AccessNode) and ie.src.data == array:
                    return True
    return False


# ------------------------------------------------------------------------------------------
# Covariance-shape WAW (two outputs write the same array), driven by nested loops over i < j.
# The input is a separate array A (no in-place read/write of the output array), so the split
# is allowed; the destination is an access node (a loop-body state), so the mirror routes
# through a direct same-array copy and exactly two single-output tasklets remain.
# ------------------------------------------------------------------------------------------
def _build_covariance_loops():
    sdfg = dace.SDFG('mo_cov_loops')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [M, M], dace.float64)
    sdfg.add_array('X', [M, M], dace.float64)
    li = LoopRegion('Li', 'i < M - 1', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(li, is_start_block=True)
    lj = LoopRegion('Lj', 'j < M', 'j', 'j = i + 1', 'j = j + 1')
    li.add_node(lj, is_start_block=True)
    body = lj.add_state('body', is_start_block=True)
    a_read = body.add_read('A')
    x_write = body.add_write('X')
    tasklet = body.add_tasklet('cov', {'a_in'}, {'cov_ij_out', 'cov_ji_out'},
                               'cov_ij_out = a_in / (N - 1)\ncov_ji_out = cov_ij_out')
    body.add_edge(a_read, None, tasklet, 'a_in', dace.Memlet('A[i, j]'))
    body.add_edge(tasklet, 'cov_ij_out', x_write, None, dace.Memlet('X[i, j]'))
    body.add_edge(tasklet, 'cov_ji_out', x_write, None, dace.Memlet('X[j, i]'))
    sdfg.validate()
    return sdfg


def test_covariance_two_output_split_and_bitexact():
    """The fused normalize+mirror covariance tasklet splits into two single-output
    tasklets, the mirror's read is routed through the array ``X``, and the whole loop nest
    is bit-exact with a numpy reference. ``N - 1 == 8`` is a power of two, so the division
    is reproduced exactly under the default ``-ffast-math`` reciprocal-multiply."""
    sdfg = _build_covariance_loops()
    assert len(_multi_output_tasklets(sdfg)) == 1

    SplitTasklets().apply_pass(sdfg=sdfg, pipeline_results={})
    sdfg.validate()

    tks = _tasklets(sdfg)
    assert len(tks) == 2, f'expected two single-output tasklets, got {len(tks)}'
    assert all(len(t.out_connectors) == 1 for t in tks), 'every split tasklet must have a single output'
    assert _reads_array(sdfg, 'cov_ji_out', 'X'), 'the mirror RAW must be materialised through the array X'

    n = 9  # N - 1 == 8
    for m in (5, 8):
        rng = np.random.default_rng(m)
        A = rng.standard_normal((m, m))
        X = np.zeros((m, m))
        sdfg(A=A, X=X, M=m, N=n)
        ref = np.zeros((m, m))
        for i in range(m - 1):
            for j in range(i + 1, m):
                ref[i, j] = A[i, j] / (n - 1)
                ref[j, i] = ref[i, j]
        assert np.array_equal(X, ref), f'covariance split not bit-exact for m={m}'


# ------------------------------------------------------------------------------------------
# Distinct output arrays with a RAW: b[i] = a[i] * 2; c[i] = b[i] + 1.
# ------------------------------------------------------------------------------------------
def _build_distinct_loop():
    sdfg = dace.SDFG('mo_distinct_loop')
    for nm in ('a', 'b', 'c'):
        sdfg.add_array(nm, [M], dace.float64)
    li = LoopRegion('Li', 'i < M', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(li, is_start_block=True)
    body = li.add_state('body', is_start_block=True)
    a_read = body.add_read('a')
    tasklet = body.add_tasklet('t', {'a_in'}, {'b_out', 'c_out'}, 'b_out = a_in * 2.0\nc_out = b_out + 1.0')
    body.add_edge(a_read, None, tasklet, 'a_in', dace.Memlet('a[i]'))
    body.add_edge(tasklet, 'b_out', body.add_write('b'), None, dace.Memlet('b[i]'))
    body.add_edge(tasklet, 'c_out', body.add_write('c'), None, dace.Memlet('c[i]'))
    sdfg.validate()
    return sdfg


def test_distinct_outputs_raw_routes_through_b():
    """Two independent output arrays with a cross-statement read: the split produces two
    single-output tasklets and the second reads ``b`` back through an access node of ``b``
    (not a private scalar), so a later loop-fission still sees the dependence."""
    sdfg = _build_distinct_loop()
    SplitTasklets().apply_pass(sdfg=sdfg, pipeline_results={})
    sdfg.validate()

    tks = _tasklets(sdfg)
    assert len(tks) == 2
    assert all(len(t.out_connectors) == 1 for t in tks)
    assert _reads_array(sdfg, 'c_out', 'b'), 'the RAW (c = b + 1) must be routed through the array b'

    m = 7
    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, ))
    b = np.zeros((m, ))
    c = np.zeros((m, ))
    sdfg(a=a, b=b, c=c, M=m)
    assert np.array_equal(b, a * 2.0)
    assert np.array_equal(c, a * 2.0 + 1.0)


# ------------------------------------------------------------------------------------------
# A three-statement transitive chain over distinct arrays.
# ------------------------------------------------------------------------------------------
def _build_three_output_chain_loop():
    sdfg = dace.SDFG('mo_chain3_loop')
    for nm in ('a', 'b', 'c', 'd'):
        sdfg.add_array(nm, [M], dace.float64)
    li = LoopRegion('Li', 'i < M', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(li, is_start_block=True)
    body = li.add_state('body', is_start_block=True)
    a_read = body.add_read('a')
    tasklet = body.add_tasklet('t', {'a_in'}, {'o1', 'o2', 'o3'}, 'o1 = a_in * 2.0\no2 = o1 + 1.0\no3 = o2 * 3.0')
    body.add_edge(a_read, None, tasklet, 'a_in', dace.Memlet('a[i]'))
    for out_conn, arr in (('o1', 'b'), ('o2', 'c'), ('o3', 'd')):
        body.add_edge(tasklet, out_conn, body.add_write(arr), None, dace.Memlet(f'{arr}[i]'))
    sdfg.validate()
    return sdfg


def test_three_output_chain_split_and_bitexact():
    """A three-statement chain splits into three single-output tasklets whose RAWs are
    routed through the produced arrays, and stays bit-exact."""
    sdfg = _build_three_output_chain_loop()
    SplitTasklets().apply_pass(sdfg=sdfg, pipeline_results={})
    sdfg.validate()
    tks = _tasklets(sdfg)
    assert len(tks) == 3 and all(len(t.out_connectors) == 1 for t in tks)
    assert _reads_array(sdfg, 'o2', 'b')
    assert _reads_array(sdfg, 'o3', 'c')

    m = 7
    rng = np.random.default_rng(2)
    a = rng.standard_normal((m, ))
    b = np.zeros((m, ))
    c = np.zeros((m, ))
    d = np.zeros((m, ))
    sdfg(a=a, b=b, c=c, d=d, M=m)
    assert np.array_equal(b, a * 2.0)
    assert np.array_equal(c, a * 2.0 + 1.0)
    assert np.array_equal(d, (a * 2.0 + 1.0) * 3.0)


# ------------------------------------------------------------------------------------------
# Map scope: the mirror must exit through the map exit, so the produced value is copied out
# by a scalar store tasklet (a same-array ``-> MapExit`` copy is not expressible directly).
# ------------------------------------------------------------------------------------------
def _build_covariance_maps():
    sdfg = dace.SDFG('mo_cov_maps')
    sdfg.add_array('A', [M, M], dace.float64)
    sdfg.add_array('X', [M, M], dace.float64)
    state = sdfg.add_state()
    ome, omx = state.add_map('outer', dict(i='0:M-1'))
    ime, imx = state.add_map('inner', dict(j='i+1:M'))
    a_read = state.add_read('A')
    x_write = state.add_write('X')
    tasklet = state.add_tasklet('cov', {'a_in'}, {'cov_ij_out', 'cov_ji_out'},
                                'cov_ij_out = a_in / 8.0\ncov_ji_out = cov_ij_out')
    state.add_memlet_path(a_read, ome, ime, tasklet, dst_conn='a_in', memlet=dace.Memlet('A[i, j]'))
    state.add_memlet_path(tasklet, imx, omx, x_write, src_conn='cov_ij_out', memlet=dace.Memlet('X[i, j]'))
    state.add_memlet_path(tasklet, imx, omx, x_write, src_conn='cov_ji_out', memlet=dace.Memlet('X[j, i]'))
    sdfg.validate()
    return sdfg


def test_map_scope_split_uses_store_tasklet_and_bitexact():
    """Inside a map the mirror routes through an access node of ``X`` and the produced
    value is copied out via a scalar store tasklet (so the map's write set stays complete).
    Two compute tasklets plus one store tasklet remain, all single-output, and the result
    is bit-exact."""
    sdfg = _build_covariance_maps()
    SplitTasklets().apply_pass(sdfg=sdfg, pipeline_results={})
    sdfg.validate()
    tks = _tasklets(sdfg)
    assert len(tks) == 3, f'two compute + one store tasklet expected, got {len(tks)}'
    assert all(len(t.out_connectors) == 1 for t in tks)
    assert _reads_array(sdfg, 'cov_ji_out', 'X')

    m = 6
    rng = np.random.default_rng(3)
    A = rng.standard_normal((m, m))
    X = np.zeros((m, m))
    sdfg(A=A, X=X, M=m)
    ref = np.zeros((m, m))
    for i in range(m - 1):
        for j in range(i + 1, m):
            ref[i, j] = A[i, j] / 8.0
            ref[j, i] = ref[i, j]
    assert np.array_equal(X, ref)


# ------------------------------------------------------------------------------------------
# In-place read-modify-write (the covariance finalize shape): the input reads the same array
# the outputs write. The split is refused and the tasklet is left intact.
# ------------------------------------------------------------------------------------------
def _build_inplace_rmw():
    sdfg = dace.SDFG('mo_inplace_rmw')
    sdfg.add_array('X', [M, M], dace.float64)
    state = sdfg.add_state()
    ome, omx = state.add_map('outer', dict(i='0:M-1'))
    ime, imx = state.add_map('inner', dict(j='i+1:M'))
    x_read = state.add_read('X')
    x_write = state.add_write('X')
    tasklet = state.add_tasklet('cov', {'x_in'}, {'o_ij', 'o_ji'}, 'o_ij = x_in / 8.0\no_ji = o_ij')
    state.add_memlet_path(x_read, ome, ime, tasklet, dst_conn='x_in', memlet=dace.Memlet('X[i, j]'))
    state.add_memlet_path(tasklet, imx, omx, x_write, src_conn='o_ij', memlet=dace.Memlet('X[i, j]'))
    state.add_memlet_path(tasklet, imx, omx, x_write, src_conn='o_ji', memlet=dace.Memlet('X[j, i]'))
    sdfg.validate()
    return sdfg


def test_inplace_read_modify_write_refused():
    """An input connector reading the same array an output writes (``x_in << X[i,j]`` and
    ``o_ij >> X[i,j]``) is left intact -- this is the polybench covariance finalize shape,
    which NaNs if split. The tasklet stays a single two-output tasklet and runs correctly."""
    sdfg = _build_inplace_rmw()
    SplitTasklets().apply_pass(sdfg=sdfg, pipeline_results={})
    sdfg.validate()
    assert len(_multi_output_tasklets(sdfg)) == 1, 'the in-place RMW tasklet must not be split'
    assert len(_tasklets(sdfg)) == 1

    m = 6
    rng = np.random.default_rng(5)
    X = rng.standard_normal((m, m))
    x_inout = X.copy()
    sdfg(X=x_inout, M=m)
    ref = X.copy()
    for i in range(m - 1):
        for j in range(i + 1, m):
            ref[i, j] = X[i, j] / 8.0
            ref[j, i] = ref[i, j]
    assert np.array_equal(x_inout, ref)


# ------------------------------------------------------------------------------------------
# Regression: the real polybench covariance kernel (in-place finalize fed by a WCR reduction)
# must be value-correct after canonicalize -- no NaN.
# ------------------------------------------------------------------------------------------
@dace.program
def _covariance_kernel(data: dace.float64[N, M], cov: dace.float64[M, M], mean: dace.float64[M]):
    mean[:] = 0.0

    @dace.map
    def comp_mean(j: _[0:M], i: _[0:N]):
        inp << data[i, j]
        out >> mean(1, lambda x, y: x + y)[j]
        out = inp

    @dace.map
    def comp_mean2(j: _[0:M]):
        inp << mean[j]
        out >> mean[j]
        out = inp / N

    @dace.map
    def sub_mean(i: _[0:N], j: _[0:M]):
        ind << data[i, j]
        me << mean[j]
        oud >> data[i, j]
        oud = ind - me

    @dace.mapscope
    def comp_cov_row(i: _[0:M]):

        @dace.mapscope
        def comp_cov_col(j: _[i:M]):
            with dace.tasklet:
                cov_ij >> cov[i, j]
                cov_ij = 0.0

            @dace.map
            def comp_cov_k(k: _[0:N]):
                indi << data[k, i]
                indj << data[k, j]
                cov_ij >> cov(1, lambda x, y: x + y)[i, j]
                cov_ij = (indi * indj)

            with dace.tasklet:
                cov_ij_in << cov[i, j]
                cov_ij_out >> cov[i, j]
                cov_ji_out >> cov[j, i]
                cov_ij_out = cov_ij_in / (N - 1)
                cov_ji_out = cov_ij_out


def _covariance_reference(data, n, m):
    d = data.copy()
    centered = d - d.mean(axis=0)
    cov = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            cov[i, j] = np.sum(centered[:, i] * centered[:, j]) / (n - 1)
            cov[j, i] = cov[i, j]
    return cov


def test_real_covariance_canonicalize_no_nan():
    """The full covariance kernel -- an in-place finalize fed by a WCR sum-reduction, inside
    nested map scopes wrapped in a NestedSDFG -- must canonicalize (the finalize is left
    intact by the in-place refusal) and stay value-correct with no NaN."""
    from dace.transformation.passes.canonicalize.pipeline import canonicalize

    n, m = 32, 28
    rng = np.random.default_rng(11)
    data = rng.standard_normal((n, m))

    sdfg = _covariance_kernel.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    # The in-place finalize must have been left intact (not split into a NaN-producing form).
    assert len(_multi_output_tasklets(sdfg)) >= 1, 'the in-place covariance finalize must be left intact'

    d = data.copy()
    cov = np.zeros((m, m))
    mean = np.zeros((m, ))
    sdfg(data=d, cov=cov, mean=mean, M=m, N=n)
    assert not np.isnan(cov).any(), 'covariance produced NaN'
    assert np.max(np.abs(cov - _covariance_reference(data, n, m))) < 1e-12


# ------------------------------------------------------------------------------------------
# Unsafe WAR (a later statement writes an array an earlier statement reads in place).
# ------------------------------------------------------------------------------------------
def _build_war_unsafe():
    sdfg = dace.SDFG('mo_war_unsafe')
    for nm in ('X', 'A', 'Y'):
        sdfg.add_array(nm, [M], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(i='0:M'))
    x_read = state.add_read('X')
    a_read = state.add_read('A')
    tasklet = state.add_tasklet('t', {'x_in', 'a_in'}, {'y_out', 'x_out'}, 'y_out = x_in\nx_out = a_in')
    state.add_memlet_path(x_read, me, tasklet, dst_conn='x_in', memlet=dace.Memlet('X[i]'))
    state.add_memlet_path(a_read, me, tasklet, dst_conn='a_in', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(tasklet, mx, state.add_write('Y'), src_conn='y_out', memlet=dace.Memlet('Y[i]'))
    state.add_memlet_path(tasklet, mx, state.add_write('X'), src_conn='x_out', memlet=dace.Memlet('X[i]'))
    sdfg.validate()
    return sdfg


def test_war_unsafe_split_refused():
    """A tasklet that reads ``X`` (entry value) and writes ``X`` is an in-place read/write of
    the same array; it is left intact and still computes the original semantics (Y == old X,
    X == A)."""
    sdfg = _build_war_unsafe()
    SplitTasklets().apply_pass(sdfg=sdfg, pipeline_results={})
    sdfg.validate()
    assert len(_multi_output_tasklets(sdfg)) == 1, 'the unsafe WAR/in-place tasklet must not be split'

    m = 6
    rng = np.random.default_rng(1)
    X = rng.standard_normal((m, ))
    A = rng.standard_normal((m, ))
    Y = np.zeros((m, ))
    x_inout = X.copy()
    sdfg(X=x_inout, A=A, Y=Y, M=m)
    assert np.array_equal(Y, X), 'Y must capture the entry value of X'
    assert np.array_equal(x_inout, A), 'X must be overwritten by A'


# ------------------------------------------------------------------------------------------
# Independent outputs in a top-level state must not be left as disconnected components.
# ------------------------------------------------------------------------------------------
def _build_independent_outputs():
    sdfg = dace.SDFG('mo_independent')
    for nm in ('a', 'b', 'c', 'd'):
        sdfg.add_array(nm, [M], dace.float64)
    sdfg.add_symbol('I', dace.int64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet('t', {'a_in', 'b_in'}, {'o1', 'o2'}, 'o1 = a_in\no2 = b_in')
    state.add_edge(state.add_read('a'), None, tasklet, 'a_in', dace.Memlet('a[I]'))
    state.add_edge(state.add_read('b'), None, tasklet, 'b_in', dace.Memlet('b[I]'))
    state.add_edge(tasklet, 'o1', state.add_write('c'), None, dace.Memlet('c[I]'))
    state.add_edge(tasklet, 'o2', state.add_write('d'), None, dace.Memlet('d[I]'))
    sdfg.validate()
    return sdfg


def test_independent_outputs_connected_and_correct():
    """Independent outputs must not be left as disconnected components: after the split the
    state is a single weakly-connected component (a sequencing edge chains the two output
    tasklets in body order), and the result is still value-correct."""
    sdfg = _build_independent_outputs()
    SplitTasklets().apply_pass(sdfg=sdfg, pipeline_results={})
    sdfg.validate()
    tks = _tasklets(sdfg)
    assert len(tks) == 2 and all(len(t.out_connectors) == 1 for t in tks)

    state = next(g for n, g in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))
    assert nx.number_weakly_connected_components(state.nx) == 1, 'split left disconnected components'
    assert nx.has_path(state.nx.to_undirected(as_view=True), tks[0], tks[1]), 'output tasklets are not sequenced'

    m = 5
    rng = np.random.default_rng(7)
    a = rng.standard_normal((m, ))
    b = rng.standard_normal((m, ))
    c = np.zeros((m, ))
    d = np.zeros((m, ))
    compiled = sdfg.compile()
    for i in range(m):
        compiled(a=a, b=b, c=c, d=d, M=m, I=i)
    assert np.array_equal(c, a)
    assert np.array_equal(d, b)


# ------------------------------------------------------------------------------------------
# A statement targeting a non-output-connector local temp is refused.
# ------------------------------------------------------------------------------------------
def _build_shared_temp():
    sdfg = dace.SDFG('mo_shared_temp')
    for nm in ('a', 'b', 'c'):
        sdfg.add_array(nm, [M], dace.float64)
    li = LoopRegion('Li', 'i < M', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(li, is_start_block=True)
    body = li.add_state('body', is_start_block=True)
    a_read = body.add_read('a')
    tasklet = body.add_tasklet('t', {'a_in'}, {'o1', 'o2'}, 'o1 = a_in * 2.0\ntmp = o1 + 1.0\no2 = tmp')
    body.add_edge(a_read, None, tasklet, 'a_in', dace.Memlet('a[i]'))
    body.add_edge(tasklet, 'o1', body.add_write('b'), None, dace.Memlet('b[i]'))
    body.add_edge(tasklet, 'o2', body.add_write('c'), None, dace.Memlet('c[i]'))
    sdfg.validate()
    return sdfg


def test_shared_local_temp_refused():
    """A statement targeting a non-output-connector local temp is a precondition failure:
    the tasklet is left intact and still runs correctly."""
    sdfg = _build_shared_temp()
    SplitTasklets().apply_pass(sdfg=sdfg, pipeline_results={})
    sdfg.validate()
    assert len(_multi_output_tasklets(sdfg)) == 1

    m = 5
    rng = np.random.default_rng(6)
    a = rng.standard_normal((m, ))
    b = np.zeros((m, ))
    c = np.zeros((m, ))
    sdfg(a=a, b=b, c=c, M=m)
    assert np.array_equal(b, a * 2.0)
    assert np.array_equal(c, a * 2.0 + 1.0)


if __name__ == '__main__':
    test_covariance_two_output_split_and_bitexact()
    test_distinct_outputs_raw_routes_through_b()
    test_three_output_chain_split_and_bitexact()
    test_map_scope_split_uses_store_tasklet_and_bitexact()
    test_inplace_read_modify_write_refused()
    test_real_covariance_canonicalize_no_nan()
    test_war_unsafe_split_refused()
    test_independent_outputs_connected_and_correct()
    test_shared_local_temp_refused()
    print('OK')
