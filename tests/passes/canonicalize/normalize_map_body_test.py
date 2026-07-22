# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :class:`NormalizeMapBody`.

The pass consolidates a map body that mixes control flow with siblings into a
single NestedSDFG (sequencing the siblings), so downstream ``ConditionFusion``
can fold same-condition guards that MapFusion left in separate nested SDFGs.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.canonicalize.normalize_map_body import NormalizeMapBody

N = dace.symbol('N')


@dace.program
def _two_guarded_loops_idx(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in range(N):
        if i % 2 == 0:
            b[i] = a[i] + 1.0
    for i in range(N):
        if i % 2 == 0:
            c[i] = a[i] * 2.0


@dace.program
def _two_plain_loops(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in range(N):
        b[i] = a[i] + 1.0
    for i in range(N):
        c[i] = a[i] * 2.0


def _body_nsdfg_count(sdfg: dace.SDFG) -> int:
    """Max number of NestedSDFG nodes inside any single top-level map body."""
    counts = [0]
    for st in sdfg.all_states():
        for n in st.nodes():
            if isinstance(n, nodes.MapEntry) and st.entry_node(n) is None:
                counts.append(
                    sum(1 for x in st.all_nodes_between(n, st.exit_node(n)) if isinstance(x, nodes.NestedSDFG)))
    return max(counts)


def _num_condblocks(sdfg: dace.SDFG) -> int:
    return sum(1 for cb in sdfg.all_control_flow_regions(recursive=True) if isinstance(cb, ConditionalBlock))


def test_merges_two_sibling_nsdfgs_into_one():
    """A fused map body with two guarded nested SDFGs -> one nested SDFG whose
    CFG sequences both; ConditionFusion then folds the two guards; valid + exact."""
    # Canonicalize up to (but not through) the map-body normalization by using
    # the standalone pass on the already-fused two-guarded-map form.
    sdfg = _two_guarded_loops_idx.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    # The pipeline now already normalizes+merges, so the end state has one guard;
    # re-running the pass must be a safe no-op, and the merged form must be valid.
    assert _num_condblocks(sdfg) == 1, "canon must leave a single merged in-map guard"
    assert _body_nsdfg_count(sdfg) <= 1, "the map body must be a single nested SDFG"
    assert NormalizeMapBody().apply_pass(sdfg, {}) is None, "re-run must be a no-op on the normalized form"
    sdfg.validate()

    n = 16
    rng = np.random.default_rng(0)
    a = rng.random(n)
    b, c = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), b=b, c=c, N=n)
    even = np.arange(n) % 2 == 0
    assert np.allclose(b, np.where(even, a + 1.0, 0.0))
    assert np.allclose(c, np.where(even, a * 2.0, 0.0))


def test_all_tasklet_body_untouched():
    """A map body with no nested SDFG (all tasklets) is left unchanged."""
    sdfg = _two_plain_loops.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    before = _body_nsdfg_count(sdfg)
    assert NormalizeMapBody().apply_pass(sdfg, {}) is None, "no control flow -> no-op"
    assert _body_nsdfg_count(sdfg) == before
    sdfg.validate()

    n = 16
    rng = np.random.default_rng(1)
    a = rng.random(n)
    b, c = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), b=b, c=c, N=n)
    assert np.allclose(b, a + 1.0)
    assert np.allclose(c, a * 2.0)


def _sibling_nsdfg(label: str, with_tmp_conn: bool, with_tmp_array: bool):
    """A one-state nested SDFG: reads scalar ``x`` -> writes scalar ``o``. Optionally the
    writing tasklet's out-connector is named ``tmp`` (``with_tmp_conn``) or an inner
    transient scalar is named ``tmp`` (``with_tmp_array``)."""
    nsdfg = dace.SDFG(label)
    nsdfg.add_scalar('x', dace.float64)
    nsdfg.add_scalar('o', dace.float64)
    st = nsdfg.add_state('st_' + label)
    conn = 'tmp' if with_tmp_conn else 'r'
    rd = st.add_access('x')
    if with_tmp_array:
        nsdfg.add_scalar('tmp', dace.float64, transient=True)
        t = st.add_tasklet(label + '_t', {'a'}, {conn}, f'{conn} = a + 1.0')
        mid = st.add_access('tmp')
        st.add_edge(rd, None, t, 'a', dace.Memlet('x[0]'))
        st.add_edge(t, conn, mid, None, dace.Memlet('tmp[0]'))
        st.add_edge(mid, None, st.add_access('o'), None, dace.Memlet('o[0]'))
    else:
        t = st.add_tasklet(label + '_t', {'a'}, {conn}, f'{conn} = a + 2.0')
        st.add_edge(rd, None, t, 'a', dace.Memlet('x[0]'))
        st.add_edge(t, conn, st.add_access('o'), None, dace.Memlet('o[0]'))
    return nsdfg


def test_merge_siblings_data_vs_connector_name_collision():
    """``_merge_siblings`` must uniquify a tail array against base's tasklet connector
    names, not only its array names: base sibling writes through a connector ``tmp``,
    tail sibling owns an array ``tmp`` -- merging the array in unchecked collides with the
    connector (``'tmp' already used as ... array name``). The tail array must be renamed."""
    sdfg = dace.SDFG('merge_conn_collision')
    for arr in ('X', 'A', 'B'):
        sdfg.add_array(arr, [N], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map('m', {'i': '0:N'})
    # sibling A: tasklet out-connector named 'tmp'; sibling B: inner array named 'tmp'.
    sibA = _sibling_nsdfg('sibA', with_tmp_conn=True, with_tmp_array=False)
    sibB = _sibling_nsdfg('sibB', with_tmp_conn=False, with_tmp_array=True)
    nA = state.add_nested_sdfg(sibA, {'x'}, {'o'})
    nB = state.add_nested_sdfg(sibB, {'x'}, {'o'})
    rd = state.add_read('X')
    state.add_memlet_path(rd, me, nA, dst_conn='x', memlet=dace.Memlet('X[i]'))
    state.add_memlet_path(rd, me, nB, dst_conn='x', memlet=dace.Memlet('X[i]'))
    state.add_memlet_path(nA, mx, state.add_write('A'), src_conn='o', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(nB, mx, state.add_write('B'), src_conn='o', memlet=dace.Memlet('B[i]'))
    sdfg.validate()

    assert NormalizeMapBody().apply_pass(sdfg, {}) is not None, 'the two siblings should merge'
    sdfg.validate()  # would raise the connector/array-name collision without the fix
    merged = [n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, nodes.NestedSDFG)]
    assert len(merged) == 1, 'siblings merged into one nested SDFG'

    n = 8
    rng = np.random.default_rng(0)
    X = rng.random(n)
    A, B = np.zeros(n), np.zeros(n)
    sdfg(X=X.copy(), A=A, B=B, N=n)
    assert np.allclose(A, X + 2.0) and np.allclose(B, X + 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
