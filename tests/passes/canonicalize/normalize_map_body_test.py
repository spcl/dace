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
from dace.transformation.dataflow.map_fusion_horizontal import MapFusionHorizontal
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.canonicalize.normalize_map_body import NormalizeMapBody
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated

N = dace.symbol('N')
M = dace.symbol('M')


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


@dace.program
def two_level_siblings(a: dace.float64[N, M], b: dace.float64[N, M], c: dace.float64[N, M]):
    for i in range(N):
        for j in range(M):
            b[i, j] = a[i, j] + 1.0
    for i in range(N):
        for j in range(M):
            c[i, j] = a[i, j] * 2.0


def two_level_sibling_sdfg() -> dace.SDFG:
    """Two independent doubly-nested loops (writing ``b``/``c`` from ``a``), taken through real
    transformations -- NOT hand-built -- to the exact shape ``correlation``'s two-level
    ``LoopToMap`` nesting plus sibling merge hits: ``LoopToMap`` converts each branch's outer AND
    inner loop (2 applications per branch, 4 total), so each branch is its own
    ``Map{NestedSDFG{Map{NestedSDFG}}}`` (3 SDFG levels). ``StateFusionExtended`` then co-locates
    the two branches' states, and ``MapFusionHorizontal`` fuses their outer maps into ONE top-level
    map holding the two branches as SIBLING NestedSDFGs -- each sibling still owning its own
    further-nested Map+NestedSDFG underneath."""
    sdfg = two_level_siblings.to_sdfg(simplify=True)
    sdfg.simplify()
    PatternMatchAndApplyRepeated([LoopToMap()]).apply_pass(sdfg, {})
    PatternMatchAndApplyRepeated([StateFusionExtended()]).apply_pass(sdfg, {})
    PatternMatchAndApplyRepeated([MapFusionHorizontal()]).apply_pass(sdfg, {})
    return sdfg


def sdfg_backpointer_violations(sdfg: dace.SDFG):
    """Every state under ``sdfg`` (through its OWN control-flow regions, never through a
    ``NestedSDFG`` codenode) must have ``state.sdfg is sdfg``; recurse into each NestedSDFG's own
    SDFG separately. Returns the ``(sdfg label, state label)`` pairs that violate this."""
    violations = []
    for state in sdfg.all_states():
        if state.sdfg is not sdfg:
            violations.append((sdfg.label, state.label))
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                violations += sdfg_backpointer_violations(node.sdfg)
    return violations


def test_merge_siblings_preserves_deep_nested_state_backpointers():
    """``_merge_siblings`` must re-point only the blocks ``base`` actually owns.

    Before the fix, its cleanup loop swept ``all_control_flow_blocks(recursive=True)`` -- which
    descends THROUGH NestedSDFG codenodes -- and stamped ``base`` onto states owned by the
    sibling's OWN further-nested SDFG (a third level down), even though that deeper SDFG has its
    own private arrays ``base`` does not. Both the direct backpointer invariant and
    ``sdfg.validate()`` must hold after the merge, and the merged SDFG must still run exact."""
    sdfg = two_level_sibling_sdfg()
    assert NormalizeMapBody().apply_pass(sdfg, {}) == 1, "the two top-level siblings must merge"
    assert sdfg_backpointer_violations(sdfg) == [], "a state's .sdfg must be its immediate owning SDFG"
    sdfg.validate()

    n, m = 6, 5
    rng = np.random.default_rng(0)
    a = rng.random((n, m))
    b, c = np.zeros((n, m)), np.zeros((n, m))
    sdfg(a=a.copy(), b=b, c=c, N=n, M=m)
    assert np.allclose(b, a + 1.0) and np.allclose(c, a * 2.0)


def _scalar_body(name: str, in_conn: str, out_conn: str, code: str) -> dace.SDFG:
    """A one-state nested SDFG computing ``out_conn = f(in_conn)`` on scalars."""
    inner = dace.SDFG(name)
    inner.add_array(in_conn, [1], dace.float64)
    inner.add_array(out_conn, [1], dace.float64)
    st = inner.add_state('body', is_start_block=True)
    t = st.add_tasklet('t', {'i'}, {'o'}, code)
    st.add_edge(st.add_read(in_conn), None, t, 'i', dace.Memlet(f'{in_conn}[0]'))
    st.add_edge(t, 'o', st.add_write(out_conn), None, dace.Memlet(f'{out_conn}[0]'))
    return inner


def _producer_consumer_siblings() -> dace.SDFG:
    """A map body of two nested SDFGs wired producer -> carrier -> consumer.

    The shape ``MapFusion`` leaves behind when the computations it co-locates are data-dependent.
    Merging them makes the carrier both a predecessor and a successor of the single surviving
    node, which no state can express.
    """
    sdfg = dace.SDFG('sibling_carrier')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('c', [N], dace.float64)
    sdfg.add_scalar('carrier', dace.float64, transient=True)

    state = sdfg.add_state('main', is_start_block=True)
    me, mx = state.add_map('m', dict(i='0:N'))
    producer = state.add_nested_sdfg(_scalar_body('produce', 'x', 'y', 'o = i + 1.0'), {'x'}, {'y'})
    consumer = state.add_nested_sdfg(_scalar_body('consume', 'y', 'z', 'o = i * 2.0'), {'y'}, {'z'})
    carrier = state.add_access('carrier')

    state.add_memlet_path(state.add_read('a'), me, producer, dst_conn='x', memlet=dace.Memlet('a[i]'))
    state.add_edge(producer, 'y', carrier, None, dace.Memlet('carrier[0]'))
    state.add_edge(carrier, None, consumer, 'y', dace.Memlet('carrier[0]'))
    state.add_memlet_path(consumer, mx, state.add_write('c'), src_conn='z', memlet=dace.Memlet('c[i]'))
    return sdfg


def test_data_dependent_siblings_merge_without_a_cycle():
    """The carrier's dependence must move INSIDE the merged nested SDFG.

    Re-adding the consumer's boundary edge after the merge points the carrier node at the very
    node that produces it -- ``State should be acyclic but contains cycles``. ``_append_cfg``
    already sequences the two bodies, so binding them to one inner array carries the value across
    that ordering edge instead.
    """
    sdfg = _producer_consumer_siblings()
    sdfg.validate()

    assert NormalizeMapBody().apply_pass(sdfg, {}) == 1, 'the two siblings must merge'
    sdfg.validate()
    state = sdfg.states()[0]
    nested = [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG)]
    assert len(nested) == 1, f'the map body must be a single nested SDFG, got {len(nested)}'
    merged = nested[0]
    assert not (set(state.predecessors(merged)) & set(state.successors(merged))), (
        'a node may not both produce and consume the same access node')
    assert not [n for n in state.data_nodes() if n.data == 'carrier'
                ], ('the carrier is produced and consumed inside the merged nested SDFG: its outer node is dead')
    me = [n for n in state.nodes() if isinstance(n, nodes.MapEntry)][0]
    assert merged in state.all_nodes_between(
        me,
        state.exit_node(me)), ('a sink the map exit cannot reach makes the whole body invisible to all_nodes_between')

    n = 16
    rng = np.random.default_rng(0)
    a = rng.random(n)
    c = np.zeros(n)
    sdfg(a=a, c=c, N=n)
    assert np.allclose(c, (a + 1.0) * 2.0), f'got {c[:3]}'


def _inner_read_write() -> dace.SDFG:
    """Sibling body ``b = a + 1``."""
    inner = dace.SDFG('inner_rw')
    inner.add_array('a', [1], dace.float64)
    inner.add_array('b', [1], dace.float64)
    st = inner.add_state()
    tasklet = st.add_tasklet('add', {'__i'}, {'__o'}, '__o = __i + 1.0')
    st.add_edge(st.add_read('a'), None, tasklet, '__i', dace.Memlet('a[0]'))
    st.add_edge(tasklet, '__o', st.add_write('b'), None, dace.Memlet('b[0]'))
    return inner


def _inner_const_write() -> dace.SDFG:
    """Sibling body ``c = 7``, no in-connectors."""
    inner = dace.SDFG('inner_const')
    inner.add_array('c', [1], dace.float64)
    st = inner.add_state()
    tasklet = st.add_tasklet('const', {}, {'__o'}, '__o = 7.0')
    st.add_edge(tasklet, '__o', st.add_write('c'), None, dace.Memlet('c[0]'))
    return inner


def _sibling_nsdfgs_with_ordering_edge(n: int) -> dace.SDFG:
    """``map i: { nsdfg(b=a+1) ; nsdfg(c=7) }``; the reader-less sibling is held by an empty memlet."""
    sdfg = dace.SDFG('ordering_edge_siblings')
    for name in ('A', 'B', 'C'):
        sdfg.add_array(name, [n], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(i=f'0:{n}'))
    me.add_in_connector('IN_A')
    me.add_out_connector('OUT_A')
    for conn in ('B', 'C'):
        mx.add_in_connector(f'IN_{conn}')
        mx.add_out_connector(f'OUT_{conn}')

    first = state.add_nested_sdfg(_inner_read_write(), {'a'}, {'b'})
    second = state.add_nested_sdfg(_inner_const_write(), {}, {'c'})

    state.add_edge(state.add_read('A'), None, me, 'IN_A', dace.Memlet(f'A[0:{n}]'))
    state.add_edge(me, 'OUT_A', first, 'a', dace.Memlet('A[i]'))
    state.add_edge(first, 'b', mx, 'IN_B', dace.Memlet('B[i]'))
    state.add_edge(mx, 'OUT_B', state.add_write('B'), None, dace.Memlet(f'B[0:{n}]'))
    state.add_edge(me, None, second, None, dace.Memlet())
    state.add_edge(second, 'c', mx, 'IN_C', dace.Memlet('C[i]'))
    state.add_edge(mx, 'OUT_C', state.add_write('C'), None, dace.Memlet(f'C[0:{n}]'))
    return sdfg


def test_ordering_memlet_into_sibling_keeps_no_connector():
    """An ordering (empty) memlet into the dropped sibling must move over WITHOUT a connector."""
    n = 8
    sdfg = _sibling_nsdfgs_with_ordering_edge(n)
    sdfg.validate()

    assert NormalizeMapBody().apply_pass(sdfg, {}) == 1, 'the two siblings must merge'

    state = sdfg.states()[0]
    nested = [nd for nd in state.nodes() if isinstance(nd, nodes.NestedSDFG)]
    assert len(nested) == 1, f'the map body must be a single nested SDFG, got {len(nested)}'
    merged = nested[0]
    assert None not in merged.in_connectors, f'ordering memlet became a connector: {merged.in_connectors}'
    assert None not in merged.out_connectors, f'ordering memlet became a connector: {merged.out_connectors}'
    me = [nd for nd in state.nodes() if isinstance(nd, nodes.MapEntry)][0]
    assert any(e.data.is_empty() for e in state.edges_between(me, merged)), (
        'the happens-before edge that held the reader-less sibling in the scope was dropped')
    sdfg.validate()

    a = np.arange(n, dtype=np.float64)
    b = np.zeros(n)
    c = np.zeros(n)
    sdfg(A=a, B=b, C=c)
    assert np.allclose(b, a + 1.0), f'got {b}'
    assert np.allclose(c, 7.0), f'got {c}'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
