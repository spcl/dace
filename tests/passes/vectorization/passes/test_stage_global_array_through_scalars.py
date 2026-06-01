# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for ``StageGlobalArrayThroughScalars``.

The pass rewrites ``Tasklet1 -> A(global) -> Tasklet2`` hops — where ``A`` is a
non-transient (global / argument) array access node that bridges a producer
tasklet and a consumer tasklet — so the producer→consumer value is staged
through fresh **transient scalars** instead of round-tripping through global
memory (which blocks vectorization / tiling).

Two rewrite shapes are exercised:

- **Case A** (``s1`` / ``s2`` PROVABLY DISJOINT): the ``T1 -> T2`` dependency
  through ``A`` is false. The producer writes a fresh scalar ``A1`` which is
  then stored to ``global[s1]``; the consumer reads ``global[s2]`` through a
  fresh scalar ``A2``.
- **Case B** (NOT provably disjoint — a real RMW): the value flows
  ``T1 -> A1 -> T2`` through one transient and is ALSO stored to ``global[s1]``
  via an assignment tasklet.

Refusal cases (intervening write to ``A``, ``wcr`` edges, transient ``A``,
multi-element hops) must leave the SDFG unchanged.

Every transformable case is checked both structurally (transient scalars
inserted, global store preserved) AND numerically: the original SDFG is
compiled + run as the reference, the pass is applied to a deep copy, that is
compiled + run, and the outputs are compared with ``numpy.allclose`` (the
e2e-numerical convention).

The pass module does not exist yet — the implementation agent creates it. Until
then this file raises an ``ImportError`` for the pass *only*; everything else
imports cleanly.
"""
import copy

import numpy
import pytest

import dace
from dace import data as dt
from dace.transformation.passes.vectorization.stage_global_array_through_scalars import (
    StageGlobalArrayThroughScalars, )

#: Symbolic map size used by the no-pattern (``add_mapped_tasklet``) fixture.
N_SYM = dace.symbol("N")

#: Number of species in the cloudsc-style ``[i, j, 5]`` global arrays.
NSPECIES = 5

#: Tolerances for the value-preserving numerical equivalence check.
RTOL = 1e-12
ATOL = 1e-12


# ---------------------------------------------------------------------------
# Structural helpers
# ---------------------------------------------------------------------------
def _transient_scalars(sdfg: dace.SDFG):
    """Collect every transient scalar descriptor name in ``sdfg`` (recursively).

    :param sdfg: The SDFG to scan.
    :returns: A set of descriptor names that are transient scalars.
    """
    names = set()
    for sd in sdfg.all_sdfgs_recursive():
        for name, desc in sd.arrays.items():
            if isinstance(desc, dt.Scalar) and desc.transient:
                names.add(name)
    return names


def _global_write_edges(sdfg: dace.SDFG, array_name: str):
    """Collect every edge that writes the global array ``array_name``.

    A write edge is an in-edge of an access node whose ``data`` is
    ``array_name``. Recurses into nested SDFGs (the cloudsc pattern lives in
    body NSDFGs).

    :param sdfg: The SDFG to scan.
    :param array_name: The global array descriptor name.
    :returns: A list of ``(edge, state)`` tuples for every write.
    """
    writes = []
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.all_states():
            for node in state.data_nodes():
                if node.data != array_name:
                    continue
                for edge in state.in_edges(node):
                    writes.append((edge, state))
    return writes


def _count_tasklets(sdfg: dace.SDFG) -> int:
    """Count tasklets across the whole SDFG (recursively).

    :param sdfg: The SDFG to scan.
    :returns: The number of ``Tasklet`` nodes.
    """
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))


def _has_global_to_tasklet_edge(sdfg: dace.SDFG, array_name: str) -> bool:
    """Whether any access node of ``array_name`` feeds a tasklet directly.

    After the pass this must be ``False`` for the staged chains — the consumer
    reads a transient scalar, not the global node.

    :param sdfg: The SDFG to scan.
    :param array_name: The global array descriptor name.
    :returns: ``True`` iff a ``A(global) -> Tasklet`` edge survives.
    """
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.all_states():
            for node in state.data_nodes():
                if node.data != array_name:
                    continue
                for edge in state.out_edges(node):
                    if isinstance(edge.dst, dace.nodes.Tasklet):
                        return True
    return False


# ---------------------------------------------------------------------------
# Input SDFG builders (construction API)
# ---------------------------------------------------------------------------
def _build_single_global_array_chain(name: str, *, s1: str, s2: str, shape, strides) -> dace.SDFG:
    """Build ``T1 -> A(global) -> T2`` inside a single-iteration Map.

    The Map body holds only tasklets and access nodes (the no-NSDFG
    restriction the staging pass enforces). For ``s1 != s2`` the read
    subset is sourced from an outer ``A`` access node via the Map's
    entry; ``s1 == s2`` is RMW (the in-iteration write satisfies the
    read directly).

    :param name: SDFG name.
    :param s1: Subset string written by ``T1`` (e.g. ``"0"`` or ``"0, 0"``).
    :param s2: Subset string read by ``T2``.
    :param shape: Shape of the global array ``A``.
    :param strides: Strides for ``A``.
    :returns: The constructed SDFG.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", shape, dace.float64, strides=strides, transient=False)
    sdfg.add_array("src", (1, ), dace.float64, transient=False)
    sdfg.add_array("dst", (1, ), dace.float64, transient=False)
    state = sdfg.add_state("main", is_start_block=True)

    src = state.add_access("src")
    dst = state.add_access("dst")
    outer_a_in = state.add_access("A")
    outer_a_out = state.add_access("A")
    me, mx = state.add_map("stage_map", dict(_i="0:1"))
    bridge = state.add_access("A")
    t1 = state.add_tasklet("t1", {"_in"}, {"_out"}, "_out = _in + 1.0")
    t2 = state.add_tasklet("t2", {"_in"}, {"_out"}, "_out = _in * 2.0")

    state.add_memlet_path(src, me, t1, dst_conn="_in", memlet=dace.Memlet("src[0]"))
    state.add_edge(t1, "_out", bridge, None, dace.Memlet(f"A[{s1}]"))
    state.add_edge(bridge, None, t2, "_in", dace.Memlet(f"A[{s2}]"))
    state.add_memlet_path(t2, mx, dst, src_conn="_out", memlet=dace.Memlet("dst[0]"))
    # Always source the bridge from the outer-A-in side so the writes
    # to ``A[s1]`` are not aliased away by the codegen when ``s1 == s2``.
    # The full-array read keeps the inner bridge in sync with the outer
    # array on entry, the write at ``s1`` then takes effect via the
    # drain on exit.
    full = ",".join(f"0:{s}" for s in shape)
    state.add_memlet_path(outer_a_in, me, bridge, memlet=dace.Memlet(f"A[{full}]"))
    state.add_memlet_path(bridge, mx, outer_a_out, memlet=dace.Memlet(f"A[{full}]"))

    sdfg.validate()
    return sdfg


def _build_intervening_write_chain(name: str) -> dace.SDFG:
    """Build a Case-B-shaped chain that ALSO has a second write to ``A[0]``.

    A single state computes:

    - ``T1``: ``A[0] = src[0] + 1.0``
    - ``Tx``: ``A[0] = src[0] * 3.0``   (a SECOND, intervening write to ``A[0]``)
    - ``T2``: ``dst[0] = A[0] * 2.0``    (RMW: reads the same element)

    The two writes to the same global element ``A[0]`` mean the value the
    transient would carry is not the sole authority for ``A[0]`` — the pass must
    REFUSE this occurrence and leave the SDFG unchanged. The second write ``Tx``
    is sequenced after ``T1`` (``a1 -dep-> a2``) so the SDFG is well-formed.

    :param name: SDFG name.
    :returns: The constructed SDFG.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (1, ), dace.float64, transient=False)
    sdfg.add_array("src", (1, ), dace.float64, transient=False)
    sdfg.add_array("dst", (1, ), dace.float64, transient=False)
    state = sdfg.add_state("main", is_start_block=True)

    src = state.add_access("src")
    dst = state.add_access("dst")
    # Two access nodes for the SAME global array; both are written.
    a1 = state.add_access("A")
    a2 = state.add_access("A")

    t1 = state.add_tasklet("t1", {"_in"}, {"_out"}, "_out = _in + 1.0")
    tx = state.add_tasklet("tx", {"_in"}, {"_out"}, "_out = _in * 3.0")
    t2 = state.add_tasklet("t2", {"_in"}, {"_out"}, "_out = _in * 2.0")

    state.add_edge(src, None, t1, "_in", dace.memlet.Memlet("src[0]"))
    state.add_edge(src, None, tx, "_in", dace.memlet.Memlet("src[0]"))

    # First write, then an intervening second write to A[0], then the read.
    state.add_edge(t1, "_out", a1, None, dace.memlet.Memlet("A[0]"))
    # Sequence the second write after the first (write-after-write ordering).
    state.add_edge(a1, None, tx, None, dace.memlet.Memlet())
    state.add_edge(tx, "_out", a2, None, dace.memlet.Memlet("A[0]"))
    state.add_edge(a2, None, t2, "_in", dace.memlet.Memlet("A[0]"))

    state.add_edge(t2, "_out", dst, None, dace.memlet.Memlet("dst[0]"))

    sdfg.validate()
    return sdfg


def _build_transient_bridge_chain(name: str) -> dace.SDFG:
    """Build ``T1 -> A(TRANSIENT) -> T2`` — must be refused (A not global).

    The bridging array ``A`` is a transient. The pass only stages *global*
    arrays, so this occurrence must be left unchanged.

    :param name: SDFG name.
    :returns: The constructed SDFG.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (1, ), dace.float64, transient=True)
    sdfg.add_array("src", (1, ), dace.float64, transient=False)
    sdfg.add_array("dst", (1, ), dace.float64, transient=False)
    state = sdfg.add_state("main", is_start_block=True)

    src = state.add_access("src")
    dst = state.add_access("dst")
    a_node = state.add_access("A")

    t1 = state.add_tasklet("t1", {"_in"}, {"_out"}, "_out = _in + 1.0")
    t2 = state.add_tasklet("t2", {"_in"}, {"_out"}, "_out = _in * 2.0")

    state.add_edge(src, None, t1, "_in", dace.memlet.Memlet("src[0]"))
    state.add_edge(t1, "_out", a_node, None, dace.memlet.Memlet("A[0]"))
    state.add_edge(a_node, None, t2, "_in", dace.memlet.Memlet("A[0]"))
    state.add_edge(t2, "_out", dst, None, dace.memlet.Memlet("dst[0]"))

    sdfg.validate()
    return sdfg


def _build_wcr_chain(name: str) -> dace.SDFG:
    """Build a ``T1 -> A(global) -> T2`` chain whose store edge carries a wcr.

    Staging a reduction edge through a plain scalar copy would drop the
    accumulation, so the pass must refuse this occurrence.

    :param name: SDFG name.
    :returns: The constructed SDFG.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (1, ), dace.float64, transient=False)
    sdfg.add_array("src", (1, ), dace.float64, transient=False)
    sdfg.add_array("dst", (1, ), dace.float64, transient=False)
    state = sdfg.add_state("main", is_start_block=True)

    src = state.add_access("src")
    dst = state.add_access("dst")
    a_node = state.add_access("A")

    t1 = state.add_tasklet("t1", {"_in"}, {"_out"}, "_out = _in + 1.0")
    t2 = state.add_tasklet("t2", {"_in"}, {"_out"}, "_out = _in * 2.0")

    state.add_edge(src, None, t1, "_in", dace.memlet.Memlet("src[0]"))

    wcr_memlet = dace.memlet.Memlet("A[0]")
    wcr_memlet.wcr = "lambda a, b: a + b"
    state.add_edge(t1, "_out", a_node, None, wcr_memlet)
    state.add_edge(a_node, None, t2, "_in", dace.memlet.Memlet("A[0]"))
    state.add_edge(t2, "_out", dst, None, dace.memlet.Memlet("dst[0]"))

    sdfg.validate()
    return sdfg


# ---------------------------------------------------------------------------
# Numerical-equivalence runner
# ---------------------------------------------------------------------------
def _run_and_compare(build_fn, name: str, arrays, params):
    """Compile + run the original SDFG, then a pass-applied deep copy, compare.

    :param build_fn: Zero-arg builder returning a fresh input SDFG.
    :param name: Base SDFG name (suffixed for each variant's build dir).
    :param arrays: Mapping of array-arg name -> numpy array (deep-copied per run).
    :param params: Mapping of scalar/symbol arg name -> value.
    :returns: The pass-applied SDFG (for additional structural assertions).
    """
    ref = build_fn(name + "_ref")
    vec = build_fn(name + "_vec")
    StageGlobalArrayThroughScalars().apply_pass(vec, {})
    vec.validate()

    ref_arrays = {k: copy.deepcopy(v) for k, v in arrays.items()}
    vec_arrays = {k: copy.deepcopy(v) for k, v in arrays.items()}
    ref.compile()(**ref_arrays, **params)
    vec.compile()(**vec_arrays, **params)
    for key in arrays:
        numpy.testing.assert_allclose(vec_arrays[key],
                                      ref_arrays[key],
                                      rtol=RTOL,
                                      atol=ATOL,
                                      err_msg=f"{name}: array {key!r} diverged after staging")
    return vec


def _run_and_compare_python(build_fn, name: str, arrays, params, python_ref):
    """Compile + run the pass-applied SDFG, then compare against a pure-Python reference.

    This is the StageGlobalArrayThroughScalars-only second arm of every
    fixture's correctness check: the pass-applied SDFG must produce the
    same output as a NumPy-only model of the kernel, NOT just the same
    output as its un-transformed twin. The SDFG-vs-SDFG check in
    :func:`_run_and_compare` catches "the pass changed the SDFG's
    output"; this check catches "the pass and the fixture are both
    consistently wrong".

    :param build_fn: Zero-arg builder returning a fresh input SDFG.
    :param name: Base SDFG name (suffixed for the build dir).
    :param arrays: Mapping of array-arg name -> numpy array.
    :param params: Mapping of scalar / symbol arg name -> value.
    :param python_ref: Callable ``(arrays_copy, params) -> arrays_copy``
        that mutates an array dict in place to the expected post-kernel
        state. The fixture's plain-Python model.
    """
    vec = build_fn(name + "_pyvec")
    StageGlobalArrayThroughScalars().apply_pass(vec, {})
    vec.validate()

    sdfg_arrays = {k: copy.deepcopy(v) for k, v in arrays.items()}
    vec.compile()(**sdfg_arrays, **params)

    expected = {k: copy.deepcopy(v) for k, v in arrays.items()}
    python_ref(expected, params)

    for key in arrays:
        numpy.testing.assert_allclose(
            sdfg_arrays[key],
            expected[key],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{name}: array {key!r} drifted from the pure-Python reference",
        )


# ---------------------------------------------------------------------------
# Case A — provably disjoint subsets
# ---------------------------------------------------------------------------
def test_disjoint_constant_dim_is_case_a_structure():
    """A 1-D global array written at ``A[0]`` and read at ``A[1]`` is a Case-A
    occurrence: the pass inserts TWO transient scalars and the global array
    keeps its store but loses the direct ``A -> T2`` feed."""
    sdfg = _build_single_global_array_chain("stage_disjoint_struct", s1="0", s2="1", shape=(4, ), strides=(1, ))
    n_before = len(_transient_scalars(sdfg))
    applied = StageGlobalArrayThroughScalars().apply_pass(sdfg, {})
    sdfg.validate()

    assert applied, "expected the disjoint chain to be staged"
    # Case A introduces two fresh transient scalars (A1 store-staging, A2 load).
    assert len(_transient_scalars(sdfg)) >= n_before + 2, "Case A must add two transient scalars"
    # The real global store survives.
    assert _global_write_edges(sdfg, "A"), "global store to A must be preserved"
    # The consumer no longer reads the global node directly.
    assert not _has_global_to_tasklet_edge(sdfg, "A"), "T2 must read a transient, not the global node"


def _python_ref_single_global_array_chain(s1, s2):
    """Build the pure-Python model of :func:`_build_single_global_array_chain`.

    The fixture's kernel:
        A[s1] = src[0] + 1.0
        dst[0] = A[s2] * 2.0  # reads the post-write A at s2 when s1 == s2

    Returns a closure mutating its first argument's ``A`` / ``dst`` in place.
    """

    def _ref(arrays, params):
        A = arrays["A"]
        src = arrays["src"]
        dst = arrays["dst"]
        # Honour single-element subset semantics: ``s1`` / ``s2`` are
        # whitespace-tolerant comma-separated index expressions.
        idx1 = tuple(int(t.strip()) for t in s1.split(","))
        idx2 = tuple(int(t.strip()) for t in s2.split(","))
        A[idx1] = src[0] + 1.0
        dst[0] = A[idx2] * 2.0

    return _ref


def test_disjoint_constant_dim_is_case_a_numerics():
    """Case-A staging is value-preserving for the disjoint 1-D kernel."""
    rng = numpy.random.default_rng(seed=0)
    arrays = {
        "A": rng.random(4),
        "src": rng.random(1),
        "dst": numpy.zeros(1),
    }
    _run_and_compare(lambda nm: _build_single_global_array_chain(nm, s1="0", s2="1", shape=(4, ), strides=(1, )),
                     "stage_disjoint_num", arrays, {})
    _run_and_compare_python(lambda nm: _build_single_global_array_chain(nm, s1="0", s2="1", shape=(4, ), strides=(1, )),
                            "stage_disjoint_num", arrays, {}, _python_ref_single_global_array_chain("0", "1"))


def test_disjoint_multidim_offsets_is_case_a_numerics():
    """Two disjoint constant offsets on a 2-D ``[NSPECIES, NSPECIES]`` global
    array (``A[1, 0]`` write, ``A[3, 0]`` read) stage as Case A and stay
    numerically equivalent."""
    rng = numpy.random.default_rng(seed=1)
    arrays = {
        "A": rng.random((NSPECIES, NSPECIES)),
        "src": rng.random(1),
        "dst": numpy.zeros(1),
    }
    _run_and_compare(
        lambda nm: _build_single_global_array_chain(
            nm, s1="1, 0", s2="3, 0", shape=(NSPECIES, NSPECIES), strides=(NSPECIES, 1)), "stage_disjoint_offset",
        arrays, {})
    _run_and_compare_python(
        lambda nm: _build_single_global_array_chain(
            nm, s1="1, 0", s2="3, 0", shape=(NSPECIES, NSPECIES), strides=(NSPECIES, 1)), "stage_disjoint_offset",
        arrays, {}, _python_ref_single_global_array_chain("1, 0", "3, 0"))


# ---------------------------------------------------------------------------
# Case B — not provably disjoint (RMW)
# ---------------------------------------------------------------------------
def test_identical_subset_is_case_b_structure():
    """A global element written and then read at the SAME subset ``A[0]`` is a
    Case-B RMW: the pass inserts ONE transient scalar that serves both producer
    and consumer, drains to the global through the enclosing ``MapExit`` chain
    (no extra assignment tasklet needed), and the consumer no longer reads
    the global node directly."""
    sdfg = _build_single_global_array_chain("stage_rmw_struct", s1="0", s2="0", shape=(1, ), strides=(1, ))
    n_scalars_before = len(_transient_scalars(sdfg))
    applied = StageGlobalArrayThroughScalars().apply_pass(sdfg, {})
    sdfg.validate()

    assert applied, "expected the RMW chain to be staged"
    # Case B introduces one fresh transient scalar (shared by producer / consumer).
    assert len(_transient_scalars(sdfg)) >= n_scalars_before + 1, "Case B must add a transient scalar"
    # The global array still receives the stored value via the MapExit drain.
    assert _global_write_edges(sdfg, "A"), "global store to A must be preserved"
    # The consumer reads the transient, not the global node.
    assert not _has_global_to_tasklet_edge(sdfg, "A"), "T2 must read the transient, not the global node"


def test_identical_subset_is_case_b_numerics():
    """Case-B (RMW) staging is value-preserving for the same-element kernel."""
    rng = numpy.random.default_rng(seed=2)
    arrays = {
        "A": rng.random(1),
        "src": rng.random(1),
        "dst": numpy.zeros(1),
    }
    _run_and_compare(lambda nm: _build_single_global_array_chain(nm, s1="0", s2="0", shape=(1, ), strides=(1, )),
                     "stage_rmw_num", arrays, {})
    _run_and_compare_python(lambda nm: _build_single_global_array_chain(nm, s1="0", s2="0", shape=(1, ), strides=(1, )),
                            "stage_rmw_num", arrays, {}, _python_ref_single_global_array_chain("0", "0"))


def test_overlapping_multidim_subset_is_case_b_numerics():
    """A 2-D same-element read/write (``A[2, 0]`` write, ``A[2, 0]`` read —
    proven overlap, the conservative non-disjoint path) stays value-preserving
    under Case-B staging."""
    rng = numpy.random.default_rng(seed=3)
    arrays = {
        "A": rng.random((NSPECIES, NSPECIES)),
        "src": rng.random(1),
        "dst": numpy.zeros(1),
    }
    _run_and_compare(
        lambda nm: _build_single_global_array_chain(
            nm, s1="2, 0", s2="2, 0", shape=(NSPECIES, NSPECIES), strides=(NSPECIES, 1)), "stage_rmw_2d", arrays, {})
    _run_and_compare_python(
        lambda nm: _build_single_global_array_chain(
            nm, s1="2, 0", s2="2, 0", shape=(NSPECIES, NSPECIES), strides=(NSPECIES, 1)), "stage_rmw_2d", arrays, {},
        _python_ref_single_global_array_chain("2, 0", "2, 0"))


# ---------------------------------------------------------------------------
# Multi-dim global array (zqx[i, j, 4]-style)
# ---------------------------------------------------------------------------
def test_multidim_global_disjoint_species_numerics():
    """A 3-D ``[NSPECIES, NSPECIES, NSPECIES]`` (zqx-style) global array written
    at species 4 and read at species 2 along the last dim is disjoint (Case A)
    and value-preserving; the staged transient scalars are present."""
    rng = numpy.random.default_rng(seed=4)
    arrays = {
        "A": rng.random((NSPECIES, NSPECIES, NSPECIES)),
        "src": rng.random(1),
        "dst": numpy.zeros(1),
    }
    vec = _run_and_compare(
        lambda nm: _build_single_global_array_chain(nm,
                                                    s1="1, 1, 4",
                                                    s2="1, 1, 2",
                                                    shape=(NSPECIES, NSPECIES, NSPECIES),
                                                    strides=(NSPECIES * NSPECIES, NSPECIES, 1)), "stage_multidim",
        arrays, {})
    # Two transient scalars for the disjoint species hop.
    assert len(_transient_scalars(vec)) >= 2, "multi-dim disjoint hop must stage through two scalars"
    _run_and_compare_python(
        lambda nm: _build_single_global_array_chain(nm,
                                                    s1="1, 1, 4",
                                                    s2="1, 1, 2",
                                                    shape=(NSPECIES, NSPECIES, NSPECIES),
                                                    strides=(NSPECIES * NSPECIES, NSPECIES, 1)), "stage_multidim",
        arrays, {}, _python_ref_single_global_array_chain("1, 1, 4", "1, 1, 2"))


# ---------------------------------------------------------------------------
# Refusal / skip cases — SDFG must be unchanged
# ---------------------------------------------------------------------------
def test_intervening_global_write_is_refused():
    """A second write to the same global element ``A[0]`` on the chain violates
    the Case-B no-other-write invariant; the pass must refuse (no-op)."""
    sdfg = _build_intervening_write_chain("stage_intervening")
    before = sdfg.to_json()
    applied = StageGlobalArrayThroughScalars().apply_pass(sdfg, {})
    sdfg.validate()
    assert not applied, "intervening-write chain must be refused"
    assert sdfg.to_json() == before, "refused SDFG must be unchanged"


def test_transient_bridge_is_refused():
    """A transient bridging array is not a global; the pass must leave it."""
    sdfg = _build_transient_bridge_chain("stage_transient_bridge")
    before = sdfg.to_json()
    applied = StageGlobalArrayThroughScalars().apply_pass(sdfg, {})
    sdfg.validate()
    assert not applied, "transient-bridge chain must be refused"
    assert sdfg.to_json() == before, "refused SDFG must be unchanged"


def test_wcr_store_edge_is_refused():
    """A wcr (reduction) store edge must not be staged (would drop the
    accumulation); the pass must leave the SDFG unchanged."""
    sdfg = _build_wcr_chain("stage_wcr")
    before = sdfg.to_json()
    applied = StageGlobalArrayThroughScalars().apply_pass(sdfg, {})
    sdfg.validate()
    assert not applied, "wcr store edge must be refused"
    assert sdfg.to_json() == before, "refused SDFG must be unchanged"


def test_no_pattern_is_noop():
    """A kernel with no ``T1 -> A(global) -> T2`` hop is a no-op."""
    sdfg = dace.SDFG("stage_nopattern")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_array("src", (N_SYM, ), dace.float64)
    sdfg.add_array("dst", (N_SYM, ), dace.float64)
    state = sdfg.add_state("main", is_start_block=True)
    t, _, _ = state.add_mapped_tasklet("m", {"i": "0:N:1"},
                                       inputs={"_in": dace.memlet.Memlet("src[i]")},
                                       code="_out = _in * 2.0",
                                       outputs={"_out": dace.memlet.Memlet("dst[i]")},
                                       external_edges=True)
    sdfg.validate()
    before = sdfg.to_json()
    applied = StageGlobalArrayThroughScalars().apply_pass(sdfg, {})
    assert not applied, "no-pattern SDFG must be a no-op"
    assert sdfg.to_json() == before, "no-op SDFG must be unchanged"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------
def test_pass_is_idempotent_case_a():
    """A second application of the pass on the already-staged Case-A kernel is
    a no-op (the global node no longer bridges two tasklets)."""
    sdfg = _build_single_global_array_chain("stage_idem", s1="0", s2="1", shape=(4, ), strides=(1, ))
    first = StageGlobalArrayThroughScalars().apply_pass(sdfg, {})
    sdfg.validate()
    assert first, "first application should stage the chain"
    second = StageGlobalArrayThroughScalars().apply_pass(sdfg, {})
    assert not second, "second application must be a no-op"


# The cloudsc-pattern (zqlhs / zsolqb reuse) tests live separately in
# ``test_stage_global_array_cloudsc.py``.

if __name__ == "__main__":
    pytest.main([__file__, "-q"])
