# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A write-only scratch scalar in a map body must not blind the passes that inspect that body.

``SDFGState.all_nodes_between`` discards its ENTIRE result the moment the walk reaches a node with
no out-edge, and a write-only scratch scalar -- a transient ``AccessNode`` with one in-edge and none
out, the shape CloudSC spells ``zanew_0`` -- is exactly such a node. A body predicate built on that
walk then reports a clean body having inspected nothing, and a body counter reports zero for a body
full of nodes.

Every case below plants that sink in an otherwise ordinary map body and pins what each
canonicalization pass must then do. Two outcomes are correct and the tests distinguish them:

* FAIL-DANGEROUS sites -- ``NormalizeLoopAndMapOrigin`` (rebases the map range, then substitutes the
  shift into the body it walked) and ``LoopToEinsum``'s transpose purity check (approves the scope
  it walked as a pure copy) -- now read scope membership, and the sink no longer changes the answer.
* FAIL-SAFE sites -- ``LoopToSymm`` and ``LiftInv`` guard on ``len(body) != 1`` and REFUSE on the
  emptied walk. The refusal loses a lift and never produces a wrong answer, so the walk stays; the
  tests here pin the refusal so a later relaxation of the guard cannot turn it dangerous silently.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

from typing import List, Tuple

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.blas.nodes.symm import Symm
from dace.libraries.linalg.nodes.inv import Inv
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState, nodes
from dace.transformation.passes.analysis import map_scope
from dace.transformation.passes.canonicalize import loop_to_einsum
from dace.transformation.passes.canonicalize.lift_inv import LiftInv
from dace.transformation.passes.canonicalize.loop_to_symm import LoopToSymm
from dace.transformation.passes.canonicalize.normalize_loop_and_map_origin import NormalizeLoopAndMapOrigin
from dace.transformation.passes.canonicalize.normalize_map_body import NormalizeMapBody

SCRATCH = "zanew_0"


def plant_write_only_scratch(sdfg: SDFG, state: SDFGState, map_entry: nodes.MapEntry, name: str = SCRATCH) -> None:
    """Give ``map_entry``'s scope a transient Register scalar that is written and never read."""
    sdfg.add_scalar(name,
                    dace.float64,
                    transient=True,
                    storage=dtypes.StorageType.Register,
                    lifetime=dtypes.AllocationLifetime.Scope)
    writer = state.add_tasklet(f"write_{name}", {}, {f"{name}_out"}, f"{name}_out = 0.0")
    state.add_edge(map_entry, None, writer, None, Memlet())
    state.add_edge(writer, f"{name}_out", state.add_access(name), None, Memlet(f"{name}[0]"))


def two_param_map_entry(sdfg: SDFG) -> Tuple[SDFGState, nodes.MapEntry]:
    """The single two-parameter map of a freshly parsed kernel, with its state."""
    found = [(st, n) for sd in sdfg.all_sdfgs_recursive() for st in sd.all_states() for n in st.nodes()
             if isinstance(n, nodes.MapEntry) and len(n.map.params) == 2]
    assert len(found) == 1, f"expected one two-parameter map, got {len(found)}"
    return found[0]


def scaled_copy_map(begin: int, extent: int, with_scratch: bool) -> Tuple[SDFG, SDFGState, nodes.MapEntry]:
    """``B[i] = 2 * A[i]`` over ``i in [begin, extent)``, optionally beside a write-only scratch."""
    sdfg = SDFG("scaled_copy")
    sdfg.add_array("A", [extent], dace.float64)
    sdfg.add_array("B", [extent], dace.float64)
    state = sdfg.add_state("s")
    read = state.add_access("A")
    entry, exit_node = state.add_map("m", dict(i=f"{begin}:{extent}"))
    body = state.add_tasklet("scale", {"inp"}, {"out"}, "out = inp * 2.0")
    state.add_memlet_path(read, entry, body, dst_conn="inp", memlet=Memlet("A[i]"))
    state.add_memlet_path(body, exit_node, state.add_access("B"), src_conn="out", memlet=Memlet("B[i]"))
    if with_scratch:
        plant_write_only_scratch(sdfg, state, entry)
    sdfg.validate()
    return sdfg, state, entry


def rebinding_map(begin: int, extent: int) -> Tuple[SDFG, nodes.MapEntry]:
    """A map whose body NestedSDFG binds the map parameter under a different inner name.

    That binding is what ``rebinds_params`` refuses: the rebase substitutes ``i -> i + begin`` in a
    body that spells the parameter ``k``, finds nothing to rewrite, and hands ``k`` the shifted
    value. The scratch beside it is what used to hide the NestedSDFG from the refusal.
    """
    inner = SDFG("inner")
    inner.add_array("iin", [extent], dace.float64)
    inner.add_array("iout", [1], dace.float64)
    istate = inner.add_state("is")
    itasklet = istate.add_tasklet("scale", {"x"}, {"y"}, "y = x * 2.0")
    istate.add_edge(istate.add_access("iin"), None, itasklet, "x", Memlet("iin[k]"))
    istate.add_edge(itasklet, "y", istate.add_access("iout"), None, Memlet("iout[0]"))

    sdfg = SDFG("rebinding")
    sdfg.add_array("A", [extent], dace.float64)
    sdfg.add_array("B", [extent], dace.float64)
    state = sdfg.add_state("s")
    read = state.add_access("A")
    entry, exit_node = state.add_map("m", dict(i=f"{begin}:{extent}"))
    nested = state.add_nested_sdfg(inner, {"iin": None}, {"iout": None}, {"k": "i"})
    state.add_memlet_path(read, entry, nested, dst_conn="iin", memlet=Memlet(f"A[0:{extent}]"))
    state.add_memlet_path(nested, exit_node, state.add_access("B"), src_conn="iout", memlet=Memlet("B[i]"))
    plant_write_only_scratch(sdfg, state, entry)
    sdfg.validate()
    return sdfg, entry


def increment_leaf(name: str) -> SDFG:
    """A one-state ``y = x + 1`` SDFG, the shape MapFusion leaves as a map-body sibling."""
    inner = SDFG(name)
    inner.add_array("x", [1], dace.float64)
    inner.add_array("y", [1], dace.float64)
    state = inner.add_state("s")
    tasklet = state.add_tasklet("inc", {"a"}, {"b"}, "b = a + 1.0")
    state.add_edge(state.add_access("x"), None, tasklet, "a", Memlet("x[0]"))
    state.add_edge(tasklet, "b", state.add_access("y"), None, Memlet("y[0]"))
    return inner


def map_of_sibling_nsdfgs(siblings: int, with_scratch: bool) -> Tuple[SDFG, SDFGState, nodes.MapEntry]:
    """A map body holding ``siblings`` independent NestedSDFGs, optionally beside a scratch sink."""
    sdfg = SDFG("siblings")
    for name in ("A", "B", "C"):
        sdfg.add_array(name, [16], dace.float64)
    state = sdfg.add_state("s")
    read = state.add_access("A")
    entry, exit_node = state.add_map("m", dict(i="0:16"))
    for index, out_name in enumerate(("B", "C")[:siblings]):
        nested = state.add_nested_sdfg(increment_leaf(f"leaf{index}"), {"x": None}, {"y": None}, {})
        state.add_memlet_path(read, entry, nested, dst_conn="x", memlet=Memlet("A[i]"))
        state.add_memlet_path(nested,
                              exit_node,
                              state.add_access(out_name),
                              src_conn="y",
                              memlet=Memlet(f"{out_name}[i]"))
    if with_scratch:
        plant_write_only_scratch(sdfg, state, entry)
    sdfg.validate()
    return sdfg, state, entry


def transpose_probe(side_effect: bool) -> SDFG:
    """A collapsed ``B[p, q] = A[q, p]`` map, the shape ``LoopToEinsum``'s probe hands the extractor.

    With ``side_effect`` the scope also writes the non-transient ``C`` through a sink -- a body that
    a Transpose lift would discard, and the very node that empties the reachability walk.
    """
    sdfg = SDFG("probe_transpose")
    sdfg.add_array("A", [8, 8], dace.float64)
    sdfg.add_array("B", [8, 8], dace.float64)
    sdfg.add_array("C", [1], dace.float64)
    state = sdfg.add_state("s")
    entry, exit_node = state.add_map("t", dict(p="0:8", q="0:8"))
    copy_tasklet = state.add_tasklet("cp", {"__inp"}, {"__out"}, "__out = __inp")
    state.add_memlet_path(state.add_access("A"), entry, copy_tasklet, dst_conn="__inp", memlet=Memlet("A[q, p]"))
    state.add_memlet_path(copy_tasklet, exit_node, state.add_access("B"), src_conn="__out", memlet=Memlet("B[p, q]"))
    if side_effect:
        side = state.add_tasklet("side", {}, {"__out"}, "__out = 3.0")
        state.add_edge(entry, None, side, None, Memlet())
        state.add_edge(side, "__out", state.add_access("C"), None, Memlet("C[0]"))
    sdfg.validate()
    return sdfg


M = dace.symbol("M")
N = dace.symbol("N")


@dace.program
def polybench_symm(C: dace.float64[M, N], A: dace.float64[M, M], B: dace.float64[M, N], alpha: dace.float64[1],
                   beta: dace.float64[1]):
    """The hand-written polybench ``symm`` nest ``LoopToSymm`` is built to recognize."""

    @dace.mapscope
    def comp_all(j: _[0:N], i: _[0:M]):
        temp2 = dace.define_local_scalar(dace.float64)

        @dace.tasklet
        def reset_tmp():
            tmp >> temp2
            tmp = 0

        @dace.map
        def comp_t2(k: _[0:i]):
            ialpha << alpha
            ia << A[i, k]
            ibi << B[i, j]
            ibk << B[k, j]
            oc >> C(1, lambda a, b: a + b)[k, j]
            ot2 >> temp2(1, lambda a, b: a + b)

            oc = ialpha * ibi * ia
            ot2 = ibk * ia

        @dace.tasklet
        def comp_rest():
            ibeta << beta
            ib << B[i, j]
            iadiag << A[i, i]
            ialpha << alpha
            it2 << temp2
            ic << C[i, j]
            oc >> C[i, j]
            oc = ibeta * ic + ialpha * ib * iadiag + ialpha * it2


@dace.program
def solve_eye(A: dace.float64[N, N], out: dace.float64[N, N]):
    """``solve(A, eye(N))`` -- the inverse spelling ``LiftInv`` is built to recognize."""
    out[:] = np.linalg.solve(A, np.eye(N))


def symm_nest(scratch_in_body: bool) -> SDFG:
    """The polybench symm nest, optionally with a write-only scratch planted in its map body."""
    sdfg = polybench_symm.to_sdfg(simplify=False)
    state, entry = two_param_map_entry(sdfg)
    planters = {True: plant_write_only_scratch, False: leave_body_alone}
    planters[scratch_in_body](sdfg, state, entry)
    return sdfg


def solve_against_identity(scratch_in_body: bool) -> SDFG:
    """``solve(A, eye(N))``, optionally with a write-only scratch in the identity-construction map."""
    sdfg = solve_eye.to_sdfg(simplify=True)
    state, entry = two_param_map_entry(sdfg)
    planters = {True: plant_write_only_scratch, False: leave_body_alone}
    planters[scratch_in_body](sdfg, state, entry)
    return sdfg


def leave_body_alone(sdfg: SDFG, state: SDFGState, map_entry: nodes.MapEntry) -> None:
    """The no-scratch half of the planter pair."""


def body_node_names(state: SDFGState, entry: nodes.MapEntry) -> List[str]:
    return sorted(str(n) for n in map_scope.map_body_nodes(state, entry))


def nsdfgs_in_body(state: SDFGState, entry: nodes.MapEntry) -> List[nodes.NestedSDFG]:
    return [n for n in map_scope.map_body_nodes(state, entry) if isinstance(n, nodes.NestedSDFG)]


def test_a_map_scope_holding_nothing_counts_zero_body_nodes():
    """The empty bracket: a map entry wired straight to its exit must count zero, or no non-zero
    count from this helper means anything."""
    sdfg = SDFG("empty_body")
    state = sdfg.add_state("s")
    entry, exit_node = state.add_map("m", dict(i="0:16"))
    state.add_edge(entry, None, exit_node, None, Memlet())
    sdfg.validate()

    assert map_scope.map_body_nodes(state, entry) == []


def test_a_write_only_scratch_scalar_is_a_map_body_node():
    """The other half of the bracket: the sink and everything beside it are counted, where the
    reachability walk reported the whole body as empty."""
    sdfg, state, entry = scaled_copy_map(begin=2, extent=16, with_scratch=True)

    scratch = next(n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == SCRATCH)
    assert state.out_edges(scratch) == [], "the planted scratch must be the sink shape that empties the walk"
    assert body_node_names(state, entry) == ["scale", "write_zanew_0", "zanew_0"]
    assert state.all_nodes_between(entry, state.exit_node(entry)) == set(), \
        "the reachability walk is expected to keep discarding this body -- that is the bug being worked around"


def test_rebasing_a_map_beside_a_write_only_scratch_shifts_the_body_reads():
    """``NormalizeLoopAndMapOrigin`` on ``i in [2, 16)``: the range becomes 0-based AND every body
    memlet gains the ``+ 2``. Shifting one without the other reads the wrong two elements."""
    sdfg, state, entry = scaled_copy_map(begin=2, extent=16, with_scratch=True)

    NormalizeLoopAndMapOrigin().apply_pass(sdfg, {})

    assert str(entry.map.range) == "0:14"
    scale = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet) and n.label == "scale")
    assert sorted(str(e.data) for e in state.all_edges(scale)) == ["A[i + 2]", "B[i + 2]"]
    sdfg.validate()

    a = np.arange(16, dtype=np.float64)
    b = np.zeros(16, dtype=np.float64)
    sdfg(A=a, B=b)
    expected = np.zeros(16, dtype=np.float64)
    expected[2:] = a[2:] * 2.0
    assert np.array_equal(b, expected)


def test_rebasing_is_refused_when_a_scratch_bearing_body_rebinds_the_map_parameter():
    """A body NestedSDFG binding ``k -> i`` cannot be shifted, and the scratch beside it must not
    hide that NestedSDFG from the refusal: the map range stays exactly as written."""
    sdfg, entry = rebinding_map(begin=2, extent=16)

    NormalizeLoopAndMapOrigin().apply_pass(sdfg, {})

    assert str(entry.map.range) == "2:16"


def test_a_pure_transposed_copy_is_read_as_a_transpose():
    """The control for the purity check: a scope holding only the copy tasklet IS a transpose."""
    spec = loop_to_einsum._extract_transpose(transpose_probe(side_effect=False), {"B": None})

    assert spec is not None
    assert (spec.src, spec.dst) == ("A", "B")


def test_a_transposed_copy_that_also_writes_another_array_is_not_a_transpose():
    """The same nest plus a write to the non-transient ``C`` through a sink is not a pure copy, and
    lifting it to a Transpose would silently drop that write."""
    spec = loop_to_einsum._extract_transpose(transpose_probe(side_effect=True), {"B": None})

    assert spec is None


def test_sibling_nested_sdfgs_beside_a_write_only_scratch_are_still_merged():
    """``NormalizeMapBody`` must see both siblings through the scratch and sequence them into one
    NestedSDFG -- the merge that exposes same-condition guards to ConditionFusion."""
    sdfg, state, entry = map_of_sibling_nsdfgs(siblings=2, with_scratch=True)
    assert len(nsdfgs_in_body(state, entry)) == 2

    merged = NormalizeMapBody().apply_pass(sdfg, {})

    assert merged == 1
    assert len(nsdfgs_in_body(state, entry)) == 1
    sdfg.validate()


def test_a_lone_nested_sdfg_beside_a_write_only_scratch_is_left_alone():
    """One sibling is nothing to merge: the pass must report no change rather than rewrite a body
    it can now finally see."""
    sdfg, state, entry = map_of_sibling_nsdfgs(siblings=1, with_scratch=True)

    merged = NormalizeMapBody().apply_pass(sdfg, {})

    assert merged is None
    assert len(nsdfgs_in_body(state, entry)) == 1
    sdfg.validate()


@pytest.mark.parametrize("scratch_in_body, expected_lifts, expected_symms", [(False, 1, 1), (True, None, 0)])
def test_symm_nest_lifts_only_while_its_body_holds_the_nested_sdfg_alone(scratch_in_body: bool, expected_lifts,
                                                                         expected_symms: int):
    """``LoopToSymm`` guards on a body of exactly one node, so the scratch costs the lift. Fail-safe
    by construction: the nest is left standing and still computes symm, never lifted wrongly."""
    sdfg = symm_nest(scratch_in_body)

    lifts = LoopToSymm().apply_pass(sdfg, {})

    assert lifts == expected_lifts
    assert sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Symm)) == expected_symms
    sdfg.validate()


@pytest.mark.parametrize("scratch_in_body, expected_lifts, expected_invs", [(False, 1, 1), (True, None, 0)])
def test_solve_against_identity_lifts_only_while_the_identity_map_holds_its_tasklet_alone(
        scratch_in_body: bool, expected_lifts, expected_invs: int):
    """``LiftInv`` guards on an identity map of exactly one tasklet, so the scratch costs the lift.
    Fail-safe: the Solve and its identity construction survive untouched."""
    sdfg = solve_against_identity(scratch_in_body)

    lifts = LiftInv().apply_pass(sdfg, {})

    assert lifts == expected_lifts
    assert sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Inv)) == expected_invs
    sdfg.validate()
