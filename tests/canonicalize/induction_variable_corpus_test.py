# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Corpus test for ``InductionVariableSubstitution`` over every loop-carried scalar the LLR
track sweep classified, split into the three outcomes a counter can have:

* TRUE_IV -- a closed form exists; the counter must vanish and the read/write subscript
  becomes the closed form. Verified two ways: a bare, single ``apply_pass`` call on a fresh
  copy (true isolation -- no other transform runs) records what IVS alone can do, and the
  production ``canonicalize`` recipe (with IVS's five entry points monkeypatched to record
  firings) gives the fully-closed end state the structural and numeric checks run against.
* REFUSE -- a conditional increment (counts true predicates) or a data-dependent step (a
  reduction, e.g. ``sum += a[i]``) has no closed form. A single isolated ``apply_pass`` call
  must return ``None`` and leave the SDFG byte-for-byte unchanged (``to_json()`` equality);
  the production recipe must still compute the right answer with none of IVS's five entry
  points ever firing.

Surprise worth flagging up front: the frontend's own ``simplify()`` already closes some
counters (``s122``, ``s125``) before IVS ever runs -- their ``to_sdfg(simplify=True)`` output
has no trace of the counter at all, isolated or not. And a bare isolated call cannot tell a
true IV needing pipeline prep (``s318``, ``s453``: both return ``None``/unchanged alone) apart
from a genuine refusal -- only the production recipe's entry-point recording can, since IVS's
own logic fires there once its prerequisite passes (constant-argument promotion, statement
splitting) have run, and never fires for a true refusal no matter how much prep precedes it.
Also: the sweep that produced this corpus mis-tagged ``s124`` as a conditional refusal. Its
``j = j + 1`` runs identically in EVERY branch of the ``if``/``else`` (branch-uniform), so
``j == i`` unconditionally -- a bare isolated call confirms it (fires, non-trivially changes
the SDFG), and ``canonicalize_iv_chain_test.py::test_s124_branch_uniform_iv_parallelizes``
already treats it as a true IV. It is filed under TRUE_IV here, not under the sweep's REFUSE
list, with the mis-tag recorded in a comment rather than silently perpetuated.

Source of the corpus: the LLR benchmark tree this repo's TSVC/TSVC-2.5 corpus modules mirror
(``tests/corpus/tsvc`` and ``tests/corpus/tsvc_2_5``); kept as a pathlib constant purely for
provenance -- the corpus modules already imported below are the executable oracle, this file
never reads the benchmark tree at test time.
"""
import contextlib
import copy
import io
import pathlib
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pytest
import sympy

import dace
from dace import symbolic
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import induction_variable_substitution as ivs
from dace.transformation.passes.canonicalize.induction_variable_substitution import InductionVariableSubstitution
from dace.transformation.passes.canonicalize.pipeline import canonicalize

from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc.tsvc_numpy import REFERENCES
from tests.corpus.tsvc_2_5 import tsvc_2_5, tsvc_2_5_numpy

#: Provenance only (see module docstring) -- the LLR benchmark tree this corpus mirrors.
BENCH_ROOT = pathlib.Path(
    "/capstor/scratch/cscs/ybudanaz/x86_64/optarena/hpcagent_bench/benchmarks/loop_level_reasoning")

_PEEL_LIMIT = 4
_TOL = 1e-9

#: Every entry point through which IVS can perform a substitution -- see
#: ``induction_variable_substitution.py``'s own module-level functions.
IVS_ENTRY_POINTS = ("_try_substitute", "try_substitute_use_site_iv", "_try_substitute_iedge_iv",
                    "_try_substitute_derived_symbol", "_hoist_branch_uniform_iv")

_CAST_FUNCS = ("int64", "int32", "float64", "float32")


def strip_casts(expr):
    """Drop ``dace.int64(x)``-style cast wrappers so a value-preserving cast does not defeat a
    symbolic zero-diff comparison (``int64(inc) - inc`` does not auto-simplify to 0, the cast is
    an opaque sympy ``Function``)."""
    return expr.replace(lambda n: isinstance(n, sympy.Function) and n.func.__name__ in _CAST_FUNCS, lambda n: n.args[0])


def single_element_reads(sdfg: dace.SDFG, array_name: str) -> List[Tuple["dace.SDFGState", object]]:
    """``(state, edge)`` for every edge reading exactly one element of ``array_name``."""
    return [(state, e) for state in sdfg.all_states() for e in state.edges() if e.data is not None
            and not e.data.is_empty() and e.data.data == array_name and e.data.subset.num_elements() == 1]


def enclosing_map_params(state, node) -> Dict[str, tuple]:
    """``{param: (start, stop, step)}`` for every ``MapEntry`` enclosing ``node`` in ``state``."""
    scope = state.scope_dict()
    out = {}
    n = node
    while n is not None:
        n = scope[n]
        if isinstance(n, nodes.MapEntry):
            out.update(zip(n.map.params, n.map.range))
    return out


def binding_before(target_node) -> Dict[str, str]:
    """Interstate-edge assignments accumulated, in execution order, up to (not including)
    ``target_node`` within its own immediate parent region.

    Sound because every seed this corpus resolves this way is bound on an ACYCLIC stretch of
    graph: either the one edge before a loop even starts (``target_node`` is the ``LoopRegion``
    itself, one level up from its own back edge), or a loop body's own straight-line interior
    (the repetition lives in the ``LoopRegion`` container, not in its body graph), or the fully
    parallelized top-level graph after a loop has already been turned into a map. A genuine
    loop-carried recurrence -- the REFUSE half of this corpus -- never goes through this helper.
    """
    region = target_node.parent_graph
    subs: Dict[str, str] = {}
    for node in nx.topological_sort(region.nx):
        if node is target_node:
            break
        for e in region.out_edges(node):
            subs.update(e.data.assignments or {})
    return subs


def resolve_zero_input_tasklet(sdfg: dace.SDFG, data_name: str):
    """Symbolic value of the tasklet writing ``data_name``, when that tasklet has NO incoming
    data edges (every operand is already a free SDFG symbol, so its code string is directly a
    closed-form expression in those symbols) -- the one-hop staging a promoted read-only
    argument or symbol goes through before it can appear in a subset. Raises if no such tasklet
    exists, rather than silently returning a wrong value for a shape this was not written for.
    """
    for state in sdfg.all_states():
        for n in state.nodes():
            if isinstance(n, nodes.AccessNode) and n.data == data_name:
                for e in state.in_edges(n):
                    if isinstance(e.src, nodes.Tasklet) and not state.in_edges(e.src):
                        rhs = e.src.code.as_string.split("=", 1)[1].strip()
                        return symbolic.pystr_to_symbolic(rhs)
    raise AssertionError(f"no zero-input tasklet writes {data_name!r}")


def isolated_apply(sdfg: dace.SDFG):
    """Apply IVS exactly once, alone, and report ``(result, unchanged)`` -- ``unchanged`` is a
    ``to_json()`` equality, the strongest available proxy for "the pass declined and touched
    nothing"."""
    before = sdfg.to_json()
    with contextlib.redirect_stdout(io.StringIO()):
        result = InductionVariableSubstitution().apply_pass(sdfg, {})
    return result, sdfg.to_json() == before


def canonicalize_recording(sdfg: dace.SDFG) -> List[str]:
    """Run the production ``canonicalize`` recipe with IVS's five entry points monkeypatched to
    record firings, so the verdict is attributed to IVS itself and not confounded with whatever
    else the recipe does (LoopToMap, BreakAntiDependence, ArgReduce lifting, ...)."""
    fired: List[str] = []
    originals = {ep: vars(ivs)[ep] for ep in IVS_ENTRY_POINTS}

    def wrap(entry_point, base):

        def wrapper(parent, loop, inner_sdfg, sdfg_free_symbols):
            out = base(parent, loop, inner_sdfg, sdfg_free_symbols)
            if out:
                fired.append(entry_point)
            return out

        return wrapper

    for ep in IVS_ENTRY_POINTS:
        setattr(ivs, ep, wrap(ep, originals[ep]))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            canonicalize(sdfg, validate=True, peel_limit=_PEEL_LIMIT)
    finally:
        for ep in IVS_ENTRY_POINTS:
            setattr(ivs, ep, originals[ep])
    return fired


def nloops(sdfg: dace.SDFG) -> int:
    return sum(1 for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)


def allclose(a, b) -> bool:
    return np.allclose(np.asarray(a), np.asarray(b), rtol=_TOL, atol=_TOL, equal_nan=True)


def tsvc_kernel(name: str):
    return [k for k in tsvc.collect() if k.name == name][0]


def tsvc25_program(name: str):
    return [p for p in tsvc_2_5.collect() if p.f.__name__ == name][0]


# =====================================================================================
# TRUE_IV -- closed forms, verified independently (Python trace + symbolic subset check).
# =====================================================================================
#
# s126: k := k+1 stepped once per inner iter and once more per outer iter (TWO sites); k=1
#   seed. Closed form k(i,j) = i*LEN_2D + j, verified by hand-tracing bb/flat_2d_array
#   indices at LEN_2D = 5, 8, 13 (k after j increments then the extra end-of-row bump lands
#   exactly on i*LEN_2D+j at the START of row i, j=1). The read is flat_2d_array[k-1], so the
#   subset is i*LEN_2D+j-1.
# s125: k := k+1, ONE site nested two loops deep, k=-1 seed -> k(i,j) = i*LEN_2D+j (a plain
#   row-major flatten index). ALREADY closed by the frontend's own simplify() -- to_sdfg(...,
#   simplify=True) already reads/writes flat_2d_array[LEN_2D*i + j] directly; IVS never runs.
# s122: k := k+j with j=1 loop-invariant, loop range(n1-1, LEN_1D, n3). Trip t at iteration i
#   is int_floor(i-(n1-1), n3); k after t+1 increments is t+1, so the read b[LEN_1D-k]
#   becomes b[LEN_1D - (int_floor(i-(n1-1),n3) + 1)]. ALSO already closed by simplify() (the
#   frontend's own scalar-to-symbol promotion resolves it) -- confirmed the same way as s125.
# s124: j := j+1 in BOTH branches of the if/else (branch-uniform, not conditional -- see
#   module docstring), j=-1 seed -> j(i) = i exactly. A SINGLE bare apply_pass call closes it
#   (_hoist_branch_uniform_iv); j survives as a loop-invariant symbol bound to -1 after that
#   one call, and a further canonicalize pass folds it away entirely.
# s318: k := k+inc with inc a read-only scalar ARGUMENT (not a literal), plus one increment
#   BEFORE the loop (k=inc going in). The frontend cannot put a data-typed step on an iedge,
#   so it stages it through a transient; IVS needs 'inc' already promoted to a symbol (a
#   canonicalize prerequisite) before its iedge-IV matcher can see the shape -- a bare
#   isolated call declines. Closed form k = i*inc, verified: at iteration i (i=1..LEN_1D-1)
#   the read is a[inc*i]; ArgReduce's lift folds the pre-loop a[0] check in as trip _j=0 of
#   the SAME map, giving subscript _j*inc for _j in [0, LEN_1D).
# s453: s := s+2.0, READ by the sibling statement a[i] = s*b[i] -- neither eliminable (the
#   read needs s's per-iteration value) nor fissionable. Use-site expansion at iteration i:
#   s == s_entry + 2.0*(i+1), matching Aho/Lam/Sethi/Ullman's closed-form-and-inline scan.
#   Needs canonicalize's statement handling first; a bare isolated call declines.


def check_s126_closed(sdfg: dace.SDFG) -> None:
    reads = [(s, e) for s, e in single_element_reads(sdfg, "flat_2d_array")]
    assert len(reads) == 1, f"s126: expected one single-element flat_2d_array read, got {len(reads)}"
    state, edge = reads[0]
    (expr, ) = edge.data.subset.min_element()
    enclosing = enclosing_map_params(state, edge.dst)
    (outer, ) = [p for p, rng in enclosing.items() if symbolic.simplify(rng[0]) == 0]
    (inner, ) = [p for p, rng in enclosing.items() if symbolic.simplify(rng[0]) == 1]
    by_name = {str(s): s for s in expr.free_symbols}
    i, j, n = by_name[outer], by_name[inner], by_name["LEN_2D"]
    assert symbolic.simplify(expr - (i * n + j - 1)) == 0, f"s126: expected i*LEN_2D+j-1, got {expr}"


def check_s125_closed(sdfg: dace.SDFG) -> None:
    # already closed straight out of the frontend -- no counter to resolve at all.
    assert "k" not in sdfg.arrays and "k" not in sdfg.symbols, "s125: 'k' survived simplify()"
    reads = single_element_reads(sdfg, "flat_2d_array")
    assert len(reads) == 1, f"s125: expected one single-element flat_2d_array write, got {len(reads)}"
    state, edge = reads[0]
    (expr, ) = edge.data.subset.min_element()
    by_name = {str(s): s for s in expr.free_symbols}
    i, j, n = by_name["i"], by_name["j"], by_name["LEN_2D"]
    assert symbolic.simplify(expr - (i * n + j)) == 0, f"s125: expected i*LEN_2D+j, got {expr}"


def check_s122_closed(sdfg: dace.SDFG) -> None:
    assert "k" not in sdfg.arrays and "k" not in sdfg.symbols, "s122: 'k' survived simplify()"
    reads = single_element_reads(sdfg, "b")
    assert len(reads) == 1, f"s122: expected one single-element b read, got {len(reads)}"
    state, edge = reads[0]
    (raw, ) = edge.data.subset.min_element()
    subs = {}
    for s in raw.free_symbols:
        name = str(s)
        if name.startswith("__sym_"):
            staged = binding_before(state)[name]  # a data name one more hop away
            subs[s] = resolve_zero_input_tasklet(sdfg, staged)
    expr = strip_casts(raw.subs(subs))
    by_name = {str(s): s for s in expr.free_symbols}
    i, n1, n3, len1d = by_name["i"], by_name["n1"], by_name["n3"], by_name["LEN_1D"]
    expected = len1d - (symbolic.int_floor(i - (n1 - 1), n3) + 1)
    assert symbolic.simplify(expr - expected) == 0, f"s122: expected LEN_1D-(int_floor(i-n1+1,n3)+1), got {expr}"


def check_s124_closed(sdfg: dace.SDFG) -> None:
    """Checked on the SDFG straight after the isolated apply_pass call (not full canonicalize):
    j survives as a loop-invariant symbol bound to -1, not yet folded away, so this checks
    i+j+1 (the surviving form) rather than a bare i."""
    loop = next(r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion))
    j_val = symbolic.pystr_to_symbolic(binding_before(loop)["j"])
    reads = single_element_reads(sdfg, "a")
    assert len(reads) == 2, f"s124: expected two single-element a accesses (read+write), got {len(reads)}"
    for state, edge in reads:
        (raw, ) = edge.data.subset.min_element()
        by_name = {str(s): s for s in raw.free_symbols}
        expr = raw.subs({by_name["j"]: j_val}) if "j" in by_name else raw
        i = by_name.get("i") or next(iter(raw.free_symbols))
        assert symbolic.simplify(expr - i) == 0, f"s124: expected a[i], got a[{raw}] (j={j_val})"


def check_s124_closed_final(sdfg: dace.SDFG) -> None:
    """Checked after the FULL canonicalize recipe: j is gone outright (folded into the map)."""
    assert "j" not in sdfg.symbols and "j" not in sdfg.arrays, "s124: 'j' survived canonicalize"


def check_s318_closed(sdfg: dace.SDFG) -> None:
    """``k += inc`` closed to an affine stride -- now read off the ArgReduce's own input memlet.

    The closure used to be observable as a gather MAP whose single-element read of ``a`` resolved
    to ``_j * inc``. The lift no longer builds that map: it folds the gather into the arg-reduction
    by giving ``_in`` a strided subset, which is what stops a LEN_1D staging buffer from being
    allocated. Same property, one level down -- if ``k`` had not closed, the step could not be a
    plain ``inc``.
    """
    from dace.libraries.standard.nodes import ArgReduce
    edges = [(state, e) for state in sdfg.all_states() for n in state.nodes() if isinstance(n, ArgReduce)
             for e in state.in_edges(n) if e.dst_conn == "_in"]
    assert len(edges) == 1, f"s318: expected exactly one ArgReduce _in edge, got {len(edges)}"
    _, edge = edges[0]
    assert edge.data.data == "a", f"s318: the arg-reduction must read a directly, got {edge.data.data}"
    ranges = edge.data.subset.ranges
    assert len(ranges) == 1, f"s318: expected a 1-D strided read, got {ranges}"
    _lo, _hi, step = ranges[0]
    by_name = {str(sym): sym for sym in step.free_symbols}
    assert "inc" in by_name, f"s318: the stride must still carry inc, got {step}"
    assert symbolic.simplify(strip_casts(step) - by_name["inc"]) == 0, f"s318: expected step inc, got {step}"


def check_s453_closed(sdfg: dace.SDFG) -> None:
    candidates = [
        n for state in sdfg.all_states() for n in state.nodes()
        if isinstance(n, nodes.Tasklet) and "__in1" in n.in_connectors and "__in2" in n.in_connectors
    ]
    assert len(candidates) == 1, f"s453: expected one fused mult tasklet, got {len(candidates)}"
    full = symbolic.pystr_to_symbolic(candidates[0].code.as_string.split("=", 1)[1].strip())
    by_name = {str(s): s for s in full.free_symbols}
    s_entry, b_in, it = by_name["__in1"], by_name["__in2"], by_name["_loop_it_0"]
    expected = (s_entry + 2.0 * (it + 1)) * b_in
    assert symbolic.simplify(full - expected) == 0, f"s453: expected (s_entry+2.0*(i+1))*b[i], got {full}"


#: (kernel, module tag, bare-isolated-call expectation, closed-form checker(s)).
#: bare_fires: whether a SINGLE ``apply_pass`` call alone substitutes something (None means
#: "declines, needs pipeline prep OR is already closed" -- both happen in this corpus and are
#: told apart by the comments above and the per-kernel test below, not by this flag alone).
TRUE_IV = [
    ("s126_d_single", True, check_s126_closed),  # fires but only closes ONE of two sites
    ("s125_d_single", None, check_s125_closed),  # already closed pre-canonicalize
    ("s122_d_single", None, check_s122_closed),  # already closed pre-canonicalize
    ("s124_d_single", True, check_s124_closed),  # single isolated call fully substitutes
    ("s318_d_single", False, check_s318_closed),  # declines alone, needs arg-promotion prep
    ("s453_d_single", False, check_s453_closed),  # declines alone, needs statement-split prep
]


@pytest.mark.parametrize("name,bare_fires,check_closed", TRUE_IV, ids=[t[0] for t in TRUE_IV])
def test_true_induction_variable_closes(name, bare_fires, check_closed):
    kernel = tsvc_kernel(name)

    # Phase 1 -- true isolation: one bare apply_pass call on its own copy, no other transform.
    scratch = tsvc.to_sdfg(kernel, "bare_" + name, simplify=True)
    result, unchanged = isolated_apply(scratch)
    if bare_fires is True:
        assert result is not None, f"{name}: expected a single isolated call to substitute something"
        assert not unchanged, f"{name}: a firing substitution must change the SDFG"
    elif bare_fires is False:
        assert result is None, f"{name}: expected a single isolated call to decline (needs pipeline prep)"
        assert unchanged, f"{name}: a declining call must not touch the SDFG"
    else:  # already closed -- isolation is a no-op because there is nothing left to do
        assert result is None and unchanged, f"{name}: expected a no-op (already closed by simplify())"

    # Phase 2 -- structural verdict + closed form, on the SDFG the numerics will run from.
    sdfg = tsvc.to_sdfg(kernel, "full_" + name, simplify=True)
    if bare_fires is None:
        # nothing for IVS to do; check the closed form is already there, pre-compile.
        fired: List[str] = []
        check_closed(sdfg)
    elif name == "s124_d_single":
        # verified once right after the single isolated call (still has the invariant j==-1
        # symbol), and once more after full canonicalize (j gone outright).
        InductionVariableSubstitution().apply_pass(sdfg, {})
        check_closed(sdfg)
        fired = canonicalize_recording(sdfg)
        check_s124_closed_final(sdfg)
    else:
        fired = canonicalize_recording(sdfg)
        assert fired, f"{name}: expected at least one IVS entry point to fire in the full pipeline"
        check_closed(sdfg)

    if bare_fires is not None and name != "s124_d_single":
        assert fired, f"{name}: IVS itself must be the one that closed this, but nothing fired"

    # Phase 3 -- numerics, compiled separately from the structural checks above.
    arrays, call_kwargs = tsvc.make_inputs(kernel)
    ref = {n: a.copy() for n, a in arrays.items()}
    REFERENCES[name](**ref, **call_kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        csdfg = sdfg.compile()
    got = {n: a.copy() for n, a in arrays.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        csdfg(**got, **call_kwargs)
    for n, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.integer):
            continue
        assert allclose(ref[n], got[n]), f"{name}/{n}: closed-form result diverges from the numpy reference"


# =====================================================================================
# REFUSE -- conditional counters and reductions wearing an induction variable's clothes.
# =====================================================================================
#
# Conditional: `if c: j += 1` makes j a COUNT of true predicates -- not affine in the loop
# indices, no closed form exists. s341/s342/s343 increment only inside the `if`, no matching
# `else` (unlike s124, which is branch-uniform and IS a true IV -- see module docstring).
# s123 increments unconditionally once per outer iteration AND again inside `if c[i] > 0.0`,
# so it is conditional on the second site.
#
# Reduction: the step reads data that changes every iteration (`a[i]`, `a[i]*b[ip[i]]`, a
# 5-term dot product, ...) -- structurally identical to `k += 1` to a naive matcher, but the
# step is DATA-DEPENDENT, not loop-invariant, so no closed form exists either; these belong to
# LoopToScan/Reduce lifting instead. s319 and s31111 additionally cover TWO/FOUR steps in one
# body; reduce_inner_carry (tsvc_2_5 corpus) covers a per-row carried reduction under an outer
# parallel loop, and s3111 is BOTH conditional and data-dependent at once (either alone would
# already force a refusal).
CONDITIONAL_REFUSALS = ["s123_d_single", "s341_d_single", "s342_d_single", "s343_d_single"]

REDUCTION_REFUSALS = [
    "s3112_d_single", "s312_d_single", "vsumr_d_single", "s3111_d_single", "s319_d_single", "s352_d_single",
    "s4115_d_single", "s4116_d_single", "s31111_d_single"
]


@pytest.mark.parametrize("name",
                         CONDITIONAL_REFUSALS + REDUCTION_REFUSALS,
                         ids=CONDITIONAL_REFUSALS + REDUCTION_REFUSALS)
def test_non_induction_variable_refuses(name):
    kernel = tsvc_kernel(name)

    # Phase 1 -- true isolation: one bare call must decline and touch nothing.
    scratch = tsvc.to_sdfg(kernel, "bare_" + name, simplify=True)
    result, unchanged = isolated_apply(scratch)
    assert result is None, f"{name}: a conditional/data-dependent step has no closed form -- must decline"
    assert unchanged, f"{name}: a declining pass must not touch the SDFG"

    # Phase 2 -- not confounded: no IVS entry point may fire even inside the full recipe.
    sdfg = tsvc.to_sdfg(kernel, "full_" + name, simplify=True)
    fired = canonicalize_recording(sdfg)
    assert not fired, f"{name}: IVS must never substitute this, but {sorted(set(fired))} fired"

    # Phase 3 -- numerics: refusal must still be VALUE-preserving (LoopToScan/Reduce, or plain
    # sequential execution, must compute the right answer).
    arrays, call_kwargs = tsvc.make_inputs(kernel)
    ref = {n: a.copy() for n, a in arrays.items()}
    REFERENCES[name](**ref, **call_kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        csdfg = sdfg.compile()
    got = {n: a.copy() for n, a in arrays.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        csdfg(**got, **call_kwargs)
    for n, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.integer):
            continue
        assert allclose(ref[n], got[n]), f"{name}/{n}: refusal changed the computed value"


def test_reduce_inner_carry_refuses():
    """``reduce_inner_carry`` (tsvc_2_5 corpus): outer loop parallel over rows, inner loop
    carries a per-row scalar reduction ``out[i] = sum_j a[i, j]`` -- same REFUSE contract as
    the tsvc REDUCTION_REFUSALS above, kept separate because it lives in a different corpus
    module with its own input/oracle plumbing."""
    program = tsvc25_program("reduce_inner_carry")

    scratch = copy.deepcopy(program.to_sdfg(simplify=True))
    scratch.name = "bare_reduce_inner_carry"
    result, unchanged = isolated_apply(scratch)
    assert result is None, "reduce_inner_carry: a data-dependent step has no closed form -- must decline"
    assert unchanged, "reduce_inner_carry: a declining pass must not touch the SDFG"

    sdfg = copy.deepcopy(program.to_sdfg(simplify=True))
    sdfg.name = "full_reduce_inner_carry"
    fired = canonicalize_recording(sdfg)
    assert not fired, f"reduce_inner_carry: IVS must never substitute this, but {sorted(set(fired))} fired"

    arrays, scalars = tsvc_2_5.make_inputs(program)
    ref = {n: a.copy() for n, a in arrays.items()}
    tsvc_2_5_numpy.ref_reduce_inner_carry(**ref)
    with contextlib.redirect_stdout(io.StringIO()):
        csdfg = sdfg.compile()
    got = {n: a.copy() for n, a in arrays.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        csdfg(**got, **scalars, LEN_2D=tsvc_2_5.SIZES["LEN_2D"])
    for n in arrays:
        assert allclose(ref[n], got[n]), f"reduce_inner_carry/{n}: refusal changed the computed value"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
