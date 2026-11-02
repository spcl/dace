# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a contraction / transpose loop nest to a single library node.

A sibling of :class:`~dace.transformation.passes.loop_to_reduce.LoopToReduce`: a
loop nest that computes one tensor contraction is replaced by a state holding a
single :class:`~dace.libraries.blas.nodes.einsum.Einsum` node, and a pure matrix
transpose by a :class:`~dace.libraries.linalg.nodes.transpose.Transpose` node.

Recognised contraction shapes (all as ``@dace.program`` loop nests)::

    for i:                for i: for j:              for i: for j: for k:
      for j:                y[j] += A[i,j]*x[i]        C[i,j] += A[i,k]*B[k,j]
        y[i] += A[i,j]*x[j] (transposed matvec:       (matmul: 'ij,jk->ik')
    (matvec: 'ij,j->i')     'ij,i->j')

and the pure transpose ``for i: for j: B[i,j] = A[j,i]`` (2-D, cross-array).

The direct matcher (:func:`_match_nest`) reads the canonical loop form in place: no copy, no pipeline. Measured on
the real form, the nest that reaches this pass is

* a CHAIN of nested single-block ``LoopRegion``s -- one iteration axis each -- and/or
  ``MapEntry`` scopes (a map carries several params at once), innermost body one
  ``SDFGState``;
* whose body accumulates via an explicit read/add/write (``C[i,j] = C[i,j] + prod`` or
  ``C[i,j] = C[i,j] - prod``, what the frontend emits -- NOT a WCR) or via a WCR-sum edge
  (what a ``@dace.map`` accumulation emits). Both are accepted directly;
* whose product is spread over several tasklets joined by single-element transient
  staging scalars (``A --A[i,k]--> A_index --> mul``), which the matcher walks through.

The einsum is then read straight off the index expressions: every axis that indexes
an operand but not the output is contracted. Acceptance mirrors ``LiftEinsum``'s
``can_be_applied`` (>=2 tensor operands, at most one scalar coefficient, indices that
are exactly axis parameters or ``0``, rectangular full ranges, non-scalar output).
"""

import ast
import copy
from typing import Any, NamedTuple, Union

import sympy

from dace import SDFG, data, dtypes, subsets, symbolic
from dace.frontend.operations import detect_reduction_type
from dace.frontend.python import astutils
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.narrowing import as_basic, as_expr
from dace.sdfg.state import AbstractControlFlowRegion, ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.symbolic import pystr_to_symbolic
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.loop_iteration_value import replace_loop_with_state
from dace.transformation.passes.canonicalize.split_statements import value_edges
from dace.transformation.transformation import explicit_cf_compatible


class EinsumSpec(NamedTuple):
    """Everything needed to rebuild an ``Einsum`` node on the original arrays."""

    einsum_str: str
    alpha: object
    beta: object
    inputs: list[tuple[str, str, subsets.Subset, object]]  # (connector, array, subset, dtype)
    output: tuple[str, str, subsets.Subset, object]  # (connector, array, subset, dtype)


class TransposeSpec(NamedTuple):
    """Everything needed to rebuild a ``Transpose`` node on the original arrays."""

    src: str
    dst: str
    dtype: object
    src_subset: subsets.Subset
    dst_subset: subsets.Subset


def _is_single_element(desc: data.Data) -> bool:
    """A ``Scalar`` or a length-1 ``Array`` -- the frontend's per-read staging shape."""
    return isinstance(desc, data.Scalar) or (isinstance(desc, data.Array) and all(str(s) == "1" for s in desc.shape))


def _has_loop_ancestor(loop: LoopRegion) -> bool:
    """True iff ``loop`` is nested inside another ``LoopRegion`` -- we match only
    the OUTERMOST loop of a nest (matching it lifts, or no-ops on, the whole nest;
    an inner loop alone is a 1-D reduction, LoopToReduce's domain, not an einsum)."""
    p = loop.parent_graph
    while p is not None and not isinstance(p, SDFG):
        if isinstance(p, LoopRegion):
            return True
        p = p.parent_graph
    return False


def _region_in_loop(region: AbstractControlFlowRegion | None) -> bool:
    """True iff ``region`` sits inside a ``LoopRegion`` (so its states' maps are already
    covered by that loop's candidate)."""
    while region is not None and not isinstance(region, SDFG):
        if isinstance(region, LoopRegion):
            return True
        region = region.parent_graph
    return False


def sweeps_whole_array(order: list[str], ends: dict[str, object], desc: data.Data) -> bool:
    """Whether ``order`` (one parameter per axis, each running ``0 .. ends[param]``) covers all of ``desc``."""
    return all(
        symbolic.simplify(pystr_to_symbolic(ends[p]) - (extent - 1)) == 0 for p, extent in zip(order, desc.shape)
    )


#: Index letters, in the order ``LiftEinsum`` hands them out (kept identical so a
#: directly-matched einsum string is spelled the way the post-LoopToMap lift spells it).
EINSUM_CHARS = "ijklmnopqrstuvwxyzabcdefgh"


class _Axis(NamedTuple):
    """One iteration dimension of a nest: ``param`` runs ``0 .. end`` inclusive, step 1."""

    param: str
    end: object


class _Nest(NamedTuple):
    """A matched iteration nest: its axes and the innermost scope holding the compute.

    ``entry`` is the innermost ``MapEntry`` (``None`` when the whole nest is
    ``LoopRegion``s and the body is the state's top level)."""

    axes: list[_Axis]
    state: SDFGState
    entry: nodes.MapEntry | None


class _Leaf(NamedTuple):
    """An array read at a single point: ``array`` indexed by one expression per axis."""

    array: str
    idx: list[str]


class _Product(NamedTuple):
    """A tasklet-level product: ``scale`` (a number) times every term."""

    scale: object
    terms: list[Union[_Leaf, "_Product"]]


def _plain_edges(region: ControlFlowRegion) -> bool:
    """True iff every interstate edge inside ``region`` is an unconditional plain link.
    A condition or an assignment means the body is more than straight-line dataflow."""
    for e in region.edges():
        if e.data.assignments:
            return False
        if e.data.condition is not None and e.data.condition.as_string.strip() not in ("1", "True"):
            return False
    return True


def _single_body_block(region: ControlFlowRegion) -> ControlFlowBlock | None:
    """``region``'s one meaningful child block, or ``None``. Empty scaffold states --
    the connective ``..._for_inline_pre_state`` blocks the map->loop lowering leaves
    behind -- are ignored; anything else must be alone."""
    if not _plain_edges(region):
        return None
    blocks = [b for b in region.nodes() if not (isinstance(b, SDFGState) and b.number_of_nodes() == 0)]
    return blocks[0] if len(blocks) == 1 else None


def _loop_axis(loop: LoopRegion) -> _Axis | None:
    """``loop`` as one ``0 .. end`` unit-stride axis, or ``None`` if it is not one.
    A non-zero start or a non-unit stride is refused for the same reason
    ``LiftEinsum._partial_range`` refuses it: the contraction spans the operands' full
    extent, so a partial sweep would compute the wrong (or a clobbering) result."""
    if loop.inverted or not loop.loop_variable:
        return None
    init = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if init is None or end is None or stride is None:
        return None
    if init != 0 or stride != 1:
        return None
    return _Axis(loop.loop_variable, end)


def _map_axes(entry: nodes.MapEntry) -> list[_Axis] | None:
    """A map's params as axes -- one MapEntry carries SEVERAL, unlike a LoopRegion."""
    axes = []
    for param, rng in zip(entry.map.params, entry.map.range):
        if rng[0] != 0 or rng[2] != 1:
            return None
        axes.append(_Axis(param, rng[1]))
    return axes


def _descend_maps(state: SDFGState, axes: list[_Axis], entry: nodes.MapEntry | None) -> _Nest | None:
    """Follow one chain of nested map scopes down from ``entry`` (or from ``state``'s
    top level when it is ``None``), appending each map's params to ``axes``. Refuses a
    level that holds compute of its own or forks into several maps -- the lifted node
    replaces the WHOLE scope, so nothing may be left behind."""
    children = state.scope_children()
    while True:
        if entry is not None:
            more = _map_axes(entry)
            if more is None:
                return None
            axes.extend(more)
        level = [n for n in children[entry] if not isinstance(n, nodes.MapExit)]
        inner = [n for n in level if isinstance(n, nodes.MapEntry)]
        if not inner:
            return _Nest(axes, state, entry)
        if len(inner) != 1 or any(isinstance(n, (nodes.Tasklet, nodes.NestedSDFG, nodes.LibraryNode)) for n in level):
            return None
        entry = inner[0]


def _nest_of(root: LoopRegion) -> _Nest | None:
    """The iteration nest rooted at a ``LoopRegion``. Loops contribute ONE axis each and
    must form a perfect chain; the innermost body may then descend through map scopes
    (a mixed ``for i: for j,k in dace.map[...]`` nest is one nest)."""
    axes: list[_Axis] = []
    block = root
    while isinstance(block, LoopRegion):
        axis = _loop_axis(block)
        if axis is None:
            return None
        axes.append(axis)
        block = _single_body_block(block)
        if block is None:
            return None
    if not isinstance(block, SDFGState):
        return None
    return _descend_maps(block, axes, None)


def _nest_of_map(state: SDFGState, entry: nodes.MapEntry) -> _Nest | None:
    """The iteration nest of a map-form contraction rooted at ``entry``."""
    return _descend_maps(state, [], entry)


def _point_index(memlet: Memlet, expect_data: str | None = None) -> tuple[str, list[str]] | None:
    """``(array, one index expression per axis)`` for a single-element point access,
    else ``None``. Mirrors ``LiftEinsum``'s per-edge screen (no dynamic memlets, exactly
    one element)."""
    if memlet is None or memlet.is_empty() or memlet.data is None or memlet.dynamic:
        return None
    if expect_data is not None and memlet.data != expect_data:
        return None
    subset = memlet.subset
    if subset is None or subset.num_elements() != 1:
        return None
    idx = []
    for rb, re_, _ in subset.ndrange():
        if rb != re_:
            return None
        idx.append(str(rb).strip())
    return memlet.data, idx


def _tasklet_expr(tasklet: nodes.Tasklet) -> symbolic.SymbolicType | None:
    """The tasklet's single assigned expression as a sympy expression, or ``None``."""
    if tasklet.code.language != dtypes.Language.Python:
        return None
    code = tasklet.code.code
    if not isinstance(code, list) or len(code) != 1:
        return None
    stmt = code[0]
    if not isinstance(stmt, (ast.Assign, ast.AnnAssign)) or stmt.value is None:
        return None
    if len(tasklet.out_connectors) != 1:
        return None
    try:
        expr = pystr_to_symbolic(astutils.unparse(stmt.value))
    except (TypeError, ValueError, AttributeError, sympy.SympifyError):
        return None
    # A comparison / logical tasklet (``__out = __in1 > __in2``) parses to a sympy
    # RELATIONAL or BooleanFunction, which cannot be divided or subtracted -- and is
    # never a contraction anyway. Reject it here rather than let the arithmetic raise.
    # NOT ``isinstance(expr, Boolean)``: sympy's ``Symbol`` derives from ``Boolean``, so
    # that would also reject the bare-symbol copy tasklet ``__out = __inp`` (the pure
    # transpose shape).
    if expr.is_Relational or isinstance(expr, sympy.logic.boolalg.BooleanFunction):
        return None
    return expr


def _product_scale(tasklet: nodes.Tasklet) -> sympy.Basic | None:
    """The numeric factor of a tasklet that multiplies ALL of its inputs together
    (``__out = 2.5 * __in2`` -> ``2.5``; ``__out = __in1 * __in2`` -> ``1``), else
    ``None``. Same test ``LiftEinsum`` applies to the fused tasklet, here applied per
    tasklet so a product split across several of them still matches."""
    expr = _tasklet_expr(tasklet)
    if expr is None or not tasklet.in_connectors:
        return None
    expected = 1
    for conn in tasklet.in_connectors:
        expected *= pystr_to_symbolic(conn)
    try:
        ratio = symbolic.simplify(expr / expected)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return ratio if as_basic(ratio).is_Number else None


def _minuend(tasklet: nodes.Tasklet) -> str | None:
    """The connector ``__in1`` of ``__out = __in1 - __in2`` -- a ``-=`` read/add/write -- else ``None``."""
    conns = list(tasklet.in_connectors)
    expr = _tasklet_expr(tasklet)
    if len(conns) != 2 or expr is None:
        return None
    first, second = (as_expr(pystr_to_symbolic(c)) for c in conns)
    try:
        if symbolic.simplify(expr - first + second) == 0:
            return conns[0]
        return conns[1] if symbolic.simplify(expr - second + first) == 0 else None
    except (TypeError, ValueError):
        return None


def _is_sum_tasklet(tasklet: nodes.Tasklet) -> bool:
    """``__out = __in1 + __in2`` -- the read/add/write half of an aug-assign."""
    conns = list(tasklet.in_connectors)
    if len(conns) != 2:
        return False
    expr = _tasklet_expr(tasklet)
    if expr is None:
        return False
    try:
        return symbolic.simplify(expr - pystr_to_symbolic(conns[0]) - as_expr(pystr_to_symbolic(conns[1]))) == 0
    except (TypeError, ValueError):
        return False


def _resolve(
    state: SDFGState, sdfg: SDFG, edge: MultiConnectorEdge[Memlet], visited: dict[nodes.Node, None]
) -> _Leaf | _Product | None:
    """The value flowing along ``edge``, chasing back through single-element transient
    staging scalars and product tasklets to the leaf array reads. ``visited`` collects
    every node consumed, so the caller can verify the body holds nothing else."""
    src = edge.src
    if isinstance(src, nodes.MapEntry):  # scope boundary: the inner memlet is the point read
        origin = state.memlet_path(edge)[0].src
        if not isinstance(origin, nodes.AccessNode):
            return None
        got = _point_index(edge.data, origin.data)
        return None if got is None else _Leaf(*got)
    if isinstance(src, nodes.AccessNode):
        desc = sdfg.arrays.get(src.data)
        if desc is None:
            return None
        if desc.transient and _is_single_element(desc):  # staging scalar: look through it
            if state.in_degree(src) != 1 or state.out_degree(src) != 1:
                return None
            visited[src] = None
            return _resolve(state, sdfg, state.in_edges(src)[0], visited)
        if state.in_degree(src) != 0:
            return None  # not a pure input to this nest
        visited[src] = None
        got = _point_index(edge.data, src.data)
        return None if got is None else _Leaf(*got)
    if isinstance(src, nodes.Tasklet):
        return _resolve_tasklet(state, sdfg, src, visited)
    return None


def _resolve_tasklet(
    state: SDFGState, sdfg: SDFG, tasklet: nodes.Tasklet, visited: dict[nodes.Node, None]
) -> _Product | None:
    """``tasklet`` as a product of its resolved inputs, or ``None``. Single-consumer
    only: a re-used intermediate is not a tree and would be counted twice."""
    if tasklet in visited or state.out_degree(tasklet) != 1:
        return None
    scale = _product_scale(tasklet)
    if scale is None:
        return None
    visited[tasklet] = None
    terms: list[_Leaf | _Product] = []
    for e in state.in_edges(tasklet):
        if e.data is None or e.data.is_empty():
            return None
        term = _resolve(state, sdfg, e, visited)
        if term is None:
            return None
        terms.append(term)
    return _Product(scale, terms) if terms else None


def _flatten(term: _Leaf | _Product) -> tuple[object, list[_Leaf]]:
    """``(numeric coefficient, operand leaves)`` of a resolved product tree."""
    if isinstance(term, _Leaf):
        return sympy.Integer(1), [term]
    scale, leaves = term.scale, []
    for sub in term.terms:
        sub_scale, sub_leaves = _flatten(sub)
        scale = scale * sub_scale
        leaves.extend(sub_leaves)
    return scale, leaves


class _BodyValue(NamedTuple):
    """What the nest's innermost body computes: ``coeff * prod(leaves)`` accumulated
    (or assigned, when ``accumulates`` is False) into ``array`` at ``idx``."""

    array: str
    idx: list[str]
    coeff: object
    leaves: list[_Leaf]
    accumulates: bool
    write_node: nodes.AccessNode | None


def _body_value(nest: _Nest, sdfg: SDFG) -> _BodyValue | None:
    """Read the nest's body as one accumulate-or-assign of a product, or ``None``."""
    state, entry = nest.state, nest.entry
    body = [n for n in state.scope_children()[entry] if not isinstance(n, (nodes.MapEntry, nodes.MapExit))]

    # Locate the single write leaving the scope.
    write_node = None
    if entry is not None:
        out_edges = value_edges(state.in_edges(state.exit_node(entry)))
        if len(out_edges) != 1:
            return None
        out_edge = out_edges[0]
        tail = state.memlet_path(out_edge)[-1].dst
        write_node = tail if isinstance(tail, nodes.AccessNode) else None
    else:
        written = [
            n
            for n in body
            if isinstance(n, nodes.AccessNode)
            and state.in_degree(n) > 0
            and not (sdfg.arrays[n.data].transient and _is_single_element(sdfg.arrays[n.data]))
        ]
        if len(written) != 1 or state.out_degree(written[0]) != 0 or state.in_degree(written[0]) != 1:
            return None
        write_node = written[0]
        out_edge = state.in_edges(write_node)[0]
    got = _point_index(out_edge.data, None if write_node is None else write_node.data)
    if got is None or not isinstance(out_edge.src, nodes.Tasklet):
        return None
    out_array, out_idx = got

    visited: dict = {}
    negated = False
    if out_edge.data.wcr is not None:
        # WCR encoding: the producing tasklet IS the product, the sum lives on the edge.
        if detect_reduction_type(out_edge.data.wcr) != dtypes.ReductionType.Sum:
            return None
        term = _resolve_tasklet(state, sdfg, out_edge.src, visited)
        if term is None:
            return None
        accumulates = True
    else:
        adder = out_edge.src
        in_edges = value_edges(state.in_edges(adder))
        minuend = _minuend(adder)
        if (minuend is not None or _is_sum_tasklet(adder)) and len(in_edges) == 2:
            # Aug-assign encoding: one input re-reads the output slot, the other is the product.
            visited[adder] = None
            resolved = [_resolve(state, sdfg, e, visited) for e in in_edges]
            if any(r is None for r in resolved):
                return None
            prior = [i for i, r in enumerate(resolved) if isinstance(r, _Leaf) and list(r) == [out_array, out_idx]]
            if len(prior) != 1 or minuend not in (None, in_edges[prior[0]].dst_conn):
                return None
            term, accumulates = resolved[1 - prior[0]], True
            negated = minuend is not None
        else:
            # Plain assignment (the pure-copy / transpose shape).
            term = _resolve_tasklet(state, sdfg, adder, visited)
            if term is None:
                return None
            accumulates = False

    if any(n not in visited and not (entry is None and n is write_node) for n in body):
        return None  # the body computes something the lifted node would drop
    coeff, leaves = _flatten(term)
    return _BodyValue(out_array, out_idx, -coeff if negated else coeff, leaves, accumulates, write_node)


def _axis_subset(idx: list[str], axes: dict[str, _Axis]) -> subsets.Range:
    """The whole-operand range a point access ``idx`` sweeps over the nest -- what
    memlet propagation out of the equivalent map produces, and what ``LiftEinsum``
    hands the ``Einsum`` node."""
    return subsets.Range([(0, 0, 1) if i == "0" else (0, axes[i].end, 1) for i in idx])


def _direct_einsum(nest: _Nest, sdfg: SDFG, value: _BodyValue) -> EinsumSpec | None:
    """The ``Einsum`` spec for an accumulated contraction, or ``None``. The acceptance
    rule mirrors ``LiftEinsum.can_be_applied`` on the map this nest would become."""
    if not value.accumulates:
        return None
    axes = {a.param: a for a in nest.axes}
    if len(axes) != len(nest.axes):
        return None  # a repeated parameter -- indices would be ambiguous
    # Rectangular iteration space: a parameter-dependent bound (a triangular syrk-style
    # nest) contracts over less than the full product the einsum expansion assumes.
    for axis in nest.axes:
        if any(s in axes for s in map(str, pystr_to_symbolic(axis.end).free_symbols)):
            return None

    tensors, coeffs = [], []
    for leaf in value.leaves:
        if any(i != "0" and i not in axes for i in leaf.idx):
            return None  # an index that is not an axis parameter
        (tensors if any(i != "0" for i in leaf.idx) else coeffs).append(leaf)
    # >=2 tensor operands: a single one is a copy / transpose / reduction, not a matmul.
    # At most one runtime scalar: the node wires exactly one ``_alpha`` connector.
    if len(tensors) < 2 or len(coeffs) > 1:
        return None
    if any(i != "0" and i not in axes for i in value.idx):
        return None
    if any(t.array == value.array for t in [*tensors, *coeffs]):
        return None  # the output feeds its own contraction -- a real loop-carried dependence
    out_chars = dict.fromkeys(i for i in value.idx if i != "0")
    in_chars = dict.fromkeys(i for t in tensors for i in t.idx if i != "0")
    if not out_chars:
        return None  # scalar output: a dot / full reduction, LoopToReduce's domain
    if not all(c in in_chars for c in out_chars):
        return None  # an output index no operand carries -- a broadcast, not a contraction
    if all(c in out_chars for c in in_chars):
        return None  # no contracted axis: elementwise or an outer product
    if in_chars != dict.fromkeys(axes):
        return None  # an axis indexes nothing -- lifting would silently drop its repetition

    # Canonical operand order. Any order is CORRECT (the einsum terms are emitted in the
    # same order as the connectors that carry them, which is the pairing the expansion's
    # ``*sorted(inputs)`` uses), but it must not depend on the frontend's edge insertion
    # order -- otherwise the same contraction gets a different einsum string depending on
    # how it was spelled. Sorting by operand identity pins it.
    tensors.sort(key=lambda t: (t.array, tuple(t.idx)))

    mapping: dict[str, str] = {}

    def term(idx: list[str]) -> str:
        out = ""
        for i in idx:
            if i == "0":
                continue
            if i not in mapping:
                mapping[i] = EINSUM_CHARS[len(mapping)]
            out += mapping[i]
        return out

    terms = [term(t.idx) for t in tensors]
    einsum_str = f"{','.join(terms)}->{term(value.idx)}"
    # Connector names are chosen so SORTED order is operand order: the ``Einsum``
    # expansion feeds ``*sorted(inputs)`` to the contraction, so the two must agree.
    inputs = [(f"_ein{i:02d}", t.array, _axis_subset(t.idx, axes), None) for i, t in enumerate(tensors)]
    if coeffs:
        inputs.append(("_alpha", coeffs[0].array, _axis_subset(coeffs[0].idx, axes), None))
    beta = _fold_coefficient(sdfg, value.array, value.write_node)
    return EinsumSpec(einsum_str, value.coeff, beta, inputs, ("_out", value.array, _axis_subset(value.idx, axes), None))


def _fold_coefficient(sdfg: SDFG, array: str, write_node: nodes.AccessNode | None) -> float:
    """The ``Einsum``'s ``beta``: ``out = alpha*contraction + beta*out_prior``. The SAME
    three signals ``LiftEinsum.apply`` uses, in the same order -- folding onto undefined
    memory and discarding a meaningful prior both miscompile:

    1. The accumulator carries a zero identity (``setzero``) -> it starts at 0, so
       OVERWRITE (beta=0).
    2. Else there is another in-SDFG writer of the array (a reset / pre-scale loop, e.g.
       atax's ``y[:] = 0`` before the matvec) -> FOLD onto it (beta=1).
    3. Else a non-transient output: the CALLER supplies the prior value that ``+=`` reads
       -> FOLD (beta=1).
    4. Else a fresh, never-written transient (k3mm's ``E = define_local(...)`` accumulator)
       -> OVERWRITE (beta=0) rather than fold uninitialized garbage.
    """
    if write_node is not None and write_node.setzero:
        return 0.0
    # Scoped to THIS sdfg so a same-named array in an unrelated nested SDFG cannot
    # false-positive a prior writer.
    # Only a non-empty in-edge is a write: an ordering memlet transfers nothing.
    has_prior_writer = any(
        n.data == array
        and n is not write_node
        and any(e.data is not None and not e.data.is_empty() for e in st.in_edges(n))
        for st in sdfg.states()
        for n in st.data_nodes()
    )
    return 1.0 if (has_prior_writer or not sdfg.arrays[array].transient) else 0.0


def _direct_transpose(nest: _Nest, sdfg: SDFG, value: _BodyValue) -> TransposeSpec | None:
    """The ``Transpose`` spec for a pure 2-D cross-array transposed copy, or ``None``."""
    if value.accumulates or len(nest.axes) != 2 or len(value.leaves) != 1 or value.coeff != 1:
        return None
    params = dict.fromkeys(a.param for a in nest.axes)
    read = value.leaves[0]
    if dict.fromkeys(read.idx) != params or dict.fromkeys(value.idx) != params:
        return None
    if read.idx != list(reversed(value.idx)):
        return None
    src, dst = read.array, value.array
    if src == dst:
        return None  # in-place symmetrization is LoopToSymmetrize's domain
    sdesc, ddesc = sdfg.arrays.get(src), sdfg.arrays.get(dst)
    if sdesc is None or ddesc is None or len(sdesc.shape) != 2 or len(ddesc.shape) != 2:
        return None
    if sdesc.dtype != ddesc.dtype:
        return None
    ends = {a.param: a.end for a in nest.axes}
    if not sweeps_whole_array(read.idx, ends, sdesc) or not sweeps_whole_array(value.idx, ends, ddesc):
        return None  # the node transposes whole arrays; a partial nest would overwrite the rest
    full = lambda desc: subsets.Range([(0, s - 1, 1) for s in desc.shape])
    return TransposeSpec(src, dst, sdesc.dtype, full(sdesc), full(ddesc))


def _scope_boundary_is_ordered(state: SDFGState, entry: nodes.MapEntry) -> bool:
    """Whether an empty memlet crosses the scope boundary -- deleting the scope would lose it."""
    exit_node = state.exit_node(entry)
    return any(e.data.is_empty() for e in state.in_edges(entry)) or any(
        e.data.is_empty() for e in state.out_edges(exit_node)
    )


def _may_hold_map_contraction(state: SDFGState) -> bool:
    """Cheap NECESSARY condition for a map-form contraction, in ONE node walk: the state
    must hold a map AND a multiplication. Building a state's scope tree is not free, and a
    big weather/physics kernel is mostly maps that multiply nothing."""
    has_map, has_mul = False, False
    for node in state.nodes():
        if isinstance(node, nodes.MapEntry):
            has_map = True
        elif isinstance(node, nodes.Tasklet) and "*" in node.code.as_string:
            has_mul = True
        if has_map and has_mul:
            return True
    return False


def _match_nest(nest: _Nest | None, sdfg: SDFG) -> EinsumSpec | TransposeSpec | None:
    """An ``EinsumSpec`` / ``TransposeSpec`` read directly off ``nest``, or ``None``."""
    if nest is None or len(nest.axes) < 2:
        return None
    value = _body_value(nest, sdfg)
    if value is None:
        return None
    return _direct_einsum(nest, sdfg, value) or _direct_transpose(nest, sdfg, value)


@explicit_cf_compatible
class LoopToEinsum(ppl.Pass):
    """Lift a contraction nest to an ``Einsum`` node, or a transpose nest to a
    ``Transpose`` node. The nest may be spelled as ``LoopRegion``s, as map scopes or as
    a mix of both (see the module docstring)."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        count = self._lift_loops(sdfg)
        count += self._lift_maps(sdfg)
        return count or None

    def _lift_loops(self, sdfg: SDFG) -> int:
        """Lift nests rooted at a ``LoopRegion``: the whole loop is spliced out."""
        # Only the OUTERMOST loop of each nest is a candidate: matching it lifts (or
        # no-ops on) the whole nest. Snapshot upfront -- a lift removes the nest.
        candidates: list[tuple[LoopRegion, SDFG]] = []
        for sd in sdfg.all_sdfgs_recursive():
            candidates.extend(
                (region, sd)
                for region in sd.all_control_flow_regions()
                if isinstance(region, LoopRegion) and region.loop_variable and not _has_loop_ancestor(region)
            )

        count = 0
        for loop, root in candidates:
            parent = loop.parent_graph
            if not isinstance(parent, ControlFlowRegion) or loop not in parent.nodes():
                continue  # already spliced out (defensive)
            spec = _match_nest(_nest_of(loop), root)
            if spec is None:
                continue
            if isinstance(spec, EinsumSpec):
                self._replace_with_einsum(parent, loop, spec)
            else:
                self._replace_with_transpose(parent, loop, spec)
            count += 1
        return count

    def _lift_maps(self, sdfg: SDFG) -> int:
        """Lift nests rooted at a top-level ``MapEntry``: the map scope is replaced in
        place, leaving the rest of its state alone. Maps under a ``LoopRegion`` are
        skipped -- the enclosing loop is the candidate that covers them."""
        candidates: list[tuple[SDFGState, nodes.MapEntry, SDFG]] = []
        for sd in sdfg.all_sdfgs_recursive():
            for state in sd.states():
                if not _may_hold_map_contraction(state):
                    continue  # cheap screen: no map, or no product to contract (one node walk)
                parent = state.parent_graph
                if isinstance(parent, LoopRegion) or (parent is not sd and _region_in_loop(parent)):
                    continue
                candidates.extend(
                    (state, node, sd) for node in state.scope_children()[None] if isinstance(node, nodes.MapEntry)
                )

        count = 0
        for state, entry, root in candidates:
            if entry not in state.nodes():
                continue  # swallowed by an earlier lift in the same state (defensive)
            if _scope_boundary_is_ordered(state, entry):
                continue  # an ordering memlet would be dropped with the scope; refuse
            spec = _match_nest(_nest_of_map(state, entry), root)
            if not isinstance(spec, EinsumSpec):
                continue  # map-form transposes stay maps: no loop scaffold to remove
            self._replace_map_with_einsum(state, entry, spec)
            count += 1
        return count

    def _replace_with_einsum(self, parent: ControlFlowRegion, loop: LoopRegion, spec: EinsumSpec) -> None:
        from dace.libraries.blas.nodes.einsum import Einsum

        state = replace_loop_with_state(parent, loop, loop.label + "_einsum")
        node = Einsum(loop.label + "_einsum")
        node.einsum_str = spec.einsum_str
        node.alpha = spec.alpha
        node.beta = spec.beta
        state.add_node(node)
        for conn, array, subset, dtype in spec.inputs:
            node.add_in_connector(conn, dtype)
            state.add_edge(state.add_read(array), None, node, conn, Memlet(data=array, subset=copy.deepcopy(subset)))
        out_conn, out_array, out_subset, out_dtype = spec.output
        node.add_out_connector(out_conn, out_dtype)
        state.add_edge(
            node, out_conn, state.add_write(out_array), None, Memlet(data=out_array, subset=copy.deepcopy(out_subset))
        )

    def _replace_map_with_einsum(self, state: SDFGState, entry: nodes.MapEntry, spec: EinsumSpec) -> None:
        """Swap a map scope for an ``Einsum`` node in place, reusing the AccessNodes
        the scope already reads from / writes to (the rest of the state is untouched)."""
        from dace.libraries.blas.nodes.einsum import Einsum

        scope = state.scope_subgraph(entry, include_entry=True, include_exit=True)
        node = Einsum(entry.map.label + "_einsum")
        node.einsum_str = spec.einsum_str
        node.alpha = spec.alpha
        node.beta = spec.beta
        state.add_node(node)
        # The operand order the spec fixes is authoritative; find each operand's outside
        # AccessNode by name among the scope's sources (they are unique per array here).
        sources = {e.src.data: e.src for e in state.in_edges(entry) if isinstance(e.src, nodes.AccessNode)}
        for conn, array, subset, dtype in spec.inputs:
            node.add_in_connector(conn, dtype)
            src = sources.get(array) or state.add_read(array)
            state.add_edge(src, None, node, conn, Memlet(data=array, subset=copy.deepcopy(subset)))
        out_conn, out_array, out_subset, out_dtype = spec.output
        node.add_out_connector(out_conn, out_dtype)
        exit_node = state.exit_node(entry)
        dst = next((e.dst for e in state.out_edges(exit_node) if isinstance(e.dst, nodes.AccessNode)), None)
        state.add_edge(
            node,
            out_conn,
            dst if dst is not None else state.add_write(out_array),
            None,
            Memlet(data=out_array, subset=copy.deepcopy(out_subset)),
        )
        state.remove_nodes_from(scope.nodes())

    def _replace_with_transpose(self, parent: ControlFlowRegion, loop: LoopRegion, spec: TransposeSpec) -> None:
        from dace.libraries.linalg.nodes.transpose import Transpose

        state = replace_loop_with_state(parent, loop, loop.label + "_transpose")
        node = Transpose(loop.label + "_transpose", dtype=spec.dtype)
        state.add_node(node)
        state.add_edge(
            state.add_read(spec.src), None, node, "_inp", Memlet(data=spec.src, subset=copy.deepcopy(spec.src_subset))
        )
        state.add_edge(
            node, "_out", state.add_write(spec.dst), None, Memlet(data=spec.dst, subset=copy.deepcopy(spec.dst_subset))
        )


__all__ = ["LoopToEinsum"]
