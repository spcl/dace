# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Memlet access policy passes: assigning :class:`~dace.sdfg.memlet_access_policy.LoopCursor` policies and lowering all
policy kinds to ordinary SDFG constructs.

* :class:`AssignLoopCursors` (analysis, optional, SDFG level): for every leaf memlet whose base element offset is
  affine in the induction variable of an enclosing :class:`~dace.sdfg.state.LoopRegion` (in its SDFG or, along the
  memlet's :class:`GlobalPath`, in an SDFG enclosing it), attaches a descriptive
  :class:`~dace.sdfg.memlet_access_policy.LoopCursor` (``memlet.access_policy``) recording the per-iteration step, the
  loop-invariant base and the lane-dependent part. Nothing else in the SDFG changes; tuners may inspect or override
  the records (including forcing or forbidding cursor sharing through ``share_key``).

* :class:`LowerMemletAccessPolicies` (codegen window): dispatches every non-default policy to its kind's
  :meth:`~dace.sdfg.memlet_access_policy.MemletAccessPolicy.lower`. For loop cursors (:func:`lower_loop_cursors`) this
  materializes one loop-carried integer *cursor symbol* per cursor class (assigned in the loop's init statement,
  advanced in its update statement), a flat :class:`~dace.data.Reference` per array set once at SDFG entry, and
  rewrites each memlet to ``flat[cursor + immediate]`` (or, for non-contiguous reads, to a per-iteration *window*
  reference). A cursor of a loop in an enclosing SDFG is passed to the nested SDFGs on the way as a symbol, and the
  flat reference is set at the entry of the memlet's own SDFG. Code generation then emits the loop as
  ``for (i = ..., cur = ...; ...; i = i + 1, cur = cur + step)`` and every access as ``flat[cur + imm]``, with no
  policy-specific code paths.

Definitions: for a leaf memlet on array ``A`` with physical base element offset ``beta = sum_k (start_k + offset_k)
* stride_k`` (the flat reference points at the array's physical element 0) and an enclosing loop with variable ``v``
and stride ``c``::

    beta = class_base(v) + immediate
    class_base(v) = terms of beta that depend on v, on the variable of an enclosing loop, or on a lane symbol
    immediate     = numeric constants, nest-invariant symbolic terms (SDFG symbols, kernel/block parameters,
                    array strides), and terms with inner symbols (inner map parameters, inner loop variables)
    step          = d(beta)/dv * c            (must be free of v and of inner symbols)

so a cursor initialized to ``class_base(v_0)`` and advanced by ``step`` per iteration equals ``class_base(v)``
at every iteration; each access then adds only its loop-invariant ``immediate``. Memlets of the same array whose
``class_base`` differ only in the ``v`` term (that is, only by an immediate) share one cursor, which is *anchored*
at one member's nest-invariant offset (the lowest one if it exists, else the coefficient-wise median) so that a
window's base is added once, on loop entry, and the immediates are small differences.
"""

import re
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Type

import sympy as sp

from dace import SDFG, SDFGState, dtypes, properties, symbolic
from dace import data as dt
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.memlet_access_policy import LoopCursor, MemletAccessPolicy
from dace.sdfg.scope import is_devicelevel_gpu
from dace.sdfg.state import LoopRegion
from dace.subsets import Range
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.analysis import loop_analysis

_SCOPE_NODES = (nodes.EntryNode, nodes.ExitNode)
_GPU_KERNEL_SCHEDULES = (dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.GPU_Persistent)
_LANE_SCHEDULES = (dtypes.ScheduleType.GPU_ThreadBlock, dtypes.ScheduleType.GPU_ThreadBlock_Dynamic)
_INT32_MAX_BYTES = 2**31
_INIT_STATE_LABEL = "__dace_memlet_access_policy_init"


# ---------------------------------------------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class OffsetDecomposition:
    """Result of decomposing a memlet's base offset w.r.t. one loop (all offsets in elements)."""

    loop: LoopRegion
    variable: str
    stride: symbolic.SymbolicType  #: loop stride ``c``
    delta: symbolic.SymbolicType  #: d(beta)/dv
    beta: symbolic.SymbolicType  #: the full base offset (relative to the array's logical origin)
    class_base: symbolic.SymbolicType  #: the part of beta a cursor tracks (contains ``v``)
    immediate: symbolic.SymbolicType  #: beta - class_base (loop invariant)
    lane_part: symbolic.SymbolicType  #: part of class_base that depends on lane/thread symbols
    cursor_key: Tuple[str, str, str]  #: (loop label, step, class_base without the ``v`` term)

    @property
    def step(self) -> symbolic.SymbolicType:
        return sp.expand(self.delta * self.stride)

    @property
    def base_invariant(self) -> symbolic.SymbolicType:
        """Everything in the base offset except the loop-variable term and the lane part."""
        v = _symbol_named(self.beta, self.variable)
        return sp.expand(self.beta - (v * self.delta if v is not None else 0) - self.lane_part)


def _symbol_named(expr: sp.Basic, name: str) -> Optional[sp.Symbol]:
    for s in expr.free_symbols:
        if str(s) == name:
            return s
    return None


def _names(expr: sp.Basic) -> Set[str]:
    return {str(s) for s in expr.free_symbols}


def _invariant_part(expr: sp.Basic, inner: Set[str]) -> sp.Basic:
    """The terms of ``expr`` that do not depend on symbols defined inside the loop body."""
    return sp.Add(*[t for t in sp.Add.make_args(sp.expand(expr)) if not (_names(t) & inner)])


def _choose_anchor(candidates: List[sp.Basic]) -> sp.Basic:
    """Pick the offset a cursor class is anchored at, among its members' nest-invariant offsets: the member that
    is smallest in every monomial coefficient if there is one (e.g. the first element of a window; every access
    then carries a non-negative immediate), otherwise the coefficient-wise (lower) median, which centres a
    stencil neighbourhood so that the immediates are the +-stride differences. Deterministic."""
    coeffs: List[Dict[sp.Basic, sp.Basic]] = []
    for cand in candidates:
        d: Dict[sp.Basic, sp.Basic] = {}
        for term in sp.Add.make_args(sp.expand(cand)):
            coeff, monomial = term.as_coeff_Mul()
            d[monomial] = d.get(monomial, sp.Integer(0)) + coeff
        coeffs.append(d)
    monomials = set().union(*coeffs)
    columns = {m: sorted((d.get(m, sp.Integer(0)) for d in coeffs), key=float) for m in monomials}
    lowest = sp.Add(*[col[0] * m for m, col in columns.items()])
    if any(sp.expand(cand - lowest) == 0 for cand in candidates):
        return lowest
    n = len(candidates)
    return sp.Add(*[col[(n - 1) // 2] * m for m, col in columns.items()])


def base_offset(desc: dt.Data, memlet: Memlet) -> sp.Basic:
    """The physical element offset of the first element of ``memlet`` in ``desc`` (descriptor offset included;
    what ``cpp_offset_expr`` emits). The flat reference is set to the array's physical element 0."""
    subset = memlet.subset.offset_new(desc.offset, False)
    return symbolic.pystr_to_symbolic(subset.at([0] * len(desc.strides), desc.strides))


def flat_length(desc: dt.Data, subset) -> Optional[symbolic.SymbolicType]:
    """Number of elements of ``subset`` if it is a contiguous run in memory (at most one dimension of size > 1,
    with unit step and unit stride), otherwise ``None``."""
    if not isinstance(subset, Range):
        return None
    sizes = subset.size()
    wide = [d for d, s in enumerate(sizes) if sp.sympify(s) != 1]
    if not wide:
        return sp.Integer(1)
    if len(wide) > 1:
        return None
    d = wide[0]
    if sp.sympify(subset.ranges[d][2]) != 1 or sp.sympify(desc.strides[d]) != 1:
        return None
    return sp.sympify(sizes[d])


def leaf_edges(state: SDFGState) -> Iterator[MultiConnectorEdge[Memlet]]:
    """Edges whose memlet produces an address in generated code: the innermost edge of each memlet path, i.e. the
    edge whose endpoint away from the memlet's data container is not a scope (map entry/exit) node. This covers
    connector bindings of tasklets and library nodes, access nodes inside scopes fed through a map entry, and
    access-node-to-access-node copies. View-defining edges (``views`` connector) and reference ``set`` edges are not
    leaves, and neither are nested SDFG bindings: a nested SDFG receives the whole container (an equivalent
    descriptor, no offset), so the path continues inside it, to its own leaf memlets (see :class:`GlobalPath`)."""
    for e in state.edges():
        if e.data.is_empty() or e.data.data is None:
            continue
        if e.dst_conn in ("set", "views") or e.src_conn == "views":
            continue
        if isinstance(e.src, nodes.NestedSDFG) or isinstance(e.dst, nodes.NestedSDFG):
            continue
        src_scope, dst_scope = isinstance(e.src, _SCOPE_NODES), isinstance(e.dst, _SCOPE_NODES)
        if src_scope and dst_scope:
            continue  # between two scope nodes: never innermost
        if not src_scope and not dst_scope:
            yield e  # direct binding or copy
            continue
        # One scope endpoint: innermost iff the other endpoint is not the data's own access node (the path root).
        other = e.dst if src_scope else e.src
        if not (isinstance(other, nodes.AccessNode) and other.data == e.data.data):
            yield e


def _scope_node(edge: MultiConnectorEdge[Memlet]) -> nodes.Node:
    """A node whose scope is the scope the leaf memlet is accessed in (used to find lane symbols)."""
    return edge.dst if not isinstance(edge.dst, _SCOPE_NODES) else edge.src


def _root(state: SDFGState, edge: MultiConnectorEdge[Memlet]) -> Tuple[Optional[nodes.AccessNode], bool]:
    """The access node of the memlet's data at the root of the edge's memlet path, and whether the memlet reads
    it. ``(None, False)`` if the data is read and written by the same path or the root is not its access node."""
    path = state.memlet_path(edge)
    data = edge.data.data
    first, last = path[0].src, path[-1].dst
    is_read = isinstance(first, nodes.AccessNode) and first.data == data
    is_write = isinstance(last, nodes.AccessNode) and last.data == data
    if is_read == is_write:
        return None, False
    return (first, True) if is_read else (last, False)


def enclosing_loops(state: SDFGState) -> List[LoopRegion]:
    """Loop regions enclosing ``state`` within its SDFG, innermost first."""
    return _enclosing_loops_of_region(state)


def _enclosing_loops_of_region(block) -> List[LoopRegion]:
    """Loop regions strictly enclosing a control flow block/region within its SDFG, innermost first."""
    result = []
    region = block.parent_graph
    while region is not None and not isinstance(region, SDFG):
        if isinstance(region, LoopRegion):
            result.append(region)
        region = region.parent_graph
    return result


def outer_loop_variables(loop: LoopRegion) -> Set[str]:
    """Induction variables of the loops enclosing ``loop`` (the variables an inner cursor can be chained on)."""
    return {outer.loop_variable for outer in _enclosing_loops_of_region(loop) if outer.loop_variable}


def inner_symbols(loop: LoopRegion) -> Set[str]:
    """Symbols (re)defined inside the loop body: parameters of maps in the body, variables of nested loops,
    and symbols assigned on interstate edges of the body. Terms of a memlet offset that depend on them vary
    *within* an iteration and therefore belong to the per-access immediate, never to the cursor."""
    result: Set[str] = set()
    for region in loop.all_control_flow_regions(recursive=False):
        if isinstance(region, LoopRegion) and region is not loop and region.loop_variable:
            result.add(region.loop_variable)
        for edge in region.edges():
            result.update(edge.data.assignments.keys())
    for state in loop.all_states():
        for node in state.nodes():
            if isinstance(node, nodes.EntryNode):
                result.update(node.map.params if hasattr(node, "map") else [])
    return result


def lane_symbols(state: SDFGState, node: nodes.Node) -> Set[str]:
    """Parameters of enclosing maps that code generation assigns to threads/lanes (GPU_ThreadBlock/GPU_Warp
    maps, or a map directly nested in a GPU kernel map -- the ``[0:W, 0:64]`` wave/lane convention)."""
    sdict = state.scope_dict()
    result: Set[str] = set()
    cur = sdict.get(node)
    while cur is not None:
        parent = sdict.get(cur)
        sched = cur.map.schedule
        if sched in _LANE_SCHEDULES or (parent is not None and parent.map.schedule in _GPU_KERNEL_SCHEDULES):
            result.update(cur.map.params)
        cur = parent
    return result


def extent_bytes_int32(desc: dt.Data) -> bool:
    """True iff the array's total byte extent is provably below 2**31 (so element offsets fit an int32)."""
    try:
        total = sp.sympify(desc.total_size) * desc.dtype.bytes
        return bool(total.is_number and int(total) < _INT32_MAX_BYTES)
    except (TypeError, ValueError):
        return False


def flat_addressable_array(desc: Optional[dt.Data]) -> bool:
    """Arrays whose base address a flat reference can hold: plain arrays (no views, references, scalars, streams)."""
    return isinstance(desc, dt.Array) and not isinstance(desc, dt.View)


@dataclass
class PathLevel:
    """One SDFG along a :class:`GlobalPath`."""

    sdfg: SDFG
    state: SDFGState  #: The state of this SDFG holding the path.
    node: nodes.Node  #: The accessing node at the leaf level, the nested SDFG node of the level below otherwise.
    array: str  #: The name of the path's container in this SDFG.


class GlobalPath:
    """The memlet path of a leaf memlet through the chain of nested SDFGs that pass its container on whole.

    Level 0 is the leaf memlet's SDFG; level ``k + 1`` is the SDFG holding the nested SDFG node of level ``k``,
    for as long as the container is a connector whose descriptor is equivalent to the outer one (the nested SDFG
    contract). An address in the leaf SDFG's symbols is then the same physical element offset in the outer
    container once its symbols are restated through the nested SDFG nodes' symbol mappings, so loops of every
    level enclose the leaf memlet and can carry its cursor.

    Restating an expression outward (:meth:`to_level`) replaces each symbol by its symbol mapping entry; a symbol
    defined inside a nested SDFG (a map parameter, a loop variable, an interstate assignment) has no outer meaning
    and becomes an *atom*, a placeholder that varies within any loop enclosing that SDFG. Restating inward
    (:meth:`to_leaf`) passes every symbol of an outer level down to the leaf SDFG, adding symbol mapping entries
    where the nested SDFGs do not already receive it.
    """

    def __init__(self, state: SDFGState, edge: MultiConnectorEdge[Memlet], is_read: bool):
        self.is_read = is_read
        self.levels: List[PathLevel] = [PathLevel(state.sdfg, state, _scope_node(edge), edge.data.data)]
        self._mappings: List[Optional[Dict[str, sp.Basic]]] = [None]  #: Per level, its node's symbol mapping.
        self._atoms: Dict[str, Tuple[int, sp.Basic]] = {}  #: Atom name -> (level, symbol it stands for).
        sdfg, array = state.sdfg, edge.data.data
        while True:
            node, outer_state = sdfg.parent_nsdfg_node, sdfg.parent
            desc = sdfg.arrays[array]
            if node is None or outer_state is None or desc.transient:
                break
            bindings = (
                outer_state.in_edges_by_connector(node, array)
                if is_read
                else outer_state.out_edges_by_connector(node, array)
            )
            binding = next(iter(bindings), None)
            if binding is None or binding.data.data is None:
                break
            outer_desc = outer_state.sdfg.arrays.get(binding.data.data)
            if not flat_addressable_array(outer_desc):
                break
            if not desc.is_equivalent(outer_desc, symbol_mapping=node.symbol_mapping):
                break
            sdfg, array = outer_state.sdfg, binding.data.data
            self.levels.append(PathLevel(sdfg, outer_state, node, array))
            self._mappings.append(
                {
                    str(k): (v if isinstance(v, sp.Basic) else symbolic.pystr_to_symbolic(v))
                    for k, v in node.symbol_mapping.items()
                }
            )

    @property
    def leaf(self) -> PathLevel:
        return self.levels[0]

    @property
    def atom_names(self) -> Set[str]:
        """Names of the atoms created so far (symbols defined inside a nested SDFG of the path)."""
        return set(self._atoms)

    def loops(self) -> List[Tuple[int, LoopRegion]]:
        """``(level, loop)`` of every loop enclosing the leaf memlet along the path, innermost first."""
        return [(k, loop) for k, level in enumerate(self.levels) for loop in enclosing_loops(level.state)]

    def find_loop(self, label: str, variable: str) -> Optional[Tuple[int, LoopRegion]]:
        """The innermost enclosing loop with the given label and induction variable, or ``None``."""
        for k, loop in self.loops():
            if loop.label == label and loop.loop_variable == variable:
                return k, loop
        return None

    def to_level(self, expr, level: int, source: int = 0) -> sp.Basic:
        """Restate an expression in the symbols of level ``source`` (default: the leaf SDFG) in the symbols of
        ``level`` and atoms."""
        expr = sp.sympify(expr)
        for k in range(source, level):
            mapping = self._mappings[k + 1]
            repl = {}
            for s in expr.free_symbols:
                if s.name in self._atoms:
                    continue
                if s.name in mapping:
                    repl[s] = mapping[s.name]
                else:
                    repl[s] = self._atom(k, s)
            expr = expr.xreplace(repl)
        return expr

    def lanes(self, level: int) -> Set[str]:
        """Thread/lane symbols of the scopes between ``level`` and the leaf, in the symbols of ``level``."""
        result: Set[str] = set()
        for k in range(level + 1):
            lv = self.levels[k]
            for name in lane_symbols(lv.state, lv.node):
                result |= _names(self.to_level(symbolic.pystr_to_symbolic(name), level, source=k))
        return result

    def to_leaf(self, expr, level: int) -> sp.Basic:
        """Restate an expression in the symbols of ``level`` (and atoms) in the leaf SDFG's symbols. Symbols that
        a nested SDFG on the way does not receive yet are added to it and to its node's symbol mapping."""
        expr = self._restore_atoms(sp.sympify(expr), level)
        for k in range(level, 0, -1):
            node: nodes.NestedSDFG = self.levels[k].node
            repl = {}
            for s in expr.free_symbols:
                if s.name in self._atoms:
                    continue
                repl[s] = self._pass_down(node, s, k)
            expr = self._restore_atoms(expr.xreplace(repl), k - 1)
        return expr

    def _pass_down(self, node: nodes.NestedSDFG, sym: sp.Basic, level: int) -> sp.Basic:
        """The symbol of ``node``'s SDFG that receives the level-``level`` symbol ``sym``, created if needed."""
        inner = node.sdfg
        # The node's live mapping: another path may already have passed the symbol down.
        for key, value in node.symbol_mapping.items():
            value = value if isinstance(value, sp.Basic) else symbolic.pystr_to_symbolic(value)
            if isinstance(value, sp.Symbol) and value.name == sym.name and str(key) in inner.symbols:
                return symbolic.symbol(str(key), inner.symbols[str(key)])
        dtype = self.levels[level].sdfg.symbols.get(sym.name, getattr(sym, "dtype", symbolic.DEFAULT_SYMBOL_TYPE))
        base = sym.name if sym.name.startswith("__dace_") else f"__dace_ap_{sym.name}"
        name, n = base, 1
        while name in inner.symbols or name in inner.arrays or name in node.symbol_mapping:
            name = f"{base}_{n}"
            n += 1
        inner.add_symbol(name, dtype)
        node.symbol_mapping[name] = sym
        self._mappings[level][name] = sym
        return symbolic.symbol(name, dtype)

    def _atom(self, level: int, sym: sp.Basic) -> sp.Basic:
        name = f"__dace_atom{level}_{sym.name}"
        self._atoms.setdefault(name, (level, sym))
        return symbolic.symbol(name)

    def _restore_atoms(self, expr: sp.Basic, level: int) -> sp.Basic:
        repl = {s: self._atoms[s.name][1] for s in expr.free_symbols if self._atoms.get(s.name, (None,))[0] == level}
        return expr.xreplace(repl) if repl else expr


def decompose(
    desc: dt.Data,
    memlet: Memlet,
    loop: LoopRegion,
    inner: Set[str],
    lanes: Set[str],
    outer: Optional[Set[str]] = None,
    translate: Optional[Callable[[Any], sp.Basic]] = None,
) -> Optional[OffsetDecomposition]:
    """Decompose ``memlet``'s base offset w.r.t. ``loop`` (see module docstring). ``None`` if the memlet is
    not cursor-addressable against this loop: dynamic, offset not affine in the loop variable, step depending on
    inner symbols, or the moved shape varying with the loop variable.

    :param inner: Symbols defined inside the loop body (see :func:`inner_symbols`).
    :param lanes: Thread/lane map parameters of the accessing node (see :func:`lane_symbols`).
    :param outer: Induction variables of the loops enclosing ``loop`` (default: derived from ``loop``).
    :param translate: Restates an expression in the memlet's symbols in the symbols of the loop's SDFG, for a loop
                      enclosing the memlet's (nested) SDFG (see :meth:`GlobalPath.to_level`). The decomposition
                      is in the loop SDFG's symbols.
    """
    var = loop.loop_variable
    if not var or memlet.dynamic or memlet.subset is None:
        return None
    stride = loop_analysis.get_loop_stride(loop)
    if stride is None or var in _names(sp.sympify(stride)):
        return None
    if outer is None:
        outer = outer_loop_variables(loop)

    if translate is None:
        translate = sp.sympify
    beta = sp.expand(translate(base_offset(desc, memlet)))
    v = _symbol_named(beta, var)
    if v is None:
        return None
    delta = sp.expand(sp.diff(beta, v))
    if delta.has(sp.Derivative) or delta.has(sp.floor) or delta.has(sp.ceiling):
        return None
    if var in _names(delta) or (_names(delta) & inner):
        return None
    # The moved shape must not vary with the loop variable (the cursor tracks a fixed footprint).
    for dim in memlet.subset.size():
        if var in _names(translate(dim)):
            return None
    for rng in memlet.subset.ndrange() if hasattr(memlet.subset, "ndrange") else []:
        if var in _names(translate(rng[2])):
            return None

    # A term belongs to the cursor only if it varies with this loop, with an enclosing loop (so that the cursor can
    # be chained to that loop's cursor), or with the lane; nest-invariant terms (numbers, SDFG symbols such as
    # strides, kernel/block parameters) and terms that vary within one iteration are per-access immediates.
    tracked = outer | lanes | {var}
    class_terms, imm_terms, lane_terms = [], [], []
    for term in sp.Add.make_args(beta):
        names = _names(term)
        if names & inner or not (names & tracked):
            imm_terms.append(term)
        else:
            class_terms.append(term)
            if names & lanes:
                lane_terms.append(term)
    class_base = sp.Add(*class_terms)
    immediate = sp.Add(*imm_terms)
    lane_part = sp.Add(*lane_terms)
    step = sp.expand(delta * stride)
    key = (loop.label, str(step), str(sp.expand(class_base - v * delta)))
    return OffsetDecomposition(loop, var, stride, delta, beta, class_base, immediate, lane_part, key)


def analyze_edge(
    path: GlobalPath, edge: MultiConnectorEdge[Memlet], inner_cache: Dict[LoopRegion, Set[str]]
) -> Optional[Tuple[int, OffsetDecomposition]]:
    """Decompose the edge's memlet against the innermost loop along its global path whose variable it depends on.

    :return: The level of the loop's SDFG on the path and the decomposition (in that SDFG's symbols), or ``None``.
    """
    desc = path.leaf.sdfg.arrays.get(edge.data.data)
    if not flat_addressable_array(desc):
        return None
    if not path.is_read and flat_length(desc, edge.data.subset) is None:
        return None  # non-contiguous writes are not lowered (window references are read-only)
    loops = path.loops()
    for idx, (level, loop) in enumerate(loops):
        dec = _decompose_on_path(path, edge, level, loop, inner_cache, loops[idx + 1 :])
        if dec is not None:
            return level, dec
    return None


def _decompose_on_path(
    path: GlobalPath,
    edge: MultiConnectorEdge[Memlet],
    level: int,
    loop: LoopRegion,
    inner_cache: Dict[LoopRegion, Set[str]],
    outer_loops: Optional[List[Tuple[int, LoopRegion]]] = None,
) -> Optional[OffsetDecomposition]:
    """Decompose the leaf memlet of ``path`` against ``loop`` of SDFG level ``level`` (see :func:`decompose`).
    Symbols defined inside the nested SDFGs below that level (atoms) vary within an iteration of ``loop``."""
    if loop not in inner_cache:
        inner_cache[loop] = inner_symbols(loop)
    desc = path.leaf.sdfg.arrays[edge.data.data]
    # Restating the offset creates the atoms (its symbols defined below ``level``) before they are needed as inner
    # symbols; the subset sizes and steps only matter for whether they depend on the loop variable.
    path.to_level(base_offset(desc, edge.data), level)
    if outer_loops is None:
        outer = outer_loop_variables(loop)
    else:
        outer = {l.loop_variable for k, l in outer_loops if k == level and l.loop_variable}
    return decompose(
        desc,
        edge.data,
        loop,
        inner_cache[loop] | path.atom_names,
        path.lanes(level),
        outer,
        translate=lambda expr: path.to_level(expr, level),
    )


# ---------------------------------------------------------------------------------------------------------------
# Analysis pass
# ---------------------------------------------------------------------------------------------------------------
@properties.make_properties
@transformation.explicit_cf_compatible
class AssignLoopCursors(ppl.Pass):
    """Attach :class:`~dace.sdfg.memlet_access_policy.LoopCursor` policies to leaf memlets whose address is affine
    in an enclosing loop's induction variable (descriptive only; see module docstring). Memlets that are not
    cursor-addressable keep their (default copy-on-access) policy."""

    scope = properties.Property(
        dtype=str,
        default="gpu",
        choices=["gpu", "all"],
        desc='"gpu": only memlets inside GPU kernel map scopes; "all": every loop.',
    )
    cursor_type = properties.TypeClassProperty(
        default=None,
        allow_none=True,
        desc="Integer type of the cursor symbols (planner may override per "
        "memlet). None = int32 when the array extent is provably < 2**31, "
        "else int64.",
    )
    arrays = properties.SetProperty(
        element_type=str, default=set(), desc="If non-empty, only assign policies to memlets of these arrays."
    )
    overwrite = properties.Property(
        dtype=bool, default=True, desc="Replace existing non-default policies (False keeps hand-set records)."
    )

    def __init__(self, **props):
        super().__init__()
        for name, value in props.items():
            if name not in ("scope", "cursor_type", "arrays", "overwrite"):
                raise TypeError(f"AssignLoopCursors has no property {name!r}")
            setattr(self, name, value)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.Memlets | ppl.Modifies.CFG | ppl.Modifies.Nodes)

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """
        :return: ``{'assigned': n, 'classes': k, 'skipped': m}`` summed over all SDFGs, or ``None`` if nothing
                 was assigned.
        """
        assigned = skipped = 0
        classes: Set[Tuple] = set()
        inner_cache: Dict[LoopRegion, Set[str]] = {}
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                for edge in leaf_edges(state):
                    memlet = edge.data
                    if self.arrays and memlet.data not in self.arrays:
                        continue
                    if not memlet.access_policy.is_default and not self.overwrite:
                        continue
                    if not flat_addressable_array(nsdfg.arrays.get(memlet.data)):
                        continue  # scalars carry no address arithmetic; views/references have no fixed base
                    if self.scope == "gpu" and not is_devicelevel_gpu(nsdfg, state, _scope_node(edge)):
                        continue
                    root, is_read = _root(state, edge)
                    if root is None:
                        skipped += bool(enclosing_loops(state))
                        continue
                    path = GlobalPath(state, edge, is_read)
                    if not path.loops():
                        continue
                    found = analyze_edge(path, edge, inner_cache)
                    if found is None:
                        skipped += 1
                        continue
                    level, dec = found
                    classes.add((id(path.levels[level].sdfg), path.levels[level].array) + dec.cursor_key)
                    memlet.access_policy = LoopCursor(
                        loop=dec.loop.label,
                        variable=dec.variable,
                        step=dec.step,
                        base_invariant=dec.base_invariant,
                        lane_part=dec.lane_part,
                        cursor_type=self.cursor_type,
                    )
                    assigned += 1
        if assigned == 0:
            return None
        return {"assigned": assigned, "classes": len(classes), "skipped": skipped}


# ---------------------------------------------------------------------------------------------------------------
# Lowering pass (codegen window)
# ---------------------------------------------------------------------------------------------------------------
@properties.make_properties
@transformation.explicit_cf_compatible
class LowerMemletAccessPolicies(ppl.Pass):
    """Lower every non-default memlet access policy to ordinary SDFG constructs by dispatching to the policy kind's
    :meth:`~dace.sdfg.memlet_access_policy.MemletAccessPolicy.lower`. Meant to run on the code-generation copy of the
    SDFG, after :class:`~dace.transformation.passes.insert_explicit_copies.InsertExplicitCopies` and before
    library-node expansion; :func:`dace.codegen.codegen.generate_code` does so automatically. Idempotent: memlets
    that are already lowered are left alone."""

    assume_int32 = properties.Property(
        dtype=bool,
        default=False,
        desc='Treat "auto" cursors as int32 even when the array extent is not '
        "provably < 2**31 (caller guarantees 31-bit offsets).",
    )
    chain_outer_loops = properties.Property(
        dtype=bool,
        default=True,
        desc="Initialize an inner-loop cursor from an outer-loop cursor when "
        "its entry value is affine in the outer loop variable (one add per "
        "loop level, no multiplies), instead of recomputing it per entry.",
    )

    def __init__(self, assume_int32: bool = False, chain_outer_loops: bool = True):
        super().__init__()
        self.assume_int32 = assume_int32
        self.chain_outer_loops = chain_outer_loops

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """
        :return: The summed counters of the policy kinds' lowerings (for loop cursors ``{'cursors': n,
                 'memlets': m, 'dropped': d}``), or ``None`` if there was nothing to lower.
        """
        totals: Dict[str, int] = {}
        options = {"assume_int32": self.assume_int32, "chain_outer_loops": self.chain_outer_loops}
        # Policies are lowered per kind over the whole SDFG tree: a memlet in a nested SDFG may be addressed
        # relative to a loop of an enclosing SDFG along its memlet path.
        by_kind: Dict[Type[MemletAccessPolicy], List[Tuple[SDFGState, MultiConnectorEdge[Memlet]]]] = {}
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                for edge in leaf_edges(state):
                    if not edge.data.access_policy.is_default:
                        by_kind.setdefault(type(edge.data.access_policy), []).append((state, edge))
        for kind, entries in by_kind.items():
            for key, value in kind.lower(sdfg, entries, **options).items():
                totals[key] = totals.get(key, 0) + value
        return totals or None


# ---------------------------------------------------------------------------------------------------------------
# Loop-cursor lowering
# ---------------------------------------------------------------------------------------------------------------
def _statement_assignments(block: Optional[CodeBlock]) -> Dict[str, str]:
    return loop_analysis.get_assignments(block)


def _append_statement(loop: LoopRegion, which: str, statement: str) -> None:
    """Append ``statement`` to the loop's init or update code block (code blocks may hold several statements)."""
    block: Optional[CodeBlock] = getattr(loop, which)
    code = statement if block is None else f"{block.as_string}\n{statement}"
    setattr(loop, which, CodeBlock(code))


def _pystr(expr) -> str:
    return symbolic.symstr(expr, cpp_mode=False)


class _CursorTable:
    """Cursor symbols of one SDFG during lowering: creation (symbol registration + loop statements), sharing per
    cursor class and chaining across loop levels."""

    def __init__(self, sdfg: SDFG, chain_outer_loops: bool):
        self.sdfg = sdfg
        self.chain = chain_outer_loops
        self.classes: Dict[Tuple, Tuple[symbolic.symbol, sp.Basic]] = {}  #: class key -> (cursor symbol, anchor)
        self.used: Set[str] = set(sdfg.symbols.keys())
        self.created = 0

    @staticmethod
    def class_key(
        loop: LoopRegion, array: str, dtype: dtypes.typeclass, share_key: Optional[str], class_base: sp.Basic
    ) -> Tuple:
        """Cursor-class identity: same loop, array, cursor type, override key, per-iteration step and tracked base
        without the loop-variable term. Members of a class differ only by a loop-invariant immediate."""
        v = _symbol_named(class_base, loop.loop_variable)
        delta = sp.expand(sp.diff(class_base, v)) if v is not None else sp.Integer(0)
        step = sp.expand(delta * loop_analysis.get_loop_stride(loop))
        base_wo_v = sp.expand(class_base - (v * delta if v is not None else 0))
        return (loop.label, array, dtype, share_key, str(step), str(base_wo_v))

    def cursor_for(
        self,
        loop: LoopRegion,
        array: str,
        dtype: dtypes.typeclass,
        class_base: sp.Basic,
        anchor: sp.Basic,
        share_key: Optional[str],
    ) -> Tuple[symbolic.symbol, sp.Basic]:
        """Return (creating if needed) the cursor symbol of ``loop`` that tracks ``class_base`` -- an expression
        affine in the loop variable and free of symbols defined inside the loop body -- together with the
        nest-invariant ``anchor`` the cursor additionally holds (the anchor requested here if the cursor is
        created now, the existing cursor's anchor otherwise; callers add the difference to their immediates).
        The symbol is the typed DaCe symbol registered in the SDFG; callers use it directly in expressions.

        Chaining: the cursor's entry value ``class_base(v_0) + anchor`` is itself an address that may be affine in
        an enclosing loop's variable. Instead of recomputing it at every entry of ``loop`` (a multiply-add per
        outer iteration), an outer cursor tracking that expression is created (or shared) on the enclosing loop
        and the inner cursor is initialized from it (``inner = outer``, or ``outer + const``). Applied recursively
        outward, so a loop nest walks its arrays with one add per loop level and no multiplies, without assuming
        anything about inner trip counts.
        """
        key = self.class_key(loop, array, dtype, share_key, class_base)
        if key in self.classes:
            return self.classes[key]

        var = loop.loop_variable
        v = _symbol_named(class_base, var)
        delta = sp.expand(sp.diff(class_base, v)) if v is not None else sp.Integer(0)
        step = sp.expand(delta * loop_analysis.get_loop_stride(loop))
        init_value = symbolic.pystr_to_symbolic(loop_analysis.get_init_assignment(loop))
        init = sp.expand((class_base.subs(v, init_value) if v is not None else class_base) + anchor)
        # Split the entry value into the part that varies with an enclosing loop (chained to that loop's cursor)
        # and the nest-invariant remainder (a constant or a symbolic expression, added once per entry).
        outer_loops = _enclosing_loops_of_region(loop)
        outer_vars = {outer.loop_variable for outer in outer_loops if outer.loop_variable}
        rest = sp.Add(*[t for t in sp.Add.make_args(init) if _names(t) & outer_vars])
        const = sp.expand(init - rest)
        init_expr = init
        if self.chain and rest != 0:
            for outer in outer_loops:
                vo = _symbol_named(rest, outer.loop_variable) if outer.loop_variable else None
                if vo is None:
                    continue
                d_out = sp.expand(sp.diff(rest, vo))
                if (
                    d_out.has(sp.Derivative)
                    or d_out.has(sp.floor)
                    or d_out.has(sp.ceiling)
                    or outer.loop_variable in _names(d_out)
                    or (_names(rest) & inner_symbols(outer))
                    or loop_analysis.get_init_assignment(outer) is None
                    or loop_analysis.get_loop_stride(outer) is None
                ):
                    break
                # The outer cursor absorbs the invariant remainder, so the inner cursor starts exactly at it.
                # A per-memlet ``share_key`` override applies to the memlet's own cursor only, so outer cursors
                # are shared by every nest that needs them.
                outer_cursor, outer_anchor = self.cursor_for(outer, array, dtype, rest, const, None)
                init_expr = sp.expand(outer_cursor + const - outer_anchor)
                break

        name = self._name(array, loop)
        self.sdfg.add_symbol(name, dtype)
        cursor = symbolic.symbol(name, dtype)
        _append_statement(loop, "init_statement", f"{name} = {_pystr(init_expr)}")
        _append_statement(loop, "update_statement", f"{name} = {_pystr(cursor + step)}")
        self.classes[key] = (cursor, anchor)
        self.created += 1
        return cursor, anchor

    def _name(self, array: str, loop: LoopRegion) -> str:
        """Deterministic identifier ``__dace_cur_<array>_<loop>`` (suffixed on collision, e.g. when one loop
        holds several cursor classes of the same array)."""
        base = re.sub(r"\W", "_", f"__dace_cur_{array}_{loop.label}")
        name, n = base, 1
        while name in self.used:
            name = f"{base}_{n}"
            n += 1
        self.used.add(name)
        return name


class _References:
    """Flat base references (one per array per SDFG, set once at SDFG entry) and per-memlet window references."""

    def __init__(self, sdfg: SDFG):
        self.sdfg = sdfg
        self.init_state: Optional[SDFGState] = None
        self.flat: Dict[str, str] = {}
        self.nodes: Dict[Tuple[SDFGState, str, bool], nodes.AccessNode] = {}
        self.windows = 0

    def flat_reference(self, array: str) -> str:
        """The flat reference of ``array`` (created and set at SDFG entry on first use)."""
        if array in self.flat:
            return self.flat[array]
        desc = self.sdfg.arrays[array]
        name = f"__dace_flat_{array}"
        if name not in self.sdfg.arrays:
            self.sdfg.add_reference(name, [desc.total_size], desc.dtype, storage=desc.storage)
            if self.init_state is None:
                start = self.sdfg.start_block
                if start.label == _INIT_STATE_LABEL and isinstance(start, SDFGState):
                    self.init_state = start
                else:
                    self.init_state = self.sdfg.add_state_before(start, _INIT_STATE_LABEL, is_start_block=True)
            # ``from_array`` covers the whole array starting at index ``-offset``, i.e. the set points the flat
            # reference at the array's physical element 0 (cursors are physical offsets, see ``base_offset``).
            self.init_state.add_edge(
                self.init_state.add_read(array),
                None,
                self.init_state.add_write(name),
                "set",
                Memlet.from_array(array, desc),
            )
        self.flat[array] = name
        return name

    def node(self, state: SDFGState, reference: str, is_read: bool) -> nodes.AccessNode:
        """One shared read (or write) access node of ``reference`` per state."""
        key = (state, reference, is_read)
        if key not in self.nodes:
            self.nodes[key] = state.add_read(reference) if is_read else state.add_write(reference)
        return self.nodes[key]

    def window(self, state: SDFGState, array: str, subset: Range, index: sp.Basic) -> nodes.AccessNode:
        """A window reference with the memlet's shape and the array's strides, set (in ``state``) to the flat
        reference at element ``index``; returns its access node."""
        desc = self.sdfg.arrays[array]
        flat = self.flat_reference(array)
        sizes = subset.size()
        total = sum((sp.sympify(s) - 1) * sp.sympify(st) for s, st in zip(sizes, desc.strides)) + 1
        name = f"__dace_win_{array}_{self.windows}"
        self.windows += 1
        self.sdfg.add_reference(name, sizes, desc.dtype, storage=desc.storage, strides=desc.strides, total_size=total)
        win = state.add_access(name)
        set_memlet = Memlet(data=flat, subset=Range([(index, index, 1)]))
        state.add_edge(self.node(state, flat, True), None, win, "set", set_memlet)
        return win


def _reroute(
    state: SDFGState, edge: MultiConnectorEdge[Memlet], new_root: nodes.AccessNode, is_read: bool, new_memlet: Memlet
) -> None:
    """Replace the memlet path of ``edge`` by one from/to ``new_root`` carrying ``new_memlet`` at the leaf
    (outer memlets are re-propagated through the scopes). The new path is added before the old one is removed,
    so scope nodes and the leaf connector are never orphaned in between."""
    path = state.memlet_path(edge)
    if is_read:
        old_root = path[0].src
        node_seq = [new_root] + [e.dst for e in path]
        src_conn, dst_conn = None, path[-1].dst_conn
    else:
        old_root = path[-1].dst
        node_seq = [e.src for e in path] + [new_root]
        src_conn, dst_conn = path[0].src_conn, None
    state.add_memlet_path(*node_seq, memlet=new_memlet, src_conn=src_conn, dst_conn=dst_conn, propagate=True)
    state.remove_memlet_path(edge, remove_orphans=True)
    if old_root in state.nodes() and state.degree(old_root) == 0:
        state.remove_node(old_root)


def lower_loop_cursors(
    sdfg: SDFG,
    entries: List[Tuple[SDFGState, MultiConnectorEdge[Memlet]]],
    assume_int32: bool = False,
    chain_outer_loops: bool = True,
    **_,
) -> Dict[str, int]:
    """Lower the :class:`~dace.sdfg.memlet_access_policy.LoopCursor` policies of an SDFG and its nested SDFGs (see
    module docstring). A cursor is created in the SDFG of its loop and passed down, as a symbol, to the nested SDFGs
    along the memlet path of each memlet it addresses; flat references are created in the SDFG of the memlet.

    :param sdfg: The root SDFG.
    :param entries: ``(state, edge)`` pairs whose memlets carry ``LoopCursor`` policies, in any SDFG of the tree.
    :param assume_int32: Treat ``auto`` cursors as int32 even when the array extent is not provably < 2**31.
    :param chain_outer_loops: Initialize inner cursors from outer cursors (see :meth:`_CursorTable.cursor_for`).
    :return: ``{'cursors': n, 'memlets': m, 'dropped': d}`` (``memlets`` includes already-lowered ones).
    """
    tables: Dict[SDFG, _CursorTable] = {}
    refs: Dict[SDFG, _References] = {}
    lowered = dropped = 0

    # Collect per loop, skipping memlets that were already lowered and dropping stale policies.
    per_loop: Dict[LoopRegion, List[Tuple[SDFGState, MultiConnectorEdge[Memlet], GlobalPath, int]]] = {}
    for state, edge in entries:
        policy: LoopCursor = edge.data.access_policy
        if policy.is_lowered:
            if edge.data.data in (policy.reference, policy.window):
                lowered += 1
            else:
                warnings.warn(f'Memlet "{edge.data}" carries a lowered policy of another memlet; dropping it.')
                edge.data.access_policy = _default()
                dropped += 1
            continue
        root, is_read = _root(state, edge)
        path = GlobalPath(state, edge, is_read) if root is not None else None
        found = path.find_loop(policy.loop, policy.variable) if path is not None else None
        if found is None:
            warnings.warn(
                f'Memlet access policy of "{edge.data}" refers to loop "{policy.loop}" (variable '
                f"{policy.variable}) which no longer encloses it; dropping the policy."
            )
            edge.data.access_policy = _default()
            dropped += 1
            continue
        level, loop = found
        per_loop.setdefault(loop, []).append((state, edge, path, level))

    # Outer loops first, so a chained inner cursor can share the class cursor an outer memlet created.
    inner_cache: Dict[LoopRegion, Set[str]] = {}
    ordered = sorted(per_loop.items(), key=lambda item: len(_enclosing_loops_of_region(item[0])))
    for loop, loop_entries in ordered:
        if loop_analysis.get_init_assignment(loop) is None:
            warnings.warn(
                f'Cannot lower memlet access policies of loop "{loop.label}": no recognizable init '
                "assignment; policies dropped."
            )
            for _, edge, _, _ in loop_entries:
                edge.data.access_policy = _default()
            dropped += len(loop_entries)
            continue
        loop_sdfg = loop.sdfg
        if loop_sdfg not in tables:
            tables[loop_sdfg] = _CursorTable(loop_sdfg, chain_outer_loops)
        # Re-derive every policy (dropping stale ones), then group the memlets into cursor classes.
        members: Dict[Tuple, List[Tuple[SDFGState, MultiConnectorEdge[Memlet], GlobalPath, int, OffsetDecomposition]]]
        members = {}
        dtypes_of: Dict[Tuple, dtypes.typeclass] = {}
        for state, edge, path, level in loop_entries:
            memlet = edge.data
            policy: LoopCursor = memlet.access_policy
            desc = state.sdfg.arrays.get(memlet.data)
            dec = None
            if flat_addressable_array(desc) and (path.is_read or flat_length(desc, memlet.subset) is not None):
                dec = _decompose_on_path(path, edge, level, loop, inner_cache)
            if dec is None or sp.expand(dec.step - policy.step) != 0:
                warnings.warn(
                    f'Memlet access policy of "{memlet}" is stale or not lowerable (recorded step '
                    f"{policy.step}, derived {None if dec is None else dec.step}); dropping it."
                )
                memlet.access_policy = _default()
                dropped += 1
                continue
            dtype = _cursor_dtype(policy, desc, assume_int32)
            array = path.levels[level].array
            key = _CursorTable.class_key(loop, array, dtype, policy.share_key, dec.class_base)
            members.setdefault(key, []).append((state, edge, path, level, dec))
            dtypes_of[key] = dtype
        for key, items in members.items():
            _, edge0, _, _, dec0 = items[0]
            array = key[1]
            # Anchor the class at the nest-invariant offset of one member (see _choose_anchor), so that a window's
            # base offset is added once, on loop entry, and each access carries only its small difference.
            anchor = _choose_anchor(
                [_invariant_part(dec.immediate, inner_cache[loop] | path.atom_names) for _, _, path, _, dec in items]
            )
            cursor, cursor_anchor = tables[loop_sdfg].cursor_for(
                loop, array, dtypes_of[key], dec0.class_base, anchor, edge0.data.access_policy.share_key
            )
            for state, edge, path, level, dec in items:
                if state.sdfg not in refs:
                    refs[state.sdfg] = _References(state.sdfg)
                _rewrite(
                    state,
                    edge,
                    path.to_leaf(cursor, level),
                    path.to_leaf(sp.expand(dec.immediate - cursor_anchor), level),
                    refs[state.sdfg],
                )
                lowered += 1
    return {"cursors": sum(t.created for t in tables.values()), "memlets": lowered, "dropped": dropped}


def _default() -> MemletAccessPolicy:
    from dace.sdfg.memlet_access_policy import CopyOnAccess

    return CopyOnAccess()


def _cursor_dtype(policy: LoopCursor, desc: dt.Data, assume_int32: bool) -> dtypes.typeclass:
    if policy.cursor_type is not None:
        return policy.cursor_type
    return dtypes.int32 if (assume_int32 or extent_bytes_int32(desc)) else dtypes.int64


def _rewrite(
    state: SDFGState,
    edge: MultiConnectorEdge[Memlet],
    cursor: sp.Basic,
    immediate: sp.Basic,
    refs: _References,
) -> None:
    """Rewrite one loop-cursor memlet to address through the cursor: ``flat[cursor + immediate]`` for contiguous
    memlets, a window reference for non-contiguous reads.

    :param cursor: The cursor symbol, in the symbols of the memlet's SDFG.
    :param immediate: The memlet's offset relative to the cursor, in the symbols of the memlet's SDFG.
    """
    old = edge.data
    policy: LoopCursor = old.access_policy
    array = old.data
    desc = state.sdfg.arrays[array]
    root, is_read = _root(state, edge)
    index = cursor + immediate
    flat = refs.flat_reference(array)
    length = flat_length(desc, old.subset)
    if length is not None:
        subset = Range([(index, index + length - 1, 1)])
        new_root = refs.node(state, flat, is_read)
        target, window = flat, None
    else:
        new_root = refs.window(state, array, old.subset, index)
        subset = Range([(0, sp.sympify(s) - 1, 1) for s in old.subset.size()])
        target, window = new_root.data, new_root.data
    new_memlet = Memlet(
        data=target,
        subset=subset,
        other_subset=old.other_subset,
        volume=old.volume,
        dynamic=old.dynamic,
        wcr=old.wcr,
        wcr_nonatomic=old.wcr_nonatomic,
        allow_oob=old.allow_oob,
        debuginfo=old.debuginfo,
    )
    policy.cursor, policy.reference, policy.window, policy.immediate = str(cursor), flat, window, immediate
    new_memlet.access_policy = policy
    _reroute(state, edge, new_root, is_read, new_memlet)
