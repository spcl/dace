# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Memlet schedule passes: assigning :class:`~dace.sdfg.memlet_schedule.LoopCursor` schedules and lowering all
schedule kinds to ordinary SDFG constructs.

* :class:`ScheduleLoopCursors` (analysis, optional, SDFG level): for every leaf memlet whose base element offset is
  affine in the induction variable of an enclosing :class:`~dace.sdfg.state.LoopRegion`, attaches a descriptive
  :class:`~dace.sdfg.memlet_schedule.LoopCursor` (``memlet.schedule``) recording the per-iteration step, the
  loop-invariant base and the lane-dependent part. Nothing else in the SDFG changes; tuners may inspect or override
  the records (including forcing or forbidding cursor sharing through ``share_key``).

* :class:`LowerMemletSchedules` (codegen window): dispatches every non-default schedule to its kind's
  :meth:`~dace.sdfg.memlet_schedule.MemletSchedule.lower`. For loop cursors (:func:`lower_loop_cursors`) this
  materializes one loop-carried integer *cursor symbol* per cursor class (assigned in the loop's init statement,
  advanced in its update statement), a flat :class:`~dace.data.Reference` per array set once at SDFG entry, and
  rewrites each memlet to ``flat[cursor + immediate]`` (or, for non-contiguous reads, to a per-iteration *window*
  reference). Code generation then emits the loop as ``for (i = ..., cur = ...; ...; i = i + 1, cur = cur + step)``
  and every access as ``flat[cur + imm]``, with no schedule-specific code paths.

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
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Type

import sympy as sp

from dace import SDFG, SDFGState, dtypes, properties, symbolic
from dace import data as dt
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.memlet_schedule import CURSOR_TYPES, LoopCursor, MemletSchedule
from dace.sdfg.state import LoopRegion
from dace.subsets import Range
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.analysis import loop_analysis

_LEAF_NODE_TYPES = (nodes.Tasklet, nodes.LibraryNode, nodes.NestedSDFG)
_GPU_KERNEL_SCHEDULES = (dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.GPU_Persistent)
_LANE_SCHEDULES = (dtypes.ScheduleType.GPU_ThreadBlock, dtypes.ScheduleType.GPU_ThreadBlock_Dynamic)
_INT32_MAX_BYTES = 2**31
_INIT_STATE_LABEL = '__dace_memlet_schedule_init'


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
    """Edges whose memlet produces an address in generated code: connector bindings of tasklets, library nodes
    and nested SDFGs, and access-node-to-access-node copies (lifted to ``CopyLibraryNode`` operands at codegen).
    Scope-crossing edges (map entry/exit connectors), view-defining edges (``views`` connector) and reference
    ``set`` edges are not leaves."""
    for e in state.edges():
        if e.data.is_empty() or e.data.data is None:
            continue
        if e.dst_conn in ('set', 'views') or e.src_conn == 'views':
            continue
        if isinstance(e.dst, _LEAF_NODE_TYPES) or isinstance(e.src, _LEAF_NODE_TYPES):
            yield e
        elif isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.AccessNode):
            yield e


def _leaf_node(edge: MultiConnectorEdge[Memlet]) -> nodes.Node:
    """The node whose scope determines the lane symbols of a leaf edge."""
    return edge.dst if isinstance(edge.dst, _LEAF_NODE_TYPES) else edge.src


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
                result.update(node.map.params if hasattr(node, 'map') else [])
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


def in_gpu_kernel(state: SDFGState, node: nodes.Node) -> bool:
    """True if ``node`` is (transitively, through nested SDFGs) inside a GPU kernel map scope."""
    sdict = state.scope_dict()
    cur = sdict.get(node)
    while cur is not None:
        if cur.map.schedule in _GPU_KERNEL_SCHEDULES:
            return True
        cur = sdict.get(cur)
    sdfg = state.sdfg
    if sdfg.parent is not None and sdfg.parent_nsdfg_node is not None:
        return in_gpu_kernel(sdfg.parent, sdfg.parent_nsdfg_node)
    return False


def extent_bytes_int32(desc: dt.Data) -> bool:
    """True iff the array's total byte extent is provably below 2**31 (so element offsets fit an int32)."""
    try:
        total = sp.sympify(desc.total_size) * desc.dtype.bytes
        return bool(total.is_number and int(total) < _INT32_MAX_BYTES)
    except (TypeError, ValueError):
        return False


def schedulable_array(desc: Optional[dt.Data]) -> bool:
    """Arrays whose base address a flat reference can hold: plain arrays (no views, references, scalars, streams)."""
    return isinstance(desc, dt.Array) and not isinstance(desc, dt.View)


def decompose(desc: dt.Data, memlet: Memlet, loop: LoopRegion, inner: Set[str], lanes: Set[str],
              outer: Optional[Set[str]] = None) -> Optional[OffsetDecomposition]:
    """Decompose ``memlet``'s base offset w.r.t. ``loop`` (see module docstring). ``None`` if the memlet is
    not schedulable against this loop: dynamic, offset not affine in the loop variable, step depending on
    inner symbols, or the moved shape varying with the loop variable.

    :param inner: Symbols defined inside the loop body (see :func:`inner_symbols`).
    :param lanes: Thread/lane map parameters of the accessing node (see :func:`lane_symbols`).
    :param outer: Induction variables of the loops enclosing ``loop`` (default: derived from ``loop``).
    """
    var = loop.loop_variable
    if not var or memlet.dynamic or memlet.subset is None:
        return None
    stride = loop_analysis.get_loop_stride(loop)
    if stride is None or var in _names(sp.sympify(stride)):
        return None
    if outer is None:
        outer = outer_loop_variables(loop)

    beta = sp.expand(base_offset(desc, memlet))
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
        if var in _names(sp.sympify(dim)):
            return None
    for rng in (memlet.subset.ndrange() if hasattr(memlet.subset, 'ndrange') else []):
        if var in _names(sp.sympify(rng[2])):
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


def analyze_edge(state: SDFGState, edge: MultiConnectorEdge[Memlet], loops: List[LoopRegion],
                 inner_cache: Dict[LoopRegion, Set[str]]) -> Optional[OffsetDecomposition]:
    """Decompose the edge's memlet against the innermost enclosing loop whose variable it depends on.

    :param loops: The loops enclosing ``state``, innermost first.
    """
    desc = state.sdfg.arrays.get(edge.data.data)
    if not schedulable_array(desc):
        return None
    root, is_read = _root(state, edge)
    if root is None:
        return None
    if not is_read and flat_length(desc, edge.data.subset) is None:
        return None  # non-contiguous writes are not lowered (window references are read-only)
    lanes = lane_symbols(state, _leaf_node(edge))
    for idx, loop in enumerate(loops):
        if loop not in inner_cache:
            inner_cache[loop] = inner_symbols(loop)
        outer = {l.loop_variable for l in loops[idx + 1:] if l.loop_variable}
        dec = decompose(desc, edge.data, loop, inner_cache[loop], lanes, outer)
        if dec is not None:
            return dec
    return None


# ---------------------------------------------------------------------------------------------------------------
# Analysis pass
# ---------------------------------------------------------------------------------------------------------------
@properties.make_properties
@transformation.explicit_cf_compatible
class ScheduleLoopCursors(ppl.Pass):
    """Attach :class:`~dace.sdfg.memlet_schedule.LoopCursor` schedules to leaf memlets whose address is affine
    in an enclosing loop's induction variable (descriptive only; see module docstring). Memlets that are not
    schedulable keep their (default copy-on-access) schedule."""

    scope = properties.Property(dtype=str,
                                default='gpu',
                                choices=['gpu', 'all'],
                                desc='"gpu": only memlets inside GPU kernel map scopes; "all": every loop.')
    cursor_type = properties.Property(dtype=str,
                                      default='auto',
                                      choices=list(CURSOR_TYPES),
                                      desc='Cursor type recorded on every schedule (planner may override per '
                                      'memlet). "auto" = int32 when the array extent is provably < 2**31.')
    arrays = properties.SetProperty(element_type=str,
                                    default=set(),
                                    desc='If non-empty, only schedule memlets of these arrays.')
    overwrite = properties.Property(dtype=bool,
                                    default=True,
                                    desc='Replace existing non-default schedules (False keeps hand-set records).')

    def __init__(self, **props):
        super().__init__()
        for name, value in props.items():
            if name not in ('scope', 'cursor_type', 'arrays', 'overwrite'):
                raise TypeError(f'ScheduleLoopCursors has no property {name!r}')
            setattr(self, name, value)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.Memlets | ppl.Modifies.CFG | ppl.Modifies.Nodes)

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """
        :return: ``{'scheduled': n, 'classes': k, 'skipped': m}`` summed over all SDFGs, or ``None`` if nothing
                 was scheduled.
        """
        scheduled = skipped = 0
        classes: Set[Tuple] = set()
        for nsdfg in sdfg.all_sdfgs_recursive():
            inner_cache: Dict[LoopRegion, Set[str]] = {}
            for state in nsdfg.states():
                loops = enclosing_loops(state)
                if not loops:
                    continue
                for edge in leaf_edges(state):
                    memlet = edge.data
                    if self.arrays and memlet.data not in self.arrays:
                        continue
                    if not memlet.schedule.is_default and not self.overwrite:
                        continue
                    if not schedulable_array(nsdfg.arrays.get(memlet.data)):
                        continue  # scalars carry no address arithmetic; views/references have no fixed base
                    if self.scope == 'gpu' and not in_gpu_kernel(state, _leaf_node(edge)):
                        continue
                    dec = analyze_edge(state, edge, loops, inner_cache)
                    if dec is None:
                        skipped += 1
                        continue
                    classes.add((nsdfg.cfg_id, memlet.data) + dec.cursor_key)
                    memlet.schedule = LoopCursor(loop=dec.loop.label,
                                                 variable=dec.variable,
                                                 step=dec.step,
                                                 base_invariant=dec.base_invariant,
                                                 lane_part=dec.lane_part,
                                                 cursor_type=self.cursor_type)
                    scheduled += 1
        if scheduled == 0:
            return None
        return {'scheduled': scheduled, 'classes': len(classes), 'skipped': skipped}


# ---------------------------------------------------------------------------------------------------------------
# Lowering pass (codegen window)
# ---------------------------------------------------------------------------------------------------------------
@properties.make_properties
@transformation.explicit_cf_compatible
class LowerMemletSchedules(ppl.Pass):
    """Lower every non-default memlet schedule to ordinary SDFG constructs by dispatching to the schedule kind's
    :meth:`~dace.sdfg.memlet_schedule.MemletSchedule.lower`. Meant to run on the code-generation copy of the
    SDFG, after :class:`~dace.transformation.passes.insert_explicit_copies.InsertExplicitCopies` and before
    library-node expansion; :func:`dace.codegen.codegen.generate_code` does so automatically. Idempotent: memlets
    that are already lowered are left alone."""

    assume_int32 = properties.Property(dtype=bool,
                                       default=False,
                                       desc='Treat "auto" cursors as int32 even when the array extent is not '
                                       'provably < 2**31 (caller guarantees 31-bit offsets).')
    chain_outer_loops = properties.Property(dtype=bool,
                                            default=True,
                                            desc='Initialize an inner-loop cursor from an outer-loop cursor when '
                                            'its entry value is affine in the outer loop variable (one add per '
                                            'loop level, no multiplies), instead of recomputing it per entry.')

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
        :return: The summed counters of the schedule kinds' lowerings (for loop cursors ``{'cursors': n,
                 'memlets': m, 'dropped': d}``), or ``None`` if there was nothing to lower.
        """
        totals: Dict[str, int] = {}
        options = {'assume_int32': self.assume_int32, 'chain_outer_loops': self.chain_outer_loops}
        for nsdfg in sdfg.all_sdfgs_recursive():
            by_kind: Dict[Type[MemletSchedule], List[Tuple[SDFGState, MultiConnectorEdge[Memlet]]]] = {}
            for state in nsdfg.states():
                for edge in leaf_edges(state):
                    if not edge.data.schedule.is_default:
                        by_kind.setdefault(type(edge.data.schedule), []).append((state, edge))
            for kind, entries in by_kind.items():
                for key, value in kind.lower(nsdfg, entries, **options).items():
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
    code = statement if block is None else f'{block.as_string}\n{statement}'
    setattr(loop, which, CodeBlock(code))


def _pystr(expr) -> str:
    return symbolic.symstr(expr, cpp_mode=False)


class _CursorTable:
    """Cursor symbols of one SDFG during lowering: creation (symbol registration + loop statements), sharing per
    cursor class and chaining across loop levels."""

    def __init__(self, sdfg: SDFG, chain_outer_loops: bool):
        self.sdfg = sdfg
        self.chain = chain_outer_loops
        self.classes: Dict[Tuple, Tuple[str, sp.Basic]] = {}  #: class key -> (symbol name, anchor)
        self.used: Set[str] = set(sdfg.symbols.keys())
        self.created = 0

    @staticmethod
    def class_key(loop: LoopRegion, array: str, dtype: dtypes.typeclass, share_key: Optional[str],
                  class_base: sp.Basic) -> Tuple:
        """Cursor-class identity: same loop, array, cursor type, override key, per-iteration step and tracked base
        without the loop-variable term. Members of a class differ only by a loop-invariant immediate."""
        v = _symbol_named(class_base, loop.loop_variable)
        delta = sp.expand(sp.diff(class_base, v)) if v is not None else sp.Integer(0)
        step = sp.expand(delta * loop_analysis.get_loop_stride(loop))
        base_wo_v = sp.expand(class_base - (v * delta if v is not None else 0))
        return (loop.label, array, dtype, share_key, str(step), str(base_wo_v))

    def cursor_for(self, loop: LoopRegion, array: str, dtype: dtypes.typeclass, class_base: sp.Basic,
                   anchor: sp.Basic, share_key: Optional[str]) -> Tuple[str, sp.Basic]:
        """Return (creating if needed) the cursor symbol of ``loop`` that tracks ``class_base`` -- an expression
        affine in the loop variable and free of symbols defined inside the loop body -- together with the
        nest-invariant ``anchor`` the cursor additionally holds (the anchor requested here if the cursor is
        created now, the existing cursor's anchor otherwise; callers add the difference to their immediates).

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
                if (d_out.has(sp.Derivative) or d_out.has(sp.floor) or d_out.has(sp.ceiling)
                        or outer.loop_variable in _names(d_out) or (_names(rest) & inner_symbols(outer))
                        or loop_analysis.get_init_assignment(outer) is None
                        or loop_analysis.get_loop_stride(outer) is None):
                    break
                # The outer cursor absorbs the invariant remainder, so the inner cursor starts exactly at it.
                # A per-memlet ``share_key`` override applies to the memlet's own cursor only, so outer cursors
                # are shared by every nest that needs them.
                outer_name, outer_anchor = self.cursor_for(outer, array, dtype, rest, const, None)
                init_expr = sp.expand(sp.Symbol(outer_name) + const - outer_anchor)
                break

        name = self._name(array, loop)
        self.sdfg.add_symbol(name, dtype)
        _append_statement(loop, 'init_statement', f'{name} = {_pystr(init_expr)}')
        _append_statement(loop, 'update_statement', f'{name} = {_pystr(sp.Symbol(name) + step)}')
        self.classes[key] = (name, anchor)
        self.created += 1
        return name, anchor

    def _name(self, array: str, loop: LoopRegion) -> str:
        """Deterministic identifier ``__dace_cur_<array>_<loop>`` (suffixed on collision, e.g. when one loop
        holds several cursor classes of the same array)."""
        base = re.sub(r'\W', '_', f'__dace_cur_{array}_{loop.label}')
        name, n = base, 1
        while name in self.used:
            name = f'{base}_{n}'
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
        name = f'__dace_flat_{array}'
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
            self.init_state.add_edge(self.init_state.add_read(array), None, self.init_state.add_write(name), 'set',
                                     Memlet.from_array(array, desc))
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
        name = f'__dace_win_{array}_{self.windows}'
        self.windows += 1
        self.sdfg.add_reference(name, sizes, desc.dtype, storage=desc.storage, strides=desc.strides, total_size=total)
        win = state.add_access(name)
        set_memlet = Memlet(data=flat, subset=Range([(index, index, 1)]))
        state.add_edge(self.node(state, flat, True), None, win, 'set', set_memlet)
        return win


def _reroute(state: SDFGState, edge: MultiConnectorEdge[Memlet], new_root: nodes.AccessNode, is_read: bool,
             new_memlet: Memlet) -> None:
    """Replace the memlet path of ``edge`` by one from/to ``new_root`` carrying ``new_memlet`` at the leaf
    (outer memlets are re-propagated through the scopes)."""
    path = state.memlet_path(edge)
    old_root = path[0].src if is_read else path[-1].dst
    if is_read:
        leaf, conn = path[-1].dst, path[-1].dst_conn
        conn_type = leaf.in_connectors.get(conn) if conn is not None else None
        node_seq = [new_root] + [e.dst for e in path]
        src_conn, dst_conn = None, conn
    else:
        leaf, conn = path[0].src, path[0].src_conn
        conn_type = leaf.out_connectors.get(conn) if conn is not None else None
        node_seq = [e.src for e in path] + [new_root]
        src_conn, dst_conn = conn, None
    state.remove_memlet_path(edge, remove_orphans=True)
    if old_root in state.nodes() and state.degree(old_root) == 0:
        state.remove_node(old_root)
    if conn is not None:  # remove_memlet_path drops a connector that no other edge uses
        if is_read and conn not in leaf.in_connectors:
            leaf.add_in_connector(conn, conn_type)
        elif not is_read and conn not in leaf.out_connectors:
            leaf.add_out_connector(conn, conn_type)
    state.add_memlet_path(*node_seq, memlet=new_memlet, src_conn=src_conn, dst_conn=dst_conn, propagate=True)


def lower_loop_cursors(sdfg: SDFG, entries: List[Tuple[SDFGState, MultiConnectorEdge[Memlet]]],
                       assume_int32: bool = False, chain_outer_loops: bool = True, **_) -> Dict[str, int]:
    """Lower the :class:`~dace.sdfg.memlet_schedule.LoopCursor` schedules of one SDFG (see module docstring).

    :param sdfg: The SDFG owning the loops and memlets.
    :param entries: ``(state, edge)`` pairs whose memlets carry ``LoopCursor`` schedules.
    :param assume_int32: Treat ``auto`` cursors as int32 even when the array extent is not provably < 2**31.
    :param chain_outer_loops: Initialize inner cursors from outer cursors (see :meth:`_CursorTable.cursor_for`).
    :return: ``{'cursors': n, 'memlets': m, 'dropped': d}`` (``memlets`` includes already-lowered ones).
    """
    cursors = _CursorTable(sdfg, chain_outer_loops)
    refs = _References(sdfg)
    lowered = dropped = 0

    # Collect per loop, skipping memlets that were already lowered and dropping stale schedules.
    per_loop: Dict[LoopRegion, List[Tuple[SDFGState, MultiConnectorEdge[Memlet]]]] = {}
    for state, edge in entries:
        sched: LoopCursor = edge.data.schedule
        if sched.is_lowered:
            if edge.data.data in (sched.reference, sched.window):
                lowered += 1
            else:
                warnings.warn(f'Memlet "{edge.data}" carries a lowered schedule of another memlet; dropping it.')
                edge.data.schedule = _default()
                dropped += 1
            continue
        loop = {l.label: l for l in enclosing_loops(state)}.get(sched.loop)
        if loop is None or loop.loop_variable != sched.variable:
            warnings.warn(f'Memlet schedule of "{edge.data}" refers to loop "{sched.loop}" (variable '
                          f'{sched.variable}) which no longer encloses it; dropping the schedule.')
            edge.data.schedule = _default()
            dropped += 1
            continue
        per_loop.setdefault(loop, []).append((state, edge))

    # Outer loops first, so a chained inner cursor can share the class cursor an outer memlet created.
    ordered = sorted(per_loop.items(), key=lambda item: len(_enclosing_loops_of_region(item[0])))
    for loop, loop_entries in ordered:
        if loop_analysis.get_init_assignment(loop) is None:
            warnings.warn(f'Cannot lower memlet schedules of loop "{loop.label}": no recognizable init '
                          'assignment; schedules dropped.')
            for _, edge in loop_entries:
                edge.data.schedule = _default()
            dropped += len(loop_entries)
            continue
        inner = inner_symbols(loop)
        # Re-derive every schedule (dropping stale ones), then group the memlets into cursor classes.
        members: Dict[Tuple, List[Tuple[SDFGState, MultiConnectorEdge[Memlet], OffsetDecomposition]]] = {}
        dtypes_of: Dict[Tuple, dtypes.typeclass] = {}
        for state, edge in loop_entries:
            memlet = edge.data
            sched: LoopCursor = memlet.schedule
            desc = sdfg.arrays.get(memlet.data)
            root, is_read = _root(state, edge) if schedulable_array(desc) else (None, False)
            dec = None
            if root is not None and (is_read or flat_length(desc, memlet.subset) is not None):
                dec = decompose(desc, memlet, loop, inner, lane_symbols(state, _leaf_node(edge)))
            if dec is None or sp.expand(dec.step - sched.step) != 0:
                warnings.warn(f'Memlet schedule of "{memlet}" is stale or not lowerable (recorded step '
                              f'{sched.step}, derived {None if dec is None else dec.step}); dropping it.')
                memlet.schedule = _default()
                dropped += 1
                continue
            dtype = _cursor_dtype(sched, desc, assume_int32)
            key = _CursorTable.class_key(loop, memlet.data, dtype, sched.share_key, dec.class_base)
            members.setdefault(key, []).append((state, edge, dec))
            dtypes_of[key] = dtype
        for key, items in members.items():
            state0, edge0, dec0 = items[0]
            array = edge0.data.data
            # Anchor the class at the nest-invariant offset of one member (see _choose_anchor), so that a window's
            # base offset is added once, on loop entry, and each access carries only its small difference.
            anchor = _choose_anchor([_invariant_part(dec.immediate, inner) for _, _, dec in items])
            name, cursor_anchor = cursors.cursor_for(loop, array, dtypes_of[key], dec0.class_base, anchor,
                                                     edge0.data.schedule.share_key)
            for state, edge, dec in items:
                _rewrite(state, edge, dec, name, cursor_anchor, refs)
                lowered += 1
    return {'cursors': cursors.created, 'memlets': lowered, 'dropped': dropped}


def _default() -> MemletSchedule:
    from dace.sdfg.memlet_schedule import CopyOnAccess
    return CopyOnAccess()


def _cursor_dtype(sched: LoopCursor, desc: dt.Data, assume_int32: bool) -> dtypes.typeclass:
    if sched.cursor_type == 'int32':
        return dtypes.int32
    if sched.cursor_type == 'int64':
        return dtypes.int64
    return dtypes.int32 if (assume_int32 or extent_bytes_int32(desc)) else dtypes.int64


def _rewrite(state: SDFGState, edge: MultiConnectorEdge[Memlet], dec: OffsetDecomposition, cursor: str,
             cursor_anchor: sp.Basic, refs: _References) -> None:
    """Rewrite one scheduled memlet to address through the cursor: ``flat[cursor + immediate]`` for contiguous
    memlets, a window reference for non-contiguous reads."""
    old = edge.data
    sched: LoopCursor = old.schedule
    array = old.data
    desc = state.sdfg.arrays[array]
    root, is_read = _root(state, edge)
    immediate = sp.expand(dec.immediate - cursor_anchor)
    index = sp.Symbol(cursor) + immediate
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
    new_memlet = Memlet(data=target,
                        subset=subset,
                        other_subset=old.other_subset,
                        volume=old.volume,
                        dynamic=old.dynamic,
                        wcr=old.wcr,
                        wcr_nonatomic=old.wcr_nonatomic,
                        allow_oob=old.allow_oob,
                        debuginfo=old.debuginfo)
    sched.cursor, sched.reference, sched.window, sched.immediate = cursor, flat, window, immediate
    new_memlet.schedule = sched
    _reroute(state, edge, new_root, is_read, new_memlet)
