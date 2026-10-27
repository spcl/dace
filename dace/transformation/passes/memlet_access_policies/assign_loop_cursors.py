# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Loop-cursor analysis: :class:`AssignLoopCursors` attaches
:class:`~dace.sdfg.memlet_access_policies.loop_cursor.LoopCursor` policies to leaf memlets.

:class:`AssignLoopCursors` (analysis, optional, SDFG level): for every leaf memlet whose base element offset is
affine in the induction variable of an enclosing :class:`~dace.sdfg.state.LoopRegion` (in its SDFG or, along the
memlet's :class:`~dace.transformation.passes.memlet_access_policies.helpers.GlobalPath`, in an SDFG enclosing it),
attaches a descriptive :class:`~dace.sdfg.memlet_access_policies.loop_cursor.LoopCursor` (``memlet.access_policy``)
recording the per-iteration step, the loop-invariant base and the lane-dependent part. Nothing else in the SDFG
changes; tuners may inspect or override the records (including forcing or forbidding cursor sharing through
``share_key``). The records are materialized by
:func:`~dace.transformation.passes.memlet_access_policies.loop_cursor_lowering.lower_loop_cursors`.

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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import sympy as sp

from dace import SDFG, properties, symbolic
from dace import data as dt
from dace.memlet import Memlet
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.memlet_access_policies import LoopCursor
from dace.sdfg.scope import is_devicelevel_gpu
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.memlet_access_policies.helpers import (
    GlobalPath,
    base_offset,
    enclosing_loops,
    flat_addressable_array,
    flat_length,
    inner_symbols,
    leaf_edges,
    outer_loop_variables,
    path_root,
    scope_node,
    symbol_named,
    symbol_names,
)


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
    cursor_key: tuple[str, str, str]  #: (loop label, step, class_base without the ``v`` term)

    @property
    def step(self) -> symbolic.SymbolicType:
        return sp.expand(self.delta * self.stride)

    @property
    def base_invariant(self) -> symbolic.SymbolicType:
        """Everything in the base offset except the loop-variable term and the lane part."""
        v = symbol_named(self.beta, self.variable)
        return sp.expand(self.beta - (v * self.delta if v is not None else 0) - self.lane_part)


def decompose(
    desc: dt.Data,
    memlet: Memlet,
    loop: LoopRegion,
    inner: set[str],
    lanes: set[str],
    outer: set[str] | None = None,
    translate: Callable[[Any], sp.Basic] | None = None,
) -> OffsetDecomposition | None:
    """Decompose ``memlet``'s base offset w.r.t. ``loop`` (see module docstring). ``None`` if the memlet is
    not cursor-addressable against this loop: dynamic, offset not affine in the loop variable, step depending on
    inner symbols, or the moved shape varying with the loop variable.

    :param inner: Symbols defined inside the loop body (see
                  :func:`~dace.transformation.passes.memlet_access_policies.helpers.inner_symbols`).
    :param lanes: Thread/lane map parameters of the accessing node (see
                  :func:`~dace.transformation.passes.memlet_access_policies.helpers.lane_symbols`).
    :param outer: Induction variables of the loops enclosing ``loop`` (default: derived from ``loop``).
    :param translate: Restates an expression in the memlet's symbols in the symbols of the loop's SDFG, for a loop
                      enclosing the memlet's (nested) SDFG (see
                      :meth:`~dace.transformation.passes.memlet_access_policies.helpers.GlobalPath.to_level`).
                      The decomposition is in the loop SDFG's symbols.
    """
    var = loop.loop_variable
    if not var or memlet.dynamic or memlet.subset is None:
        return None
    stride = loop_analysis.get_loop_stride(loop)
    if stride is None or var in symbol_names(sp.sympify(stride)):
        return None
    if outer is None:
        outer = outer_loop_variables(loop)

    if translate is None:
        translate = sp.sympify
    beta = sp.expand(translate(base_offset(desc, memlet)))
    v = symbol_named(beta, var)
    if v is None:
        return None
    delta = sp.expand(sp.diff(beta, v))
    if delta.has(sp.Derivative) or delta.has(sp.floor) or delta.has(sp.ceiling):
        return None
    if var in symbol_names(delta) or (symbol_names(delta) & inner):
        return None
    # The moved shape must not vary with the loop variable (the cursor tracks a fixed footprint).
    for dim in memlet.subset.size():
        if var in symbol_names(translate(dim)):
            return None
    for rng in memlet.subset.ndrange() if hasattr(memlet.subset, "ndrange") else []:
        if var in symbol_names(translate(rng[2])):
            return None

    # A term belongs to the cursor only if it varies with this loop, with an enclosing loop (so that the cursor can
    # be chained to that loop's cursor), or with the lane; nest-invariant terms (numbers, SDFG symbols such as
    # strides, kernel/block parameters) and terms that vary within one iteration are per-access immediates.
    tracked = outer | lanes | {var}
    class_terms, imm_terms, lane_terms = [], [], []
    for term in sp.Add.make_args(beta):
        names = symbol_names(term)
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
    path: GlobalPath, edge: MultiConnectorEdge[Memlet], inner_cache: dict[LoopRegion, set[str]]
) -> tuple[int, OffsetDecomposition] | None:
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
        dec = decompose_on_path(path, edge, level, loop, inner_cache, loops[idx + 1 :])
        if dec is not None:
            return level, dec
    return None


def decompose_on_path(
    path: GlobalPath,
    edge: MultiConnectorEdge[Memlet],
    level: int,
    loop: LoopRegion,
    inner_cache: dict[LoopRegion, set[str]],
    outer_loops: list[tuple[int, LoopRegion]] | None = None,
) -> OffsetDecomposition | None:
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


@properties.make_properties
@transformation.explicit_cf_compatible
class AssignLoopCursors(ppl.Pass):
    """Attach :class:`~dace.sdfg.memlet_access_policies.loop_cursor.LoopCursor` policies to leaf memlets whose
    address is affine in an enclosing loop's induction variable (descriptive only; see module docstring). Memlets
    that are not cursor-addressable keep their (default copy-on-access) policy."""

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

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> dict[str, int] | None:
        """
        :return: ``{'assigned': n, 'classes': k, 'skipped': m}`` summed over all SDFGs, or ``None`` if nothing
                 was assigned.
        """
        assigned = skipped = 0
        classes: set[tuple] = set()
        inner_cache: dict[LoopRegion, set[str]] = {}
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
                    if self.scope == "gpu" and not is_devicelevel_gpu(nsdfg, state, scope_node(edge)):
                        continue
                    root, is_read = path_root(state, edge)
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
