# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Promote every non-scalar transient array to the root SDFG.

For each iteration:

1. Find a non-scalar transient ``Array`` that still lives in a
   ``NestedSDFG.sdfg.arrays`` table.
2. Lift it one level up to the enclosing SDFG. If the ``NestedSDFG``
   node sits inside a ``MapEntry`` / ``MapExit`` scope, the enclosing
   map's parameters extend the descriptor's shape so every concurrent
   map-iteration writes into a disjoint slab. Memlets and connectors are
   extended along the way.
3. Stop when no transient non-scalar array lives in any nested SDFG.

Layout preservation
-------------------
The pass inspects each lifted descriptor's strides via
``Array.is_packed_fortran_strides`` / ``is_packed_c_strides``:

 * Fortran-packed (column-major, strides grow leftward): map-iter dims
   are APPENDED (they become the slowest dims). Default when the strides
   match neither (e.g. a freshly materialised descriptor with symbolic
   strides).
 * C-packed (row-major, strides grow rightward): map-iter dims are
   PREPENDED. Each slab stays contiguous in memory in either case.

The outer descriptor's new strides use the same packed layout as the
inner one so the inner NSDFG's stride-dependent code still sees the
expected layout.

Scalars (``shape == (1,)``) are treated as thread-local scratch and left
untouched.

``map_range_suggestions``
-------------------------
Optional mapping from ``(begin_symbol, end_symbol)`` (strings
identifying a map range's begin/end variables) to a replacement
dimension-size expression. Used when a map iterates
``begin_symbol .. end_symbol`` with bounds that are assigned by
interstate edges -- without a suggestion the lifted shape would use
``end - begin + 1`` and the allocator couldn't size the array at SDFG
entry. With a suggestion (e.g. ``"nblks_c"``) the new dim uses the
suggested size and each memlet indexes that dim via ``(param - begin)``,
so every map iteration still writes a disjoint slab.
"""
import copy
from typing import Any, Dict, List, Optional, Set, Tuple

import dace
from dace import SDFG, data, dtypes, memlet as mm, subsets, symbolic
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl


RangeSuggestions = Dict[Tuple[str, str], str]


class LiftTransients(ppl.Pass):
    """Lift every non-scalar transient array to the root SDFG.

    See module docstring for the full algorithm. Use the constructor's
    ``map_range_suggestions`` parameter to pre-declare how to size any
    map dim whose loop bounds are assigned via interstate edges."""

    CATEGORY: str = 'Optimization Preparation'

    def __init__(self, map_range_suggestions: Optional[RangeSuggestions] = None):
        super().__init__()
        self._suggestions: RangeSuggestions = map_range_suggestions or {}

    def modifies(self) -> ppl.Modifies:
        return (ppl.Modifies.Descriptors
                | ppl.Modifies.Edges
                | ppl.Modifies.Nodes
                | ppl.Modifies.States)

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.Descriptors | ppl.Modifies.Nodes) != ppl.Modifies.Nothing

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[int]:
        """Run the fixpoint lift; return the total number of promotions
        (or ``None`` if nothing was lifted, per DaCe Pass convention)."""
        total = 0
        while True:
            promoted = _lift_one_level(sdfg, self._suggestions)
            if promoted == 0:
                break
            total += promoted
        _verify_postconditions(sdfg)
        return total if total > 0 else None


def lift_transients(sdfg: SDFG,
                    map_range_suggestions: Optional[RangeSuggestions] = None) -> int:
    """Functional entry point. Runs :class:`LiftTransients` once and
    returns the number of promotions (0 if none)."""
    pass_obj = LiftTransients(map_range_suggestions=map_range_suggestions)
    return pass_obj.apply_pass(sdfg, {}) or 0


def _verify_postconditions(sdfg: SDFG):
    """Post-conditions enforced after the fixpoint:

    - every top-level transient has ``AllocationLifetime.SDFG``;
    - no multi-element transient remains in any nested SDFG;
    - no lifted shape depends on a symbol assigned by an interstate-edge
      assignment anywhere in the tree (such a symbol isn't known at SDFG
      entry, so the allocator can't size the array).
    """
    errors: List[str] = []
    for name, desc in sdfg.arrays.items():
        if desc.transient and desc.lifetime != dtypes.AllocationLifetime.SDFG:
            errors.append(
                f"top-level transient {name!r} has lifetime={desc.lifetime}, "
                f"expected AllocationLifetime.SDFG")

    for n, _ in sdfg.all_nodes_recursive():
        if not isinstance(n, nodes.NestedSDFG):
            continue
        for name, desc in n.sdfg.arrays.items():
            if not desc.transient:
                continue
            if tuple(desc.shape) == (1, ):
                continue
            errors.append(
                f"nested SDFG {n.label!r} still declares transient {name!r} "
                f"(shape={tuple(desc.shape)})")

    assigned = _assigned_symbols(sdfg)
    for name, desc in sdfg.arrays.items():
        if not desc.transient:
            continue
        for dim in desc.shape:
            syms = symbolic.symlist(dim)
            bad = [s for s in syms if str(s) in assigned]
            if bad:
                errors.append(
                    f"top-level transient {name!r} shape {tuple(desc.shape)} "
                    f"depends on internally-assigned symbol(s) {bad}")
                break

    if errors:
        raise ValueError(
            "LiftTransients post-conditions failed:\n  - "
            + "\n  - ".join(errors))


def _assigned_symbols(sdfg: SDFG) -> Set[str]:
    names: Set[str] = set()
    for g in _all_sdfgs(sdfg):
        for e in g.all_interstate_edges():
            names.update(e.data.assignments.keys())
    return names


def _all_sdfgs(sdfg: SDFG):
    yield sdfg
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.NestedSDFG):
            yield n.sdfg


def _lift_one_level(root: SDFG, suggestions: RangeSuggestions) -> int:
    victims: List[Tuple[nodes.NestedSDFG, dace.SDFGState, str]] = []
    for n, parent_state in root.all_nodes_recursive():
        if not isinstance(n, nodes.NestedSDFG):
            continue
        if not isinstance(parent_state, dace.SDFGState):
            continue
        for name, desc in n.sdfg.arrays.items():
            if _should_lift(desc):
                victims.append((n, parent_state, name))

    for nsdfg_node, parent_state, name in victims:
        _lift_one(nsdfg_node, parent_state, name, suggestions)
    return len(victims)


def _should_lift(desc: data.Data) -> bool:
    if not desc.transient:
        return False
    if not isinstance(desc, data.Array):
        return False
    return tuple(desc.shape) != (1, )


def _is_fortran_layout(desc: data.Array) -> bool:
    """Inspect descriptor strides; default to Fortran when ambiguous."""
    if desc.is_packed_c_strides() and not desc.is_packed_fortran_strides():
        return False
    return True


def _lift_one(nsdfg_node: nodes.NestedSDFG, parent_state: dace.SDFGState, name: str,
              suggestions: RangeSuggestions):
    inner = nsdfg_node.sdfg
    parent_sdfg: SDFG = parent_state.sdfg
    desc = inner.arrays[name]
    is_fortran = _is_fortran_layout(desc)

    scope_entry = parent_state.entry_node(nsdfg_node)
    scope_exit = parent_state.exit_node(scope_entry) if scope_entry is not None else None
    map_params = list(scope_entry.map.params) if scope_entry is not None else []
    map_range: Optional[subsets.Range] = (
        scope_entry.map.range if scope_entry is not None else None)

    # axes[i] = (param_name, begin_expr) for each added dim.
    new_shape, axes = _extend_shape_with_map_dims(
        desc.shape, map_range, map_params, suggestions, is_fortran)

    new_name = _unique(parent_sdfg.arrays, name)
    new_desc = copy.deepcopy(desc)
    # ``set_shape`` invalidates the cached packed-strides so later
    # ``is_packed_fortran_strides`` / ``is_packed_c_strides`` calls see
    # the post-lift layout. Setting ``.shape`` directly doesn't.
    new_desc.set_shape(
        new_shape,
        strides=_packed_strides(new_shape, is_fortran),
        total_size=_product(new_shape),
        offset=[0] * len(new_shape),
    )
    new_desc.transient = True
    new_desc.lifetime = dtypes.AllocationLifetime.SDFG
    parent_sdfg.add_datadesc(new_name, new_desc)

    inner_desc = copy.deepcopy(desc)
    inner_desc.transient = False
    inner.remove_data(name, validate=False)
    inner.add_datadesc(name, inner_desc)

    reads, writes = _inner_rw(inner, name)
    if not reads and not writes:
        return

    _wire(parent_state, nsdfg_node, name, new_name, new_desc,
          scope_entry, scope_exit, axes, is_fortran, reads, writes)


def _extend_shape_with_map_dims(orig_shape, map_range: Optional[subsets.Range],
                                map_params: List[str], suggestions: RangeSuggestions,
                                is_fortran: bool):
    """Return ``(new_shape, axes)`` where ``axes`` is one
    ``(param_name, begin_expr)`` entry per added dim. For Fortran
    (column-major) the map dims are APPENDED (so each slab stays
    contiguous); for C they are PREPENDED."""
    if map_range is None:
        return list(orig_shape), []
    extra = []
    axes: List[Tuple[str, symbolic.SymbolicType]] = []
    for p, (rb, re, _rs) in zip(map_params, map_range.ndrange()):
        size = None
        key = _range_key(rb, re)
        if key is not None and key in suggestions:
            size = symbolic.pystr_to_symbolic(suggestions[key])
        if size is None:
            size = symbolic.simplify(re - rb + 1)
        extra.append(size)
        axes.append((p, rb))
    if is_fortran:
        return list(orig_shape) + extra, axes
    return extra + list(orig_shape), axes


def _range_key(begin, end) -> Optional[Tuple[str, str]]:
    b_syms = list(symbolic.symlist(begin).keys())
    e_syms = list(symbolic.symlist(end).keys())
    if len(b_syms) != 1 or len(e_syms) != 1:
        return None
    return (str(b_syms[0]), str(e_syms[0]))


def _packed_strides(shape, is_fortran: bool):
    strides = [1] * len(shape)
    if is_fortran:
        for i in range(1, len(shape)):
            strides[i] = strides[i - 1] * shape[i - 1]
    else:
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def _product(shape):
    acc = 1
    for d in shape:
        acc = acc * d
    return acc


def _unique(existing, base: str) -> str:
    if base not in existing:
        return base
    i = 0
    while True:
        cand = f"{base}_{i}"
        if cand not in existing:
            return cand
        i += 1


def _inner_rw(inner: SDFG, name: str) -> Tuple[bool, bool]:
    reads = writes = False
    for state in inner.all_states():
        for n in state.nodes():
            if not isinstance(n, nodes.AccessNode) or n.data != name:
                continue
            if state.in_edges(n):
                writes = True
            if state.out_edges(n):
                reads = True
    return reads, writes


def _wire(parent_state: dace.SDFGState, nsdfg_node: nodes.NestedSDFG,
         inner_name: str, outer_name: str, outer_desc: data.Array,
         scope_entry: Optional[nodes.MapEntry], scope_exit: Optional[nodes.MapExit],
         axes: List[Tuple[str, symbolic.SymbolicType]], is_fortran: bool,
         reads: bool, writes: bool):
    full = mm.Memlet.from_array(outer_name, outer_desc)

    if reads:
        nsdfg_node.add_in_connector(inner_name, force=True)
        access = parent_state.add_read(outer_name)
        if scope_entry is None:
            parent_state.add_edge(access, None, nsdfg_node, inner_name,
                                  copy.deepcopy(full))
        else:
            ic = "IN_" + outer_name
            oc = "OUT_" + outer_name
            scope_entry.add_in_connector(ic)
            scope_entry.add_out_connector(oc)
            parent_state.add_edge(access, None, scope_entry, ic, copy.deepcopy(full))
            parent_state.add_edge(scope_entry, oc, nsdfg_node, inner_name,
                                  _inner_memlet(outer_name, outer_desc, axes, is_fortran))

    if writes:
        nsdfg_node.add_out_connector(inner_name, force=True)
        write = parent_state.add_write(outer_name)
        if scope_exit is None:
            parent_state.add_edge(nsdfg_node, inner_name, write, None,
                                  copy.deepcopy(full))
        else:
            ic = "IN_" + outer_name
            oc = "OUT_" + outer_name
            scope_exit.add_in_connector(ic)
            scope_exit.add_out_connector(oc)
            parent_state.add_edge(nsdfg_node, inner_name, scope_exit, ic,
                                  _inner_memlet(outer_name, outer_desc, axes, is_fortran))
            parent_state.add_edge(scope_exit, oc, write, None, copy.deepcopy(full))


def _inner_memlet(name: str, desc: data.Array,
                  axes: List[Tuple[str, symbolic.SymbolicType]],
                  is_fortran: bool) -> mm.Memlet:
    """Per-iteration slab memlet addressing the OUTER array. For each
    ``(param, begin)`` in ``axes`` the memlet has a single-point range
    ``(param - begin, param - begin, 1)``; remaining dims are the full
    original range. Fortran appends the per-iter points to the tail of
    the subset; C prepends them to the head."""
    n_extra = len(axes)
    outer_shape = list(desc.shape)
    if is_fortran:
        orig = outer_shape[:-n_extra] if n_extra else outer_shape
    else:
        orig = outer_shape[n_extra:]
    orig_ranges = [(0, symbolic.simplify(d - 1), 1) for d in orig]

    point_ranges = []
    for p, begin in axes:
        sym = symbolic.pystr_to_symbolic(p)
        idx = symbolic.simplify(sym - begin)
        point_ranges.append((idx, idx, 1))

    ranges = orig_ranges + point_ranges if is_fortran else point_ranges + orig_ranges
    return mm.Memlet(data=name, subset=subsets.Range(ranges))
