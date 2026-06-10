# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Normalize the iteration ranges of two "close" maps so subsequent
:class:`~dace.transformation.dataflow.map_fusion.MapFusion` can fuse them.

Two maps are *close* iff they have the same dimensionality and, per dimension, their
start and end bounds differ by a **constant** (a sympy ``Number``). The optional
``max_constant_diff`` knob lets the user further require ``|diff| <= max_constant_diff``
per dim; the default of ``0`` means "no upper bound -- accept any constant difference".

Rewrite per matched pair: take the per-dim union ``(min(start), max(end))`` as the
common range, extend each map's range to it, and wrap each map's body in an if-guard
that re-imposes the original per-map bounds. After running, both maps have identical
ranges; the existing vertical :class:`MapFusion` (in the pass form here, only the
vertical producer -> AccessNode -> consumer chain is targeted) can then fuse them
normally.

The transformation form :class:`UnifyCloseIterationDomains` matches the vertical
chain pattern ``MapExit -> AccessNode -> MapEntry``. The pass form
:class:`UnifyCloseIterationDomainsPass` walks every such pair in every state and
applies the transformation when the maps are close.
"""
import copy
from typing import Any, Dict, List, Optional, Tuple

import sympy

import dace
from dace import SDFG, SDFGState, dtypes, properties
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import SubgraphView
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.helpers import nest_state_subgraph


def _ranges_are_close(range_a, range_b, max_constant_diff: int) -> bool:
    """True iff ``range_a`` and ``range_b`` differ only by per-dim **constant** start/end
    bounds (and identical strides).

    :param range_a: First map's ``dace.subsets.Range``.
    :param range_b: Second map's ``dace.subsets.Range``.
    :param max_constant_diff: ``0`` accepts any constant difference. A positive value
                              further requires every per-dim ``|diff| <= max_constant_diff``.
    :return: ``True`` if the ranges are close, ``False`` otherwise.
    """
    if len(range_a) != len(range_b):
        return False
    for (ba, ea, sa), (bb, eb, sb) in zip(range_a, range_b):
        if sympy.simplify(sa - sb) != 0:
            return False
        start_diff = sympy.simplify(ba - bb)
        end_diff = sympy.simplify(ea - eb)
        if not start_diff.is_Number or not end_diff.is_Number:
            return False
        if max_constant_diff > 0:
            if abs(int(start_diff)) > max_constant_diff or abs(int(end_diff)) > max_constant_diff:
                return False
    return True


def _union_range(range_a, range_b) -> 'dace.subsets.Range':
    """Per-dim ``(min(start), max(end), step)`` union of two close ranges."""
    out = []
    for (ba, ea, sa), (bb, eb, sb) in zip(range_a, range_b):
        out.append((sympy.simplify(sympy.Min(ba, bb)), sympy.simplify(sympy.Max(ea, eb)), sa))
    return dace.subsets.Range(out)


def _bounds_check_expr(params, orig_range, new_range) -> Optional[str]:
    """Build the minimal Python ``and``-joined bounds-check for a map's params.

    Per dim, emit ``p >= orig_start`` **only when** ``new_start < orig_start`` and
    emit ``p <= orig_end`` **only when** ``new_end > orig_end`` -- those are the
    cases where the extension introduced iterations that the original range did
    not cover. Dimensions whose ``orig`` bound equals ``new`` are dropped entirely.

    :param params: Iteration parameter names.
    :param orig_range: The map's original (smaller) range that the body must respect.
    :param new_range: The post-extension range; what the map will actually iterate.
    :return: The bounds-check expression, or ``None`` when every dimension's original
             range is equal to the new range (no guard needed).
    """
    parts = []
    for p, (ob, oe, _os), (nb, ne, _ns) in zip(params, orig_range, new_range):
        per_dim = []
        if sympy.simplify(nb - ob) != 0:
            per_dim.append(f"({p} >= {ob})")
        if sympy.simplify(ne - oe) != 0:
            per_dim.append(f"({p} <= {oe})")
        if per_dim:
            parts.append(" and ".join(per_dim))
    return " and ".join(parts) if parts else None


def _wrap_map_body_in_bounds_check(state: SDFGState, map_entry: nodes.MapEntry, orig_range: 'dace.subsets.Range',
                                   new_range: 'dace.subsets.Range') -> None:
    """Wrap the body of ``map_entry`` in a ``NestedSDFG`` whose top-level block is a
    :class:`ConditionalBlock` guarded by ``orig_range`` on ``map_entry.map.params``.

    Pattern: ``MapEntry -> body_nodes -> MapExit`` becomes
    ``MapEntry -> NestedSDFG{ConditionalBlock(body_nodes)} -> MapExit``.

    Built on :func:`dace.transformation.helpers.nest_state_subgraph` -- that helper does
    the boundary rewiring (memlets through ``MapEntry``/``MapExit`` connectors) cleanly;
    we then promote the resulting NSDFG's start state into a ConditionalBlock branch.

    :param state: State containing ``map_entry``.
    :param map_entry: MapEntry whose body should be guarded.
    :param orig_range: The pre-extension range; the if-guard re-imposes it.
    """
    bounds_expr = _bounds_check_expr(map_entry.map.params, orig_range, new_range)
    if bounds_expr is None:
        # Every dimension's original range matches the new range -- no guard needed,
        # i.e. extending this map's range introduced no extra iterations.
        return

    map_exit = state.exit_node(map_entry)
    body_nodes = [n for n in state.all_nodes_between(map_entry, map_exit) if n is not map_entry and n is not map_exit]
    if not body_nodes:
        return

    # Standard helper: encapsulate the body into a NestedSDFG; properly wires the
    # boundary memlets through MapEntry/MapExit.
    # ``full_data=False`` keeps the per-iteration boundary subset (e.g. ``arr[i, j]``)
    # instead of promoting it to the full-array shape. Promotion would break vertical
    # ``MapFusion``'s producer-covers-consumer check downstream: the producer writes one
    # element per iteration but the consumer would appear to read the whole array.
    nsdfg_node = nest_state_subgraph(state.sdfg,
                                     state,
                                     SubgraphView(state, body_nodes),
                                     name=f"unify_close_{map_entry.label}",
                                     full_data=False)
    inner_sdfg = nsdfg_node.sdfg

    # nest_state_subgraph leaves the inner SDFG with one state holding the original body.
    # Wrap that state in a ConditionalBlock guarded by the if-bound-check.
    body_state = inner_sdfg.start_block
    assert isinstance(body_state, SDFGState), type(body_state)

    # Detach the body state from inner_sdfg and reparent it as the (single) branch of a
    # new ConditionalBlock at inner_sdfg's top.
    inner_sdfg.remove_node(body_state)

    if_block = ConditionalBlock(label=f"bound_check_{map_entry.label}", sdfg=inner_sdfg, parent=inner_sdfg)
    branch_region = ControlFlowRegion(label=f"body_{map_entry.label}", sdfg=inner_sdfg, parent=if_block)
    branch_region.add_node(body_state, is_start_block=True)
    body_state.parent_graph = branch_region

    if_block.add_branch(condition=CodeBlock(bounds_expr), branch=branch_region)
    inner_sdfg.add_node(if_block, is_start_block=True)

    # Thread the map's own iteration params -- the if-guard references them.
    for sym in map_entry.map.params:
        if sym not in inner_sdfg.symbols:
            inner_sdfg.add_symbol(sym, dtypes.int32)
        if sym not in nsdfg_node.symbol_mapping:
            nsdfg_node.symbol_mapping[sym] = sym

    sdutil.set_nested_sdfg_parent_references(state.sdfg)
    state.sdfg.reset_cfg_list()


@properties.make_properties
class UnifyCloseIterationDomains(transformation.SingleStateTransformation):
    """Single-state transformation: take two maps in a vertical
    ``MapExit -> AccessNode -> MapEntry`` chain and, if their ranges are close, extend
    both to their per-dim union and guard each body with the original range.

    After applying, both maps have identical ranges -- the existing vertical
    :class:`~dace.transformation.dataflow.map_fusion.MapFusion` can fuse them.

    :param max_constant_diff: ``0`` (default) accepts any constant per-dim difference;
        a positive value caps it to ``|diff| <= max_constant_diff``.
    """

    first_map_exit = transformation.PatternNode(nodes.MapExit)
    array = transformation.PatternNode(nodes.AccessNode)
    second_map_entry = transformation.PatternNode(nodes.MapEntry)

    max_constant_diff = properties.Property(
        dtype=int,
        default=0,
        desc="0 = accept any constant per-dim range difference. >0 = require |diff| <= max_constant_diff per dim.")

    @classmethod
    def expressions(cls) -> Any:
        return [sdutil.node_path_graph(cls.first_map_exit, cls.array, cls.second_map_entry)]

    def can_be_applied(self, state: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        first_entry = state.entry_node(self.first_map_exit)
        if first_entry is None:
            return False
        if not _ranges_are_close(first_entry.map.range, self.second_map_entry.map.range, int(self.max_constant_diff)):
            return False
        # Refuse if the ranges are already identical -- nothing to unify.
        if all(
                sympy.simplify(a - b) == 0
                for (a,
                     b) in zip(_flatten_range(first_entry.map.range), _flatten_range(self.second_map_entry.map.range))):
            return False
        return True

    def apply(self, state: SDFGState, sdfg: SDFG) -> None:
        first_entry = state.entry_node(self.first_map_exit)
        second_entry = self.second_map_entry
        orig_first = copy.deepcopy(first_entry.map.range)
        orig_second = copy.deepcopy(second_entry.map.range)
        unified = _union_range(first_entry.map.range, second_entry.map.range)

        # Extend each map's range and wrap the body where the original range was strictly
        # smaller -- when the union equals the original on every dim the if-guard would
        # always evaluate true, so skip the wrap.
        def _same(r_a, r_b) -> bool:
            return all(sympy.simplify(a - b) == 0 for a, b in zip(_flatten_range(r_a), _flatten_range(r_b)))

        if not _same(unified, orig_first):
            first_entry.map.range = copy.deepcopy(unified)
            _wrap_map_body_in_bounds_check(state, first_entry, orig_first, unified)
        if not _same(unified, orig_second):
            second_entry.map.range = copy.deepcopy(unified)
            _wrap_map_body_in_bounds_check(state, second_entry, orig_second, unified)


def _flatten_range(rng) -> List:
    """Flatten a ``Range`` to a list of (start, end, step) symbols in order."""
    out = []
    for (b, e, s) in rng:
        out += [b, e, s]
    return out


@properties.make_properties
@transformation.explicit_cf_compatible
class UnifyCloseIterationDomainsPass(ppl.Pass):
    """Walk every vertical ``MapExit -> AccessNode -> MapEntry`` chain in every state
    and apply :class:`UnifyCloseIterationDomains` where the ranges are close.

    Pass form only -- handles the producer -> consumer (vertical) chain. Independent
    sibling maps are not addressed.
    """

    max_constant_diff = properties.Property(dtype=int,
                                            default=0,
                                            desc="Per-map-pair tolerance forwarded to UnifyCloseIterationDomains. "
                                            "0 = accept any constant per-dim range difference.")

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[int]:
        applied = 0
        for nested in list(sdfg.all_sdfgs_recursive()):
            for state in list(nested.states()):
                for pair in self._collect_vertical_pairs(state):
                    if self._try_apply_one(nested, state, *pair):
                        applied += 1
        return applied if applied > 0 else None

    @staticmethod
    def _collect_vertical_pairs(state: SDFGState) -> List[Tuple[nodes.MapExit, nodes.AccessNode, nodes.MapEntry]]:
        pairs: List[Tuple[nodes.MapExit, nodes.AccessNode, nodes.MapEntry]] = []
        for me_exit in [n for n in state.nodes() if isinstance(n, nodes.MapExit)]:
            for oe in state.out_edges(me_exit):
                if not isinstance(oe.dst, nodes.AccessNode):
                    continue
                an = oe.dst
                for ne in state.out_edges(an):
                    if not isinstance(ne.dst, nodes.MapEntry):
                        continue
                    pairs.append((me_exit, an, ne.dst))
        return pairs

    def _try_apply_one(self, sdfg: SDFG, state: SDFGState, first_exit: nodes.MapExit, array: nodes.AccessNode,
                       second_entry: nodes.MapEntry) -> bool:
        xform = UnifyCloseIterationDomains()
        xform.max_constant_diff = int(self.max_constant_diff)
        xform.setup_match(
            sdfg, sdfg.cfg_id, state.block_id, {
                UnifyCloseIterationDomains.first_map_exit: state.node_id(first_exit),
                UnifyCloseIterationDomains.array: state.node_id(array),
                UnifyCloseIterationDomains.second_map_entry: state.node_id(second_entry),
            }, 0)
        if not xform.can_be_applied(state, 0, sdfg):
            return False
        xform.apply(state, sdfg)
        return True
