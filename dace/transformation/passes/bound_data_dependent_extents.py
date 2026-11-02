# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Size a transient whose extent names a symbol read from data by the largest extent that symbol allows.

``cum_x[order - dil_index - diu_index + 2]``, with ``dil_index = dil[p]`` assigned per particle, has a new extent
every iteration: it is allocated per iteration, and no kernel can own it. When every such symbol is nonnegative (every
store into what it is read from is, :func:`~dace.transformation.passes.analysis.array_facts.nonnegative_containers`)
and enters the extent linearly with a nonpositive coefficient, the extent is at most its value with those symbols at
zero (``order + 2``). The transient takes that extent; every access keeps its subset.

A transient no such extent fits may still have a capacity. An out-of-bounds access is undefined, so a subset of a
literal-shaped array bounds the symbols it names: ``pol[0:nlp, rel[i] + 2]`` with ``rel[i] = i - span`` and ``pol`` of
shape ``(5, 5)`` gives ``nlp <= 5`` and ``span <= 2``. Shape and strides stay exact, the packed views need them, and
only ``total_size`` becomes a literal, the most any iteration needs, so a thread can own the array. A site counts when
its trip counts are literals or extents of the arrays being sized, which are taken to be nonzero: an extent that is zero
runs no access, so it proves nothing about another one.
"""

import re
from typing import Any

import sympy

from dace import SDFG, dtypes, symbolic
from dace import data as dt
from dace.sdfg import nodes
from dace.sdfg.sdfg import interstate_bound_names
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.sdfg.utils import THREAD_CAPACITY
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.analysis import array_facts
from dace.utils import prod

Interval = tuple[Any, Any]


def parsed(text: Any) -> Any:
    return symbolic.pystr_to_symbolic(re.sub(r"\bint64_t\b", "", str(text)))


def stored_value(state: SDFGState, edge: Any) -> tuple[Any, nodes.Tasklet] | None:
    """The value a chain of one-assignment tasklets leaves on ``edge``, and the first of them."""
    source, _ = array_facts.producer(state, edge)
    if isinstance(source, nodes.AccessNode) and state.in_degree(source) == 1:
        return stored_value(state, state.in_edges(source)[0])
    if not isinstance(source, nodes.Tasklet) or array_facts.assignment_value(source) is None:
        return None
    value = parsed(array_facts.assignment_value(source))
    for inner in state.in_edges(source):
        if inner.data.is_empty():
            continue
        if not isinstance(inner.src, nodes.AccessNode) or state.in_degree(inner.src) != 1:
            return None
        operand = stored_value(state, state.in_edges(inner.src)[0])
        if operand is None:
            return None
        value = value.subs(symbolic.symbol(inner.dst_conn), operand[0])
    return value, source


def element_range(sdfg: SDFG, name: str) -> Interval | None:
    """The values of the array ``name``, stored by one map as an affine function of its parameter."""
    edges = array_facts.write_edges(sdfg, name)
    stored = stored_value(*edges[0]) if len(edges) == 1 else None
    if stored is None:
        return None
    state, value = edges[0][0], stored[0]
    entry = state.entry_node(stored[1])
    if not isinstance(entry, nodes.MapEntry) or entry.map.get_param_num() != 1:
        return None
    param, (low, high, _) = entry.map.params[0], entry.map.range[0]
    coefficients = symbolic.affine_coefficients(value, param)
    if coefficients is None or not coefficients[0].is_number:
        return None
    ends = [coefficients[0] * end + coefficients[1] for end in (low, high)]
    return (ends[0], ends[1]) if coefficients[0] >= 0 else (ends[1], ends[0])


def extreme(expr: Any, ranges: dict[str, Interval], upper: bool) -> Any:
    """The largest (or smallest) value of ``expr``, linear in the names ``ranges`` holds."""
    expr = parsed(expr)
    for name in sorted({str(s) for s in expr.free_symbols} & ranges.keys()):
        coefficients = symbolic.affine_coefficients(expr, name)
        if coefficients is not None and coefficients[0].is_number:
            expr = coefficients[0] * ranges[name][int(upper == (coefficients[0] >= 0))] + coefficients[1]
    return expr


def within(low: Any, high: Any, first: Any, last: Any, ranges: dict[str, Interval]) -> list[symbolic.Relation]:
    """``first <= [low, high] <= last``, with the names of ``ranges`` at their extremes."""
    return [
        symbolic.Relation(symbolic.RelationKind.LE, parsed(first), extreme(low, ranges, False)),
        symbolic.Relation(symbolic.RelationKind.LE, extreme(high, ranges, True), parsed(last)),
    ]


def state_relations(state: SDFGState, ranges: dict[str, Interval], nonempty: set[str]) -> list[symbolic.Relation]:
    """What the in-bounds accesses of ``state`` to literal-shaped arrays imply; only exact leaf memlets count."""
    sdfg, scopes, relations = state.sdfg, state.scope_dict(), []
    known: dict[nodes.EntryNode | None, bool] = {None: True}

    def runs(scope: nodes.EntryNode | None) -> bool:
        if scope not in known:
            sizes = [parsed(size) for size in scope.map.range.size()]
            known[scope] = all(size.is_number or str(size) in nonempty for size in sizes) and runs(scopes[scope])
        return known[scope]

    for node in state.nodes():
        # ``__arr[__inp0]``, a gather: the row its one memlet names bounds the index array read on the other.
        if isinstance(node, nodes.Tasklet) and array_facts.assignment_value(node) == "__arr[__inp0]" and runs(scopes[node]):
            memlets = {edge.dst_conn: edge.data for edge in state.in_edges(node)}
            values = element_range(sdfg, memlets["__inp0"].data)
            if values is not None and memlets["__arr"].subset[-1][0].is_number:
                first, last, _ = memlets["__arr"].subset[-1]
                relations += within(*values, first, last, ranges)
            continue
        if not isinstance(node, nodes.AccessNode) or not all(parsed(e).is_number for e in node.desc(sdfg).shape):
            continue
        for edge in state.all_edges(node):
            if edge.data.is_empty() or edge.data.dynamic or not runs(scopes[node]):
                continue
            other = edge.dst if edge.src is node else edge.src
            if isinstance(other, (nodes.NestedSDFG, nodes.EntryNode, nodes.ExitNode)):
                continue
            memlet = edge.data
            subset = memlet.get_src_subset(edge, state) if edge.src is node else memlet.get_dst_subset(edge, state)
            for (low, high, _), extent in zip(subset, node.desc(sdfg).shape):
                relations += within(low, high, 0, extent - 1, ranges)
    return relations


def guaranteed_relations(
    region: ControlFlowRegion, ranges: dict[str, Interval], nonempty: set[str]
) -> list[symbolic.Relation]:
    """What holds on every path through ``region``: a conditional gives what all its arms, an ``else`` among them, do."""
    if not array_facts.straight(region):
        return []
    relations = []
    for block in region.nodes():
        if isinstance(block, SDFGState):
            relations += state_relations(block, ranges, nonempty)
        elif isinstance(block, LoopRegion):
            bounds = array_facts.loop_range(block)
            size = None if bounds is None else parsed(bounds[1]) - parsed(bounds[0]) + 1
            if size is not None and (size.is_number or str(size) in nonempty):
                relations += guaranteed_relations(block, ranges, nonempty)
        elif isinstance(block, ConditionalBlock) and block.branches[-1][0] is None:
            arms = [guaranteed_relations(body, ranges, nonempty) for _, body in block.branches]
            shared = {str(relation) for relation in arms[0]}
            for arm in arms[1:]:
                shared &= {str(relation) for relation in arm}
            relations += [relation for relation in arms[0] if str(relation) in shared]
    return relations


def one_value_per_path(sdfg: SDFG) -> tuple[dict[str, Interval], dict[str, Any], set[str]]:
    """Over ``sdfg``'s interstate assignments: the ranges of symbols read from an index array, the values of those
    assigned one expression once, and the names assigned at most once on any path."""
    sites: dict[str, list[tuple[ControlFlowRegion, str]]] = {}
    for region in sdfg.all_control_flow_regions():
        for edge in region.edges():
            for name, value in edge.data.assignments.items():
                sites.setdefault(name, []).append((region, value))
    ranges, values, single = {}, {}, set()
    for name, assigned in sites.items():
        owners = {region.parent_graph for region, _ in assigned}
        if len(assigned) == 1 or (len(owners) == 1 and isinstance(next(iter(owners)), ConditionalBlock)):
            single.add(name)
        if len(assigned) != 1:
            continue
        read = re.fullmatch(r"\(?(\w+)\[.*\]\)?", assigned[0][1].strip())
        if read is not None and read[1] in sdfg.arrays:
            found = element_range(sdfg, read[1])
            if found is not None:
                ranges[name] = found
        elif not {str(s) for s in parsed(assigned[0][1]).free_symbols} & sdfg.arrays.keys():
            values[name] = parsed(assigned[0][1])
    return ranges, values, single


def capacity(extent: Any, facts: symbolic.Facts, known: dict[Any, int | None]) -> int | None:
    """The smallest literal the ``facts`` show ``extent`` stays under, up to :data:`THREAD_CAPACITY`; a product, of
    nonnegative extents, by its factors. ``known`` holds the answers so far."""
    extent = parsed(extent)
    if extent.is_number:
        return int(extent)
    if extent in known:
        return known[extent]
    factors = [factor.as_base_exp() for factor in sympy.Mul.make_args(extent)]
    if len(factors) > 1 or factors[0][1] != 1:
        sizes = [capacity(base, facts, known) for base, _ in factors]
        result = None if None in sizes else prod(size**exponent for size, (_, exponent) in zip(sizes, factors))
    else:
        def fits(limit: int) -> symbolic.Truth:
            return symbolic.ask(symbolic.Relation(symbolic.RelationKind.LE, extent, sympy.Integer(limit)), facts)

        result = None
        if fits(THREAD_CAPACITY) is symbolic.Truth.TRUE:
            low, high = 0, THREAD_CAPACITY
            while high - low > 1:
                middle = (low + high) // 2
                low, high = (low, middle) if fits(middle) is symbolic.Truth.TRUE else (middle, high)
            result = high
    known[extent] = result
    return result


def set_capacities(sdfg: SDFG, assigned: set[str]) -> int:
    """Give every per-iteration transient of ``sdfg`` its capacity, or none of them: one left over stays on the host."""
    sized = {
        name: desc
        for name, desc in sdfg.arrays.items()
        if isinstance(desc, dt.Array) and desc.transient and {str(s) for s in desc.free_symbols} & assigned
    }
    if not sized:
        return 0
    ranges, values, single = one_value_per_path(sdfg)
    nonempty = {str(parsed(extent)) for desc in sized.values() for extent in desc.shape}
    nonempty |= {str(value) for name, value in values.items() if name in nonempty}
    relations = guaranteed_relations(sdfg, ranges, nonempty)
    for name, value in values.items():
        target = symbolic.symbol(name)
        relations += [
            symbolic.Relation(symbolic.RelationKind.LE, target, value),
            symbolic.Relation(symbolic.RelationKind.LE, value, target),
        ]
    relations = [
        relation
        for relation in dict.fromkeys(relations)
        if symbolic.relation_names(relation) and symbolic.relation_names(relation) & assigned <= single
    ]
    integers = frozenset(n for r in relations for n in symbolic.relation_names(r))
    facts = symbolic.Facts(relations, integers)
    known: dict[Any, int | None] = {}
    sizes = {name: [capacity(extent, facts, known) for extent in desc.shape] for name, desc in sized.items()}
    owned = {n: e for n, e in sizes.items() if not isinstance(sized[n], dt.View)}
    if any(None in extents or prod(extents) > THREAD_CAPACITY for extents in owned.values()):
        return 0
    for name, extents in sizes.items():
        if None in extents:
            continue
        sized[name].total_size = prod(extents)
        if not isinstance(sized[name], dt.View):
            sized[name].lifetime = dtypes.AllocationLifetime.SDFG
    return len(sizes)


def extent_upper_bound(extent: Any, assigned: set[str], nonnegative: set[str]) -> Any | None:
    """``extent`` with every name of ``assigned`` it holds set to zero, when each is in ``nonnegative`` and enters
    linearly with a nonpositive constant coefficient (so zero maximizes it); ``None`` otherwise."""
    expr = symbolic.pystr_to_symbolic(extent)
    for name in sorted({str(s) for s in expr.free_symbols} & assigned):
        coefficients = symbolic.affine_coefficients(expr, name)
        if name not in nonnegative or coefficients is None:
            return None
        lead, rest = coefficients
        if not lead.is_number or lead > 0:
            return None
        expr = rest
    return expr


def keeps_strides(desc: dt.Array, shape: list) -> bool:
    """Whether ``desc`` is packed row-major and ``shape`` changes at most its leading extent, so no stride moves."""
    packed = dt.Array(desc.dtype, desc.shape).strides
    return all(symbolic.equal(a, b) is True for a, b in zip(desc.strides, packed)) and all(
        symbolic.equal(a, b) is True for a, b in zip(desc.shape[1:], shape[1:])
    )


def resize_plan(sdfg: SDFG, name: str, shape: list) -> list[tuple[SDFG, str, list]] | None:
    """Every descriptor that must take ``shape`` with ``sdfg``'s ``name``: the nested SDFG connectors it binds,
    restated in their own symbols; ``None`` if a view or a connector cannot follow without moving a stride."""
    desc = sdfg.arrays[name]
    plan = [(sdfg, name, shape)]
    strides_kept = keeps_strides(desc, shape)
    for state in sdfg.states():
        for node in state.data_nodes():
            if node.data != name:
                continue
            # Through the scopes in between: the far end of each memlet path is what reads the descriptor
            ends = [
                (leaf.dst, leaf.dst_conn) if edge.src is node else (leaf.src, leaf.src_conn)
                for edge in state.all_edges(node)
                for leaf in state.memlet_tree(edge).leaves()
            ]
            for other, conn in ends:
                if isinstance(other, nodes.AccessNode) and isinstance(other.desc(sdfg), dt.View) and not strides_kept:
                    return None
                if not isinstance(other, nodes.NestedSDFG):
                    continue
                inner = other.sdfg.arrays[conn]
                if not strides_kept or not keeps_strides(inner, shape):
                    return None
                inverse = {str(value): key for key, value in other.symbol_mapping.items()}
                names = {str(s) for extent in shape for s in symbolic.pystr_to_symbolic(extent).free_symbols}
                if not names <= inverse.keys():
                    return None
                rename = {symbolic.symbol(n): symbolic.symbol(inverse[n]) for n in names}
                inner_shape = [symbolic.pystr_to_symbolic(extent).subs(rename) for extent in shape]
                if any(entry[0] is other.sdfg and entry[1] == conn for entry in plan):
                    continue
                deeper = resize_plan(other.sdfg, conn, inner_shape)
                if deeper is None:
                    return None
                plan += deeper
    return plan


@transformation.explicit_cf_compatible
class BoundDataDependentExtents(ppl.Pass):
    """Give each transient sized by a data-read symbol the extent that bounds it (module docstring)."""

    CATEGORY: str = "Optimization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> set[type[ppl.Pass] | ppl.Pass]:
        return set()

    def nonnegative(self, sdfg: SDFG, known: dict[SDFG, set[str]]) -> set[str]:
        """The containers and symbols of ``sdfg`` provably nonnegative, given those of the SDFGs around it."""
        if sdfg in known:
            return known[sdfg]
        assumed: set[str] = set()
        node = sdfg.parent_nsdfg_node
        if node is not None:
            outer = self.nonnegative(sdfg.parent_sdfg, known)
            assumed = {e.dst_conn for e in sdfg.parent.in_edges(node) if e.dst_conn and e.data.data in outer}
            assumed |= {k for k, v in node.symbol_mapping.items() if array_facts.parses_nonnegative(str(v), outer)}
        known[sdfg] = array_facts.nonnegative_containers(sdfg, frozenset(assumed))
        return known[sdfg]

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """Bound every transient extent the module docstring describes, in ``sdfg`` and its nested SDFGs.

        :returns: How many transients were resized, or ``None`` if none.
        """
        known: dict[SDFG, set[str]] = {}
        resized = 0
        for nested in sdfg.all_sdfgs_recursive():
            assigned = interstate_bound_names(nested)
            if not assigned:
                continue
            for name, desc in nested.arrays.items():
                if (
                    not desc.transient
                    or not isinstance(desc, dt.Array)
                    or isinstance(desc, dt.View)
                    or not {str(s) for s in desc.free_symbols} & assigned
                ):
                    continue
                nonnegative = self.nonnegative(nested, known)
                shape = [extent_upper_bound(extent, assigned, nonnegative) for extent in desc.shape]
                if any(extent is None for extent in shape):
                    continue
                plan = resize_plan(nested, name, shape)
                if plan is None:
                    continue
                for owner, resized_name, resized_shape in plan:
                    owner.arrays[resized_name].set_shape(resized_shape)
                resized += 1
            if nested.parent_nsdfg_node is not None:
                resized += set_capacities(nested, assigned)
        return resized or None
