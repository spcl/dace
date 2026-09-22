# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Moves a loop around a map into the map """

import copy
import dataclasses
from dace.sdfg.state import AbstractControlFlowRegion, ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
import dace.transformation.helpers as helpers
from dace import graphlib as nx
from dace.libraries.standard.nodes.copy import node as copy_node
from dace.libraries.standard.nodes.fill import node as fill_node
from dace.ordered import OrderedSet
from dace.sdfg.scope import ScopeTree
from dace import Memlet, data as dt, dtypes, nodes, properties, sdfg as sd, subsets as sbs, symbolic, symbol
from dace.sdfg import graph as gr, nodes, propagation, utils as sdutil
from dace.transformation import transformation
from sympy import diff
from typing import List, Set, Tuple

from dace.transformation.passes.analysis import loop_analysis


def fold(memlet_subset_ranges, itervar, lower, upper):
    return [(r[0].replace(symbol(itervar), lower), r[1].replace(symbol(itervar), upper), r[2])
            for r in memlet_subset_ranges]


def offset(memlet_subset_ranges, value):
    return (memlet_subset_ranges[0] + value, memlet_subset_ranges[1] + value, memlet_subset_ranges[2])


def _collect_nested_lane_accesses(body: SDFGState, nsdfg: nodes.NestedSDFG, reads: List, writes: List) -> None:
    """Append the per-lane accesses hidden inside ``nsdfg`` to ``reads`` / ``writes``.

    ``LoopToMap`` nests the map body, leaving WHOLE-ARRAY memlets on the NestedSDFG's connectors
    -- the per-lane indices that decide whether a carry crosses map lanes only exist inside. The
    inner subsets are rewritten into the outer symbol space through ``symbol_mapping``; an inner
    symbol with no mapping (a nested loop's own iterator) is left alone, which can only make two
    subsets look MORE different, i.e. can only lead to a refusal.
    """
    outer_data = {}
    for e in body.in_edges(nsdfg):
        if e.dst_conn and e.data is not None and e.data.data is not None:
            outer_data[e.dst_conn] = e.data.data
    for e in body.out_edges(nsdfg):
        if e.src_conn and e.data is not None and e.data.data is not None:
            outer_data[e.src_conn] = e.data.data
    subs = {symbol(k): symbolic.pystr_to_symbolic(v) for k, v in nsdfg.symbol_mapping.items()}
    for state in nsdfg.sdfg.all_states():
        for node in state.data_nodes():
            name = outer_data.get(node.data)
            if name is None:
                continue
            for edge, bucket, incoming in ([(ie, writes, True)
                                            for ie in state.in_edges(node)] + [(oe, reads, False)
                                                                               for oe in state.out_edges(node)]):
                if edge.data is None:
                    continue
                # ``get_*_subset`` resolves which side of the memlet indexes ``node`` even when the
                # memlet's ``data`` names the OTHER endpoint (a copy edge), where ``.subset`` would
                # be the other container's range.
                sub = (edge.data.get_dst_subset(edge, state) if incoming else edge.data.get_src_subset(edge, state))
                sub = sub if sub is not None else edge.data.subset
                if sub is None:
                    continue
                rewritten = [tuple(symbolic.pystr_to_symbolic(t).subs(subs) for t in dim) for dim in sub.ndrange()]
                bucket.append((name, sbs.Range(rewritten)))


def _differs_on_map_axis(read: sbs.Subset, write: sbs.Subset, mparams: Set[str]) -> bool:
    """True if ``read`` and ``write`` address a different position on some dimension indexed by
    a map parameter.

    That is a dependence that crosses map LANES. Before the interchange the map is re-entered
    once per loop iteration, so the loop's sequential order separates lane ``p``'s read from
    lane ``q``'s write; afterwards the map is the outer parallel axis and both run
    concurrently, so the interchange would introduce a race. The classic distance-vector
    statement: swapping the two axes turns a dependence ``(d_loop, d_map)`` into
    ``(d_map, d_loop)``, which stays a legal *parallel* outer map only when ``d_map == 0``.
    """
    rnd, wnd = list(read.ndrange()), list(write.ndrange())
    if len(rnd) != len(wnd):
        return True  # shape mismatch -> cannot pair the axes up; stay safe
    for r, w in zip(rnd, wnd):
        names = set()
        for token in (*r, *w):
            if symbolic.issymbolic(token):
                names |= {str(s) for s in token.free_symbols}
        if not (names & mparams):
            continue
        for rt, wt in zip(r, w):
            # One name can be TWO sympy symbols -- identity folds in the assumptions and the DaCe
            # dtype, and a subset rebuilt through arithmetic carries a differently-tagged ``i`` from
            # the one a bound was reparsed into. They never cancel, so ``i - i`` would read as a
            # nonzero lane distance and refuse a legal interchange.
            if symbolic.issymbolic(rt) and symbolic.issymbolic(wt):
                rt, wt = symbolic.equalize_symbols(rt, wt)
            if symbolic.simplify(rt - wt) != 0:
                return True
    return False


#: Library nodes that act element by element on their memlet subsets, so one lane's share of the node is the node
#: restricted to that lane's element.
ELEMENTWISE_LIBRARY_NODES = (copy_node.CopyLibraryNode, fill_node.FillLibraryNode)

#: One lane-indexed dimension of an access: ``(dimension, lane position, offset)``, indexing ``lane + offset``.
LaneDim = tuple[int, int, symbolic.SymbolicType]


@dataclasses.dataclass(slots=True)
class LaneFacts:
    """What the lane analysis of a loop body found; ``refusal`` is ``None`` when the interchange is legal."""
    lanes: list[tuple[SDFGState, nodes.MapEntry]]
    lane_containers: OrderedSet
    narrow: list[tuple[gr.MultiConnectorEdge, tuple[LaneDim, ...]]]
    refusal: str | None = None


def single_map_body(loop: LoopRegion) -> SDFGState | None:
    """The body state if ``loop`` has the one-state, one-map, one-component shape of the classic interchange."""
    if len(loop.nodes()) != 1 or not isinstance(loop.nodes()[0], SDFGState):
        return None
    body = loop.nodes()[0]
    if len(list(nx.weakly_connected_components(body._nx))) > 1:
        return None
    if sum(1 for node in body.nodes() if isinstance(node, nodes.MapEntry)) != 1:
        return None
    return body


def lane_maps(loop: LoopRegion) -> list[tuple[SDFGState, nodes.MapEntry]]:
    """Every top-level map of every state in ``loop``'s body, nested control flow included, nested SDFGs not."""
    found = []
    for state in loop.all_states():
        scope = state.scope_dict()
        found.extend(
            (state, node) for node in state.nodes() if isinstance(node, nodes.MapEntry) and scope[node] is None)
    return found


def names_of(expr) -> OrderedSet:
    return OrderedSet(str(s) for s in expr.free_symbols) if symbolic.issymbolic(expr) else OrderedSet()


def renamed(expr, rename: dict[str, str]):
    expr = symbolic.pystr_to_symbolic(expr)
    if not symbolic.issymbolic(expr):
        return expr
    return expr.subs({s: symbolic.symbol(rename[str(s)]) for s in expr.free_symbols if str(s) in rename})


def same_value(a, b) -> bool:
    a, b = symbolic.equalize_symbols(symbolic.pystr_to_symbolic(a), symbolic.pystr_to_symbolic(b))
    return symbolic.simplify(a - b) == 0


def lane_signature(subset: sbs.Range, rename: dict[str, str], lanes: list[str],
                   invariant: OrderedSet) -> tuple[LaneDim, ...] | None:
    """The lane dimensions of a map access, its parameters renamed onto the common ``lanes``; ``None`` unless every
    lane indexes exactly one dimension as ``lane + offset`` with a loop-invariant offset."""
    dims = []
    for d, (begin, end, _) in enumerate(subset.ndrange()):
        begin, end = (renamed(x, rename) for x in (begin, end))
        hit = [j for j, lane in enumerate(lanes) if lane in names_of(begin) | names_of(end)]
        if not hit:
            continue
        if len(hit) != 1 or not same_value(begin, end):
            return None
        lane = next(s for s in begin.free_symbols if str(s) == lanes[hit[0]])
        offset = symbolic.simplify(begin - lane)
        if not names_of(offset) <= invariant:
            return None
        dims.append((d, hit[0], offset))
    return tuple(dims) if sorted(j for _, j, _ in dims) == list(range(len(lanes))) else None


def footprint_signature(subset: sbs.Range, ref: sbs.Range, invariant: OrderedSet) -> tuple[LaneDim, ...] | None:
    """The lane dimensions of an elementwise access outside the maps: each non-point dimension must be one lane's
    range shifted by a loop-invariant offset. ``()`` for an all-point subset, ``None`` for any other range."""
    dims = []
    for d, (begin, end, step) in enumerate(subset.ndrange()):
        if same_value(begin, end):
            continue
        hit = [
            j for j, (rb, re, rs) in enumerate(ref)
            if same_value(step, 1) and same_value(rs, 1) and same_value(end - begin, re - rb)
        ]
        if len(hit) != 1:
            return None
        offset = symbolic.simplify(symbolic.pystr_to_symbolic(begin) - symbolic.pystr_to_symbolic(ref[hit[0]][0]))
        if not names_of(offset) <= invariant:
            return None
        dims.append((d, hit[0], offset))
    return tuple(dims)


def same_signature(a: tuple[LaneDim, ...], b: tuple[LaneDim, ...]) -> bool:
    return len(a) == len(b) and all(x[:2] == y[:2] and same_value(x[2], y[2]) for x, y in zip(a, b))


def map_accesses(state: SDFGState, entry: nodes.MapEntry) -> list[tuple[str, sbs.Range, bool]]:
    """``(container, subset, is_write)`` of every access a map scope makes, nested SDFGs included. A nested
    SDFG's connector memlet bounds every access behind it, so it stands for them when it pins each
    dimension a map parameter indexes to one point; any coarser connector memlet is looked through."""
    found = []
    scope = state.scope_subgraph(entry)
    params = OrderedSet(entry.map.params)
    bound = {}  # nested SDFG -> containers its connector memlets already bind per iteration
    for e in scope.edges():
        if e.data.is_empty():
            continue
        path = state.memlet_path(e)
        src, dst = path[0].src, path[-1].dst
        for nested in (e.src, e.dst):
            if isinstance(nested, nodes.NestedSDFG):
                dims = [(b, x) for b, x, _ in e.data.subset.ndrange() if (names_of(b) | names_of(x)) & params]
                if not dims or not all(same_value(b, x) for b, x in dims):
                    break
                bound.setdefault(nested, OrderedSet()).add(e.data.data)
        else:
            if isinstance(src, nodes.AccessNode) and src.data == e.data.data:
                found.append((src.data, e.data.get_src_subset(e, state) or e.data.subset, False))
            if isinstance(dst, nodes.AccessNode) and dst.data == e.data.data:
                found.append((dst.data, e.data.get_dst_subset(e, state) or e.data.subset, True))
    for node in scope.nodes():
        if isinstance(node, nodes.NestedSDFG):
            reads, writes = [], []
            _collect_nested_lane_accesses(state, node, reads, writes)
            skip = bound.get(node, OrderedSet())
            found.extend((name, sub, False) for name, sub in reads if name not in skip)
            found.extend((name, sub, True) for name, sub in writes if name not in skip)
    return found


def assigned_symbols(loop: LoopRegion) -> OrderedSet:
    names = OrderedSet(k for e in loop.all_interstate_edges() for k in e.data.assignments)
    names.update(r.loop_variable for r in loop.all_control_flow_regions() if isinstance(r, LoopRegion))
    return names


def escaping_symbol(loop: LoopRegion, sdfg: sd.SDFG) -> str | None:
    """A symbol the loop assigns that ``nest_sdfg_subgraph`` would export out of the nest, where every lane would
    race on it. Mirrors that helper's own internal/external split."""
    use_sites, descriptor_symbols = loop_analysis.symbol_use_sites(sdfg)
    incoming = sdfg.free_symbols
    for edge in loop.all_interstate_edges():
        for name, value in edge.data.assignments.items():
            if (name in symbolic.free_symbols_and_functions(value) or name in incoming
                    or loop_analysis.counter_used_outside_loop(name, loop, sdfg, use_sites, descriptor_symbols)):
                return name
    for region in loop.all_control_flow_regions():
        if not isinstance(region, LoopRegion) or not region.loop_variable:
            continue
        init = loop_analysis.get_init_assignment(region) if region.init_statement else None
        outright = not region.init_statement or (init is not None and region.loop_variable
                                                 not in symbolic.free_symbols_and_functions(init))
        if not outright or loop_analysis.counter_used_outside_loop(region.loop_variable, region, sdfg, use_sites,
                                                                   descriptor_symbols):
            return region.loop_variable
    return None


def names_outside_loop(loop: LoopRegion, sdfg: sd.SDFG) -> OrderedSet:
    """Containers referenced outside ``loop``, by the same rule ``nest_sdfg_subgraph`` uses to keep them outside."""
    inside = OrderedSet([loop]) | OrderedSet(loop.all_control_flow_blocks())
    names = OrderedSet()
    for block in sdfg.all_control_flow_blocks():
        if block in inside:
            continue
        if isinstance(block, SDFGState):
            names.update(n.data for n in block.data_nodes())
        elif isinstance(block, ConditionalBlock):
            names.update(s for c, _ in block.branches if c is not None for s in c.get_free_symbols())
        elif isinstance(block, LoopRegion):
            names.update(block.loop_condition.get_free_symbols())
    for edge in sdfg.all_interstate_edges():
        if edge.src not in inside or edge.dst not in inside:
            names.update(edge.data.free_symbols)
    return names


def control_flow_reads(loop: LoopRegion, sdfg: sd.SDFG) -> OrderedSet:
    """Containers the loop's conditions, headers and interstate assignments read."""
    names = OrderedSet(s for e in loop.all_interstate_edges() for s in e.data.free_symbols)
    for region in loop.all_control_flow_regions():
        if isinstance(region, ConditionalBlock):
            names.update(s for c, _ in region.branches if c is not None for s in c.get_free_symbols())
        elif isinstance(region, LoopRegion):
            for code in (region.loop_condition, region.init_statement, region.update_statement):
                if code is not None:
                    names.update(code.get_free_symbols())
    return OrderedSet(n for n in names if n in sdfg.arrays)


def edge_containers(state: SDFGState, e: gr.MultiConnectorEdge) -> OrderedSet:
    path = state.memlet_path(e)
    ends = [n.data for n in (path[0].src, path[-1].dst) if isinstance(n, nodes.AccessNode)]
    return OrderedSet([e.data.data] + ends)


def elementwise_refusal(state: SDFGState, node: nodes.LibraryNode, ref: sbs.Range, invariant: OrderedSet,
                        accesses: dict, uniform: tuple[OrderedSet, OrderedSet], narrow: list, wide: list) -> str | None:
    """Classify one top-level Copy/Fill: lane-shaped edges are narrowed to the lane, all-point edges are uniform, and
    a Fill over whole dimensions goes to ``wide`` for :func:`wide_fill_refusal`."""
    lane_edges, plain_edges = [], []
    for e in state.all_edges(node):
        if e.data.is_empty():
            continue
        sig = footprint_signature(e.data.subset, ref, invariant)
        if sig is None and isinstance(node, fill_node.FillLibraryNode) and e.src is node:
            if not isinstance(e.dst, nodes.AccessNode) or e.data.wcr is not None or e.data.other_subset is not None:
                return f'{node.label} has an ambiguous memlet {e.data}'
            wide.append((node, e))
            continue
        if sig is None or (sig and sorted(j for _, j, _ in sig) != list(range(len(ref)))):
            return f'{node.label} covers {e.data.subset}, which is not the lanes'
        (lane_edges if sig else plain_edges).append((e, sig))
    if not lane_edges:
        for e, _ in plain_edges:
            uniform[e.src is node].update(edge_containers(state, e))
        return None
    if any(e.dst_conn != fill_node.FillLibraryNode.VALUE_CONNECTOR_NAME for e, _ in plain_edges):
        return f'{node.label} mixes lane and non-lane subsets'
    for e, sig in lane_edges:
        end = e.dst if e.src is node else e.src
        if not isinstance(end, nodes.AccessNode) or end.data != e.data.data or e.data.other_subset is not None:
            return f'{node.label} has an ambiguous memlet {e.data}'
        if e.data.wcr is not None:
            return f'{node.label} writes {e.data.data} with a conflict resolution'
        accesses.setdefault(end.data, []).append((e.src is node, sig))
        narrow.append((e, sig))
    for e, _ in plain_edges:
        uniform[False].update(edge_containers(state, e))
    return None


def provably_nonnegative(expr) -> bool:
    value = symbolic.simplify(symbolic.pystr_to_symbolic(expr))
    return value.is_Number and value >= 0


def wide_fill_refusal(loop: LoopRegion, sdfg: sd.SDFG, node: nodes.LibraryNode, edge: gr.MultiConnectorEdge,
                      sig: tuple[LaneDim, ...] | None, ref: sbs.Range) -> str | None:
    """A Fill wider than the lanes may shrink to each lane's element when the part outside the lanes is dead: the
    container is a transient the maps index per lane (``sig``), the Fill spans its whole lane dimensions (so it
    covers every lane), and every read of it anywhere in ``sdfg`` stays within the lanes."""
    data = edge.dst.data
    desc = sdfg.arrays[data]
    if sig is None or not desc.transient:
        return f'{node.label} fills {data} beyond the lanes'
    dims = list(edge.data.subset.ndrange())
    lane_dims = {d for d, _, _ in sig}
    for d, (begin, end, _) in enumerate(dims):
        whole = same_value(begin, 0) and same_value(end, desc.shape[d] - 1)
        if (d in lane_dims and not whole) or (d not in lane_dims and not same_value(begin, end)):
            return f'{node.label} fills {data}[{edge.data.subset}], neither the lanes nor whole dimensions'
    if data in control_flow_reads_outside(loop, sdfg):
        return f'{data}, filled beyond the lanes, is read by control flow outside the loop'
    inside = OrderedSet(loop.all_states())
    for state in sdfg.all_states():
        if state in inside:
            continue
        for access in state.data_nodes():
            if access.data != data:
                continue
            reads = [(e, e.data.get_src_subset(e, state)) for e in state.out_edges(access) if not e.data.is_empty()]
            reads += [(e, e.data.get_dst_subset(e, state)) for e in state.in_edges(access) if e.data.wcr is not None]
            for e, sub in reads:
                sub = sub if sub is not None else e.data.subset
                for d, j, offset in sig:
                    begin, end, _ = sub.ndrange()[d]
                    if not (provably_nonnegative(begin - ref[j][0] - offset)
                            and provably_nonnegative(ref[j][1] + offset - end)):
                        return f'{data}, filled beyond the lanes, is read outside them in {state.label}'
    return None


def control_flow_reads_outside(loop: LoopRegion, sdfg: sd.SDFG) -> OrderedSet:
    """Containers read by conditions and interstate edges outside ``loop``."""
    inside = OrderedSet([loop]) | OrderedSet(loop.all_control_flow_blocks())
    names = OrderedSet()
    for edge in sdfg.all_interstate_edges():
        if edge.src not in inside or edge.dst not in inside:
            names.update(edge.data.free_symbols)
    for region in sdfg.all_control_flow_regions():
        if region in inside:
            continue
        if isinstance(region, ConditionalBlock):
            names.update(s for c, _ in region.branches if c is not None for s in c.get_free_symbols())
        elif isinstance(region, LoopRegion):
            for code in (region.loop_condition, region.init_statement, region.update_statement):
                if code is not None:
                    names.update(code.get_free_symbols())
    return OrderedSet(n for n in names if n in sdfg.arrays)


def lane_refusal(loop: LoopRegion, sdfg: sd.SDFG, facts: LaneFacts) -> str | None:
    """Why ``loop`` cannot become one parallel map over its maps' common range with the loop inside, or ``None``.

    The body may be any control flow of states, branches and loops. Every top-level map in it must span the same
    range: these are the lanes. A container a map writes must be indexed ``lane + c`` (``c`` loop-invariant) by
    every access, so no lane ever touches another lane's element; Copy/Fill nodes outside the maps count as maps
    when their range is exactly the lanes, or is a whole-dimension Fill whose remainder is dead
    (:func:`wide_fill_refusal`). Everything else must be lane-independent: control flow and code outside
    the maps may not read what the maps write, and what it writes becomes a private copy per lane, so it must not be
    observed outside the loop.
    """
    for block in loop.all_control_flow_blocks():
        if not isinstance(block, (SDFGState, AbstractControlFlowRegion)):
            return f'{type(block).__name__} {block.label} in the body'
    if not facts.lanes:
        return 'no map in the body'
    ref_state, ref_entry = facts.lanes[0]
    ref = ref_entry.map.range
    invariant = (OrderedSet(sdfg.symbols) | OrderedSet(sdfg.constants)) - assigned_symbols(loop)
    for state, entry in facts.lanes:
        if len(entry.map.range) != len(ref) or not all(
                same_value(x, y) for r, q in zip(entry.map.range, ref) for x, y in zip(r, q)):
            return f'map {entry.map.label} spans {entry.map.range}, not {ref}'
        if any(not c.startswith('IN_') for c in entry.in_connectors):
            return f'map {entry.map.label} has a dynamic range'
    if all(same_value(b, e) for b, e, _ in ref):
        return 'the maps span a single lane'
    if not OrderedSet(str(s) for s in ref.free_symbols) <= invariant:
        return f'the lane range {ref} changes inside the loop'

    lanes = list(ref_entry.map.params)
    accesses: dict[str, list[tuple[bool, tuple[LaneDim, ...] | None]]] = {}
    uniform = (OrderedSet(), OrderedSet())  # (reads, writes) of lane-independent code
    top_names = OrderedSet()
    wide = []
    for state, entry in facts.lanes:
        rename = dict(zip(entry.map.params, lanes))
        for data, subset, write in map_accesses(state, entry):
            accesses.setdefault(data, []).append((write, lane_signature(subset, rename, lanes, invariant)))
    for state in loop.all_states():
        scope = state.scope_dict()
        for node in state.nodes():
            if scope[node] is not None or isinstance(node, (nodes.MapEntry, nodes.MapExit)):
                continue
            if isinstance(node, nodes.AccessNode):
                top_names.add(node.data)
                for e in state.out_edges(node):
                    if isinstance(e.dst, nodes.AccessNode) and not e.data.is_empty():
                        uniform[0].add(node.data)
                        uniform[1].add(e.dst.data)
            elif isinstance(node, ELEMENTWISE_LIBRARY_NODES):
                reason = elementwise_refusal(state, node, ref, invariant, accesses, uniform, facts.narrow, wide)
                if reason is not None:
                    return reason
            else:
                for e in state.all_edges(node):
                    if not e.data.is_empty():
                        uniform[e.src is node].update(edge_containers(state, e))
                        if e.src is node and e.data.wcr is not None:
                            uniform[0].update(edge_containers(state, e))
    uniform[0].update(control_flow_reads(loop, sdfg))

    lane_written = OrderedSet(d for d, found in accesses.items() if any(w for w, _ in found))
    outside = names_outside_loop(loop, sdfg)
    for data in uniform[1] | lane_written:
        desc = sdfg.arrays[data]
        private = desc.transient and data not in outside
        if data in uniform[1]:
            if data in lane_written:
                return f'{data} is written both per lane and by lane-independent code'
            if not private:
                return f'lane-independent code writes {data}, which is observed outside the loop'
            continue
        if isinstance(desc, dt.View):
            return f'the maps write through the view {data}'
        if data not in top_names and data not in uniform[0] and private:
            continue  # map-internal scratch: each lane gets its own copy
        if data in uniform[0]:
            return f'lane-independent code reads {data}, which the maps write per lane'
        signatures = [sig for _, sig in accesses[data]]
        if signatures[0] is None or not all(s is not None and same_signature(signatures[0], s) for s in signatures):
            return f'{data} is accessed across lanes'
        facts.lane_containers.add(data)
    for node, edge in wide:
        sig = next((s for _, s in accesses.get(edge.dst.data, ()) if s), None)
        reason = wide_fill_refusal(loop, sdfg, node, edge, sig if edge.dst.data in facts.lane_containers else None, ref)
        if reason is not None:
            return reason
        facts.narrow.append((edge, sig))

    escaped = escaping_symbol(loop, sdfg)
    if escaped is not None:
        return f'symbol {escaped}, set in the loop, is read after it'
    return None


def analyze_lanes(loop: LoopRegion, sdfg: sd.SDFG) -> LaneFacts:
    facts = LaneFacts(lane_maps(loop), OrderedSet(), [])
    facts.refusal = lane_refusal(loop, sdfg, facts)
    return facts


def move_loop_into_lane_maps(loop: LoopRegion, sdfg: sd.SDFG) -> nodes.MapEntry:
    """Rewrite ``for(...) { body }`` into ``map(lanes) { for(...) { body' } }``, where ``body'`` runs every map of
    ``body`` on its own lane only. Each map keeps its scope, narrowed to the one-iteration range ``[lane, lane]``,
    so no memlet inside it changes. The caller has checked :func:`lane_refusal`.

    :returns: The entry of the new lane map.
    """
    facts = analyze_lanes(loop, sdfg)
    graph = loop.parent_graph
    ref_state, ref_entry = facts.lanes[0]
    ref = copy.deepcopy(ref_entry.map.range)
    types = ref_entry.new_symbols(sdfg, ref_state, sdfg.symbols)
    taken = OrderedSet(sdfg.symbols) | OrderedSet(sdfg.arrays) | OrderedSet(p for _, e in facts.lanes
                                                                            for p in e.map.params)
    lane_syms = [symbolic.symbol(dt.find_new_name(f'{p}_lane', taken), types[p]) for p in ref_entry.map.params]

    state = helpers.nest_sdfg_subgraph(sdfg, gr.SubgraphView(graph, [loop]), keep_outside=facts.lane_containers)
    nsdfg = next(n for n in state.nodes() if isinstance(n, nodes.NestedSDFG))
    for lane in lane_syms:
        nsdfg.sdfg.add_symbol(lane.name, lane.dtype)
        nsdfg.symbol_mapping[lane.name] = lane
    for _, entry in facts.lanes:
        entry.map.range = sbs.Range([(lane, lane, 1) for lane in lane_syms])
    for edge, sig in facts.narrow:
        dims = list(edge.data.subset.ndrange())
        for d, j, offset in sig:
            dims[d] = (lane_syms[j] + offset, lane_syms[j] + offset, 1)
        edge.data.subset = sbs.Range(dims)
        edge.data.volume = edge.data.subset.num_elements()

    entry, _ = helpers.wrap_code_node_in_unit_map(state, nsdfg, dtypes.ScheduleType.Default, '_lanes')
    entry.map.label = f'{loop.label}_lanes'
    entry.map.params = [lane.name for lane in lane_syms]
    entry.map.range = ref
    propagation.propagate_memlets_state(sdfg, state)
    sdfg.reset_cfg_list()
    return entry


@properties.make_properties
@transformation.explicit_cf_compatible
class MoveLoopIntoMap(transformation.MultiStateTransformation):
    """
    Moves a loop around a map into the map.

    With ``cfg_body`` the body may also be any control flow whose maps share one range (see :func:`lane_refusal`):
    the loop then moves into one map over that range and every map of the body runs on its own lane. A body of the
    classic one-state, one-map shape takes the classic path either way.
    """

    loop = transformation.PatternNode(LoopRegion)

    cfg_body = properties.Property(dtype=bool,
                                   default=False,
                                   desc='Also interchange a loop whose body is control flow over maps of one range')

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.loop)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        if self.cfg_body and single_map_body(self.loop) is None:
            return analyze_lanes(self.loop, sdfg).refusal is None

        # If loop information cannot be determined, fail.
        start = loop_analysis.get_init_assignment(self.loop)
        end = loop_analysis.get_loop_end(self.loop)
        step = loop_analysis.get_loop_stride(self.loop)
        itervar = self.loop.loop_variable
        if start is None or end is None or step is None or itervar is None:
            return False

        if step not in [-1, 1]:
            return False

        # Body must contain a single state
        if len(self.loop.nodes()) != 1 or not isinstance(self.loop.nodes()[0], SDFGState):
            return False
        body: SDFGState = self.loop.nodes()[0]

        # Body must have only a single connected component
        # NOTE: This is a strict check that can be potentially relaxed.
        # If only one connected component includes a Map and the others do not create RW dependencies, then we could
        # proceed with the transformation. However, that would be a case of an SDFG with redundant computation/copying,
        # which is unlikely after simplification transformations. Alternatively, we could try to apply the
        # transformation to each component separately, but this would require a lot more checks.
        if len(list(nx.weakly_connected_components(body._nx))) > 1:
            return False

        # Check if body contains exactly one map
        maps = [node for node in body.nodes() if isinstance(node, nodes.MapEntry)]
        if len(maps) != 1:
            return False

        map_entry = maps[0]
        map_exit = body.exit_node(map_entry)
        subgraph = body.scope_subgraph(map_entry)
        read_set, write_set = body.read_and_write_sets()

        # Check for iteration variable in map and data descriptors
        if str(itervar) in map_entry.free_symbols:
            return False
        for arr in (read_set | write_set):
            if str(itervar) in set(map(str, sdfg.arrays[arr].free_symbols)):
                return False

        # Check that everything else outside the Map is independent of the loop's itervar
        for e in body.edges():
            if e.src in subgraph.nodes() or e.dst in subgraph.nodes():
                continue
            if e.dst is map_entry and isinstance(e.src, nodes.AccessNode):
                continue
            if e.src is map_exit and isinstance(e.dst, nodes.AccessNode):
                continue
            if str(itervar) in e.data.free_symbols:
                return False
            if isinstance(e.dst, nodes.AccessNode) and e.dst.data in read_set:
                # NOTE: This is strict check that can be potentially relaxed.
                # If some data written indirectly by the Map (i.e., it is not an immediate output of the MapExit) is
                # also read, then abort. In practice, we could follow the edges and with subset compositions figure out
                # if there is a RW dependency on the loop variable. However, in such complicated cases, it is far more
                # likely that the simplification redundant array/copying transformations trigger first. If they don't,
                # this is a good hint that there is a RW dependency.
                if nx.has_path(body._nx, map_exit, e.dst):
                    return False
        for n in body.nodes():
            if n in subgraph.nodes():
                continue
            if str(itervar) in n.free_symbols:
                return False

        def test_subset_dependency(subset: sbs.Subset, mparams: Set[int]) -> Tuple[bool, List[int]]:
            dims = []
            for i, r in enumerate(subset):
                if not isinstance(r, (list, tuple)):
                    r = [r]
                fsymbols = set()
                for token in r:
                    if symbolic.issymbolic(token):
                        fsymbols = fsymbols.union({str(s) for s in token.free_symbols})
                if itervar in fsymbols:
                    if fsymbols.intersection(mparams):
                        return (False, [])
                    else:
                        # Strong checks
                        if not permissive:
                            # Only indices allowed
                            if len(r) > 1 and r[0] != r[1]:
                                return (False, [])
                            # Injective IN THE LOOP ITERATOR, so differentiate by it explicitly.
                            # Omitting the variable only works while the index happens to hold a
                            # single free symbol; a two-symbol index (``_loop_it_0 + _loop_it_2``,
                            # which polybench ``lu`` produces once loop origins are rebased) makes
                            # sympy refuse to guess and raise instead of answering the question.
                            # Injective IN THE LOOP ITERATOR, so differentiate by it explicitly.
                            # Letting sympy infer the variable only works while the index holds a
                            # single free symbol; a two-symbol index (``_loop_it_0 + _loop_it_2``,
                            # which polybench ``lu`` produces once loop origins are rebased) makes
                            # it refuse to guess and raise instead of answering the question.
                            # Take the symbol INSTANCE out of the expression rather than building a
                            # fresh one -- a reconstructed symbol carries different assumptions, so
                            # ``diff`` would not match it and would answer 0 for every index.
                            ivsym = next((s for s in r[0].free_symbols if str(s) == str(itervar)), None)
                            derivative = diff(r[0], ivsym) if ivsym is not None else 0
                            # Index function must be injective
                            if not (((derivative > 0) == True) or ((derivative < 0) == True)):
                                return (False, [])
                        dims.append(i)
            return (True, dims)

        # Check that Map memlets depend on itervar in a consistent manner
        # a. A container must either not depend at all on itervar, or depend on it always in the same dimensions.
        # b. Abort when a dimension depends on both the itervar and a Map parameter.
        mparams = set(map_entry.map.params)
        data_dependency = dict()
        for e in body.edges():
            if e.src in subgraph.nodes() and e.dst in subgraph.nodes():
                if itervar in e.data.free_symbols:
                    e.data.try_initialize(sdfg, subgraph, e)
                    for i, subset in enumerate((e.data.src_subset, e.data.dst_subset)):
                        if subset:
                            if i == 0:
                                access = body.memlet_path(e)[0].src
                            else:
                                access = body.memlet_path(e)[-1].dst
                            passed, dims = test_subset_dependency(subset, mparams)
                            if not passed:
                                return False
                            if dims:
                                if access.data in data_dependency:
                                    if data_dependency[access.data] != dims:
                                        return False
                                else:
                                    data_dependency[access.data] = dims

        # A container both read and written INSIDE the map may carry a dependence from one loop
        # iteration to the next. The interchange only preserves it while that dependence stays
        # inside a single map lane; a read/write pair that lands on different positions of a
        # map-parameter axis (``a[i-1, j+1]`` read vs ``a[i, j]`` write) crosses lanes, and after
        # the swap the lanes run concurrently -- a race. Refuse. The ``for(seq) { map }``
        # recurrence sweeps this pass exists for (TSVC s231 / s233 / s235,
        # ``aa[j, i] = aa[j-1, i] + ...``) carry only on the loop axis and read the map axis at the
        # same position, so they still interchange.
        lane_reads: List[Tuple[str, sbs.Subset]] = []
        lane_writes: List[Tuple[str, sbs.Subset]] = []
        for e in body.edges():
            if e.src not in subgraph.nodes() or e.dst not in subgraph.nodes():
                continue
            if e.data is None or e.data.data is None:
                continue
            if isinstance(e.src, nodes.NestedSDFG) or isinstance(e.dst, nodes.NestedSDFG):
                continue  # coarse connector memlet; the descent below reads the precise indices
            path = body.memlet_path(e)
            src, dst = path[0].src, path[-1].dst
            if isinstance(src, nodes.AccessNode) and src.data == e.data.data:
                sub = e.data.get_src_subset(e, body) or e.data.subset
                if sub is not None:
                    lane_reads.append((src.data, sub))
            if isinstance(dst, nodes.AccessNode) and dst.data == e.data.data:
                sub = e.data.get_dst_subset(e, body) or e.data.subset
                if sub is not None:
                    lane_writes.append((dst.data, sub))
        for node in subgraph.nodes():
            if isinstance(node, nodes.NestedSDFG):
                _collect_nested_lane_accesses(body, node, lane_reads, lane_writes)
        for wdata, wsub in lane_writes:
            for rdata, rsub in lane_reads:
                if rdata == wdata and _differs_on_map_axis(rsub, wsub, mparams):
                    return False

        return True

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        if self.cfg_body and single_map_body(self.loop) is None:
            move_loop_into_lane_maps(self.loop, sdfg)
            return
        body: sd.SDFGState = self.loop.nodes()[0]
        itervar = self.loop.loop_variable

        for node in body.nodes():
            if isinstance(node, nodes.MapEntry):
                map_entry = node
            if isinstance(node, nodes.MapExit):
                map_exit = node

        # nest map's content in sdfg
        map_subgraph = body.scope_subgraph(map_entry, include_entry=False, include_exit=False)
        nsdfg = helpers.nest_state_subgraph(sdfg, body, map_subgraph, full_data=True)
        nested_state: SDFGState = nsdfg.sdfg.nodes()[0]

        # replicate loop in nested sdfg
        inner_loop = LoopRegion(self.loop.label, self.loop.loop_condition, self.loop.loop_variable,
                                self.loop.init_statement, self.loop.update_statement, self.loop.inverted, nsdfg,
                                self.loop.update_before_condition)
        inner_loop.add_node(nested_state, is_start_block=True)
        nsdfg.sdfg.remove_node(nested_state)
        nsdfg.sdfg.add_node(inner_loop, is_start_block=True)

        # ``body`` was created inside the LoopRegion, so its label is only unique *there*.
        #  Hoisting it into ``graph`` with a raw ``add_node`` bypasses the block-name
        #  uniquifier that ``add_state`` applies, and the label it carries is a fixed one
        #  (``LoopToMap`` names every one of them ``single_state_body``). The second loop in
        #  a region to go through LoopToMap + this transformation therefore produced a
        #  duplicate, and ``validate_sdfg`` rejects the SDFG with "Found multiple blocks with
        #  the same name". Rename against the labels actually present rather than the CFG's
        #  cached ``_labels`` set, which a re-parenting like this one leaves stale.
        body.label = dt.find_new_name(body.label, {block.label for block in graph.nodes()})
        graph.add_node(body, is_start_block=(graph.start_block is self.loop), ensure_unique_name=True)
        for ie in graph.in_edges(self.loop):
            graph.add_edge(ie.src, body, ie.data)
        for oe in graph.out_edges(self.loop):
            graph.add_edge(body, oe.dst, oe.data)
        graph.remove_node(self.loop)

        if itervar in nsdfg.symbol_mapping:
            del nsdfg.symbol_mapping[itervar]
        if itervar in sdfg.symbols:
            del sdfg.symbols[itervar]

        # Add missing data/symbols
        for s in nsdfg.sdfg.free_symbols:
            if s in nsdfg.symbol_mapping:
                continue
            if s in sdfg.symbols:
                nsdfg.symbol_mapping[s] = s
                if s not in nsdfg.sdfg.symbols:
                    nsdfg.sdfg.add_symbol(s, sdfg.symbols[s])
            elif s in sdfg.arrays:
                desc = sdfg.arrays[s]
                access = body.add_access(s)
                conn = nsdfg.sdfg.add_datadesc(s, copy.deepcopy(desc))
                nsdfg.sdfg.arrays[s].transient = False
                nsdfg.add_in_connector(conn)
                body.add_memlet_path(access, map_entry, nsdfg, memlet=Memlet.from_array(s, desc), dst_conn=conn)
            else:
                raise NotImplementedError(f"Free symbol {s} is neither a symbol nor data.")
        to_delete = set()
        for s in nsdfg.symbol_mapping:
            if s not in nsdfg.sdfg.free_symbols:
                to_delete.add(s)
        for s in to_delete:
            del nsdfg.symbol_mapping[s]

        # propagate scope for correct volumes
        scope_tree = ScopeTree(map_entry, map_exit)
        scope_tree.parent = ScopeTree(None, None)
        # The first execution helps remove apperances of symbols
        # that are now defined only in the nested SDFG in memlets.
        propagation.propagate_memlets_scope(sdfg, body, scope_tree)

        # ``is_symbol_unused`` walks every descriptor, every state and every interstate edge PER
        # symbol. It never reads ``sdfg.symbols``, and ``remove_symbol`` touches only that (plus the
        # PARENT nested-SDFG node's symbol_mapping, which lives outside ``sdfg``), so one union of
        # the used names answers all of them.
        used_symbols = set()
        for desc in sdfg.arrays.values():
            used_symbols.update(str(s) for s in desc.free_symbols)
        for state in sdfg.states():
            used_symbols.update(state.free_symbols)
        for e in sdfg.all_interstate_edges():
            used_symbols.update(e.data.free_symbols)
        for s in to_delete:
            if s not in used_symbols:
                sdfg.remove_symbol(s)

        sdfg.reset_cfg_list()

        from dace.transformation.interstate import RefineNestedAccess
        transformation = RefineNestedAccess()
        transformation.setup_match(sdfg, body.parent_graph.cfg_id, body.block_id,
                                   {RefineNestedAccess.nsdfg: body.node_id(nsdfg)}, 0)
        transformation.apply(body, sdfg)

        # Second propagation for refined accesses.
        propagation.propagate_memlets_scope(sdfg, body, scope_tree)
