# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shrink a transient that only ever lives inside one map body down to the box that body touches."""

import re
from typing import Any

from dace import SDFG, SDFGState, data, dtypes, properties, subsets, symbolic
from dace.memlet import Memlet
from dace.optionals import required
from dace.sdfg import graph as gr
from dace.sdfg import nodes
from dace.sdfg.scope import is_devicelevel_gpu
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.dataflow.redundant_array import same_extent
from dace.transformation.passes.canonicalize.prune_unreferenced_transients import code_text

#: Storages whose buffer is a plain allocation this pass may resize.
RESIZABLE_STORAGE: tuple[dtypes.StorageType, ...] = (
    dtypes.StorageType.Default,
    dtypes.StorageType.CPU_Heap,
    dtypes.StorageType.Register,
)


def named_in_text(sdfg: SDFG, name: str, code: dict[int, str]) -> bool:
    """Whether ``name`` appears as a word in code a structural walk cannot see."""
    if id(sdfg) not in code:
        texts = [code_text(sdfg.init_code), code_text(sdfg.exit_code)]
        for state in sdfg.states():
            texts += [code_text(n.code) for n in state.nodes() if isinstance(n, nodes.Tasklet)]
        for edge in sdfg.all_interstate_edges():
            texts.append(code_text(edge.data.condition))
            texts += [code_text(a) for a in edge.data.assignments.values()]
        code[id(sdfg)] = "\n".join(texts)
    return re.search(rf"\b{re.escape(name)}\b", code[id(sdfg)]) is not None


#: Per descriptor name, each state naming it with the edges and the access nodes that do.
Uses = dict[str, list[tuple[SDFGState, list[gr.MultiConnectorEdge[Memlet]], list[nodes.AccessNode]]]]


def uses_by_name(sdfg: SDFG) -> Uses:
    """Every state of ``sdfg`` per descriptor it names, read once: the pass renames nothing."""
    uses: Uses = {}
    for state in sdfg.states():
        here: dict[str, tuple[list[gr.MultiConnectorEdge[Memlet]], list[nodes.AccessNode]]] = {}
        for edge in state.edges():
            if edge.data.data is not None:
                here.setdefault(edge.data.data, ([], []))[0].append(edge)
        for node in state.data_nodes():
            here.setdefault(node.data, ([], []))[1].append(node)
        for name, (edges, nodes_here) in here.items():
            uses.setdefault(name, []).append((state, edges, nodes_here))
    return uses


def map_local_accesses(uses: Uses, name: str) -> tuple[SDFGState, list[gr.MultiConnectorEdge[Memlet]]] | None:
    """The state and edges of ``name`` when every access to it sits strictly inside ONE map body."""
    if len(uses.get(name, ())) != 1:
        return None
    ((state, edges, nodes_here),) = uses[name]
    sdict = state.scope_dict()
    scopes = {sdict[n] for n in nodes_here}
    scopes.update(sdict[e.src] for e in edges)
    scopes.update(sdict[e.dst] for e in edges)
    if len(scopes) != 1 or not isinstance(next(iter(scopes)), nodes.MapEntry):
        return None
    if named_by_other_side(state, name):
        return None
    return state, edges


def named_by_other_side(state: SDFGState, name: str) -> bool:
    """Whether an edge at an access node of ``name`` in ``state`` is a copy whose memlet names the other container: its
    subset of ``name`` is implicit or an ``other_subset``, out of what the box is read from and what is rewritten."""
    return any(
        e.data.data != name and not e.data.is_empty()
        for node in state.data_nodes()
        if node.data == name
        for e in state.all_edges(node)
    )


def uniform_subset(edges: list[gr.MultiConnectorEdge[Memlet]]) -> subsets.Range | None:
    """The one subset every edge in ``edges`` reads or writes, or ``None`` when they disagree."""
    shared: subsets.Range | None = None
    for edge in edges:
        memlet = edge.data
        # A dynamic memlet names the box all the same: only how often it is touched is unknown.
        if memlet.other_subset is not None or memlet.wcr is not None:
            return None
        if not isinstance(memlet.subset, subsets.Range):
            return None
        if shared is None:
            shared = memlet.subset
        elif not same_extent(memlet.subset, shared):
            return None
    return shared


def map_body_accesses(
    uses: Uses, name: str
) -> tuple[SDFGState, list[gr.MultiConnectorEdge[Memlet]], dict[str, None]] | None:
    """The state and edges of ``name`` when its access nodes sit in ONE map body and every edge naming it lies in that
    body or in a map nested in it, with the symbols those nested maps define."""
    if len(uses.get(name, ())) != 1:
        return None
    ((state, edges, nodes_here),) = uses[name]
    sdict = state.scope_dict()
    body = sdict[nodes_here[0]] if nodes_here else None
    if not isinstance(body, nodes.MapEntry) or any(sdict[n] is not body for n in nodes_here):
        return None
    if named_by_other_side(state, name):
        return None

    def inside(node: nodes.Node) -> bool:
        scope = sdict[node]
        while scope is not None and scope is not body:
            scope = sdict[scope]
        return scope is body

    body_exit = state.exit_node(body)
    for edge in edges:
        for node in (edge.src, edge.dst):
            if node is body_exit or isinstance(node, nodes.NestedSDFG) or not inside(node):
                return None
            # A View keeps the strides it was given, which a resized buffer no longer has
            if isinstance(node, nodes.AccessNode) and isinstance(node.desc(state.sdfg), data.View):
                return None
    defined: dict[str, None] = {}
    for node in state.nodes():
        if isinstance(node, nodes.EntryNode) and node is not body and inside(node):
            if not isinstance(node, nodes.MapEntry):
                return None
            defined.update(dict.fromkeys(node.map.params))
            defined.update(dict.fromkeys(c for c in node.in_connectors if not c.startswith("IN_")))
    return state, edges, defined


def shared_dimensions(
    edges: list[gr.MultiConnectorEdge[Memlet]], defined: dict[str, None]
) -> list[tuple | None] | None:
    """Per dimension, the range every edge names when it is one unit-step range through symbols outside ``defined``,
    else ``None`` (the dimension stays whole); ``None`` overall when an edge is a copy or no dimension is shared."""
    shared: list[tuple | None] | None = None
    for edge in edges:
        memlet = edge.data
        if memlet.other_subset is not None or not isinstance(memlet.subset, subsets.Range):
            return None
        ranges = memlet.subset.ranges
        if shared is None:
            shared = list(ranges)
        elif len(ranges) != len(shared):
            return None
        for dim, (mine, kept) in enumerate(zip(ranges, shared)):
            if kept is not None and not same_extent(mine, kept):
                shared[dim] = None
    if shared is None:
        return None
    for dim, kept in enumerate(shared):
        if kept is None:
            continue
        if kept[2] != 1 or any(symbolic.free_symbols_and_functions(b) & defined.keys() for b in kept[:2]):
            shared[dim] = None
    return shared if any(kept is not None for kept in shared) else None


def device_level_accesses(sdfg: SDFG, accesses: list[tuple[SDFGState, gr.MultiConnectorEdge[Memlet]]]) -> bool:
    """Whether every access in ``accesses`` runs in device code, under a ``GPU_Device`` map."""
    return all(is_devicelevel_gpu(sdfg, state, edge.dst) for state, edge in accesses)


def nest_invariant_accesses(
    sdfg: SDFG, uses: Uses, name: str
) -> list[tuple[SDFGState, gr.MultiConnectorEdge[Memlet]]] | None:
    """Every access to ``name`` when ``sdfg`` is a nested SDFG and all of them sit outside any map."""
    if sdfg.parent_nsdfg_node is None:
        return None
    found: list[tuple[SDFGState, gr.MultiConnectorEdge[Memlet]]] = []
    for state, edges, nodes_here in uses.get(name, ()):
        sdict = state.scope_dict()
        if any(sdict[n] is not None for n in nodes_here):
            return None
        if named_by_other_side(state, name):
            return None
        for edge in edges:
            if sdict[edge.src] is not None or sdict[edge.dst] is not None:
                return None
            found.append((state, edge))
    return found or None


def box_is_invariant(sdfg: SDFG, box: subsets.Range) -> bool:
    """Whether the symbols ``box`` names keep one value for a whole execution of ``sdfg``."""
    assigned = set()
    for edge in sdfg.all_interstate_edges():
        assigned.update(edge.data.assignments.keys())
    for region in sdfg.all_control_flow_regions():
        if isinstance(region, LoopRegion) and region.loop_variable:
            assigned.add(region.loop_variable)
    return not ({str(sym) for sym in box.free_symbols} & assigned)


def bound_in_nests(
    accesses: list[tuple[SDFGState, gr.MultiConnectorEdge[Memlet]]], box: subsets.Range, code: dict[int, str]
) -> list[tuple[SDFG, str, list[gr.MultiConnectorEdge[Memlet]]]] | None:
    """The descriptors nested SDFGs bind to a buffer whose ``accesses`` all name ``box``, each with its own accesses."""
    found: list[tuple[SDFG, str, list[gr.MultiConnectorEdge[Memlet]]]] = []
    for _, outer in accesses:
        for node, name in ((outer.dst, outer.dst_conn), (outer.src, outer.src_conn)):
            if not isinstance(node, nodes.NestedSDFG) or any(sd is node.sdfg and name == n for sd, n, _ in found):
                continue
            if named_in_text(node.sdfg, name, code):
                return None
            inner: list[tuple[SDFGState, gr.MultiConnectorEdge[Memlet]]] = []
            for state in node.sdfg.states():
                if named_by_other_side(state, name):
                    return None
                inner += [(state, edge) for edge in state.edges() if edge.data.data == name]
            for _, edge in inner:
                named = uniform_subset([edge])
                if named is None or not box_is_invariant(node.sdfg, named):
                    return None
                if not names_the_box(named, node.symbol_mapping, box):
                    return None
            deeper = bound_in_nests(inner, inner[0][1].data.subset, code) if inner else []
            if deeper is None:
                return None
            found.append((node.sdfg, name, [edge for _, edge in inner]))
            found += deeper
    return found


def names_the_box(inner: subsets.Range, mapping: dict[str, Any], box: subsets.Range) -> bool:
    """Whether ``inner``, a subset in a nested SDFG's symbols, is ``box`` once ``mapping`` binds those symbols."""
    if len(inner.ranges) != len(box.ranges):
        return False
    for mine, theirs in zip(inner.ranges, box.ranges):
        for bound, expected in zip(mine, theirs):
            bound = symbolic.pystr_to_symbolic(bound)
            # A symbol the nest defines itself may share its name with one of the box.
            if any(s.name not in mapping for s in bound.free_symbols):
                return False
            bound = bound.subs({s: mapping[s.name] for s in bound.free_symbols}, simultaneous=True)
            if symbolic.simplify(bound - expected) != 0 and not same_extent(bound, expected):
                return False
    return True


def shrinks_the_buffer(desc: data.Array, size: tuple[Any, ...]) -> bool:
    """Whether a descriptor of extent ``size`` is provably smaller than ``desc``."""
    smaller = desc.total_size
    for dim in size:
        smaller = smaller / dim
    return symbolic.simplify(smaller) != 1 and bool(symbolic.equal(desc.total_size, 1) is not True)


@properties.make_properties
@transformation.explicit_cf_compatible
class ShrinkMapLocalTransients(ppl.Pass):
    """Resize a map-body-local transient to the single box its accesses name."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Descriptors | ppl.Modifies.Memlets | ppl.Modifies.Nodes))

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """Shrink every map-body-local transient that names one box."""
        shrunk = 0
        code: dict[int, str] = {}
        for sd in sdfg.all_sdfgs_recursive():
            uses = uses_by_name(sd)
            for name, desc in list(sd.arrays.items()):
                if not self.is_candidate(sd, name, desc):
                    continue
                if self.shrink_to_one_box(sd, uses, name, desc, code) or self.shrink_shared_dimensions(
                    sd, uses, name, desc, code
                ):
                    shrunk += 1
        return shrunk or None

    @staticmethod
    def shrink_to_one_box(sdfg: SDFG, uses: Uses, name: str, desc: data.Array, code: dict[int, str]) -> bool:
        """Resize ``name`` to the one box all its accesses name, in ``sdfg`` and the nests it is bound into."""
        in_map = map_local_accesses(uses, name)
        if in_map is not None:
            accesses = [(in_map[0], edge) for edge in in_map[1]]
        else:
            accesses = nest_invariant_accesses(sdfg, uses, name)
            if accesses is None:
                return False
        on_device = desc.storage == dtypes.StorageType.GPU_Global
        if on_device and not device_level_accesses(sdfg, accesses):
            return False
        shared = uniform_subset([edge for _, edge in accesses])
        if shared is None:
            return False
        if in_map is None and not box_is_invariant(sdfg, shared):
            return False
        size = tuple(shared.size())
        if len(size) != len(desc.shape) or not shrinks_the_buffer(desc, size):
            return False
        if named_in_text(sdfg, name, code):
            return False
        nested = bound_in_nests(accesses, shared, code)
        if nested is None:
            return False
        desc.set_shape(size)
        if on_device:
            # Thread-private by construction: every access sits under a kernel and names one iteration's box.
            desc.storage = dtypes.StorageType.Register(force=True)
        for inner, inner_name, _ in nested:
            inner.arrays[inner_name].set_shape(size)
            if on_device:
                inner.arrays[inner_name].storage = desc.storage
        for edge in [edge for _, edge in accesses] + [edge for _, _, edges in nested for edge in edges]:
            edge.data.subset = subsets.Range([(0, dim - 1, 1) for dim in size])
            edge.data.volume = required(edge.data.subset).num_elements()
        return True

    @staticmethod
    def shrink_shared_dimensions(sdfg: SDFG, uses: Uses, name: str, desc: data.Array, code: dict[int, str]) -> bool:
        """Resize ``name`` in the dimensions every access names alike (``t[i, 0:K]`` written, ``t[i, k]`` read under
        ``map[i]``: the ``i`` dimension), keeping the others whole."""
        if desc.storage == dtypes.StorageType.GPU_Global or any(o != 0 for o in desc.offset):
            return False
        # The kept dimensions keep their layout: a packed row- or column-major buffer stays one
        fortran = len(desc.shape) > 1 and not desc.is_packed_c_strides()
        if fortran and not desc.is_packed_fortran_strides():
            return False
        found = map_body_accesses(uses, name)
        if found is None:
            return False
        _, edges, defined = found
        shared = shared_dimensions(edges, defined)
        if shared is None or len(shared) != len(desc.shape):
            return False
        size = tuple(extent if kept is None else kept[1] - kept[0] + 1 for extent, kept in zip(desc.shape, shared))
        if not shrinks_the_buffer(desc, size) or named_in_text(sdfg, name, code):
            return False
        desc.set_shape(size)
        if fortran:
            desc.set_strides_from_layout(*range(len(size)))
        for edge in edges:
            edge.data.subset = subsets.Range(
                [
                    mine if kept is None else (0, mine[1] - mine[0], 1)
                    for mine, kept in zip(edge.data.subset.ranges, shared)
                ]
            )
        return True

    def is_candidate(self, sdfg: SDFG, name: str, desc: data.Data) -> bool:
        """Whether ``name`` is a plain resizable transient array of ``sdfg``."""
        if type(desc) is not data.Array or not desc.transient:
            return False
        if name in sdfg.constants_prop:
            return False
        if desc.lifetime != dtypes.AllocationLifetime.Scope:
            return False
        return desc.storage in RESIZABLE_STORAGE or desc.storage == dtypes.StorageType.GPU_Global
