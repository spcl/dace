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
from dace.transformation.passes.canonicalize.prune_unreferenced_transients import code_text

#: Storages whose buffer is a plain allocation this pass may resize.
RESIZABLE_STORAGE: tuple[dtypes.StorageType, ...] = (
    dtypes.StorageType.Default,
    dtypes.StorageType.CPU_Heap,
    dtypes.StorageType.Register,
)


def named_in_text(sdfg: SDFG, name: str, code: dict[int, str]) -> bool:
    """Whether ``name`` appears as a word in code a structural walk cannot see.

    :param sdfg: SDFG to read (its own blocks only).
    :param name: Descriptor name.
    :param code: The code of each SDFG read so far, by ``id``: the pass rewrites none of it.
    :returns: ``True`` when a tasklet body, interstate edge or loop header spells the name.
    """
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
    """The state and edges of ``name`` when every access to it sits strictly inside ONE map body.

    :param uses: :func:`uses_by_name` of the SDFG owning the descriptor.
    :param name: Descriptor name.
    :returns: ``(state, edges)``, or ``None`` when the accesses are not confined to one map body.
    """
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
    """The one subset every edge in ``edges`` reads or writes, or ``None`` when they disagree.

    :param edges: Edges naming one descriptor.
    :returns: The shared subset, or ``None``.
    """
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
        elif memlet.subset != shared:
            return None
    return shared


def device_level_accesses(sdfg: SDFG, accesses: list[tuple[SDFGState, gr.MultiConnectorEdge[Memlet]]]) -> bool:
    """Whether every access in ``accesses`` runs in device code, under a ``GPU_Device`` map.

    A ``GPU_Global`` transient is resizable only there. Shrunk under a kernel, it becomes
    thread-private; under a host map it would stay one device buffer shared by the host loop's
    iterations.

    :param sdfg: SDFG owning the states.
    :param accesses: ``(state, edge)`` pairs naming the descriptor.
    :returns: ``True`` when every edge is device-level.
    """
    return all(is_devicelevel_gpu(sdfg, state, edge.dst) for state, edge in accesses)


def nest_invariant_accesses(
    sdfg: SDFG, uses: Uses, name: str
) -> list[tuple[SDFGState, gr.MultiConnectorEdge[Memlet]]] | None:
    """Every access to ``name`` when ``sdfg`` is a nested SDFG and all of them sit outside any map.

    A per-iteration buffer also appears in a map body nested into its own SDFG, as a top-level
    transient of the nest indexed by a symbol the nest receives from the enclosing map
    (``tx_times_tx[_loop_it_27]`` in npbench warpx_boris_push). Each nest instance then touches one
    box if nothing inside the nest reassigns the symbols the box names. :func:`box_is_invariant`
    checks that condition.

    :param sdfg: SDFG owning the descriptor.
    :param uses: :func:`uses_by_name` of ``sdfg``.
    :param name: Descriptor name.
    :returns: ``(state, edge)`` pairs, or ``None`` when ``sdfg`` is the root or an access is in a map.
    """
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
    """Whether the symbols ``box`` names keep one value for a whole execution of ``sdfg``.

    :param sdfg: SDFG whose control flow is checked (its own regions only).
    :param box: The subset every access names.
    :returns: ``False`` when an interstate edge assigns one of them or a loop iterates over one.
    """
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
    """The descriptors nested SDFGs bind to a buffer whose ``accesses`` all name ``box``, each with its own accesses.

    A nested SDFG holds its own descriptor of the buffer, laid out like the outer one, and indexes it itself: the
    buffer may shrink only if every access in the nest names the same box under the nest's symbol mapping, through
    symbols nothing in the nest rebinds. The descriptors of nests inside the nest follow.

    :param accesses: ``(state, edge)`` pairs naming the buffer in the SDFG that owns it.
    :param box: The subset each of them names.
    :param code: See :func:`named_in_text`.
    :returns: ``(nested SDFG, name, edges)`` per bound descriptor, or ``None`` when a nest touches anything else.
    """
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
            if symbolic.simplify(bound - expected) != 0:
                return False
    return True


def shrinks_the_buffer(desc: data.Array, size: tuple[Any, ...]) -> bool:
    """Whether a descriptor of extent ``size`` is provably smaller than ``desc``.

    :param desc: The descriptor as it stands.
    :param size: The candidate extent.
    :returns: ``True`` when the candidate is provably the smaller buffer.
    """
    smaller = desc.total_size
    for dim in size:
        smaller = smaller / dim
    return symbolic.simplify(smaller) != 1 and bool(symbolic.equal(desc.total_size, 1) is not True)


@properties.make_properties
@transformation.explicit_cf_compatible
class ShrinkMapLocalTransients(ppl.Pass):
    """Resize a map-body-local transient to the single box its accesses name.

    A pass that internalizes a computation -- an inline of a nested SDFG, a fusion that pulls a
    producer into its consumer -- can leave a FULL-EXTENT transient whose every access sits inside
    one map body and names one box, e.g. ``rsq[i, j]`` under ``map[i, j, k]``. The descriptor is
    then scope-local, so codegen allocates the whole extent ONCE PER ITERATION, and the scope
    default storage inside a ``CPU_Multicore`` map is ``Register``: an ``N*N`` stack array per
    iteration, which segfaults the moment ``N`` leaves the stack (measured on examinimd, where the
    canonicalized form died in the OpenMP body while the plain lowering ran).

    Only the buffer shrinks. The accesses stay private to the iteration that makes them -- each
    one writes the box and reads it back within its own iteration, which is what makes rewriting
    the subset to the origin an identity on the values.

    A map body nested into its own SDFG is the same case one level down: the buffer is a
    top-level transient of the nest and the box is named by symbols the nest receives, so it is
    shrunk when every access names that one box and nothing in the nest reassigns its symbols.
    """

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Descriptors | ppl.Modifies.Memlets | ppl.Modifies.Nodes))

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """Shrink every map-body-local transient that names one box.

        :param sdfg: SDFG to rewrite in place.
        :param pipeline_results: Unused; the pass resolves its own facts.
        :returns: The number of descriptors resized, or ``None`` when none were.
        """
        shrunk = 0
        code: dict[int, str] = {}
        for sd in sdfg.all_sdfgs_recursive():
            uses = uses_by_name(sd)
            for name, desc in list(sd.arrays.items()):
                if not self.is_candidate(sd, name, desc):
                    continue
                in_map = map_local_accesses(uses, name)
                if in_map is not None:
                    accesses = [(in_map[0], edge) for edge in in_map[1]]
                else:
                    accesses = nest_invariant_accesses(sd, uses, name)
                    if accesses is None:
                        continue
                on_device = desc.storage == dtypes.StorageType.GPU_Global
                if on_device and not device_level_accesses(sd, accesses):
                    continue
                shared = uniform_subset([edge for _, edge in accesses])
                if shared is None:
                    continue
                if in_map is None and not box_is_invariant(sd, shared):
                    continue
                size = tuple(shared.size())
                if len(size) != len(desc.shape) or not shrinks_the_buffer(desc, size):
                    continue
                if named_in_text(sd, name, code):
                    continue
                nested = bound_in_nests(accesses, shared, code)
                if nested is None:
                    continue
                desc.set_shape(size)
                if on_device:
                    # Thread-private by construction: every access sits under a kernel and names
                    # one iteration's box. Left in GPU_Global it would be one
                    # buffer shared by every thread.
                    desc.storage = dtypes.StorageType.Register(force=True)
                for inner, inner_name, _ in nested:
                    inner.arrays[inner_name].set_shape(size)
                    if on_device:
                        inner.arrays[inner_name].storage = desc.storage
                for edge in [edge for _, edge in accesses] + [edge for _, _, edges in nested for edge in edges]:
                    edge.data.subset = subsets.Range([(0, dim - 1, 1) for dim in size])
                    edge.data.volume = required(edge.data.subset).num_elements()
                shrunk += 1
        return shrunk or None

    def is_candidate(self, sdfg: SDFG, name: str, desc: data.Data) -> bool:
        """Whether ``name`` is a plain resizable transient array of ``sdfg``.

        :param sdfg: SDFG owning the descriptor.
        :param name: Descriptor name.
        :param desc: The descriptor.
        :returns: ``True`` when the descriptor is one this pass may resize.
        """
        if type(desc) is not data.Array or not desc.transient:
            return False
        if name in sdfg.constants_prop:
            return False
        if desc.lifetime != dtypes.AllocationLifetime.Scope:
            return False
        return desc.storage in RESIZABLE_STORAGE or desc.storage == dtypes.StorageType.GPU_Global
