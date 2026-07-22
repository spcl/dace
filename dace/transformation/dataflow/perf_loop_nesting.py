# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``PerfLoopNesting`` — duplicate a parent map around each inner child.

Single-state transformation applied at a top-level ``MapEntry``. The
base match requires:

- The ``MapEntry``'s body contains exactly one ``NestedSDFG``.
- That ``NestedSDFG`` has exactly one state.
- That state has K >= 2 top-level children -- either ``MapEntry`` nodes
  or ``Tasklet`` nodes outside any map scope.

Apply fissions the parent map into K copies, each wrapping a pruned
``NestedSDFG`` that retains exactly one child's subgraph. Top-level
``Tasklet`` children are first wrapped in a single-iteration trivial map
so every sibling is a ``MapEntry``; this preserves perfect-nesting shape
in the output.

Outer-state edges are re-routed: each duplicate receives only the inputs
its child reads and only the outputs its child writes. Arrays no longer
referenced inside a duplicate's pruned state are dropped from that
``NestedSDFG`` so validation's non-transient-as-connector requirement
holds.

Design — generalization to imperfectly-nested parent bodies
-----------------------------------------------------------

The base match required the parent ``MapEntry``'s body to be *exactly*
the single ``NestedSDFG`` plus the ``AccessNode`` s wired through the
parent ``MapEntry`` / ``MapExit``. Real Python-frontend output for an
imperfect nest (a statement *between* the outer ``dace.map`` and the
inner loop, e.g. ``for j: s = f(j); for i: x[i, j] = s``) instead emits
the parent body as an *intervening* chain of ``Tasklet`` / transient
``AccessNode`` nodes feeding a ``NestedSDFG`` input connector. The base
``apply`` deleted every such body node without replicating it, silently
dropping the intervening computation (or failing validation).

This pass generalizes the match to admit an intervening chain of the
shape ``parent MapEntry -> (Tasklet | transient AccessNode)+ -> NestedSDFG
input connector`` and *sinks* the whole chain into the ``NestedSDFG``'s
single inner state once, before fission. After sinking, the chain's final
``AccessNode`` is an ordinary inner producer that the existing K-way
fission already replicates per child (the very same shared-producer path
exercised by the velocity ``cfl_clipping`` pattern). No outer-state edge
surgery is therefore required, and the proven duplication machinery is
fully reused.

Soundness. A chain is admitted only when every chain node is a
side-effect-free, deterministic dataflow node (a ``Tasklet`` or a
*transient* ``AccessNode``), no carried memlet uses write-conflict
resolution (``wcr``), the chain reads no array data through the parent
``MapEntry`` (only the parent map's iteration symbols / free symbols /
constants -- already available inside the ``NestedSDFG`` via its
``symbol_mapping``), and the chain's result is observed only by the
``NestedSDFG``. Under these conditions, evaluating the chain once per
fissioned parent yields bit-identical results to evaluating it once and
fanning it out, because the duplicated parent map replays the identical
iteration space and every chain node is a pure function of the parent
map's symbols. Any body that violates the conditions is rejected, so
forcing perfect nesting can never change results. Bare two-level perfect
nests (no intervening nodes) keep their original fast path unchanged.
"""
import copy as _copy
from typing import Dict, List, Set, Tuple

from dace import memlet as mm, nodes, properties
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import utils as sdutil
from dace.transformation import transformation as xf

#: Node types admitted in a replicable intervening chain in an imperfectly
#: nested parent-map body (side-effect-free, deterministic dataflow).
REPLICABLE_INTERVENING_TYPES = (nodes.Tasklet, nodes.AccessNode)


@properties.make_properties
class PerfLoopNesting(xf.SingleStateTransformation):
    """Duplicate a parent map so each inner map gets its own wrapper."""

    parent_entry = xf.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.parent_entry)]

    def can_be_applied(self, graph: SDFGState, expr_index, sdfg: SDFG, permissive=False) -> bool:
        from dace.transformation.dataflow.map_fission import MapFission

        pe = self.parent_entry
        px = graph.exit_node(pe)
        body = [n for n in graph.nodes() if graph.entry_node(n) is pe]
        nsdfgs = [n for n in body if isinstance(n, nodes.NestedSDFG)]

        # Inlined same-state form (no NestedSDFG indirection): a parent map --
        # at any nesting depth -- directly encloses two or more inner maps.
        # Duplicating the parent per independent inner map while respecting
        # data dependencies is exactly ``MapFission`` (its no-NestedSDFG
        # expression); delegate to that proven, dependency-aware logic.
        # Applied repeatedly this cascades an arbitrarily deep nest, e.g.
        # ``m1[m2[m3; m4]]`` into ``m1[m2[m3]]`` and ``m1[m2[m4]]``.
        if len(nsdfgs) == 0:
            if len([n for n in body if isinstance(n, nodes.MapEntry)]) < 2:
                return False
            return MapFission.can_be_applied_to(sdfg, map_entry=pe)

        # NestedSDFG-wrapped path (original behavior): only at the top level.
        if graph.entry_node(pe) is not None:
            return False
        if len(nsdfgs) != 1:
            return False
        nsdfg = nsdfgs[0]
        inner = nsdfg.sdfg
        if len(list(inner.states())) != 1:
            return False

        # Every body node other than the single NestedSDFG and the AccessNodes
        # wired directly through the parent MapEntry/MapExit (the base
        # perfect-nest shape) must form a replicable intervening chain.
        if not _intervening_chain_is_replicable(graph, pe, px, nsdfg, body):
            return False

        inner_state = list(inner.states())[0]
        top_children = [
            n for n in inner_state.nodes()
            if inner_state.entry_node(n) is None and isinstance(n, (nodes.MapEntry, nodes.Tasklet))
        ]
        # Each child becomes its own parent-map copy, so the children must be
        # data-independent: a child map's output is not replicated into the
        # other copies, so the split is invalid if another child reads or
        # writes a container that a child map writes.
        return len(top_children) >= 2 and _children_data_independent(inner_state, top_children, nsdfg)

    def apply(self, graph: SDFGState, sdfg: SDFG):
        pe = self.parent_entry
        px = graph.exit_node(pe)
        body = [n for n in graph.nodes() if graph.entry_node(n) is pe]

        # Inlined same-state form: delegate to MapFission (see can_be_applied).
        if not any(isinstance(n, nodes.NestedSDFG) for n in body):
            from dace.transformation.dataflow.map_fission import MapFission
            MapFission.apply_to(sdfg, map_entry=pe, verify=False, save=False)
            return

        orig_nsdfg = next(n for n in body if isinstance(n, nodes.NestedSDFG))
        inner_state = list(orig_nsdfg.sdfg.states())[0]

        # Imperfect-nest generalization: sink any intervening chain into the
        # NestedSDFG's only state so the existing fission path replicates it.
        sunk = _sink_intervening_chain(graph, pe, orig_nsdfg, body)

        # Bare top-level Tasklets become siblings via a trivial map, except
        # sunk-chain producers, which are replicated per child by
        # ``_child_subgraph`` and must not turn into independent siblings.
        for n in list(inner_state.nodes()):
            if isinstance(n, nodes.Tasklet) and inner_state.entry_node(n) is None and n not in sunk:
                _wrap_tasklet_in_trivial_map(inner_state, n)

        child_entries = [
            n for n in inner_state.nodes() if isinstance(n, nodes.MapEntry) and inner_state.entry_node(n) is None
        ]
        outer_in_by, outer_out_by = _outer_plumbing(graph, pe, px, orig_nsdfg)

        for ch_entry in child_entries:
            keep = _child_subgraph(inner_state, ch_entry)
            _build_duplicate(graph, pe, px, orig_nsdfg, inner_state, keep, outer_in_by, outer_out_by)

        for ie in list(graph.in_edges(pe)):
            graph.remove_edge(ie)
        for oe in list(graph.out_edges(px)):
            graph.remove_edge(oe)
        body = [n for n in graph.nodes() if graph.entry_node(n) is pe]
        for n in body + [pe, px]:
            if n in graph.nodes():
                graph.remove_node(n)


def _direct_io_access_nodes(graph: SDFGState, pe: nodes.MapEntry, px: nodes.MapExit) -> Set[nodes.Node]:
    """``AccessNode`` s wired directly to the parent ``MapEntry`` outputs or
    ``MapExit`` inputs (the base perfect-nest plumbing).

    :param graph: The state containing the parent map scope.
    :param pe: The parent ``MapEntry``.
    :param px: The parent ``MapExit``.
    :returns: The set of directly-wired ``AccessNode`` s.
    """
    direct: Set[nodes.Node] = set()
    for e in graph.out_edges(pe):
        if isinstance(e.dst, nodes.AccessNode):
            direct.add(e.dst)
    for e in graph.in_edges(px):
        if isinstance(e.src, nodes.AccessNode):
            direct.add(e.src)
    return direct


def _child_data_io(state: SDFGState, child: nodes.Node) -> Tuple[Set[str], Set[str]]:
    """The data containers read and written across one top-level child's scope
    boundary (a child ``MapEntry`` and its body, or a bare ``Tasklet``).

    :param state: The inner state holding the child.
    :param child: A top-level ``MapEntry`` or ``Tasklet``.
    :returns: ``(reads, writes)`` data-container name sets.
    """
    if isinstance(child, nodes.MapEntry):
        cx = state.exit_node(child)
        scope = set(state.all_nodes_between(child, cx)) | {child, cx}
    else:
        scope = {child}
    reads: Set[str] = set()
    writes: Set[str] = set()
    for e in state.edges():
        if e.data is None or e.data.data is None:
            continue
        if e.dst in scope and e.src not in scope:
            reads.add(e.data.data)
        if e.src in scope and e.dst not in scope:
            writes.add(e.data.data)
    return reads, writes


def _children_data_independent(state: SDFGState, children: List[nodes.Node], nsdfg: nodes.NestedSDFG) -> bool:
    """Whether the children can be split into independent parent-map copies.

    Each child becomes its own parent-map copy wrapping a pruned NestedSDFG,
    wired only with the NestedSDFG's existing in/out connectors plus the
    replicated producer chain (tasklets / access nodes) feeding it. A child
    *map*'s output is not replicated. So the split is invalid only when an
    *internal* container -- one that is not already a NestedSDFG connector --
    is written by a child map and read or written by another child: it would
    have to be materialised across the two parent maps, which the duplication
    does not do. A shared container that is already a NestedSDFG connector
    stays wired (e.g. two maps doing read-modify-write on an in-out array), and
    an internal container written by a producer ``Tasklet`` is fine because
    that producer is replicated into each consuming copy.

    :param state: The inner state holding the children.
    :param children: The top-level ``MapEntry`` / ``Tasklet`` children.
    :param nsdfg: The NestedSDFG whose connectors stay wired across the split.
    :returns: ``True`` if the split preserves every cross-child dependency.
    """
    # A reader child can only be re-wired to a container that is a NestedSDFG
    # *input* connector (its copy reads it from the same outside source). A
    # container a child map merely writes out (an out-only connector) or an
    # internal transient cannot reach another child's copy.
    inputs = set(nsdfg.in_connectors)
    io = {id(c): _child_data_io(state, c) for c in children}
    for mc in children:
        if not isinstance(mc, nodes.MapEntry):
            continue
        internal_writes = io[id(mc)][1] - inputs
        for other in children:
            if other is mc:
                continue
            other_reads, other_writes = io[id(other)]
            if internal_writes & (other_reads | other_writes):
                return False
    return True


def _intervening_chain(graph: SDFGState, pe: nodes.MapEntry, px: nodes.MapExit, nsdfg: nodes.NestedSDFG,
                       body: List[nodes.Node]) -> List[nodes.Node]:
    """Return the body nodes that form the intervening chain (everything that
    is neither the ``NestedSDFG`` nor a directly-wired ``AccessNode``).

    :param graph: The state containing the parent map scope.
    :param pe: The parent ``MapEntry``.
    :param px: The parent ``MapExit``.
    :param nsdfg: The single ``NestedSDFG`` in the parent body.
    :param body: All nodes scoped directly under ``pe``.
    :returns: The intervening nodes (unordered, may be empty).
    """
    direct = _direct_io_access_nodes(graph, pe, px)
    # ``body`` (nodes whose enclosing scope is ``pe``) includes the parent
    # ``MapExit`` itself; it is the scope boundary, not intervening
    # computation, so exclude it -- otherwise a bare perfect nest (whose only
    # body content is the ``NestedSDFG``) gets a non-empty chain containing the
    # ``MapExit`` and is wrongly rejected as a non-replicable chain.
    return [n for n in body if n is not nsdfg and n is not px and n not in direct]


def _intervening_chain_is_replicable(graph: SDFGState, pe: nodes.MapEntry, px: nodes.MapExit, nsdfg: nodes.NestedSDFG,
                                     body: List[nodes.Node]) -> bool:
    """Validate that the intervening chain may be soundly sunk into the
    ``NestedSDFG`` and replicated per fissioned duplicate.

    See the module docstring for the full soundness argument.

    :param graph: The state containing the parent map scope.
    :param pe: The parent ``MapEntry``.
    :param px: The parent ``MapExit``.
    :param nsdfg: The single ``NestedSDFG`` in the parent body.
    :param body: All nodes scoped directly under ``pe``.
    :returns: ``True`` if the chain (possibly empty) is replicable.
    """
    chain = _intervening_chain(graph, pe, px, nsdfg, body)
    if not chain:
        return True  # Base perfect-nest fast path.

    # Conservative supported shape (the canonical frontend imperfect-nest
    # emission): for every fed connector, exactly one producing ``Tasklet``
    # whose only inputs are parent-map symbols, optionally followed by a
    # single transient ``AccessNode``, then the ``NestedSDFG`` input. This
    # keeps the sink a pure data-name remap with no intermediate transients.
    if len(chain) > 2:
        return False
    chain_set = set(chain)
    tasklets = [n for n in chain if isinstance(n, nodes.Tasklet)]
    accesses = [n for n in chain if isinstance(n, nodes.AccessNode)]
    if len(tasklets) != 1 or len(accesses) != len(chain) - 1:
        return False
    t = tasklets[0]

    for n in chain:
        if not isinstance(n, REPLICABLE_INTERVENING_TYPES):
            return False
        if isinstance(n, nodes.AccessNode):
            desc = graph.sdfg.arrays.get(n.data)
            if desc is None or not desc.transient:
                return False
        for ie in graph.in_edges(n):
            if ie.data is not None and ie.data.wcr is not None:
                return False
            # Chain inputs may only be parent-map symbols (an empty dependency
            # edge with no data and no destination connector) or another chain
            # node; no array data may be read through the parent MapEntry.
            if ie.src is pe:
                if ie.dst_conn is not None or (ie.data is not None and ie.data.data is not None):
                    return False
            elif ie.src not in chain_set:
                return False
        for oe in graph.out_edges(n):
            if oe.data is not None and oe.data.wcr is not None:
                return False
            # The chain result must be observed only by the NestedSDFG or by
            # another chain node -- never by the parent MapExit or elsewhere.
            if not (oe.dst is nsdfg or oe.dst in chain_set):
                return False

    # The Tasklet must produce exactly the value(s) consumed by the
    # NestedSDFG, and the chain must terminate at NestedSDFG *input*
    # connectors only.
    feeds_nsdfg = False
    for n in chain:
        for oe in graph.out_edges(n):
            if oe.dst is nsdfg:
                if oe.dst_conn is None:
                    return False
                feeds_nsdfg = True
    return feeds_nsdfg and t is not None


def _sink_intervening_chain(graph: SDFGState, pe: nodes.MapEntry, nsdfg: nodes.NestedSDFG,
                            body: List[nodes.Node]) -> Set[nodes.Node]:
    """Move the intervening chain from the parent scope into the
    ``NestedSDFG``'s single inner state.

    After this, the chain's terminal transient ``AccessNode`` (the value the
    inner maps consume through what used to be a ``NestedSDFG`` input
    connector) is produced *inside* the inner state, so the subsequent K-way
    fission replicates it per child via the existing shared-producer path.
    The vacated ``NestedSDFG`` input connectors are removed and the inner
    arrays they backed are marked transient.

    :param graph: The state containing the parent map scope.
    :param pe: The parent ``MapEntry``.
    :param nsdfg: The single ``NestedSDFG`` in the parent body.
    :param body: All nodes scoped directly under ``pe``.
    :returns: The set of clone nodes inserted into the inner state (empty if
        there was no intervening chain).
    """
    px = graph.exit_node(pe)
    chain = _intervening_chain(graph, pe, px, nsdfg, body)
    if not chain:
        return set()
    inner = nsdfg.sdfg
    inner_state = list(inner.states())[0]

    # Validated supported shape: a single producing Tasklet, then (optionally)
    # one transient AccessNode, then the NestedSDFG input connector(s). Clone
    # only the Tasklet; the terminal AccessNode is *absorbed* -- its inner role
    # is already served by the readers of the vacated connector's array.
    t = next(n for n in chain if isinstance(n, nodes.Tasklet))
    inner_t = _copy.deepcopy(t)
    inner_state.add_node(inner_t)

    # Map each fed NestedSDFG-input connector to the Tasklet out-connector
    # whose value reaches it (directly or via the absorbed AccessNode).
    chain_set = set(chain)
    conn_to_srcconn: Dict[str, str] = {}
    for e in graph.out_edges(t):
        if e.dst is nsdfg and e.dst_conn is not None:
            conn_to_srcconn[e.dst_conn] = e.src_conn
        elif e.dst in chain_set and isinstance(e.dst, nodes.AccessNode):
            for ee in graph.out_edges(e.dst):
                if ee.dst is nsdfg and ee.dst_conn is not None:
                    conn_to_srcconn[ee.dst_conn] = e.src_conn

    # Wire the cloned Tasklet to every inner reader of each vacated connector's
    # array (one reader per consuming inner map); each reader must gain the
    # producer edge or it would read uninitialized data. The memlet references
    # the inner array (the connector name), not the outer transient.
    for conn, src_conn in conn_to_srcconn.items():
        readers = [
            a for a in inner_state.nodes()
            if isinstance(a, nodes.AccessNode) and a.data == conn and inner_state.in_degree(a) == 0
        ]
        if not readers:
            readers = [inner_state.add_access(conn)]
        for consumer in readers:
            inner_state.add_edge(inner_t, src_conn, consumer, None, mm.Memlet.from_array(conn, inner.arrays[conn]))

    # Drop the now-internally-produced input connectors and their outer edges,
    # and make the inner arrays they backed transient (no longer fed from
    # outside the NestedSDFG).
    for conn in conn_to_srcconn:
        for ie in list(graph.in_edges(nsdfg)):
            if ie.dst_conn == conn:
                graph.remove_edge(ie)
        if conn in nsdfg.in_connectors:
            nsdfg.remove_in_connector(conn)
        desc = inner.arrays.get(conn)
        if desc is not None and not desc.transient and conn not in nsdfg.out_connectors:
            desc.transient = True

    # Remove the now-sunk chain nodes (and their dangling parent-scope edges).
    for n in chain:
        for e in list(graph.in_edges(n)) + list(graph.out_edges(n)):
            graph.remove_edge(e)
    for n in chain:
        if n in graph.nodes():
            graph.remove_node(n)

    return {inner_t}


def _copy_state_contents(src: SDFGState, dst: SDFGState) -> Dict[nodes.Node, nodes.Node]:
    """Deep-copy all nodes and edges from ``src`` into ``dst``."""
    nmap: Dict[nodes.Node, nodes.Node] = {}
    for n in src.nodes():
        new_n = _copy.deepcopy(n)
        dst.add_node(new_n)
        nmap[n] = new_n
    for e in src.edges():
        dst.add_edge(nmap[e.src], e.src_conn, nmap[e.dst], e.dst_conn, _copy.deepcopy(e.data))
    return nmap


def _child_subgraph(inner_state: SDFGState, ch_entry: nodes.MapEntry) -> Set[nodes.Node]:
    """Subgraph belonging to one child map: its MapEntry, MapExit, every
    node scoped under the MapEntry, the top-level AccessNodes directly
    connected to the entry's in-edges or exit's out-edges, and -- for an
    imperfect nest -- the transitive top-level producer chain of any such
    input AccessNode (the sunk intervening chain)."""
    ch_exit = inner_state.exit_node(ch_entry)
    keep: Set[nodes.Node] = {ch_entry, ch_exit}
    keep.update(n for n in inner_state.nodes() if inner_state.entry_node(n) is ch_entry)
    frontier: List[nodes.Node] = []
    for e in inner_state.in_edges(ch_entry):
        if isinstance(e.src, nodes.AccessNode):
            keep.add(e.src)
            frontier.append(e.src)
    for e in inner_state.out_edges(ch_exit):
        if isinstance(e.dst, nodes.AccessNode):
            keep.add(e.dst)
    # Pull in the producer chain feeding kept top-level input AccessNodes so
    # the sunk intervening computation travels into every duplicate.
    while frontier:
        node = frontier.pop()
        for ie in inner_state.in_edges(node):
            src = ie.src
            if src in keep or inner_state.entry_node(src) is not None:
                continue
            if isinstance(src, (nodes.Tasklet, nodes.AccessNode)):
                keep.add(src)
                frontier.append(src)
    return keep


def _outer_plumbing(graph: SDFGState, pe: nodes.MapEntry, px: nodes.MapExit, orig_nsdfg: nodes.NestedSDFG):
    in_by: Dict[str, list] = {}
    for e in graph.in_edges(orig_nsdfg):
        n_conn = e.dst_conn
        me_in_conn = e.src_conn.replace("OUT_", "IN_") if e.src_conn else None
        outer_edges = [ie for ie in graph.in_edges(pe) if ie.dst_conn == me_in_conn]
        in_by.setdefault(n_conn, []).extend((ie.src, ie.data, e.data) for ie in outer_edges)
    out_by: Dict[str, list] = {}
    for e in graph.out_edges(orig_nsdfg):
        n_conn = e.src_conn
        mx_out_conn = e.dst_conn.replace("IN_", "OUT_") if e.dst_conn else None
        outer_edges = [oe for oe in graph.out_edges(px) if oe.src_conn == mx_out_conn]
        out_by.setdefault(n_conn, []).extend((oe.dst, oe.data, e.data) for oe in outer_edges)
    return in_by, out_by


def _wrap_tasklet_in_trivial_map(state: SDFGState, t: nodes.Tasklet):
    """Enclose a bare top-level ``Tasklet`` in a single-iteration map."""
    in_edges = list(state.in_edges(t))
    out_edges = list(state.out_edges(t))

    me, mx = state.add_map(f"{t.label}_wrap", {"_p": "0:1"})

    for idx, e in enumerate(in_edges):
        key = e.dst_conn if e.dst_conn else f"in_{idx}"
        in_c = f"IN_{key}"
        out_c = f"OUT_{key}"
        me.add_in_connector(in_c)
        me.add_out_connector(out_c)
        state.remove_edge(e)
        state.add_edge(e.src, e.src_conn, me, in_c, _copy.deepcopy(e.data))
        state.add_edge(me, out_c, t, e.dst_conn, _copy.deepcopy(e.data))

    for idx, e in enumerate(out_edges):
        key = e.src_conn if e.src_conn else f"out_{idx}"
        in_c = f"IN_{key}"
        out_c = f"OUT_{key}"
        mx.add_in_connector(in_c)
        mx.add_out_connector(out_c)
        state.remove_edge(e)
        state.add_edge(t, e.src_conn, mx, in_c, _copy.deepcopy(e.data))
        state.add_edge(mx, out_c, e.dst, e.dst_conn, _copy.deepcopy(e.data))

    if not in_edges:
        state.add_nedge(me, t, mm.Memlet())
    if not out_edges:
        state.add_nedge(t, mx, mm.Memlet())


def _build_duplicate(graph: SDFGState, pe: nodes.MapEntry, px: nodes.MapExit, orig_nsdfg: nodes.NestedSDFG,
                     orig_inner_state: SDFGState, keep: Set[nodes.Node], outer_in_by: Dict[str, list],
                     outer_out_by: Dict[str, list]):
    new_inner_sdfg = SDFG(orig_nsdfg.sdfg.name + "_pn")
    for name, desc in orig_nsdfg.sdfg.arrays.items():
        new_inner_sdfg.add_datadesc(name, _copy.deepcopy(desc))
    for name, stype in orig_nsdfg.sdfg.symbols.items():
        if name not in new_inner_sdfg.symbols:
            new_inner_sdfg.add_symbol(name, stype)

    new_inner_state = new_inner_sdfg.add_state("body", is_start_block=True)
    nmap = _copy_state_contents(orig_inner_state, new_inner_state)

    to_remove = [nmap[n] for n in orig_inner_state.nodes() if n not in keep]
    for n in to_remove:
        for e in list(new_inner_state.in_edges(n)) + list(new_inner_state.out_edges(n)):
            new_inner_state.remove_edge(e)
    for n in to_remove:
        if n in new_inner_state.nodes():
            new_inner_state.remove_node(n)

    used_in: Set[str] = set()
    used_out: Set[str] = set()
    for n in new_inner_state.nodes():
        if not isinstance(n, nodes.AccessNode):
            continue
        if n.data in orig_nsdfg.in_connectors and new_inner_state.out_degree(n) > 0:
            used_in.add(n.data)
        if n.data in orig_nsdfg.out_connectors and new_inner_state.in_degree(n) > 0:
            used_out.add(n.data)

    referenced: Set[str] = set()
    for n in new_inner_state.nodes():
        if isinstance(n, nodes.AccessNode):
            referenced.add(n.data)
    for e in new_inner_state.edges():
        if e.data is not None and e.data.data:
            referenced.add(e.data.data)
    for name in list(new_inner_sdfg.arrays):
        if name not in referenced:
            try:
                new_inner_sdfg.remove_data(name, validate=False)
            except Exception:
                new_inner_sdfg.arrays.pop(name, None)

    new_map = _copy.deepcopy(pe.map)
    new_pe = nodes.MapEntry(new_map)
    new_px = nodes.MapExit(new_map)
    graph.add_node(new_pe)
    graph.add_node(new_px)

    new_nsdfg = graph.add_nested_sdfg(new_inner_sdfg,
                                      used_in,
                                      used_out,
                                      symbol_mapping=_copy.copy(orig_nsdfg.symbol_mapping))

    for conn in used_in:
        new_pe.add_in_connector("IN_" + conn)
        new_pe.add_out_connector("OUT_" + conn)
        for (outer_access, outer_memlet, inner_memlet) in outer_in_by.get(conn, []):
            graph.add_edge(outer_access, None, new_pe, "IN_" + conn, _copy.deepcopy(outer_memlet))
            graph.add_edge(new_pe, "OUT_" + conn, new_nsdfg, conn, _copy.deepcopy(inner_memlet))
    for conn in used_out:
        new_px.add_in_connector("IN_" + conn)
        new_px.add_out_connector("OUT_" + conn)
        for (outer_access, outer_memlet, inner_memlet) in outer_out_by.get(conn, []):
            graph.add_edge(new_nsdfg, conn, new_px, "IN_" + conn, _copy.deepcopy(inner_memlet))
            graph.add_edge(new_px, "OUT_" + conn, outer_access, None, _copy.deepcopy(outer_memlet))

    if not used_in:
        graph.add_nedge(new_pe, new_nsdfg, mm.Memlet())
    if not used_out:
        graph.add_nedge(new_nsdfg, new_px, mm.Memlet())
