# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Rematerialize a map temporary that is a pure elementwise function of values the consumer already holds.

Vertical map fusion routinely pulls a sub-expression of a consumer map UP into the producer map, because
the value it is built from lives in a register there. When a THIRD map still consumes that
sub-expression, fusion cannot delete it and it is forced across the map boundary as a full transient
array ``T``::

    map1[i]:  X[x(i)] = <expression>        ; T[t(i)] = f(<the register that expression landed in>)
    map2[j]:  ... = g(T[s(j)], X[..], ..)

``T`` costs a full array store plus a full array load per execution -- on GPU an extra global-memory
round trip across a kernel boundary -- to save recomputing ``f``. On every machine we target, memory
balance is around 0.1 bytes/flop, so a stored-and-reloaded 8-byte double buys roughly a hundred flops
of recomputation. Recomputing ``f`` inside map2 is therefore the cheaper side of the trade, PROVIDED
the recomputation needs no data map2 does not already read: this pass only fires when every input of
the recomputed chain sits on an EXISTING read edge of the consumer map, so the rewrite adds zero
memory traffic and removes a whole array.

The rewrite is literal rematerialization: clone the producing tasklet subgraph into the consumer's
scope, wire its inputs to the consumer's existing reads at the index the inversion demands, feed the
consumer from the clone, and delete ``T``.

Nothing here is tied to a kernel or a constant. What generalizes is the pair of facts fusion creates:
a transient that is a pure elementwise function of one produced value, and a consumer that already
reads the container that value was also stored to.
"""
import copy
from typing import Any, Dict, List, Optional, Tuple

from ordered_set import OrderedSet

from dace import SDFG, data, dtypes, properties, subsets, symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.propagation import propagate_memlets_map_scope
from dace.sdfg.state import ConditionalBlock, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl, transformation

# A producer-side value the clone needs: container, point to read it at, and the access node that
# stores it there (None when the producer simply reads the container through its map entry).
SourceRead = Tuple[str, List[symbolic.SymbolicType], Optional[nodes.AccessNode]]


def point_of(sub: Optional[subsets.Subset]) -> Optional[List[symbolic.SymbolicType]]:
    """The single element a subset addresses, or None if it is not a unit-stride single element.

    :param sub: The subset to inspect.
    :returns: One index expression per dimension, or None.
    """
    if not isinstance(sub, subsets.Range):
        return None
    pt = []
    for beg, end, step in sub.ranges:
        if step != 1 or symbolic.simplify(end - beg) != 0:
            return None
        pt.append(beg)
    return pt


def same_point(lhs: List[symbolic.SymbolicType], rhs: List[symbolic.SymbolicType]) -> bool:
    """Whether two point subsets address the same element."""
    return len(lhs) == len(rhs) and all(symbolic.simplify(a - b) == 0 for a, b in zip(lhs, rhs))


def point_memlet(container: str, point: List[symbolic.SymbolicType]) -> Memlet:
    """A fresh single-element memlet. Never share a memlet or a subset between two edges."""
    return Memlet(data=container, subset=subsets.Range([(idx, idx, 1) for idx in point]))


def symbolic_references(sdfg: SDFG) -> OrderedSet:
    """Every name referenced by control flow (interstate edges, loop headers, branch conditions).

    A container named here is read outside dataflow, so its access nodes do not tell the whole story.

    :param sdfg: The SDFG to scan (this level only; a nested SDFG uses its own names).
    :returns: The referenced names.
    """
    names: OrderedSet = OrderedSet()
    for edge in sdfg.all_interstate_edges():
        names |= OrderedSet(edge.data.free_symbols)
    for block in sdfg.all_control_flow_blocks():
        if isinstance(block, LoopRegion):
            for code in (block.init_statement, block.loop_condition, block.update_statement):
                if code is not None:
                    names |= OrderedSet(code.get_free_symbols())
        elif isinstance(block, ConditionalBlock):
            for cond, _ in block.branches:
                if cond is not None:
                    names |= OrderedSet(cond.get_free_symbols())
    return names


@properties.make_properties
@transformation.explicit_cf_compatible
class RematerializeDerivedTemporaries(ppl.Pass):
    """Delete a map temporary by recomputing it in its consumer from reads the consumer already has."""

    max_recompute_tasklets = properties.Property(
        dtype=int,
        default=8,
        desc='Refuse when the total number of cloned tasklets (chain length times consumer reads) exceeds this. '
        'The saving is bytes and the cost is flops; at the ~0.1 bytes/flop machine balance of every CPU and '
        'GPU we target a stored-and-reloaded double is worth ~100 flops, so this bound is deliberately '
        'generous and only excludes runaway chains.')

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Tasklets | ppl.Modifies.Memlets | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # Fresh instances appear only when maps fuse, which reshapes access nodes and memlets.
        return bool(modified & (ppl.Modifies.AccessNodes | ppl.Modifies.Memlets))

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        removed = 0
        for nested in sdfg.all_sdfgs_recursive():
            controlled = symbolic_references(nested)
            arglist = OrderedSet(nested.arglist().keys())
            for name in list(nested.arrays.keys()):
                if name in controlled or name in arglist:
                    continue
                if self.rematerialize(nested, name):
                    removed += 1
        return removed or None

    # ------------------------------------------------------------------ matching

    def temporary_node(self, sdfg: SDFG, name: str) -> Optional[Tuple[SDFGState, nodes.AccessNode]]:
        """The unique access node of ``name``, if it has exactly one and is a real in-memory array.

        :param sdfg: The SDFG owning ``name``.
        :param name: Candidate container.
        :returns: ``(state, node)`` or None.
        """
        desc = sdfg.arrays[name]
        if not desc.transient or not isinstance(desc, data.Array) or isinstance(desc, (data.View, data.Reference)):
            return None
        # A register or a single element is not worth rematerializing; the win is deleting an array.
        if desc.storage == dtypes.StorageType.Register or desc.total_size == 1:
            return None
        found = None
        for state in sdfg.states():
            for node in state.data_nodes():
                if node.data != name:
                    continue
                if found is not None:
                    return None
                found = (state, node)
        return found

    def match(self, sdfg: SDFG, name: str) -> Optional[Dict[str, Any]]:
        """Recognize the rematerialization shape around temporary ``name``.

        :param sdfg: The SDFG owning ``name``.
        :param name: Candidate container.
        :returns: A plan, or None when any condition refuses.
        """
        located = self.temporary_node(sdfg, name)
        if located is None:
            return None
        state, tnode = located
        if state.in_degree(tnode) != 1 or state.out_degree(tnode) < 1:
            return None
        # An empty memlet on the temporary is an ORDERING edge; deleting the temporary would drop it.
        for edge in state.all_edges(tnode):
            if edge.data.is_empty():
                return None

        outer_write = state.in_edges(tnode)[0]
        exit1 = outer_write.src
        if not isinstance(exit1, nodes.MapExit) or outer_write.data.wcr is not None:
            return None
        entry1 = state.entry_node(exit1)

        produced = [e for e in state.in_edges(exit1) if e.data.data == name]
        if len(produced) != 1:
            return None
        write = produced[0]
        if write.data.wcr is not None or write.data.dynamic or not str(write.dst_conn).startswith('IN_'):
            return None
        wpoint = point_of(write.data.dst_subset if write.data.dst_subset is not None else write.data.subset)
        if wpoint is None:
            return None

        inverse = self.invert(entry1.map.params, wpoint)
        if inverse is None:
            return None

        sliced = self.slice_back(sdfg, state, entry1, write)
        if sliced is None:
            return None
        tasklets, sources = sliced

        reads: List[Dict[str, Any]] = []
        for outer_read in state.out_edges(tnode):
            entry2 = outer_read.dst
            if not isinstance(entry2, nodes.MapEntry) or entry2 is entry1:
                return None
            if not str(outer_read.dst_conn).startswith('IN_'):
                return None
            # Every element the consumer spans must have been written by this producer, else the value
            # rematerialization reproduces was never in the temporary to begin with.
            if not outer_write.data.subset.covers(outer_read.data.subset):
                return None
            inner = [e for e in state.out_edges(entry2) if e.data.data == name]
            if not inner:
                return None
            for edge in inner:
                plan = self.plan_read(sdfg, state, entry2, edge, inverse, sources)
                if plan is None:
                    return None
                reads.append(plan)

        distinct = len(OrderedSet(r['key'] for r in reads))
        if len(tasklets) * distinct > self.max_recompute_tasklets:
            return None

        return {'state': state, 'tnode': tnode, 'exit1': exit1, 'write': write, 'tasklets': tasklets, 'reads': reads}

    def invert(self, params: List[str], wpoint: List[symbolic.SymbolicType]) -> Optional[Dict[str, Any]]:
        """Invert the producer's write index ``t(i)``, requiring a per-dimension shifted bijection.

        Each dimension must be ``param + const`` and every map parameter must appear exactly once.
        Anything else -- a scaled index, two parameters in one dimension, a parameter that never appears
        (several iterations writing one element) -- leaves the value at a temporary element ambiguous or
        the index map non-invertible.

        :param params: The producer map's parameters.
        :param wpoint: The point the producer writes the temporary at.
        :returns: ``{'params': per-dim parameter, 'shift': per-dim constant}`` or None.
        """
        if len(wpoint) != len(params):
            return None
        pset = OrderedSet(params)
        used: OrderedSet = OrderedSet()
        dimparams: List[str] = []
        shifts: List[symbolic.SymbolicType] = []
        for expr in wpoint:
            here = [p for p in params if p in symbolic.symlist(expr)]
            if len(here) != 1 or here[0] in used:
                return None
            used.add(here[0])
            shift = symbolic.simplify(expr - symbolic.pystr_to_symbolic(here[0]))
            if OrderedSet(symbolic.symlist(shift).keys()) & pset:
                return None
            dimparams.append(here[0])
            shifts.append(shift)
        if len(used) != len(pset):
            return None
        return {'params': dimparams, 'shift': shifts}

    def slice_back(self, sdfg: SDFG, state: SDFGState, entry1: nodes.MapEntry,
                   write) -> Optional[Tuple[List[nodes.Tasklet], Dict[Any, SourceRead]]]:
        """Backward slice of the temporary's write, cut at values that are also stored to a container.

        Walks back from whatever feeds the temporary. A tasklet is cloned. A scope-local register access
        node is CUT when the producer also stores its value to a container -- that store is exactly what
        the consumer can read back -- and otherwise the walk continues through its writer. A read straight
        off the producer's map entry becomes a source requirement on that container.

        :returns: ``(cloned tasklets, source requirement per boundary edge)`` or None when the chain is
            impure, not elementwise, or reaches something that cannot be re-read.
        """
        tasklets: List[nodes.Tasklet] = []
        sources: Dict[Any, SourceRead] = {}
        queue = [write]
        seen: OrderedSet = OrderedSet()
        while queue:
            edge = queue.pop()
            if edge.data.is_empty() or edge.data.wcr is not None or edge.data.dynamic:
                return None
            node = edge.src
            if isinstance(node, nodes.AccessNode):
                cut = self.cut_read(sdfg, state, node)
                if cut is not None:
                    sources[edge] = cut
                    continue
                # Not container-backed: keep slicing through its producer, if it is a private register.
                desc = sdfg.arrays[node.data]
                if state.entry_node(node) is not entry1 or state.in_degree(node) != 1:
                    return None
                if not desc.transient or desc.total_size != 1:
                    return None
                inedge = state.in_edges(node)[0]
                if inedge not in seen:
                    seen.add(inedge)
                    queue.append(inedge)
                continue
            if isinstance(node, nodes.MapEntry):
                if node is not entry1 or edge.data.data is None:
                    return None
                pt = point_of(edge.data.subset)
                if pt is None:
                    return None
                sources[edge] = (edge.data.data, pt, None)
                continue
            if not isinstance(node, nodes.Tasklet):
                # NestedSDFG / LibraryNode: purity is not a local question and the body may hold state.
                return None
            if node.code.language != dtypes.Language.Python or node.has_side_effects(sdfg):
                return None
            if node in tasklets:
                continue
            tasklets.append(node)
            for inedge in state.in_edges(node):
                if point_of(inedge.data.subset) is None:
                    return None
                queue.append(inedge)
        return tasklets, sources

    def copy_class(self, sdfg: SDFG, state: SDFGState, node: nodes.AccessNode) -> List[nodes.AccessNode]:
        """Every access node holding the same value as ``node``, following plain single-element copies.

        Fusion routinely renames a value through a copy (``__map_fusion_B_0 = __map_fusion_B``) and stores
        only the copy, so the store is one hop away from the register the temporary is derived from. A
        copy edge preserves the value in both directions, provided each member has a single writer.

        :returns: The value-equal access nodes, ``node`` first.
        """
        members = [node]
        queue = [node]
        while queue:
            current = queue.pop()
            if state.in_degree(current) != 1:
                continue
            neighbours = [(e, e.dst) for e in state.out_edges(current) if isinstance(e.dst, nodes.AccessNode)]
            neighbours += [(e, e.src) for e in state.in_edges(current) if isinstance(e.src, nodes.AccessNode)]
            for edge, other in neighbours:
                if other in members or edge.data.is_empty() or edge.data.wcr is not None or edge.data.dynamic:
                    continue
                if edge.data.volume != 1 or state.in_degree(other) != 1:
                    continue
                if sdfg.arrays[other.data].dtype != sdfg.arrays[current.data].dtype:
                    continue
                members.append(other)
                queue.append(other)
        return members

    def cut_read(self, sdfg: SDFG, state: SDFGState, node: nodes.AccessNode) -> Optional[SourceRead]:
        """Whether this value is also stored to a container, and where the consumer can re-read it.

        :returns: ``(container, point, storing access node)`` to re-read the value at, or None.
        """
        desc = sdfg.arrays[node.data]
        for member in self.copy_class(sdfg, state, node):
            for edge in state.out_edges(member):
                if edge.data.is_empty() or not isinstance(edge.dst, nodes.MapExit):
                    continue
                if edge.data.data is None or edge.data.data == member.data:
                    continue
                if edge.data.wcr is not None or edge.data.dynamic:
                    continue
                # Store-then-reload returns the same bits only when no conversion happens on the way.
                if sdfg.arrays[edge.data.data].dtype != desc.dtype:
                    continue
                pt = point_of(edge.data.dst_subset if edge.data.dst_subset is not None else edge.data.subset)
                if pt is None:
                    continue
                return (edge.data.data, pt, member)
        return None

    def plan_read(self, sdfg: SDFG, state: SDFGState, entry2: nodes.MapEntry, edge, inverse: Dict[str, Any],
                  sources: Dict[Any, SourceRead]) -> Optional[Dict[str, Any]]:
        """Check one consumer read of the temporary and record how to feed it instead.

        :returns: ``{'edge', 'entry', 'wiring', 'key'}`` where wiring maps each boundary edge of the
            producer slice to the consumer map-entry out-connector and the point to read it at, or None.
        """
        if edge.data.wcr is not None or edge.data.dynamic:
            return None
        rpoint = point_of(edge.data.subset)
        if rpoint is None:
            return None
        subst = {
            param: symbolic.simplify(coord - shift)
            for param, shift, coord in zip(inverse['params'], inverse['shift'], rpoint)
        }
        available = self.consumer_reads(state, entry2)
        written = self.consumer_writes(state, entry2)
        wiring: Dict[Any, Tuple[str, List[symbolic.SymbolicType], str]] = {}
        for boundary, (container, pt, origin) in sources.items():
            # A container the consumer also writes gives no guarantee about which value the read sees.
            if container in written:
                return None
            need = [symbolic.simplify(symbolic.pystr_to_symbolic(coord).subs(subst)) for coord in pt]
            match = None
            for conn, existing, srcnode in available.get(container, []):
                if same_point(existing, need):
                    match = (conn, srcnode)
                    break
            if match is None:
                return None
            # The consumer must reach the same access node the producer's value did, or the two read
            # different versions of the container.
            if not self.same_version(state, container, match[1], boundary, origin):
                return None
            wiring[boundary] = (match[0], need, container)
        key = tuple(str(v) for v in subst.values())
        return {'edge': edge, 'entry': entry2, 'wiring': wiring, 'key': (id(entry2), key)}

    def consumer_reads(self, state: SDFGState, entry2: nodes.MapEntry) -> Dict[str, List[Any]]:
        """Every point read already crossing the consumer map entry, keyed by container."""
        out: Dict[str, List[Any]] = {}
        for edge in state.out_edges(entry2):
            if edge.data.is_empty() or edge.data.data is None or edge.data.wcr is not None or edge.data.dynamic:
                continue
            pt = point_of(edge.data.subset)
            if pt is None:
                continue
            src = None
            for outer in state.in_edges(entry2):
                if outer.data.data == edge.data.data and isinstance(outer.src, nodes.AccessNode):
                    src = outer.src
            out.setdefault(edge.data.data, []).append((edge.src_conn, pt, src))
        return out

    def consumer_writes(self, state: SDFGState, entry2: nodes.MapEntry) -> OrderedSet:
        """Containers written anywhere inside the consumer map's scope."""
        written: OrderedSet = OrderedSet()
        scope = state.scope_subgraph(entry2, include_entry=False, include_exit=True)
        for edge in scope.edges():
            if edge.data.is_empty() or edge.data.data is None:
                continue
            if isinstance(edge.dst, (nodes.AccessNode, nodes.MapExit)):
                written.add(edge.data.data)
        return written

    def same_version(self, state: SDFGState, container: str, consumer_src: Optional[nodes.AccessNode], boundary,
                     origin: Optional[nodes.AccessNode]) -> bool:
        """Whether the consumer reads the very access node the producer's value lives in.

        For a cut register the producer STORES into an access node through its map exit; for a map-entry
        read the producer READS one. Either way the consumer has to reach the same node, or the two see
        different versions of the container.
        """
        if consumer_src is None:
            return False
        producer_side = None
        if origin is not None:
            for edge in state.out_edges(origin):
                if edge.data.data != container or not isinstance(edge.dst, nodes.MapExit):
                    continue
                for outer in state.out_edges(edge.dst):
                    if outer.data.data == container and isinstance(outer.dst, nodes.AccessNode):
                        producer_side = outer.dst
        else:
            for outer in state.in_edges(boundary.src):
                if outer.data.data == container and isinstance(outer.src, nodes.AccessNode):
                    producer_side = outer.src
        return producer_side is consumer_src

    # ------------------------------------------------------------------- rewrite

    def rematerialize(self, sdfg: SDFG, name: str) -> bool:
        """Apply the rewrite for ``name`` if it matches.

        :returns: Whether the temporary was removed.
        """
        plan = self.match(sdfg, name)
        if plan is None:
            return False
        state: SDFGState = plan['state']
        touched: OrderedSet = OrderedSet()
        built: Dict[Any, nodes.AccessNode] = {}
        for read in plan['reads']:
            edge = read['edge']
            dst, dst_conn = edge.dst, edge.dst_conn
            state.remove_edge(edge)
            if read['key'] not in built:
                built[read['key']] = self.clone_into(sdfg, state, plan, read)
            sink = built[read['key']]
            state.add_edge(sink, None, dst, dst_conn, Memlet(data=sink.data, subset='0'))
            touched.add(read['entry'])

        for outer_read in list(state.out_edges(plan['tnode'])):
            entry2, conn = outer_read.dst, outer_read.dst_conn
            state.remove_edge(outer_read)
            entry2.remove_in_connector(conn)
            entry2.remove_out_connector(conn.replace('IN_', 'OUT_', 1))
        state.remove_edge(state.in_edges(plan['tnode'])[0])
        state.remove_node(plan['tnode'])
        write, exit1 = plan['write'], plan['exit1']
        state.remove_edge(write)
        exit1.remove_in_connector(write.dst_conn)
        exit1.remove_out_connector(write.dst_conn.replace('IN_', 'OUT_', 1))
        self.prune_dead(sdfg, state, plan['tasklets'])
        sdfg.remove_data(name, validate=False)

        for entry2 in touched:
            propagate_memlets_map_scope(sdfg, state, entry2)
        return True

    def clone_into(self, sdfg: SDFG, state: SDFGState, plan: Dict[str, Any], read: Dict[str, Any]) -> nodes.AccessNode:
        """Clone the producer slice into the consumer scope.

        :returns: The access node holding the rematerialized value.
        """
        entry2, wiring = read['entry'], read['wiring']
        sink_name, _ = sdfg.add_scalar('remat_' + plan['tnode'].data,
                                       sdfg.arrays[plan['tnode'].data].dtype,
                                       storage=dtypes.StorageType.Register,
                                       transient=True,
                                       find_new_name=True)
        sink = state.add_access(sink_name)

        write = plan['write']
        if not plan['tasklets']:
            # Degenerate chain: the temporary was a plain copy of the value, so the consumer's existing
            # read of the container IS the value. Route it straight through a private register.
            conn, point, container = wiring[write]
            state.add_edge(entry2, conn, sink, None, point_memlet(container, point))
            return sink

        clones: Dict[nodes.Tasklet, nodes.Tasklet] = {}
        for tasklet in plan['tasklets']:
            clone = copy.deepcopy(tasklet)
            state.add_node(clone)
            clones[tasklet] = clone
        # Inputs first, so every intermediate register exists before the writes that fill it. Every
        # value inside the slice passes through an access node -- a code-to-code edge carries no subset
        # and so was already refused when slicing -- which is why there is no tasklet-to-tasklet case.
        carriers: Dict[nodes.AccessNode, nodes.AccessNode] = {}
        for tasklet, clone in clones.items():
            for edge in state.in_edges(tasklet):
                if edge in wiring:
                    conn, point, container = wiring[edge]
                    state.add_edge(entry2, conn, clone, edge.dst_conn, point_memlet(container, point))
                else:
                    carrier = self.carrier(sdfg, state, edge.src, carriers)
                    state.add_edge(carrier, None, clone, edge.dst_conn, Memlet(data=carrier.data, subset='0'))
            if state.in_degree(clone) == 0:
                # A clone built only from literals still has to live inside the consumer's scope.
                state.add_edge(entry2, None, clone, None, Memlet())
        for tasklet, clone in clones.items():
            for edge in state.out_edges(tasklet):
                if edge is write:
                    state.add_edge(clone, edge.src_conn, sink, None, Memlet(data=sink_name, subset='0'))
                elif edge.dst in carriers:
                    carrier = carriers[edge.dst]
                    state.add_edge(clone, edge.src_conn, carrier, None, Memlet(data=carrier.data, subset='0'))
        return sink

    def carrier(self, sdfg: SDFG, state: SDFGState, origin: nodes.AccessNode,
                carriers: Dict[nodes.AccessNode, nodes.AccessNode]) -> nodes.AccessNode:
        """A fresh private register mirroring an intermediate value of the producer slice."""
        if origin in carriers:
            return carriers[origin]
        name, _ = sdfg.add_scalar('remat_' + origin.data,
                                  sdfg.arrays[origin.data].dtype,
                                  storage=dtypes.StorageType.Register,
                                  transient=True,
                                  find_new_name=True)
        node = state.add_access(name)
        carriers[origin] = node
        return node

    def prune_dead(self, sdfg: SDFG, state: SDFGState, tasklets: List[nodes.Tasklet]) -> None:
        """Remove producer-slice nodes whose only consumer was the deleted temporary."""
        queue = list(tasklets)
        while queue:
            node = queue.pop()
            if node not in state.nodes() or state.out_degree(node) != 0:
                continue
            preds = [e.src for e in state.in_edges(node)]
            state.remove_node(node)
            for pred in preds:
                if isinstance(pred, nodes.Tasklet):
                    queue.append(pred)
                elif isinstance(pred, nodes.AccessNode) and sdfg.arrays[pred.data].transient:
                    queue.append(pred)
