# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import collections.abc
from collections import defaultdict, deque
from dataclasses import dataclass

import sympy

from dace.sdfg.state import AbstractControlFlowRegion, ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion
from dace.subsets import Range
from dace.transformation import pass_pipeline as ppl, transformation
from dace import SDFG, SDFGState, properties, InterstateEdge, Memlet, data as dt, symbolic
from dace.sdfg.graph import Edge
from dace.sdfg import nodes as nd, utils as sdutil
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.sdfg.propagation import align_memlet
from typing import Dict, Iterable, Iterator, List, Set, Tuple, Any, Optional, Union
import networkx as nx
from networkx.algorithms import shortest_paths as nxsp
from ordered_set import OrderedSet

from dace.transformation.passes.analysis import loop_analysis

WriteScopeDict = Dict[str, Dict[Optional[Tuple[SDFGState, nd.AccessNode]],
                                Set[Union[Tuple[SDFGState, nd.AccessNode], Tuple[ControlFlowBlock, InterstateEdge]]]]]
SymbolScopeDict = Dict[str, Dict[Edge[InterstateEdge], Set[Union[Edge[InterstateEdge], ControlFlowBlock]]]]


@properties.make_properties
@transformation.explicit_cf_compatible
class StateReachability(ppl.Pass):
    """
    Evaluates state reachability (which other states can be executed after each state).
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified & ppl.Modifies.CFG

    def depends_on(self):
        return [ControlFlowBlockReachability]

    def apply_pass(self, top_sdfg: SDFG, pipeline_res: Dict) -> Dict[int, Dict[SDFGState, Set[SDFGState]]]:
        """
        :return: A dictionary mapping each state to its other reachable states. The reachable states are a read-only
                 set view, which is computed on demand.
        """
        # Ensure control flow block reachability is run if not run within a pipeline.
        if pipeline_res is None or not ControlFlowBlockReachability.__name__ in pipeline_res:
            cf_block_reach_dict = ControlFlowBlockReachability().apply_pass(top_sdfg, {})
        else:
            cf_block_reach_dict = pipeline_res[ControlFlowBlockReachability.__name__]
        reachable: Dict[int, Dict[SDFGState, Set[SDFGState]]] = {}
        for sdfg in top_sdfg.all_sdfgs_recursive():
            result: Dict[SDFGState, Set[SDFGState]] = defaultdict(OrderedSet)
            for state in sdfg.states():
                block_reach = cf_block_reach_dict[state.parent_graph.cfg_id][state]
                if isinstance(block_reach, ReachableBlocks):
                    result[state] = ReachableBlocks(block_reach.index, state, states_only=True)
                else:
                    result[state] = OrderedSet(r for r in block_reach if isinstance(r, SDFGState))
            reachable[sdfg.cfg_id] = result
        return reachable


@properties.make_properties
@transformation.explicit_cf_compatible
class ControlFlowBlockReachability(ppl.Pass):
    """
    Evaluates control flow block reachability (which control flow block can be executed after each control flow block)
    """

    CATEGORY: str = 'Analysis'

    contain_to_single_level = properties.Property(dtype=bool, default=False)

    def __init__(self, contain_to_single_level=False) -> None:
        super().__init__()

        self.contain_to_single_level = contain_to_single_level

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.CFG

    def apply_pass(self, top_sdfg: SDFG, _) -> Dict[int, Dict[ControlFlowBlock, Set[ControlFlowBlock]]]:
        """
        :return: For each control flow region, a dictionary mapping each control flow block to its other reachable
                 control flow blocks. The reachable blocks are a read-only set view (see :class:`ReachableBlocks`),
                 which answers membership queries without materializing the set.
        """
        top_sdfg.reset_cfg_list()
        index = BlockReachabilityIndex(top_sdfg)

        if self.contain_to_single_level:
            single_level_reachable: Dict[int, Dict[ControlFlowBlock,
                                                   Set[ControlFlowBlock]]] = defaultdict(lambda: defaultdict(set))
            for cfg in top_sdfg.all_control_flow_regions(recursive=True):
                for block in cfg.nx.nodes():
                    single_level_reachable[cfg.cfg_id][block] = ReachableBlocks(index, block, single_level=True)
            return single_level_reachable

        reachable: Dict[int, Dict[ControlFlowBlock, Set[ControlFlowBlock]]] = {}
        for sdfg in top_sdfg.all_sdfgs_recursive():
            for cfg in sdfg.all_control_flow_regions():
                result: Dict[ControlFlowBlock, Set[ControlFlowBlock]] = defaultdict(OrderedSet)
                for block in cfg.nodes():
                    result[block] = ReachableBlocks(index, block)
                reachable[cfg.cfg_id] = result
        return reachable


class BlockReachabilityIndex:
    """
    A snapshot of the control flow block hierarchy of an SDFG (and all nested SDFGs), with the transitive closure of
    each control flow region's edges stored as one bitset per block.

    A block ``b`` reaches a block ``t`` if, for ``b`` or any of its ancestor regions ``x`` (up to the SDFG), ``x``
    reaches ``t`` or an ancestor of ``t`` through the edges of ``x``'s parent region, or if ``x``'s parent region is a
    loop that contains ``t`` (any point in a loop may reach any other point in it again, including the loop itself).
    Branches of a conditional block do not reach each other. In single-level mode, only the parent region of ``b`` is
    considered, and a block in a loop reaches the loop's direct children (but not their contents). The index takes memory linear in the number of blocks
    (plus a quadratic number of bits per region), whereas materializing the reachable sets takes memory quadratic in
    the number of blocks for deeply nested control flow.
    """

    def __init__(self, top_sdfg: SDFG) -> None:
        #: The region every indexed block is a direct child of.
        self.parent: Dict[ControlFlowBlock, AbstractControlFlowRegion] = {}
        #: The direct child blocks of every region.
        self.children: Dict[AbstractControlFlowRegion, List[ControlFlowBlock]] = {}
        #: Per region, the position of each block in the region's graph and the reachability bitset of each position.
        self._position: Dict[AbstractControlFlowRegion, Dict[ControlFlowBlock, int]] = {}
        self._nodes: Dict[AbstractControlFlowRegion, List[ControlFlowBlock]] = {}
        self._reach: Dict[AbstractControlFlowRegion, List[int]] = {}
        for sdfg in top_sdfg.all_sdfgs_recursive():
            for cfg in sdfg.all_control_flow_regions():
                children = list(cfg.nodes())
                self.children[cfg] = children
                for block in children:
                    self.parent[block] = cfg
                graph = cfg.nx
                nodes = list(graph.nodes())
                position = {n: i for i, n in enumerate(nodes)}
                self._nodes[cfg] = nodes
                self._position[cfg] = position
                succ = [[position[d] for d in graph.successors(n)] for n in nodes]
                self._reach[cfg] = _transitive_closure_bitsets(succ)

    def reaches_in_region(self, region: AbstractControlFlowRegion, src: ControlFlowBlock,
                          dst: ControlFlowBlock) -> bool:
        """ Whether ``dst`` can execute after ``src`` through the edges of ``region``, both direct children of it. """
        position = self._position[region]
        i, j = position.get(src), position.get(dst)
        if i is None or j is None:
            return False
        return bool((self._reach[region][i] >> j) & 1)

    def reachable_in_region(self, region: AbstractControlFlowRegion, src: ControlFlowBlock) -> List[ControlFlowBlock]:
        """ The direct children of ``region`` that can execute after ``src`` through the edges of ``region``. """
        i = self._position[region].get(src)
        if i is None:
            return []
        nodes = self._nodes[region]
        bits = self._reach[region][i]
        result = []
        while bits:
            low = bits & -bits
            result.append(nodes[low.bit_length() - 1])
            bits ^= low
        return result

    def descendants(self, region: ControlFlowBlock) -> Iterator[ControlFlowBlock]:
        """ All blocks nested in ``region`` (not crossing into nested SDFGs), parents before their children. """
        stack = [iter(self.children.get(region, ()))]
        while stack:
            block = next(stack[-1], None)
            if block is None:
                stack.pop()
                continue
            yield block
            if block in self.children:
                stack.append(iter(self.children[block]))

    def reaches(self, src: ControlFlowBlock, dst: Any, single_level: bool = False) -> bool:
        """ Whether ``dst`` can execute after ``src`` (see the class documentation for the definition). """
        # Map each region enclosing ``dst`` (within its SDFG) to the child of that region containing ``dst``.
        dst_in: Dict[AbstractControlFlowRegion, ControlFlowBlock] = {}
        block = dst
        while True:
            region = self.parent.get(block)
            if region is None:
                break
            dst_in[region] = block
            if isinstance(region, SDFG):
                break
            block = region

        block = src
        while True:
            region = self.parent.get(block)
            if region is None:
                return False
            target = dst_in.get(region)
            if target is not None and self.reaches_in_region(region, block, target):
                return True
            if isinstance(region, LoopRegion):
                if single_level:
                    # Within one level, a loop's blocks reach its other direct children (but not their contents).
                    return target is dst and block in self._position[region]
                if dst is region or region in dst_in:
                    return True
            if single_level or isinstance(region, SDFG):
                return False
            block = region

    def reachable(self, src: ControlFlowBlock, single_level: bool = False) -> Iterator[ControlFlowBlock]:
        """ Iterates over the blocks that can execute after ``src``, without duplicates. """
        seen: Set[ControlFlowBlock] = set()
        block = src
        while True:
            region = self.parent.get(block)
            if region is None:
                return
            for reached in self.reachable_in_region(region, block):
                if reached not in seen:
                    seen.add(reached)
                    yield reached
                    for nested in self.descendants(reached):
                        if nested not in seen:
                            seen.add(nested)
                            yield nested
            if isinstance(region, LoopRegion) and single_level:
                if block in self._position[region]:
                    for child in self.children[region]:
                        if child not in seen:
                            seen.add(child)
                            yield child
                return
            if single_level or isinstance(region, SDFG):
                return
            if isinstance(region, LoopRegion):
                for nested in self.descendants(region):
                    if nested not in seen:
                        seen.add(nested)
                        yield nested
                if region not in seen:
                    seen.add(region)
                    yield region
            block = region


class ReachableBlocks(collections.abc.Set):
    """
    A read-only set view of the control flow blocks (or only the states) that can execute after a given block, backed
    by a :class:`BlockReachabilityIndex`. Membership tests take time proportional to the nesting depth; iterating
    computes the set on demand.
    """

    __slots__ = ('index', 'block', 'states_only', 'single_level')

    def __init__(self,
                 index: BlockReachabilityIndex,
                 block: ControlFlowBlock,
                 states_only: bool = False,
                 single_level: bool = False) -> None:
        self.index = index
        self.block = block
        self.states_only = states_only
        self.single_level = single_level

    def __contains__(self, item: Any) -> bool:
        if self.states_only and not isinstance(item, SDFGState):
            return False
        try:
            return self.index.reaches(self.block, item, self.single_level)
        except TypeError:  # Unhashable item
            return False

    def __iter__(self) -> Iterator[ControlFlowBlock]:
        for reached in self.index.reachable(self.block, self.single_level):
            if not self.states_only or isinstance(reached, SDFGState):
                yield reached

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __bool__(self) -> bool:
        return next(iter(self), None) is not None

    def __repr__(self) -> str:
        return f'{type(self).__name__}({list(self)!r})'

    @classmethod
    def _from_iterable(cls, it: Iterable[ControlFlowBlock]) -> OrderedSet:
        return OrderedSet(it)

    # Named methods of (ordered) sets, returning new ordered sets.
    def copy(self) -> OrderedSet:
        return OrderedSet(self)

    def union(self, *others: Iterable) -> OrderedSet:
        return OrderedSet(self).union(*others)

    def intersection(self, *others: Iterable) -> OrderedSet:
        return OrderedSet(self).intersection(*others)

    def difference(self, *others: Iterable) -> OrderedSet:
        return OrderedSet(self).difference(*others)

    def issubset(self, other: Iterable) -> bool:
        return self <= (other if isinstance(other, collections.abc.Set) else set(other))

    def issuperset(self, other: Iterable) -> bool:
        return all(o in self for o in other)


def _transitive_closure_bitsets(succ: List[List[int]]) -> List[int]:
    """
    Computes, for every node of a graph given as successor lists, the bitset of nodes reachable from it by a path of
    at least one edge (a node reaches itself only through a cycle). Uses Tarjan's algorithm, which emits strongly
    connected components in reverse topological order, so each component's successors are complete when it is
    visited.
    """
    n = len(succ)
    index = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    component = [-1] * n
    stack: List[int] = []
    # Per component (in the order Tarjan's algorithm emits them): the nodes reachable from its members, and the same
    # plus the members themselves (what a predecessor reaches through the component).
    comp_reach: List[int] = []
    comp_through: List[int] = []
    counter = 0
    for root in range(n):
        if index[root] != -1:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True
        work = [(root, 0)]
        while work:
            v, i = work[-1]
            if i < len(succ[v]):
                work[-1] = (v, i + 1)
                w = succ[v][i]
                if index[w] == -1:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack[w] = True
                    work.append((w, 0))
                elif on_stack[w] and index[w] < low[v]:
                    low[v] = index[w]
                continue
            work.pop()
            if work and low[v] < low[work[-1][0]]:
                low[work[-1][0]] = low[v]
            if low[v] != index[v]:
                continue
            # ``v`` is the root of a strongly connected component: pop it and compute its reachability.
            c = len(comp_reach)
            members = 0
            while True:
                w = stack.pop()
                on_stack[w] = False
                component[w] = c
                members |= 1 << w
                if w == v:
                    break
            reach = 0
            cyclic = members != (1 << v)
            m = members
            while m:
                low_bit = m & -m
                for w in succ[low_bit.bit_length() - 1]:
                    d = component[w]
                    if d == c:
                        cyclic = True  # Self-loop or edge within the component
                    else:
                        reach |= comp_through[d]
                m ^= low_bit
            comp_through.append(reach | members)
            comp_reach.append(reach | members if cyclic else reach)
    return [comp_reach[component[i]] for i in range(n)]


@properties.make_properties
@transformation.explicit_cf_compatible
class SymbolAccessSets(ppl.ControlFlowRegionPass):
    """
    Evaluates symbol access sets (which symbols are read/written in each control flow block or interstate edge).
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified & ppl.Modifies.States | ppl.Modifies.Edges | ppl.Modifies.Symbols | ppl.Modifies.Nodes

    def apply(self, region: ControlFlowRegion,
              _) -> Dict[Union[ControlFlowBlock, Edge[InterstateEdge]], Tuple[OrderedSet[str], OrderedSet[str]]]:
        adesc = set(region.sdfg.arrays.keys())
        result: Dict[ControlFlowBlock, Tuple[OrderedSet[str], OrderedSet[str]]] = {}
        for block in region.nodes():
            # No symbols may be written to inside blocks.
            result[block] = (block.free_symbols, set())
            for oedge in region.out_edges(block):
                edge_readset = oedge.data.read_symbols() - adesc
                edge_writeset = set(oedge.data.assignments.keys())
                result[oedge] = (edge_readset, edge_writeset)
        return result


@properties.make_properties
@transformation.explicit_cf_compatible
class AccessSets(ppl.Pass):
    """
    Evaluates memory access sets (which arrays/data descriptors are read/written in each control flow block).
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If access nodes were modified, reapply
        return modified & ppl.Modifies.AccessNodes

    def _get_loop_region_readset(self, loop: LoopRegion, arrays: OrderedSet[str]) -> OrderedSet[str]:
        readset = set()
        exprs = {loop.loop_condition.as_string}
        update_stmt = loop_analysis.get_update_assignment(loop)
        init_stmt = loop_analysis.get_init_assignment(loop)
        if update_stmt:
            exprs.add(update_stmt)
        if init_stmt:
            exprs.add(init_stmt)
        for expr in exprs:
            readset |= (symbolic.free_symbols_and_functions(expr) | symbolic.arrays(expr)) & arrays
        return readset

    def apply_pass(self, top_sdfg: SDFG, _) -> Dict[ControlFlowBlock, Tuple[OrderedSet[str], OrderedSet[str]]]:
        """
        :return: A dictionary mapping each control flow block to a tuple of its (read, written) data descriptors.
        """
        result: Dict[ControlFlowBlock, Tuple[OrderedSet[str], OrderedSet[str]]] = {}
        for sdfg in top_sdfg.all_sdfgs_recursive():
            arrays: OrderedSet[str] = OrderedSet(sdfg.arrays.keys())
            for block in sdfg.all_control_flow_blocks():
                readset, writeset = OrderedSet(), OrderedSet()
                if isinstance(block, SDFGState):
                    for anode in block.data_nodes():
                        if block.in_degree(anode) > 0:
                            writeset.add(anode.data)
                        if block.out_degree(anode) > 0:
                            readset.add(anode.data)
                elif isinstance(block, AbstractControlFlowRegion):
                    for state in block.all_states():
                        for anode in state.data_nodes():
                            if state.in_degree(anode) > 0:
                                writeset.add(anode.data)
                            if state.out_degree(anode) > 0:
                                readset.add(anode.data)
                    if isinstance(block, LoopRegion):
                        readset |= self._get_loop_region_readset(block, arrays)
                    elif isinstance(block, ConditionalBlock):
                        for cond, _ in block.branches:
                            if cond is not None:
                                readset |= (symbolic.free_symbols_and_functions(cond.as_string)
                                            | symbolic.arrays(cond.as_string)) & arrays

                result[block] = (readset, writeset)

            # Edges that read from arrays add to both ends' access sets
            anames = sdfg.arrays.keys()
            for e in sdfg.all_interstate_edges():
                fsyms = sorted(e.data.free_symbols & anames)
                if fsyms:
                    result[e.src][0].update(fsyms)
                    result[e.dst][0].update(fsyms)
        return result


@properties.make_properties
@transformation.explicit_cf_compatible
class FindAccessStates(ppl.Pass):
    """
    For each data descriptor, creates a set of states in which access nodes of that data are used.
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified & ppl.Modifies.AccessNodes

    def apply_pass(self, top_sdfg: SDFG, _) -> Dict[int, Dict[str, OrderedSet[SDFGState]]]:
        """
        :return: A dictionary mapping each data descriptor name to states where it can be found in.
        """
        top_result: Dict[int, Dict[str, OrderedSet[SDFGState]]] = {}

        for sdfg in top_sdfg.all_sdfgs_recursive():
            result: Dict[str, OrderedSet[SDFGState]] = defaultdict(OrderedSet)
            for state in sdfg.states():
                for anode in state.data_nodes():
                    result[anode.data].add(state)

            # Edges that read from arrays add to both ends' access sets
            anames = sdfg.arrays.keys()
            for e in sdfg.all_interstate_edges():
                fsyms = sorted(e.data.free_symbols & anames)
                for access in fsyms:
                    result[access].update((e.src, e.dst))

            top_result[sdfg.cfg_id] = result
        return top_result


@properties.make_properties
@transformation.explicit_cf_compatible
class FindSingleUseData(ppl.Pass):
    """
    For each SDFG find all data descriptors that are referenced in exactly one location.

    In addition to the requirement that there exists exactly one AccessNode that
    refers to a data descriptor the following conditions have to be meet as well:
    - The data is not read on an interstate edge.
    - The data is not accessed in the branch condition, loop condition, etc. of
        control flow regions.
    - There must be at least one AccessNode that refers to the data. I.e. if it exists
        inside `SDFG.arrays` but there is no AccessNode, then it is _not_ included.

    It is also important to note that the degree of the AccessNodes are ignored.
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified & ppl.Modifies.AccessNodes & ppl.Modifies.CFG

    def apply_pass(self, sdfg: SDFG, _) -> Dict[SDFG, OrderedSet[str]]:
        """
        :return: A dictionary mapping SDFGs to a `set` of strings containing the name
            of the data descriptors that are only used once.
        """
        # TODO(pschaad): Should we index on cfg or the SDFG itself.
        exclusive_data: Dict[SDFG, OrderedSet[str]] = {}
        for nsdfg in sdfg.all_sdfgs_recursive():
            exclusive_data[nsdfg] = self._find_single_use_data_in_sdfg(nsdfg)
        return exclusive_data

    def _find_single_use_data_in_sdfg(self, sdfg: SDFG) -> OrderedSet[str]:
        """Scans an SDFG and computes the data that is only used once in the SDFG.

        The rules used to classify data descriptors are outlined above. The function
        will not scan nested SDFGs.

        :return: The set of data descriptors that are used once in the SDFG.
        """
        # If we encounter a data descriptor for the first time we immediately
        #  classify it as single use. We will undo this decision as soon as
        #  learn that it is used somewhere else.
        single_use_data: OrderedSet[str] = OrderedSet()
        previously_seen: OrderedSet[str] = OrderedSet()

        for state in sdfg.states():
            for dnode in state.data_nodes():
                data_name: str = dnode.data
                if data_name in single_use_data:
                    single_use_data.discard(data_name)  # Classified too early -> Undo
                elif data_name not in previously_seen:
                    single_use_data.add(data_name)  # Never seen -> Assume single use
                previously_seen.add(data_name)

        # By definition, data that is referenced by interstate edges is not single
        #  use data, also remove it.
        for edge in sdfg.all_interstate_edges():
            single_use_data.difference_update(edge.data.free_symbols)

        # By definition, data that is referenced by the conditions (branching condition,
        #  loop condition, ...) is not single use data, also remove that.
        for cfr in sdfg.all_control_flow_regions():
            single_use_data.difference_update(cfr.used_symbols(all_symbols=True, with_contents=False))

        return single_use_data


@properties.make_properties
@transformation.explicit_cf_compatible
class FindAccessNodes(ppl.Pass):
    """
    For each data descriptor, creates a dictionary mapping states to all read and write access nodes with the given
    data descriptor.
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.AccessNodes

    def apply_pass(
            self, top_sdfg: SDFG,
            _) -> Dict[int, Dict[str, Dict[SDFGState, Tuple[OrderedSet[nd.AccessNode], OrderedSet[nd.AccessNode]]]]]:
        """
        :return: A dictionary mapping each data descriptor name to a dictionary keyed by states with all access nodes
                 that use that data descriptor.
        """
        top_result: Dict[int, Dict[str, OrderedSet[nd.AccessNode]]] = dict()

        for sdfg in top_sdfg.all_sdfgs_recursive():
            result: Dict[str, Dict[SDFGState,
                                   Tuple[OrderedSet[nd.AccessNode], OrderedSet[nd.AccessNode]]]] = defaultdict(
                                       lambda: defaultdict(lambda: [OrderedSet(), OrderedSet()]))
            for state in sdfg.states():
                for anode in state.data_nodes():
                    # An EMPTY memlet is an ordering edge, not a transfer: it moves no data, so a node
                    # reached only by such edges is neither read nor written. Classifying by raw degree
                    # reports it as a write, and ``ScalarWriteShadowScopes`` then lists it both as a read
                    # of the real write and as a write scope of its own -- ``ScalarFission`` renames it
                    # twice and versions a later read onto a container nothing ever writes.
                    if any(not e.data.is_empty() for e in state.in_edges(anode)):
                        result[anode.data][state][1].add(anode)
                    if any(not e.data.is_empty() for e in state.out_edges(anode)):
                        result[anode.data][state][0].add(anode)
            top_result[sdfg.cfg_id] = result
        return top_result


@properties.make_properties
@transformation.explicit_cf_compatible
class SymbolWriteScopes(ppl.ControlFlowRegionPass):
    """
    For each symbol, create a dictionary mapping each interstate edge writing to that symbol to the set of interstate
    edges and states reading that symbol that are dominated by that write.
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Symbols | ppl.Modifies.CFG | ppl.Modifies.Edges | ppl.Modifies.Nodes

    def depends_on(self):
        return [SymbolAccessSets, ControlFlowBlockReachability]

    def _find_dominating_write(self, sym: str, read: Union[ControlFlowBlock, Edge[InterstateEdge]],
                               block_idom: Dict[ControlFlowBlock, ControlFlowBlock]) -> Optional[Edge[InterstateEdge]]:
        last_block: ControlFlowBlock = read if isinstance(read, ControlFlowBlock) else read.src

        in_edges = last_block.parent_graph.in_edges(last_block)
        deg = len(in_edges)
        if deg == 0:
            return None
        elif deg == 1 and any([sym == k for k in in_edges[0].data.assignments.keys()]):
            return in_edges[0]

        write_isedge = None
        n_block = block_idom[last_block] if block_idom[last_block] != last_block else None
        while n_block is not None and write_isedge is None:
            oedges = n_block.parent_graph.out_edges(n_block)
            odeg = len(oedges)
            if odeg == 1:
                if any([sym == k for k in oedges[0].data.assignments.keys()]):
                    write_isedge = oedges[0]
            else:
                dom_edge = None
                for cand in oedges:
                    if nxsp.has_path(n_block.parent_graph.nx, cand.dst, last_block):
                        if dom_edge is not None:
                            dom_edge = None
                            break
                        elif any([sym == k for k in cand.data.assignments.keys()]):
                            dom_edge = cand
                write_isedge = dom_edge
            n_block = block_idom[n_block] if block_idom[n_block] != n_block else None
        return write_isedge

    def apply(self, region, pipeline_results) -> SymbolScopeDict:
        result: SymbolScopeDict = defaultdict(lambda: defaultdict(lambda: OrderedSet()))

        idom = nx.immediate_dominators(region.nx, region.start_block)
        all_doms = cfg_analysis.all_dominators(region, idom)

        b_reach: Dict[ControlFlowBlock, OrderedSet[ControlFlowBlock]] = pipeline_results[
            ControlFlowBlockReachability.__name__][region.cfg_id]
        symbol_access_sets: Dict[Union[ControlFlowBlock, Edge[InterstateEdge]],
                                 Tuple[OrderedSet[str],
                                       OrderedSet[str]]] = pipeline_results[SymbolAccessSets.__name__][region.cfg_id]

        for read_loc, (reads, _) in symbol_access_sets.items():
            for sym in reads:
                dominating_write = self._find_dominating_write(sym, read_loc, idom)
                result[sym][dominating_write].add(read_loc)

        # If any write A is dominated by another write B and any reads in B's scope are also reachable by A, then merge
        # A and its scope into B's scope.
        to_remove = set()
        for sym in result.keys():
            for write, accesses in result[sym].items():
                if write is None:
                    continue
                dominators = all_doms[write.dst]
                reach = b_reach[write.dst]
                for dom in dominators:
                    iedges = dom.parent_graph.in_edges(dom)
                    if len(iedges) == 1 and iedges[0] in result[sym]:
                        other_accesses = result[sym][iedges[0]]
                        coarsen = False
                        for a_state_or_edge in other_accesses:
                            if isinstance(a_state_or_edge, SDFGState):
                                if a_state_or_edge in reach:
                                    coarsen = True
                                    break
                            else:
                                if a_state_or_edge.src in reach:
                                    coarsen = True
                                    break
                        if coarsen:
                            other_accesses.update(accesses)
                            other_accesses.add(write)
                            to_remove.add((sym, write))
                            result[sym][write] = OrderedSet()
        for sym, write in to_remove:
            del result[sym][write]

        return result


@properties.make_properties
@transformation.explicit_cf_compatible
class ScalarWriteShadowScopes(ppl.Pass):
    """
    For each scalar or array of size 1, create a dictionary mapping writes to that data container to the set of reads
    and writes that are dominated by that write.
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified & ppl.Modifies.States

    def depends_on(self):
        return [AccessSets, FindAccessNodes, ControlFlowBlockReachability]

    def _find_dominating_write(self,
                               desc: str,
                               block: ControlFlowBlock,
                               read: Union[nd.AccessNode, InterstateEdge],
                               access_nodes: Dict[SDFGState, Tuple[OrderedSet[nd.AccessNode],
                                                                   OrderedSet[nd.AccessNode]]],
                               idom_dict: Dict[ControlFlowRegion, Dict[ControlFlowBlock, ControlFlowBlock]],
                               access_sets: Dict[ControlFlowBlock, Tuple[OrderedSet[str], OrderedSet[str]]],
                               no_self_shadowing: bool = False) -> Optional[Tuple[SDFGState, nd.AccessNode]]:
        if isinstance(read, nd.AccessNode):
            state: SDFGState = block
            # If the read is also a write, it shadows itself.
            iedges = state.in_edges(read)
            if len(iedges) > 0 and any(not e.data.is_empty() for e in iedges) and not no_self_shadowing:
                return (state, read)

            # Find a dominating write within the same state.
            # TODO: Can this be done more efficiently?
            closest_candidate = None
            write_nodes = access_nodes[desc][state][1]
            for cand in write_nodes:
                if cand != read and nxsp.has_path(state._nx, cand, read):
                    if closest_candidate is None or nxsp.has_path(state._nx, closest_candidate, cand):
                        closest_candidate = cand
            if closest_candidate is not None:
                return (state, closest_candidate)
        elif isinstance(read, InterstateEdge) and isinstance(block, SDFGState):
            # Attempt to find a shadowing write in the current state.
            # TODO: Can this be done more efficiently?
            closest_candidate = None
            write_nodes = access_nodes[desc][block][1]
            for cand in write_nodes:
                if closest_candidate is None or nxsp.has_path(block._nx, closest_candidate, cand):
                    closest_candidate = cand
            if closest_candidate is not None:
                return (block, closest_candidate)

        # Find the dominating write state if the current block is not the dominating write state.
        write_state = None
        pivot_block = block
        region = block.parent_graph
        while region is not None and write_state is None:
            nblock = idom_dict[region][pivot_block] if idom_dict[region][pivot_block] != block else None
            while nblock is not None and write_state is None:
                if isinstance(nblock, SDFGState) and desc in access_sets[nblock][1]:
                    write_state = nblock
                nblock = idom_dict[region][nblock] if idom_dict[region][nblock] != nblock else None
            # No dominating write found in the current control flow graph, check one further up.
            if write_state is None:
                pivot_block = region
                region = region.parent_graph

        # Find a dominating write in the write state, i.e., the 'last' write to the data container.
        if write_state is not None:
            closest_candidate = None
            for cand in access_nodes[desc][write_state][1]:
                if write_state.out_degree(cand) == 0:
                    closest_candidate = cand
                    break
                elif closest_candidate is None or nxsp.has_path(write_state._nx, closest_candidate, cand):
                    closest_candidate = cand
            if closest_candidate is not None:
                return (write_state, closest_candidate)

        return None

    def apply_pass(self, top_sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Dict[int, WriteScopeDict]:
        """
        :return: A dictionary mapping each data descriptor name to a dictionary, where writes to that data descriptor
                 and the states they are contained in are mapped to the set of reads and writes (and their states) that
                 are dominated by that write.
        """
        top_result: Dict[int, WriteScopeDict] = dict()

        access_sets: Dict[ControlFlowBlock, Tuple[OrderedSet[str],
                                                  OrderedSet[str]]] = pipeline_results[AccessSets.__name__]

        for sdfg in top_sdfg.all_sdfgs_recursive():
            result: WriteScopeDict = defaultdict(lambda: defaultdict(lambda: OrderedSet()))
            idom_dict: Dict[ControlFlowRegion, Dict[ControlFlowBlock, ControlFlowBlock]] = {}
            all_doms_transitive: Dict[ControlFlowBlock,
                                      OrderedSet[ControlFlowBlock]] = defaultdict(lambda: OrderedSet())
            for cfg in sdfg.all_control_flow_regions():
                if isinstance(cfg, ConditionalBlock):
                    idom_dict[cfg] = {b: b for _, b in cfg.branches}
                    all_doms = {b: OrderedSet([b]) for _, b in cfg.branches}
                else:
                    idom_dict[cfg] = nx.immediate_dominators(cfg.nx, cfg.start_block)
                    all_doms = cfg_analysis.all_dominators(cfg, idom_dict[cfg])

                # Since all_control_flow_regions goes top-down in the graph hierarchy, we can build a transitive
                # closure of all dominators her.
                for k in all_doms.keys():
                    all_doms_transitive[k].update(all_doms[k])
                    all_doms_transitive[k].add(cfg)
                    all_doms_transitive[k].update(all_doms_transitive[cfg])

            access_nodes: Dict[str, Dict[SDFGState, Tuple[OrderedSet[nd.AccessNode],
                                                          OrderedSet[nd.AccessNode]]]] = pipeline_results[
                                                              FindAccessNodes.__name__][sdfg.cfg_id]

            block_reach: Dict[ControlFlowBlock,
                              OrderedSet[ControlFlowBlock]] = pipeline_results[ControlFlowBlockReachability.__name__]

            anames = sdfg.arrays.keys()
            for desc in sdfg.arrays:
                desc_states_with_nodes = OrderedSet(access_nodes[desc].keys())
                for state in desc_states_with_nodes:
                    for read_node in access_nodes[desc][state][0]:
                        write = self._find_dominating_write(desc, state, read_node, access_nodes, idom_dict,
                                                            access_sets)
                        result[desc][write].add((state, read_node))
                # Ensure accesses to interstate edges are also considered.
                for block, accesses in access_sets.items():
                    if desc in accesses[0]:
                        out_edges = block.parent_graph.out_edges(block)
                        for oedge in out_edges:
                            syms = oedge.data.free_symbols & anames
                            if desc in syms:
                                write = self._find_dominating_write(desc, block, oedge.data, access_nodes, idom_dict,
                                                                    access_sets)
                                result[desc][write].add((block, oedge.data))
                # Take care of any write nodes that have not been assigned to a scope yet, i.e., writes that are not
                # dominating any reads and are thus not part of the results yet.
                for state in desc_states_with_nodes:
                    for write_node in access_nodes[desc][state][1]:
                        if not (state, write_node) in result[desc]:
                            write = self._find_dominating_write(desc,
                                                                state,
                                                                write_node,
                                                                access_nodes,
                                                                idom_dict,
                                                                access_sets,
                                                                no_self_shadowing=True)
                            result[desc][write].add((state, write_node))

                # If any write A is dominated by another write B and any reads in B's scope are also reachable by A,
                # then merge A and its scope into B's scope.
                to_remove = OrderedSet()
                for write, accesses in result[desc].items():
                    if write is None:
                        continue
                    write_state, write_node = write
                    dominators = all_doms_transitive[write_state]
                    reach = block_reach[write_state.parent_graph.cfg_id][write_state]
                    for other_write, other_accesses in result[desc].items():
                        if other_write is not None and other_write[1] is write_node and other_write[0] is write_state:
                            continue
                        if other_write is None or other_write[0] in dominators:
                            noa = len(other_accesses)
                            if noa > 0 and (noa > 1 or list(other_accesses)[0] != other_write):
                                if any([a_state in reach for a_state, _ in other_accesses]):
                                    other_accesses.update(accesses)
                                    other_accesses.add(write)
                                    to_remove.add(write)
                                    result[desc][write] = OrderedSet()
                for write in to_remove:
                    del result[desc][write]
            top_result[sdfg.cfg_id] = result
        return top_result


@properties.make_properties
@transformation.explicit_cf_compatible
class AccessRanges(ppl.Pass):
    """
    For each data descriptor, finds all memlets used to access it (read/write ranges).
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Memlets

    def apply_pass(self, top_sdfg: SDFG, _) -> Dict[int, Dict[str, OrderedSet[Memlet]]]:
        """
        :return: A dictionary mapping each data descriptor name to a set of memlets.
        """
        top_result: Dict[int, Dict[str, OrderedSet[Memlet]]] = dict()

        for sdfg in top_sdfg.all_sdfgs_recursive():
            result: Dict[str, OrderedSet[Memlet]] = defaultdict(OrderedSet)
            for state in sdfg.states():
                for anode in state.data_nodes():
                    for e in state.all_edges(anode):
                        if e.dst is anode and e.dst_conn == 'set':  # Skip reference sets
                            continue
                        if e.data.is_empty():  # Skip empty memlets
                            continue
                        # Find (hopefully propagated) root memlet
                        e = state.memlet_tree(e).root().edge
                        result[anode.data].add(e.data)
            top_result[sdfg.cfg_id] = result
        return top_result


@dataclass(unsafe_hash=True)
@properties.make_properties
@transformation.explicit_cf_compatible
class FindReferenceSources(ppl.Pass):
    """
    For each Reference data descriptor, finds all memlets used to set it. If a Tasklet was used
    to set the reference, the Tasklet is given as a source.
    """

    CATEGORY: str = 'Analysis'

    trace_through_code = properties.Property(dtype=bool, default=False, desc='Trace inputs through tasklets.')
    recursive = properties.Property(dtype=bool, default=False, desc='Add reference of reference dependencies.')

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Memlets

    def apply_pass(self, top_sdfg: SDFG, _) -> Dict[int, Dict[str, OrderedSet[Union[Memlet, nd.CodeNode]]]]:
        """
        :return: A dictionary mapping each data descriptor name to a set of memlets.
        """
        top_result: Dict[int, Dict[str, OrderedSet[Union[Memlet, nd.CodeNode]]]] = dict()

        for sdfg in top_sdfg.all_sdfgs_recursive():
            result: Dict[str, OrderedSet[Memlet]] = defaultdict(OrderedSet)
            reference_descs = OrderedSet(k for k, v in sdfg.arrays.items() if isinstance(v, dt.Reference))
            for state in sdfg.states():
                code_sources: Dict[str, OrderedSet[nd.CodeNode]] = defaultdict(OrderedSet)
                for anode in state.data_nodes():
                    if anode.data not in reference_descs:
                        continue
                    for e in state.in_edges(anode):
                        if e.dst_conn != 'set':
                            continue
                        true_src = state.memlet_path(e)[0].src
                        if isinstance(true_src, nd.CodeNode):
                            # Code  -> Reference
                            result[anode.data].add(true_src)
                            code_sources[anode.data].add(true_src)
                        else:
                            # Array -> Reference
                            result[anode.data].add(align_memlet(state, e, dst=False))

                            # If array is view, add view targets
                            view_targets = sdutil.get_all_view_edges(state, true_src)
                            for te in view_targets:
                                result[anode.data].add(align_memlet(state, te, dst=False))

                        if 'views' in anode.out_connectors:  # Reference and view
                            out_edge, = state.out_edges_by_connector(anode, 'views')
                            if isinstance(out_edge.dst, nd.AccessNode):
                                view_targets = sdutil.get_all_view_nodes(state, out_edge.dst)
                            for target in view_targets:
                                if isinstance(true_src, nd.CodeNode):
                                    # Code  -> Reference
                                    result[target.data].add(true_src)
                                    code_sources[target.data].add(true_src)
                                else:
                                    # Array -> Reference
                                    result[target.data].add(align_memlet(state, e, dst=False))

                # Trace back through code nodes
                if self.trace_through_code:
                    for name, codes in code_sources.items():
                        sources = deque(codes)
                        while sources:
                            src = sources.pop()
                            if isinstance(src, nd.CodeNode):
                                for e in state.in_edges(src):
                                    true_src = state.memlet_path(e)[0].src
                                    if isinstance(true_src, nd.CodeNode):
                                        # Keep traversing backwards
                                        sources.append(true_src)
                                    else:
                                        result[name].add(e.data)

            # Recursively add dependencies of reference dependencies
            if self.recursive:
                for k, v in result.items():
                    for src in list(v):
                        if not isinstance(v, nd.CodeNode) and src.data in result:
                            v.update(result[src.data])

            top_result[sdfg.cfg_id] = result
        return top_result


@properties.make_properties
@transformation.explicit_cf_compatible
class DeriveSDFGConstraints(ppl.Pass):

    CATEGORY: str = 'Analysis'

    assume_max_data_size = properties.Property(dtype=int,
                                               default=None,
                                               allow_none=True,
                                               desc='Assume that all data containers have no dimension larger than ' +
                                               'this value. If None, no assumption is made.')

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified & ppl.Modifies.Everything

    def _derive_parameter_datasize_constraints(self, sdfg: SDFG, invariants: Dict[str, OrderedSet[str]]) -> None:
        handled = OrderedSet()
        for arr in sdfg.arrays.values():
            for dim in arr.shape:
                if isinstance(dim, symbolic.symbol) and not dim in handled:
                    ds = str(dim)
                    if ds not in invariants:
                        invariants[ds] = OrderedSet()
                    invariants[ds].add(f'{ds} > 0')
                    if self.assume_max_data_size is not None:
                        invariants[ds].add(f'{ds} <= {self.assume_max_data_size}')
                    handled.add(ds)

    def apply_pass(self, sdfg: SDFG,
                   _) -> Tuple[Dict[str, OrderedSet[str]], Dict[str, OrderedSet[str]], Dict[str, OrderedSet[str]]]:
        invariants: Dict[str, OrderedSet[str]] = {}
        self._derive_parameter_datasize_constraints(sdfg, invariants)
        return {}, invariants, {}


@transformation.explicit_cf_compatible
class StatePropagation(ppl.ControlFlowRegionPass):
    """
    Analyze a control flow region to determine the number of times each block inside of it is executed in the form of a
    symbolic expression, or a concrete number where possible.
    Each control flow block is marked with a symbolic expression for the number of executions, and a boolean flag to
    indicate whether the number of executions is dynamic or not. A combination of dynamic being set to true and the
    number of executions being 0 indicates that the number of executions is dynamically unbounded.
    Additionally, the pass annotates each block with a `ranges` property, which indicates for loop variables defined
    at that block what range of values the variable may take on.
    Note: This path directly annotates the graph.
    This pass supersedes `dace.sdfg.propagation.propagate_states` and is based on its algorithm, with significant
    simplifications thanks to the use of control flow regions.
    """

    CATEGORY: str = 'Analysis'

    def __init__(self):
        super().__init__()
        self.top_down = True
        self.apply_to_conditionals = True

    def depends_on(self):
        return [ControlFlowBlockReachability]

    def _propagate_in_cfg(self, cfg: ControlFlowRegion, reachable: Dict[ControlFlowBlock, OrderedSet[ControlFlowBlock]],
                          starting_executions: int, starting_dynamic_executions: bool):
        visited_blocks: OrderedSet[ControlFlowBlock] = OrderedSet()
        traversal_q: deque[Tuple[ControlFlowBlock, int, bool, List[str]]] = deque()
        traversal_q.append((cfg.start_block, starting_executions, starting_dynamic_executions, []))
        while traversal_q:
            (block, proposed_executions, proposed_dynamic, itvar_stack) = traversal_q.pop()
            out_edges = cfg.out_edges(block)
            if block in visited_blocks:
                # This block has already been visited, meaning there are multiple paths towards this block.
                if proposed_executions == 0 and proposed_dynamic:
                    block.executions = 0
                    block.dynamic_executions = True
                else:
                    block.executions = sympy.Max(block.executions, proposed_executions).doit()
                    block.dynamic_executions = (block.dynamic_executions or proposed_dynamic)
            elif proposed_dynamic and proposed_executions == 0:
                # We're propagating a dynamic unbounded number of executions, which always gets propagated
                # unconditionally. Propagate to all children.
                visited_blocks.add(block)
                block.executions = proposed_executions
                block.dynamic_executions = proposed_dynamic
                # This gets pushed through to all children unconditionally.
                if len(out_edges) > 0:
                    for oedge in out_edges:
                        traversal_q.append((oedge.dst, proposed_executions, proposed_dynamic, itvar_stack))
            else:
                # If the state hasn't been visited yet and we're not propagating a dynamic unbounded number of
                # executions, we calculate the number of executions for the next state(s) and continue propagating.
                visited_blocks.add(block)
                block.executions = proposed_executions
                block.dynamic_executions = proposed_dynamic
                if len(out_edges) == 1:
                    # Continue with the only child state.
                    if not out_edges[0].data.is_unconditional():
                        # If the transition to the child state is based on a condition, this state could be an implicit
                        # exit state. The child state's number of executions is thus only given as an upper bound and
                        # marked as dynamic.
                        proposed_dynamic = True
                    traversal_q.append((out_edges[0].dst, proposed_executions, proposed_dynamic, itvar_stack))
                elif len(out_edges) > 1:
                    # Conditional split
                    for oedge in out_edges:
                        traversal_q.append((oedge.dst, block.executions, True, itvar_stack))

        # Check if the CFG contains any cycles. Any cycles left in the graph (after control flow raising) are
        # irreducible control flow and thus lead to a dynamically unbounded number of executions. Mark any block
        # inside and reachable from any block inside the cycle as dynamically unbounded, irrespectively of what it was
        # marked as before.
        cycles: Iterable[Iterable[ControlFlowBlock]] = cfg.find_cycles()
        for cycle in cycles:
            for blk in cycle:
                blk.executions = 0
                blk.dynamic_executions = True
                for reached in reachable[blk]:
                    reached.executions = 0
                    blk.dynamic_executions = True

    def apply(self, region, pipeline_results) -> None:
        if isinstance(region, ConditionalBlock):
            # In a conditional block, each branch is executed up to as many times as the conditional block itself is.
            # TODO(later): We may be able to derive ranges here based on the branch conditions too.
            for _, b in region.branches:
                b.executions = region.executions
                b.dynamic_executions = True
                b.ranges = region.ranges
        else:
            if isinstance(region, SDFG):
                # The root SDFG is executed exactly once, any other, nested SDFG is executed as many times as the parent
                # state is.
                if region is region.root_sdfg:
                    region.executions = 1
                    region.dynamic_executions = False
                elif region.parent:
                    region.executions = region.parent.executions
                    region.dynamic_executions = region.parent.dynamic_executions

            # Clear existing annotations.
            for blk in region.nodes():
                blk.executions = 0
                blk.dynamic_executions = True
                blk.ranges = region.ranges

            # Determine the number of executions for the start block within this region. In the case of loops, this
            # is dependent on the number of loop iterations - where they can be determined. Where they may not be
            # determined, the number of iterations is assumed to be dynamically unbounded. For any other control flow
            # region, the start block is executed as many times as the region itself is.
            starting_execs = region.executions
            starting_dynamic = region.dynamic_executions
            if isinstance(region, LoopRegion):
                # If inside a loop, add range information if possible.
                start = loop_analysis.get_init_assignment(region)
                stop = loop_analysis.get_loop_end(region)
                stride = loop_analysis.get_loop_stride(region)
                if start is not None and stop is not None and stride is not None and region.loop_variable:
                    # This inequality needs to be checked exactly like this due to constraints in sympy/symbolic
                    # expressions, do not simplify!
                    if (stride < 0) == True:
                        rng = (stop, start, -stride)
                    else:
                        rng = (start, stop, stride)
                    for blk in region.nodes():
                        blk.ranges[str(region.loop_variable)] = Range([rng])

                    # Get surrounding iteration variables for the case of nested loops.
                    itvar_stack = []
                    par = region.parent_graph
                    while par is not None and not isinstance(par, SDFG):
                        if isinstance(par, LoopRegion) and par.loop_variable:
                            itvar_stack.append(par.loop_variable)
                        par = par.parent_graph

                    # Calculate the number of loop executions.
                    # This resolves ranges based on the order of iteration variables from surrounding loops.
                    loop_executions = sympy.ceiling(((stop + 1) - start) / stride)
                    for outer_itvar_string in itvar_stack:
                        outer_range = region.ranges[outer_itvar_string]
                        outer_start = outer_range[0][0]
                        outer_stop = outer_range[0][1]
                        outer_stride = outer_range[0][2]
                        outer_itvar = symbolic.pystr_to_symbolic(outer_itvar_string)
                        exec_repl = loop_executions.subs({outer_itvar: (outer_itvar * outer_stride + outer_start)})
                        sum_rng = (outer_itvar, 0, sympy.ceiling((outer_stop - outer_start) / outer_stride))
                        loop_executions = sympy.Sum(exec_repl, sum_rng)
                    starting_execs = loop_executions.doit()
                    starting_dynamic = region.dynamic_executions
                else:
                    starting_execs = 0
                    starting_dynamic = True

            # Propagate the number of executions.
            self._propagate_in_cfg(region, pipeline_results[ControlFlowBlockReachability.__name__][region.cfg_id],
                                   starting_execs, starting_dynamic)


@properties.make_properties
@transformation.explicit_cf_compatible
class ConditionUniqueWrites(ppl.Pass):
    """
    Finds all access nodes in ConditionalBlocks, which are not written in all branches (i.e. these locations are not guaranteed to be written to).
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified & ppl.Modifies.CFG

    def depends_on(self):
        return []

    def apply_pass(self, top_sdfg: SDFG, pipeline_res: Dict) -> OrderedSet[nd.AccessNode]:
        """
        :return: A set of access nodes, which are unique writes in conditional blocks.
        """
        cond_unique = OrderedSet()
        for cfb in top_sdfg.all_control_flow_blocks(recursive=True):
            if not isinstance(cfb, ConditionalBlock):
                continue

            # No else branch -> all access nodes are unique
            if not any(cnd is None for cnd, br in cfb.branches):
                cond_unique.update(an for an in cfb.data_nodes())
                continue

            # Build a mapping of access_node -> written subset -> set of branches it appears in
            access_write_branch = {}
            for _, br in cfb.branches:
                for st in br.all_states():
                    for an in st.data_nodes():
                        array_name = an.data
                        write_subsets = OrderedSet(e.data.dst_subset for e in st.in_edges(an))
                        wss = str(write_subsets)
                        if array_name not in access_write_branch:
                            access_write_branch[array_name] = {}
                        if wss not in access_write_branch[array_name]:
                            access_write_branch[array_name][wss] = {
                                "branches": OrderedSet(),
                                "access_nodes": OrderedSet()
                            }
                        access_write_branch[array_name][wss]["branches"].add(br)
                        access_write_branch[array_name][wss]["access_nodes"].add(an)

            # Eliminate all write subset that appear in all branches
            for array_name, ws_br in list(access_write_branch.items()):
                to_remove = []
                for wss, brd in list(ws_br.items()):
                    if len(brd["branches"]) == len(cfb.branches):
                        to_remove.append(wss)
                for wss in to_remove:
                    del ws_br[wss]
                if len(ws_br) == 0:
                    del access_write_branch[array_name]

            # All remaining access nodes are unique
            for array_name, ws_br in access_write_branch.items():
                for wss, brd in ws_br.items():
                    cond_unique.update(brd["access_nodes"])

        return cond_unique
