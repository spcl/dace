# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Type

from dace import SDFG, Memlet, SDFGState, data, dtypes
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.analysis import cfg
from dace.sdfg import infer_types
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes import analysis as ap

PROTECTED_NAMES = {'__pystate'}  #: A set of names that are not allowed to be erased


@dataclass(unsafe_hash=True)
class DeadDataflowElimination(ppl.Pass):
    """
    Removes unused computations from SDFG states.
    Traverses the graph backwards, removing any computations that result in transient descriptors
    that are not used again. Removal propagates through scopes (maps), tasklets, and optionally library nodes.
    """
    skip_library_nodes: bool = False  #: If True, does not remove library nodes if their results are unused. Otherwise removes library nodes without side effects.
    remove_persistent_memory: bool = False  #: If True, marks code with Persistent allocation lifetime as dead

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If dataflow or states changed, new dead code may be exposed
        return modified & (ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.States)

    def depends_on(self) -> Set[Type[ppl.Pass]]:
        return {ap.StateReachability, ap.AccessSets}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[SDFGState, Set[str]]]:
        """
        Removes unreachable dataflow throughout SDFG states.
        
        :param sdfg: The SDFG to modify.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: A dictionary mapping states to removed data descriptor names, or None if nothing changed.
        """
        # Depends on the following analysis passes:
        #  * State reachability
        #  * Read/write access sets per state
        reachable: Dict[SDFGState, Set[SDFGState]] = pipeline_results['StateReachability'][sdfg.sdfg_id]
        access_sets: Dict[SDFGState, Tuple[Set[str], Set[str]]] = pipeline_results['AccessSets'][sdfg.sdfg_id]
        result: Dict[SDFGState, Set[str]] = defaultdict(set)

        # Traverse SDFG backwards
        try:
            state_order = list(cfg.stateorder_topological_sort(sdfg))
        except KeyError:
            return None
        for state in reversed(state_order):
            #############################################
            # Analysis
            #############################################

            # Compute states where memory will no longer be read
            writes = access_sets[state][1]
            descendants = reachable[state]
            descendant_reads = set().union(*(access_sets[succ][0] for succ in descendants))
            no_longer_used: Set[str] = set(data for data in writes if data not in descendant_reads)

            # Compute dead nodes
            dead_nodes: List[nodes.Node] = []

            # Propagate deadness backwards within a state
            for node in sdutil.dfs_topological_sort(state, reverse=True):
                if self._is_node_dead(node, sdfg, state, dead_nodes, no_longer_used, access_sets[state]):
                    dead_nodes.append(node)

            # Scope exit nodes are only dead if their corresponding entry nodes are
            live_nodes = set()
            for node in dead_nodes:
                if isinstance(node, nodes.ExitNode) and state.entry_node(node) not in dead_nodes:
                    live_nodes.add(node)
            dead_nodes = dtypes.deduplicate([n for n in dead_nodes if n not in live_nodes])

            if not dead_nodes:
                continue

            # Remove nodes while preserving scopes
            scopes_to_reconnect: Set[nodes.Node] = set()
            for node in state.nodes():
                # Look for scope exits that will be disconnected
                if isinstance(node, nodes.ExitNode) and node not in dead_nodes:
                    if any(n in dead_nodes for n in state.predecessors(node)):
                        scopes_to_reconnect.add(node)

            # Two types of scope disconnections may occur:
            # 1. Two scope exits will no longer be connected
            # 2. A predecessor of dead nodes is in a scope and not connected to its exit
            # Case (1) is taken care of by ``remove_memlet_path``
            # Case (2) is handled below
            # Reconnect scopes
            if scopes_to_reconnect:
                schildren = state.scope_children()
                for exit_node in scopes_to_reconnect:
                    entry_node = state.entry_node(exit_node)
                    for node in schildren[entry_node]:
                        if node is exit_node:
                            continue
                        if isinstance(node, nodes.EntryNode):
                            node = state.exit_node(node)
                        # If node will be disconnected from exit node, add an empty memlet
                        if all(succ in dead_nodes for succ in state.successors(node)):
                            state.add_nedge(node, exit_node, Memlet())

            #############################################
            # Removal
            #############################################
            predecessor_nsdfgs: Dict[nodes.NestedSDFG, Set[str]] = defaultdict(set)
            for node in dead_nodes:
                # Remove memlet paths and connectors pertaining to dead nodes
                try:
                    for e in state.in_edges(node):
                        mtree = state.memlet_tree(e)
                        for leaf in mtree.leaves():
                            # Keep track of predecessors of removed nodes for connector pruning
                            if isinstance(leaf.src, nodes.NestedSDFG):
                                if not leaf.data.is_empty():
                                    predecessor_nsdfgs[leaf.src].add(leaf.src_conn)

                            # Pruning connectors on tasklets sometimes needs to change their code
                            elif (isinstance(leaf.src, nodes.Tasklet) and leaf.src.code.language != dtypes.Language.Python):
                                if leaf.src.code.language == dtypes.Language.CPP:
                                    ctype = infer_types.infer_out_connector_type(sdfg, state, leaf.src, leaf.src_conn)
                                    if ctype is None:
                                        raise NotImplementedError(f'Cannot eliminate dead connector "{leaf.src_conn}" on '
                                                                'tasklet due to connector type inference failure.')
                                    # Add definition
                                    leaf.src.code.code = f'{ctype.as_arg(leaf.src_conn)};\n' + leaf.src.code.code
                                else:
                                    raise NotImplementedError(f'Cannot eliminate dead connector "{leaf.src_conn}" on '
                                                            'tasklet due to its code language.')
                            state.remove_memlet_path(leaf)

                    # Remove the node itself as necessary
                    state.remove_node(node)
                except KeyError:  # Node already removed
                    continue

            result[state].update(dead_nodes)

            # Remove isolated access nodes after elimination
            access_nodes = set(state.data_nodes())
            for node in access_nodes:
                if state.degree(node) == 0:
                    state.remove_node(node)
                    result[state].add(node)

            # Prune now-dead connectors
            for node, dead_conns in predecessor_nsdfgs.items():
                for conn in dead_conns:
                    # If removed connector belonged to a nested SDFG, and no other input connector shares name,
                    # make nested data transient (dead dataflow elimination would remove internally as necessary)
                    if conn not in node.in_connectors:
                        node.sdfg.arrays[conn].transient = True

            # Update read sets for the predecessor states to reuse
            access_nodes -= result[state]
            access_node_names = set(n.data for n in access_nodes if state.out_degree(n) > 0)
            access_sets[state] = (access_node_names, access_sets[state][1])

        return result or None

    def report(self, pass_retval: Dict[SDFGState, Set[str]]) -> str:
        n = sum(len(v) for v in pass_retval.values())
        return f'Eliminated {n} nodes in {len(pass_retval)} states: {pass_retval}'

    def _is_node_dead(self, node: nodes.Node, sdfg: SDFG, state: SDFGState, dead_nodes: Set[nodes.Node],
                      no_longer_used: Set[str], access_set: Tuple[Set[str], Set[str]]) -> bool:
        # Conditions for dead node:
        # * All successors are dead
        # * Access node that can no longer be read
        #   * Sub-case: Persistent allocation lifetime may block this, if configured
        # * Dead tasklets may not contain any callbacks
        # * Library nodes being dead depend on configuration (and side-effects)

        # Check that all successors are dead
        if any(succ not in dead_nodes for succ in state.successors(node)):
            return False

        # Determine on a case-by-case basis
        if isinstance(node, nodes.LibraryNode):
            # Library nodes must not have any side effects to be considered dead
            if self.skip_library_nodes:
                return False
            return not node.has_side_effects
        elif isinstance(node, nodes.Tasklet):
            # If a tasklet has any callbacks, mark as "live" due to potential side effects
            return not node.has_side_effects(sdfg)

        elif isinstance(node, nodes.AccessNode):
            # Search for names that are disallowed to remove
            if node.data in PROTECTED_NAMES:
                return False

            desc = sdfg.arrays[node.data]

            # If data descriptor is global, it cannot be removed
            if not desc.transient:
                return False

            # If access node is persistent, mark as dead only if self.remove_persistent_memory is set
            if not self.remove_persistent_memory:
                if desc.lifetime == dtypes.AllocationLifetime.Persistent:
                    return False

            # If data will be used later, cannot remove
            if node.data not in no_longer_used:
                return False

            # Check incoming edges
            for e in state.in_edges(node):
                for l in state.memlet_tree(e).leaves():
                    # If data is connected to a side-effect tasklet/library node, cannot remove
                    if _has_side_effects(l.src, sdfg):
                        return False

                    # If data is connected to a nested SDFG as an input/output, do not remove
                    if (isinstance(l.src, nodes.NestedSDFG)
                            and any(ie.data.data == node.data for ie in state.in_edges(l.src))):
                        return False

            # If it is a stream and is read somewhere in the state, it may be popped after pushing
            if isinstance(desc, data.Stream) and node.data in access_set[0]:
                return False

        # Any other case can be marked as dead
        return True


def _has_side_effects(node, sdfg):
    try:
        return node.has_side_effects(sdfg)
    except (AttributeError, TypeError):
        try:
            return node.has_side_effects
        except AttributeError:
            return False
