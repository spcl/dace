# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

from collections import defaultdict
from dataclasses import dataclass
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes import analysis as ap
from dace.sdfg.analysis import cfg
from dace import SDFG, SDFGState, dtypes
from dace.sdfg import nodes, utils as sdutil
from typing import Any, Dict, Set, Optional, Tuple, Type


@dataclass(unsafe_hash=True)
class DeadDataflowElimination(ppl.Pass):
    """
    Removes unused computations from SDFG states.
    Traverses the graph backwards, removing any computations that result in transient descriptors
    that are not used again. Removal propagates through scopes (maps), tasklets, and optionally library nodes.
    """
    skip_library_nodes: bool = False  #: If True, does not remove library nodes if their results are unused. Otherwise removes library nodes without side effects.
    remove_persistent_memory: bool = True  #: If True, marks code with Persistent allocation lifetime as dead

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
        reachable: Dict[SDFGState, Set[SDFGState]] = pipeline_results['StateReachability']
        access_sets: Dict[SDFGState, Tuple[Set[str], Set[str]]] = pipeline_results['AccessSets']
        result: Dict[SDFGState, Set[str]] = defaultdict(set)

        # TODO: Compute states where memory will no longer be read
        no_longer_used = set()

        # Traverse SDFG backwards
        for state in reversed(list(cfg.stateorder_topological_sort(sdfg))):
            dead_nodes: Set[nodes.Node] = set()

            # Propagate deadness backwards within a state
            for node in sdutil.dfs_topological_sort(state, reverse=True):
                if self._is_node_dead(node, sdfg, state, dead_nodes, no_longer_used):
                    dead_nodes.add(node)

            # Scope exit nodes are only dead if their corresponding entry nodes are
            for node in dead_nodes:
                if isinstance(node, nodes.ExitNode) and state.entry_node(node) not in dead_nodes:
                    dead_nodes.remove(node)

            if not dead_nodes:
                continue

            # Remove nodes while preserving scopes
            scopes_to_reconnect: Set[nodes.Node] = set()
            for node in state.nodes():
                # Look for scope exits that will be disconnected
                if isinstance(node, nodes.ExitNode) and node not in dead_nodes:
                    if any(n in dead_nodes for n in state.predecessors(node)):
                        scopes_to_reconnect.add(node)

            # TODO: Reconnect scopes
            if scopes_to_reconnect:
                raise NotImplementedError

            state.remove_nodes_from(dead_nodes)
            result[state].update(dead_nodes)

            # Remove isolated access nodes after elimination
            for node in list(state.data_nodes()):
                if state.degree(node) == 0:
                    state.remove_node(node)
                    result[state].add(node)

        return result or None

    def _is_node_dead(self, node: nodes.Node, sdfg: SDFG, state: SDFGState, dead_nodes: Set[nodes.Node],
                      no_longer_used: Set[str]) -> bool:
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
            if node.side_effects is True:
                return False
            elif node.side_effects is None:
                # TODO(later): If a tasklet has any callbacks, mark as "live" due to potential side effects
                pass

        elif isinstance(node, nodes.AccessNode):
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

        # Any other case can be marked as dead
        return True
