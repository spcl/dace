# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg import utils as sdutil
from dace.transformation import pass_pipeline as ppl
from dace import SDFG, SDFGState, symbolic
from typing import Any, Dict, Set, Optional

from dace.transformation.passes.scalar_to_symbol import ScalarToSymbolPromotion


class _UnknownValue:
    """ A helper class that indicates a symbol value is ambiguous. """
    pass


class ConstantPropagation(ppl.Pass):
    """
    Propagates constants and symbols that were assigned to one value forward through the SDFG, reducing
    the number of overall symbols.
    """
    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Symbols | ppl.Modifies.Edges | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified != ppl.Modifies.Nothing

    def should_apply(self, sdfg: SDFG) -> bool:
        """
        Fast check (O(m)) whether the pass should early-exit without traversing the SDFG.
        """
        for edge in sdfg.edges():
            # If there are no assignments, there are no constants to propagate
            if len(edge.data.assignments) == 0:
                continue
            # If no assignment assigns a constant to a symbol, no constants can be propagated
            if any(not symbolic.issymbolic(aval) for aval in edge.data.assignments.values()):
                return True

        return False

    def apply_pass(self, sdfg: SDFG, _) -> Optional[Set[str]]:
        """
        Propagates constants throughout the SDFG.
        :param sdfg: The SDFG to modify.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: A set of propagated constants, or None if nothing was changed.
        """
        # Early exit if no constants can be propagated
        if not self.should_apply(sdfg):
            return None

        # Trace all constants and symbols through states
        per_state_constants: Dict[SDFGState, Dict[str, Any]] = self.collect_constants(sdfg)

        # Keep track of replaced and ambiguous symbols
        symbols_replaced: Set[str] = set()
        remaining_unknowns: Set[str] = set()

        # Replace constants per state
        for state, mapping in per_state_constants.items():
            remaining_unknowns.update({k for k, v in mapping.items() if v is _UnknownValue})
            mapping = {k: v for k, v in mapping.items() if v is not _UnknownValue}
            symbols_replaced.update(mapping.keys())

            # TODO: Remove symbol assignments in interstate edges

            # Replace in state contents
            state.replace_dict(mapping)
            # Replace in outgoing edges as well
            for e in sdfg.out_edges(state):
                e.data.replace_dict(mapping)

        # If symbols are never unknown any longer, remove from SDFG
        result = (symbols_replaced - remaining_unknowns)
        for sym in result:
            sdfg.remove_symbol(sym)

        # Return result
        if not result:
            return None
        return result

    def collect_constants(self, sdfg: SDFG) -> Dict[SDFGState, Dict[str, Any]]:
        """
        Finds all constants and constant-assigned symbols in the SDFG for each state.
        :param sdfg: The SDFG to traverse.
        :return: A dictionary mapping an SDFG state to a mapping of constants and their corresponding values.
        """
        arrays: Set[str] = set(sdfg.arrays.keys() | sdfg.constants_prop.keys())
        result: Dict[SDFGState, Dict[str, Any]] = {}

        # Process:
        # * Collect constants in topologically ordered states
        # * If unvisited state has one incoming edge - propagate symbols forward and edge assignments
        # * If unvisited state has more than one incoming edge, consider all paths (use reverse DFS on unvisited paths)
        #   * If value is ambiguous (not the same), set value to UNKNOWN

        # Traverse SDFG topologically
        for state in sdfg.topological_sort(sdfg.start_state):
            if state in result:
                continue
            result[state] = {}

            # Get predecessors
            in_edges = sdfg.in_edges(state)
            if len(in_edges) == 1:  # Special case, propagate as-is
                # First the prior state
                result[state].update(result[in_edges[0].src])

                # Then assignments on the incoming edge
                result[state].update(self._data_independent_assignments(in_edges[0].data, arrays))
                continue

            # More than one incoming edge: may require reversed traversal
            assignments = {}
            for edge in in_edges:
                # If source was already visited, use its propagated constants
                constants: Dict[str, Any]
                if edge.src in result:
                    constants = result[edge.src]
                else:  # Otherwise, reverse DFS to find constants until a visited state
                    constants = self._constants_from_unvisited_state(sdfg, edge.src, arrays, result)

                # Update constants with incoming edge
                constants.update(self._data_independent_assignments(edge.data, arrays))

                for aname, aval in constants.items():
                    # If something was assigned more than once (to a different value), it's not a constant
                    if aname in assignments and aval != assignments[aname]:
                        assignments[aname] = _UnknownValue
                    else:
                        assignments[aname] = aval

            result[state].update(assignments)

            # TODO: Update results with values of other propagated symbols

        return result

    def _data_independent_assignments(self, edge: InterstateEdge, arrays: Set[str]) -> Dict[str, Any]:
        """
        Return symbol assignments that only depend on other symbols and constants, rather than data descriptors.
        """
        return {k: v for k, v in edge.assignments.items() if not symbolic.free_symbols_and_functions(v) & arrays}

    def _constants_from_unvisited_state(self, sdfg: SDFG, state: SDFGState, arrays: Set[str],
                                        existing_constants: Dict[SDFGState, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collects constants from an unvisited state, traversing backwards until reaching states that do have
        collected constants.
        """
        result: Dict[str, Any] = {}

        def _update_if_new(existing: Dict[Any, Any], new: Dict[Any, Any]) -> None:
            for k, v in new.items():
                if k not in existing:
                    existing[k] = v

        for parent, node in sdutil.dfs_conditional(sdfg,
                                                   sources=[state],
                                                   reverse=True,
                                                   condition=lambda p, c: c not in existing_constants,
                                                   yield_parent=True):
            # Skip first node
            if parent is None:
                continue

            # Get connecting edge (reversed)
            edge = sdfg.edges_between(node, parent)[0]

            # If node already has propagated constants, update dictionary and stop traversal
            if node in existing_constants:
                self._data_independent_assignments(edge.data, arrays)
                _update_if_new(result, existing_constants[node])
                continue

        return result
