# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import ast
from dataclasses import dataclass
from dace.frontend.python import astutils
from dace.sdfg.analysis import cfg
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg import nodes, utils as sdutil
from dace.transformation import pass_pipeline as ppl, transformation
from dace.cli.progress import optional_progressbar
from dace import data, SDFG, SDFGState, dtypes, symbolic, properties
from typing import Any, Dict, Set, Optional, Tuple


class _UnknownValue:
    """ A helper class that indicates a symbol value is ambiguous. """
    pass


@dataclass(unsafe_hash=True)
@properties.make_properties
@transformation.single_level_sdfg_only
class ConstantPropagation(ppl.Pass):
    """
    Propagates constants and symbols that were assigned to one value forward through the SDFG, reducing
    the number of overall symbols.
    """

    CATEGORY: str = 'Simplification'

    recursive = properties.Property(dtype=bool, default=True, desc='Propagate recursively through nested SDFGs')
    progress = properties.Property(dtype=bool, default=None, allow_none=True, desc='Show progress')

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

    def apply_pass(self, sdfg: SDFG, _, initial_symbols: Optional[Dict[str, Any]] = None) -> Optional[Set[str]]:
        """
        Propagates constants throughout the SDFG.

        :param sdfg: The SDFG to modify.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :param initial_symbols: If not None, sets values of initial symbols.
        :return: A set of propagated constants, or None if nothing was changed.
        """
        initial_symbols = initial_symbols or {}

        # Early exit if no constants can be propagated
        if not initial_symbols and not self.should_apply(sdfg):
            result = {}
        else:
            # Trace all constants and symbols through states
            per_state_constants: Dict[SDFGState, Dict[str, Any]] = self.collect_constants(sdfg, initial_symbols)

            # Keep track of replaced and ambiguous symbols
            symbols_replaced: Dict[str, Any] = {}
            remaining_unknowns: Set[str] = set()

            # Collect symbols from symbol-dependent data descriptors
            # If there can be multiple values over the SDFG, the symbols are not propagated
            desc_symbols, multivalue_desc_symbols = self._find_desc_symbols(sdfg, per_state_constants)

            # Replace constants per state
            for state, mapping in optional_progressbar(per_state_constants.items(),
                                                       'Propagating constants',
                                                       n=len(per_state_constants),
                                                       progress=self.progress):
                remaining_unknowns.update(
                    {k
                     for k, v in mapping.items() if v is _UnknownValue or k in multivalue_desc_symbols})
                mapping = {
                    k: v
                    for k, v in mapping.items() if v is not _UnknownValue and k not in multivalue_desc_symbols
                }
                if not mapping:
                    continue

                # Update replaced symbols for later replacements
                symbols_replaced.update(mapping)

                # Replace in state contents
                state.replace_dict(mapping)
                # Replace in outgoing edges as well
                for e in sdfg.out_edges(state):
                    e.data.replace_dict(mapping, replace_keys=False)

            # Gather initial propagated symbols
            result = {k: v for k, v in symbols_replaced.items() if k not in remaining_unknowns}

            # Remove single-valued symbols from data descriptors (e.g., symbolic array size)
            sdfg.replace_dict({k: v
                               for k, v in result.items() if k in desc_symbols},
                              replace_in_graph=False,
                              replace_keys=False)

            # Remove constant symbol assignments in interstate edges
            for edge in sdfg.edges():
                intersection = result & edge.data.assignments.keys()
                for sym in intersection:
                    del edge.data.assignments[sym]

            # If symbols are never unknown any longer, remove from SDFG
            fsyms = sdfg.used_symbols(all_symbols=False)
            result = {k: v for k, v in result.items() if k not in fsyms}
            for sym in result:
                if sym in sdfg.symbols:
                    # Remove from symbol repository and nested SDFG symbol mapping
                    sdfg.remove_symbol(sym)

        result = set(result.keys())

        if self.recursive:
            # Change result to set of tuples
            sid = sdfg.cfg_id
            result = set((sid, sym) for sym in result)

            for state in sdfg.nodes():
                for node in state.nodes():
                    if isinstance(node, nodes.NestedSDFG):
                        nested_id = node.sdfg.cfg_id
                        const_syms = {k: v for k, v in node.symbol_mapping.items() if not symbolic.issymbolic(v)}
                        internal = self.apply_pass(node.sdfg, _, const_syms)
                        if internal:
                            for nid, removed in internal:
                                result.add((nid, removed))
                                # Remove symbol mapping if constant was completely propagated
                                if nid == nested_id and removed in node.symbol_mapping:
                                    del node.symbol_mapping[removed]

        # Return result
        if not result:
            return None
        return result

    def report(self, pass_retval: Set[str]) -> str:
        return f'Propagated {len(pass_retval)} constants.'

    def collect_constants(self,
                          sdfg: SDFG,
                          initial_symbols: Optional[Dict[str, Any]] = None) -> Dict[SDFGState, Dict[str, Any]]:
        """
        Finds all constants and constant-assigned symbols in the SDFG for each state.

        :param sdfg: The SDFG to traverse.
        :param initial_symbols: If not None, sets values of initial symbols.
        :return: A dictionary mapping an SDFG state to a mapping of constants and their corresponding values.
        """
        arrays: Set[str] = set(sdfg.arrays.keys() | sdfg.constants_prop.keys())
        result: Dict[SDFGState, Dict[str, Any]] = {}

        # Add nested data to arrays
        def _add_nested_datanames(name: str, desc: data.Structure):
            for k, v in desc.members.items():
                if isinstance(v, data.Structure):
                    _add_nested_datanames(f'{name}.{k}', v)
                elif isinstance(v, data.ContainerArray):
                    # TODO: How are we handling this?
                    pass
                arrays.add(f'{name}.{k}')

        for name, desc in sdfg.arrays.items():
            if isinstance(desc, data.Structure):
                _add_nested_datanames(name, desc)

        # Process:
        # * Collect constants in topologically ordered states
        # * Propagate forward symbols forward and edge assignments
        #   * If value is ambiguous (not the same), set value to UNKNOWN
        # * Repeat until no update is performed

        start_state = sdfg.start_state
        if initial_symbols:
            result[start_state] = {}
            result[start_state].update(initial_symbols)

        redo = True
        while redo:
            redo = False
            # Traverse SDFG topologically
            for state in optional_progressbar(cfg.blockorder_topological_sort(sdfg), 'Collecting constants',
                                              sdfg.number_of_nodes(), self.progress):

                # Get predecessors
                in_edges = sdfg.in_edges(state)
                assignments = {}
                for edge in in_edges:
                    # If source was already visited, use its propagated constants
                    constants: Dict[str, Any] = {}
                    if edge.src in result:
                        constants.update(result[edge.src])

                    # Update constants with incoming edge
                    self._propagate(constants, self._data_independent_assignments(edge.data, arrays))

                    for aname, aval in constants.items():
                        # If something was assigned more than once (to a different value), it's not a constant
                        # If a symbol appearing in the replacing expression of a constant is modified,
                        # the constant is not valid anymore
                        if ((aname in assignments and aval != assignments[aname]) or
                                symbolic.free_symbols_and_functions(aval) & edge.data.assignments.keys()):
                            assignments[aname] = _UnknownValue
                        else:
                            assignments[aname] = aval

                for edge in sdfg.out_edges(state):
                    for aname, aval in assignments.items():
                        # If the specific replacement would result in the value
                        # being both used and reassigned on the same inter-state
                        # edge, remove it from consideration.
                        replacements = symbolic.free_symbols_and_functions(aval)
                        used_in_assignments = {
                            k
                            for k, v in edge.data.assignments.items() if aname in symbolic.free_symbols_and_functions(v)
                        }
                        reassignments = replacements & edge.data.assignments.keys()
                        if reassignments and (used_in_assignments - reassignments):
                            assignments[aname] = _UnknownValue

                if state not in result:  # Condition may evaluate to False when state is the start-state
                    result[state] = {}
                redo |= self._propagate(result[state], assignments)

        return result

    def _find_desc_symbols(self, sdfg: SDFG, constants: Dict[SDFGState, Dict[str, Any]]) -> Tuple[Set[str], Set[str]]:
        """
        Finds constant symbols that data descriptors (e.g., arrays) depend on.

        :param sdfg: The SDFG to scan.
        :param constants: Constant symbols found in ``collect_constants``.
        :return: A tuple of two sets: (all descriptor-related symbols, symbols that take multiple values).
        """
        symbols_in_data: Set[str] = set()
        symbols_in_data_with_multiple_values: Set[str] = set()
        for arr in sdfg.arrays.values():
            symbols_in_data |= set(map(str, arr.free_symbols))

        values: Dict[str, Any] = {}
        for mapping in constants.values():
            # Symbols that data descriptors depend on must receive a single value, otherwise mark as unknown
            for k, v in mapping.items():
                if k not in symbols_in_data:
                    continue
                if k in values:
                    if v is _UnknownValue or v != values[k]:
                        symbols_in_data_with_multiple_values.add(k)
                else:
                    if v is _UnknownValue:
                        symbols_in_data_with_multiple_values.add(k)
                    values[k] = v

        return symbols_in_data, symbols_in_data_with_multiple_values

    def _propagate(self, symbols: Dict[str, Any], new_symbols: Dict[str, Any]) -> bool:
        """
        Updates symbols dictionary in-place with new symbols, propagating existing ones within.
        
        :param symbols: The symbols dictionary to update.
        :param new_symbols: The new symbols to include (and propagate ``symbols`` into).
        :return: True if symbols was modified, False otherwise
        """
        if not new_symbols:
            return False

        repl = {k: v for k, v in symbols.items() if v is not _UnknownValue}

        # Replace interstate edge assignment (which is Python code)
        def _replace_assignment(v, assignment):
            # Special cases to speed up replacement
            v = str(v)
            if not v:
                return v
            if dtypes.validate_name(v) and v in repl:
                return repl[v]

            vast = ast.parse(v)
            replacer = astutils.ASTFindReplace(repl, assignment)
            try:
                vast = replacer.visit(vast)
            except astutils.NameFound:
                # If any of the unknowns were found in the expression, mark assignment as unknown
                return _UnknownValue
            return astutils.unparse(vast)

        # Update results with values of other propagated symbols
        propagated_symbols = {
            k: _replace_assignment(v, {k}) if v is not _UnknownValue else _UnknownValue
            for k, v in new_symbols.items()
        }
        original_symbols = symbols.copy()
        symbols.update(propagated_symbols)

        return original_symbols != symbols

    def _data_independent_assignments(self, edge: InterstateEdge, arrays: Set[str]) -> Dict[str, Any]:
        """
        Return symbol assignments that only depend on other symbols and constants, rather than data descriptors.
        """
        return {
            k: v if (not (symbolic.free_symbols_and_functions(v) & arrays)) else _UnknownValue
            for k, v in edge.assignments.items()
        }
