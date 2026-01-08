# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import ast
from dataclasses import dataclass
from dace.frontend.python import astutils
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg import nodes
from dace.sdfg.state import AbstractControlFlowRegion, ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation
from dace.cli.progress import optional_progressbar
from dace import data, SDFG, SDFGState, dtypes, symbolic, properties
from typing import Any, Dict, Set, Optional, Tuple


class _UnknownValue:
    """ A helper class that indicates a symbol value is ambiguous. """
    pass


ConstsT = Dict[str, Any]
BlockConstsT = Dict[ControlFlowBlock, ConstsT]


@dataclass(unsafe_hash=True)
@properties.make_properties
@transformation.explicit_cf_compatible
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
        for edge in sdfg.all_interstate_edges():
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
            arrays: Set[str] = set(sdfg.arrays.keys() | sdfg.constants_prop.keys())

            # Add nested data to arrays
            def _add_nested_datanames(name: str, desc: data.Structure):
                for k, v in desc.members.items():
                    if isinstance(v, data.Structure):
                        _add_nested_datanames(f'{name}.{k}', v)
                    elif isinstance(v, data.ContainerArray):
                        pass
                    arrays.add(f'{name}.{k}')

            for name, desc in sdfg.arrays.items():
                if isinstance(desc, data.Structure):
                    _add_nested_datanames(name, desc)

            # Trace all constants and symbols through blocks
            in_constants: BlockConstsT = {sdfg: initial_symbols}
            pre_constants: BlockConstsT = {}
            post_constants: BlockConstsT = {}
            out_constants: BlockConstsT = {}
            self._collect_constants_for_region(sdfg, arrays, in_constants, pre_constants, post_constants, out_constants)

            # Keep track of replaced and ambiguous symbols
            symbols_replaced: Dict[str, Any] = {}
            remaining_unknowns: Set[str] = set()

            # Collect symbols from symbol-dependent data descriptors
            # If there can be multiple values over the SDFG, the symbols are not propagated
            desc_symbols, multivalue_desc_symbols = self._find_desc_symbols(sdfg, in_constants)

            # Replace constants per state
            for block, mapping in optional_progressbar(in_constants.items(),
                                                       'Propagating constants',
                                                       n=len(in_constants),
                                                       progress=self.progress):
                if block is sdfg:
                    continue

                remaining_unknowns.update(
                    {k
                     for k, v in mapping.items() if v is _UnknownValue or k in multivalue_desc_symbols})
                mapping = {
                    k: v
                    for k, v in mapping.items() if v is not _UnknownValue and k not in multivalue_desc_symbols
                }
                out_mapping = {
                    k: v
                    for k, v in out_constants[block].items()
                    if v is not _UnknownValue and k not in multivalue_desc_symbols
                }

                if mapping:
                    # Update replaced symbols for later replacements
                    symbols_replaced.update(mapping)

                    if isinstance(block, SDFGState):
                        # Replace in state contents
                        block.replace_dict(mapping)
                    elif isinstance(block, AbstractControlFlowRegion):
                        block.replace_dict(mapping, replace_in_graph=False, replace_keys=False)

                if out_mapping:
                    # Replace in outgoing edges as well
                    for e in block.parent_graph.out_edges(block):
                        e.data.replace_dict(out_mapping, replace_keys=False)

                if isinstance(block, LoopRegion):
                    self._propagate_loop(block, post_constants, multivalue_desc_symbols)

            # Gather initial propagated symbols
            result = {k: v for k, v in symbols_replaced.items() if k not in remaining_unknowns}

            # Remove single-valued symbols from data descriptors (e.g., symbolic array size)
            sdfg.replace_dict({
                k: v
                for k, v in result.items() if k in desc_symbols
            },
                              replace_in_graph=False,
                              replace_keys=False)

            # Remove constant symbol assignments in interstate edges
            for edge in sdfg.all_interstate_edges():
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

            for state in sdfg.states():
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

    def _propagate_loop(self, loop: LoopRegion, post_constants: BlockConstsT,
                        multivalue_desc_symbols: Set[str]) -> None:
        if loop in post_constants and post_constants[loop] is not None:
            if loop.update_statement is not None and (loop.inverted and loop.update_before_condition
                                                      or not loop.inverted):
                # Replace the RHS of the update experssion
                post_mapping = {
                    k: v
                    for k, v in post_constants[loop].items()
                    if v is not _UnknownValue and k not in multivalue_desc_symbols
                }
                update_stmt = loop.update_statement
                updates = update_stmt.code if isinstance(update_stmt.code, list) else [update_stmt.code]
                for update in updates:
                    astutils.ASTReplaceAssignmentRHS(post_mapping).visit(update)
                loop.update_statement.code = updates

    def _collect_constants_for_conditional(self, conditional: ConditionalBlock, arrays: Set[str],
                                           in_const_dict: BlockConstsT, pre_const_dict: BlockConstsT,
                                           post_const_dict: BlockConstsT, out_const_dict: BlockConstsT) -> None:
        """
        Collect the constants for and inside of a conditional region.
        Recursively collects constants inside of nested regions.

        :param conditional: The conditional region to traverse.
        :param arrays: A set of data descriptors in the SDFG.
        :param in_const_dict: Dictionary mapping each control flow block to the set of constants observed right before
                              the block is executed. Populated by this function.
        :param pre_const_dict: Dictionary mapping each control flow block to the set of constants observed before its
                               contents are executed. Populated by this function.
        :param post_const_dict: Dictionary mapping each control flow block to the set of constants observed after its
                                contents are executed. Populated by this function.
        :param out_const_dict: Dictionary mapping each control flow block to the set of constants observed right after
                               the block is executed. Populated by this function.
        """
        in_consts = in_const_dict[conditional]
        # First, collect all constants for each of the branches.
        for _, branch in conditional.branches:
            in_const_dict[branch] = in_consts
            self._collect_constants_for_region(branch, arrays, in_const_dict, pre_const_dict, post_const_dict,
                                               out_const_dict)
        # Second, determine the 'post constants' (constants at the end of the conditional region) as an intersection
        # between the output constants of each of the branches.
        post_consts = {}
        post_consts_intersection = None
        has_else = False
        for cond, branch in conditional.branches:
            if post_consts_intersection is None:
                post_consts_intersection = set(out_const_dict[branch].keys())
            else:
                post_consts_intersection &= set(out_const_dict[branch].keys())
            if cond is None:
                has_else = True
        for _, branch in conditional.branches:
            for k, v in out_const_dict[branch].items():
                if k in post_consts_intersection:
                    if k not in post_consts:
                        post_consts[k] = v
                    elif post_consts[k] != _UnknownValue and post_consts[k] != v:
                        post_consts[k] = _UnknownValue
                else:
                    post_consts[k] = _UnknownValue
        post_const_dict[conditional] = post_consts

        # Finally, determine the conditional region's output constants.
        if has_else:
            # If there is an else, at least one branch will certainly be taken, so the output constants are the region's
            # post constants.
            out_const_dict[conditional] = post_consts
        else:
            # No else branch is present, so it is possible that no branch is executed. In this case the out constants
            # are the intersection between the in constants and the post constants.
            out_consts = in_consts.copy()
            for k, v in post_consts.items():
                if k not in out_consts:
                    out_consts[k] = _UnknownValue
                elif out_consts[k] != _UnknownValue and out_consts[k] != v:
                    out_consts[k] = _UnknownValue
            out_const_dict[conditional] = out_consts

    def _assignments_in_loop(self, loop: LoopRegion) -> Set[str]:
        assignments_within = set()
        for e in loop.all_interstate_edges():
            for k in e.data.assignments.keys():
                assignments_within.add(k)
        if loop.loop_variable is not None:
            assignments_within.add(loop.loop_variable)
        return assignments_within

    def _collect_constants_for_region(self, cfg: ControlFlowRegion, arrays: Set[str], in_const_dict: BlockConstsT,
                                      pre_const_dict: BlockConstsT, post_const_dict: BlockConstsT,
                                      out_const_dict: BlockConstsT) -> None:
        """
        Finds all constants and constant-assigned symbols in the control flow graph for each block.
        Recursively collects constants for nested control flow regions.

        :param cfg: The CFG to traverse.
        :param arrays: A set of data descriptors in the SDFG.
        :param in_const_dict: Dictionary mapping each control flow block to the set of constants observed right before
                              the block is executed. Populated by this function.
        :param pre_const_dict: Dictionary mapping each control flow block to the set of constants observed before its
                               contents are executed. Populated by this function.
        :param post_const_dict: Dictionary mapping each control flow block to the set of constants observed after its
                                contents are executed. Populated by this function.
        :param out_const_dict: Dictionary mapping each control flow block to the set of constants observed right after
                               the block is executed. Populated by this function.
        """
        # Given the 'in constants', i.e., the constants for before the current region is executed, compute the 'pre
        # constants', i.e., the set of constants seen inside the region when executing.
        if cfg in in_const_dict:
            in_const = in_const_dict[cfg]
            if isinstance(cfg, LoopRegion):
                # In the case of a loop, the 'pre constants' are equivalent to the 'in constants', with the exception
                # of values that may at any point be re-assigned inside the loop, since that assignment would carry over
                # into the next iteration (including increments to the loop variable, if present).
                assigned_in_loop = self._assignments_in_loop(cfg)
                pre_const = {k: (v if k not in assigned_in_loop else _UnknownValue) for k, v in in_const.items()}
            else:
                # In any other case, the 'pre constants' are equivalent to the 'in constants'.
                pre_const = {}
                pre_const.update(in_const)
        else:
            # No 'in constants' for the current region - so initialize to nothing.
            pre_const = {}
            pre_const_dict[cfg] = pre_const
            in_const = {}
        pre_const_dict[cfg] = pre_const

        # Process:
        # * Collect constants in topologically ordered blocks
        # * Propagate forward symbols forward and edge assignments
        #   * If value is ambiguous (not the same), set value to UNKNOWN
        # * Repeat until no update is performed

        start_block = cfg.start_block
        if pre_const:
            in_const_dict[start_block] = {}
            in_const_dict[start_block].update(pre_const)

        redo = True
        while redo:
            redo = False
            # Traverse CFG topologically
            for block in optional_progressbar(cfg_analysis.blockorder_topological_sort(cfg, recursive=False),
                                              'Collecting constants for ' + cfg.label, cfg.number_of_nodes(),
                                              self.progress):
                # Get predecessors
                in_edges = cfg.in_edges(block)
                assignments = {}
                for edge in in_edges:
                    # If source was already visited, use its propagated constants
                    constants: Dict[str, Any] = {}
                    if edge.src in out_const_dict:
                        constants.update(out_const_dict[edge.src])

                    # Update constants with incoming edge
                    self._propagate(constants, self._data_independent_assignments(edge.data, arrays))

                    for aname, aval in constants.items():
                        # If something was assigned more than once (to a different value), it's not a constant
                        # If a symbol appearing in the replacing expression of a constant is modified,
                        # the constant is not valid anymore
                        if ((aname in assignments and aval != assignments[aname])
                                or symbolic.free_symbols_and_functions(aval) & edge.data.assignments.keys()):
                            assignments[aname] = _UnknownValue
                        else:
                            assignments[aname] = aval

                for edge in cfg.out_edges(block):
                    for aname, aval in assignments.items():
                        # If the specific replacement would result in the value being both used and reassigned on the
                        # same inter-state edge, remove it from consideration.
                        replacements = symbolic.free_symbols_and_functions(aval)
                        used_in_assignments = {
                            k
                            for k, v in edge.data.assignments.items() if aname in symbolic.free_symbols_and_functions(v)
                        }
                        reassignments = replacements & edge.data.assignments.keys()
                        if reassignments and (used_in_assignments - reassignments):
                            assignments[aname] = _UnknownValue

                if isinstance(block, LoopRegion):
                    # Any constants before a loop that may be overwritten inside the loop cannot be assumed as constants
                    # for the loop itself.
                    assigned_in_loop = self._assignments_in_loop(block)
                    for k in assignments.keys():
                        if k in assigned_in_loop:
                            assignments[k] = _UnknownValue

                if block not in in_const_dict:
                    in_const_dict[block] = {}
                if assignments:
                    redo |= self._propagate(in_const_dict[block], assignments)

                if isinstance(block, ControlFlowRegion):
                    self._collect_constants_for_region(block, arrays, in_const_dict, pre_const_dict, post_const_dict,
                                                       out_const_dict)
                elif isinstance(block, ConditionalBlock):
                    self._collect_constants_for_conditional(block, arrays, in_const_dict, pre_const_dict,
                                                            post_const_dict, out_const_dict)
                else:
                    # Simple case, no change in constants through this block (states and other basic blocks).
                    pre_const_dict[block] = in_const_dict[block].copy()
                    post_const_dict[block] = in_const_dict[block].copy()
                    out_const_dict[block] = in_const_dict[block].copy()

        # For all sink nodes, compute the overlapping set of constants between them, making sure all constants in the
        # resulting intersection are actually constants (i.e., all blocks see the same constant value for them). This
        # resulting overlap forms the 'post constants' of this CFG.
        post_consts = {}
        post_consts_intersection = None
        sinks = cfg.sink_nodes()
        for sink in sinks:
            if post_consts_intersection is None:
                post_consts_intersection = set(out_const_dict[sink].keys())
            else:
                post_consts_intersection &= set(out_const_dict[sink].keys())
        for sink in sinks:
            for k, v in out_const_dict[sink].items():
                if k in post_consts_intersection:
                    if k not in post_consts:
                        post_consts[k] = v
                    elif post_consts[k] != _UnknownValue and post_consts[k] != v:
                        post_consts[k] = _UnknownValue
                else:
                    post_consts[k] = _UnknownValue
        post_const_dict[cfg] = post_consts

        out_consts = {}
        if isinstance(cfg, LoopRegion):
            # For a loop we can not determine if it is being executed and how many times it would be executed. The 'out
            # constants' are thus formed from the intersection of the loop's 'in constants' and 'post constants'.
            out_consts.update(in_const)
            for k, v in post_consts.items():
                if k not in out_consts:
                    out_consts[k] = _UnknownValue
                elif out_consts[k] != _UnknownValue and out_consts[k] != v:
                    out_consts[k] = _UnknownValue
        else:
            out_consts.update(post_consts)
        out_const_dict[cfg] = out_consts

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
