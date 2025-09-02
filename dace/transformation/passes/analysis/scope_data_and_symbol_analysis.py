# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation import pass_pipeline as ppl, transformation
from typing import Dict, Set, NamedTuple
from dace import properties
import dace.sdfg.utils as sdutils


class ScopeAnalysis(NamedTuple):
    """Container for scope analysis results."""
    used_data: Set[str]
    used_symbols: Set[str]
    constant_data: Set[str]
    constant_symbols: Set[str]


@properties.make_properties
@transformation.explicit_cf_compatible
class ScopeDataAndSymbolAnalysis(ppl.Pass):
    """
    Analyzes data and symbol usage patterns within GPU device scopes and nested SDFGs
    (the scopes that result in generating function calls in the generated source code).

    This pass identifies which data containers and symbols are:
    - Used within the scope (may be read or written)
    - Constant within the scope (values don't change during execution of the scope)

    Note: This analysis identifies ALL data used within a scope, not just
    those that need to be passed as external arguments. All data containers used
    within the scope are included.

    Note: This analysis identifies only the symbols defined at the time of scope
    generation. This means that if a symbol is defined within the scope and has a
    lifetime shorter than the scope currently being analyzed, the symbol will not be
    included. This difference in behavior with respect to data is due to the current
    implementation of the lifetime scope of data in DaCe.

    Returns:
    Dict mapping node GUIDs to ScopeAnalysis containing:
    - used_data: All data containers accessed within the scope
    - used_symbols: All symbols required by the scope, including ones defined by the scope
    - constant_data: Data containers that remain constant within the scope
    - constant_symbols: The subset of used symbols that remain constant within the scope
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.CFG | ppl.Modifies.SDFG | ppl.Modifies.Nodes)

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: dace.SDFG, pipeline_res: Dict) -> Dict[str, ScopeAnalysis]:
        """
        Analyze data and symbol usage within GPU device maps and nested SDFGs.

        Args:
            sdfg: The SDFG to analyze
            pipeline_res: Results from previous pipeline passes (unused)

        Returns:
            Dictionary mapping node GUIDs to their scope analysis results
        """
        analysis_results = {}

        for node, parent_graph in sdfg.all_nodes_recursive():
            # Analyze GPU device maps
            if (isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device):

                analysis_results[node.guid] = ScopeAnalysis(
                    used_data=sdutils.get_used_data(node, parent_state=parent_graph),
                    used_symbols=sdutils.get_used_symbols(node,
                                                          parent_state=parent_graph,
                                                          include_symbols_for_offset_calculations=True),
                    constant_data=sdutils.get_constant_data(node, parent_state=parent_graph),
                    constant_symbols=sdutils.get_constant_symbols(node,
                                                                  parent_state=parent_graph,
                                                                  include_symbols_for_offset_calculations=True))

            # Analyze nested SDFGs
            elif isinstance(node, dace.sdfg.nodes.NestedSDFG):
                analysis_results[node.guid] = ScopeAnalysis(
                    used_data=sdutils.get_used_data(node, parent_state=parent_graph),
                    used_symbols=sdutils.get_used_symbols(node,
                                                          parent_state=parent_graph,
                                                          include_symbols_for_offset_calculations=True),
                    constant_data=sdutils.get_constant_data(node, parent_state=parent_graph),
                    constant_symbols=sdutils.get_constant_symbols(node,
                                                                  parent_state=parent_graph,
                                                                  include_symbols_for_offset_calculations=True))

        return analysis_results
