# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Analysis pass that infers which SDFG arguments are compile-time constant."""
import dace
from dace.transformation import pass_pipeline as ppl, transformation
from typing import Dict, Set, Tuple
from dace import properties
import dace.sdfg.utils as sdutils


@properties.make_properties
@transformation.explicit_cf_compatible
class InferConstantArguments(ppl.Pass):
    """Infer the compile-time-constant data and symbols of each ``GPU_Device`` map and NestedSDFG."""

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.CFG & ppl.Modifies.SDFG & ppl.Modifies.Nodes

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: dace.SDFG, pipeline_res: Dict) -> Dict[str, Tuple[Set[str], Set[str]]]:
        """Map each GPU device map / NestedSDFG ``guid`` to its ``(constant_data, constant_symbols)``.

        :param sdfg: the SDFG to analyze.
        :param pipeline_res: results of previously applied passes (unused).
        :returns: a dict from node ``guid`` to a ``(constant_data, constant_symbols)`` pair.
        """
        const_args_dict = dict()
        for node, parent_graph in sdfg.all_nodes_recursive():
            if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device:
                const_args_dict[node.guid] = (sdutils.get_constant_data(node, parent_state=parent_graph),
                                              sdutils.get_constant_symbols(node, parent_state=parent_graph))
            elif isinstance(node, dace.sdfg.nodes.NestedSDFG):
                const_args_dict[node.guid] = (sdutils.get_constant_data(node.sdfg, parent_state=parent_graph),
                                              sdutils.get_constant_symbols(node.sdfg, parent_state=parent_graph))

        return const_args_dict
