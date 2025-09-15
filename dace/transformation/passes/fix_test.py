# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict, Set, Type, Union

import numpy as np
import sympy as sp

import dace
from dace import dtypes, properties, SDFG
from dace.codegen import common
from dace.config import Config
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.gpustream.gpustream_scheduling import NaiveGPUStreamScheduler
from dace.transformation.passes.gpustream.insert_gpu_streams_to_sdfgs import InsertGPUStreamsToSDFGs

from dace.codegen.targets.experimental_cuda_helpers.gpu_utils import is_within_schedule_types


@properties.make_properties
@transformation.explicit_cf_compatible
class Fix(ppl.Pass):
    """
    """

    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        return {NaiveGPUStreamScheduler, InsertGPUStreamsToSDFGs}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Dict[str, dace.data.Data]:

        from dace.transformation.helpers import get_parent_map

        skip = set()
        to_be_moved = set()
        names: Dict = dict()
        for node, parent_state in sdfg.all_nodes_recursive():
            if not isinstance(node, nodes.AccessNode):
                continue

            map_parent = None
            state = parent_state
            current = node
            while current is not None:
                if isinstance(current, nodes.MapEntry):
                    if current.map.schedule == dace.dtypes.ScheduleType.GPU_Device:
                        map_parent = current
                        break

                parent = get_parent_map(state, current)
                if parent is None:
                    break
                current, state = parent

            if map_parent is None:
                continue
            
            if node.data not in parent_state.sdfg.arrays:
                continue

            data_desc = node.desc(parent_state)
            if not data_desc.storage == dtypes.StorageType.Register:
                continue

            if isinstance(data_desc, dace.data.View) or data_desc.lifetime == dtypes.AllocationLifetime.Persistent:
                continue

            break_cond = False
            for edge, parent in sdfg.all_edges_recursive():
                if not isinstance(parent, dace.SDFGState):
                    continue
                src = edge.src
                if edge.dst_conn == node.data and isinstance(src, nodes.AccessNode) and src.data != node.data:
                    break_cond = True
                    skip.add(src.data)
            
            if break_cond:
                continue

            shape = data_desc.shape
            size_expr = np.prod(shape)

            # Try to evaluate the inequality
            cmp = sp.simplify(size_expr > 64)

            if cmp is sp.true:  # definitely larger
                move_out = True
            elif cmp is sp.false:  # definitely safe
                move_out = False
            else:
                # TODO: explain yakup and myself
                # undecidable case (symbolic expression)
                move_out = False  # or warn, depending on policy

            if move_out:
                to_be_moved.add((node.data, data_desc, map_parent))


        for name, desc, map_parent in to_be_moved:
            if name in skip:
                continue

            desc.storage = dtypes.StorageType.GPU_Global
            desc.transient = True
            names[name] = map_parent

    

        return names
