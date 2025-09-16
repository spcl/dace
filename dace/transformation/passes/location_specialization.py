# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Tuple, List, Union, Dict

import dace
from dace.properties import make_properties, DictProperty, ShapeProperty
from dace import subsets, symbolic
from dace import Config, dtypes, symbolic
from dace.properties import make_properties
from dace.sdfg import SDFG, SDFGState, nodes, utils as sdutil
from dace.codegen.targets.experimental_cuda_helpers import gpu_utils
from dace.transformation import helpers, transformation
from dace.transformation import helpers, pass_pipeline as ppl, transformation

from dace.codegen.targets.cpp import sym2cpp


@make_properties
class LocationSpecialization(ppl.Pass):


    def apply_pass(self, sdfg: SDFG, kernel_dimensions_map: Dict) -> None:

        for node, state in sdfg.all_nodes_recursive():
            
            if not isinstance(node, dace.nodes.Tasklet):
                continue

            if not self._applicable(state, node):
                continue

            tasklet = node
            block_dims = self._get_block_dims(state, tasklet, kernel_dimensions_map)

            # Generate preambles for thread/warp/block
            preamble_thread = self._generate_condition_from_location("gpu_thread", self._get_thread_id(block_dims), tasklet)
            preamble_warp   = self._generate_condition_from_location("gpu_warp",   self._get_warp_id(block_dims),  tasklet)
            preamble_block  = self._generate_condition_from_location("gpu_block",  self._get_block_id(block_dims), tasklet)

            # Keep only non-empty preambles
            preambles = [p for p in (preamble_thread, preamble_warp, preamble_block) if p]


            for preamble in preambles:
                if tasklet.code.language == dace.dtypes.Language.Python:
                    cond = preamble.strip()[3:-1].strip()
                    cond = cond.replace("&&", "and").replace("||", "or")

                pre_tasklet = state.add_tasklet(f"specialization", {}, {}, cond)

                state.add_edge(pre_tasklet, None, tasklet, None, dace.Memlet())
                for pred in state.predecessors(tasklet):
                    state.add_edge(pred, None, pre_tasklet, None, dace.Memlet())
                

            import textwrap
            # Wrap tasklet code with preambles and closing braces
            for preamble in preambles:
                original = tasklet.code.as_string or ""
                if tasklet.code.language == dace.dtypes.Language.Python:
                    # Turn CUDA-style preamble into a Python if-statement
                    cond = preamble.strip()[3:-1].strip()  # strip "if (" at start and "{"
                    cond = cond.replace("&&", "and").replace("||", "or")
                    tasklet.code.as_string = f"if {cond}:\n" + textwrap.indent(original, "    ")
                else:
                    # Leave CUDA/C++ unchanged
                    tasklet.code.as_string = preamble + original + "}\n"


    def _applicable(self, state: SDFGState, tasklet: dace.nodes.Tasklet) -> bool:
            """
            Check if this transformation is applicable.

            Applicable if:
            * The tasklet is scheduled to run within a GPU kernel, and
            * Its location dictionary contains at least one of:
                - "gpu_block"
                - "gpu_thread"
                - "gpu_warp"
            """

            # Not within the kernel - skip
            if not gpu_utils.is_within_schedule_types(state, tasklet, dtypes.GPU_SCHEDULES):
                return False

            # return if the location dictionary contain block, thread of warp specialization
            return any(k in tasklet.location for k in ("gpu_block", "gpu_thread", "gpu_warp"))

    def _generate_condition_from_location(self, name: str, index_expr: str, node: nodes.Tasklet) -> str:
        if name not in node.location:
            return ''

        location: Union[int, str, subsets.Range] = node.location[name]
        if isinstance(location, str) and ':' in location:
            location = subsets.Range.from_string(location)
        elif symbolic.issymbolic(location):
            location = sym2cpp(location)

        if isinstance(location, subsets.Range):
            # Range of indices
            if len(location) != 1:
                raise ValueError(f'Only one-dimensional ranges are allowed for {name} specialization, {location} given')
            begin, end, stride = location[0]
            rb, re, rs = sym2cpp(begin), sym2cpp(end), sym2cpp(stride)
            cond = ''
            cond += f'(({index_expr}) >= {rb}) && (({index_expr}) <= {re})'
            if stride != 1:
                cond += f' && ((({index_expr}) - {rb}) % {rs} == 0)'

            return (f'if ({cond}) {{\n')
        else:
            # Single-element
            return(f'if (({index_expr}) == {location}) {{\n')

    def _get_thread_id(self, block_dims: List) -> str:
        result = 'threadIdx.x'
        if block_dims[1] != 1:
            result += f' + ({sym2cpp(block_dims[0])}) * threadIdx.y'
        if block_dims[2] != 1:
            result += f' + ({sym2cpp(block_dims[0] * block_dims[1])}) * threadIdx.z'
        return result

    def _get_warp_id(self, block_dims: List) -> str:
        return f'(({self._get_thread_id(block_dims)}) / warpSize)'

    def _get_block_id(self, block_dims: List) -> str:
        result = 'blockIdx.x'
        if block_dims[1] != 1:
            result += f' + gridDim.x * blockIdx.y'
        if block_dims[2] != 1:
            result += f' + gridDim.x * gridDim.y * blockIdx.z'
        return result
    
    def _get_block_dims(self, state, tasklet, kernel_dimensions_map) -> List:

        parent_map, parent_map_state = gpu_utils.get_parent_map(state, tasklet)
        while parent_map.map.schedule != dtypes.ScheduleType.GPU_Device:
            parent_map, parent_map_state = gpu_utils.get_parent_map(parent_map_state, parent_map)
        
        _, block_size = kernel_dimensions_map[parent_map]
        return block_size

    @staticmethod
    def annotates_memlets():
        return False
