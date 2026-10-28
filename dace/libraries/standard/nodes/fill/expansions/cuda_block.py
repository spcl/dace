# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Block-collective fill inside a GPU kernel: the block's lanes split the elements, the innermost axis strided across
them so adjacent lanes write adjacent elements, and a barrier publishes the fill to every lane."""

import dace
from dace import library
from dace.libraries.standard.nodes.fill.common import VALUE_CONNECTOR_NAME, make_fill_skeleton, python_literal
from dace.libraries.standard.nodes.fill.node import FillLibraryNode
from dace.sdfg.propagation import propagate_memlets_state
from dace.transformation.transformation import ExpandTransformation


@library.register_expansion(FillLibraryNode, "CUDA (block)")
class ExpandCUDABlock(ExpandTransformation):
    environments = []
    runs_inside_kernel = True

    @staticmethod
    def expansion(node: "FillLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
        from dace.libraries.standard.block_reduce import BLOCK_COLLECTIVE_THREADS

        sdfg, state, out_name, out, map_lengths = make_fill_skeleton(node, parent_state)
        lanes, lanes_exit = state.add_map(
            f"{node.label}_lanes", {"__lane": f"0:{BLOCK_COLLECTIVE_THREADS}"}, dace.ScheduleType.GPU_ThreadBlock
        )
        lanes.map.gpu_force_syncthreads = True
        params = [f"__i{i}" for i in range(len(map_lengths))]
        ranges = {param: (0, length - 1, 1) for param, length in zip(params, map_lengths)}
        ranges[params[-1]] = ("__lane", map_lengths[-1] - 1, BLOCK_COLLECTIVE_THREADS)
        strided, strided_exit = state.add_map(f"{node.label}_elements", ranges, dace.ScheduleType.Sequential)

        value_info = node.value_descriptor(parent_state)
        if value_info is not None:
            sdfg.add_array(VALUE_CONNECTOR_NAME, [1], value_info[1].dtype, value_info[1].storage)
            tasklet = state.add_tasklet(f"{node.label}_tasklet", {"_val"}, {"_out"}, "_out = _val")
            state.add_memlet_path(
                state.add_read(VALUE_CONNECTOR_NAME),
                lanes,
                strided,
                tasklet,
                dst_conn="_val",
                memlet=dace.Memlet(f"{VALUE_CONNECTOR_NAME}[0]"),
            )
        else:
            tasklet = state.add_tasklet(
                f"{node.label}_tasklet", {}, {"_out"}, f"_out = {python_literal(node.value, out.dtype)}"
            )
            state.add_nedge(lanes, strided, dace.Memlet())
            state.add_nedge(strided, tasklet, dace.Memlet())
        state.add_memlet_path(
            tasklet,
            strided_exit,
            lanes_exit,
            state.add_write(out_name),
            src_conn="_out",
            memlet=dace.Memlet(f"{out_name}[{', '.join(params)}]"),
        )
        propagate_memlets_state(sdfg, state)
        return sdfg
