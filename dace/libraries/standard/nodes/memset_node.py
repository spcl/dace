# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import library, nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from functools import reduce
import operator
from dace.codegen.common import sym2cpp
import copy


@library.expansion
class ExpandPure(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        out_name, out, out_subset = node.validate(parent_sdfg, parent_state)
        map_lengths = [(e + 1 - b) // s for (b, e, s) in out_subset]

        sdfg = dace.SDFG(f"{node.label}_sdfg")
        sdfg.add_array(out_name, out.shape, out.dtype, out.storage, strides=out.strides)

        state = sdfg.add_state(f"{node.label}_state")
        map_params = [f"__i{i}" for i in range(len(map_lengths))]
        map_rng = {i: f"0:{s}" for i, s in zip(map_params, map_lengths)}
        access_expr = ','.join(map_params)
        outputs = {"_out": dace.memlet.Memlet(f"{out_name}[{access_expr}]")}
        code = "_out = 0"
        if out.storage == dace.dtypes.StorageType.GPU_Global:
            schedule = dace.dtypes.ScheduleType.GPU_Device
        else:
            schedule = dace.dtypes.ScheduleType.Default
        state.add_mapped_tasklet(f"{node.label}_tasklet",
                                 map_rng,
                                 dict(),
                                 code,
                                 outputs,
                                 schedule=schedule,
                                 external_edges=True)

        return sdfg


@library.expansion
class ExpandCUDA(ExpandTransformation):
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG):
        out_name, out, out_subset = node.validate(parent_sdfg, parent_state)
        map_lengths = [(e + 1 - b) // s for (b, e, s) in out_subset]
        memset_size = reduce(operator.mul, map_lengths, 1)

        sdfg = dace.SDFG(f"{node.label}_sdfg")
        sdfg.add_array(out_name, out.shape, out.dtype, out.storage, strides=out.strides)

        state = sdfg.add_state(f"{node.label}_main")

        out_access = state.add_access(out_name)
        tasklet = state.add_tasklet(
            name=f"memcpy_tasklet",
            inputs={},
            outputs={"_out"},
            code=
            f"cudaMemsetAsync(_out, 0, {sym2cpp(memset_size)} * sizeof({out.dtype.ctype}), __dace_current_stream);",
            language=dace.Language.CPP,
            code_global=f"#include <cuda_runtime.h>\n")

        state.add_edge(tasklet, "_out", out_access, None,
                       dace.memlet.Memlet(data=out_name, subset=copy.deepcopy(out_subset)))

        return sdfg


@library.node
class MemsetLibraryNode(nodes.LibraryNode):
    implementations = {"pure": ExpandPure, "CUDA": ExpandCUDA}
    default_implementation = 'pure'

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def validate(self, sdfg, state):
        """
        Validates the tensor transposition operation.
        :return: A tuple (inp, out) for the data descriptors in the parent SDFG.
        """

        out_name, out, out_subset = None, None, None
        if len(state.out_edges(self)) != 1:
            raise ValueError("Number of out edges unequal to one")
        if len(state.in_edges(self)) != 0:
            raise ValueError("Number of in edges unequal to one")

        oe = next(iter(state.out_edges(self)))
        out = sdfg.arrays[oe.data.data]
        out_subset = oe.data.subset
        out_name = oe.data.data

        if not out:
            raise ValueError("Missing the output tensor.")

        return out_name, out, out_subset
