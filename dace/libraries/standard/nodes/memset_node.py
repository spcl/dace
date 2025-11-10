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
        map_lengths = [(e + 1 - b) // s for (b, e, s) in out_subset if (e + 1 - b) // s != 1]
        cp_size = reduce(operator.mul, map_lengths, 1)

        out_shape_collapsed = map_lengths
        out_strides_collapsed = [
            stride for (b, e, s), stride in zip(out_subset, out.strides) if ((e + 1 - b) // s) != 1
        ]

        sdfg = dace.SDFG(f"{node.label}_sdfg")
        sdfg.add_array(out_name, out_shape_collapsed, out.dtype, out.storage, strides=out_strides_collapsed)

        state = sdfg.add_state(f"{node.label}_state")
        map_params = [f"__i{i}" for i in range(len(map_lengths))]
        map_rng = {i: f"0:{s}" for i, s in zip(map_params, map_lengths)}
        access_expr = ','.join(map_params)
        outputs = {"_memset_out": dace.memlet.Memlet(f"{out_name}[{access_expr}]")}
        code = "_memset_out = 0"
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
        cp_size = reduce(operator.mul, map_lengths, 1)

        out_shape_collapsed = [ml for ml in map_lengths if ml != 1]
        out_strides_collapsed = [
            stride for (b, e, s), stride in zip(out_subset, out.strides) if ((e + 1 - b) // s) != 1
        ]

        sdfg = dace.SDFG(f"{node.label}_sdfg")
        sdfg.add_array(out_name, out_shape_collapsed, out.dtype, out.storage, strides=out_strides_collapsed)

        state = sdfg.add_state(f"{node.label}_state")

        out_access = state.add_access(out_name)
        tasklet = state.add_tasklet(
            name=f"memcpy_tasklet",
            inputs={},
            outputs={"_memset_out"},
            code=
            f"cudaMemsetAsync(_memset_out, 0, {sym2cpp(cp_size)} * sizeof({out.dtype.ctype}), __dace_current_stream);",
            language=dace.Language.CPP,
            code_global=f"#include <cuda_runtime.h>\n")

        state.add_edge(
            tasklet, "_memset_out", out_access, None,
            dace.memlet.Memlet(data=out_name, subset=dace.subsets.Range([(0, e - 1, 1) for e in map_lengths])))


        return sdfg


@library.expansion
class ExpandCPU(ExpandTransformation):
    environments = [environments.CPU]

    @staticmethod
    def expansion(node, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG):
        out_name, out, out_subset = node.validate(parent_sdfg, parent_state)
        map_lengths = [(e + 1 - b) // s for (b, e, s) in out_subset if (e + 1 - b) // s != 1]
        cp_size = reduce(operator.mul, map_lengths, 1)

        out_shape_collapsed = map_lengths
        out_strides_collapsed = [
            stride for (b, e, s), stride in zip(out_subset, out.strides) if ((e + 1 - b) // s) != 1
        ]

        sdfg = dace.SDFG(f"{node.label}_sdfg")
        sdfg.add_array(out_name, out_shape_collapsed, out.dtype, out.storage, strides=out_strides_collapsed)

        state = sdfg.add_state(f"{node.label}_state")

        # Access the original output
        out_access = state.add_access(out_name)

        # Add a tasklet that does standard CPU memset
        tasklet = state.add_tasklet(name=f"memset_tasklet",
                                    inputs={},
                                    outputs={"_memset_out"},
                                    code=f"memset(_memset_out, 0, {sym2cpp(cp_size)} * sizeof({out.dtype.ctype}));",
                                    language=dace.Language.CPP,
                                    code_global="#include <cstring>")  # include C++ memset header

        # Connect tasklet to the output
        state.add_edge(
            tasklet, "_memset_out", out_access, None,
            dace.memlet.Memlet(data=out_name, subset=dace.subsets.Range([(0, e - 1, 1) for e in map_lengths])))

        return sdfg


@library.node
class MemsetLibraryNode(nodes.LibraryNode):
    implementations = {"pure": ExpandPure, "CUDA": ExpandCUDA, "CPU": ExpandCPU}
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

        oe = next(iter(state.out_edges(self)))
        out = sdfg.arrays[oe.data.data]
        out_subset = oe.data.subset
        out_name = oe.src_conn

        if not out:
            raise ValueError("Missing the output tensor.")

        for ie in state.in_edges(self):
            state.remove_edge(ie)
            if state.degree(ie.src) == 0:
                state.remove_node(ie.src)
            state.add_edge(ie.src, None, self, None, dace.memlet.Memlet(None))
            self.remove_in_connector(ie.dst_conn)

        return out_name, out, out_subset
