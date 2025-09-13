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
        inp_name, inp, in_subset, out_name, out, _ = node.validate(parent_sdfg, parent_state)
        map_lengths = [(e + 1 - b) // s for (b, e, s) in in_subset]

        sdfg = dace.SDFG(f"{node.label}_sdfg")
        sdfg.add_array(inp_name, map_lengths, inp.dtype, inp.storage, strides=inp.strides)
        sdfg.add_array(out_name, map_lengths, out.dtype, out.storage, strides=out.strides)

        state = sdfg.add_state(f"{node.label}_state")
        sdfg.schedule = dace.dtypes.ScheduleType.Default

        map_params = [f"__i{i}" for i in range(len(map_lengths))]
        map_rng = {i: f"0:{s}" for i, s in zip(map_params, map_lengths)}
        access_expr = ','.join(map_params)
        inputs = {"_inp": dace.memlet.Memlet(f"{inp_name}[{access_expr}]")}
        outputs = {"_out": dace.memlet.Memlet(f"{out_name}[{access_expr}]")}
        code = "_out = _inp"
        if inp.storage == dace.dtypes.StorageType.GPU_Global:
            schedule = dace.dtypes.ScheduleType.GPU_Device
        else:
            schedule = dace.dtypes.ScheduleType.Default
        state.add_mapped_tasklet(f"{node.label}_tasklet",
                                 map_rng,
                                 inputs,
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
        inp_name, inp, in_subset, out_name, out, out_subset = node.validate(parent_sdfg, parent_state)
        map_lengths = [(e + 1 - b) // s for (b, e, s) in in_subset]
        cp_size = reduce(operator.mul, map_lengths, 1)

        sdfg = dace.SDFG(f"{node.label}_sdfg")
        sdfg.add_array(inp_name, map_lengths, inp.dtype, inp.storage, strides=inp.strides)
        sdfg.add_array(out_name, map_lengths, out.dtype, out.storage, strides=out.strides)

        state = sdfg.add_state(f"{node.label}_main")

        in_access = state.add_access(inp_name)
        out_access = state.add_access(out_name)
        tasklet = state.add_tasklet(
            name=f"memcpy_tasklet",
            inputs={"_in"},
            outputs={"_out"},
            code=
            f"cudaMemcpyAsync(_out, _in, {sym2cpp(cp_size)} * sizeof({inp.dtype.ctype}), cudaMemcpyDeviceToDevice, __dace_current_stream);",
            language=dace.Language.CPP,
            code_global=f"#include <cuda_runtime.h>\n")
        tasklet.schedule = dace.dtypes.ScheduleType.GPU_Device

        state.add_edge(
            in_access, None, tasklet, "_in",
            dace.memlet.Memlet(data=inp_name, subset=dace.subsets.Range([(0, e - 1, 1) for e in map_lengths])))
        state.add_edge(
            tasklet, "_out", out_access, None,
            dace.memlet.Memlet(data=out_name, subset=dace.subsets.Range([(0, e - 1, 1) for e in map_lengths])))
        return sdfg


@library.node
class CopyLibraryNode(nodes.LibraryNode):
    implementations = {"pure": ExpandPure, "CUDA": ExpandCUDA}
    default_implementation = 'pure'

    def _extend_in_out_edges(self, parent_state, parent_sdfg):
        # Set connection from parent state to the nested to cover the complete array subset
        oes = {oe for oe in parent_state.edges() if oe.src == self}
        ies = {ie for ie in parent_state.edges() if ie.dst == self}
        assert len(oes) == 1 and len(
            ies) == 1, f"Length of in edges / out edges of the library node is not both 1: {len(ies)}, {len(oes)}"
        # New subset covers the complete array, as the correct subset offset is done in the inner map
        for e in {next(iter(oes)), next(iter(ies))}:
            e.data = dace.memlet.Memlet.from_array(e.data.data, parent_sdfg.arrays[e.data.data])

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def validate(self, sdfg, state):
        """
        Validates the tensor transposition operation.
        :return: A tuple (inp, out) for the data descriptors in the parent SDFG.
        """

        inp_name, inp, in_subset, out_name, out, out_subset = None, None, None, None, None, None
        if len(state.out_edges(self)) != 1:
            raise ValueError("Number of out edges unequal to one")
        if len(state.in_edges(self)) != 1:
            raise ValueError("Number of in edges unequal to one")

        oe = next(iter(state.out_edges(self)))
        out = sdfg.arrays[oe.data.data]
        out_subset = oe.data.subset
        out_name = oe.data.data
        ie = next(iter(state.in_edges(self)))
        inp = sdfg.arrays[ie.data.data]
        in_subset = ie.data.subset
        inp_name = ie.data.data

        if not inp:
            raise ValueError("Missing the input tensor.")
        if not out:
            raise ValueError("Missing the output tensor.")

        if inp.dtype != out.dtype:
            raise ValueError("The datatype of the input and output tensors must match.")

        if inp.storage != out.storage:
            raise ValueError("The storage of the input and output tensors must match.")

        return inp_name, inp, in_subset, out_name, out, out_subset
