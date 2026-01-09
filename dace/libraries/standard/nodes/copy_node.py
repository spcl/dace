# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import library, nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from functools import reduce
import operator
from dace.codegen.common import sym2cpp

from dace.libraries.standard.helper import collapse_shape_and_strides, add_dynamic_inputs


@library.expansion
class ExpandPure(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        inp_name, inp, in_subset, out_name, out, out_subset, dynamic_inputs = node.validate(parent_sdfg, parent_state)
        map_lengths = [(e + 1 - b) // s for (b, e, s) in in_subset]

        in_shape_collapsed, in_strides_collapsed = collapse_shape_and_strides(in_subset, inp.strides)
        out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

        sdfg = dace.SDFG(f"{node.label}_sdfg")
        sdfg.add_array(inp_name, in_shape_collapsed, inp.dtype, inp.storage, strides=in_strides_collapsed)
        sdfg.add_array(out_name, out_shape_collapsed, out.dtype, out.storage, strides=out_strides_collapsed)

        state = sdfg.add_state(f"{node.label}_state", is_start_block=True)

        map_lengths = add_dynamic_inputs(dynamic_inputs, sdfg, in_subset, state)
        sdfg.schedule = dace.dtypes.ScheduleType.Sequential

        map_params = [f"__i{i}" for i in range(len(map_lengths))]
        map_rng = {i: f"0:{s}" for i, s in zip(map_params, map_lengths)}
        in_access_expr = ','.join(map_params)
        out_access_expr = ','.join(map_params)
        inputs = {"_memcpy_inp": dace.memlet.Memlet(f"{inp_name}[{in_access_expr}]")}
        outputs = {"_memcpy_out": dace.memlet.Memlet(f"{out_name}[{out_access_expr}]")}
        code = "_memcpy_out = _memcpy_inp"
        state.add_mapped_tasklet(f"{node.label}_tasklet",
                                 map_rng,
                                 inputs,
                                 code,
                                 outputs,
                                 schedule=dace.dtypes.ScheduleType.GPU_Device if out.storage
                                 == dace.dtypes.StorageType.GPU_Global else dace.dtypes.ScheduleType.Default,
                                 external_edges=True)

        return sdfg


@library.expansion
class ExpandCUDA(ExpandTransformation):
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG):
        inp_name, inp, in_subset, out_name, out, out_subset, dynamic_inputs = node.validate(parent_sdfg, parent_state)

        map_lengths = [(e + 1 - b) // s for (b, e, s) in in_subset]
        cp_size = reduce(operator.mul, map_lengths, 1)

        in_shape_collapsed, in_strides_collapsed = collapse_shape_and_strides(in_subset, inp.strides)
        out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

        sdfg = dace.SDFG(f"{node.label}_sdfg")
        sdfg.add_array(inp_name, in_shape_collapsed, inp.dtype, inp.storage, strides=in_strides_collapsed)
        sdfg.add_array(out_name, out_shape_collapsed, out.dtype, out.storage, strides=out_strides_collapsed)
        sdfg.schedule = dace.dtypes.ScheduleType.Sequential

        # Add dynamic inputs
        state = sdfg.add_state(f"{node.label}_state")
        map_lengths = add_dynamic_inputs(dynamic_inputs, sdfg, in_subset, state)

        in_access = state.add_access(inp_name)
        out_access = state.add_access(out_name)
        tasklet = state.add_tasklet(
            name=f"memcpy_tasklet",
            inputs={"_memcpy_in"},
            outputs={"_memcpy_out"},
            code=
            f"cudaMemcpyAsync(_memcpy_out, _memcpy_in, {sym2cpp(cp_size)} * sizeof({inp.dtype.ctype}), cudaMemcpyDeviceToDevice, __dace_current_stream);",
            language=dace.Language.CPP)

        tasklet.schedule = dace.dtypes.ScheduleType.GPU_Device

        state.add_edge(
            in_access, None, tasklet, "_memcpy_in",
            dace.memlet.Memlet(data=inp_name, subset=dace.subsets.Range([(0, e - 1, 1) for e in map_lengths])))
        state.add_edge(
            tasklet, "_memcpy_out", out_access, None,
            dace.memlet.Memlet(data=out_name, subset=dace.subsets.Range([(0, e - 1, 1) for e in map_lengths])))

        return sdfg


@library.expansion
class ExpandCPU(ExpandTransformation):
    environments = [environments.CPU]

    @staticmethod
    def expansion(node, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG):
        inp_name, inp, in_subset, out_name, out, out_subset, dynamic_inputs = node.validate(parent_sdfg, parent_state)
        map_lengths = [(e + 1 - b) // s for (b, e, s) in in_subset]
        cp_size = reduce(operator.mul, map_lengths, 1)

        in_shape_collapsed, in_strides_collapsed = collapse_shape_and_strides(in_subset, inp.strides)
        out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

        sdfg = dace.SDFG(f"{node.label}_sdfg")
        sdfg.add_array(inp_name, in_shape_collapsed, inp.dtype, inp.storage, strides=in_strides_collapsed)
        sdfg.add_array(out_name, out_shape_collapsed, out.dtype, out.storage, strides=out_strides_collapsed)
        sdfg.schedule = dace.dtypes.ScheduleType.Sequential
        state = sdfg.add_state(f"{node.label}_state")

        # Add dynamic inputs
        map_lengths = add_dynamic_inputs(dynamic_inputs, sdfg, in_subset, state)

        # Add CPU access nodes
        in_access = state.add_access(inp_name)
        out_access = state.add_access(out_name)

        # Tasklet performing standard CPU memcpy
        tasklet = state.add_tasklet(
            name=f"memcpy_tasklet",
            inputs={"_memcpy_in"},
            outputs={"_memcpy_out"},
            code=f"memcpy(_memcpy_out, _memcpy_in, {sym2cpp(cp_size)} * sizeof({inp.dtype.ctype}));",
            language=dace.Language.CPP)

        # Connect input and output to the tasklet
        state.add_edge(
            in_access, None, tasklet, "_memcpy_in",
            dace.memlet.Memlet(data=inp_name, subset=dace.subsets.Range([(0, e - 1, 1) for e in map_lengths])))
        state.add_edge(
            tasklet, "_memcpy_out", out_access, None,
            dace.memlet.Memlet(data=out_name, subset=dace.subsets.Range([(0, e - 1, 1) for e in map_lengths])))

        return sdfg


@library.node
class CopyLibraryNode(nodes.LibraryNode):
    implementations = {"pure": ExpandPure, "CUDA": ExpandCUDA, "CPU": ExpandCPU}
    default_implementation = 'pure'

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def validate(self, sdfg, state):
        """
        Validates the tensor transposition operation.
        :return: A tuple (inp, out) for the data descriptors in the parent SDFG.
        """

        if len(state.out_edges(self)) != 1:
            raise ValueError("Number of out edges unequal to one")

        oe = next(iter(state.out_edges(self)))
        out = sdfg.arrays[oe.data.data]
        out_subset = oe.data.subset
        out_name = oe.src_conn

        # Add dynamic connectors
        dynamic_ies = {ie for ie in state.in_edges(self) if ie.dst_conn != "_in"}
        dynamic_inputs = dict()
        for ie in dynamic_ies:
            dataname = ie.data.data
            datadesc = state.sdfg.arrays[dataname]
            if not isinstance(datadesc, dace.data.Scalar):
                raise ValueError("Dynamic inputs (not connected to `_in`) need to be all scalars")
            dynamic_inputs[ie.dst_conn] = datadesc

        data_ies = {ie for ie in state.in_edges(self) if ie.dst_conn == "_in"}
        if len(data_ies) != 1:
            raise ValueError("Only when edge should be to dst connector `_in`")
        ie = data_ies.pop()
        inp = sdfg.arrays[ie.data.data]

        in_subset = ie.data.subset
        inp_name = ie.dst_conn
        if not inp:
            raise ValueError("Missing the input tensor.")
        if not out:
            raise ValueError("Missing the output tensor.")

        if inp.dtype != out.dtype:
            raise ValueError(
                "The datatype of the input and output tensors must match. For copies that also perform type conversion, please extend the library node expansion."
            )

        if inp.storage != out.storage:
            raise ValueError(
                "The storage of the input and output tensors must match. For CPU<->GPU copy nodes, please extend the library node expansion."
            )

        return inp_name, inp, in_subset, out_name, out, out_subset, dynamic_inputs
