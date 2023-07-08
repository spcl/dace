# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import functools
from copy import deepcopy as dc
from typing import List

from dace.config import Config
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.libraries.blas import blas_helpers
from dace.transformation.transformation import ExpandTransformation
from .. import environments
import warnings


def _get_permute_input(node, state, sdfg):
    """Returns the permute input edge, array, and shape."""
    for edge in state.in_edges(node):
        if edge.dst_conn == "_inp":
            subset = dc(edge.data.subset)
            subset.squeeze()
            size = subset.size()
            outer_array = sdfg.data(dace.sdfg.find_input_arraynode(state, edge).data)
            return edge, outer_array, size
    raise ValueError("Permute input connector \"_inp\" not found.")


def _get_permute_output(node, state, sdfg):
    """Returns the permute output edge, array, and shape."""
    for edge in state.out_edges(node):
        if edge.src_conn == "_out":
            subset = dc(edge.data.subset)
            subset.squeeze()
            size = subset.size()
            outer_array = sdfg.data(dace.sdfg.find_output_arraynode(state, edge).data)
            return edge, outer_array, size
    raise ValueError("Permute output connector \"_out\" not found.")


@dace.library.expansion
class ExpandPermutePure(ExpandTransformation):
    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):

        in_edge, in_outer_array, in_shape = _get_permute_input(node, parent_state, parent_sdfg)
        out_edge, out_outer_array, out_shape = _get_permute_output(node, parent_state, parent_sdfg)
        dtype = node.dtype
        axes = node.axes
        sdfg = dace.SDFG(node.label + "_sdfg")
        state = sdfg.add_state(node.label + "_state")

        _, in_array = sdfg.add_array("_inp",
                                     in_shape,
                                     dtype,
                                     strides=in_outer_array.strides,
                                     storage=in_outer_array.storage)
        _, out_array = sdfg.add_array("_out",
                                      out_shape,
                                      dtype,
                                      strides=out_outer_array.strides,
                                      storage=out_outer_array.storage)

        num_elements = functools.reduce(lambda x, y: x * y, in_array.shape)
        if num_elements == 1:
            inp = state.add_read("_inp")
            out = state.add_write("_out")
            tasklet = state.add_tasklet("permute", {"__inp"}, {"__out"}, "__out = __inp")
            state.add_edge(inp, None, tasklet, "__inp", dace.memlet.Memlet.from_array("_inp", in_array))
            state.add_edge(tasklet, "__out", out, None, dace.memlet.Memlet.from_array("_out", out_array))
        else:
            state.add_mapped_tasklet(
                "_permute_", {"_i{}".format(i): "0:{}".format(s)
                                for i, s in enumerate(in_array.shape)},
                dict(_tmp_in=dace.memlet.Memlet.simple("_inp", ", ".join("_i{}".format(i) for i, _ in enumerate(in_array.shape)))),
                "_tmp_out = _tmp_in",
                dict(_tmp_out=dace.memlet.Memlet.simple("_out", ", ".join("_i{}".format(axes[i]) for i, _ in enumerate(in_array.shape)))),
                external_edges=True)

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        return ExpandPermutePure.make_sdfg(node, state, sdfg)


# @dace.library.expansion
# class ExpandPermuteCuTENSOR(ExpandTransformation):
#
#     environments = [environments.cublas.cuBLAS]
#
#     @staticmethod
#     def expansion(node, state, sdfg, **kwargs):
#         node.validate(sdfg, state)
#         dtype = node.dtype
#
#         try:
#             func, cdtype, factort = blas_helpers.cublas_type_metadata(dtype)
#         except TypeError as ex:
#             warnings.warn(f'{ex}. Falling back to pure expansion')
#             return ExpandPermutePure.expansion(node, state, sdfg, **kwargs)
#
#         func = func + 'geam'
#
#         alpha = f"__state->cublas_handle.Constants(__dace_cuda_device).{factort}Pone()"
#         beta = f"__state->cublas_handle.Constants(__dace_cuda_device).{factort}Zero()"
#         _, _, (m, n) = _get_permute_input(node, state, sdfg)
#
#         code = (environments.cublas.cuBLAS.handle_setup_code(node) + f"""cublas{func}(
#                     __dace_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
#                     {m}, {n}, {alpha}, ({cdtype}*)_inp, {n}, {beta}, ({cdtype}*)_inp, {m}, ({cdtype}*)_out, {m});
#                 """)
#
#         tasklet = dace.sdfg.nodes.Tasklet(node.name,
#                                           node.in_connectors,
#                                           node.out_connectors,
#                                           code,
#                                           language=dace.dtypes.Language.CPP)
#
#         return tasklet


@dace.library.node
class Permute(dace.sdfg.nodes.LibraryNode):
    # Global properties
    implementations = {
        "pure": ExpandPermutePure,
        # "cuTENSOR": ExpandPermuteCuTensor
    }
    default_implementation = None

    dtype = dace.properties.TypeClassProperty(allow_none=True)
    axes = dace.properties.ListProperty(element_type=int, allow_none=True,
                                        desc="Axes to permute.")

    def __init__(self, name, axes, dtype=None, location=None, ):
        super().__init__(name, location=location, inputs={'_inp'}, outputs={'_out'})
        self.dtype = dtype
        self.axes = axes

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) != 1:
            raise ValueError("Expected exactly one input to permute operation")
        in_size = None
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == '_inp':
                subset = dc(memlet.subset)
                subset.squeeze()
                in_size = subset.size()
        if in_size is None:
            raise ValueError("Input connector not found.")
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from permute operation")
        out_memlet = out_edges[0].data

        out_subset = dc(out_memlet.subset)
        out_subset.squeeze()
        out_size = out_subset.size()
        if len(out_size) != len(in_size):
            raise ValueError("Permute operation only supported on matrices of same dimensionalities.")
        if set(out_size) != set(in_size):
            raise ValueError("Expected input size to be a permutation of output size.")
