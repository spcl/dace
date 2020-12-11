# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
from dace.symbolic import symstr
from dace.properties import Property
from dace.transformation.transformation import ExpandTransformation
from dace.frontend.common import op_repository as oprepo
from dace.sdfg.nodes import LibraryNode
from dace.libraries.blas.nodes.matmul import _get_matmul_operands
import dace.sdfg.nodes
import dace.library as library
from dace.sdfg import SDFG, SDFGState
from dace import memlet as mm
import copy
import numpy as np


@library.expansion
class ExpandGerPure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):

        sdfg = dace.SDFG(node.label + "_sdfg")

        ((edge_x, outer_array_x, shape_x, strides_x), (edge_y, outer_array_y,
                                                       shape_y, strides_y),
         cdata) = _get_matmul_operands(node,
                                       parent_state,
                                       parent_sdfg,
                                       name_lhs="_x",
                                       name_rhs="_y",
                                       name_out="_res")

        dtype_x = outer_array_x.dtype.type
        dtype_y = outer_array_y.dtype.type
        dtype_c = dace.DTYPE_TO_TYPECLASS[np.result_type(dtype_x, dtype_y).type]

        if (len(shape_x) != 1 or len(shape_y) != 1):
            raise SyntaxError("Vectors must have single dimension")

        M, N = shape_x[0], shape_y[0]
        shape_c = (M, N)

        if outer_array_x.storage != outer_array_y.storage:
            raise ValueError("Input vectors must have same storage.")
        storage = outer_array_x.storage

        _, array_x = sdfg.add_array("_x",
                                    shape_x,
                                    dtype_x,
                                    strides=strides_x,
                                    storage=storage)
        _, array_y = sdfg.add_array("_y",
                                    shape_y,
                                    dtype_y,
                                    strides=strides_y,
                                    storage=storage)
        _, array_a = sdfg.add_array("_A", shape_c, dtype_c, storage=storage)
        _, array_res = sdfg.add_array("_res", shape_c, dtype_c, storage=storage)

        if node.alpha == 1.0:
            mul_program = "__out = __x * __y"
        else:
            mul_program = "__out = {} * __x * __y".format(node.alpha)

        init_state = sdfg.add_state(node.label + "_initstate")
        state = sdfg.add_state_after(init_state, node.label + "_state")

        # prepare beta*C
        mul_out, mul_out_array = tmp, array_tmp = sdfg.add_temp_transient(
            shape_c, dtype_c, storage=storage)
        access_tmp = state.add_read(tmp)
        output_nodes = {mul_out: access_tmp}

        # init map
        init_state.add_mapped_tasklet(
            '_GER_init',
            {'_o%d' % i: '0:%s' % symstr(d)
             for i, d in enumerate(shape_c)}, {},
            'out = 0', {
                'out':
                dace.Memlet.simple(
                    mul_out, ','.join(['_o%d' % i
                                       for i in range(len(shape_c))]))
            },
            external_edges=True)

        # outer product map
        state.add_mapped_tasklet(
            "_GER_outer_",
            {"__i%d" % i: "0:%s" % s
             for i, s in enumerate([M, N])}, {
                 "__x": dace.Memlet.simple("_x", "__i0"),
                 "__y": dace.Memlet.simple("_y", "__i1")
             },
            mul_program, {
                "__out":
                dace.Memlet.simple(
                    mul_out, "__i0, __i1", wcr_str="lambda x, y: x + y")
            },
            external_edges=True,
            output_nodes=output_nodes)

        add_program = "__res = __A + __tmp"

        # manually broadcasting C to [M, N]
        if list(shape_c) == [M, N]:
            memlet_idx = '__i0, __i1'
        elif list(shape_c) == [1, N]:
            memlet_idx = '0, __i1'
        elif list(shape_c) == [M, 1]:
            memlet_idx = '__i0, 0'
        elif list(shape_c) == [N]:
            memlet_idx = '__i1'
        else:
            raise ValueError(
                "Could not broadcast input _res to ({}, {})".format(M, N))

        # addition map
        state.add_mapped_tasklet(
            "_GER_add_",
            {"__i%d" % i: "0:%s" % s
             for i, s in enumerate([M, N])}, {
                 "__A": dace.Memlet.simple("_A", memlet_idx),
                 "__tmp": dace.Memlet.simple(mul_out, "__i0, __i1"),
             },
            add_program, {"__res": dace.Memlet.simple("_res", "__i0, __i1")},
            external_edges=True,
            input_nodes={mul_out: access_tmp})

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandGerPure.make_sdfg(node, state, sdfg)


@library.node
class Ger(LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandGerPure,
    }
    default_implementation = None

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)

    alpha = Property(
        dtype=tuple(dace.dtypes._CONSTANT_TYPES),
        default=1,
        desc=
        "A scalar which will be multiplied with the outer product x*yT before adding matrix A"
    )

    def __init__(self, name, dtype=None, location=None, alpha=1):
        super().__init__(name,
                         location=location,
                         inputs={"_x", "_y", "_A"},
                         outputs={"_res"})
        self.dtype = dtype
        self.alpha = alpha

    def validate(self, sdfg, state):

        in_edges = state.in_edges(self)
        if len(in_edges) != 3:
            raise ValueError(
                "Expected exactly three inputs to the ger operation (vectors x, y and matrix A)"
            )

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError(
                "Expected exactly one output from the ger operation")

        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == "_A":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_a = subset.size()
            if dst_conn == "_x":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_x = subset.size()
            if dst_conn == "_y":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_y = subset.size()

        if len(size_a) != 2:
            raise ValueError("A must be a matrix")
        if len(size_x) != 1:
            raise ValueError("x must be a vector")
        if len(size_y) != 1:
            raise ValueError("y must be a vector")

        if size_a[0] != size_x[0] or size_a[1] != size_y[0]:
            raise ValueError(
                "Input vectors x and y (outer product) must match with the matrix A dimensions."
            )

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError(
                "Expected exactly one output from ger rank 1 operation.")
        out_memlet = out_edges[0].data

        out_subset = copy.deepcopy(out_memlet.subset)
        out_subset.squeeze()
        size_out = out_subset.size()
        if (len(size_out) != 2 or size_out[0] != size_a[0]
                or size_out[1] != size_a[1]):
            raise ValueError(
                "Output matrix must match input matrix a and outer product x*yT."
            )


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.ger')
@oprepo.replaces('dace.libraries.blas.Ger')
def ger_libnode(sdfg: SDFG, state: SDFGState, A, x, y, output, alpha):
    # Add nodes
    A_in, x_in, y_in = (state.add_read(name) for name in (A, x, y))
    out = state.add_write(output)

    libnode = Ger('ger',
                   dtype=sdfg.arrays[A].dtype,
                   alpha=alpha)
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(A_in, None, libnode, '_A', mm.Memlet(A))
    state.add_edge(x_in, None, libnode, '_x', mm.Memlet(x))
    state.add_edge(y_in, None, libnode, '_y', mm.Memlet(y))
    state.add_edge(libnode, '_res', out, None, mm.Memlet(output))

    return []
