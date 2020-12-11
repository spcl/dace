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
from dace import memlet as mm, subsets as sbs
import dace
import copy
import numpy as np

            
@library.expansion
class ExpandGerPure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        inputs = ('_A', '_x', '_y')
        outputs = ('_res',)
        in_edges = [next(parent_state.in_edges_by_connector(node, conn)) for conn in inputs]
        out_edges = [next(parent_state.out_edges_by_connector(node, conn)) for conn in outputs]
        arrays = {}
        arrays.update({inp: parent_sdfg.arrays[e.data.data] for inp, e in zip(inputs, in_edges)})
        arrays.update({out: parent_sdfg.arrays[e.data.data] for out, e in zip(outputs, out_edges)})

        # TODO: Support memlet subsets
        if any(e.data.subset != sbs.Range.from_array(arrays[a]) for a, e in zip(inputs, in_edges)):
            raise NotImplementedError
        if any(e.data.subset != sbs.Range.from_array(arrays[a]) for a, e in zip(outputs, out_edges)):
            raise NotImplementedError
        
        alpha = dace.symbol('alpha', arrays['_A'].dtype)
        m, n = arrays['_A'].shape
        @dace.program
        def ger(_A: arrays['_A'], _x: arrays['_x'], _y: arrays['_y'], _res: arrays['_res']):
            for i,j in dace.map[0:m, 0:n]:
                with dace.tasklet:
                    a << _A[i, j]
                    xin << _x[i]
                    yin << _y[j]
        
                    aout = alpha * xin * yin + a
        
                    aout >> _res[i, j]

        return ger.to_sdfg()

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
