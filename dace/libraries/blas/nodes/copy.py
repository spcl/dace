# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas import blas_helpers
from dace import memlet as mm, SDFG, SDFGState
from dace.frontend.common import op_repository as oprepo


@dace.library.expansion
class ExpandCopyPure(ExpandTransformation):
    """Expand a BLAS COPY by delegating to the standard :class:`CopyLibraryNode`.

    A BLAS ``xcopy`` is a plain element move, so there is no arithmetic for a
    vendor kernel to accelerate -- ``cblas_dcopy`` is a strided ``memcpy``.  The
    standard copy node already picks the right lowering (contiguous ``memcpy``,
    strided map, host/GPU) from the array descriptors, so routing through it
    keeps a single copy implementation instead of a parallel vendor one.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        from dace.libraries.standard.nodes import CopyLibraryNode
        (desc_x, stride_x), (desc_y, stride_y), sz = node.validate(parent_sdfg, parent_state)
        n = n or node.n or sz

        sdfg = dace.SDFG(node.label + "_sdfg")
        sdfg.add_array("_x", [n], desc_x.dtype, strides=[stride_x], storage=desc_x.storage)
        sdfg.add_array("_y", [n], desc_y.dtype, strides=[stride_y], storage=desc_y.storage)

        state = sdfg.add_state(node.label + "_state")
        cp = CopyLibraryNode(name=node.label + "_copy")
        state.add_node(cp)
        state.add_edge(state.add_read("_x"), None, cp, CopyLibraryNode.INPUT_CONNECTOR_NAME,
                       mm.Memlet.from_array("_x", sdfg.arrays["_x"]))
        state.add_edge(cp, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, state.add_write("_y"), None,
                       mm.Memlet.from_array("_y", sdfg.arrays["_y"]))
        return sdfg


@dace.library.node
class Copy(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandCopyPure,
    }
    default_implementation = "pure"

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    def __init__(self, name, n=None, **kwargs):
        super().__init__(name, inputs={"_x"}, outputs={"_y"}, **kwargs)
        self.n = n

    def validate(self, sdfg, state):
        """
        :return: A three-tuple ((x, stride_x), (y, stride_y), n).
        """
        desc_xs, desc_ys, n = blas_helpers.validate_level1_vector_to_vector(self, sdfg, state, "COPY")
        if desc_xs[0].dtype.base_type != desc_ys[0].dtype.base_type:
            raise TypeError(f"COPY dtype mismatch: {desc_xs[0].dtype} vs {desc_ys[0].dtype}")
        return desc_xs, desc_ys, n


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.copy')
@oprepo.replaces('dace.libraries.blas.Copy')
def copy_libnode(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, x, y):
    x_in = state.add_read(x)
    y_out = state.add_write(y)

    libnode = Copy('copy', n=sdfg.arrays[x].shape[0])
    state.add_node(libnode)

    state.add_edge(x_in, None, libnode, '_x', mm.Memlet(x))
    state.add_edge(libnode, '_y', y_out, None, mm.Memlet(y))

    return []
