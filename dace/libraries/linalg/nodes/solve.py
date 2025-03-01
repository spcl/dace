# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace
import dace.library
import dace.properties
import dace.sdfg.nodes
import numpy as np

from dace import Memlet
from dace.libraries.lapack import Getrf, Getrs
from dace.libraries.standard import Transpose
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.lapack import environments
from dace.libraries.blas import environments as blas_environments


def _make_sdfg_getrs(node, parent_state, parent_sdfg, implementation):

    arr_desc = node.validate(parent_sdfg, parent_state)
    (ain_shape, ain_dtype, ain_strides, bin_shape, bin_dtype, bin_strides, out_shape, out_dtype, out_strides, n,
     rhs) = arr_desc
    dtype = ain_dtype

    sdfg = dace.SDFG("{l}_sdfg".format(l=node.label))

    ain_arr = sdfg.add_array('_ain', ain_shape, dtype=ain_dtype, strides=ain_strides)
    ainout_arr = sdfg.add_array('_ainout', [n, n], dtype=ain_dtype, transient=True)
    bin_arr = sdfg.add_array('_bin', bin_shape, dtype=bin_dtype, strides=bin_strides)
    binout_shape = [n, rhs]
    if implementation == 'cuSolverDn':
        binout_shape = [rhs, n]
    binout_arr = sdfg.add_array('_binout', binout_shape, dtype=out_dtype, transient=True)
    bout_arr = sdfg.add_array('_bout', out_shape, dtype=out_dtype, strides=out_strides)
    ipiv_arr = sdfg.add_array('_pivots', [n], dtype=dace.int32, transient=True)
    info_arr = sdfg.add_array('_info', [1], dtype=dace.int32, transient=True)

    state = sdfg.add_state("{l}_state".format(l=node.label))

    getrf_node = Getrf('getrf')
    getrf_node.implementation = implementation
    getrs_node = Getrs('getrs')
    getrs_node.implementation = implementation

    ain = state.add_read('_ain')
    ainout1 = state.add_read('_ainout')
    ainout2 = state.add_access('_ainout')
    bin = state.add_read('_bin')
    binout1 = state.add_read('_binout')
    binout2 = state.add_read('_binout')
    bout = state.add_access('_bout')
    if implementation == 'cuSolverDn':
        transpose_ain = Transpose('AT', dtype=ain_dtype)
        transpose_ain.implementation = 'cuBLAS'
        state.add_edge(ain, None, transpose_ain, '_inp', Memlet.from_array(*ain_arr))
        state.add_edge(transpose_ain, '_out', ainout1, None, Memlet.from_array(*ainout_arr))
        transpose_bin = Transpose('bT', dtype=bin_dtype)
        transpose_bin.implementation = 'cuBLAS'
        state.add_edge(bin, None, transpose_bin, '_inp', Memlet.from_array(*bin_arr))
        state.add_edge(transpose_bin, '_out', binout1, None, Memlet.from_array(*binout_arr))
        transpose_out = Transpose('XT', dtype=bin_dtype)
        transpose_out.implementation = 'cuBLAS'
        state.add_edge(binout2, None, transpose_out, '_inp', Memlet.from_array(*binout_arr))
        state.add_edge(transpose_out, '_out', bout, None, Memlet.from_array(*bout_arr))
    else:
        state.add_nedge(ain, ainout1, Memlet.from_array(*ain_arr))
        state.add_nedge(bin, binout1, Memlet.from_array(*bin_arr))
        state.add_nedge(binout2, bout, Memlet.from_array(*bout_arr))

    ipiv = state.add_access('_pivots')
    info1 = state.add_write('_info')
    info2 = state.add_write('_info')

    state.add_memlet_path(ainout1, getrf_node, dst_conn="_xin", memlet=Memlet.from_array(*ainout_arr))
    state.add_memlet_path(getrf_node, info1, src_conn="_res", memlet=Memlet.from_array(*info_arr))
    state.add_memlet_path(getrf_node, ipiv, src_conn="_ipiv", memlet=Memlet.from_array(*ipiv_arr))
    state.add_memlet_path(getrf_node, ainout2, src_conn="_xout", memlet=Memlet.from_array(*ainout_arr))
    state.add_memlet_path(ainout2, getrs_node, dst_conn="_a", memlet=Memlet.from_array(*ainout_arr))
    state.add_memlet_path(binout1, getrs_node, dst_conn="_rhs_in", memlet=Memlet.from_array(*binout_arr))
    state.add_memlet_path(ipiv, getrs_node, dst_conn="_ipiv", memlet=Memlet.from_array(*ipiv_arr))
    state.add_memlet_path(getrs_node, info2, src_conn="_res", memlet=Memlet.from_array(*info_arr))
    state.add_memlet_path(getrs_node, binout2, src_conn="_rhs_out", memlet=Memlet.from_array(*binout_arr))

    return sdfg


@dace.library.expansion
class ExpandSolvePure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        raise NotImplementedError("Missing pure implementation of linalg.solve.")

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) + ".")
        return ExpandSolvePure.make_sdfg(node, state, sdfg)


@dace.library.expansion
class ExpandSolveOpenBLAS(ExpandTransformation):

    environments = [blas_environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return _make_sdfg_getrs(node, parent_state, parent_sdfg, "OpenBLAS")


@dace.library.expansion
class ExpandSolveMKL(ExpandTransformation):

    environments = [blas_environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return _make_sdfg_getrs(node, parent_state, parent_sdfg, "MKL")


@dace.library.expansion
class ExpandSolveCuSolverDn(ExpandTransformation):

    environments = [environments.cusolverdn.cuSolverDn]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return _make_sdfg_getrs(node, parent_state, parent_sdfg, "cuSolverDn")


@dace.library.node
class Solve(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {"OpenBLAS": ExpandSolveOpenBLAS, "MKL": ExpandSolveMKL, "cuSolverDn": ExpandSolveCuSolverDn}
    default_implementation = None

    overwrite = dace.properties.Property(dtype=bool, default=False)

    # Object fields
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_ain", "_bin"}, outputs={"_bout"}, **kwargs)
        # NOTE: We currently do not support overwrite == True
        self.overwrite = False

    def validate(self, sdfg, state):
        """
        :return: A four-tuple (ain, aout, ipiv, info) of the three data
                 descriptors in the parent SDFG.
        """

        in_edges = state.in_edges(self)
        if len(in_edges) != 2:
            raise ValueError("Expected exactly two inputs to inv")
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from inv")

        desc_ain, desc_bin, desc_out = None, None, None
        for e in state.in_edges(self):
            if e.dst_conn == "_ain":
                desc_ain = sdfg.arrays[e.data.data]
                ain_memlet = e.data
            if e.dst_conn == "_bin":
                desc_bin = sdfg.arrays[e.data.data]
                bin_memlet = e.data
        for e in state.out_edges(self):
            if e.src_conn == "_bout":
                desc_out = sdfg.arrays[e.data.data]
                out_memlet = e.data

        # Squeeze input memlets
        squeezed_ain = copy.deepcopy(ain_memlet.subset)
        dims_ain = squeezed_ain.squeeze()
        squeezed_bin = copy.deepcopy(bin_memlet.subset)
        dims_bin = squeezed_bin.squeeze()
        # Squeeze output memlets
        squeezed_out = copy.deepcopy(out_memlet.subset)
        dims_out = squeezed_out.squeeze()

        if (desc_ain.dtype.base_type != desc_out.dtype.base_type
                or desc_ain.dtype.base_type != desc_bin.dtype.base_type):
            raise ValueError("Basetype of inputs and output must be equal!")

        if (len(squeezed_ain.size()) != 2 or len(squeezed_bin.size()) > 2 or len(squeezed_out.size()) > 2):
            raise ValueError("linalg.solve only supported with first input a "
                             " matrix and second input vector or matrix")

        shape_ain = squeezed_ain.size()
        shape_bin = squeezed_bin.size()
        shape_out = squeezed_out.size()
        if shape_ain[0] != shape_ain[1]:
            raise ValueError("linalg.solve only supported with first input a " "square matrix")
        if shape_ain[-1] != shape_bin[0]:
            raise ValueError("A column must be equal to B rows")
        if not np.array_equal(shape_bin, shape_out):
            raise ValueError("Squeezed shape of second input and output must be the same")

        strides_ain = np.array(desc_ain.strides)[dims_ain].tolist()
        strides_bin = np.array(desc_bin.strides)[dims_bin].tolist()
        strides_out = np.array(desc_out.strides)[dims_out].tolist()
        if strides_ain[-1] != 1:
            raise ValueError("Matrices with column strides greater than 1 are unsupported")

        if desc_bin is desc_out:
            raise ValueError("Overwriting input B is not supported")

        return (shape_ain, desc_ain.dtype, strides_ain, shape_bin, desc_bin.dtype, strides_bin, shape_out,
                desc_out.dtype, strides_out, shape_out[0], shape_out[1])
