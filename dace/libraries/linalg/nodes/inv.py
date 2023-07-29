# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace
import dace.library
import dace.properties
import dace.sdfg.nodes
import numpy as np
from dace import Memlet
from dace.libraries.lapack.nodes import Getrf, Getri, Getrs
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.lapack import environments
from dace.libraries.blas import environments as blas_environments


def _make_sdfg(node, parent_state, parent_sdfg, implementation):

    arr_desc = node.validate(parent_sdfg, parent_state)
    if node.overwrite:
        in_shape, in_dtype, in_strides, n = arr_desc
    else:
        (in_shape, in_dtype, in_strides, out_shape, out_dtype, out_strides, n) = arr_desc
    dtype = in_dtype

    sdfg = dace.SDFG("{l}_sdfg".format(l=node.label))

    a_arr = sdfg.add_array('_ain', in_shape, dtype=in_dtype, strides=in_strides)
    if not node.overwrite:
        ain_arr = a_arr
        a_arr = sdfg.add_array('_aout', out_shape, dtype=out_dtype, strides=out_strides)
    ipiv_arr = sdfg.add_array('_pivots', [n], dtype=dace.int32, transient=True)
    info_arr = sdfg.add_array('_info', [1], dtype=dace.int32, transient=True)

    state = sdfg.add_state("{l}_state".format(l=node.label))

    getrf_node = Getrf('getrf')
    getrf_node.implementation = implementation
    getri_node = Getri('getri')
    getri_node.implementation = implementation

    if node.overwrite:
        ain = state.add_read('_ain')
        ainout = state.add_access('_ain')
        aout = state.add_write('_ain')
    else:
        a = state.add_read('_ain')
        ain = state.add_read('_aout')
        ainout = state.add_access('_aout')
        aout = state.add_write('_aout')
        state.add_nedge(a, ain, Memlet.from_array(*ain_arr))

    ipiv = state.add_access('_pivots')
    info1 = state.add_write('_info')
    info2 = state.add_write('_info')

    state.add_memlet_path(ain, getrf_node, dst_conn="_xin", memlet=Memlet.from_array(*a_arr))
    state.add_memlet_path(getrf_node, info1, src_conn="_res", memlet=Memlet.from_array(*info_arr))
    state.add_memlet_path(getrf_node, ipiv, src_conn="_ipiv", memlet=Memlet.from_array(*ipiv_arr))
    state.add_memlet_path(getrf_node, ainout, src_conn="_xout", memlet=Memlet.from_array(*a_arr))
    state.add_memlet_path(ainout, getri_node, dst_conn="_xin", memlet=Memlet.from_array(*a_arr))
    state.add_memlet_path(ipiv, getri_node, dst_conn="_ipiv", memlet=Memlet.from_array(*ipiv_arr))
    state.add_memlet_path(getri_node, info2, src_conn="_res", memlet=Memlet.from_array(*info_arr))
    state.add_memlet_path(getri_node, aout, src_conn="_xout", memlet=Memlet.from_array(*a_arr))

    return sdfg


def _make_sdfg_getrs(node, parent_state, parent_sdfg, implementation):

    arr_desc = node.validate(parent_sdfg, parent_state)
    if node.overwrite:
        in_shape, in_dtype, in_strides, n = arr_desc
    else:
        (in_shape, in_dtype, in_strides, out_shape, out_dtype, out_strides, n) = arr_desc
    dtype = in_dtype

    sdfg = dace.SDFG("{l}_sdfg".format(l=node.label))

    a_arr = sdfg.add_array('_ain', in_shape, dtype=in_dtype, strides=in_strides)
    if not node.overwrite:
        ain_arr = a_arr
        a_arr = sdfg.add_array('_ainout', [n, n], dtype=in_dtype, transient=True)
        b_arr = sdfg.add_array('_aout', out_shape, dtype=out_dtype, strides=out_strides)
    else:
        b_arr = sdfg.add_array('_b', [n, n], dtype=dtype, transient=True)
    ipiv_arr = sdfg.add_array('_pivots', [n], dtype=dace.int32, transient=True)
    info_arr = sdfg.add_array('_info', [1], dtype=dace.int32, transient=True)

    state = sdfg.add_state("{l}_state".format(l=node.label))

    getrf_node = Getrf('getrf')
    getrf_node.implementation = implementation
    getrs_node = Getrs('getrs')
    getrs_node.implementation = implementation

    if node.overwrite:
        ain = state.add_read('_ain')
        ainout = state.add_access('_ain')
        aout = state.add_write('_ain')
        bin_name = '_b'
        bout = state.add_write('_b')
        state.add_nedge(bout, aout, Memlet.from_array(*a_arr))
    else:
        a = state.add_read('_ain')
        ain = state.add_read('_ainout')
        ainout = state.add_access('_ainout')
        # aout = state.add_write('_aout')
        state.add_nedge(a, ain, Memlet.from_array(*ain_arr))
        bin_name = '_aout'
        bout = state.add_access('_aout')

    _, _, mx = state.add_mapped_tasklet('_eye_',
                                        dict(i=f"0:{n}", j=f"0:{n}"), {},
                                        '_out = (i == j) ? 1 : 0;',
                                        dict(_out=Memlet.simple(bin_name, 'i, j')),
                                        language=dace.dtypes.Language.CPP,
                                        external_edges=True)
    bin = state.out_edges(mx)[0].dst

    ipiv = state.add_access('_pivots')
    info1 = state.add_write('_info')
    info2 = state.add_write('_info')

    state.add_memlet_path(ain, getrf_node, dst_conn="_xin", memlet=Memlet.from_array(*a_arr))
    state.add_memlet_path(getrf_node, info1, src_conn="_res", memlet=Memlet.from_array(*info_arr))
    state.add_memlet_path(getrf_node, ipiv, src_conn="_ipiv", memlet=Memlet.from_array(*ipiv_arr))
    state.add_memlet_path(getrf_node, ainout, src_conn="_xout", memlet=Memlet.from_array(*a_arr))
    state.add_memlet_path(ainout, getrs_node, dst_conn="_a", memlet=Memlet.from_array(*a_arr))
    state.add_memlet_path(bin, getrs_node, dst_conn="_rhs_in", memlet=Memlet.from_array(*b_arr))
    state.add_memlet_path(ipiv, getrs_node, dst_conn="_ipiv", memlet=Memlet.from_array(*ipiv_arr))
    state.add_memlet_path(getrs_node, info2, src_conn="_res", memlet=Memlet.from_array(*info_arr))
    state.add_memlet_path(getrs_node, bout, src_conn="_rhs_out", memlet=Memlet.from_array(*b_arr))

    return sdfg


@dace.library.expansion
class ExpandInvPure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        raise NotImplementedError("Missing pure implementation of linalg.inv.")

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) + ".")
        return ExpandInvPure.make_sdfg(node, state, sdfg)


@dace.library.expansion
class ExpandInvOpenBLAS(ExpandTransformation):

    environments = [blas_environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        if node.use_getri:
            return _make_sdfg(node, parent_state, parent_sdfg, "OpenBLAS")
        else:
            return _make_sdfg_getrs(node, parent_state, parent_sdfg, "OpenBLAS")


@dace.library.expansion
class ExpandInvMKL(ExpandTransformation):

    environments = [blas_environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        if node.use_getri:
            return _make_sdfg(node, parent_state, parent_sdfg, "MKL")
        else:
            return _make_sdfg_getrs(node, parent_state, parent_sdfg, "MKL")


@dace.library.expansion
class ExpandInvCuSolverDn(ExpandTransformation):

    environments = [environments.cusolverdn.cuSolverDn]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return _make_sdfg_getrs(node, parent_state, parent_sdfg, "cuSolverDn")


@dace.library.node
class Inv(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {"OpenBLAS": ExpandInvOpenBLAS, "MKL": ExpandInvMKL, "cuSolverDn": ExpandInvCuSolverDn}
    default_implementation = None

    overwrite = dace.properties.Property(dtype=bool, default=False)
    use_getri = dace.properties.Property(dtype=bool, default=True)

    # Object fields
    def __init__(self, name, overwrite_a=False, use_getri=True, *args, **kwargs):
        super().__init__(name, *args, inputs={"_ain"}, outputs={"_aout"}, **kwargs)
        self.overwrite = overwrite_a
        self.use_getri = use_getri

    def validate(self, sdfg, state):
        """
        :return: A four-tuple (ain, aout, ipiv, info) of the three data
                 descriptors in the parent SDFG.
        """

        in_edges = state.in_edges(self)
        if len(in_edges) != 1:
            raise ValueError("Expected exactly one input to inv")
        in_memlet = in_edges[0].data
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from inv")
        out_memlet = out_edges[0].data

        # Squeeze input memlets
        squeezed1 = copy.deepcopy(in_memlet.subset)
        dims1 = squeezed1.squeeze()
        # Squeeze output memlets
        squeezed2 = copy.deepcopy(out_memlet.subset)
        dims2 = squeezed2.squeeze()

        desc_ain, desc_aout = None, None,
        for e in state.in_edges(self):
            if e.dst_conn == "_ain":
                desc_ain = sdfg.arrays[e.data.data]
        for e in state.out_edges(self):
            if e.src_conn == "_aout":
                desc_aout = sdfg.arrays[e.data.data]

        if desc_ain.dtype.base_type != desc_aout.dtype.base_type:
            raise ValueError("Basetype of input and output must be equal!")

        if len(squeezed1.size()) != 2 or len(squeezed2.size()) != 2:
            raise ValueError("linalg.inv only supported on matrices")

        shape1 = squeezed1.size()
        shape2 = squeezed2.size()
        if shape1[0] != shape1[1]:
            raise ValueError("linalg.inv only supported on square matrices")
        if not np.array_equal(shape1, shape2):
            raise ValueError("Squeezed shape of input and output must be the same")

        strides1 = np.array(desc_ain.strides)[dims1].tolist()
        strides2 = np.array(desc_aout.strides)[dims2].tolist()
        if strides2[-1] != 1:
            raise ValueError("Matrices with column strides greater than 1 are unsupported")

        if self.overwrite and desc_ain is not desc_aout:
            raise ValueError("Overwrite enabled but output is different than input")
        if not self.overwrite and desc_ain is desc_aout:
            raise ValueError("Overwrite disabled but output is same as input")

        if self.overwrite:
            return shape1, desc_ain.dtype, strides1, shape1[0]
        else:
            return (shape1, desc_ain.dtype, strides1, shape2, desc_aout.dtype, strides2, shape1[0])
