# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace.library
import dace.properties
import dace.sdfg.nodes

from dace import Memlet
from dace.libraries.lapack import Potrf
from dace.libraries.standard import Transpose
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.lapack import environments
from dace.libraries.blas import environments as blas_environments


def _make_sdfg(node, parent_state, parent_sdfg, implementation):

    inp_desc, inp_shape, out_desc, out_shape = node.validate(parent_sdfg, parent_state)
    dtype = inp_desc.dtype

    sdfg = dace.SDFG("{l}_sdfg".format(l=node.label))

    ain_arr = sdfg.add_array('_a', inp_shape, dtype=dtype, strides=inp_desc.strides)
    bout_arr = sdfg.add_array('_b', out_shape, dtype=dtype, strides=out_desc.strides)
    info_arr = sdfg.add_array('_info', [1], dtype=dace.int32, transient=True)
    if implementation == 'cuSolverDn':
        binout_arr = sdfg.add_array('_bt', inp_shape, dtype=dtype, transient=True)
    else:
        binout_arr = bout_arr

    state = sdfg.add_state("{l}_state".format(l=node.label))

    potrf_node = Potrf('potrf', lower=node.lower)
    potrf_node.implementation = implementation

    _, me, mx = state.add_mapped_tasklet('_uzero_',
                                         dict(__i="0:%s" % out_shape[0], __j="0:%s" % out_shape[1]),
                                         dict(_inp=Memlet.simple('_b', '__i, __j')),
                                         '_out = (__i < __j) ? 0 : _inp;',
                                         dict(_out=Memlet.simple('_b', '__i, __j')),
                                         language=dace.dtypes.Language.CPP,
                                         external_edges=True)

    ain = state.add_read('_a')
    if implementation == 'cuSolverDn':
        binout1 = state.add_access('_bt')
        binout2 = state.add_access('_bt')
        binout3 = state.in_edges(me)[0].src
        bout = state.out_edges(mx)[0].dst
        transpose_ain = Transpose('AT', dtype=dtype)
        transpose_ain.implementation = 'cuBLAS'
        state.add_edge(ain, None, transpose_ain, '_inp', Memlet.from_array(*ain_arr))
        state.add_edge(transpose_ain, '_out', binout1, None, Memlet.from_array(*binout_arr))
        transpose_out = Transpose('BT', dtype=dtype)
        transpose_out.implementation = 'cuBLAS'
        state.add_edge(binout2, None, transpose_out, '_inp', Memlet.from_array(*binout_arr))
        state.add_edge(transpose_out, '_out', binout3, None, Memlet.from_array(*bout_arr))
    else:
        binout1 = state.add_access('_b')
        binout2 = state.in_edges(me)[0].src
        binout3 = state.out_edges(mx)[0].dst
        state.add_nedge(ain, binout1, Memlet.from_array(*ain_arr))

    info = state.add_write('_info')

    state.add_memlet_path(binout1, potrf_node, dst_conn="_xin", memlet=Memlet.from_array(*binout_arr))
    state.add_memlet_path(potrf_node, info, src_conn="_res", memlet=Memlet.from_array(*info_arr))
    state.add_memlet_path(potrf_node, binout2, src_conn="_xout", memlet=Memlet.from_array(*binout_arr))

    return sdfg


@dace.library.expansion
class ExpandCholeskyPure(ExpandTransformation):
    """
    Naive backend-agnostic expansion of LAPACK POTRF.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        raise NotImplementedError


@dace.library.expansion
class ExpandCholeskyOpenBLAS(ExpandTransformation):

    environments = [blas_environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return _make_sdfg(node, parent_state, parent_sdfg, "OpenBLAS")


@dace.library.expansion
class ExpandCholeskyMKL(ExpandTransformation):

    environments = [blas_environments.intel_mkl.IntelMKL]

    staticmethod

    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return _make_sdfg(node, parent_state, parent_sdfg, "MKL")


@dace.library.expansion
class ExpandCholeskyCuSolverDn(ExpandTransformation):

    environments = [environments.cusolverdn.cuSolverDn]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        staticmethod

    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return _make_sdfg(node, parent_state, parent_sdfg, "cuSolverDn")


@dace.library.node
class Cholesky(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "OpenBLAS": ExpandCholeskyOpenBLAS,
        "MKL": ExpandCholeskyMKL,
        "cuSolverDn": ExpandCholeskyCuSolverDn
    }
    default_implementation = None

    lower = dace.properties.Property(dtype=bool, default=True)

    def __init__(self, name, lower=True, *args, **kwargs):
        super().__init__(name, *args, inputs={"_a"}, outputs={
            "_b",
        }, **kwargs)
        self.lower = lower

    def validate(self, sdfg, state):
        """
        :return: A two-tuple of the input and output descriptors
        """
        in_edges = state.in_edges(self)
        if len(in_edges) != 1:
            raise ValueError("Expected exactly one input to pcholesky")
        in_memlet = in_edges[0].data
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one input from cholesky node")
        out_memlet = out_edges[0].data

        # Squeeze input memlets
        squeezed1 = copy.deepcopy(in_memlet.subset)
        sqdims1 = squeezed1.squeeze()
        # Squeeze output memlets
        squeezed2 = copy.deepcopy(out_memlet.subset)
        sqdims2 = squeezed2.squeeze()

        desc_ain, desc_aout, = None, None
        for e in state.in_edges(self):
            if e.dst_conn == "_a":
                desc_ain = sdfg.arrays[e.data.data]
        for e in state.out_edges(self):
            if e.src_conn == "_b":
                desc_bout = sdfg.arrays[e.data.data]

        if desc_ain.dtype.base_type != desc_bout.dtype.base_type:
            raise ValueError("Basetype of input and output must be equal!")

        stride_a = desc_ain.strides[sqdims1[0]]
        shape_a = squeezed1.size()
        rows_a = shape_a[0]
        cols_a = shape_a[1]
        stride_b = desc_bout.strides[sqdims2[0]]
        shape_b = squeezed2.size()
        rows_b = shape_b[0]
        cols_b = shape_b[1]

        if len(squeezed1.size()) != 2:
            print(str(squeezed1))
            raise ValueError("Choleksy only supported on 2-dimensional arrays")

        return desc_ain, shape_a, desc_bout, shape_b
