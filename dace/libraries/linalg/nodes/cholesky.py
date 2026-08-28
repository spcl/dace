# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import math

import dace.library
from dace.codegen import common
import dace.properties
import dace.sdfg.nodes
from dace import dtypes

from dace import Memlet
from dace.libraries.lapack import Potrf
from dace.libraries.linalg.nodes.solve import restride
from dace.libraries.linalg.nodes.transpose import Transpose
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.lapack import environments
from dace.libraries.blas import environments as blas_environments

#: The vendor GPU solvers. The branches below are about whether the factorization runs ON THE
#: DEVICE -- which decides the column-major transposes and the device-side info code -- and not
#: about which vendor it is, so a second backend must not need a second branch.
GPU_SOLVERS = ("cuSolverDn", "rocSOLVER")

#: The vendor BLAS that goes with each, for the transposes staged around the factorization. Naming
#: a cuBLAS transpose inside a rocSOLVER graph selects an expansion whose environment is not
#: installed, and every one of them silently drops to the serial loop.
SOLVER_BLAS = {"cuSolverDn": "cuBLAS", "rocSOLVER": "rocBLAS"}


def _make_sdfg(node, parent_state, parent_sdfg, implementation):

    inp_desc, inp_shape, out_desc, out_shape = node.validate(parent_sdfg, parent_state)
    dtype = inp_desc.dtype
    storage = inp_desc.storage

    sdfg = dace.SDFG("{l}_sdfg".format(l=node.label))

    ain_arr = sdfg.add_array('_a', inp_shape, dtype=dtype, strides=inp_desc.strides)
    bout_arr = sdfg.add_array('_b', out_shape, dtype=dtype, strides=out_desc.strides)
    # cuSolverDn writes the LAPACK info code via a device pointer, so ``_info``
    # must stay on the GPU. We additionally allocate ``_info_host`` on the CPU
    # and connect an implicit edge ``_info -> _info_host`` so the new GPU
    # pipeline's InsertExplicitGPUGlobalMemoryCopies lowers it to an explicit
    # D2H copy -- the host then has a readable status code.
    info_arr = sdfg.add_array('_info', [1], dtype=dace.int32, transient=True, storage=storage)
    if implementation in GPU_SOLVERS:
        info_host_arr = sdfg.add_array('_info_host', [1],
                                       dtype=dace.int32,
                                       transient=True,
                                       storage=dtypes.StorageType.CPU_Heap)
        binout_arr = sdfg.add_array('_bt', inp_shape, dtype=dtype, transient=True, storage=storage)
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
    if implementation in GPU_SOLVERS:
        binout1 = state.add_access('_bt')
        binout2 = state.add_access('_bt')
        binout3 = state.in_edges(me)[0].src
        bout = state.out_edges(mx)[0].dst
        transpose_ain = Transpose('AT', dtype=dtype)
        transpose_ain.implementation = SOLVER_BLAS[implementation]
        state.add_edge(ain, None, transpose_ain, '_inp', Memlet.from_array(*ain_arr))
        state.add_edge(transpose_ain, '_out', binout1, None, Memlet.from_array(*binout_arr))
        transpose_out = Transpose('BT', dtype=dtype)
        transpose_out.implementation = SOLVER_BLAS[implementation]
        state.add_edge(binout2, None, transpose_out, '_inp', Memlet.from_array(*binout_arr))
        state.add_edge(transpose_out, '_out', binout3, None, Memlet.from_array(*bout_arr))
    else:
        binout1 = state.add_access('_b')
        binout2 = state.in_edges(me)[0].src
        binout3 = state.out_edges(mx)[0].dst
        state.add_nedge(ain, binout1, Memlet.from_array(*ain_arr))

    info = state.add_access('_info')

    state.add_memlet_path(binout1, potrf_node, dst_conn="_xin", memlet=Memlet.from_array(*binout_arr))
    state.add_memlet_path(potrf_node, info, src_conn="_res", memlet=Memlet.from_array(*info_arr))
    state.add_memlet_path(potrf_node, binout2, src_conn="_xout", memlet=Memlet.from_array(*binout_arr))

    if implementation in GPU_SOLVERS:
        info_host = state.add_write('_info_host')
        state.add_nedge(info, info_host, Memlet.from_array(*info_host_arr))

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
class ExpandCholeskyPure(ExpandTransformation):
    """Cholesky as loops and tasklets, with no library behind it.

    Exists so a Cholesky can be rendered, read and edited on its own -- MPR emits one translation
    unit with no BLAS to link, and an SDFG that reaches a vendor implementation cannot be rendered
    at all. It is the textbook right-looking factorization, so it is correct rather than fast; a
    build that has MKL or OpenBLAS should keep using them.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        inp_desc, inp_shape, out_desc, out_shape = node.validate(parent_sdfg, parent_state)
        dtype = inp_desc.dtype
        n = inp_shape[0]
        lower = node.lower

        @dace.program
        def cholesky_pure(_a: dtype[n, n], _b: dtype[n, n]):
            factor = dace.define_local([n, n], dtype)
            factor[:] = 0  # the strict upper triangle is never assigned, and is read back on the copy
            for j in range(n):
                diagonal = _a[j, j]
                for k in range(j):
                    diagonal = diagonal - factor[j, k] * factor[j, k]
                factor[j, j] = math.sqrt(diagonal)
                for i in range(j + 1, n):
                    off = _a[i, j]
                    for k in range(j):
                        off = off - factor[i, k] * factor[j, k]
                    factor[i, j] = off / factor[j, j]
            # ``factor`` is always the LOWER triangle; ``lower=False`` asks for the upper one, which
            # is its transpose (A = L L^T = U^T U), so the orientation is a copy and not a second
            # factorization.
            for i, j in dace.map[0:n, 0:n]:
                _b[i, j] = factor[i, j] if lower else factor[j, i]

        nsdfg = cholesky_pure.to_sdfg(simplify=True)
        # See ``restride``: a connector may be a strided slice of a bigger array, and a contiguous
        # reading of it is silently wrong rather than an error.
        restride(nsdfg, (('_a', inp_shape, inp_desc.strides), ('_b', out_shape, out_desc.strides)), dtype)
        return nsdfg


@dace.library.expansion
class ExpandCholeskyOpenBLAS(ExpandTransformation):

    environments = [blas_environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return _make_sdfg(node, parent_state, parent_sdfg, "OpenBLAS")


@dace.library.expansion
class ExpandCholeskyMKL(ExpandTransformation):

    environments = [blas_environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return _make_sdfg(node, parent_state, parent_sdfg, "MKL")


@dace.library.expansion
class ExpandCholeskyCuSolverDn(ExpandTransformation):

    environments = [environments.cusolverdn.cuSolverDn]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return _make_sdfg(node, parent_state, parent_sdfg, "cuSolverDn")


@dace.library.expansion
class ExpandCholeskyRocSolver(ExpandTransformation):

    environments = [environments.rocsolver.rocSOLVER]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return _make_sdfg(node, parent_state, parent_sdfg, "rocSOLVER")


@dace.library.node
class Cholesky(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandCholeskyPure,
        "OpenBLAS": ExpandCholeskyOpenBLAS,
        "MKL": ExpandCholeskyMKL,
        "cuSolverDn": ExpandCholeskyCuSolverDn,
        "rocSOLVER": ExpandCholeskyRocSolver
    }
    default_implementation = None

    lower = dace.properties.Property(dtype=bool, default=True)

    def __init__(self, name, lower=True, *args, **kwargs):
        super().__init__(name, *args, inputs={"_a"}, outputs={
            "_b",
        }, **kwargs)
        self.lower = lower

    def expand(self, state, sdfg=None, *args, **kwargs):
        # Storage-aware auto-pick: the device solver for GPU input, OpenBLAS otherwise.
        # Without this, ``apply_gpu_transformations + expand_library_nodes`` lands
        # on OpenBLAS for a GPU-resident matrix (alphabetical default), which
        # then puts ``_info`` on GPU storage but writes it from a CPU library and
        # fails validation. WHICH device solver follows the configured backend --
        # picking cuSolverDn on an AMD node names an environment that is not
        # installed, which lands back on the CPU library this exists to avoid.
        actual_sdfg = sdfg if (sdfg is not None and not isinstance(sdfg, str)) else state.parent
        if self.implementation is None:
            in_edges = [e for e in state.in_edges(self) if e.dst_conn == "_a"]
            if in_edges:
                outer = state.memlet_path(in_edges[0])[0].src
                if isinstance(outer, dace.sdfg.nodes.AccessNode):
                    if actual_sdfg.arrays[outer.data].storage == dtypes.StorageType.GPU_Global:
                        self.implementation = ('rocSOLVER' if common.get_gpu_backend() == 'hip' else 'cuSolverDn')
        if sdfg is not None:
            return super().expand(state, sdfg, *args, **kwargs)
        return super().expand(state, *args, **kwargs)

    def validate(self, sdfg, state):
        """
        :return: A two-tuple of the input and output descriptors
        """
        # Filter on the data connector -- the GPU stream pipeline may attach
        # a separate ``stream`` in-edge to GPU library nodes which is not part
        # of the data flow and must not be counted here.
        in_edges = [e for e in state.in_edges(self) if e.dst_conn == "_a"]
        if len(in_edges) != 1:
            raise ValueError("Expected exactly one input to pcholesky")
        in_memlet = in_edges[0].data
        out_edges = [e for e in state.out_edges(self) if e.src_conn == "_b"]
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
