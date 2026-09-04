# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace
import dace.library
import dace.properties
import dace.sdfg.nodes
import numpy as np

from dace import Memlet, SDFG, SDFGState
from dace import symbolic
from dace.libraries.standard.helper import host_accessible_info_storage
from dace.libraries.lapack import Getrf, Getrs
from dace.libraries.linalg.nodes.transpose import Transpose
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.lapack import environments
from dace.libraries.blas import environments as blas_environments
from dace.ordered import OrderedSet


def gesv_core_program(dtype, n, rhs):
    """``dace.program`` solving ``a x = b`` in place, for a given element type and shape.

    A factory rather than a plain program because a ``dace.program``'s annotations need concrete
    types, and because ``linalg.inv`` runs the same elimination against the identity -- the
    algorithm is written once and both expansions call it.

    :param dtype: element type of both operands.
    :param n: matrix order.
    :param rhs: number of right-hand sides.
    :returns: the program, which overwrites ``a`` with its factorization and ``x`` with the solution.
    """

    @dace.program
    def gesv_core(a: dtype[n, n], x: dtype[n, rhs]):
        for k in range(n):
            pivot = k
            for i in range(k + 1, n):
                if abs(a[i, k]) > abs(a[pivot, k]):
                    pivot = i
            if pivot != k:
                for j in range(n):
                    swap_a = a[k, j]
                    a[k, j] = a[pivot, j]
                    a[pivot, j] = swap_a
                for j in range(rhs):
                    swap_x = x[k, j]
                    x[k, j] = x[pivot, j]
                    x[pivot, j] = swap_x
            for i in range(k + 1, n):
                factor = a[i, k] / a[k, k]
                a[i, k] = 0
                for j in range(k + 1, n):
                    a[i, j] = a[i, j] - factor * a[k, j]
                for j in range(rhs):
                    x[i, j] = x[i, j] - factor * x[k, j]
        for step in range(n):
            i = n - 1 - step
            for j in range(rhs):
                total = x[i, j]
                for m in range(i + 1, n):
                    total = total - a[i, m] * x[m, j]
                x[i, j] = total / a[i, i]

    return gesv_core


def restride(nsdfg, connectors, dtype):
    """Give ``nsdfg``'s connector arrays the strides the caller's containers actually have.

    A pure expansion built from a ``dace.program`` gets contiguous strides, which is wrong whenever
    the library node reads a strided slice of a bigger array -- the elements are then read from the
    wrong addresses, and the result is silently numeric garbage rather than an error.

    :param nsdfg: the expansion SDFG, modified in place.
    :param connectors: ``(name, shape, strides)`` per connector.
    :param dtype: element type of every connector.
    """
    for name, shape, strides in connectors:
        if len(strides) != len(shape):
            # The caller passed a slice of a higher-rank array whose descriptor was never squeezed
            # down to the connector's rank, so which strides belong to the slice is not recoverable
            # here. Refuse: reading it as contiguous would be numeric garbage, not an error.
            raise NotImplementedError('%s is a rank-%d slice of a rank-%d container; pass it as an array of its '
                                      'own rank.' % (name, len(shape), len(strides)))
        transient = nsdfg.arrays[name].transient
        nsdfg.remove_data(name, validate=False)
        nsdfg.add_array(name, shape, dtype=dtype, strides=strides, transient=transient)


#: The vendor GPU solvers. The branches below are about whether the factorization runs ON THE
#: DEVICE -- which decides the column-major transposes and the device-side info code -- and not
#: about which vendor it is, so a second backend must not need a second branch.
GPU_SOLVERS = ("cuSolverDn", "rocSOLVER")

#: The vendor BLAS that goes with each, for the transposes staged around the factorization. Naming
#: a cuBLAS transpose inside a rocSOLVER graph selects an expansion whose environment is not
#: installed, and every one of them silently drops to the serial loop.
SOLVER_BLAS = {"cuSolverDn": "cuBLAS", "rocSOLVER": "rocBLAS"}


def _make_sdfg_getrs(node: 'Solve', parent_state, parent_sdfg, implementation):

    arr_desc = node.validate(parent_sdfg, parent_state)
    (ain_shape, ain_dtype, ain_strides, bin_shape, bin_dtype, bin_strides, out_shape, out_dtype, out_strides, n, rhs,
     storage) = arr_desc

    # ``validate`` squeezes the memlets, so an (n, 1) right-hand side arrives here rank-1 like a
    # plain vector. One column is contiguous in either layout, so it is staged as a rank-1
    # ``_binout`` and getrs reads nrhs == 1 off the rank.
    single_rhs = len(out_shape) == 1

    sdfg = dace.SDFG("{l}_sdfg".format(l=node.label))

    ain_arr = sdfg.add_array('_ain', ain_shape, dtype=ain_dtype, strides=ain_strides)
    ainout_arr = sdfg.add_array('_ainout', [n, n], dtype=ain_dtype, transient=True, storage=storage)
    bin_arr = sdfg.add_array('_bin', bin_shape, dtype=bin_dtype, strides=bin_strides)
    if single_rhs:
        binout_shape = [n]
    elif implementation in GPU_SOLVERS:
        binout_shape = [rhs, n]
    else:
        binout_shape = [n, rhs]
    binout_arr = sdfg.add_array('_binout', binout_shape, dtype=out_dtype, transient=True, storage=storage)
    bout_arr = sdfg.add_array('_bout', out_shape, dtype=out_dtype, strides=out_strides)
    ipiv_arr = sdfg.add_array('_pivots', [n], dtype=dace.int32, transient=True, storage=storage)
    # cuSOLVER writes ``devInfo`` through a raw pointer; the status scalar is host-checkable, so it
    # must live in host-accessible (pinned) memory instead of the input's GPU_Global storage.
    info_arr = sdfg.add_array('_info', [1],
                              dtype=dace.int32,
                              transient=True,
                              storage=host_accessible_info_storage(storage))

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
    if implementation in GPU_SOLVERS:
        transpose_ain = Transpose('AT', dtype=ain_dtype)
        transpose_ain.implementation = SOLVER_BLAS[implementation]
        state.add_edge(ain, None, transpose_ain, '_inp', Memlet.from_array(*ain_arr))
        state.add_edge(transpose_ain, '_out', ainout1, None, Memlet.from_array(*ainout_arr))
    else:
        state.add_nedge(ain, ainout1, Memlet.from_array(*ain_arr))

    if implementation in GPU_SOLVERS and not single_rhs:
        transpose_bin = Transpose('bT', dtype=bin_dtype)
        transpose_bin.implementation = SOLVER_BLAS[implementation]
        state.add_edge(bin, None, transpose_bin, '_inp', Memlet.from_array(*bin_arr))
        state.add_edge(transpose_bin, '_out', binout1, None, Memlet.from_array(*binout_arr))
        transpose_out = Transpose('XT', dtype=bin_dtype)
        transpose_out.implementation = SOLVER_BLAS[implementation]
        state.add_edge(binout2, None, transpose_out, '_inp', Memlet.from_array(*binout_arr))
        state.add_edge(transpose_out, '_out', bout, None, Memlet.from_array(*bout_arr))
    else:
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
    """``A x = b`` as loops and tasklets, with no library behind it.

    Gaussian elimination with partial pivoting, then back substitution -- the same thing LAPACK's
    ``?GESV`` does, spelled so that MPR can render it into a translation unit that links against
    nothing. Correct rather than fast; a build with MKL or OpenBLAS should keep using them.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        (shape_ain, dtype, strides_ain, shape_bin, _, strides_bin, shape_out, _, strides_out, n, rhs,
         _) = node.validate(parent_sdfg, parent_state)

        gesv_core = gesv_core_program(dtype, n, rhs)

        # ``overwrite`` is always False, so the elimination runs on a copy; a single right-hand side
        # is widened to one column so the core has one shape to handle rather than two.
        if len(shape_bin) == 1:

            @dace.program
            def solve_pure(_ain: dtype[n, n], _bin: dtype[n], _bout: dtype[n]):
                work = dace.define_local([n, n], dtype)
                work[:] = _ain
                columns = dace.define_local([n, rhs], dtype)
                for i in dace.map[0:n]:
                    columns[i, 0] = _bin[i]
                gesv_core(work, columns)
                for i in dace.map[0:n]:
                    _bout[i] = columns[i, 0]
        else:

            @dace.program
            def solve_pure(_ain: dtype[n, n], _bin: dtype[n, rhs], _bout: dtype[n, rhs]):
                work = dace.define_local([n, n], dtype)
                work[:] = _ain
                columns = dace.define_local([n, rhs], dtype)
                columns[:] = _bin
                gesv_core(work, columns)
                _bout[:] = columns

        nsdfg = solve_pure.to_sdfg(simplify=True)
        # ``to_sdfg`` gives the connectors contiguous strides, but they are VIEWS of the caller's
        # containers and may be strided slices of a larger array. Restating them is what makes the
        # expansion read the same elements the vendor path does.
        restride(nsdfg, (('_ain', shape_ain, strides_ain), ('_bin', shape_bin, strides_bin),
                         ('_bout', shape_out, strides_out)), dtype)
        return nsdfg


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


@dace.library.expansion
class ExpandSolveRocSolver(ExpandTransformation):

    environments = [environments.rocsolver.rocSOLVER]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return _make_sdfg_getrs(node, parent_state, parent_sdfg, "rocSOLVER")


@dace.library.node
class Solve(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandSolvePure,
        "OpenBLAS": ExpandSolveOpenBLAS,
        "MKL": ExpandSolveMKL,
        "cuSolverDn": ExpandSolveCuSolverDn,
        "rocSOLVER": ExpandSolveRocSolver
    }
    default_implementation = None

    overwrite = dace.properties.Property(dtype=bool, default=False)

    # Object fields
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs=OrderedSet(('_ain', '_bin')), outputs={"_bout"}, **kwargs)
        # NOTE: We currently do not support overwrite == True
        self.overwrite = False

    def validate(
        self, sdfg: SDFG, state: SDFGState
    ) -> tuple[list[symbolic.SymbolicType], dace.dtypes.typeclass, list[symbolic.SymbolicType],
               list[symbolic.SymbolicType], dace.dtypes.typeclass, list[symbolic.SymbolicType],
               list[symbolic.SymbolicType], dace.dtypes.typeclass, list[symbolic.SymbolicType], symbolic.SymbolicType,
               symbolic.SymbolicType, dace.dtypes.StorageType]:
        """
        :return: A tuple containing shapes, dtypes, strides, sizes, and storage:
                 (ain_shape, ain_dtype, ain_strides, bin_shape, bin_dtype, bin_strides,
                  out_shape, out_dtype, out_strides, n, rhs, storage).
        """

        in_edges = state.in_edges(self)
        if len(in_edges) != 2:
            raise ValueError("Expected exactly two inputs to solve")
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from solve")

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
            raise ValueError("linalg.solve only supported with first input a "
                             "square matrix")
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

        # A single right-hand side is a VECTOR, whose squeezed shape has no second entry; it is one
        # column. Reading shape_out[1] unconditionally raised IndexError for that case.
        return (shape_ain, desc_ain.dtype, strides_ain, shape_bin, desc_bin.dtype, strides_bin, shape_out,
                desc_out.dtype, strides_out, shape_out[0], shape_out[1] if len(shape_out) > 1 else 1, desc_ain.storage)
