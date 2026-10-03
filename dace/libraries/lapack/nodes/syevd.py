# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LAPACK ``?SYEVD`` / ``?HEEVD`` library node: all eigenvalues and eigenvectors of a symmetric or
Hermitian matrix, by divide and conquer (the driver ``numpy.linalg.eigh`` calls).

In place, like :class:`~dace.libraries.lapack.nodes.potrf.Potrf`: ``_xin`` and ``_xout`` name one
buffer, which holds the matrix on entry and the eigenvectors (as columns) on exit. ``_evals`` receives
the eigenvalues in ascending order and ``_res`` the info code. The CPU expansions read the matrix
row-major (``LAPACK_ROW_MAJOR``) and the GPU solvers column-major, as for the other LAPACK nodes.

A nonzero info code -- an illegal argument, or a tridiagonal eigensolve that did not converge -- is
raised as ``std::runtime_error`` by the expansion itself, so no caller can read eigenvectors that
were never computed. On the device that check costs one stream synchronization.
"""
import copy
from typing import Any

import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import SDFG, SDFGState, dtypes, symbolic
from dace.libraries.blas import blas_helpers
from dace.libraries.blas import environments as blas_environments
from dace.libraries.lapack import environments
from dace.ordered import OrderedSet
from dace.transformation.transformation import ExpandTransformation

#: The eigenvalue type of each vendor matrix type: the real type underneath a complex one.
REAL_CTYPE = {'cuComplex': 'float', 'cuDoubleComplex': 'double'}


def lapack_driver(dtype: dtypes.typeclass) -> str:
    """``syevd`` for a real ``dtype``, ``heevd`` for a complex one."""
    return 'heevd' if dtype in (dtypes.complex64, dtypes.complex128) else 'syevd'


def info_error(func: str, info: str) -> str:
    """C++ statement raising a nonzero info code ``info`` of solver call ``func``."""
    return (f'if ({info} != 0) throw std::runtime_error(std::string("{func} failed with info ") + '
            f'std::to_string({info}));\n')


@dace.library.expansion
class ExpandSyevdOpenBLAS(ExpandTransformation):

    environments = [blas_environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node: 'Syevd', parent_state: SDFGState, parent_sdfg: SDFG, **kwargs: Any) -> dace.sdfg.nodes.Tasklet:
        dtype, n, lda = node.validate(parent_sdfg, parent_state)
        lapack_dtype = blas_helpers.to_blastype(dtype.type).lower()
        cast = {"c": "(lapack_complex_float*)", "z": "(lapack_complex_double*)"}.get(lapack_dtype, "")
        func = f'LAPACKE_{lapack_dtype}{lapack_driver(dtype)}'
        uplo = "'L'" if node.lower else "'U'"
        code = (f"_res = {func}(LAPACK_ROW_MAJOR, 'V', {uplo}, {n}, {cast}_xin, {lda}, _evals);\n" +
                info_error(func, '_res'))
        return dace.sdfg.nodes.Tasklet(node.name,
                                       node.in_connectors,
                                       node.out_connectors,
                                       code,
                                       language=dace.dtypes.Language.CPP)


@dace.library.expansion
class ExpandSyevdMKL(ExpandTransformation):

    environments = [blas_environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args: Any, **kwargs: Any) -> dace.sdfg.nodes.Tasklet:
        return ExpandSyevdOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandSyevdGPUSolver(ExpandTransformation):
    """Eigendecomposition on a vendor GPU solver, column-major.

    The dialects differ in the call only: cuSolverDn sizes a workspace through ``*_bufferSize``,
    rocSOLVER takes a length-``n`` buffer for the off-diagonal of its tridiagonal form. Either way
    the info code lands in ``_res``, which the caller keeps in host-accessible memory, and is read
    once the stream has drained.
    """

    environments = []

    @classmethod
    def expansion(cls, node: 'Syevd', parent_state: SDFGState, parent_sdfg: SDFG,
                  **kwargs: Any) -> dace.sdfg.nodes.Tasklet:
        dtype, n, lda = node.validate(parent_sdfg, parent_state)
        letter, ctype, _ = blas_helpers.cublas_type_metadata(dtype)
        func = letter + lapack_driver(dtype)
        matrix = f"{cls.fill_enum(node.lower)}, {n}, ({cls.cast_ctype(ctype)}*)_xin, {lda}"
        code = (cls.environments[0].handle_setup_code(node) + cls.call(func, ctype, matrix, n) +
                'DACE_GPU_CHECK(gpuStreamSynchronize(__dace_current_stream));\n' + info_error(func, '*_res'))
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        tasklet.out_connectors = {
            c: (dtypes.pointer(dtypes.int32) if c == '_res' else t)
            for c, t in tasklet.out_connectors.items()
        }
        return tasklet

    @classmethod
    def fill_enum(cls, lower: bool) -> str:
        raise NotImplementedError

    @classmethod
    def cast_ctype(cls, ctype: str) -> str:
        return ctype

    @classmethod
    def call(cls, func: str, ctype: str, matrix: str, n: symbolic.SymbolicType) -> str:
        """The solver call on the ``matrix`` arguments (fill mode, order, pointer, leading dimension)."""
        raise NotImplementedError


@dace.library.expansion
class ExpandSyevdCuSolverDn(ExpandSyevdGPUSolver):
    environments = [environments.cusolverdn.cuSolverDn]

    @classmethod
    def fill_enum(cls, lower: bool) -> str:
        return "CUBLAS_FILL_MODE_LOWER" if lower else "CUBLAS_FILL_MODE_UPPER"

    @classmethod
    def call(cls, func: str, ctype: str, matrix: str, n: symbolic.SymbolicType) -> str:
        args = f"__dace_cusolverDn_handle, CUSOLVER_EIG_MODE_VECTOR, {matrix}, ({REAL_CTYPE.get(ctype, ctype)}*)_evals"
        return f"""
            int __dace_workspace_size = 0;
            {ctype}* __dace_workspace;
            dace::lapack::CheckCusolverDnError(cusolverDn{func}_bufferSize({args}, &__dace_workspace_size));
            gpuMalloc<{ctype}>(&__dace_workspace, sizeof({ctype}) * __dace_workspace_size);
            dace::lapack::CheckCusolverDnError(
                cusolverDn{func}({args}, __dace_workspace, __dace_workspace_size, _res));
            gpuFree(__dace_workspace);
            """


@dace.library.expansion
class ExpandSyevdRocSolver(ExpandSyevdGPUSolver):
    environments = [environments.rocsolver.rocSOLVER]

    @classmethod
    def fill_enum(cls, lower: bool) -> str:
        return "rocblas_fill_lower" if lower else "rocblas_fill_upper"

    @classmethod
    def cast_ctype(cls, ctype: str) -> str:
        return blas_helpers.rocblas_type(ctype)

    @classmethod
    def call(cls, func: str, ctype: str, matrix: str, n: symbolic.SymbolicType) -> str:
        wtype = REAL_CTYPE.get(ctype, ctype)
        return f"""
            {wtype}* __dace_offdiagonal;
            gpuMalloc<{wtype}>(&__dace_offdiagonal, sizeof({wtype}) * ({n}));
            dace::lapack::CheckRocsolverError(rocsolver_{func.lower()}(
                __dace_rocblas_handle, rocblas_evect_original, {matrix}, ({wtype}*)_evals, __dace_offdiagonal, _res));
            gpuFree(__dace_offdiagonal);
            """


@dace.library.node
class Syevd(dace.sdfg.nodes.LibraryNode):
    """LAPACK ``?SYEVD`` / ``?HEEVD``, eigenvalues and eigenvectors (``jobz = 'V'``).

    Inputs: ``_xin``. Outputs: ``_xout`` (the eigenvectors, same buffer as ``_xin``), ``_evals``
    (eigenvalues, ascending, of the real type of the matrix), ``_res`` (info).
    """

    implementations = {
        "OpenBLAS": ExpandSyevdOpenBLAS,
        "MKL": ExpandSyevdMKL,
        "cuSolverDn": ExpandSyevdCuSolverDn,
        "rocSOLVER": ExpandSyevdRocSolver
    }
    default_implementation = None

    lower = dace.properties.Property(dtype=bool, default=True, desc="Read the lower triangle of the matrix")

    def __init__(self, name: str, lower: bool = True, **kwargs: Any) -> None:
        super().__init__(name, inputs={"_xin"}, outputs=OrderedSet(("_xout", "_evals", "_res")), **kwargs)
        self.lower = lower

    def validate(self, sdfg: SDFG, state: SDFGState) -> tuple[dtypes.typeclass, Any, Any]:
        """:return: ``(dtype, n, lda)`` of the matrix operand."""
        edges = [e for e in state.in_edges(self) if e.dst_conn == "_xin"]
        if len(edges) != 1:
            raise ValueError("syevd expects exactly one _xin input")
        subset = copy.deepcopy(edges[0].data.subset)
        dims = subset.squeeze()
        desc = sdfg.arrays[edges[0].data.data]
        if len(subset.size()) != 2:
            raise ValueError("syevd only supports 2-dimensional matrices")
        n, cols = subset.size()
        if symbolic.equal(n, cols) is False:
            raise ValueError("syevd needs a square matrix")
        if desc.dtype.veclen > 1:
            raise NotImplementedError("syevd does not support vector types")
        if symbolic.equal(desc.strides[dims[1]], 1) is not True:
            raise NotImplementedError("syevd needs a unit stride along the matrix rows")
        return desc.dtype.base_type, n, desc.strides[dims[0]]
