# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``numpy.linalg.eigh``: eigenvalues and eigenvectors of a symmetric or Hermitian matrix.

``_a`` (n x n) in, ``_w`` (n, ascending, real) and ``_v`` (n x n, eigenvectors as columns) out. Only
the triangle named by ``lower`` is read, as numpy's ``UPLO`` does. The vendor expansions call
``?syevd`` / ``?heevd`` (:class:`~dace.libraries.lapack.nodes.syevd.Syevd`); ``pure`` is a cyclic
Jacobi sweep for a build with no LAPACK at all.
"""
import copy
from typing import Any, List, NamedTuple

import numpy as np

import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import SDFG, Memlet, SDFGState, dtypes, symbolic
from dace.libraries.blas import environments as blas_environments
from dace.libraries.lapack import environments
from dace.libraries.lapack.nodes.syevd import Syevd
from dace.libraries.linalg.nodes.cholesky import GPU_SOLVERS, SOLVER_BLAS, device_solver_implementation
from dace.libraries.linalg.nodes.solve import restride
from dace.libraries.linalg.nodes.transpose import Transpose
from dace.libraries.standard.helper import host_accessible_info_storage
from dace.transformation.transformation import ExpandTransformation
from dace.optionals import required
from dace.sdfg.narrowing import as_range, as_typeclass

#: Jacobi sweeps before the pure expansion gives up; cyclic Jacobi converges quadratically, so a
#: well-scaled matrix needs well under twenty.
MAX_JACOBI_SWEEPS = 100

#: The eigenvalue type of each matrix type.
REAL_TYPE: dict[dtypes.typeclass, dtypes.typeclass] = {
    as_typeclass(dtypes.complex64): as_typeclass(dtypes.float32),
    as_typeclass(dtypes.complex128): as_typeclass(dtypes.float64),
}


class Operand(NamedTuple):
    """One connector of an ``Eigh`` node, as its expansion must declare it."""
    dtype: dtypes.typeclass
    storage: dtypes.StorageType
    shape: list
    strides: list


def drop_unused_outputs(node: 'Eigh', state: SDFGState, sdfg: SDFG) -> None:
    """Make each output nothing reads a transient of ``sdfg`` and drop it from ``node``.

    Dead-dataflow elimination removes the write of an unread eigenvector matrix (``eigvalsh``, or
    ``w, _ = eigh(a)``), but every expansion still has to produce it.
    """
    used = {e.src_conn for e in state.out_edges(node)}
    for conn in ('_w', '_v'):
        if conn not in used:
            sdfg.arrays[conn].transient = True
            node.remove_out_connector(conn)


def add_transpose(state: SDFGState, name: str, src: dace.sdfg.nodes.AccessNode, dst: dace.sdfg.nodes.AccessNode,
                  implementation: str) -> None:
    """``dst = src^T`` between two access nodes, on the vendor BLAS of the GPU solver ``implementation``."""
    src_desc, dst_desc = src.desc(state.sdfg), dst.desc(state.sdfg)
    transpose = Transpose(name, dtype=src_desc.dtype)
    transpose.implementation = SOLVER_BLAS[implementation]
    state.add_edge(src, None, transpose, '_inp', Memlet.from_array(src.data, src_desc))
    state.add_edge(transpose, '_out', dst, None, Memlet.from_array(dst.data, dst_desc))


def make_vendor_sdfg(node: 'Eigh', parent_state: SDFGState, parent_sdfg: SDFG, implementation: str) -> SDFG:
    """The eigensolve as one ``Syevd`` node, with the layout staging each implementation needs.

    LAPACKE reads the row-major operand directly, so the matrix is copied into ``_v`` and solved in
    place. The GPU solvers are column-major: the matrix is transposed into a scratch buffer (which
    is also the copy the solver destroys), solved there, and the column-major eigenvectors are
    transposed back into ``_v``. Both transposes are plain, not conjugating, so a complex Hermitian
    operand needs no special case.
    """
    a_op, w_op, v_op = node.validate(parent_sdfg, parent_state)
    dtype, storage = a_op.dtype, a_op.storage
    sdfg = dace.SDFG(f"{node.label}_sdfg")
    a_arr = sdfg.add_array('_a', a_op.shape, dtype, strides=a_op.strides, storage=storage)
    w_arr = sdfg.add_array('_w', w_op.shape, w_op.dtype, strides=w_op.strides, storage=w_op.storage)
    v_arr = sdfg.add_array('_v', v_op.shape, dtype, strides=v_op.strides, storage=v_op.storage)
    info_arr = sdfg.add_array('_info', [1], dace.int32, transient=True, storage=host_accessible_info_storage(storage))
    drop_unused_outputs(node, parent_state, sdfg)
    state = sdfg.add_state(f"{node.label}_state")

    syevd = Syevd('syevd', lower=node.lower)
    syevd.implementation = implementation
    a, v = state.add_read('_a'), state.add_write('_v')
    if implementation in GPU_SOLVERS:
        work_arr = sdfg.add_array('_vt', a_op.shape, dtype, transient=True, storage=storage)
        work_in, work_out = state.add_access('_vt'), state.add_access('_vt')
        add_transpose(state, 'AT', a, work_in, implementation)
        add_transpose(state, 'VT', work_out, v, implementation)
    else:
        work_arr = v_arr
        work_in, work_out = state.add_access('_v'), v
        state.add_nedge(a, work_in, Memlet.from_array(*a_arr))
    state.add_edge(work_in, None, syevd, '_xin', Memlet.from_array(*work_arr))
    state.add_edge(syevd, '_xout', work_out, None, Memlet.from_array(*work_arr))
    state.add_edge(syevd, '_evals', state.add_write('_w'), None, Memlet.from_array(*w_arr))
    state.add_edge(syevd, '_res', state.add_write('_info'), None, Memlet.from_array(*info_arr))
    return sdfg


def jacobi_rotation(dtype: dtypes.typeclass, n: symbolic.SymbolicType) -> Any:
    """The unitary rotation of ``work`` (and the accumulated ``vectors``) that zeroes ``work[p, q]``:
    the phase of ``work[p, q]``, then a real Jacobi angle."""

    @dace.program
    def rotate(work: dtype[n, n], vectors: dtype[n, n], p: dace.int64, q: dace.int64):
        magnitude = np.abs(work[p, q])
        if magnitude > 0.0:
            phase = work[p, q] / magnitude
            tau = (np.real(work[q, q]) - np.real(work[p, p])) / (2.0 * magnitude)
            sign = 1.0
            if tau < 0.0:
                sign = -1.0
            t = sign / (np.abs(tau) + np.sqrt(tau * tau + 1.0))
            c = 1.0 / np.sqrt(t * t + 1.0)
            s = t * c
            for k in range(n):
                akp = work[k, p]
                akq = work[k, q]
                work[k, p] = c * akp - s * np.conj(phase) * akq
                work[k, q] = s * phase * akp + c * akq
            for k in range(n):
                apk = work[p, k]
                aqk = work[q, k]
                work[p, k] = c * apk - s * phase * aqk
                work[q, k] = s * np.conj(phase) * apk + c * aqk
            for k in range(n):
                vkp = vectors[k, p]
                vkq = vectors[k, q]
                vectors[k, p] = c * vkp - s * np.conj(phase) * vkq
                vectors[k, q] = s * phase * vkp + c * vkq

    return rotate


def ascending_sort(dtype: dtypes.typeclass, wtype: dtypes.typeclass, n: symbolic.SymbolicType) -> Any:
    """Selection sort of the eigenvalues ``_w``, permuting the eigenvector columns with them."""

    @dace.program
    def sort_ascending(_w: wtype[n], vectors: dtype[n, n]):
        for i in range(n):
            smallest = i
            for j in range(i + 1, n):
                if _w[j] < _w[smallest]:
                    smallest = j
            if smallest != i:
                swap = _w[i]
                _w[i] = _w[smallest]
                _w[smallest] = swap
                for k in range(n):
                    vswap = vectors[k, i]
                    vectors[k, i] = vectors[k, smallest]
                    vectors[k, smallest] = vswap

    return sort_ascending


def jacobi_program(dtype: dtypes.typeclass, wtype: dtypes.typeclass, n: symbolic.SymbolicType, lower: bool) -> Any:
    """``eigh`` of an ``n`` x ``n`` matrix by cyclic Jacobi, as a program over ``_a``, ``_w``, ``_v``."""
    tolerance = float(np.finfo(wtype.type).eps)**2
    rotate = jacobi_rotation(dtype, n)
    sort_ascending = ascending_sort(dtype, wtype, n)

    @dace.program
    def eigh_pure(_a: dtype[n, n], _w: wtype[n], _v: dtype[n, n]):
        work = dace.define_local([n, n], dtype)
        vectors = dace.define_local([n, n], dtype)
        vectors[:] = 0
        for i in range(n):
            for j in range(i + 1):
                # Entry (i, j) of the lower triangle, read from whichever triangle is stored.
                entry = _a[i, j] if lower else np.conj(_a[j, i])
                work[i, j] = entry
                work[j, i] = np.conj(entry)
            work[i, i] = np.real(work[i, i])
            vectors[i, i] = 1
        scale = 0.0
        for i in range(n):
            for j in range(n):
                scale = scale + np.real(work[i, j] * np.conj(work[i, j]))
        for _ in range(MAX_JACOBI_SWEEPS):
            off = 0.0
            for p in range(n):
                for q in range(p + 1, n):
                    off = off + np.real(work[p, q] * np.conj(work[p, q]))
            if off <= tolerance * scale:
                break
            for p in range(n):
                for q in range(p + 1, n):
                    rotate(work, vectors, p, q)
        for i in range(n):
            _w[i] = np.real(work[i, i])
        sort_ascending(_w, vectors)
        for i, j in dace.map[0:n, 0:n]:
            _v[i, j] = vectors[i, j]

    return eigh_pure


@dace.library.expansion
class ExpandEighPure(ExpandTransformation):
    """Cyclic Jacobi as loops and tasklets, with no library behind it.

    Correct rather than fast: every sweep rotates each off-diagonal pair to zero with a unitary
    rotation (the phase of ``a[p, q]``, then a real Jacobi angle), until the off-diagonal mass falls
    to rounding level relative to the matrix. The eigenvalues are then sorted ascending and the
    eigenvector columns permuted with them. A build with LAPACK or a GPU solver takes those instead.
    """

    environments: List[type] = []

    @staticmethod
    def expansion(node: 'Eigh', parent_state: SDFGState, parent_sdfg: SDFG, **kwargs: Any) -> SDFG:
        a_op, w_op, v_op = node.validate(parent_sdfg, parent_state)
        dtype, wtype = a_op.dtype, w_op.dtype
        n = a_op.shape[0]
        nsdfg = jacobi_program(dtype, wtype, n, node.lower).to_sdfg(simplify=True)
        restride(nsdfg, (('_a', a_op.shape, a_op.strides), ('_v', v_op.shape, v_op.strides)), dtype)
        restride(nsdfg, (('_w', w_op.shape, w_op.strides), ), wtype)
        drop_unused_outputs(node, parent_state, nsdfg)
        return nsdfg


@dace.library.expansion
class ExpandEighOpenBLAS(ExpandTransformation):

    environments = [blas_environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node: 'Eigh', parent_state: SDFGState, parent_sdfg: SDFG, **kwargs: Any) -> SDFG:
        return make_vendor_sdfg(node, parent_state, parent_sdfg, "OpenBLAS")


@dace.library.expansion
class ExpandEighMKL(ExpandTransformation):

    environments = [blas_environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(node: 'Eigh', parent_state: SDFGState, parent_sdfg: SDFG, **kwargs: Any) -> SDFG:
        return make_vendor_sdfg(node, parent_state, parent_sdfg, "MKL")


@dace.library.expansion
class ExpandEighCuSolverDn(ExpandTransformation):

    environments = [environments.cusolverdn.cuSolverDn]

    @staticmethod
    def expansion(node: 'Eigh', parent_state: SDFGState, parent_sdfg: SDFG, **kwargs: Any) -> SDFG:
        return make_vendor_sdfg(node, parent_state, parent_sdfg, "cuSolverDn")


@dace.library.expansion
class ExpandEighRocSolver(ExpandTransformation):

    environments = [environments.rocsolver.rocSOLVER]

    @staticmethod
    def expansion(node: 'Eigh', parent_state: SDFGState, parent_sdfg: SDFG, **kwargs: Any) -> SDFG:
        return make_vendor_sdfg(node, parent_state, parent_sdfg, "rocSOLVER")


@dace.library.node
class Eigh(dace.sdfg.nodes.LibraryNode):
    """``w, v = numpy.linalg.eigh(a, UPLO)``: ``_a`` in, ``_w`` and ``_v`` out."""

    implementations = {
        "pure": ExpandEighPure,
        "OpenBLAS": ExpandEighOpenBLAS,
        "MKL": ExpandEighMKL,
        "cuSolverDn": ExpandEighCuSolverDn,
        "rocSOLVER": ExpandEighRocSolver
    }
    default_implementation = None

    lower = dace.properties.Property(dtype=bool, default=True, desc="Read the lower triangle (numpy UPLO='L')")

    def __init__(self, name: str, lower: bool = True, **kwargs: Any) -> None:
        super().__init__(name, inputs={"_a"}, outputs={"_w", "_v"}, **kwargs)
        self.lower = lower

    def expand(self, state_or_sdfg: SDFGState | SDFG, *args: Any, **kwargs: Any) -> str:
        if self.implementation is None:
            state = state_or_sdfg if isinstance(state_or_sdfg, SDFGState) else args[0]
            self.implementation = device_solver_implementation(self, state, "_a")
        return super().expand(state_or_sdfg, *args, **kwargs)

    def validate(self, sdfg: SDFG, state: SDFGState) -> tuple[Operand, Operand, Operand]:
        """:return: the :class:`Operand` of ``_a``, ``_w`` and ``_v``, squeezed to the matrix.

        An output nothing reads (see :func:`drop_unused_outputs`) is described as a packed buffer in
        the storage of ``_a``.
        """
        memlets = {e.dst_conn: e.data for e in state.in_edges(self) if e.dst_conn == "_a"}
        memlets.update({e.src_conn: e.data for e in state.out_edges(self) if e.src_conn in ("_w", "_v")})
        if "_a" not in memlets:
            raise ValueError("eigh needs an _a input")
        operands = {}
        for conn, memlet in memlets.items():
            subset = copy.deepcopy(memlet.subset)
            dims = as_range(subset).squeeze()
            desc = sdfg.arrays[required(memlet.data)]
            operands[conn] = Operand(desc.dtype, desc.storage, as_range(subset).size(), [desc.strides[d] for d in dims])
        a_op = operands["_a"]
        if len(a_op.shape) != 2 or symbolic.equal(a_op.shape[0], a_op.shape[1]) is False:
            raise ValueError("eigh needs a square matrix")
        n = a_op.shape[0]
        real = REAL_TYPE[a_op.dtype] if a_op.dtype in REAL_TYPE else a_op.dtype
        w_op = operands.get("_w", Operand(real, a_op.storage, [n], [1]))
        v_op = operands.get("_v", Operand(a_op.dtype, a_op.storage, [n, n], [n, 1]))
        if len(w_op.shape) != 1 or len(v_op.shape) != 2:
            raise ValueError("eigh writes a vector of eigenvalues and a matrix of eigenvectors")
        if v_op.dtype != a_op.dtype:
            raise ValueError("eigh eigenvectors must have the type of the matrix")
        if w_op.dtype != real:
            raise ValueError(f"eigh eigenvalues of a {a_op.dtype} matrix must be {real}, not {w_op.dtype}")
        return a_op, w_op, v_op
