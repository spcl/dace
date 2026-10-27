# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LU factorization WITHOUT pivoting, in place: ``A = L @ U`` with ``L`` unit lower triangular.

The factors overwrite the operand the way LAPACK's ``getrf`` leaves them: ``U`` on and above the diagonal, the
strict lower triangle of ``L`` below it, ``L``'s unit diagonal implied. Unlike :class:`~.getrf.Getrf` there is no
pivot vector, so the result is the textbook (Doolittle) elimination, and the node is what a hand-written
no-pivot elimination nest lifts to (``LoopToGetrfNoPivot``).

The operand is ONE in/out connector ``_x``: the factorization reads what it has already written, so the input and
the output are one container by contract, and the pure expansion's nest has to see a single array to keep that
order (two inner names would let an analysis inside the expansion treat the reads as of a read-only input).

Expansions:

- ``pure``: the plain right-looking elimination nest -- per column ``k``, scale the column below the diagonal by
  the pivot, then the rank-1 update of the trailing submatrix -- as sequential loops. Canonicalization expands the
  node on CPU and derives the nest's parallelism itself (the trailing update is a parallel map per ``k``).
- ``cuSolverDn`` / ``hipSolverDn``: the vendor GPU solver. Both are column-major and the operand is row-major, so a
  call on the operand as it lies would factor its TRANSPOSE (``A^T = L' U'`` gives ``A = U'^T L'^T``, a unit UPPER
  factor). The expansion transposes into a column-major scratch with ``geam``, factors that, and transposes back.
  cuSolverDn takes no pivoting when its pivot pointer is ``NULL``; hipSOLVER is the same call renamed.
- ``MKL``: ``LAPACKE_mkl_?getrfnp``, MKL's no-pivot LU (no other CPU LAPACK has one).
"""

import copy

import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import data as dt
from dace import dtypes, symbolic
from dace.libraries import hip_dialect
from dace.libraries.blas import blas_helpers
from dace.libraries.blas import environments as blas_environments
from dace.libraries.lapack import environments
from dace.libraries.standard import pure_components
from dace.memlet import Memlet
from dace.transformation.transformation import ExpandTransformation

#: The in/out connector of the factored matrix.
OPERAND = "_x"


def _edge(node: "GetrfNoPivot", state: dace.SDFGState):
    (edge,) = [e for e in state.in_edges(node) if e.dst_conn == OPERAND]
    return edge


def _same(a, b) -> bool:
    """Whether two extents are one value; through text, so a symbol's dtype does not split one name in two."""
    return symbolic.simplify(symbolic.pystr_to_symbolic(f"({a}) - ({b})")) == 0


@dace.library.expansion
class ExpandGetrfNoPivotPure(ExpandTransformation):
    """The plain right-looking elimination nest over the operand, as sequential loops."""

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        _, n = node.validate(parent_sdfg, parent_state)
        sdfg = dace.SDFG(node.label + "_sdfg")
        pure_components.operand_array(sdfg, OPERAND, _edge(node, parent_state), parent_sdfg)
        k, i, j = "__lu_k", "__lu_i", "__lu_j"

        def at(row: str, col: str) -> Memlet:
            return Memlet(f"{OPERAND}[{row}, {col}]")

        # A[i, k] = A[i, k] / A[k, k] for i > k
        scale = pure_components.counted_loop("lu_scale", i, f"{k} + 1", n)
        scale.add_node(
            pure_components.tasklet_state(
                sdfg, "lu_scale", "out = a / d", {"a": at(i, k), "d": at(k, k)}, {"out": at(i, k)}
            ),
            is_start_block=True,
        )
        # A[i, j] = A[i, j] - A[i, k] * A[k, j] for i, j > k
        rows = pure_components.counted_loop("lu_update_rows", i, f"{k} + 1", n)
        cols = pure_components.counted_loop("lu_update_cols", j, f"{k} + 1", n)
        rows.add_node(cols, is_start_block=True)
        cols.add_node(
            pure_components.tasklet_state(
                sdfg,
                "lu_update",
                "out = a - l * u",
                {"a": at(i, j), "l": at(i, k), "u": at(k, j)},
                {"out": at(i, j)},
            ),
            is_start_block=True,
        )
        column = pure_components.counted_loop("lu_column", k, "0", n)
        pure_components.chain(column, [scale, rows])
        sdfg.add_node(column, is_start_block=True)
        return sdfg


def _wrap_call(node, parent_state, parent_sdfg, code: str, environments_: list) -> dace.SDFG:
    """A one-state SDFG around the CPP tasklet ``code``, which reads the operand as ``_xin`` and writes it as
    ``_xout``: a tasklet cannot hold one connector name on both sides, the node does."""
    sdfg = dace.SDFG(node.label + "_sdfg")
    desc = pure_components.operand_array(sdfg, OPERAND, _edge(node, parent_state), parent_sdfg)
    state = sdfg.add_state(node.label, is_start_block=True)
    tasklet = state.add_tasklet(node.name, {"_xin"}, {"_xout"}, code, language=dtypes.Language.CPP)
    tasklet.environments = {env.full_class_path() for env in environments_}
    full = Memlet.from_array(OPERAND, desc)
    state.add_edge(state.add_read(OPERAND), None, tasklet, "_xin", full)
    state.add_edge(tasklet, "_xout", state.add_write(OPERAND), None, copy.deepcopy(full))
    return sdfg


def _vendor_operand(node, parent_state, parent_sdfg) -> tuple[dt.Data, object, object]:
    """``(descriptor, lda, n)`` of an operand a vendor routine can address: unit column stride."""
    lda, n = node.validate(parent_sdfg, parent_state)
    edge = _edge(node, parent_state)
    desc = parent_sdfg.arrays[edge.data.data]
    dims = copy.deepcopy(edge.data.subset).squeeze()
    if not _same(desc.strides[dims[1]] * edge.data.subset.ranges[dims[1]][2], 1):
        raise NotImplementedError(f"{type(node).__name__}: a vendor routine needs unit column stride")
    if desc.dtype.veclen != 1:
        raise NotImplementedError(f"{type(node).__name__} on vector types")
    return desc, lda, n


@dace.library.expansion
class ExpandGetrfNoPivotMKL(ExpandTransformation):
    """MKL's ``LAPACKE_mkl_?getrfnp``: row-major, in place, no pivoting."""

    environments = [blas_environments.intel_mkl.IntelMKL]

    @classmethod
    def expansion(cls, node, parent_state, parent_sdfg, **kwargs):
        desc, lda, n = _vendor_operand(node, parent_state, parent_sdfg)
        lapack_dtype = blas_helpers.to_blastype(desc.dtype.type).lower()
        cast = {"c": "(MKL_Complex8*)", "z": "(MKL_Complex16*)"}.get(lapack_dtype, "")
        code = f"LAPACKE_mkl_{lapack_dtype}getrfnp(LAPACK_ROW_MAJOR, {n}, {n}, {cast}_xin, {lda});"
        return _wrap_call(node, parent_state, parent_sdfg, code, cls.environments)


@dace.library.expansion
class ExpandGetrfNoPivotGPUSolver(ExpandTransformation):
    """No-pivot LU on a vendor GPU solver, through a column-major scratch copy (see the module docstring).

    Each dialect supplies its handle setup, its ``geam`` spelling, and the factorization call; the transposes,
    the scratch and the info word are shared. The pivot-free factorization itself never fails: a zero pivot
    yields the infinities the loop nest would, and the info word only records where.
    """

    environments = []

    @classmethod
    def expansion(cls, node, parent_state, parent_sdfg, **kwargs):
        desc, lda, n = _vendor_operand(node, parent_state, parent_sdfg)
        func, ctype, factort = blas_helpers.cublas_type_metadata(desc.dtype.base_type)
        ctype = cls.ctype(ctype)
        one = f"__state->{cls.blas_handle_field}.Constants().{factort}Pone()"
        zero = f"__state->{cls.blas_handle_field}.Constants().{factort}Zero()"
        # geam(T, N): C (column-major, ldc) = op(A); B is C itself under beta = 0, which geam allows in place.
        to_scratch = cls.geam(func, n, one, f"({ctype}*)_xin", lda, zero, "__dace_lu", n)
        back = cls.geam(func, n, one, "__dace_lu", n, zero, f"({ctype}*)_xout", lda)
        code = (
            cls.setup(node)
            + f"""
{ctype}* __dace_lu;
int* __dace_lu_info;
gpuMalloc<{ctype}>(&__dace_lu, sizeof({ctype}) * ({n}) * ({n}));
gpuMalloc<int>(&__dace_lu_info, sizeof(int));
{to_scratch}
{cls.factor(func, ctype, n)}
{back}
gpuFree(__dace_lu_info);
gpuFree(__dace_lu);
"""
        )
        return _wrap_call(node, parent_state, parent_sdfg, code, cls.environments)


@dace.library.expansion
class ExpandGetrfNoPivotCuSolverDn(ExpandGetrfNoPivotGPUSolver):
    environments = [environments.cusolverdn.cuSolverDn, blas_environments.cublas.cuBLAS]
    blas_handle_field = "cublas_handle"

    @classmethod
    def ctype(cls, ctype: str) -> str:
        return ctype

    @classmethod
    def setup(cls, node) -> str:
        return environments.cusolverdn.cuSolverDn.handle_setup_code(
            node
        ) + blas_environments.cublas.cuBLAS.handle_setup_code(node)

    @classmethod
    def geam(cls, func, n, one, a, lda, zero, c, ldc) -> str:
        return (
            f"dace::blas::CheckCublasError(cublas{func}geam(__dace_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, "
            f"{n}, {n}, {one}, {a}, {lda}, {zero}, {c}, {ldc}, {c}, {ldc}));"
        )

    @classmethod
    def factor(cls, func, ctype, n) -> str:
        return f"""int __dace_lu_lwork = 0;
cusolverDn{func}getrf_bufferSize(__dace_cusolverDn_handle, {n}, {n}, __dace_lu, {n}, &__dace_lu_lwork);
{ctype}* __dace_lu_work;
gpuMalloc<{ctype}>(&__dace_lu_work, sizeof({ctype}) * __dace_lu_lwork);
// A NULL pivot array is cuSolverDn's no-pivoting mode.
cusolverDn{func}getrf(__dace_cusolverDn_handle, {n}, {n}, __dace_lu, {n}, __dace_lu_work, nullptr, __dace_lu_info);
gpuFree(__dace_lu_work);"""


#: The cuSolverDn expansion stages its operand through cuBLAS ``geam`` transposes, so it is renamed as BLAS code too.
ExpandGetrfNoPivotHipSolverDn = hip_dialect.hip_expansion(
    ExpandGetrfNoPivotCuSolverDn,
    environments.hipsolverdn.hipSolverDn,
    (("cusolverDn", "hipsolverDn"),) + hip_dialect.BLAS,
)
ExpandGetrfNoPivotHipSolverDn.environments = [environments.hipsolverdn.hipSolverDn, blas_environments.hipblas.hipBLAS]


@dace.library.node
class GetrfNoPivot(dace.sdfg.nodes.LibraryNode):
    """In-place LU factorization without pivoting (see the module docstring)."""

    implementations = {
        "pure": ExpandGetrfNoPivotPure,
        "MKL": ExpandGetrfNoPivotMKL,
        "cuSolverDn": ExpandGetrfNoPivotCuSolverDn,
        "hipSolverDn": ExpandGetrfNoPivotHipSolverDn,
    }
    default_implementation = "pure"

    n = dace.properties.SymbolicProperty(allow_none=True, default=None, category="Semantics")

    def __init__(self, name, n=None, *args, **kwargs):
        super().__init__(name, *args, inputs={OPERAND}, outputs={OPERAND}, **kwargs)
        self.n = n

    def validate(self, sdfg, state):
        """
        :return: ``(lda, n)``: the operand's row stride and its order.
        :raises ValueError: unless the one operand is read and written as the same square 2-D region of one
                            container.
        """
        ins = [e for e in state.in_edges(self) if e.dst_conn == OPERAND]
        outs = [e for e in state.out_edges(self) if e.src_conn == OPERAND]
        if len(ins) != 1 or len(outs) != 1 or len(state.in_edges(self)) != 1 or len(state.out_edges(self)) != 1:
            raise ValueError("getrf_nopivot takes exactly one in/out operand _x")
        (inp,), (out,) = ins, outs
        if inp.data.data != out.data.data or inp.data.subset != out.data.subset:
            raise ValueError("getrf_nopivot factors in place: _x must read and write one region of one container")
        squeezed = copy.deepcopy(inp.data.subset)
        dims = squeezed.squeeze()
        if len(dims) != 2:
            raise ValueError("getrf_nopivot only supported on 2-dimensional arrays")
        rows, cols = squeezed.size()
        if not _same(rows, cols):
            raise ValueError(f"getrf_nopivot factors a square matrix, got {rows} x {cols}")
        if self.n is not None and not _same(self.n, rows):
            raise ValueError(f"getrf_nopivot: n = {self.n} but the operand is {rows} x {cols}")
        desc = sdfg.arrays[inp.data.data]
        return desc.strides[dims[0]] * inp.data.subset.ranges[dims[0]][2], rows
