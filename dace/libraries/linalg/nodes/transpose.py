# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import functools
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import symbolic
from dace.libraries.blas import blas_helpers
from dace import symbolic
from dace.libraries.blas import environments as blas_environments
from dace.libraries.standard.environments.tiled_transpose import TiledTranspose
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.sdfg import scope
from dace.transformation.transformation import ExpandTransformation
import warnings


def _get_transpose_input(node, state, sdfg):
    """Returns the transpose input edge, array, and shape."""
    for edge in state.in_edges(node):
        if edge.dst_conn == "_inp":
            size, idx = blas_helpers.matrix_view(edge.data.subset)
            outer_array = sdfg.data(dace.sdfg.find_input_arraynode(state, edge).data)
            return edge, outer_array, (size[0], size[1]), (outer_array.strides[idx[0]], outer_array.strides[idx[1]])
    raise ValueError("Transpose input connector \"_inp\" not found.")


def _get_transpose_output(node, state, sdfg):
    """Returns the transpose output edge, array, and shape."""
    for edge in state.out_edges(node):
        if edge.src_conn == "_out":
            size, idx = blas_helpers.matrix_view(edge.data.subset)
            outer_array = sdfg.data(dace.sdfg.find_output_arraynode(state, edge).data)
            return edge, outer_array, (size[0], size[1]), (outer_array.strides[idx[0]], outer_array.strides[idx[1]])
    raise ValueError("Transpose output connector \"_out\" not found.")


def _is_single_element(node, state, sdfg) -> bool:
    """``True`` when the operand provably holds exactly one element.

    A 1x1 transpose is a one-element copy, but every BLAS path below takes ``_inp`` / ``_out`` as
    BUFFERS, and the codegen types a single-element subset as a SCALAR -- so the emitted
    ``omatcopy`` / ``geam`` call is handed a value where it declared a pointer and the build fails.
    :class:`ExpandTransposePure` already carries the one-element tasklet for exactly this case.

    A symbolic extent answers ``None``, which reads as False: an extent that is not KNOWN to be one
    keeps the BLAS path it has today.
    """
    _, _, (m, n), _ = _get_transpose_input(node, state, sdfg)
    return symbolic.equal(m * n, 1) is True


def _is_blas_packed(node, state, sdfg) -> bool:
    """``True`` when both operands have the packed layout the omatcopy / geam calls below assume.

    Those calls state the leading dimension as the operand's own extent -- ``lda = n`` for the
    ``m x n`` input, ``ldb = m`` for the ``n x m`` output -- which is the stride of a matrix packed
    row after row with nothing between. A VIEW is not that: ``pos[:, 0:1]`` is an ``(N, 1)`` operand
    whose consecutive elements sit ``3`` apart, and the call then walks the wrong memory and answers
    plausible wrong numbers (the npbench ``nbody`` regression). The innermost stride is checked for
    the same reason and admits no leading dimension at all: no ``ld`` can express a gap INSIDE a row.

    A stride that is only symbolically equal to the packed one answers ``None``, which reads as
    False and keeps the operand on the element-wise expansion -- the safe side.
    """
    _, _, (m, n), (in_row, in_elem) = _get_transpose_input(node, state, sdfg)
    _, _, _, (out_row, out_elem) = _get_transpose_output(node, state, sdfg)
    return all(
        symbolic.equal(have, want) is True for have, want in ((in_elem, 1), (out_elem, 1), (in_row, n), (out_row, m)))


@dace.library.expansion
class ExpandTransposePure(ExpandTransformation):

    environments = []

    @staticmethod
    def map_schedule(node, parent_state, parent_sdfg):
        """Schedule for the element-wise map this expansion builds.

        Schedule inference has already run by the time a library node expands, so a map left at the
        default schedule is whatever surrounds it -- and on device-resident data at host level that
        is a host loop over ``GPU_Global`` memory. npbench nbody lands here exactly: ``pos[:, 0:1]``
        is a strided view, so every packed-layout expansion declines it and this one takes over.
        Inside a kernel the opposite holds: everything below is device code already, and a nested
        device map is not allowed.
        """
        if scope.is_devicelevel_gpu(parent_sdfg, parent_state, node):
            return dace.dtypes.ScheduleType.Sequential
        operands = [parent_state.in_edges(node)[0].data.data, parent_state.out_edges(node)[0].data.data]
        if any(parent_sdfg.arrays[name].storage in GPU_RESIDENT_STORAGES for name in operands):
            return dace.dtypes.ScheduleType.GPU_Device
        return dace.dtypes.ScheduleType.Default

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):

        in_edge, in_outer_array, in_shape, in_strides = _get_transpose_input(node, parent_state, parent_sdfg)
        out_edge, out_outer_array, out_shape, out_strides = _get_transpose_output(node, parent_state, parent_sdfg)
        dtype = node.dtype

        sdfg = dace.SDFG(node.label + "_sdfg")
        state = sdfg.add_state(node.label + "_state")

        _, in_array = sdfg.add_array("_inp", in_shape, dtype, strides=in_strides, storage=in_outer_array.storage)
        _, out_array = sdfg.add_array("_out", out_shape, dtype, strides=out_strides, storage=out_outer_array.storage)

        num_elements = functools.reduce(lambda x, y: x * y, in_array.shape)
        if num_elements == 1:
            inp = state.add_read("_inp")
            out = state.add_write("_out")
            tasklet = state.add_tasklet("transpose", {"__inp"}, {"__out"}, "__out = __inp")
            state.add_edge(inp, None, tasklet, "__inp", dace.memlet.Memlet.from_array("_inp", in_array))
            state.add_edge(tasklet, "__out", out, None, dace.memlet.Memlet.from_array("_out", out_array))
        else:
            state.add_mapped_tasklet(
                name="transpose",
                schedule=ExpandTransposePure.map_schedule(node, parent_state, parent_sdfg),
                map_ranges={
                    "__i%d" % i: "0:%s" % n
                    for i, n in enumerate(in_array.shape)
                },
                inputs={
                    "__inp": dace.memlet.Memlet.simple("_inp",
                                                       ",".join(["__i%d" % i for i in range(len(in_array.shape))]))
                },
                code="__out = __inp",
                outputs={
                    "__out":
                    dace.memlet.Memlet.simple("_out",
                                              ",".join(["__i%d" % i for i in range(len(in_array.shape) - 1, -1, -1)]))
                },
                external_edges=True)

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        return ExpandTransposePure.make_sdfg(node, state, sdfg)


@dace.library.expansion
class ExpandTransposeMKL(ExpandTransformation):

    environments = [blas_environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)

        # A one-element operand is a copy, not a BLAS call (see :func:`_is_single_element`).
        if _is_single_element(node, state, sdfg):
            return ExpandTransposePure.make_sdfg(node, state, sdfg)

        # A strided operand is not the packed matrix the call states (see :func:`_is_blas_packed`).
        if not _is_blas_packed(node, state, sdfg):
            return ExpandTransposePure.make_sdfg(node, state, sdfg)

        # Fall back to native implementation if input and output types are not the same
        if (sdfg.arrays[list(state.in_edges_by_connector(node, '_inp'))[0].data.data].dtype
                != sdfg.arrays[list(state.out_edges_by_connector(node, '_out'))[0].data.data].dtype):
            return ExpandTransposePure.make_sdfg(node, state, sdfg)

        dtype = node.dtype
        if dtype == dace.float32:
            func = "somatcopy"
            alpha = "1.0f"
            cast = ''
        elif dtype == dace.float64:
            func = "domatcopy"
            alpha = "1.0"
            cast = ''
        elif dtype == dace.complex64:
            func = "comatcopy"
            alpha = "*(MKL_Complex8*)dace::blas::BlasConstants::Get().Complex64Pone()"
            cast = '(MKL_Complex8*)'
        elif dtype == dace.complex128:
            func = "zomatcopy"
            alpha = "*(MKL_Complex16*)dace::blas::BlasConstants::Get().Complex128Pone()"
            cast = '(MKL_Complex16*)'
        else:
            warnings.warn("Unsupported type for MKL omatcopy extension: " + str(dtype) + ", falling back to pure")
            return ExpandTransposePure.expansion(node, state, sdfg)

        _, _, (m, n), _ = _get_transpose_input(node, state, sdfg)
        code = ("mkl_{f}('R', 'T', {m}, {n}, {a}, {cast}_inp, "
                "{n}, {cast}_out, {m});").format(f=func, m=m, n=n, a=alpha, cast=cast)
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandTransposeOpenBLAS(ExpandTransformation):

    environments = [blas_environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)

        # A one-element operand is a copy, not a BLAS call (see :func:`_is_single_element`).
        if _is_single_element(node, state, sdfg):
            return ExpandTransposePure.make_sdfg(node, state, sdfg)

        # A strided operand is not the packed matrix the call states (see :func:`_is_blas_packed`).
        if not _is_blas_packed(node, state, sdfg):
            return ExpandTransposePure.make_sdfg(node, state, sdfg)

        # Fall back to native implementation if input and output types are not the same
        if (sdfg.arrays[list(state.in_edges_by_connector(node, '_inp'))[0].data.data].dtype
                != sdfg.arrays[list(state.out_edges_by_connector(node, '_out'))[0].data.data].dtype):
            return ExpandTransposePure.make_sdfg(node, state, sdfg)

        dtype = node.dtype
        cast = ""
        if dtype == dace.float32:
            func = "somatcopy"
            alpha = "1.0f"
            cast = ''
        elif dtype == dace.float64:
            func = "domatcopy"
            alpha = "1.0"
            cast = ''
        elif dtype == dace.complex64:
            func = "comatcopy"
            alpha = "dace::blas::BlasConstants::Get().Complex64Pone()"
            cast = '(float*)'
        elif dtype == dace.complex128:
            func = "zomatcopy"
            alpha = "dace::blas::BlasConstants::Get().Complex128Pone()"
            cast = '(double*)'
        else:
            # OpenBLAS omatcopy only covers the four BLAS floating types; any other
            # element type (e.g. an int64 index/count grid) falls back to the native
            # element-wise transpose, matching the differing-type fallback above.
            return ExpandTransposePure.make_sdfg(node, state, sdfg)
        _, _, (m, n), _ = _get_transpose_input(node, state, sdfg)
        # Adaptations for BLAS API
        order = 'CblasRowMajor'
        trans = 'CblasTrans'
        code = ("cblas_{f}({o}, {t}, {m}, {n}, {cast}{a}, {cast}_inp, "
                "{n}, {cast}_out, {m});").format(f=func, o=order, t=trans, m=m, n=n, a=alpha, cast=cast)
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandTransposeGPUBLAS(ExpandTransformation):
    """Transpose as a vendor ``?geam`` with alpha=1, beta=0; the two backends differ only below."""

    environments = []

    @classmethod
    def expansion(cls, node, state, sdfg, **kwargs):
        node.validate(sdfg, state)
        dtype = node.dtype

        # A one-element operand is a copy, not a BLAS call (see :func:`_is_single_element`).
        if _is_single_element(node, state, sdfg):
            return ExpandTransposePure.make_sdfg(node, state, sdfg)

        # A strided operand is not the packed matrix the call states (see :func:`_is_blas_packed`).
        if not _is_blas_packed(node, state, sdfg):
            return ExpandTransposePure.make_sdfg(node, state, sdfg)

        # Fall back to native implementation if input and output types are not the same
        if (sdfg.arrays[list(state.in_edges_by_connector(node, '_inp'))[0].data.data].dtype
                != sdfg.arrays[list(state.out_edges_by_connector(node, '_out'))[0].data.data].dtype):
            return ExpandTransposePure.make_sdfg(node, state, sdfg)

        try:
            func, cdtype, factort = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandTransposePure.expansion(node, state, sdfg, **kwargs)

        func = func + 'geam'

        alpha = f"__state->{cls.handle_field}.Constants().{factort}Pone()"
        beta = f"__state->{cls.handle_field}.Constants().{factort}Zero()"
        _, _, (m, n), (istride, _) = _get_transpose_input(node, state, sdfg)
        _, _, _, (ostride, _) = _get_transpose_output(node, state, sdfg)

        code = (cls.environments[0].handle_setup_code(node) + f"""{cls.check_error}({cls.funcname(func)}(
                    {cls.handle}, {cls.op('T')}, {cls.op('N')},
                    {m}, {n}, {alpha}, ({cdtype}*)_inp, {n}, {beta}, ({cdtype}*)_inp, {m}, ({cdtype}*)_out, {m}));
                """)

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)

        return tasklet


@dace.library.expansion
class ExpandTransposeCuBLAS(ExpandTransposeGPUBLAS):
    environments = [blas_environments.cublas.cuBLAS]
    handle = "__dace_cublas_handle"
    handle_field = "cublas_handle"
    check_error = "dace::blas::CheckCublasError"

    @classmethod
    def op(cls, mode: str) -> str:
        return f"CUBLAS_OP_{mode}"

    @classmethod
    def funcname(cls, func: str) -> str:
        return f"cublas{func}"


@dace.library.expansion
class ExpandTransposeRocBLAS(ExpandTransposeGPUBLAS):
    environments = [blas_environments.rocblas.rocBLAS]
    handle = "__dace_rocblas_handle"
    handle_field = "rocblas_handle"
    check_error = "dace::blas::CheckRocblasError"

    @classmethod
    def op(cls, mode: str) -> str:
        return "rocblas_operation_transpose" if mode == "T" else "rocblas_operation_none"

    @classmethod
    def funcname(cls, func: str) -> str:
        return f"rocblas_{func.lower()}"


@dace.library.expansion
class ExpandTransposeCUDA(ExpandTransformation):
    """Our own tiled kernel: ``dace::cuda_transpose::transpose``.

    Preferred over ``cuBLAS`` ``geam`` for a plain matrix transpose. ``geam`` is a general
    ``alpha*op(A) + beta*op(B)`` and pays for reading a second operand and scaling it; a transpose
    only needs one coalesced read and one coalesced write per element, which is what the 32x32
    shared-memory tile gives. Measured on an RTX 4050 at 88% of a straight device-to-device copy.
    """

    environments = [TiledTranspose]

    @staticmethod
    def expansion(node, state, sdfg, **kwargs):
        from dace.codegen.targets.cpp import sym2cpp
        node.validate(sdfg, state)
        in_edge, in_outer, (m, n), (istride, in_elem) = _get_transpose_input(node, state, sdfg)
        out_edge, out_outer, _, (ostride, out_elem) = _get_transpose_output(node, state, sdfg)
        dtype = node.dtype

        # The kernel indexes rows by a leading dimension and elements by 1, so a non-unit element
        # stride (a transposed VIEW handed in as the operand) is not addressable this way.
        # ``ExpandTransformation.apply`` attaches THIS class's ``environments`` to whatever is
        # returned, so a delegation has to reset them -- the pure expansion needs none, and leaving
        # the CUDA header attached would pull it into a unit that never calls the kernel.
        if symbolic.equal(in_elem, 1) is not True or symbolic.equal(out_elem, 1) is not True:
            ExpandTransposeCUDA.environments = []
            return ExpandTransposePure.expansion(node, state, sdfg, **kwargs)
        if sdfg.arrays[in_edge.data.data].dtype != sdfg.arrays[out_edge.data.data].dtype:
            ExpandTransposeCUDA.environments = []
            return ExpandTransposePure.make_sdfg(node, state, sdfg)
        ExpandTransposeCUDA.environments = [TiledTranspose]

        state_id = state.parent_graph.node_id(state)
        idstr = f'{sdfg.name}_{state_id}_{state.node_id(node)}'
        ctype = dtype.base_type.ctype
        prototype = (f'DACE_EXPORTED gpuError_t __dace_transpose_{idstr}(const {ctype} *__tr_in, {ctype} *__tr_out, '
                     f'int __tr_rows, int __tr_cols, int __tr_ldin, int __tr_ldout, gpuStream_t __tr_stream);')
        sdfg.append_global_code(prototype + '\n')
        # No ``DACE_GPU_CHECK`` in this body: the macro reports through ``__state``, which a free
        # function in the CUDA unit does not have. The status is returned and checked at the call.
        sdfg.append_global_code(
            f'{prototype}\n'
            f'gpuError_t __dace_transpose_{idstr}(const {ctype} *__tr_in, {ctype} *__tr_out, int __tr_rows, '
            f'int __tr_cols, int __tr_ldin, int __tr_ldout, gpuStream_t __tr_stream) {{\n'
            f'    return ::dace::cuda_transpose::transpose<{ctype}>(__tr_in, __tr_out, __tr_rows, __tr_cols, '
            f'__tr_ldin, __tr_ldout, __tr_stream);\n'
            f'}}\n', 'cuda')

        code = (f'DACE_GPU_CHECK(__dace_transpose_{idstr}(_inp, _out, (int)({sym2cpp(m)}), (int)({sym2cpp(n)}), '
                f'(int)({sym2cpp(istride)}), (int)({sym2cpp(ostride)}), __dace_current_stream));')
        return dace.sdfg.nodes.Tasklet(node.name,
                                       node.in_connectors,
                                       node.out_connectors,
                                       code,
                                       language=dace.dtypes.Language.CPP)


@dace.library.node
class Transpose(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandTransposePure,
        "MKL": ExpandTransposeMKL,
        "OpenBLAS": ExpandTransposeOpenBLAS,
        "cuBLAS": ExpandTransposeCuBLAS,
        "rocBLAS": ExpandTransposeRocBLAS,
        "CUDA": ExpandTransposeCUDA,
    }
    default_implementation = 'pure'

    dtype = dace.properties.TypeClassProperty(allow_none=True)

    def __init__(self, name, dtype=None, location=None):
        super().__init__(name, location=location, inputs={'_inp'}, outputs={'_out'})
        self.dtype = dtype

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) != 1:
            raise ValueError("Expected exactly one input to transpose operation")
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == '_inp':
                in_size, _ = blas_helpers.matrix_view(memlet.subset)
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from transpose operation")
        out_memlet = out_edges[0].data
        if len(in_size) != 2:
            raise ValueError("Transpose operation only supported on matrices")
        out_size, _ = blas_helpers.matrix_view(out_memlet.subset)
        if len(out_size) != 2:
            raise ValueError("Transpose operation only supported on matrices")
        if list(out_size) != [in_size[1], in_size[0]]:
            raise ValueError("Output to transpose operation must agree in the m and n dimensions")
