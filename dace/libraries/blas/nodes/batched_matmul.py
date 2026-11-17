# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numbers
import warnings
from copy import deepcopy as dc

import dace.library
import dace.sdfg.nodes
from dace import data as dt
from dace import dtypes, properties
from dace import memlet as mm
from dace.frontend.common import op_repository as oprepo
from dace.libraries import hip_dialect
from dace.libraries.blas import block, environments, gpu_dialect
from dace.libraries.blas.blas_helpers import (
    check_access,
    check_one_device,
    dtype_to_cudadatatype,
    get_gemm_opts,
    packed_unit_extent,
    to_blastype,
    to_cublas_computetype,
)
from dace.libraries.blas.nodes.matmul import _get_batchmm_opts, _get_codegen_gemm_opts, _get_matmul_operands
from dace.libraries.standard.environments.cuda import CUDA
from dace.ordered import OrderedSet
from dace.symbolic import equal, equal_valued, is_small_extent, symstr
from dace.transformation.transformation import ExpandTransformation


def refuse_broadcast_batches(node, state, sdfg) -> None:
    """Refuse a broadcast batch dimension for the fixed-stride BLAS lowerings.

    ``cublas*StridedBatched`` and ``*gemm_batch`` advance each operand by one matrix per batch
    element. A dimension of extent 1 on one side must instead be re-read for every element, which
    no single stride expresses, so such a call would read past the end of the smaller operand.
    The pure expansion handles it by indexing that dimension at 0.

    Refused on the dimensions where broadcasting is what the pairing MEANS: one side provably 1,
    or the two provably different. Two dimensions whose equality is merely unprovable -- distinct
    symbols, ``a: [B, M, K] @ b: [L, K, N]`` -- are not a broadcast. numpy pairs those only when
    they are equal at run time, the same thing the frontend assumed when it gave the result its
    batch shape; refusing them sends every symbolically-batched matmul to a failed expansion.

    :param node: The library node being expanded.
    :param state: The state the node lives in.
    :param sdfg: The SDFG the state belongs to.
    :raise ValueError: If either operand carries a batch dimension that must broadcast.
    """
    (_, _, shape_a, _, _, _), (_, _, shape_b, _, _, _), _ = _get_matmul_operands(node, state, sdfg)
    batch_a, batch_b = shape_a[:-2], shape_b[:-2]
    if not batch_a or not batch_b:
        return
    offset = len(batch_a) - len(batch_b)
    paired = zip(batch_a[offset:], batch_b) if offset >= 0 else zip(batch_a, batch_b[-offset:])
    for d0, d1 in paired:
        same = equal(d0, d1)
        if same is True:
            continue
        if same is False or equal(d0, 1) is True or equal(d1, 1) is True:
            raise ValueError(
                f"{type(node).__name__} cannot broadcast batch dimensions {batch_a} against {batch_b}: "
                'this expansion walks a fixed batch stride. Use the "pure" implementation, or '
                "materialize both operands at the same batch shape."
            )


def shares_out_batch(node, state, sdfg) -> bool:
    """Whether a CPU BLAS lowering runs its batch loop as an OpenMP team, one single-core GEMM per element.

    Only a top-level node whose schedule is an OpenMP team does, and only over a batch that is not ``SMALL``: a
    re-entered node or a ``SMALL`` batch keeps the loop sequential, each threaded BLAS call then the parallel axis.
    Inside the team an OpenMP OpenBLAS runs each call on one thread (a pthreads build already does,
    ``DeferOpenBlasThreadsToOpenMP``).

    :param node: The library node being expanded.
    :param state: The state the node lives in.
    :param sdfg: The SDFG the state belongs to.
    """
    from dace.transformation.auto.auto_optimize import libnode_is_sequential, libnode_runs_multicore  # import cycle

    if libnode_is_sequential(node, state, sdfg) or not libnode_runs_multicore(node):
        return False
    (_, _, shape_a, strides_a, _, _), (_, _, shape_b, strides_b, _, _), _ = _get_matmul_operands(node, state, sdfg)
    cdesc = sdfg.arrays[state.out_edges(node)[0].data.data]
    bopt = _get_batchmm_opts(shape_a, strides_a, shape_b, strides_b, cdesc.shape, cdesc.strides)
    return bool(bopt) and not is_small_extent(bopt["b"], sdfg.facts())


def format_scalar_literal(value, ctype: "dtypes.typeclass") -> str:
    """Render a compile-time alpha/beta value as a DaCe-typed Python-tasklet literal.

    :param value: The scalar property value (real or complex).
    :param ctype: The DaCe typeclass the literal is cast to.
    :return: A ``dace.<type>(...)`` constructor expression, valid inside a Python tasklet body.
    """
    if not isinstance(value, numbers.Number):
        return f"dace.{ctype.to_string()}({value})"
    if ctype.is_complex():
        cvalue = complex(value)
        return f"dace.{ctype.to_string()}({cvalue.real}, {cvalue.imag})"
    return f"dace.{ctype.to_string()}({float(value)})"


#: The optional gather inputs: batch element ``p`` multiplies ``_a[_a_batch_index[p]]`` (``_b`` likewise) into ``_c[p]``.
BATCH_INDEX = {"_a": "_a_batch_index", "_b": "_b_batch_index"}


def gathered_operands(node) -> dict[str, str]:
    """The operands of ``node`` gathered through a batch index, mapped to the index connector."""
    return {operand: index for operand, index in BATCH_INDEX.items() if index in node.in_connectors}


def gathered_sdfg(node, state, sdfg, label: str) -> tuple[dace.SDFG, dict[str, list[str]]]:
    """The nested SDFG of a gathered product: its operands, output and indices as arrays, and per operand the
    subscripts ``__p, __m, __k``-style of batch element ``__p``'s matrix (``__g`` for a gathered pool index)."""
    nsdfg = dace.SDFG(label)
    matrices = {"_a": ["__k", "__m"] if node.transA else ["__m", "__k"]}
    matrices["_b"] = ["__n", "__k"] if node.transB else ["__k", "__n"]
    matrices["_c"] = ["__m", "__n"]
    gathered = gathered_operands(node)
    subscripts = {}
    for name, operand in zip(("_a", "_b", "_c"), _get_matmul_operands(node, state, sdfg)):
        desc, shape, strides = operand[1:4]
        nsdfg.add_array(name, shape, desc.dtype, strides=strides, storage=desc.storage)
        batch = ["__g"] if name in gathered else ["__p"] if len(shape) == 3 else []
        subscripts[name] = batch + matrices[name]
    for edge in state.in_edges(node):
        if edge.dst_conn in gathered.values():
            desc = sdfg.arrays[edge.data.data]
            nsdfg.add_array(edge.dst_conn, edge.data.subset.size(), desc.dtype, storage=desc.storage)
    return nsdfg, subscripts


@dace.library.expansion
class ExpandBatchedMatMulPure(ExpandTransformation):
    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        # Get metadata from parent SDFG
        (
            (edge_a, outer_array_a, shape_a, strides_a, _, _),
            (edge_b, outer_array_b, shape_b, strides_b, _, _),
            cdata,
        ) = _get_matmul_operands(node, parent_state, parent_sdfg)
        outedge = parent_state.out_edges(node)[0]
        cdesc = parent_sdfg.arrays[outedge.data.data]
        bopt = _get_batchmm_opts(shape_a, strides_a, shape_b, strides_b, cdesc.shape, cdesc.strides)

        rows_a, cols_a = shape_a[-1:-3:-1] if node.transA else shape_a[-2:]
        rows_b, cols_b = shape_b[-1:-3:-1] if node.transB else shape_b[-2:]
        res = equal(cols_a, rows_b)
        if res is None:
            warnings.warn(f"First matrix columns {cols_a} may not match second matrix rows {rows_b}", UserWarning)
        elif not res:
            raise SyntaxError("Matrix sizes must match")

        # Determine output shape based on batch options
        if bopt:
            # Use batch dimensions from bopt (may be multi-dimensional)
            batch_dims = bopt.get("batch_dims", [bopt["b"]])
            shape_c = tuple(batch_dims) + (rows_a, cols_b)
        else:
            shape_c = (rows_a, cols_b)

        dtype_a = outer_array_a.dtype.type
        dtype_b = outer_array_b.dtype.type
        dtype_c = cdesc.dtype.type

        check_one_device(outer_array_a, outer_array_b)

        # Create replacement SDFG
        sdfg = dace.SDFG(node.label + "_sdfg")

        _, array_a = sdfg.add_array("_a", shape_a, dtype_a, strides=strides_a, storage=outer_array_a.storage)
        _, array_b = sdfg.add_array("_b", shape_b, dtype_b, strides=strides_b, storage=outer_array_b.storage)
        _, array_c = sdfg.add_array("_c", shape_c, dtype_c, strides=cdata[-3], storage=cdesc.storage)

        # C is read and written in place through the sole "_c" connector -- BatchedMatMul carries
        # no _cin, mirroring Gemm(cin=False). beta==1 leaves C for the WCR add below to accumulate
        # onto directly (no init state at all); beta==0 zeroes it first; any other beta scales the
        # existing value in an init state, reading and writing the same array.
        c_dims = {"_o%d" % i: "0:%s" % symstr(d) for i, d in enumerate(shape_c)}
        c_full_index = ",".join(["_o%d" % i for i in range(len(shape_c))])
        if equal_valued(1, node.beta):
            state = sdfg.add_state(node.label + "_state")
        else:
            init_state = sdfg.add_state(node.label + "_initstate")
            if equal_valued(0, node.beta):
                init_state.add_mapped_tasklet(
                    "batched_matmul_init",
                    c_dims,
                    {},
                    "out = 0",
                    {"out": dace.Memlet.simple("_c", c_full_index)},
                    external_edges=True,
                )
            else:
                beta_lit = format_scalar_literal(node.beta, dace.dtype_to_typeclass(dtype_c))
                init_state.add_mapped_tasklet(
                    "batched_matmul_beta_scale",
                    c_dims,
                    {"inp": dace.Memlet.simple("_c", c_full_index)},
                    f"out = {beta_lit} * inp",
                    {"out": dace.Memlet.simple("_c", c_full_index)},
                    external_edges=True,
                )
            state = sdfg.add_state_after(init_state, node.label + "_state")

        # Calculate number of batch dimensions in output
        num_batch_dims = len(shape_c) - 2

        # Build map parameters: batch dimensions + M, N, K
        map_params = {}
        for i in range(num_batch_dims):
            map_params["__i%d" % i] = f"0:{symstr(shape_c[i])}"

        # M, N, K dimensions
        map_params["__im"] = f"0:{symstr(rows_a)}"
        map_params["__in"] = f"0:{symstr(cols_b)}"
        map_params["__ik"] = f"0:{symstr(cols_a)}"

        def batch_indices(operand_shape) -> str:
            """The operand's batch subscripts, following NumPy's broadcasting rules.

            Batch dimensions align to the RIGHT against the output's, and a dimension of extent 1
            broadcasts, so it is read at 0 for every output batch element. Taking the output index
            for such a dimension -- or aligning from the left -- reads past the end of the operand.
            """
            num_operand_batch = len(operand_shape) - 2
            offset = num_batch_dims - num_operand_batch
            subscripts = []
            for j, dim in enumerate(operand_shape[:-2]):
                broadcast = equal(dim, 1) is True
                subscripts.append("0" if broadcast else "__i%d" % (offset + j))
            return ", ".join(subscripts)

        # For A: if 2D, use [M, K]; if 3D+, use [batch_indices..., M, K]
        matrix_a = "__ik, __im" if node.transA else "__im, __ik"
        memlet_a = matrix_a if len(array_a.shape) == 2 else f"{batch_indices(array_a.shape)}, {matrix_a}"

        # For B: if 2D, use [K, N]; if 3D+, use [batch_indices..., K, N]
        matrix_b = "__in, __ik" if node.transB else "__ik, __in"
        memlet_b = matrix_b if len(array_b.shape) == 2 else f"{batch_indices(array_b.shape)}, {matrix_b}"

        # For C: always has batch dimensions
        c_indices = ", ".join(["__i%d" % i for i in range(num_batch_dims)]) + ", __im, __in"

        # Each product is scaled by alpha, so the summed result is alpha * (A @ B).
        product = "__a * __b"
        if not equal_valued(1, node.alpha):
            alpha_lit = format_scalar_literal(node.alpha, dace.dtype_to_typeclass(dtype_c))
            product = f"{alpha_lit} * {product}"

        state.add_mapped_tasklet(
            "_BatchedMatMult_",
            map_params,
            {"__a": dace.Memlet.simple("_a", memlet_a), "__b": dace.Memlet.simple("_b", memlet_b)},
            f"__c = {product}",
            {"__c": dace.Memlet.simple("_c", c_indices, wcr_str="lambda x, y: x + y")},
            external_edges=True,
        )

        return sdfg

    @staticmethod
    def make_gathered_sdfg(node, parent_state, parent_sdfg):
        """``_c[p] = alpha * _a[_a_batch_index[p]] @ _b[_b_batch_index[p]]`` as one map, reading the pool through
        the index; an operand without an index is batched (3D) or shared (2D)."""
        if not (equal_valued(0, node.beta) or equal_valued(1, node.beta)):
            raise NotImplementedError("A gathered batched product takes beta 0 or 1")
        sdfg, subscripts = gathered_sdfg(node, parent_state, parent_sdfg, node.label + "_sdfg")
        shape_a, shape_c = sdfg.arrays["_a"].shape, sdfg.arrays["_c"].shape
        extents = {"__p": shape_c[0], "__m": shape_c[1], "__n": shape_c[2], "__k": shape_a[-2 if node.transA else -1]}
        state = sdfg.add_state(node.label + "_state")
        if equal_valued(0, node.beta):
            init = sdfg.add_state_before(state, node.label + "_initstate")
            init.add_mapped_tasklet(
                "batched_matmul_init",
                {
                    "__p": f"0:{symstr(extents['__p'])}",
                    "__m": f"0:{symstr(extents['__m'])}",
                    "__n": f"0:{symstr(extents['__n'])}",
                },
                {},
                "out = 0",
                {"out": dace.Memlet.simple("_c", "__p, __m, __n")},
                external_edges=True,
            )
        inputs, factors = {}, []
        for name, conn in (("_a", "__a"), ("_b", "__b")):
            index = BATCH_INDEX[name]
            if index in sdfg.arrays:
                pool = symstr(sdfg.arrays[name].shape[0])
                inputs[f"{conn}_at"] = dace.Memlet.simple(index, "__p")
                inputs[conn] = dace.Memlet.simple(name, ", ".join([f"0:{pool}", *subscripts[name][1:]]))
                factors.append(f"{conn}[{conn}_at]")
            else:
                inputs[conn] = dace.Memlet.simple(name, ", ".join(subscripts[name]))
                factors.append(conn)
        product = " * ".join(factors)
        if not equal_valued(1, node.alpha):
            product = f"{format_scalar_literal(node.alpha, sdfg.arrays['_c'].dtype)} * {product}"
        state.add_mapped_tasklet(
            "_BatchedMatMult_",
            {var: f"0:{symstr(extent)}" for var, extent in extents.items()},
            inputs,
            f"__c = {product}",
            {"__c": dace.Memlet.simple("_c", "__p, __m, __n", wcr_str="lambda x, y: x + y")},
            external_edges=True,
        )
        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if gathered_operands(node):
            return ExpandBatchedMatMulPure.make_gathered_sdfg(node, state, sdfg)
        return ExpandBatchedMatMulPure.make_sdfg(node, state, sdfg)


@dace.library.expansion
class ExpandBatchedMatMulCUDABlock(ExpandTransformation):
    """In-kernel batched GEMM by ONE thread block: each lane computes elements of ``_c``, strided over the block.
    A batch dimension of extent 1 broadcasts, as in the ``pure`` expansion."""

    runs_inside_kernel = True
    environments = [CUDA]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        node.validate(parent_sdfg, parent_state)
        if gathered_operands(node):
            return ExpandBatchedMatMulPure.expansion(node, parent_state, parent_sdfg)
        operands = _get_matmul_operands(node, parent_state, parent_sdfg)
        sdfg = dace.SDFG(node.label + "_block")
        for name, operand in zip(("_a", "_b", "_c"), operands):
            desc, shape, strides = operand[1:4]
            sdfg.add_array(name, shape, desc.dtype, strides=strides, storage=desc.storage)
        if block.vector_types(sdfg):
            return ExpandBatchedMatMulPure.expansion(node, parent_state, parent_sdfg)
        shape_a, shape_b, shape_c = (sdfg.arrays[name].shape for name in ("_a", "_b", "_c"))
        batch = [(f"__b{i}", extent) for i, extent in enumerate(shape_c[:-2])]

        def indices(shape, matrix: list[str]) -> list[str]:
            """Batch subscripts aligned to the RIGHT against the output's; an extent of 1 is read at 0."""
            offset = len(batch) - (len(shape) - 2)
            return ["0" if equal(d, 1) is True else batch[offset + j][0] for j, d in enumerate(shape[:-2])] + matrix

        a = block.at(sdfg, "_a", *indices(shape_a, ["__k", "__m"] if node.transA else ["__m", "__k"]))
        b = block.at(sdfg, "_b", *indices(shape_b, ["__n", "__k"] if node.transB else ["__k", "__n"]))
        c = block.at(sdfg, "_c", *(var for var, extent in batch), "__m", "__n")
        result = "__alpha * __acc" if equal_valued(0, node.beta) else f"__alpha * __acc + __beta * {c}"
        values = {"alpha": node.alpha} if equal_valued(0, node.beta) else {"alpha": node.alpha, "beta": node.beta}
        dtype = sdfg.arrays["_c"].dtype
        code = block.coefficients(dtype, **values) + block.block_contraction_code(
            batch + [("__m", shape_c[-2]), ("__n", shape_c[-1])],
            [("__k", shape_a[-2] if node.transA else shape_a[-1])],
            f"{a} * {b}",
            f"{c} = {result};",
            dtype.ctype,
        )
        return block.block_sdfg(node, parent_state, sdfg, code, tuple(values.values()))


@dace.library.expansion
class ExpandBatchedMatMulMKL(ExpandTransformation):
    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if gathered_operands(node):
            return ExpandBatchedMatMulPure.expansion(node, state, sdfg)
        refuse_broadcast_batches(node, state, sdfg)
        (_, adesc, ashape, astrides, _, _), (_, bdesc, bshape, bstrides, _, _), _ = _get_matmul_operands(
            node, state, sdfg
        )
        cdesc: dt.Array = sdfg.arrays[state.out_edges(node)[0].data.data]
        check_access(dtypes.ScheduleType.CPU_Multicore, adesc, bdesc, cdesc)
        dtype = cdesc.dtype.base_type
        func = to_blastype(dtype.type).lower() + "gemm"
        # ``?gemm_batch`` is the Fortran-style call: the coefficient and operand arrays hold MKL's own
        # element type, so a complex one is MKL_Complex8/16 initialized by value, not dace::complex.
        # Both structs are aggregates with no converting constructor, so a complex coefficient stays a
        # brace literal. beta scales C in place -- there is no separate C read here, exactly like the
        # pure expansion: ?gemm_batch reads and writes the same pointer, so beta != 0 is honored for
        # free by passing node.beta through instead of hard-coding 0.
        ctype = cdesc.dtype.ctype

        def mkl_coeff(value) -> str:
            if not isinstance(value, numbers.Number):
                if dtype in (dace.float32, dace.float64):
                    return f"({dtype.ctype})({value})"
                return f"{{({dtype.base_type.ctype})({value}), 0}}"
            if dtype == dace.float32:
                return f"{float(value)}f"
            if dtype == dace.float64:
                return f"{float(value)}"
            cvalue = complex(value)
            suffix = "f" if dtype == dace.complex64 else ""
            return f"{{{cvalue.real}{suffix}, {cvalue.imag}{suffix}}}"

        if dtype == dace.float32:
            prefix = "s"
        elif dtype == dace.float64:
            prefix = "d"
        elif dtype == dace.complex64:
            prefix = "c"
            ctype = "MKL_Complex8"
        elif dtype == dace.complex128:
            prefix = "z"
            ctype = "MKL_Complex16"
        else:
            raise ValueError("Unsupported type for BLAS dot product: " + str(dtype))
        alpha = mkl_coeff(node.alpha)
        beta = mkl_coeff(node.beta)
        opt = _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc, alpha, beta, ctype, func)

        opt["prefix"] = prefix

        code = """
        const MKL_INT group_count = 1;
        MKL_INT group_sizes[group_count] = {{ (MKL_INT)({BATCH}) }};
        MKL_INT m_array[group_count] = {{ (MKL_INT)({M}) }};
        MKL_INT n_array[group_count] = {{ (MKL_INT)({N}) }};
        MKL_INT k_array[group_count] = {{ (MKL_INT)({K}) }};
        char transa[group_count] = {{ '{ta}' }};
        char transb[group_count] = {{ '{tb}' }};
        {dtype} alpha_array[group_count] = {{ {alpha} }};
        {dtype} beta_array[group_count] = {{ {beta} }};
        MKL_INT lda_array[group_count] = {{ (MKL_INT)({lda}) }};
        MKL_INT ldb_array[group_count] = {{ (MKL_INT)({ldb}) }};
        MKL_INT ldc_array[group_count] = {{ (MKL_INT)({ldc}) }};

        const {dtype}** __mkl_BMM_A = new const {dtype}*[{BATCH}];
        const {dtype}** __mkl_BMM_B = new const {dtype}*[{BATCH}];
        {dtype}** __mkl_BMM_C = new {dtype}*[{BATCH}];
        for (int __ib = 0; __ib < {BATCH}; __ib++) {{
            __mkl_BMM_A[__ib] = (({dtype}*){x}) + __ib*{stride_a};
            __mkl_BMM_B[__ib] = (({dtype}*){y}) + __ib*{stride_b};
            __mkl_BMM_C[__ib] = (({dtype}*)_c) + __ib*{stride_c};
        }}

        {prefix}gemm_batch(transa, transb, m_array, n_array, k_array, alpha_array, __mkl_BMM_A, lda_array, __mkl_BMM_B, ldb_array, beta_array, __mkl_BMM_C, ldc_array, &group_count, group_sizes);

        delete[] __mkl_BMM_A;
        delete[] __mkl_BMM_B;
        delete[] __mkl_BMM_C;
        """.format_map(opt)

        tasklet = dace.sdfg.nodes.Tasklet(
            node.name, node.in_connectors, node.out_connectors, code, language=dace.dtypes.Language.CPP
        )
        return tasklet


@dace.library.expansion
class ExpandBatchedMatMulOpenBLAS(ExpandTransformation):
    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if gathered_operands(node):
            return ExpandBatchedMatMulPure.expansion(node, state, sdfg)
        refuse_broadcast_batches(node, state, sdfg)
        (_, adesc, ashape, astrides, _, _), (_, bdesc, bshape, bstrides, _, _), _ = _get_matmul_operands(
            node, state, sdfg
        )
        cdesc = sdfg.arrays[state.out_edges(node)[0].data.data]
        check_access(dtypes.ScheduleType.CPU_Multicore, adesc, bdesc, cdesc)
        dtype = cdesc.dtype.base_type
        func = to_blastype(dtype.type).lower() + "gemm"
        # cblas_[cz]gemm takes alpha/beta as ``const void*`` (see cblas.h), so a complex coefficient
        # is materialized into a local and passed by address; the real ?gemm calls take them by
        # value. beta scales C in place through the same pointer cblas already writes through, so
        # passing node.beta needs no separate C read here (mirrors the pure expansion).
        decls = ""
        if dtype in (dace.float32, dace.float64):
            alpha = f"{dtype.ctype}({node.alpha})"
            beta = f"{dtype.ctype}({node.beta})"
        elif dtype in (dace.complex64, dace.complex128):
            calpha, cbeta = complex(node.alpha), complex(node.beta)
            decls = (
                f"{dtype.ctype} __bmm_alpha = {dtype.ctype}({calpha.real}, {calpha.imag});\n"
                f"        {dtype.ctype} __bmm_beta = {dtype.ctype}({cbeta.real}, {cbeta.imag});\n        "
            )
            alpha = "(const void*)&__bmm_alpha"
            beta = "(const void*)&__bmm_beta"
        else:
            raise ValueError("Unsupported type for BLAS dot product: " + str(dtype))
        opt = _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc, alpha, beta, cdesc.dtype.ctype, func)
        opt["decls"] = decls

        # Adaptations for MKL/BLAS API
        opt["ta"] = "CblasNoTrans" if opt["ta"] == "N" else "CblasTrans"
        opt["tb"] = "CblasNoTrans" if opt["tb"] == "N" else "CblasTrans"
        opt["pragma"] = "#pragma omp parallel for\n        " if shares_out_batch(node, state, sdfg) else ""

        code = """
        {decls}{pragma}for (int __ib = 0; __ib < {BATCH}; ++__ib) {{
            cblas_{func}(CblasColMajor, {ta}, {tb}, {M}, {N}, {K}, {alpha},
                         (({dtype}*){x}) + __ib*{stride_a}, {lda},
                         (({dtype}*){y}) + __ib*{stride_b}, {ldb},
                         {beta},
                         (({dtype}*)_c) + __ib*{stride_c}, {ldc});
        }}""".format_map(opt)

        tasklet = dace.sdfg.nodes.Tasklet(
            node.name, node.in_connectors, node.out_connectors, code, language=dace.dtypes.Language.CPP
        )
        return tasklet


@dace.library.expansion
class ExpandBatchedMatMulLIBXSMM(ExpandTransformation):
    """The OpenBLAS batch loop over ``libxsmm_?gemm``: a JIT kernel per small product, no per-call buffer allocation
    (cp2k_trs4's 1.2M 5x5 block products). A complex batch takes the OpenBLAS calls."""

    environments = [environments.libxsmm.LIBXSMM, environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if gathered_operands(node):
            return ExpandBatchedMatMulPure.expansion(node, state, sdfg)
        refuse_broadcast_batches(node, state, sdfg)
        (_, adesc, _, _, _, _), (_, bdesc, _, _, _, _), _ = _get_matmul_operands(node, state, sdfg)
        cdesc = sdfg.arrays[state.out_edges(node)[0].data.data]
        dtype = cdesc.dtype.base_type
        if dtype not in (dace.float32, dace.float64):
            return ExpandBatchedMatMulOpenBLAS.expansion(node, state, sdfg)
        check_access(dtypes.ScheduleType.CPU_Multicore, adesc, bdesc, cdesc)
        func = to_blastype(dtype.type).lower() + "gemm"
        alpha, beta = f"{dtype.ctype}({node.alpha})", f"{dtype.ctype}({node.beta})"
        opt = _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc, alpha, beta, cdesc.dtype.ctype, func)
        opt["pragma"] = "#pragma omp parallel for\n        " if shares_out_batch(node, state, sdfg) else ""
        code = """
        const dace::blas::SmallGemm<{dtype}> __gemm('{ta}', '{tb}', {M}, {N}, {K}, {lda}, {ldb}, {ldc}, {alpha}, {beta});
        {pragma}for (int64_t __ib = 0; __ib < {BATCH}; ++__ib) {{
            __gemm((({dtype}*){x}) + __ib*{stride_a}, (({dtype}*){y}) + __ib*{stride_b},
                   (({dtype}*)_c) + __ib*{stride_c});
        }}""".format_map(opt)
        return dace.sdfg.nodes.Tasklet(
            node.name, node.in_connectors, node.out_connectors, code, language=dace.dtypes.Language.CPP
        )


def cuda_ctype(dtype: dtypes.typeclass) -> str:
    """The CUDA C type a GPU BLAS call takes ``dtype`` as."""
    ctypes = {
        dace.float16: "__half",
        dace.float32: "float",
        dace.float64: "double",
        dace.complex64: "cuComplex",
        dace.complex128: "cuDoubleComplex",
    }
    if dtype not in ctypes:
        raise ValueError("Unsupported type: " + str(dtype))
    return ctypes[dtype]


@dace.library.expansion
class ExpandBatchedMatMulGPUBLAS(ExpandTransformation):
    environments = []

    @classmethod
    def gathered_expansion(cls, node, state, sdfg):
        """One ``<T>gemmBatched`` over device pointer arrays that a GPU map fills from the batch indices."""
        from dace.codegen.common import sym2cpp  # Avoid import loops

        nsdfg, subscripts = gathered_sdfg(node, state, sdfg, node.label + "_gathered")
        operands = ("_a", "_b", "_c")
        on_device = all(nsdfg.arrays[name].storage == dtypes.StorageType.GPU_Global for name in nsdfg.arrays)
        if not on_device or not (equal_valued(0, node.beta) or equal_valued(1, node.beta)):
            return ExpandBatchedMatMulPure.make_gathered_sdfg(node, state, sdfg)
        dtype = nsdfg.arrays["_c"].dtype.base_type
        batch = symstr(nsdfg.arrays["_c"].shape[0])

        def matrix(name: str, transposed: bool) -> list:
            desc = nsdfg.arrays[name]
            shape, strides = list(desc.shape[-2:]), list(desc.strides[-2:])
            return packed_unit_extent(shape[::-1], strides[::-1]) if transposed else packed_unit_extent(shape, strides)

        opt = get_gemm_opts(matrix("_a", node.transA), matrix("_b", node.transB), matrix("_c", False))
        rows, depth = nsdfg.arrays["_a"].shape[-1:-3:-1] if node.transA else nsdfg.arrays["_a"].shape[-2:]
        cols = nsdfg.arrays["_b"].shape[-2] if node.transB else nsdfg.arrays["_b"].shape[-1]
        x, y, lda, ldb, ta, tb = "_a", "_b", opt["lda"], opt["ldb"], opt["ta"], opt["tb"]
        if opt["swap"]:
            x, y, lda, ldb, ta, tb, rows, cols = y, x, ldb, lda, tb, ta, cols, rows

        fill = nsdfg.add_state(node.label + "_pointers")
        entry, exit_node = fill.add_map(node.label + "_pointers", {"__p": f"0:{batch}"}, dtypes.ScheduleType.GPU_Device)
        tasklet = fill.add_tasklet(node.label + "_pointers", {}, {}, "", language=dtypes.Language.CPP)
        code, writes = [], []
        # Inputs first: a tasklet the entry does not yet reach is outside the map scope
        for name in operands:
            desc = nsdfg.arrays[name]
            base = f"__base{name}"
            at = {"__g": f"__at{name}", "__p": "__p"}.get(subscripts[name][0], "0")
            code.append(f"*__out{name} = ({dtype.ctype} *){base} + ({at}) * ({symstr(desc.strides[0])});")
            whole = dace.Memlet.from_array(name, desc)
            if name == "_c":
                # The output's address only; the call writes it
                tasklet.add_out_connector(base, dtypes.pointer(dtype))
                whole.dynamic = True
                writes.append((base, name, whole))
            else:
                tasklet.add_in_connector(base, dtypes.pointer(dtype))
                fill.add_memlet_path(fill.add_read(name), entry, tasklet, dst_conn=base, memlet=whole)
            if at == f"__at{name}":
                tasklet.add_in_connector(at)
                index = BATCH_INDEX[name]
                fill.add_memlet_path(
                    fill.add_read(index), entry, tasklet, dst_conn=at, memlet=dace.Memlet(f"{index}[__p]")
                )
            pointers, _ = nsdfg.add_array(
                f"__ptr{name}", [batch], dtypes.pointer(dtype), dtypes.StorageType.GPU_Global, transient=True
            )
            tasklet.add_out_connector(f"__out{name}", dtypes.pointer(dtypes.pointer(dtype)))
            writes.append((f"__out{name}", pointers, dace.Memlet(f"{pointers}[__p]")))
        for conn, data, memlet in writes:
            fill.add_memlet_path(tasklet, exit_node, fill.add_write(data), src_conn=conn, memlet=memlet)
        tasklet.code = properties.CodeBlock("\n".join(code), dtypes.Language.CPP)

        cdtype = cls.dialect.ctype(cuda_ctype(dtype))
        alpha = (
            f"{dtype.ctype}({node.alpha.real}, {node.alpha.imag})"
            if isinstance(node.alpha, complex)
            else f"{dtype.ctype}({node.alpha})"
        )
        beta = (
            f"{dtype.ctype}({node.beta.real}, {node.beta.imag})"
            if isinstance(node.beta, complex)
            else f"{dtype.ctype}({node.beta})"
        )
        routine = cls.dialect.routine(f"{to_blastype(dtype.type)}gemmBatched")
        body = f"""{dtype.ctype} __alpha = {alpha}, __beta = {beta};
{cls.dialect.check_error}({routine}({cls.dialect.handle}, {cls.dialect.op(ta)}, {cls.dialect.op(tb)},
    {sym2cpp(rows)}, {sym2cpp(cols)}, {sym2cpp(depth)}, ({cdtype} *)&__alpha,
    (const {cdtype} * const *)__in_ptr{x}, {sym2cpp(lda)}, (const {cdtype} * const *)__in_ptr{y}, {sym2cpp(ldb)},
    ({cdtype} *)&__beta, ({cdtype} * const *)__in_ptr_c, {sym2cpp(opt["ldc"])}, {batch}));"""
        call_state = nsdfg.add_state_after(fill, node.label + "_call")
        pointer = dtypes.pointer(dtypes.pointer(dtype))
        call = call_state.add_tasklet(
            node.label + "_call",
            {f"__in_ptr{name}": pointer for name in operands},
            {"__c": dtypes.pointer(dtype)},
            cls.environments[0].handle_setup_code(node) + gpu_dialect.host_scalar_mode(cls.dialect, body),
            language=dtypes.Language.CPP,
        )
        for name in operands:
            pointers = f"__ptr{name}"
            call_state.add_edge(
                call_state.add_read(pointers),
                None,
                call,
                f"__in_ptr{name}",
                dace.Memlet.from_array(pointers, nsdfg.arrays[pointers]),
            )
        call_state.add_edge(
            call, "__c", call_state.add_write("_c"), None, dace.Memlet.from_array("_c", nsdfg.arrays["_c"])
        )
        return nsdfg

    @classmethod
    def expansion(cls, node, state, sdfg):
        node.validate(sdfg, state)
        if gathered_operands(node):
            return cls.gathered_expansion(node, state, sdfg)
        refuse_broadcast_batches(node, state, sdfg)

        # Find inputs and output
        adesc, bdesc, cdesc = None, None, None
        for e in state.in_edges(node):
            if e.dst_conn == "_a":
                anode = state.memlet_path(e)[0].src
                if isinstance(anode, dace.sdfg.nodes.AccessNode):
                    adesc: dt.Array = sdfg.arrays[anode.data]
            elif e.dst_conn == "_b":
                bnode = state.memlet_path(e)[0].src
                if isinstance(bnode, dace.sdfg.nodes.AccessNode):
                    bdesc: dt.Array = sdfg.arrays[bnode.data]
        for e in state.out_edges(node):
            if e.src_conn == "_c":
                cnode = state.memlet_path(e)[-1].dst
                if isinstance(cnode, dace.sdfg.nodes.AccessNode):
                    cdesc: dt.Array = sdfg.arrays[cnode.data]
        if not adesc or not bdesc or not cdesc:
            raise ValueError("Unsupported input/output arrays")

        needs_copy = any(
            desc.storage not in (dace.StorageType.GPU_Global, dace.StorageType.CPU_Pinned)
            for desc in (adesc, bdesc, cdesc)
        )

        dtype = cdesc.dtype.base_type
        func = f"{to_blastype(dtype.type)}gemm"
        cdtype = cls.dialect.ctype(cuda_ctype(dtype))

        call_prefix = cls.environments[0].handle_setup_code(node)
        call_suffix = ""
        # Alpha and beta go in as host values, under host pointer mode for this call: under device mode rocBLAS
        # copies both back to the host (a synchronizing copy each) to test for a quick return. cuBLAS reads and
        # writes C in place through the one pointer this call already takes, so a nonzero beta needs no separate C
        # connector here.
        if isinstance(node.alpha, complex):
            alpha_lit = f"{dtype.ctype}({node.alpha.real}, {node.alpha.imag})"
        else:
            alpha_lit = f"{dtype.ctype}({node.alpha})"
        if isinstance(node.beta, complex):
            beta_lit = f"{dtype.ctype}({node.beta.real}, {node.beta.imag})"
        else:
            beta_lit = f"{dtype.ctype}({node.beta})"

        # Set pointer mode to host
        call_prefix += f"""{cls.dialect.check_error}(
            {cls.dialect.set_pointer_mode}({cls.dialect.handle}, {cls.dialect.pointer_host}));
            {dtype.ctype} alpha = {alpha_lit};
            {dtype.ctype} beta = {beta_lit};
            """
        call_suffix += f"""
    {cls.dialect.check_error}({cls.dialect.set_pointer_mode}({cls.dialect.handle}, {cls.dialect.pointer_device}));
            """
        beta = f"({cdtype} *)&beta"
        alpha = f"({cdtype} *)&alpha"

        # Set up options for code formatting
        opt = _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc, alpha, beta, cdtype, func)
        opt["array_prefix"] = "_" if needs_copy else ""

        # Matrix multiplication
        if node.compute_type is None and node.accumulator_type is None and node.algorithm is None:
            call = f"""{cls.dialect.check_error}({cls.dialect.strided_batched(func)}({cls.dialect.handle},
                {cls.dialect.op(opt["ta"])}, {cls.dialect.op(opt["tb"])},
                {opt["M"]}, {opt["N"]}, {opt["K"]},
                {opt["alpha"]},
                ({opt["dtype"]}*){opt["array_prefix"]}{opt["x"]}, {opt["lda"]}, {opt["stride_a"]},
                ({opt["dtype"]}*){opt["array_prefix"]}{opt["y"]}, {opt["ldb"]}, {opt["stride_b"]},
                {opt["beta"]},
                ({opt["dtype"]}*){opt["array_prefix"]}_c, {opt["ldc"]}, {opt["stride_c"]},
                {opt["BATCH"]}));"""
        else:
            if node.compute_type is not None:
                acctype = node.compute_type
            elif node.accumulator_type is not None:
                acc_dtype: dtypes.typeclass = node.accumulator_type
                acctype = f"CUBLAS_COMPUTE_{to_cublas_computetype(acc_dtype)}"
            else:
                acctype = f"CUBLAS_COMPUTE_{to_cublas_computetype(dtype)}"

            algorithm = "CUBLAS_GEMM_DEFAULT_TENSOR_OP"
            if node.algorithm is not None:
                algorithm = node.algorithm

            call = f"""
            {cls.dialect.check_error}(cublasGemmStridedBatchedEx({cls.dialect.handle},
                {cls.dialect.op(opt["ta"])}, {cls.dialect.op(opt["tb"])},
                {opt["M"]}, {opt["N"]}, {opt["K"]},
                {alpha},
                {opt["array_prefix"]}{opt["x"]},
                {dtype_to_cudadatatype(opt["xdtype"])},
                {opt["lda"]}, {opt["stride_a"]},
                {opt["array_prefix"]}{opt["y"]},
                {dtype_to_cudadatatype(opt["ydtype"])},
                {opt["ldb"]}, {opt["stride_b"]},
                {beta},
                {opt["array_prefix"]}_c,
                {dtype_to_cudadatatype(opt["cdtype"])},
                {opt["ldc"]}, {opt["stride_c"]},
                {opt["BATCH"]},
                {acctype}, {algorithm}));
            """

        code = call_prefix + call + call_suffix
        tasklet = dace.sdfg.nodes.Tasklet(
            node.name, node.in_connectors, node.out_connectors, code, language=dace.dtypes.Language.CPP
        )

        # If buffers are not on the GPU, copy them
        if needs_copy:
            nsdfg = dace.SDFG("nested_batched_matmul")
            tasklet = dace.sdfg.nodes.Tasklet(
                node.name,
                {"__a": dtypes.pointer(adesc.dtype), "__b": dtypes.pointer(bdesc.dtype)},
                {"__c": dtypes.pointer(cdesc.dtype)},
                code,
                language=dace.dtypes.Language.CPP,
            )

            for name, desc in [("_a", adesc), ("_b", bdesc), ("_c", cdesc)]:
                if isinstance(desc, dt.View):
                    dcopy = desc.as_array()
                else:
                    dcopy = dc(desc)
                dcopy.transient = False
                dcopy.lifetime = dtypes.AllocationLifetime.Scope
                dcopy_gpu = dc(dcopy)
                nsdfg.add_datadesc(name, dcopy)
                dcopy_gpu.transient = True
                dcopy_gpu.storage = dace.StorageType.GPU_Global
                nsdfg.add_datadesc(name + "_gpu", dcopy_gpu)

            # A nonzero beta needs the host C staged into `_c_gpu` BEFORE the call -- unlike A/B, C is
            # not otherwise copied in, so a fresh GPU allocation would read uninitialized memory as
            # the prior value. A separate state (not a second edge into the same access node) avoids
            # a write/write conflict with the tasklet's own write below; `_c_gpu` has Scope lifetime,
            # so the value persists into the next state exactly like the pure expansion's init state.
            if not equal_valued(0, node.beta):
                copyin_state = nsdfg.add_state(node.label + "_c_copyin")
                copyin_state.add_nedge(
                    copyin_state.add_read("_c"), copyin_state.add_access("_c_gpu"), dace.Memlet.from_array("_c", cdesc)
                )
                nstate = nsdfg.add_state_after(copyin_state)
            else:
                nstate = nsdfg.add_state()
            a = nstate.add_read("_a")
            ga = nstate.add_access("_a_gpu")
            b = nstate.add_read("_b")
            gb = nstate.add_access("_b_gpu")
            c = nstate.add_write("_c")
            gc = nstate.add_access("_c_gpu")
            nstate.add_node(tasklet)
            nstate.add_nedge(a, ga, dace.Memlet.from_array("_a", adesc))
            nstate.add_nedge(b, gb, dace.Memlet.from_array("_b", bdesc))
            nstate.add_edge(ga, None, tasklet, "__a", dace.Memlet.from_array("_a_gpu", adesc))
            nstate.add_edge(gb, None, tasklet, "__b", dace.Memlet.from_array("_b_gpu", bdesc))
            nstate.add_edge(tasklet, "__c", gc, None, dace.Memlet.from_array("_c_gpu", cdesc))
            nstate.add_nedge(gc, c, dace.Memlet.from_array("_c", cdesc))

            return nsdfg
        # End of copy to GPU

        return tasklet


@dace.library.expansion
class ExpandBatchedMatMulCuBLAS(ExpandBatchedMatMulGPUBLAS):
    environments = [environments.cublas.cuBLAS]
    dialect = gpu_dialect.CUBLAS


ExpandBatchedMatMulHipBLAS = hip_dialect.hip_expansion(
    ExpandBatchedMatMulCuBLAS, environments.hipblas.hipBLAS, hip_dialect.BLAS
)


@dace.library.node
class BatchedMatMul(dace.sdfg.nodes.LibraryNode):
    # Global properties
    implementations = {
        "pure": ExpandBatchedMatMulPure,
        "MKL": ExpandBatchedMatMulMKL,
        "OpenBLAS": ExpandBatchedMatMulOpenBLAS,
        "LIBXSMM": ExpandBatchedMatMulLIBXSMM,
        "cuBLAS": ExpandBatchedMatMulCuBLAS,
        "hipBLAS": ExpandBatchedMatMulHipBLAS,
        "rocBLAS": ExpandBatchedMatMulHipBLAS,
        "CUDA (block strided)": ExpandBatchedMatMulCUDABlock,
    }
    transA = properties.Property(dtype=bool, category="Semantics", desc="Whether to transpose A before multiplying")
    transB = properties.Property(dtype=bool, category="Semantics", desc="Whether to transpose B before multiplying")
    alpha = properties.Property(
        allow_none=False,
        default=1,
        category="Semantics",
        desc="A scalar which will be multiplied with A @ B before adding C",
    )

    def folds_onto_output(self) -> bool:
        return self.folds_by_beta(self.beta)

    beta = properties.Property(
        allow_none=False,
        default=0,
        category="Semantics",
        desc="A scalar which will be multiplied with C before adding C",
    )
    algorithm = properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        category="Code Generation",
        desc="If applicable, chooses the vendor-provided implementation (algorithm) for the multiplication",
    )
    accumulator_type = properties.TypeClassProperty(
        default=None,
        allow_none=True,
        category="Semantics",
        desc="Accumulator or intermediate storage type used in multiplication",
    )
    compute_type = properties.Property(
        default=None,
        dtype=str,
        allow_none=True,
        category="Code Generation",
        desc="If applicable, overrides computation type (CUBLAS-specific, see ``cublasComputeType_t``)",
    )

    default_implementation = None

    def __init__(self, name, location=None):
        super().__init__(name, location=location, inputs=OrderedSet(("_a", "_b")), outputs={"_c"})

    def validate(self, sdfg, state):
        gathered = gathered_operands(self)
        in_edges = [edge for edge in state.in_edges(self) if edge.dst_conn not in gathered.values()]
        if len(in_edges) != 2:
            raise ValueError("Expected exactly two inputs to batched matrix-matrix product")
        # NumPy matmul semantics on the real extents: squeezing would turn B[b, k, 1] into a (b, k) matrix
        sizes = {edge.dst_conn: edge.data.subset.size() for edge in in_edges}
        size0, size1 = sizes["_a"], sizes["_b"]
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from batched matrix-matrix product")
        out_memlet = out_edges[0].data

        # Both inputs must be at least 2D
        if len(size0) < 2:
            raise ValueError(f"First input must be at least 2D, got shape with {len(size0)} dimensions")
        if len(size1) < 2:
            raise ValueError(f"Second input must be at least 2D, got shape with {len(size1)} dimensions")

        # At least one input must have batch dimensions (3D or higher) for batched operation
        if len(size0) <= 2 and len(size1) <= 2:
            raise ValueError(
                "Batched matrix-matrix product requires at least one input to have batch dimensions (3D or higher)"
            )

        # Batch dimensions follow NumPy's broadcasting rules: they align to the RIGHT, and a
        # dimension of extent 1 stretches. Only a genuine mismatch is rejected here -- whether a
        # given expansion can express a broadcast is the expansion's business, not the node's
        # (the pure lowering indexes a broadcast dimension at 0; the BLAS ones walk a fixed batch
        # stride and refuse, see ``refuse_broadcast_batches``).
        batch0, batch1 = size0[:-2], size1[:-2]
        if gathered:
            # The index, not the pool, sets the batch: a gathered operand is a pool of matrices
            if len(out_memlet.subset) != 3 or any(len(sizes[operand]) != 3 for operand in gathered):
                raise ValueError("A gathered batched product takes 3D operands and output")
            batch0 = batch1 = ()
        if batch0 and batch1:
            offset = len(batch0) - len(batch1)
            paired = zip(batch0[offset:], batch1) if offset >= 0 else zip(batch0, batch1[-offset:])
            for d0, d1 in paired:
                if equal(d0, d1) is False and equal(d0, 1) is not True and equal(d1, 1) is not True:
                    raise ValueError(
                        f"Batch dimensions of the two inputs do not broadcast, got {batch0} and "
                        f"{batch1}: {d0} and {d1} differ and neither is 1"
                    )

        # Validate K-dimension compatibility
        modes = size0[-2 if self.transA else -1], size1[-1 if self.transB else -2]
        res = equal(*modes)
        if res is None:
            warnings.warn(
                f"First tensor's mode {modes[0]} and second tensor's mode {modes[1]} may not match", UserWarning
            )
        elif not res:
            raise ValueError("Inputs to matrix-matrix product must agree in the k-dimension")

        # Output must have batch dimensions
        if len(out_memlet.subset) < 3:
            raise ValueError(
                f"Batched matrix-matrix product output must be at least 3D, got {len(out_memlet.subset)} dimensions"
            )


# Numpy replacement
@oprepo.replaces("dace.libraries.blas.bmm")
def bmmnode(pv, sdfg: dace.SDFG, state: dace.SDFGState, A, B, C, alpha=1, beta=0, trans_a=False, trans_b=False):
    # Add nodes
    A_in, B_in = (state.add_read(name) for name in (A, B))
    C_out = state.add_write(C)

    libnode = BatchedMatMul("bmm")
    libnode.alpha = alpha
    libnode.beta = beta
    libnode.transA = trans_a
    libnode.transB = trans_b
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(A_in, None, libnode, "_a", mm.Memlet(A))
    state.add_edge(B_in, None, libnode, "_b", mm.Memlet(B))
    state.add_edge(libnode, "_c", C_out, None, mm.Memlet(C))

    return []
