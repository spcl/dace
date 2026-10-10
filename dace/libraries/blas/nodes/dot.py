# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import warnings

import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import SDFG, SDFGState, dtypes, symbolic
from dace import memlet as mm
from dace.frontend.common import op_repository as oprepo
from dace.libraries.blas import blas_helpers, environments
from dace.libraries.standard.environments.cuda import CUDA
from dace.ordered import OrderedSet
from dace.transformation.transformation import ExpandTransformation


def pure_dot_sdfg(node, parent_state, parent_sdfg, n=None):
    """The operands of a pure DOT expansion: its SDFG with ``_x``, ``_y`` and ``_result``, the length, the
    multiply tasklet code and the result type."""
    (desc_x, stride_x), (desc_y, stride_y), desc_res, sz = node.validate(parent_sdfg, parent_state)

    n = n or node.n or sz

    dtype_x = desc_x.dtype.type
    dtype_y = desc_y.dtype.type
    dtype_result = desc_res.dtype.type
    sdfg = dace.SDFG(node.label + "_sdfg")

    if desc_x.dtype.veclen > 1 or desc_y.dtype.veclen > 1:
        raise NotImplementedError("Pure expansion not implemented for vector types.")

    sdfg.add_array("_x", [n], dtype_x, strides=[stride_x], storage=desc_x.storage)
    sdfg.add_array("_y", [n], dtype_y, strides=[stride_y], storage=desc_y.storage)
    sdfg.add_array("_result", [1], dtype_result, storage=desc_res.storage)

    # Fortran DOT_PRODUCT(a, b) for complex a is SUM(CONJG(a)*b) = BLAS ?dotc; the
    # default Dot models ?dotu (no conjugation). conj on a real type would promote to
    # complex, so only apply it for complex operands.
    if node.conjugate and desc_x.dtype.is_complex():
        mul_program = "__out = conj(__x) * __y"
    else:
        mul_program = "__out = __x * __y"
    return sdfg, n, mul_program, dtype_result


@dace.library.expansion
class ExpandDotPure(ExpandTransformation):
    """
    Naive backend-agnostic expansion of DOT.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        sdfg, n, mul_program, dtype_result = pure_dot_sdfg(node, parent_state, parent_sdfg, n)

        init_state = sdfg.add_state(node.label + "_initstate")
        state = sdfg.add_state_after(init_state, node.label + "_state")

        if dace.dtypes.can_access(dace.dtypes.ScheduleType.CPU_Multicore, sdfg.arrays["_result"].storage):
            # A bare tasklet: a one-iteration map would fork a thread team to write a single scalar
            init_tasklet = init_state.add_tasklet("_dot_init", {}, {"_out"}, "_out = 0")
            init_state.add_edge(init_tasklet, "_out", init_state.add_write("_result"), None, dace.Memlet("_result[0]"))
        else:
            # A result in device memory is initialized by a one-iteration map, which is scheduled on the device
            init_state.add_mapped_tasklet(
                "_dot_init",
                {"__i_unused": "0:1"},
                {},
                "_out = 0",
                {"_out": dace.Memlet("_result[0]")},
                external_edges=True,
            )

        # Multiplication map
        state.add_mapped_tasklet(
            "dot",
            {"__i": f"0:{n}"},
            {"__x": dace.Memlet("_x[__i]"), "__y": dace.Memlet("_y[__i]")},
            mul_program,
            {"__out": dace.Memlet("_result[0]", wcr="lambda x, y: x + y")},
            external_edges=True,
            output_nodes=None,
        )

        return sdfg


@dace.library.expansion
class ExpandDotPureAccumulate(ExpandTransformation):
    """Single-state DOT for a map body: a register accumulator, so the expansion inlines and its map fuses
    with an operand's producer (spmv's gathered ``x[cols]``); in a kernel each lane sums its share and a
    block reduce folds the lanes."""

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        sdfg, n, mul_program, dtype_result = pure_dot_sdfg(node, parent_state, parent_sdfg, n)

        sdfg.add_scalar("_acc", dtype_result, transient=True, storage=dace.StorageType.Register)
        state = sdfg.add_state(node.label + "_state")
        seeded = state.add_access("_acc")
        zero = state.add_tasklet("_dot_init", {}, {"_out"}, "_out = 0")
        state.add_edge(zero, "_out", seeded, None, dace.Memlet("_acc[0]"))
        entry, exit_node = state.add_map("dot", {"__i": f"0:{n}"})
        mul = state.add_tasklet("dot", {"__x", "__y"}, {"__out"}, mul_program)
        state.add_memlet_path(state.add_read("_x"), entry, mul, dst_conn="__x", memlet=dace.Memlet("_x[__i]"))
        state.add_memlet_path(state.add_read("_y"), entry, mul, dst_conn="__y", memlet=dace.Memlet("_y[__i]"))
        summed = state.add_access("_acc")
        state.add_memlet_path(
            mul, exit_node, summed, src_conn="__out", memlet=dace.Memlet("_acc[0]", wcr="lambda x, y: x + y")
        )
        state.add_nedge(seeded, entry, dace.Memlet())
        state.add_nedge(summed, state.add_write("_result"), dace.Memlet("_acc[0] -> [0]"))

        return sdfg


@dace.library.expansion
class ExpandDotCUDABlock(ExpandTransformation):
    """In-kernel DOT by ONE thread block: a block-strided multiply folded by ``gpucub::BlockReduce``.

    The same collective the in-kernel ``Reduce`` lowers to (:func:`block_reduce_code`), with the product
    as the element. Every lane reads the whole run; the lane map only supplies the threads.
    """

    runs_inside_kernel = True
    environments = [CUDA]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        from dace.codegen.common import global_code_id
        from dace.libraries.standard.block_reduce import (
            BLOCK_COLLECTIVE_THREADS,
            add_block_lane_map,
            block_redop,
            block_reduce_code,
        )

        (desc_x, stride_x), (desc_y, stride_y), desc_res, sz = node.validate(parent_sdfg, parent_state)
        if desc_x.dtype.veclen > 1 or desc_y.dtype.veclen > 1:
            return ExpandDotPure.expansion(node, parent_state, parent_sdfg, n=n, **kwargs)
        n = n or node.n or sz
        ctype = desc_res.dtype.base_type.ctype
        x_element = f"__x[__bri * ({symbolic.symstr(stride_x)})]"
        if node.conjugate and desc_x.dtype.is_complex():
            x_element = f"dace::math::conj({x_element})"
        code = block_reduce_code(
            idstr=global_code_id(parent_sdfg, parent_state, node),
            ctype=ctype,
            lanes=BLOCK_COLLECTIVE_THREADS,
            count_expr=symbolic.symstr(n),
            element_expr=f"{x_element} * __y[__bri * ({symbolic.symstr(stride_y)})]",
            redop=block_redop(dtypes.ReductionType.Sum, ctype),
            identity=f"static_cast<{ctype}>(0)",
            out_expr="__res[0]",
        )

        sdfg = dace.SDFG(node.label + "_block")
        sdfg.add_array("_x", [n], desc_x.dtype, strides=[stride_x], storage=desc_x.storage)
        sdfg.add_array("_y", [n], desc_y.dtype, strides=[stride_y], storage=desc_y.storage)
        sdfg.add_array("_result", [1], desc_res.dtype, storage=desc_res.storage)
        state = sdfg.add_state(node.label + "_block_state")
        tasklet = state.add_tasklet(
            node.label + "_block_dot",
            {"__x": dace.pointer(desc_x.dtype.base_type), "__y": dace.pointer(desc_y.dtype.base_type)},
            {"__res": dace.pointer(desc_res.dtype.base_type)},
            code,
            language=dace.Language.CPP,
        )
        entry, exit_node = add_block_lane_map(state, node.label + "_block_lanes")
        for conn, name in (("__x", "_x"), ("__y", "_y")):
            state.add_memlet_path(
                state.add_read(name),
                entry,
                tasklet,
                dst_conn=conn,
                memlet=dace.Memlet.from_array(name, sdfg.arrays[name]),
            )
        state.add_memlet_path(
            tasklet, exit_node, state.add_write("_result"), src_conn="__res", memlet=dace.Memlet("_result[0]")
        )
        return sdfg


@dace.library.expansion
class ExpandDotOpenBLAS(ExpandTransformation):
    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x), (desc_y, stride_y), desc_res, sz = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type
        veclen = desc_x.dtype.veclen

        # A conjugated (?dotc) complex dot is not modelled by this cblas_?dot emission; route it
        # to the pure conj expansion rather than silently emit an unconjugated ?dotu.
        if node.conjugate and desc_x.dtype.is_complex():
            return ExpandDotPure.expansion(node, parent_state, parent_sdfg, n, **kwargs)

        try:
            func, _, _ = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f"{ex}. Falling back to pure expansion")
            return ExpandDotPure.expansion(node, parent_state, parent_sdfg, n, **kwargs)

        func = func.lower() + "dot"

        # ``cblas_?dot`` accumulates in the operand type; a wider accumulator needs the pure expansion.
        if node.accumulator_type is not None:
            warnings.warn("CBLAS has no mixed-precision dot. Falling back to pure expansion")
            return ExpandDotPure.expansion(node, parent_state, parent_sdfg, n, **kwargs)

        n = n or node.n or sz
        if veclen != 1:
            n /= veclen
        code = f"_result = cblas_{func}({n}, _x, {stride_x}, _y, {stride_y});"
        # The return type is scalar in cblas_?dot signature
        tasklet = dace.sdfg.nodes.Tasklet(
            node.name, node.in_connectors, {"_result": dtype}, code, language=dace.dtypes.Language.CPP
        )
        return tasklet


@dace.library.expansion
class ExpandDotMKL(ExpandTransformation):
    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandDotOpenBLAS.expansion(*args, **kwargs)


class ExpandDotGPUBLAS(ExpandTransformation):
    """``?dot`` on a vendor GPU BLAS. The two backends differ only in the vocabulary below.

    Same split as :class:`~dace.libraries.blas.nodes.gemm.ExpandGemmGPUBLAS`: the body -- operand
    validation, the veclen division, the conjugate and unsupported-dtype fallbacks -- is identical
    for both, and only the handle, the error check and the routine spelling move.
    """

    environments = []

    @classmethod
    def expansion(cls, node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x), (desc_y, stride_y), desc_res, sz = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type
        veclen = desc_x.dtype.veclen

        # Conjugated (?dotc) complex dot is not emitted here; use the pure conj expansion
        # rather than a silently unconjugated cublas ?dotu.
        if node.conjugate and desc_x.dtype.is_complex():
            return ExpandDotPure.expansion(node, parent_state, parent_sdfg, n, **kwargs)

        try:
            func, _, _ = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f"{ex}. Falling back to pure expansion")
            return ExpandDotPure.expansion(node, parent_state, parent_sdfg, n, **kwargs)
        func = func + "dot"

        n = n or node.n or sz
        if veclen != 1:
            n /= veclen

        code = cls.environments[0].handle_setup_code(node)
        if node.accumulator_type is None:
            code += f"""{cls.check_error}({cls.funcname(func)}({cls.handle}, {n}, _x, {stride_x}, _y,
                             {stride_y}, _result));"""
        else:
            code += f"""
            {cls.check_error}({cls.ex_name}(
                {cls.handle},
                {n},
                _x,
                {blas_helpers.dtype_to_cudadatatype(dtype)},
                {stride_x},
                _y,
                {blas_helpers.dtype_to_cudadatatype(desc_y.dtype)},
                {stride_y},
                _result,
                {blas_helpers.dtype_to_cudadatatype(desc_res.dtype)},
                {blas_helpers.dtype_to_cudadatatype(node.accumulator_type)}));
            """

        tasklet = dace.sdfg.nodes.Tasklet(
            node.name, node.in_connectors, {"_result": dtypes.pointer(dtype)}, code, language=dace.dtypes.Language.CPP
        )

        return tasklet


@dace.library.expansion
class ExpandDotCuBLAS(ExpandDotGPUBLAS):
    environments = [environments.cublas.cuBLAS]
    handle = "__dace_cublas_handle"
    check_error = "dace::blas::CheckCublasError"
    ex_name = "cublasDotEx"

    @classmethod
    def funcname(cls, func: str) -> str:
        return f"cublas{func}"


@dace.library.expansion
class ExpandDotRocBLAS(ExpandDotGPUBLAS):
    environments = [environments.rocblas.rocBLAS]
    handle = "__dace_rocblas_handle"
    check_error = "dace::blas::CheckRocblasError"
    #: No mixed-precision path: ``rocblas_dot_ex`` takes ``rocblas_datatype_*`` enums, which the
    #: shared body has no mapping for. Empty makes the base fall back to ``pure`` for that case
    #: instead of emitting a rocBLAS call carrying CUDA enum names.
    ex_name = ""

    @classmethod
    def funcname(cls, func: str) -> str:
        # ``Ddot`` -> ``rocblas_ddot``: rocBLAS is snake_case with the type letter lowered.
        return f"rocblas_{func.lower()}"


@dace.library.node
class Dot(dace.sdfg.nodes.LibraryNode):
    # Global properties
    implementations = {
        "pure": ExpandDotPure,
        "pure_accumulate": ExpandDotPureAccumulate,
        "OpenBLAS": ExpandDotOpenBLAS,
        "MKL": ExpandDotMKL,
        "cuBLAS": ExpandDotCuBLAS,
        "CUDA (block strided)": ExpandDotCUDABlock,
        "rocBLAS": ExpandDotRocBLAS,
    }
    default_implementation = None

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None, category="Semantics")
    accumulator_type = dace.properties.TypeClassProperty(
        default=None, allow_none=True, category="Semantics", desc="Accumulator or intermediate storage type"
    )
    conjugate = dace.properties.Property(
        dtype=bool,
        default=False,
        desc="Conjugate operand _x (BLAS ?dotc / Fortran complex DOT_PRODUCT); no-op for real operands",
    )

    def __init__(self, name, n=None, accumulator_type=None, conjugate=False, **kwargs):
        super().__init__(name, inputs=OrderedSet(("_x", "_y")), outputs={"_result"}, **kwargs)
        self.n = n
        self.accumulator_type = accumulator_type
        self.conjugate = conjugate

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (x, y, res) of the three data descriptors in the
                 parent SDFG.
        """
        in_edges = state.in_edges(self)
        if len(in_edges) != 2:
            raise ValueError("Expected exactly two inputs to dot product")
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from dot product")
        out_memlet = out_edges[0].data

        desc_x, desc_y, desc_res = None, None, None
        in_memlets = [None, None]
        for e in state.in_edges(self):
            if e.dst_conn == "_x":
                desc_x = sdfg.arrays[e.data.data]
                in_memlets[0] = e.data
            elif e.dst_conn == "_y":
                desc_y = sdfg.arrays[e.data.data]
                in_memlets[1] = e.data
        for e in state.out_edges(self):
            if e.src_conn == "_result":
                desc_res = sdfg.arrays[e.data.data]

        if desc_x.dtype != desc_y.dtype:
            raise TypeError(f"Data types of input operands must be equal: {desc_x.dtype}, {desc_y.dtype}")
        if desc_x.dtype.base_type != desc_res.dtype.base_type:
            raise TypeError(f"Data types of input and output must be equal: {desc_x.dtype}, {desc_res.dtype}")

        # Squeeze input memlets
        squeezed1 = copy.deepcopy(in_memlets[0].subset)
        squeezed2 = copy.deepcopy(in_memlets[1].subset)
        sqdims1 = squeezed1.squeeze()
        sqdims2 = squeezed2.squeeze()

        if len(squeezed1.size()) != 1 or len(squeezed2.size()) != 1:
            raise ValueError("dot product only supported on 1-dimensional arrays")
        if out_memlet.subset.num_elements() != 1:
            raise ValueError("Output of dot product must be a single element")

        # We are guaranteed that there is only one non-squeezed dimension
        stride_x = desc_x.strides[sqdims1[0]]
        stride_y = desc_y.strides[sqdims2[0]]
        n = squeezed1.num_elements()
        if symbolic.inequal_symbols(squeezed1.num_elements(), squeezed2.num_elements()):
            raise ValueError("Size mismatch in inputs")

        return (desc_x, stride_x), (desc_y, stride_y), desc_res, n


# Numpy replacement
@oprepo.replaces("dace.libraries.blas.dot")
@oprepo.replaces("dace.libraries.blas.Dot")
def dot_libnode(pv: "ProgramVisitor", sdfg: SDFG, state: SDFGState, x, y, result, acctype=None):
    # Add nodes
    x_in, y_in = (state.add_read(name) for name in (x, y))
    res = state.add_write(result)

    libnode = Dot("dot", n=sdfg.arrays[x].shape[0], accumulator_type=acctype)
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(x_in, None, libnode, "_x", mm.Memlet(x))
    state.add_edge(y_in, None, libnode, "_y", mm.Memlet(y))
    state.add_edge(libnode, "_result", res, None, mm.Memlet(result))

    return []
