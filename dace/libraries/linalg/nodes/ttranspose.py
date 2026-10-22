# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""TensorTranspose library node and its pure / HPTT / cuTENSOR expansions."""

import warnings
from collections.abc import Sequence
from numbers import Number
from typing import Any

import dace
from dace import dtypes, library, nodes, properties, symbolic
from dace.codegen.common import global_code_id
from dace.data import core as datacore
from dace.libraries.blas import blas_helpers
from dace.libraries.linalg import environments
from dace.libraries.standard.environments.tiled_transpose import TiledTranspose
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES, single_core_expansion
from dace.sdfg.scope import is_devicelevel_gpu
from dace.transformation.transformation import ExpandTransformation


def moves_whole_container(desc: dace.data.Data, shape: Sequence[Any]) -> bool:
    """True iff a memlet carrying ``shape`` spans all of ``desc``."""
    return symbolic.shapes_equal(shape, desc.shape)


@library.expansion
class ExpandPure(ExpandTransformation):
    """Implements the pure expansion of TensorTranspose library node."""

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        inp_tensor, out_tensor, inp_shape, out_shape = node.validate(parent_sdfg, parent_state)

        sdfg = dace.SDFG(f"{node.label}_sdfg")
        # Shape from the memlet, strides from the container: the connector sees the SUBSET the edge
        # carries, laid out the way the array it is cut from is laid out.
        _, inp_arr = sdfg.add_array(
            "_inp_tensor", inp_shape, inp_tensor.dtype, inp_tensor.storage, strides=inp_tensor.strides
        )
        _, out_arr = sdfg.add_array(
            "_out_tensor", out_shape, out_tensor.dtype, out_tensor.storage, strides=out_tensor.strides
        )

        state = sdfg.add_state(f"{node.label}_state")
        map_params = [f"__i{i}" for i in range(len(inp_arr.shape))]
        map_rng = {i: f"0:{s}" for i, s in zip(map_params, inp_arr.shape)}
        inp_mem = dace.Memlet(expr=f"_inp_tensor[{','.join(map_params)}]")
        out_mem = dace.Memlet(expr=f"_out_tensor[{','.join([map_params[i] for i in node.axes])}]")
        inputs = {"_inp": inp_mem}
        outputs = {"_out": out_mem}
        if node.alpha == 1:
            code = "_out = _inp"
        else:
            code = f"_out = decltype(_inp)({node.alpha}) * _inp"
        if node.beta != 0:
            inputs["_inout"] = out_mem
            code = f"_out = {node.alpha} * _inp + {node.beta} * _inout"
        # A device-resident operand needs a device map: the default schedule is host code, which
        # cannot touch GPU_Global memory. ls3df_scf's double tensor transpose hits this case; it
        # reaches this expansion because hipTensor does not permute doubles. Inside a kernel the
        # same map would be a nested kernel, so there it stays sequential.
        if is_devicelevel_gpu(parent_sdfg, parent_state, node):
            schedule = dtypes.ScheduleType.Sequential
        elif out_tensor.storage in GPU_RESIDENT_STORAGES:
            schedule = dtypes.ScheduleType.GPU_Device
        else:
            schedule = dtypes.ScheduleType.Default
        state.add_mapped_tasklet(
            f"{node.label}_tasklet", map_rng, inputs, code, outputs, schedule=schedule, external_edges=True
        )

        return sdfg


@library.expansion
class ExpandSequential(ExpandTransformation):
    """One core's transpose for a node a parallel region encloses: ``dace::linalg::transpose_blocked`` tiles the
    axis contiguous in the input with the one contiguous in the output, so both stream through cache lines (the
    pure map walks the input order and scatters every write). A memlet whose rank differs from its container's
    takes the pure map."""

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        from dace.codegen.common import sym2cpp  # Avoid import loop

        inp_tensor, out_tensor, inp_shape, out_shape = node.validate(parent_sdfg, parent_state)
        rank = len(inp_shape)
        if rank != len(inp_tensor.strides) or len(out_shape) != len(out_tensor.strides) or rank < 2:
            return single_core_expansion(ExpandPure).expansion(node, parent_state, parent_sdfg)
        out_stride_of_input = [None] * rank
        for k, axis in enumerate(node.axes):
            out_stride_of_input[axis] = out_tensor.strides[k]
        ctype = out_tensor.dtype.ctype
        inp = parent_sdfg._find_new_name("_inp_tensor")
        out = parent_sdfg._find_new_name("_out_tensor")
        for edge in parent_state.in_edges_by_connector(node, "_inp_tensor"):
            edge.dst_conn = inp
        for edge in parent_state.out_edges_by_connector(node, "_out_tensor"):
            edge.src_conn = out
        listed = lambda values: ", ".join(f"int64_t({sym2cpp(v)})" for v in values)
        code = f"""
            const int64_t __shape[{rank}] = {{{listed(inp_shape)}}};
            const int64_t __in_strides[{rank}] = {{{listed(inp_tensor.strides)}}};
            const int64_t __out_strides[{rank}] = {{{listed(out_stride_of_input)}}};
            dace::linalg::transpose_blocked<{ctype}, {rank}>({inp}, {out}, __shape, __in_strides, __out_strides,
                                                         {ctype}({sym2cpp(node.alpha)}), {ctype}({sym2cpp(node.beta)}));
        """
        return nodes.Tasklet(
            node.name,
            {inp: node.in_connectors["_inp_tensor"]},
            {out: node.out_connectors["_out_tensor"]},
            code,
            language=dace.dtypes.Language.CPP,
        )


@library.expansion
class ExpandHPTT(ExpandTransformation):
    """
    Implements the TensorTranspose library node using the High-Performance Tensor Transpose Library (HPTT).
    For more information, see https://github.com/springer13/hptt.
    """

    environments = [environments.HPTT]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        from dace.codegen.common import sym2cpp  # Avoid import loop

        inp_tensor, out_tensor, inp_shape, out_shape = node.validate(parent_sdfg, parent_state)
        # HPTT expresses a non-packed operand only as an "outer size" per mode -- it cannot take a
        # stride array -- so a memlet moving a slice of a larger container has no faithful call here.
        if not (moves_whole_container(inp_tensor, inp_shape) and moves_whole_container(out_tensor, out_shape)):
            warnings.warn(
                "HPTT takes no stride array, so it cannot transpose a subset of a larger "
                "container, falling back to the pure implementation"
            )
            return ExpandPure.expansion(node, parent_state, parent_sdfg)
        axes = ",".join([sym2cpp(a) for a in node.axes])
        shape = ",".join([sym2cpp(s) for s in inp_shape])
        dchar = blas_helpers.to_blastype(inp_tensor.dtype.type).lower()
        if dchar not in ("s", "d", "c", "z"):
            raise TypeError("HPTT supports only single and double (and corresponding complex) FP datatypes")
        alpha = sym2cpp(node.alpha)
        beta = sym2cpp(node.beta)
        # The complex entry points take whether to conjugate A right after alpha, and C99 complex pointers
        conj_a = "false, " if dchar in ("c", "z") else ""
        ctype = {"c": "float _Complex", "z": "double _Complex"}.get(dchar)
        # A tasklet's connectors may not share a name with a container or symbol of its SDFG, and TensorDot's TTGT
        # expansion puts this node into an SDFG holding its own ``_out_tensor``: rename a clashing connector.
        inp = parent_sdfg._find_new_name("_inp_tensor")
        out = parent_sdfg._find_new_name("_out_tensor")
        for edge in parent_state.in_edges_by_connector(node, "_inp_tensor"):
            edge.dst_conn = inp
        for edge in parent_state.out_edges_by_connector(node, "_out_tensor"):
            edge.src_conn = out
        inp_arg = f"reinterpret_cast<const {ctype}*>({inp})" if ctype else inp
        out_arg = f"reinterpret_cast<{ctype}*>({out})" if ctype else out
        code = f"""
            int perm[{len(inp_shape)}] = {{{axes}}};
            int size[{len(inp_shape)}] = {{{shape}}};
            {dchar}TensorTranspose(perm, {len(inp_shape)}, {alpha}, {conj_a}{inp_arg}, size, NULL, {beta}, {out_arg}, NULL, omp_get_max_threads(), 1);
        """

        return nodes.Tasklet(
            node.name,
            {inp: node.in_connectors["_inp_tensor"]},
            {out: node.out_connectors["_out_tensor"]},
            code,
            language=dace.dtypes.Language.CPP,
        )


@library.expansion
class ExpandGPUTensor(ExpandTransformation):
    """
    Implements the TensorTranspose library node through a vendor tensor library's v2 permutation
    API. cuTENSOR (>= 2.0) and hipTensor expose that surface call for call -- descriptor,
    permutation, plan preference, plan, execute, destroy -- so the body below is shared and the two
    subclasses carry only the names. Their TYPE_MAPs are NOT shared: cuTENSOR takes the CUDA-wide
    ``cudaDataType`` and hipTensor its own ``hiptensorDataType_t``.

    The permutation is expressed as:
        C_{modesC} = alpha * A_{modesA}
    where modesA is the identity [0, 1, ..., n-1] and modesC encodes the
    axis permutation.

    Neither vendor's permute takes a beta (``(handle, plan, alpha, A, B, stream)``, out-of-place
    ``B = alpha*op(A)``), so ``C = alpha*perm(A) + beta*C`` permutes into a scratch tensor and an
    elementwise map folds it onto ``C``.
    See https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html#cutensorpermute
    """

    environments = []

    #: The v2 descriptor says where the elements are, so the listing order should not matter, and for
    #: cuTENSOR it does not. hipTensor's permute reads A's strides but IGNORES C's and packs C densely
    #: in the order its modes are listed (measured, ROCm 7.2.3 / gfx90a), so its subclass lists every
    #: descriptor fastest-mode-first and refuses a non-packed output.
    fastest_mode_first = False

    @classmethod
    def permute_then_fold_beta(cls, node, inp_tensor, out_tensor, inp_shape, out_shape) -> dace.SDFG:
        """``C = alpha*perm(A) + beta*C`` as this vendor's permute into a scratch tensor, then a device map
        ``C = scratch + beta*C``."""
        sdfg = dace.SDFG(f"{node.label}_beta_sdfg")
        _, inp_arr = sdfg.add_array(
            "_inp_tensor", inp_shape, inp_tensor.dtype, inp_tensor.storage, strides=inp_tensor.strides
        )
        _, out_arr = sdfg.add_array(
            "_out_tensor", out_shape, out_tensor.dtype, out_tensor.storage, strides=out_tensor.strides
        )
        _, tmp_arr = sdfg.add_array("_permuted", out_shape, out_tensor.dtype, out_tensor.storage, transient=True)
        state = sdfg.add_state(f"{node.label}_state")
        permute = TensorTranspose(f"{node.label}_permute", list(node.axes), alpha=node.alpha)
        permute.implementation = next(k for k, v in TensorTranspose.implementations.items() if v is cls)
        tmp = state.add_access("_permuted")
        state.add_edge(
            state.add_read("_inp_tensor"), None, permute, "_inp_tensor", dace.Memlet.from_array("_inp_tensor", inp_arr)
        )
        state.add_edge(permute, "_out_tensor", tmp, None, dace.Memlet.from_array("_permuted", tmp_arr))
        params = [f"__i{i}" for i in range(len(out_shape))]
        index = ",".join(params)
        state.add_mapped_tasklet(
            f"{node.label}_fold_beta",
            {p: f"0:{s}" for p, s in zip(params, out_shape)},
            {"_p": dace.Memlet(f"_permuted[{index}]"), "_c": dace.Memlet(f"_out_tensor[{index}]")},
            f"_out = _p + {node.beta} * _c",
            {"_out": dace.Memlet(f"_out_tensor[{index}]")},
            schedule=dtypes.ScheduleType.GPU_Device,
            input_nodes={"_permuted": tmp},
            external_edges=True,
        )
        return sdfg

    @classmethod
    def expansion(cls, node, parent_state, parent_sdfg):
        from dace.codegen.common import sym2cpp  # Avoid import loop

        inp_tensor, out_tensor, inp_shape, out_shape = node.validate(parent_sdfg, parent_state)

        if node.beta != 0:
            return cls.permute_then_fold_beta(node, inp_tensor, out_tensor, inp_shape, out_shape)

        ndim = len(inp_shape)
        dtype = inp_tensor.dtype.base_type

        if dtype not in cls.environments[0].TYPE_MAP:
            # Fall back to pure expansion for unsupported types (integers, etc.).
            # The pure expansion generates a GPU map when data is GPU_Global,
            # so integer transposes still execute on the GPU.
            warnings.warn(
                f"{cls.vendor} does not support {dtype} tensors, falling back to the pure "
                "implementation (still a GPU map on GPU-resident data)"
            )
            return ExpandPure.expansion(node, parent_state, parent_sdfg)

        tensor_dtype, compute_desc, alpha_type = cls.environments[0].TYPE_MAP[dtype]
        alpha_val = f"({alpha_type}){node.alpha}"

        # Input modes: identity mapping  [0, 1, ..., n-1]
        modes_a = list(range(ndim))
        # Output modes: the permutation   [axes[0], axes[1], ...]
        modes_c = list(node.axes)

        # Extents from the memlets, strides from the containers: the v2 descriptor takes both, so a
        # subset is expressed exactly -- the pointer already arrives at the subset origin.
        extent_a = [sym2cpp(s) for s in inp_shape]
        extent_c = [sym2cpp(s) for s in out_shape]
        stride_a = [sym2cpp(s) for s in inp_tensor.strides]
        stride_c = [sym2cpp(s) for s in out_tensor.strides]

        if cls.fastest_mode_first:
            # Packedness is a property of the moved REGION: a slice of a packed container is not
            # itself packed, and this vendor writes the output densely whatever its strides say.
            if not datacore.strides_equal(datacore.packed_c_strides(out_shape), out_tensor.strides):
                warnings.warn(
                    f"{cls.vendor} permute ignores the output tensor's strides and would pack a "
                    "non-packed output, falling back to the pure implementation (still a GPU map "
                    "on GPU-resident data)"
                )
                return ExpandPure.expansion(node, parent_state, parent_sdfg)
            modes_a, modes_c = modes_a[::-1], modes_c[::-1]
            extent_a, extent_c = extent_a[::-1], extent_c[::-1]
            stride_a, stride_c = stride_a[::-1], stride_c[::-1]

        modes_a_str = ", ".join(str(m) for m in modes_a)
        modes_c_str = ", ".join(str(m) for m in modes_c)
        extent_a_str = ", ".join(extent_a)
        extent_c_str = ", ".join(extent_c)
        stride_a_str = ", ".join(stride_a)
        stride_c_str = ", ".join(stride_c)

        code = f"""\
{cls.environments[0].handle_setup_code(node)}
{{
    // vendor tensor-library v2 permutation
    const uint32_t kNdim = {ndim};

    int32_t  modesA[{ndim}]   = {{{modes_a_str}}};
    int32_t  modesC[{ndim}]   = {{{modes_c_str}}};
    int64_t  extentA[{ndim}]  = {{{extent_a_str}}};
    int64_t  extentC[{ndim}]  = {{{extent_c_str}}};
    int64_t  stridesA[{ndim}] = {{{stride_a_str}}};
    int64_t  stridesC[{ndim}] = {{{stride_c_str}}};

    {alpha_type} alpha = {alpha_val};

    // tensor descriptors (v2: alignment hint in bytes, 256 is safe)
    {cls.vendor_lower}TensorDescriptor_t descA, descC;
    {cls.check}({cls.vendor_lower}CreateTensorDescriptor(
        {cls.handle}, &descA, kNdim,
        extentA, stridesA, {tensor_dtype}, 256));
    {cls.check}({cls.vendor_lower}CreateTensorDescriptor(
        {cls.handle}, &descC, kNdim,
        extentC, stridesC, {tensor_dtype}, 256));

    // operation descriptor (permutation)
    {cls.vendor_lower}OperationDescriptor_t opDesc;
    {cls.check}({cls.vendor_lower}CreatePermutation(
        {cls.handle}, &opDesc,
        descA, modesA, {cls.op_identity},
        descC, modesC,
        {compute_desc}));

    // plan preference & plan
    {cls.vendor_lower}PlanPreference_t planPref;
    {cls.check}({cls.vendor_lower}CreatePlanPreference(
        {cls.handle}, &planPref,
        {cls.algo_default}, {cls.jit_default}));

    {cls.vendor_lower}Plan_t plan;
    {cls.check}({cls.vendor_lower}CreatePlan(
        {cls.handle}, &plan, opDesc, planPref, 0));

    // execute
    {cls.check}({cls.vendor_lower}Permute(
        {cls.handle}, plan,
        (const void*)&alpha, _inp_tensor, _out_tensor,
        __dace_current_stream));

    // cleanup
    {cls.vendor_lower}DestroyPlan(plan);
    {cls.vendor_lower}DestroyPlanPreference(planPref);
    {cls.vendor_lower}DestroyOperationDescriptor(opDesc);
    {cls.vendor_lower}DestroyTensorDescriptor(descC);
    {cls.vendor_lower}DestroyTensorDescriptor(descA);
}}
"""

        tasklet = nodes.Tasklet(
            node.name, node.in_connectors, node.out_connectors, code, language=dace.dtypes.Language.CPP
        )
        return tasklet


@library.expansion
class ExpandCuTensor(ExpandGPUTensor):
    environments = [environments.cuTensor]
    vendor = "cuTENSOR"
    vendor_lower = "cutensor"
    handle = "__dace_cutensor_handle"
    check = "dace::linalg::CheckCuTensorError"
    op_identity = "CUTENSOR_OP_IDENTITY"
    algo_default = "CUTENSOR_ALGO_DEFAULT"
    jit_default = "CUTENSOR_JIT_MODE_DEFAULT"


@library.expansion
class ExpandHipTensor(ExpandGPUTensor):
    environments = [environments.hipTensor]
    vendor = "hipTensor"
    vendor_lower = "hiptensor"
    handle = "__dace_hiptensor_handle"
    check = "dace::linalg::CheckHipTensorError"
    op_identity = "HIPTENSOR_OP_IDENTITY"
    fastest_mode_first = True
    algo_default = "HIPTENSOR_ALGO_DEFAULT"
    #: NOT the literal translation of cuTENSOR's ``JIT_MODE_DEFAULT``. hipTensor accepts that name
    #: and then refuses the plan at execution with ``HIPTENSOR_STATUS_NOT_SUPPORTED``, for every
    #: rank and dtype (measured); ``NONE`` -- no just-in-time compilation, which is what cuTENSOR's
    #: DEFAULT means anyway -- is the mode its plans are actually built for.
    jit_default = "HIPTENSOR_JIT_MODE_NONE"


@library.expansion
class ExpandTensorTransposeCUDA(ExpandTransformation):
    """A rank-2 permutation IS a matrix transpose, so lower it with our own tiled kernel.

    ``cuTENSOR`` stays the right answer for a genuine tensor permutation, but for ``axes == [1, 0]``
    it is a general contraction engine doing a job one coalesced pass can do. Anything that is not a
    plain rank-2 swap with unit scaling falls through to ``cuTENSOR``'s expansion.
    """

    environments = [TiledTranspose]

    @staticmethod
    def tensor_delegate():
        """The vendor tensor library to hand a genuine permutation to, for THIS backend.

        Named by backend rather than hardcoded: a cuTENSOR delegate on an AMD build selects an
        environment that is not installed, and the configure step fails for a graph that has a
        working expansion available. For the same reason a backend whose tensor library this host
        cannot build against gets the pure permutation instead, the availability rule
        :func:`~dace.transformation.auto.auto_optimize.find_fast_library` applies to the
        ``hipTENSOR`` / ``cuTENSOR`` keys: on ROCm 6.3, whose hipTensor lacks the v2 header, the
        delegation carried ``dace_hiptensor.h`` into the build even where an fp64 permute had
        already fallen back to the pure map.
        """
        from dace.codegen.common import get_gpu_backend  # Avoid import loop

        try:
            backend = get_gpu_backend()
        except RuntimeError:
            backend = "cuda"
        delegate = ExpandHipTensor if backend == "hip" else ExpandCuTensor
        return delegate if delegate.environments[0].is_installed() else ExpandPure

    @staticmethod
    def expansion(node, state, sdfg, **kwargs):
        from dace.codegen.targets.cpp import sym2cpp

        inp_tensor, out_tensor, inp_shape, out_shape = node.validate(sdfg, state)
        plain_swap = list(node.axes) == [1, 0] and len(inp_shape) == 2 and node.alpha == 1 and node.beta == 0
        if not plain_swap:
            # ``ExpandTransformation.apply`` attaches THIS class's ``environments`` to whatever is
            # returned, so a delegation has to carry the delegate's -- otherwise cuTENSOR's expansion
            # is emitted without its library and the unit fails to link.
            delegate = ExpandTensorTransposeCUDA.tensor_delegate()
            ExpandTensorTransposeCUDA.environments = list(delegate.environments)
            return delegate.expansion(node, state, sdfg, **kwargs)
        ExpandTensorTransposeCUDA.environments = [TiledTranspose]

        # The kernel takes leading dimensions separately, so the extents are the region's and the
        # strides stay the containers' -- that pair is what makes a subset transpose correct here.
        rows, cols = inp_shape
        idstr = global_code_id(sdfg, state, node)
        ctype = inp_tensor.dtype.base_type.ctype
        prototype = (
            f"DACE_EXPORTED gpuError_t __dace_ttranspose_{idstr}(const {ctype} *__tr_in, {ctype} *__tr_out, "
            f"int __tr_rows, int __tr_cols, int __tr_ldin, int __tr_ldout, gpuStream_t __tr_stream);"
        )
        sdfg.append_global_code(prototype + "\n")
        # No ``DACE_GPU_CHECK`` in this body: the macro reports through ``__state``, which a free
        # function in the CUDA unit does not have. The status is returned and checked at the call.
        sdfg.append_global_code(
            f"{prototype}\n"
            f"gpuError_t __dace_ttranspose_{idstr}(const {ctype} *__tr_in, {ctype} *__tr_out, int __tr_rows, "
            f"int __tr_cols, int __tr_ldin, int __tr_ldout, gpuStream_t __tr_stream) {{\n"
            f"    return ::dace::cuda_transpose::transpose<{ctype}>(__tr_in, __tr_out, __tr_rows, __tr_cols, "
            f"__tr_ldin, __tr_ldout, __tr_stream);\n"
            f"}}\n",
            "cuda",
        )

        code = (
            f"DACE_GPU_CHECK(__dace_ttranspose_{idstr}(_inp_tensor, _out_tensor, (int)({sym2cpp(rows)}), "
            f"(int)({sym2cpp(cols)}), (int)({sym2cpp(inp_tensor.strides[0])}), "
            f"(int)({sym2cpp(out_tensor.strides[0])}), __dace_current_stream));"
        )
        return nodes.Tasklet(node.name, node.in_connectors, node.out_connectors, code, language=dtypes.Language.CPP)


@library.node
class TensorTranspose(nodes.LibraryNode):
    """Implements out-of-place tensor transpositions."""

    implementations = {
        "pure": ExpandPure,
        "sequential": ExpandSequential,
        "HPTT": ExpandHPTT,
        "cuTENSOR": ExpandCuTensor,
        "hipTENSOR": ExpandHipTensor,
        "CUDA": ExpandTensorTransposeCUDA,
    }
    default_implementation = "pure"

    axes = properties.ListProperty(
        element_type=int, default=[], category="Semantics", desc="Permutation of input tensor's modes"
    )
    alpha = properties.Property(dtype=Number, default=1, category="Semantics", desc="Input tensor scaling factor")

    def folds_onto_output(self) -> bool:
        return self.folds_by_beta(self.beta)

    beta = properties.Property(dtype=Number, default=0, category="Semantics", desc="Output tensor scaling factor")

    def __init__(self, name, axes=[], alpha=1, beta=0, *args, **kwargs):
        super().__init__(name, *args, inputs={"_inp_tensor"}, outputs={"_out_tensor"}, **kwargs)
        self.axes = axes
        self.alpha = alpha
        self.beta = beta

    def validate(self, sdfg, state):
        """
        Validates the tensor transposition operation.

        :return: A tuple (inp_tensor, out_tensor, inp_shape, out_shape) -- the data descriptors in
                 the parent SDFG, and the extents the memlets actually move.
        """

        inp_tensor, out_tensor = None, None
        inp_shape, out_shape = None, None
        for e in state.out_edges(self):
            if e.src_conn == "_out_tensor":
                out_tensor = sdfg.arrays[e.data.data]
                # The extents come from the SUBSET, not the descriptor: a transpose may write into a
                # slice of a larger container -- densenet appends 32 channels into channels 64:96 of
                # a 256-channel concatenation buffer -- and comparing the permuted input against the
                # container rejects a write that is correct.
                out_shape = e.data.subset.size()
        for e in state.in_edges(self):
            if e.dst_conn == "_inp_tensor":
                inp_tensor = sdfg.arrays[e.data.data]
                inp_shape = e.data.subset.size()

        if not inp_tensor:
            raise ValueError("Missing the input tensor.")
        if not out_tensor:
            raise ValueError("Missing the output tensor.")

        if inp_tensor.dtype != out_tensor.dtype:
            raise ValueError("The datatype of the input and output tensors must match.")

        if inp_tensor.storage != out_tensor.storage:
            raise ValueError("The storage of the input and output tensors must match.")

        if len(inp_shape) != len(out_shape):
            raise ValueError("The input and output tensors must have the same number of modes.")
        if len(inp_shape) != len(self.axes):
            raise ValueError("The axes list property must have as many elements as the number of tensor modes.")
        if sorted(self.axes) != list(range(len(inp_shape))):
            raise ValueError("The axes list property is not a perimutation of the input tensor's modes.")

        transposed_shape = [inp_shape[t] for t in self.axes]
        # Compared by NAME: one extent reaches the two sides through different rewrites and arrives
        # as two spellings that raw ``!=`` calls unequal, rejecting shapes that match.
        if not symbolic.shapes_equal(transposed_shape, out_shape):
            raise ValueError("The permutation of the input shape does not match the output shape.")

        return inp_tensor, out_tensor, inp_shape, out_shape
