# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""TensorTranspose library node and its pure / HPTT / cuTENSOR expansions."""
import dace
import multiprocessing
from dace import library, nodes, properties
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas import blas_helpers
from numbers import Number
from dace.libraries.linalg import environments
import warnings


@library.expansion
class ExpandPure(ExpandTransformation):
    """ Implements the pure expansion of TensorTranspose library node. """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        inp_tensor, out_tensor = node.validate(parent_sdfg, parent_state)

        sdfg = dace.SDFG(f"{node.label}_sdfg")
        _, inp_arr = sdfg.add_array("_inp_tensor",
                                    inp_tensor.shape,
                                    inp_tensor.dtype,
                                    inp_tensor.storage,
                                    strides=inp_tensor.strides)
        _, out_arr = sdfg.add_array("_out_tensor",
                                    out_tensor.shape,
                                    out_tensor.dtype,
                                    out_tensor.storage,
                                    strides=out_tensor.strides)

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
        state.add_mapped_tasklet(f"{node.label}_tasklet", map_rng, inputs, code, outputs, external_edges=True)

        return sdfg


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

        inp_tensor, out_tensor = node.validate(parent_sdfg, parent_state)
        axes = ','.join([sym2cpp(a) for a in node.axes])
        shape = ','.join([sym2cpp(s) for s in inp_tensor.shape])
        dchar = blas_helpers.to_blastype(inp_tensor.dtype.type).lower()
        if dchar not in ('s', 'd', 'c', 'z'):
            raise TypeError("HPTT supports only single and double (and corresponding complex) FP datatypes")
        alpha = sym2cpp(node.alpha)
        beta = sym2cpp(node.beta)
        code = f"""
            int perm[{len(inp_tensor.shape)}] = {{{axes}}};
            int size[{len(inp_tensor.shape)}] = {{{shape}}};
            {dchar}TensorTranspose(perm, {len(inp_tensor.shape)}, {alpha}, _inp_tensor, size, NULL, {beta}, _out_tensor, NULL, {multiprocessing.cpu_count()}, 1);
        """

        tasklet = nodes.Tasklet(node.name,
                                node.in_connectors,
                                node.out_connectors,
                                code,
                                language=dace.dtypes.Language.CPP)

        return tasklet


@library.expansion
class ExpandCuTensor(ExpandTransformation):
    """
    Implements the TensorTranspose library node using the cuTENSOR v2 API
    (cutensorPermute). Requires cuTENSOR >= 2.0.

    The permutation is expressed as:
        C_{modesC} = alpha * A_{modesA}
    where modesA is the identity [0, 1, ..., n-1] and modesC encodes the
    axis permutation.

    NOTE: beta != 0 is not supported by cutensorPermute (its signature is
    ``(handle, plan, alpha, A, B, stream)`` -- out-of-place ``B = alpha*op(A)``,
    no beta term). For C = alpha * perm(A) + beta * C, use ExpandPure or
    implement via cutensorElementwiseBinary.
    See https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html#cutensorpermute
    """

    environments = [environments.cuTensor]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        from dace.codegen.common import sym2cpp  # Avoid import loop

        inp_tensor, out_tensor = node.validate(parent_sdfg, parent_state)

        if node.beta != 0:
            raise NotImplementedError("cuTENSOR v2 cutensorPermute does not support beta != 0. "
                                      "Use the 'pure' expansion or implement via cutensorElementwiseBinary.")

        ndim = len(inp_tensor.shape)
        dtype = inp_tensor.dtype.base_type

        if dtype not in environments.cuTensor.TYPE_MAP:
            # Fall back to pure expansion for unsupported types (integers, etc.).
            # The pure expansion generates a GPU map when data is GPU_Global,
            # so integer transposes still execute on the GPU.
            warnings.warn("CuTensor does not support integer tensors, falling back to pure implementation")
            return ExpandPure.expansion(node, parent_state, parent_sdfg)

        cutensor_dtype, compute_desc, alpha_type = environments.cuTensor.TYPE_MAP[dtype]
        alpha_val = f"({alpha_type}){node.alpha}"

        # Input modes: identity mapping  [0, 1, ..., n-1]
        modes_a = list(range(ndim))
        # Output modes: the permutation   [axes[0], axes[1], ...]
        modes_c = list(node.axes)

        modes_a_str = ', '.join(str(m) for m in modes_a)
        modes_c_str = ', '.join(str(m) for m in modes_c)
        extent_a_str = ', '.join(sym2cpp(s) for s in inp_tensor.shape)
        extent_c_str = ', '.join(sym2cpp(s) for s in out_tensor.shape)
        stride_a_str = ', '.join(sym2cpp(s) for s in inp_tensor.strides)
        stride_c_str = ', '.join(sym2cpp(s) for s in out_tensor.strides)

        code = f"""\
{environments.cuTensor.handle_setup_code(node)}
{{
    // cuTENSOR v2 permutation
    const uint32_t kNdim = {ndim};

    int32_t  modesA[{ndim}]   = {{{modes_a_str}}};
    int32_t  modesC[{ndim}]   = {{{modes_c_str}}};
    int64_t  extentA[{ndim}]  = {{{extent_a_str}}};
    int64_t  extentC[{ndim}]  = {{{extent_c_str}}};
    int64_t  stridesA[{ndim}] = {{{stride_a_str}}};
    int64_t  stridesC[{ndim}] = {{{stride_c_str}}};

    {alpha_type} alpha = {alpha_val};

    // tensor descriptors (v2: alignment hint in bytes, 256 is safe)
    cutensorTensorDescriptor_t descA, descC;
    dace::linalg::CheckCuTensorError(cutensorCreateTensorDescriptor(
        __dace_cutensor_handle, &descA, kNdim,
        extentA, stridesA, {cutensor_dtype}, 256));
    dace::linalg::CheckCuTensorError(cutensorCreateTensorDescriptor(
        __dace_cutensor_handle, &descC, kNdim,
        extentC, stridesC, {cutensor_dtype}, 256));

    // operation descriptor (permutation)
    cutensorOperationDescriptor_t opDesc;
    dace::linalg::CheckCuTensorError(cutensorCreatePermutation(
        __dace_cutensor_handle, &opDesc,
        descA, modesA, CUTENSOR_OP_IDENTITY,
        descC, modesC,
        {compute_desc}));

    // plan preference & plan
    cutensorPlanPreference_t planPref;
    dace::linalg::CheckCuTensorError(cutensorCreatePlanPreference(
        __dace_cutensor_handle, &planPref,
        CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_DEFAULT));

    cutensorPlan_t plan;
    dace::linalg::CheckCuTensorError(cutensorCreatePlan(
        __dace_cutensor_handle, &plan, opDesc, planPref, 0));

    // execute
    dace::linalg::CheckCuTensorError(cutensorPermute(
        __dace_cutensor_handle, plan,
        (const void*)&alpha, _inp_tensor, _out_tensor,
        __dace_current_stream));

    // cleanup
    cutensorDestroyPlan(plan);
    cutensorDestroyPlanPreference(planPref);
    cutensorDestroyOperationDescriptor(opDesc);
    cutensorDestroyTensorDescriptor(descC);
    cutensorDestroyTensorDescriptor(descA);
}}
"""

        tasklet = nodes.Tasklet(node.name,
                                node.in_connectors,
                                node.out_connectors,
                                code,
                                language=dace.dtypes.Language.CPP)
        return tasklet


@library.node
class TensorTranspose(nodes.LibraryNode):
    """ Implements out-of-place tensor transpositions. """

    implementations = {
        "pure": ExpandPure,
        "HPTT": ExpandHPTT,
        "cuTENSOR": ExpandCuTensor,
    }
    default_implementation = 'pure'

    axes = properties.ListProperty(element_type=int, default=[], desc="Permutation of input tensor's modes")
    alpha = properties.Property(dtype=Number, default=1, desc="Input tensor scaling factor")
    beta = properties.Property(dtype=Number, default=0, desc="Output tensor scaling factor")

    def __init__(self, name, axes=[], alpha=1, beta=0, *args, **kwargs):
        super().__init__(name, *args, inputs={"_inp_tensor"}, outputs={"_out_tensor"}, **kwargs)
        self.axes = axes
        self.alpha = alpha
        self.beta = beta

    def validate(self, sdfg, state):
        """
        Validates the tensor transposition operation.
        :return: A tuple (inp_tensor, out_tensor) for the data descriptors in the parent SDFG.
        """

        inp_tensor, out_tensor = None, None
        for e in state.out_edges(self):
            if e.src_conn == "_out_tensor":
                out_tensor = sdfg.arrays[e.data.data]
        for e in state.in_edges(self):
            if e.dst_conn == "_inp_tensor":
                inp_tensor = sdfg.arrays[e.data.data]

        if not inp_tensor:
            raise ValueError("Missing the input tensor.")
        if not out_tensor:
            raise ValueError("Missing the output tensor.")

        if inp_tensor.dtype != out_tensor.dtype:
            raise ValueError("The datatype of the input and output tensors must match.")

        if inp_tensor.storage != out_tensor.storage:
            raise ValueError("The storage of the input and output tensors must match.")

        if len(inp_tensor.shape) != len(out_tensor.shape):
            raise ValueError("The input and output tensors must have the same number of modes.")
        if len(inp_tensor.shape) != len(self.axes):
            raise ValueError("The axes list property must have as many elements as the number of tensor modes.")
        if sorted(self.axes) != list(range(len(inp_tensor.shape))):
            raise ValueError("The axes list property is not a perimutation of the input tensor's modes.")

        transposed_shape = [inp_tensor.shape[t] for t in self.axes]
        if transposed_shape != list(out_tensor.shape):
            raise ValueError("The permutation of the input shape does not match the output shape.")

        return inp_tensor, out_tensor
