# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Convolution library node with PyTorch ``aten.convolution`` forward semantics.

Data layout follows PyTorch: inputs and outputs are ``(N, C, spatial...)`` and
weights are ``(Cout, Cin // groups, k...)``. Arbitrary input/output strides are
taken from the connected data descriptors, so channels-last tensors are
supported transparently through their strides.
"""

from dace import SDFG, Memlet, SDFGState, dtypes, library, nodes, properties, symbolic
from dace.libraries.ml import environments as env
from dace.transformation.transformation import ExpandTransformation


def _spatial_output_size(h, k, pad, dilation, stride):
    """Floor((H + 2*pad - dilation*(k-1) - 1) / stride) + 1 (PyTorch semantics)."""
    return symbolic.int_floor(h + 2 * pad - dilation * (k - 1) - 1, stride) + 1


def _edges_by_connector(state, node):
    ins = {e.dst_conn: e for e in state.in_edges(node) if e.dst_conn}
    outs = {e.src_conn: e for e in state.out_edges(node) if e.src_conn}
    return ins, outs


@library.node
class Convolution(nodes.LibraryNode):
    """
    N-dimensional (1-3 spatial dims) convolution, PyTorch ``aten.convolution``
    forward semantics. Connectors: inputs ``_x`` (data), ``_w`` (weights), and
    optional ``_b`` (bias, shape ``[Cout]``); output ``_y``.
    """

    implementations = {}
    default_implementation = "pure"

    strides = properties.ListProperty(
        element_type=int, default=[], category="Semantics", desc="Convolution stride per spatial dimension"
    )
    pads = properties.ListProperty(
        element_type=int,
        default=[],
        category="Semantics",
        desc="Symmetric zero-padding per spatial dimension (PyTorch semantics)",
    )
    dilations = properties.ListProperty(
        element_type=int, default=[], category="Semantics", desc="Kernel dilation per spatial dimension"
    )
    output_padding = properties.ListProperty(
        element_type=int,
        default=[],
        category="Semantics",
        desc="Extra size added to the output (transposed convolutions only)",
    )
    groups = properties.Property(dtype=int, default=1, category="Semantics", desc="Number of convolution groups")
    transposed = properties.Property(
        dtype=bool,
        default=False,
        category="Semantics",
        desc="Whether this is a transposed (fractionally-strided) convolution",
    )

    def __init__(
        self,
        name,
        strides=None,
        pads=None,
        dilations=None,
        output_padding=None,
        groups=1,
        transposed=False,
        *args,
        **kwargs,
    ):
        super().__init__(name, *args, inputs={"_x", "_w"}, outputs={"_y"}, **kwargs)
        if strides is not None:
            self.strides = list(strides)
        if pads is not None:
            self.pads = list(pads)
        if dilations is not None:
            self.dilations = list(dilations)
        if output_padding is not None:
            self.output_padding = list(output_padding)
        self.groups = groups
        self.transposed = transposed

    def validate(self, sdfg: SDFG, state: SDFGState):
        """
        Validate ranks and consistency, and confirm that the output spatial
        shape matches the PyTorch formula. Returns a tuple of descriptors
        ``(x, w, b, y)`` (``b`` is ``None`` when no bias is connected).
        """
        ins, outs = _edges_by_connector(state, self)
        if "_x" not in ins or "_w" not in ins:
            raise ValueError("Convolution requires '_x' and '_w' input connectors")
        if "_y" not in outs:
            raise ValueError("Convolution requires a '_y' output connector")

        x = sdfg.arrays[ins["_x"].data.data]
        w = sdfg.arrays[ins["_w"].data.data]
        y = sdfg.arrays[outs["_y"].data.data]
        b = sdfg.arrays[ins["_b"].data.data] if "_b" in ins else None

        ndim = len(w.shape) - 2  # number of spatial dims
        if ndim < 1 or ndim > 3:
            raise ValueError("Convolution supports 1-3 spatial dimensions")
        if len(x.shape) != ndim + 2 or len(y.shape) != ndim + 2:
            raise ValueError("Input/output rank must equal (number of spatial dims) + 2")

        for lst, lname in ((self.strides, "strides"), (self.pads, "pads"), (self.dilations, "dilations")):
            if len(lst) != ndim:
                raise ValueError(f"'{lname}' must have one entry per spatial dimension ({ndim})")

        if self.groups < 1:
            raise ValueError("'groups' must be at least 1")

        if self.transposed:
            # Only forward (non-transposed) convolution is validated/implemented here.
            return x, w, b, y

        cout, cin_g = w.shape[0], w.shape[1]
        if symbolic.equal(y.shape[1], cout) is False:
            raise ValueError("Output channel dimension must match the number of weight output channels")
        if symbolic.equal(x.shape[1], cin_g * self.groups) is False:
            raise ValueError("Input channels must equal (weight input channels) * groups")
        if symbolic.equal(cout % self.groups, 0) is False:
            raise ValueError("Output channels must be divisible by 'groups'")
        if symbolic.equal(x.shape[0], y.shape[0]) is False:
            raise ValueError("Batch dimension of input and output must match")

        if b is not None:
            if len(b.shape) != 1 or symbolic.equal(b.shape[0], cout) is False:
                raise ValueError("Bias must be a 1-D array of length Cout")

        # Spatial shape check.
        for i in range(ndim):
            expected = _spatial_output_size(
                x.shape[2 + i], w.shape[2 + i], self.pads[i], self.dilations[i], self.strides[i]
            )
            if symbolic.equal(y.shape[2 + i], expected) is False:
                raise ValueError(
                    f"Output spatial dimension {i} ({y.shape[2 + i]}) does not match the expected size {expected}"
                )

        return x, w, b, y


@library.register_expansion(Convolution, "pure")
class ExpandConvolutionPure(ExpandTransformation):
    """
    Backend-agnostic native SDFG expansion. One outer map iterates over output
    elements ``(n, co, o...)``; an inner map reduces over ``(ci-in-group, k...)``
    with kernel ranges clipped per output element so that no out-of-bounds input
    read is ever generated (no padded copy of the input). Products accumulate
    into a transient of the output shape and ``_y`` is written exactly once
    (with the bias added when ``_b`` is connected).
    """

    environments = []

    @staticmethod
    def expansion(node: "Convolution", parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        x, w, b, y = node.validate(parent_sdfg, parent_state)
        if node.transposed:
            raise NotImplementedError("Transposed convolution is not implemented in the 'pure' expansion")

        ins, outs = _edges_by_connector(parent_state, node)
        ndim = len(w.shape) - 2
        has_bias = b is not None

        sdfg = SDFG("conv_pure")
        sdfg.add_array("_x", x.shape, x.dtype, strides=x.strides, storage=x.storage)
        sdfg.add_array("_w", w.shape, w.dtype, strides=w.strides, storage=w.storage)
        sdfg.add_array("_y", y.shape, y.dtype, strides=y.strides, storage=y.storage)
        if has_bias:
            sdfg.add_array("_b", b.shape, b.dtype, strides=b.strides, storage=b.storage)
        # Transient accumulator with the output shape (default/contiguous strides).
        sdfg.add_transient("_acc", y.shape, y.dtype, storage=y.storage)

        dtype = y.dtype
        cout = w.shape[0]
        cin_g = w.shape[1]
        n_batch = y.shape[0]
        cout_per_group = cout // node.groups
        out_spatial = [y.shape[2 + i] for i in range(ndim)]
        ksize = [w.shape[2 + i] for i in range(ndim)]
        in_spatial = [x.shape[2 + i] for i in range(ndim)]

        # Outer map parameter names.
        n, oc = "__n", "__oc"
        ox = [f"__o{i}" for i in range(ndim)]
        ic = "__ic"
        kk = [f"__k{i}" for i in range(ndim)]

        out_index = ", ".join([n, oc] + ox)
        out_map_ranges = {p: f"0:{symbolic.symstr(s)}" for p, s in zip([n, oc] + ox, [n_batch, cout] + out_spatial)}

        # --- State 1: zero the accumulator ---
        init_state = sdfg.add_state("init")
        init_state.add_mapped_tasklet(
            "conv_init", out_map_ranges, {}, "__out = 0", {"__out": Memlet(f"_acc[{out_index}]")}, external_edges=True
        )

        # --- State 2: accumulate products with clipped kernel ranges ---
        acc_state = sdfg.add_state_after(init_state, "accumulate")
        rx = acc_state.add_read("_x")
        rw = acc_state.add_read("_w")
        wacc = acc_state.add_write("_acc")

        ome, omx = acc_state.add_map("conv_outer", out_map_ranges)

        # Clipped inner ranges: for each spatial dim i the valid kernel taps are
        #   k in [max(0, ceil((pad - o*stride)/dil)), min(K, ceil((H + pad - o*stride)/dil)))
        inner_ranges = {ic: f"0:{symbolic.symstr(cin_g)}"}
        for i in range(ndim):
            p, s, d = node.pads[i], node.strides[i], node.dilations[i]
            lo = f"max(0, int_ceil({p} - ({ox[i]})*{s}, {d}))"
            hi = f"min({symbolic.symstr(ksize[i])}, int_ceil({symbolic.symstr(in_spatial[i])} + {p} - ({ox[i]})*{s}, {d}))"
            inner_ranges[kk[i]] = f"{lo}:{hi}"
        ime, imx = acc_state.add_map("conv_inner", inner_ranges)

        # Input channel index accounts for grouping.
        in_channel = f"({oc}) // {symbolic.symstr(cout_per_group)} * {symbolic.symstr(cin_g)} + ({ic})"
        x_pos = [f"({ox[i]})*{node.strides[i]} - {node.pads[i]} + ({kk[i]})*{node.dilations[i]}" for i in range(ndim)]
        x_index = ", ".join([n, in_channel] + x_pos)
        w_index = ", ".join([oc, ic] + kk)

        tasklet = acc_state.add_tasklet("conv_mul", {"__x", "__w"}, {"__o"}, "__o = __x * __w")

        acc_state.add_memlet_path(rx, ome, ime, tasklet, dst_conn="__x", memlet=Memlet(f"_x[{x_index}]"))
        acc_state.add_memlet_path(rw, ome, ime, tasklet, dst_conn="__w", memlet=Memlet(f"_w[{w_index}]"))
        acc_state.add_memlet_path(
            tasklet, imx, omx, wacc, src_conn="__o", memlet=Memlet(f"_acc[{out_index}]", wcr="lambda a, b: a + b")
        )

        # --- State 3: write the output exactly once (adding bias when present) ---
        fin_state = sdfg.add_state_after(acc_state, "finalize")
        if has_bias:
            fin_state.add_mapped_tasklet(
                "conv_finalize",
                out_map_ranges,
                {
                    "__acc": Memlet(f"_acc[{out_index}]"),
                    "__b": Memlet(f"_b[{oc}]"),
                },
                "__y = __acc + __b",
                {"__y": Memlet(f"_y[{out_index}]")},
                external_edges=True,
            )
        else:
            fin_state.add_mapped_tasklet(
                "conv_finalize",
                out_map_ranges,
                {"__acc": Memlet(f"_acc[{out_index}]")},
                "__y = __acc",
                {"__y": Memlet(f"_y[{out_index}]")},
                external_edges=True,
            )

        return sdfg


def _cudnn_dtype(dtype):
    mapping = {
        dtypes.float16: "CUDNN_DATA_HALF",
        dtypes.float32: "CUDNN_DATA_FLOAT",
        dtypes.float64: "CUDNN_DATA_DOUBLE",
    }
    if dtype not in mapping:
        raise NotImplementedError(f"cuDNN convolution does not support dtype {dtype}")
    return mapping[dtype]


def _miopen_dtype(dtype):
    mapping = {
        dtypes.float16: "miopenHalf",
        dtypes.float32: "miopenFloat",
        dtypes.float64: "miopenDouble",
    }
    if dtype not in mapping:
        raise NotImplementedError(f"MIOpen convolution does not support dtype {dtype}")
    return mapping[dtype]


def _c_array(values):
    from dace.codegen.targets import cpp

    return "{" + ", ".join(cpp.sym2cpp(v) for v in values) + "}"


def _gpu_validate(node, x, w, y):
    if node.transposed:
        raise NotImplementedError("Transposed convolution is not implemented in the GPU expansions")
    for desc, name in ((x, "_x"), (w, "_w"), (y, "_y")):
        if desc.storage != dtypes.StorageType.GPU_Global:
            raise ValueError(f"cuDNN/MIOpen convolution requires '{name}' to be in GPU_Global storage")


@library.register_expansion(Convolution, "cudnn")
class ExpandConvolutionCuDNN(ExpandTransformation):
    """
    cuDNN ``cudnnConvolutionForward`` expansion. Tensor descriptors are built
    from the actual shapes and strides (``cudnnSetTensorNdDescriptor`` with
    explicit strides), the forward algorithm is chosen via
    ``cudnnGetConvolutionForwardAlgorithm_v7``, and the workspace is a scratch
    allocation. Bias (when ``_b`` is connected) is added with ``cudnnAddTensor``.
    """

    environments = [env.cudnn.cuDNN]
    uid = 0

    @staticmethod
    def expansion(node: "Convolution", parent_state: SDFGState, parent_sdfg: SDFG):
        x, w, b, y = node.validate(parent_sdfg, parent_state)
        _gpu_validate(node, x, w, y)
        ndim = len(w.shape) - 2
        has_bias = b is not None

        tag = ExpandConvolutionCuDNN.uid
        ExpandConvolutionCuDNN.uid += 1
        xd = f"__state->conv_xDesc_{tag}"
        yd = f"__state->conv_yDesc_{tag}"
        wd = f"__state->conv_wDesc_{tag}"
        cd = f"__state->conv_convDesc_{tag}"
        bd = f"__state->conv_bDesc_{tag}"

        fields = [
            f"cudnnTensorDescriptor_t conv_xDesc_{tag};",
            f"cudnnTensorDescriptor_t conv_yDesc_{tag};",
            f"cudnnFilterDescriptor_t conv_wDesc_{tag};",
            f"cudnnConvolutionDescriptor_t conv_convDesc_{tag};",
        ]
        if has_bias:
            fields.append(f"cudnnTensorDescriptor_t conv_bDesc_{tag};")

        cudnn_dt = _cudnn_dtype(y.dtype)
        compute_dt = "CUDNN_DATA_FLOAT" if y.dtype == dtypes.float16 else cudnn_dt
        nb = ndim + 2

        x_dims, x_strides = _c_array(x.shape), _c_array(x.strides)
        y_dims, y_strides = _c_array(y.shape), _c_array(y.strides)
        w_dims = _c_array(w.shape)
        pads, convstr, dils = _c_array(node.pads), _c_array(node.strides), _c_array(node.dilations)

        init_code = f"""
cudnnCreateTensorDescriptor(&{xd});
cudnnCreateTensorDescriptor(&{yd});
cudnnCreateFilterDescriptor(&{wd});
cudnnCreateConvolutionDescriptor(&{cd});
{{
    int __xdims[] = {x_dims};
    int __xstrides[] = {x_strides};
    dace::ml::CheckCudnnError(cudnnSetTensorNdDescriptor({xd}, {cudnn_dt}, {nb}, __xdims, __xstrides));
    int __ydims[] = {y_dims};
    int __ystrides[] = {y_strides};
    dace::ml::CheckCudnnError(cudnnSetTensorNdDescriptor({yd}, {cudnn_dt}, {nb}, __ydims, __ystrides));
    int __wdims[] = {w_dims};
    dace::ml::CheckCudnnError(cudnnSetFilterNdDescriptor({wd}, {cudnn_dt}, CUDNN_TENSOR_NCHW, {nb}, __wdims));
    int __pads[] = {pads};
    int __convstrides[] = {convstr};
    int __dilations[] = {dils};
    dace::ml::CheckCudnnError(cudnnSetConvolutionNdDescriptor({cd}, {ndim}, __pads, __convstrides, __dilations,
        CUDNN_CROSS_CORRELATION, {compute_dt}));
    dace::ml::CheckCudnnError(cudnnSetConvolutionGroupCount({cd}, {node.groups}));
}}
"""
        exit_code = f"""
cudnnDestroyTensorDescriptor({xd});
cudnnDestroyTensorDescriptor({yd});
cudnnDestroyFilterDescriptor({wd});
cudnnDestroyConvolutionDescriptor({cd});
"""
        if has_bias:
            b_dims = _c_array([1, w.shape[0]] + [1] * ndim)
            b_strides = _c_array([w.shape[0]] + [1] * (ndim + 1))
            init_code += f"""
cudnnCreateTensorDescriptor(&{bd});
{{
    int __bdims[] = {b_dims};
    int __bstrides[] = {b_strides};
    dace::ml::CheckCudnnError(cudnnSetTensorNdDescriptor({bd}, {cudnn_dt}, {nb}, __bdims, __bstrides));
}}
"""
            exit_code += f"cudnnDestroyTensorDescriptor({bd});\n"

        code = (
            env.cudnn.cuDNN.handle_setup_code(node)
            + f"""
const float __alpha = 1.0f, __beta = 0.0f;
cudnnConvolutionFwdAlgoPerf_t __perf;
int __returned = 0;
dace::ml::CheckCudnnError(cudnnGetConvolutionForwardAlgorithm_v7(__dace_cudnn_handle, {xd}, {wd}, {cd}, {yd},
    1, &__returned, &__perf));
cudnnConvolutionFwdAlgo_t __algo = __perf.algo;
size_t __ws_size = 0;
dace::ml::CheckCudnnError(cudnnGetConvolutionForwardWorkspaceSize(__dace_cudnn_handle, {xd}, {wd}, {cd}, {yd},
    __algo, &__ws_size));
void *__workspace = nullptr;
if (__ws_size > 0) cudaMalloc(&__workspace, __ws_size);
dace::ml::CheckCudnnError(cudnnConvolutionForward(__dace_cudnn_handle, &__alpha, {xd}, _x, {wd}, _w, {cd}, __algo,
    __workspace, __ws_size, &__beta, {yd}, _y));
"""
        )
        if has_bias:
            code += f"""
dace::ml::CheckCudnnError(cudnnAddTensor(__dace_cudnn_handle, &__alpha, {bd}, _b, &__alpha, {yd}, _y));
"""
        code += "if (__workspace) cudaFree(__workspace);\n"

        in_conns = {"_x", "_w"} | ({"_b"} if has_bias else set())
        tasklet = nodes.Tasklet(
            node.name + "_cudnn",
            in_conns,
            {"_y"},
            code,
            language=dtypes.Language.CPP,
            state_fields=fields,
            code_init=init_code,
            code_exit=exit_code,
        )
        return tasklet


@library.register_expansion(Convolution, "miopen")
class ExpandConvolutionMIOpen(ExpandTransformation):
    """
    MIOpen ``miopenConvolutionForward`` expansion. Tensor descriptors are built
    from the actual shapes and strides, the forward algorithm is chosen via
    ``miopenFindConvolutionForwardAlgorithm``, and the workspace is a scratch
    allocation. Bias (when ``_b`` is connected) is added with
    ``miopenConvolutionForwardBias``.
    """

    environments = [env.miopen.MIOpen]
    uid = 0

    @staticmethod
    def expansion(node: "Convolution", parent_state: SDFGState, parent_sdfg: SDFG):
        x, w, b, y = node.validate(parent_sdfg, parent_state)
        _gpu_validate(node, x, w, y)
        ndim = len(w.shape) - 2
        has_bias = b is not None

        tag = ExpandConvolutionMIOpen.uid
        ExpandConvolutionMIOpen.uid += 1
        xd = f"__state->conv_xDesc_{tag}"
        yd = f"__state->conv_yDesc_{tag}"
        wd = f"__state->conv_wDesc_{tag}"
        cd = f"__state->conv_convDesc_{tag}"
        bd = f"__state->conv_bDesc_{tag}"

        fields = [
            f"miopenTensorDescriptor_t conv_xDesc_{tag};",
            f"miopenTensorDescriptor_t conv_yDesc_{tag};",
            f"miopenTensorDescriptor_t conv_wDesc_{tag};",
            f"miopenConvolutionDescriptor_t conv_convDesc_{tag};",
        ]
        if has_bias:
            fields.append(f"miopenTensorDescriptor_t conv_bDesc_{tag};")

        miopen_dt = _miopen_dtype(y.dtype)
        nb = ndim + 2
        x_dims, x_strides = _c_array(x.shape), _c_array(x.strides)
        y_dims, y_strides = _c_array(y.shape), _c_array(y.strides)
        w_dims, w_strides = _c_array(w.shape), _c_array(w.strides)
        pads, convstr, dils = _c_array(node.pads), _c_array(node.strides), _c_array(node.dilations)

        init_code = f"""
miopenCreateTensorDescriptor(&{xd});
miopenCreateTensorDescriptor(&{yd});
miopenCreateTensorDescriptor(&{wd});
miopenCreateConvolutionDescriptor(&{cd});
{{
    int __xdims[] = {x_dims};
    int __xstrides[] = {x_strides};
    dace::ml::CheckMiopenError(miopenSetTensorDescriptor({xd}, {miopen_dt}, {nb}, __xdims, __xstrides));
    int __ydims[] = {y_dims};
    int __ystrides[] = {y_strides};
    dace::ml::CheckMiopenError(miopenSetTensorDescriptor({yd}, {miopen_dt}, {nb}, __ydims, __ystrides));
    int __wdims[] = {w_dims};
    int __wstrides[] = {w_strides};
    dace::ml::CheckMiopenError(miopenSetTensorDescriptor({wd}, {miopen_dt}, {nb}, __wdims, __wstrides));
    int __pads[] = {pads};
    int __convstrides[] = {convstr};
    int __dilations[] = {dils};
    dace::ml::CheckMiopenError(miopenInitConvolutionNdDescriptor({cd}, {ndim}, __pads, __convstrides, __dilations,
        miopenConvolution));
    dace::ml::CheckMiopenError(miopenSetConvolutionGroupCount({cd}, {node.groups}));
}}
"""
        exit_code = f"""
miopenDestroyTensorDescriptor({xd});
miopenDestroyTensorDescriptor({yd});
miopenDestroyTensorDescriptor({wd});
miopenDestroyConvolutionDescriptor({cd});
"""
        if has_bias:
            b_dims = _c_array([1, w.shape[0]] + [1] * ndim)
            b_strides = _c_array([w.shape[0]] + [1] * (ndim + 1))
            init_code += f"""
miopenCreateTensorDescriptor(&{bd});
{{
    int __bdims[] = {b_dims};
    int __bstrides[] = {b_strides};
    dace::ml::CheckMiopenError(miopenSetTensorDescriptor({bd}, {miopen_dt}, {nb}, __bdims, __bstrides));
}}
"""
            exit_code += f"miopenDestroyTensorDescriptor({bd});\n"

        code = (
            env.miopen.MIOpen.handle_setup_code(node)
            + f"""
const float __alpha = 1.0f, __beta = 0.0f;
size_t __ws_size = 0;
dace::ml::CheckMiopenError(miopenConvolutionForwardGetWorkSpaceSize(__dace_miopen_handle, {wd}, {xd}, {cd}, {yd},
    &__ws_size));
void *__workspace = nullptr;
if (__ws_size > 0) cudaMalloc(&__workspace, __ws_size);
miopenConvAlgoPerf_t __perf;
int __returned = 0;
dace::ml::CheckMiopenError(miopenFindConvolutionForwardAlgorithm(__dace_miopen_handle, {xd}, _x, {wd}, _w, {cd},
    {yd}, _y, 1, &__returned, &__perf, __workspace, __ws_size, false));
dace::ml::CheckMiopenError(miopenConvolutionForward(__dace_miopen_handle, &__alpha, {xd}, _x, {wd}, _w, {cd},
    __perf.fwd_algo, &__beta, {yd}, _y, __workspace, __ws_size));
"""
        )
        if has_bias:
            code += f"""
dace::ml::CheckMiopenError(miopenConvolutionForwardBias(__dace_miopen_handle, &__alpha, {bd}, _b, &__beta, {yd}, _y));
"""
        code += "if (__workspace) cudaFree(__workspace);\n"

        in_conns = {"_x", "_w"} | ({"_b"} if has_bias else set())
        tasklet = nodes.Tasklet(
            node.name + "_miopen",
            in_conns,
            {"_y"},
            code,
            language=dtypes.Language.CPP,
            state_fields=fields,
            code_init=init_code,
            code_exit=exit_code,
        )
        return tasklet


# ---------------------------------------------------------------------------------------------- backward data
@library.node
class ConvolutionBackwardData(nodes.LibraryNode):
    """
    Gradient of a (non-transposed) N-d convolution with respect to its input. Connectors: inputs ``_dy`` (output
    gradient) and ``_w`` (weights); output ``_dx`` (input gradient). Same semantics/properties as :class:`Convolution`.
    """

    implementations = {}
    default_implementation = "pure"

    strides = properties.ListProperty(element_type=int, default=[], category="Semantics", desc="Stride per spatial dim")
    pads = properties.ListProperty(element_type=int, default=[], category="Semantics", desc="Symmetric padding")
    dilations = properties.ListProperty(element_type=int, default=[], category="Semantics", desc="Dilation per dim")
    groups = properties.Property(dtype=int, default=1, category="Semantics", desc="Number of convolution groups")
    transposed = properties.Property(dtype=bool, default=False, category="Semantics", desc="Transposed convolution")

    def __init__(self, name, strides=None, pads=None, dilations=None, groups=1, transposed=False, **kwargs):
        super().__init__(name, inputs={"_dy", "_w"}, outputs={"_dx"}, **kwargs)
        if strides is not None:
            self.strides = list(strides)
        if pads is not None:
            self.pads = list(pads)
        if dilations is not None:
            self.dilations = list(dilations)
        self.groups = groups
        self.transposed = transposed

    def validate(self, sdfg: SDFG, state: SDFGState):
        ins, outs = _edges_by_connector(state, self)
        if "_dy" not in ins or "_w" not in ins:
            raise ValueError("ConvolutionBackwardData requires '_dy' and '_w' input connectors")
        if "_dx" not in outs:
            raise ValueError("ConvolutionBackwardData requires a '_dx' output connector")
        dy = sdfg.arrays[ins["_dy"].data.data]
        w = sdfg.arrays[ins["_w"].data.data]
        dx = sdfg.arrays[outs["_dx"].data.data]
        ndim = len(w.shape) - 2
        if ndim < 1 or ndim > 3:
            raise ValueError("ConvolutionBackwardData supports 1-3 spatial dimensions")
        if len(dy.shape) != ndim + 2 or len(dx.shape) != ndim + 2:
            raise ValueError("Input/output rank must equal (number of spatial dims) + 2")
        for lst, lname in ((self.strides, "strides"), (self.pads, "pads"), (self.dilations, "dilations")):
            if len(lst) != ndim:
                raise ValueError(f"'{lname}' must have one entry per spatial dimension ({ndim})")
        if symbolic.equal(dx.shape[1], w.shape[1] * self.groups) is False:
            raise ValueError("Input-gradient channels must equal (weight input channels) * groups")
        if symbolic.equal(dy.shape[1], w.shape[0]) is False:
            raise ValueError("Output-gradient channels must equal the number of weight output channels")
        if not self.transposed:
            for i in range(ndim):
                expected = _spatial_output_size(
                    dx.shape[2 + i], w.shape[2 + i], self.pads[i], self.dilations[i], self.strides[i]
                )
                if symbolic.equal(dy.shape[2 + i], expected) is False:
                    raise ValueError(f"Output-gradient spatial dim {i} ({dy.shape[2 + i]}) does not match {expected}")
        return dy, w, dx


@library.register_expansion(ConvolutionBackwardData, "pure")
class ExpandConvolutionBackwardDataPure(ExpandTransformation):
    """
    Gather formulation: one outer map over ``_dx`` elements, an inner reduction over the output channels of the group
    and the output positions that could have read this input element (ranges clipped so every read is in bounds, plus a
    divisibility guard for strided/dilated windows). No padded copy; accumulation is write-conflict-resolved.
    """

    environments = []

    @staticmethod
    def expansion(node: "ConvolutionBackwardData", parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        dy, w, dx = node.validate(parent_sdfg, parent_state)
        if node.transposed:
            raise NotImplementedError("Transposed convolution is not implemented in the 'pure' expansion")
        ndim = len(w.shape) - 2

        sdfg = SDFG("conv_backward_data_pure")
        sdfg.add_array("_dy", dy.shape, dy.dtype, strides=dy.strides, storage=dy.storage)
        sdfg.add_array("_w", w.shape, w.dtype, strides=w.strides, storage=w.storage)
        sdfg.add_array("_dx", dx.shape, dx.dtype, strides=dx.strides, storage=dx.storage)
        sdfg.add_transient("_acc", dx.shape, dx.dtype, storage=dx.storage)

        cout = w.shape[0]
        cin_g = w.shape[1]
        cout_pg = cout // node.groups
        out_spatial = [dy.shape[2 + i] for i in range(ndim)]
        ksize = [w.shape[2 + i] for i in range(ndim)]

        n, ci = "__n", "__ci"
        ipos = [f"__i{i}" for i in range(ndim)]
        cog = "__cog"
        o = [f"__o{i}" for i in range(ndim)]

        dx_index = ", ".join([n, ci] + ipos)
        dx_ranges = {
            p: f"0:{symbolic.symstr(s)}"
            for p, s in zip([n, ci] + ipos, [dx.shape[0], dx.shape[1]] + [dx.shape[2 + i] for i in range(ndim)])
        }

        init = sdfg.add_state("init")
        init.add_mapped_tasklet(
            "cbd_init", dx_ranges, {}, "__out = 0", {"__out": Memlet(f"_acc[{dx_index}]")}, external_edges=True
        )

        acc_state = sdfg.add_state_after(init, "accumulate")
        rdy = acc_state.add_read("_dy")
        rw = acc_state.add_read("_w")
        wacc = acc_state.add_write("_acc")
        ome, omx = acc_state.add_map("cbd_outer", dx_ranges)

        inner_ranges = {cog: f"0:{symbolic.symstr(cout_pg)}"}
        for i in range(ndim):
            p, s, d = node.pads[i], node.strides[i], node.dilations[i]
            lo = f"max(0, int_ceil(({ipos[i]}) + {p} - ({symbolic.symstr(ksize[i])} - 1)*{d}, {s}))"
            hi = f"min({symbolic.symstr(out_spatial[i])}, int_floor(({ipos[i]}) + {p}, {s}) + 1)"
            inner_ranges[o[i]] = f"{lo}:{hi}"
        ime, imx = acc_state.add_map("cbd_inner", inner_ranges)

        co = f"(({ci}) // ({symbolic.symstr(cin_g)})) * {symbolic.symstr(cout_pg)} + ({cog})"
        ci_g = f"({ci}) % ({symbolic.symstr(cin_g)})"
        kpos = [
            f"(({ipos[i]}) + {node.pads[i]} - ({o[i]})*{node.strides[i]}) // {node.dilations[i]}" for i in range(ndim)
        ]
        dy_index = ", ".join([n, co] + o)
        w_index = ", ".join([co, ci_g] + kpos)
        valid = " and ".join(
            f"((({ipos[i]}) + {node.pads[i]} - ({o[i]})*{node.strides[i]}) % {node.dilations[i]} == 0)"
            for i in range(ndim)
        )
        tasklet = acc_state.add_tasklet("cbd_mul", {"__g", "__w"}, {"__o"}, f"__o = (__g * __w) if ({valid}) else 0")
        acc_state.add_memlet_path(rdy, ome, ime, tasklet, dst_conn="__g", memlet=Memlet(f"_dy[{dy_index}]"))
        acc_state.add_memlet_path(rw, ome, ime, tasklet, dst_conn="__w", memlet=Memlet(f"_w[{w_index}]"))
        acc_state.add_memlet_path(
            tasklet, imx, omx, wacc, src_conn="__o", memlet=Memlet(f"_acc[{dx_index}]", wcr="lambda a, b: a + b")
        )

        fin = sdfg.add_state_after(acc_state, "finalize")
        fin.add_mapped_tasklet(
            "cbd_copy",
            dx_ranges,
            {"__acc": Memlet(f"_acc[{dx_index}]")},
            "__dx = __acc",
            {"__dx": Memlet(f"_dx[{dx_index}]")},
            external_edges=True,
        )
        return sdfg


# ---------------------------------------------------------------------------------------------- backward filter
@library.node
class ConvolutionBackwardFilter(nodes.LibraryNode):
    """
    Gradient of a (non-transposed) N-d convolution with respect to its weights. Connectors: inputs ``_x`` (data) and
    ``_dy`` (output gradient); output ``_dw`` (weight gradient). Same semantics/properties as :class:`Convolution`.
    """

    implementations = {}
    default_implementation = "pure"

    strides = properties.ListProperty(element_type=int, default=[], category="Semantics", desc="Stride per spatial dim")
    pads = properties.ListProperty(element_type=int, default=[], category="Semantics", desc="Symmetric padding")
    dilations = properties.ListProperty(element_type=int, default=[], category="Semantics", desc="Dilation per dim")
    groups = properties.Property(dtype=int, default=1, category="Semantics", desc="Number of convolution groups")
    transposed = properties.Property(dtype=bool, default=False, category="Semantics", desc="Transposed convolution")

    def __init__(self, name, strides=None, pads=None, dilations=None, groups=1, transposed=False, **kwargs):
        super().__init__(name, inputs={"_x", "_dy"}, outputs={"_dw"}, **kwargs)
        if strides is not None:
            self.strides = list(strides)
        if pads is not None:
            self.pads = list(pads)
        if dilations is not None:
            self.dilations = list(dilations)
        self.groups = groups
        self.transposed = transposed

    def validate(self, sdfg: SDFG, state: SDFGState):
        ins, outs = _edges_by_connector(state, self)
        if "_x" not in ins or "_dy" not in ins:
            raise ValueError("ConvolutionBackwardFilter requires '_x' and '_dy' input connectors")
        if "_dw" not in outs:
            raise ValueError("ConvolutionBackwardFilter requires a '_dw' output connector")
        x = sdfg.arrays[ins["_x"].data.data]
        dy = sdfg.arrays[ins["_dy"].data.data]
        dw = sdfg.arrays[outs["_dw"].data.data]
        ndim = len(dw.shape) - 2
        if ndim < 1 or ndim > 3:
            raise ValueError("ConvolutionBackwardFilter supports 1-3 spatial dimensions")
        if len(x.shape) != ndim + 2 or len(dy.shape) != ndim + 2:
            raise ValueError("Input/output rank must equal (number of spatial dims) + 2")
        for lst, lname in ((self.strides, "strides"), (self.pads, "pads"), (self.dilations, "dilations")):
            if len(lst) != ndim:
                raise ValueError(f"'{lname}' must have one entry per spatial dimension ({ndim})")
        if symbolic.equal(dw.shape[0], dy.shape[1]) is False:
            raise ValueError("Weight-gradient output channels must equal the output-gradient channels")
        if symbolic.equal(x.shape[1], dw.shape[1] * self.groups) is False:
            raise ValueError("Input channels must equal (weight input channels) * groups")
        return x, dy, dw


@library.register_expansion(ConvolutionBackwardFilter, "pure")
class ExpandConvolutionBackwardFilterPure(ExpandTransformation):
    """
    Reduction formulation: one outer map over ``_dw`` elements, an inner reduction over the batch and output positions
    (clipped so the correlated input index stays in bounds). No padded copy; accumulation is write-conflict-resolved.
    """

    environments = []

    @staticmethod
    def expansion(node: "ConvolutionBackwardFilter", parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        x, dy, dw = node.validate(parent_sdfg, parent_state)
        if node.transposed:
            raise NotImplementedError("Transposed convolution is not implemented in the 'pure' expansion")
        ndim = len(dw.shape) - 2

        sdfg = SDFG("conv_backward_filter_pure")
        sdfg.add_array("_x", x.shape, x.dtype, strides=x.strides, storage=x.storage)
        sdfg.add_array("_dy", dy.shape, dy.dtype, strides=dy.strides, storage=dy.storage)
        sdfg.add_array("_dw", dw.shape, dw.dtype, strides=dw.strides, storage=dw.storage)
        sdfg.add_transient("_acc", dw.shape, dw.dtype, storage=dw.storage)

        cout = dw.shape[0]
        cin_g = dw.shape[1]
        cout_pg = cout // node.groups
        n_batch = x.shape[0]
        out_spatial = [dy.shape[2 + i] for i in range(ndim)]
        in_spatial = [x.shape[2 + i] for i in range(ndim)]
        ksize = [dw.shape[2 + i] for i in range(ndim)]

        co, cig = "__co", "__cig"
        kpos = [f"__k{i}" for i in range(ndim)]
        n = "__n"
        o = [f"__o{i}" for i in range(ndim)]

        dw_index = ", ".join([co, cig] + kpos)
        dw_ranges = {p: f"0:{symbolic.symstr(s)}" for p, s in zip([co, cig] + kpos, [cout, cin_g] + ksize)}

        init = sdfg.add_state("init")
        init.add_mapped_tasklet(
            "cbf_init", dw_ranges, {}, "__out = 0", {"__out": Memlet(f"_acc[{dw_index}]")}, external_edges=True
        )

        acc_state = sdfg.add_state_after(init, "accumulate")
        rx = acc_state.add_read("_x")
        rdy = acc_state.add_read("_dy")
        wacc = acc_state.add_write("_acc")
        ome, omx = acc_state.add_map("cbf_outer", dw_ranges)

        inner_ranges = {n: f"0:{symbolic.symstr(n_batch)}"}
        for i in range(ndim):
            p, s, d = node.pads[i], node.strides[i], node.dilations[i]
            lo = f"max(0, int_ceil({p} - ({kpos[i]})*{d}, {s}))"
            hi = f"min({symbolic.symstr(out_spatial[i])}, int_ceil({symbolic.symstr(in_spatial[i])} + {p} - ({kpos[i]})*{d}, {s}))"
            inner_ranges[o[i]] = f"{lo}:{hi}"
        ime, imx = acc_state.add_map("cbf_inner", inner_ranges)

        ci = f"(({co}) // {symbolic.symstr(cout_pg)}) * {symbolic.symstr(cin_g)} + ({cig})"
        xpos = [f"({o[i]})*{node.strides[i]} - {node.pads[i]} + ({kpos[i]})*{node.dilations[i]}" for i in range(ndim)]
        dy_index = ", ".join([n, co] + o)
        x_index = ", ".join([n, ci] + xpos)
        tasklet = acc_state.add_tasklet("cbf_mul", {"__g", "__x"}, {"__o"}, "__o = __g * __x")
        acc_state.add_memlet_path(rdy, ome, ime, tasklet, dst_conn="__g", memlet=Memlet(f"_dy[{dy_index}]"))
        acc_state.add_memlet_path(rx, ome, ime, tasklet, dst_conn="__x", memlet=Memlet(f"_x[{x_index}]"))
        acc_state.add_memlet_path(
            tasklet, imx, omx, wacc, src_conn="__o", memlet=Memlet(f"_acc[{dw_index}]", wcr="lambda a, b: a + b")
        )

        fin = sdfg.add_state_after(acc_state, "finalize")
        fin.add_mapped_tasklet(
            "cbf_copy",
            dw_ranges,
            {"__acc": Memlet(f"_acc[{dw_index}]")},
            "__dw = __acc",
            {"__dw": Memlet(f"_dw[{dw_index}]")},
            external_edges=True,
        )
        return sdfg
