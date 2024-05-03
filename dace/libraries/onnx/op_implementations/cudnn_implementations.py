import functools
from typing import Union, Optional, Tuple, List

import dace
from dace import SDFGState, nodes as nd, SDFG, dtypes, data as dt
from dace.codegen.common import sym2cpp

from dace.libraries.onnx import environments
from dace.libraries.onnx.converters import clean_onnx_name
from dace.libraries.onnx.forward_implementation_abc import ONNXForward
from dace.libraries.onnx.nodes import onnx_op
from dace.libraries.onnx.op_implementations import op_implementation, empty_sdfg_for_node
from dace.util import all_equal, in_desc_with_name, out_desc_with_name, remove_output_connector


def _prod(sequence):
    return functools.reduce(lambda a, b: a * b, sequence, 1)


def _get_tensor_layout(desc: dt.Array) -> Optional[str]:
    """ Detect the layout of a 3d/4d tensor.

        :param desc: the tensor.
        :return: "NCHW", "NHWC" or None.
    """

    if len(desc.shape) == 1:
        # just return anything
        return "NCHW"

    if len(desc.shape) != 3 and len(desc.shape) != 4:
        raise ValueError("Tensor with dimension != {3,4} is not supported")

    # in ONNX, tensor the dimensions are ordered N C H W
    # strides that the contiguous tensor would have
    cont_strides = [_prod(desc.shape[i + 1:]) for i in range(len(desc.shape))]

    if len(desc.shape) == 4:
        nhwc_shape = [
            desc.shape[0], desc.shape[3], desc.shape[1], desc.shape[2]
        ]
    elif len(desc.shape) == 3:
        nhwc_shape = [desc.shape[0], desc.shape[2], desc.shape[1]]

    # strides that a nhwc tensor would have if it was contiguous
    nhwc_contiguous_strides = [
        _prod(nhwc_shape[i + 1:]) for i in range(len(desc.shape))
    ]
    # strides that the nhwc tensor would have if viewed as a nchw tensor
    if len(desc.shape) == 4:
        nhwc_reshaped_strides = [
            nhwc_contiguous_strides[0], nhwc_contiguous_strides[3],
            nhwc_contiguous_strides[1], nhwc_contiguous_strides[2]
        ]
    elif len(desc.shape) == 3:
        nhwc_reshaped_strides = [
            nhwc_contiguous_strides[0], nhwc_contiguous_strides[2],
            nhwc_contiguous_strides[1]
        ]

    if all_equal(desc.strides, cont_strides):
        return "NCHW"
    elif all_equal(desc.strides, nhwc_reshaped_strides):
        return "NHWC"
    else:
        return None


def _cudnn_tensor_descriptor_code(
        desc: dt.Array,
        state_field_name: str,
        filter: bool,
        shape: Optional[List[int]] = None) -> Tuple[str, str]:
    """ Emit the cudnn code for the tensor descriptor for a given dace descriptor.

        :param desc: the descriptor of the dace tensor.
        :param state_field_name: the name of the pointer variable where the descriptor should be stored.
        :param filter: True if the tensor is a filter.
        :param shape: (optional) the shape to override the shape of the tensor
        :return: the init and exit code
    """

    # detect layout
    layout = _get_tensor_layout(desc)
    if shape is None:
        shape = desc.shape
    if len(shape) < 4:
        shape = list(shape) + [1] * (4 - len(shape))
    elif len(shape) > 4:
        raise ValueError("Tensor with dimension > 4 is not supported")

    assert layout is not None, "layout changed after can_be_applied"
    f_or_t_str = 'Filter' if filter else 'Tensor'

    layout_str = f"CUDNN_TENSOR_{layout}"
    dtype_str = _DACE_DTYPE_TO_CUDNN_DTYPE[desc.dtype]
    init_code = f"""
    __state->{state_field_name} = new cudnn{f_or_t_str}Descriptor_t;
    dace::cudnn::CheckCudnnError(cudnnCreate{f_or_t_str}Descriptor(__state->{state_field_name}));
    dace::cudnn::CheckCudnnError(cudnnSet{f_or_t_str}4dDescriptor(
        *__state->{state_field_name}, 
        {dtype_str if filter else layout_str},
        {layout_str if filter else dtype_str},
        {",".join(str(s) for s in shape)}
    ));
    """
    exit_code = f"""\
    dace::cudnn::CheckCudnnError(cudnnDestroy{f_or_t_str}Descriptor(*__state->{state_field_name}));
    delete __state->{state_field_name};
    """
    return init_code, exit_code


_DACE_DTYPE_TO_CUDNN_DTYPE = {
    dace.float32: "CUDNN_DATA_FLOAT",
    dace.float64: "CUDNN_DATA_DOUBLE",
    dace.uint8: "CUDNN_DATA_UINT8",
    dace.int8: "CUDNN_DATA_INT8",
    dace.int32: "CUDNN_DATA_INT32",
}


@op_implementation(op="Conv", name="cuDNN")
class CudnnConvolution(ONNXForward):
    """ Convolution implementation that uses cuDNN.

        This node will check for the existence of a _algorithm attribute on the ONNXConv node it is expanding.
        If this attribute does not exist, it will use `CudnnConvolution.default_algorithm`.

    """
    environments = [environments.cuDNN]
    default_algorithm = "auto"

    # choices for algorithms
    algorithms = [
        "auto"
        "implicit_gemm",
        "implicit_precomp_gemm",
        "gemm",
        "direct",
        "fft",
        "fft_tiling",
        "winograd",
        "winograd_nonfused",
    ]
    search_ws_size = 32 * 1024 * 1024

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:

        descs = [("X", in_desc_with_name(node, state, sdfg, "X")),
                 ("W", in_desc_with_name(node, state, sdfg, "W")),
                 ("Y", out_desc_with_name(node, state, sdfg, "Y"))]

        if "B" in node.in_connectors:
            descs.append(("B", in_desc_with_name(node, state, sdfg, "B")))

        for name, desc in descs:
            # check that the dtype is supported by cudnn
            if desc.dtype not in [
                    dace.float32, dace.float64, dace.uint8, dace.int8,
                    dace.int32
            ]:
                return False
            # only 1d/2d convs for now; ONNX supports N dimensional
            if name != "B" and len(desc.shape) not in {3, 4}:
                return False

            if not isinstance(desc, dt.Array):
                return False

            # check that the layout is supported by cudnn
            if name != "B" and _get_tensor_layout(desc) is None:
                return False

        # padding must be symmetric
        dims = len(descs[0][1].shape) - 2
        for i in range(dims):
            if node.pads[i] != node.pads[dims + i]:
                return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> Union[nd.Node, SDFG]:

        nsdfg, nstate, inputs, outputs = empty_sdfg_for_node(sdfg, state, node)

        if "B" in inputs:
            nstate.remove_node(inputs["B"])
            Y = out_desc_with_name(node, state, sdfg, "Y")
            # add broadcast state
            init_state = nsdfg.add_state_before(nstate, label="broadcast_bias")
            # yapf: disable
            init_state.add_mapped_tasklet("broadcast_bias",
                                          map_ranges={
                                              "i{}".format(i): "0:{}".format(s)
                                              for i, s in enumerate(Y.shape)
                                          },
                                          inputs=dict(
                                              b=dace.Memlet("B[i1]")
                                          ),
                                          code="y = b".format(),
                                          outputs=dict(
                                              y=dace.Memlet("Y[{}]".format(
                                                  ", ".join("i{}".format(i)
                                                            for i, _ in enumerate(Y.shape))))
                                          ),
                                          external_edges=True)
            # yapf: enable

        X_desc = in_desc_with_name(node, state, sdfg, "X")

        T = X_desc.dtype

        unique_id = "{}_{}_{}_{}".format(clean_onnx_name(node.name),
                                         sdfg.sdfg_id, sdfg.node_id(state),
                                         state.node_id(node))

        init_code = ""
        finalize_code = ""

        # setup tensor descriptors
        for edge, is_input in node.iter_edges(state):
            conn = edge.dst_conn if is_input else edge.src_conn
            desc = in_desc_with_name(node, state, sdfg,
                                     conn) if is_input else out_desc_with_name(
                                         node, state, sdfg, conn)
            assert isinstance(desc, dt.Array)
            if conn == "B":
                # bias will be handled separately
                continue
            is_filter = conn == "W"
            init, exit = _cudnn_tensor_descriptor_code(
                desc, f"{unique_id}_{conn}_desc", is_filter)
            init_code += init
            finalize_code += exit

        # setup conv descriptor
        # we know padding is symmetric
        if len(node.strides) == 1:
            pad_h, pad_w = node.pads[0], 0
            stride_h, stride_w = node.strides[0], 1
            dilation_h, dilation_w = node.dilations[0], 1
        else:
            pad_h, pad_w = node.pads[0], node.pads[1]
            stride_h, stride_w = node.strides
            dilation_h, dilation_w = node.dilations
        init_code += f"""
        __state->{unique_id}_conv_desc = new cudnnConvolutionDescriptor_t; 
        dace::cudnn::CheckCudnnError(cudnnCreateConvolutionDescriptor(__state->{unique_id}_conv_desc));
        dace::cudnn::CheckCudnnError(cudnnSetConvolution2dDescriptor(
            *__state->{unique_id}_conv_desc,
            {pad_h},
            {pad_w},
            {stride_h},
            {stride_w},
            {dilation_h},
            {dilation_w},
            CUDNN_CROSS_CORRELATION,
            {_DACE_DTYPE_TO_CUDNN_DTYPE[T]}));
        dace::cudnn::CheckCudnnError(cudnnSetConvolutionMathType(
            *__state->{unique_id}_conv_desc,
            CUDNN_DEFAULT_MATH));
        """

        if node.group != 1:
            init_code += f"""
            dace::cudnn::CheckCudnnError(cudnnSetConvolutionGroupCount(
                *__state->{unique_id}_conv_desc,
                {node.group}
                ));
            """

        finalize_code += f"""
        dace::cudnn::CheckCudnnError(cudnnDestroyConvolutionDescriptor(*__state->{unique_id}_conv_desc));
        delete __state->{unique_id}_conv_desc;
        """
        # setup algo
        init_code += f"""
        {environments.cuDNN.handle_setup_code(node, init_stream=False)}
        """

        if hasattr(node, "_algorithm"):
            algo = node._algorithm
        else:
            algo = CudnnConvolution.default_algorithm
        if algo == "auto":

            # setup fake data
            free_fake_data_code, fake_data_init_code = setup_fake_data(
                node, sdfg, state, False)

            init_code += f"""
            // setup fake data
            {fake_data_init_code}

            // setup workspace
            void *search_ws; 
            cudaMalloc(&search_ws, {CudnnConvolution.search_ws_size});
            // run search
            cudnnConvolutionFwdAlgoPerf_t results;
            int algo_count = 1;
            dace::cudnn::CheckCudnnError(cudnnFindConvolutionForwardAlgorithmEx(
                __dace_cudnn_handle,
                *__state->{unique_id}_X_desc,
                fake_X,
                *__state->{unique_id}_W_desc,
                fake_W,
                *__state->{unique_id}_conv_desc,
                *__state->{unique_id}_Y_desc,
                fake_Y,
                1,
                &algo_count,
                &results,
                search_ws,
                {CudnnConvolution.search_ws_size}
            ));
            cudaFree(search_ws);
            __state->{unique_id}_algo = new cudnnConvolutionFwdAlgo_t;
            *__state->{unique_id}_algo = results.algo;
            printf("{unique_id} using algo %d\\n", *__state->{unique_id}_algo);
            
            {free_fake_data_code}
            """
        else:
            init_code += f"""
            __state->{unique_id}_algo = new cudnnConvolutionFwdAlgo_t;
            *__state->{unique_id}_algo = CUDNN_CONVOLUTION_FWD_ALGO_{algo.upper()};
            """

        finalize_code += f"""
             delete __state->{unique_id}_algo;
        """

        # setup workspace
        init_code += \
            f"""
        // Setup workspace for {unique_id}
        size_t ws_size;
        dace::cudnn::CheckCudnnError(cudnnGetConvolutionForwardWorkspaceSize(
            __dace_cudnn_handle,
            *__state->{unique_id}_X_desc,
            *__state->{unique_id}_W_desc,
            *__state->{unique_id}_conv_desc,
            *__state->{unique_id}_Y_desc,
            *__state->{unique_id}_algo,
            &ws_size));
        __state->{unique_id}_workspace_size = new size_t;
        *__state->{unique_id}_workspace_size = ws_size;
        cudaMalloc(&__state->{unique_id}_workspace, ws_size);
        """
        finalize_code += f"""
        cudaFree(__state->{unique_id}_workspace);
        delete __state->{unique_id}_workspace_size;
        """

        tasklet_code = f"""
        {environments.cuDNN.handle_setup_code(node)}
        float alpha = 1.f;
        float beta = {"1.f" if "B" in inputs else "0.f"};
        dace::cudnn::CheckCudnnError(cudnnConvolutionForward(
            __dace_cudnn_handle,
            &alpha,
            *__state->{unique_id}_X_desc,
            _X,
            *__state->{unique_id}_W_desc,
            _W,
            *__state->{unique_id}_conv_desc,
            *__state->{unique_id}_algo,
            __state->{unique_id}_workspace,
            *__state->{unique_id}_workspace_size,
            &beta,
            *__state->{unique_id}_Y_desc,
            _Y
        ));
        """

        init_code = "{\n" + init_code + "\n}"
        finalize_code = "{\n" + finalize_code + "\n}"

        tasklet = nstate.add_tasklet(
            unique_id, {
                "_X": dace.pointer(T),
                "_W": dace.pointer(T)
            }, {"_Y": dace.pointer(T)},
            tasklet_code,
            dtypes.Language.CPP,
            code_init=init_code,
            code_exit=finalize_code,
            state_fields=[
                f"cudnnConvolutionDescriptor_t *{unique_id}_conv_desc;",
                f"cudnnTensorDescriptor_t *{unique_id}_Y_desc;",
                f"cudnnTensorDescriptor_t *{unique_id}_X_desc;",
                f"cudnnConvolutionFwdAlgo_t *{unique_id}_algo;",
                f"cudnnFilterDescriptor_t *{unique_id}_W_desc;",
                f"float *{unique_id}_workspace;",
                f"size_t *{unique_id}_workspace_size;"
            ])
        nstate.add_edge(inputs["X"], None, tasklet, "_X",
                        nsdfg.make_array_memlet("X"))
        nstate.add_edge(inputs["W"], None, tasklet, "_W",
                        nsdfg.make_array_memlet("W"))

        nstate.add_edge(tasklet, "_Y", outputs["Y"], None,
                        nsdfg.make_array_memlet("Y"))

        return nsdfg


def setup_fake_data(node, sdfg, state, bwd) -> Tuple[str, str]:
    free_fake_data_code = ""
    init_code = ""

    for edge, is_input in node.iter_edges(state):
        conn = edge.dst_conn if is_input else edge.src_conn
        desc = in_desc_with_name(node, state, sdfg,
                                 conn) if is_input else out_desc_with_name(
                                     node, state, sdfg, conn)
        assert isinstance(desc, dt.Array)
        init_code += f"""
            void *fake_{conn};
            cudaMalloc(&fake_{conn}, {sym2cpp(desc.total_size * desc.dtype.bytes)});
            cudaMemset(fake_{conn}, 0, {sym2cpp(desc.total_size * desc.dtype.bytes)});
            """
        free_fake_data_code += f"""
            cudaFree(fake_{conn});
            """
        if bwd:
            init_code += f"""
                void *fake_d{conn};
                cudaMalloc(&fake_d{conn}, {sym2cpp(desc.total_size * desc.dtype.bytes)});
                cudaMemset(fake_d{conn}, 0, {sym2cpp(desc.total_size * desc.dtype.bytes)});
                """
            free_fake_data_code += f"""
                cudaFree(fake_d{conn});
                """

    return free_fake_data_code, init_code


@op_implementation(op="BatchNormalization", name="cuDNN")
class CudnnBatchNormalizationTraining(ONNXForward):
    environments = [environments.cuDNN]

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        X = in_desc_with_name(node, state, sdfg, "X")
        if len(X.shape) != 4:
            return False

        # only for training for now
        if not {"out_mean", "out_var", "saved_mean", "saved_var"}.issubset(
                node.out_connectors):
            return False
        if not {"scale", "B"}.issubset(node.in_connectors):
            return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp,
                state: SDFGState,
                sdfg: SDFG,
                reserved_ptr=False) -> Union[nd.Node, SDFG]:

        nsdfg, nstate, inputs, outputs = empty_sdfg_for_node(sdfg, state, node)

        X_desc = in_desc_with_name(node, state, sdfg, "X")
        T = X_desc.dtype

        unique_id = "{}_{}_{}_{}".format(clean_onnx_name(node.name),
                                         sdfg.sdfg_id, sdfg.node_id(state),
                                         state.node_id(node))

        init_code = ""
        finalize_code = ""

        init, exit = _cudnn_tensor_descriptor_code(inputs["X"].desc(nsdfg),
                                                   f"{unique_id}_X_desc",
                                                   False)
        init_code += init
        finalize_code += exit

        init, exit = _cudnn_tensor_descriptor_code(outputs["Y"].desc(nsdfg),
                                                   f"{unique_id}_Y_desc",
                                                   False)
        init_code += init
        finalize_code += exit

        # setup scale descriptor
        init_code += f"""
        __state->{unique_id}_scale_desc = new cudnnTensorDescriptor_t; 
        dace::cudnn::CheckCudnnError(cudnnCreateTensorDescriptor(__state->{unique_id}_scale_desc));
        dace::cudnn::CheckCudnnError(cudnnDeriveBNTensorDescriptor(
            *__state->{unique_id}_scale_desc,
            *__state->{unique_id}_X_desc,
            CUDNN_BATCHNORM_SPATIAL));
        """
        finalize_code += f"""
        dace::cudnn::CheckCudnnError(cudnnDestroyTensorDescriptor(*__state->{unique_id}_scale_desc));
        delete __state->{unique_id}_scale_desc;
        """

        # setup workspace and reserve space
        init_code += \
            f"""
        {environments.cuDNN.handle_setup_code(node, init_stream=False)}
        // Setup workspace and reserved space for {unique_id}
        size_t ws_size;
        dace::cudnn::CheckCudnnError(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
            __dace_cudnn_handle,
            CUDNN_BATCHNORM_SPATIAL,
            CUDNN_BATCHNORM_OPS_BN,
            *__state->{unique_id}_X_desc,
            nullptr,
            *__state->{unique_id}_Y_desc,
            *__state->{unique_id}_scale_desc,
            nullptr,
            &ws_size));
        __state->{unique_id}_workspace_size = new size_t;
        *__state->{unique_id}_workspace_size = ws_size;
        cudaMalloc(&__state->{unique_id}_workspace, ws_size);

        size_t rs_size;
        dace::cudnn::CheckCudnnError(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
            __dace_cudnn_handle,
            CUDNN_BATCHNORM_SPATIAL,
            CUDNN_BATCHNORM_OPS_BN,
            nullptr,
            *__state->{unique_id}_X_desc,
            &rs_size));
        __state->{unique_id}_reserved_size = new size_t;
        *__state->{unique_id}_reserved_size = rs_size;
        cudaMalloc(&__state->{unique_id}_reserved, rs_size);
        """
        finalize_code += f"""
        cudaFree(__state->{unique_id}_workspace);
        cudaFree(__state->{unique_id}_reserved);
        delete __state->{unique_id}_reserved_size;
        delete __state->{unique_id}_workspace_size;
        """

        tasklet_code = f"""
        {environments.cuDNN.handle_setup_code(node)}
        float alpha = 1.f;
        float beta = 0.f;
        dace::cudnn::CheckCudnnError(cudnnBatchNormalizationForwardTrainingEx(
            __dace_cudnn_handle,
            CUDNN_BATCHNORM_SPATIAL,
            CUDNN_BATCHNORM_OPS_BN,
            &alpha,
            &beta,
            *__state->{unique_id}_X_desc,
            _X,
            *__state->{unique_id}_X_desc,
            nullptr,
            *__state->{unique_id}_Y_desc,
            _Y,
            *__state->{unique_id}_scale_desc,
            _scale,
            _B,
            {1 - node.momentum},
            _in_mean,
            _in_var,
            {node.epsilon},
            _saved_mean,
            _saved_var,
            nullptr,
            __state->{unique_id}_workspace,
            *__state->{unique_id}_workspace_size,
            __state->{unique_id}_reserved,
            *__state->{unique_id}_reserved_size));

            // save the reserved ptr as an output if required
            {f"_reserved_ptr = __state->{unique_id}_reserved;" if reserved_ptr else ""}
            {f"_reserved_size = *__state->{unique_id}_reserved_size;" if reserved_ptr else ""}
        """

        in_connectors = ["X", "B", "scale", "in_mean", "in_var"]
        out_connectors = {
            "Y": dace.pointer(T),
            "saved_mean": dace.pointer(T),
            "saved_var": dace.pointer(T)
        }
        if reserved_ptr:
            out_connectors["reserved_size"] = dace.int64
            out_connectors["reserved_ptr"] = dace.pointer(dace.typeclass(None))
            nsdfg.add_scalar(f"reserved_ptr",
                             dace.pointer(dace.typeclass(None)),
                             storage=dtypes.StorageType.CPU_Heap,
                             transient=True)
            nsdfg.add_scalar(f"reserved_size",
                             dace.int64,
                             storage=dtypes.StorageType.CPU_Heap,
                             transient=True)
            outputs["reserved_ptr"] = nstate.add_write("reserved_ptr")
            outputs["reserved_size"] = nstate.add_write("reserved_size")

        init_code = "{\n" + init_code + "\n}"
        finalize_code = "{\n" + finalize_code + "\n}"

        tasklet = nstate.add_tasklet(
            unique_id, {f"_{i}": dace.pointer(T)
                        for i in in_connectors},
            {f"_{i}": t
             for i, t in out_connectors.items()},
            tasklet_code,
            dtypes.Language.CPP,
            code_init=init_code,
            code_exit=finalize_code,
            state_fields=[
                f"cudnnTensorDescriptor_t *{unique_id}_Y_desc;",
                f"cudnnTensorDescriptor_t *{unique_id}_X_desc;",
                f"cudnnTensorDescriptor_t *{unique_id}_scale_desc;",
                f"float *{unique_id}_workspace;",
                f"size_t *{unique_id}_workspace_size;"
                f"float *{unique_id}_reserved;",
                f"size_t *{unique_id}_reserved_size;"
            ])

        for inp in in_connectors:
            nstate.add_edge(inputs[inp], None, tasklet, f"_{inp}",
                            nsdfg.make_array_memlet(inp))

        for inp, anode in inputs.items():
            if f"_{inp}" not in tasklet.in_connectors:
                nstate.remove_node(anode)
                del node.in_connectors[inp]

        for outp in out_connectors:
            nstate.add_edge(tasklet, f"_{outp}", outputs[outp], None,
                            nsdfg.make_array_memlet(outp))

        # remove out_mean and out_var. We write these out to the same pointers as the inputs
        remove_output_connector(sdfg, state, node, "out_mean")
        remove_output_connector(sdfg, state, node, "out_var")
        del nsdfg.arrays["out_mean"]
        del nsdfg.arrays["out_var"]

        for outp, anode in outputs.items():
            if f"_{outp}" not in tasklet.out_connectors:
                nstate.remove_node(anode)
                del node.out_connectors[outp]

        return nsdfg
