import copy
import ctypes
import itertools
from typing import List, Optional, Tuple, Dict, Union

import numpy as np

import dace
from dace.frontend.common import einsum
import dace.libraries
from dace.registry import autoregister_params
from dace import nodes as nd, dtypes, subsets
import dace.transformation.transformation as xf

import dace.libraries.onnx as donnx
from dace.libraries.onnx.converters import clean_onnx_name
from dace.libraries.onnx.op_implementations import pure_implementations, \
    cudnn_implementations, CudnnBatchNormalizationTraining, setup_fake_data
import dace.autodiff.utils as butils
from dace.autodiff.base_abc import BackwardImplementation, BackwardContext, BackwardResult
from dace.transformation.onnx.replacement import onnx_constant_or_none
from dace.util import in_desc_with_name


def reverse_einsum_wrt_input(forward_node: donnx.ONNXEinsum,
                             input_name: str) -> Tuple[List[str], str]:
    """ Produce the einsum string that computes the grad of ``forward_node`` w.r.t. ``input_name``.

       :Note:
            There is an edge case we currently don't handle (can be implemented though). Something like ``'ii->i'``
            would become ``'i->ii'``. This is invalid because ``i`` is repeated in the output.

        :param forward_node: the einsum node to reverse.
        :param input_name: the connector on the forward node the produce the gradient computation for.
        :return: the list of forward node connectors required as inputs, and the einsum string. The first parameter of
                 the produced einsum string is implicitly the grad of ``Output``.
    """

    _, input_idx = donnx.parse_variadic_param(input_name)
    parser = einsum.EinsumParser(forward_node.equation)

    backward_input_expressions = [
        parser.output
    ] + parser.inputs[:input_idx] + parser.inputs[input_idx + 1:]
    backward_input_arrays = [
        f"Inputs__{i}" for i in itertools.chain(
            range(input_idx), range(input_idx + 1, len(parser.inputs)))
    ]

    einsum_str = f"{','.join(backward_input_expressions)}->{parser.inputs[input_idx]}"
    return backward_input_arrays, einsum_str


@autoregister_params(op="Einsum", name="default")
class DefaultEinsumBackward(BackwardImplementation):
    """ The symbolic autodiff can automatically derive matmuls, but the produced maps are more difficult to optimize.
    """
    @staticmethod
    def backward_can_be_applied(node: nd.Node, state: dace.SDFGState,
                                sdfg: dace.SDFG) -> bool:
        return pure_implementations.PureEinsum.forward_can_be_applied(
            node, state, sdfg)

    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[nd.Node, BackwardResult]:

        nsdfg = dace.SDFG(forward_node.label + "_backward")
        nstate = nsdfg.add_state()

        # setup arrays
        output_desc = butils.forward_out_desc_with_name(
            forward_node, context, "Output")
        result = BackwardResult.empty()
        result.given_grad_names["Output"] = butils.add_backward_desc(
            nsdfg, context.forward_sdfg, output_desc, "Output")
        access_output_grad = nstate.add_read(result.given_grad_names["Output"])

        def create_access_node(connector: str) -> nd.AccessNode:
            nsdfg.add_datadesc(
                connector,
                butils.forward_in_desc_with_name(forward_node, context,
                                                 connector))
            return nstate.add_read(connector)

        # the forward inputs we will require
        # maps the connector name to the accessnode
        required_forward_inputs: Dict[str, nd.AccessNode] = {}

        for input_name in sorted(required_gradients):
            # we add an einsum for each required gradient
            forward_inputs, einsum_str = reverse_einsum_wrt_input(
                forward_node, input_name)

            einsum_node = donnx.ONNXEinsum(input_name + "_backward",
                                           equation=einsum_str)
            nstate.add_node(einsum_node)

            # the first input is always the output grad
            einsum_node.add_in_connector(f"Inputs__0")
            nstate.add_edge(
                access_output_grad, None, einsum_node, "Inputs__0",
                nsdfg.make_array_memlet(result.given_grad_names["Output"]))

            # add the other inputs from forward that we need
            for i, forward_input in enumerate(sorted(forward_inputs)):
                connector = f"Inputs__{i + 1}"
                einsum_node.add_in_connector(connector)
                if forward_input not in required_forward_inputs:
                    required_forward_inputs[
                        forward_input] = create_access_node(forward_input)

                nstate.add_edge(required_forward_inputs[forward_input], None,
                                einsum_node, connector,
                                nsdfg.make_array_memlet(forward_input))

            # write out the gradient
            butils.forward_in_desc_with_name(forward_node, context, input_name)
            result.required_grad_names[
                input_name] = butils.add_backward_desc_for_connector(
                    nsdfg, forward_node, context, input_name, True)
            memlet = nsdfg.make_array_memlet(
                result.required_grad_names[input_name])
            nstate.add_edge(
                einsum_node, "Output",
                nstate.add_write(result.required_grad_names[input_name]), None,
                memlet)

        result_node = context.backward_state.add_nested_sdfg(
            nsdfg, None,
            set(result.given_grad_names.values()).union(
                required_forward_inputs),
            set(result.required_grad_names.values()))

        return result_node, result


@autoregister_params(op="Clip", name="default")
class DefaultClipBackward(BackwardImplementation):
    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[Union[nd.Node, dace.SDFG], BackwardResult]:

        result_node, result = butils.add_empty_sdfg_for_node(
            forward_node, ["input_grad", "output_grad", "input"], context)

        nstate = result_node.sdfg.add_state()

        min_node = next(
            context.forward_state.in_edges_by_connector(forward_node,
                                                        'min')).src
        max_node = next(
            context.forward_state.in_edges_by_connector(forward_node,
                                                        'max')).src
        minval = onnx_constant_or_none(context.forward_sdfg, min_node)
        maxval = onnx_constant_or_none(context.forward_sdfg, max_node)

        idesc = butils.forward_in_desc_with_name(forward_node, context,
                                                 "input")
        shape = idesc.shape
        map_ranges = {f"i{i}": f"0:{s}" for i, s in enumerate(shape)}

        input_dtype = idesc.dtype
        minstr = f"dace.{input_dtype.to_string()}({minval})"
        maxstr = f"dace.{input_dtype.to_string()}({maxval})"

        index_str = f"{', '.join(map_ranges.keys())}"
        code = f"""
if __input < {minstr} or __input > {maxstr}:
    __input_grad = 0
else:
    __input_grad = __output_grad
                """
        nstate.add_mapped_tasklet(
            forward_node.label + "_backward",
            map_ranges=map_ranges,
            inputs={
                f"__output_grad": dace.Memlet(f"output_grad[{index_str}]"),
                f"__input": dace.Memlet(f"input[{index_str}]"),
            },
            code=code,
            outputs={f"__input_grad": dace.Memlet(f"input_grad[{index_str}]")},
            external_edges=True)

        return result_node, result


@autoregister_params(op="Dropout", name="default")
class DefaultDropoutBackward(BackwardImplementation):
    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[Union[nd.Node, dace.SDFG], BackwardResult]:

        # one day maybe:
        # def dropout_backward(data_grad, output_grad, mask, ratio):
        #     scale = dace.float64(ratio) / (1 - dace.float64(ratio))
        #     data_grad[:] = scale * output_grad * mask

        result_node, result = butils.add_empty_sdfg_for_node(
            forward_node, ["data_grad", "output_grad", "mask", "ratio"],
            context)

        nstate = result_node.sdfg.add_state()

        shape = butils.forward_in_desc_with_name(forward_node, context,
                                                 "data").shape
        map_ranges = {f"i{i}": f"0:{s}" for i, s in enumerate(shape)}
        index_str = f"{', '.join(map_ranges.keys())}"
        code = f"""
scale = dace.float32(1.0) / (1 - __ratio)
__data_grad = __output_grad * __mask * scale
                """
        nstate.add_mapped_tasklet(
            forward_node.label + "_backward",
            map_ranges=map_ranges,
            inputs={
                "__output_grad": dace.Memlet(f"output_grad[{index_str}]"),
                "__mask": dace.Memlet(f"mask[{index_str}]"),
                "__ratio": dace.Memlet("ratio[0]")
            },
            code=code,
            outputs={f"__data_grad": dace.Memlet(f"data_grad[{index_str}]")},
            external_edges=True)

        return result_node, result


@autoregister_params(op="Softmax", name="default")
class DefaultSoftmaxBackward(BackwardImplementation):
    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[Union[nd.Node, dace.SDFG], BackwardResult]:

        dim = forward_node.axis

        output_shape = butils.forward_out_desc_with_name(
            forward_node, context, "output").shape
        output_dtype = butils.forward_out_desc_with_name(
            forward_node, context, "output").dtype

        sums_shape = list(copy.deepcopy(output_shape))
        sums_shape[dim] = 1

        def softmax_backward(output, output_grad, input_grad):
            prod = dace.define_local(output_shape, output_dtype)
            sums = dace.define_local(sums_shape, output_dtype)
            donnx.ONNXMul(A=output, B=output_grad, C=prod)
            donnx.ONNXReduceSum(data=prod,
                                reduced=sums,
                                keepdims=1,
                                axes=[dim])

            donnx.ONNXMul(A=output, B=sums, C=input_grad)
            # let's not use ONNXSub here; not sure how this inplace op is handled by ORT...
            input_grad[:] = prod - input_grad

        result_node, result = butils.backward_program_for_node(
            softmax_backward, context, forward_node)

        butils.connect_output_from_forward(forward_node, result_node, context,
                                           "output")

        return result_node, result


def _find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


@autoregister_params(op="MaxPool", name="default")
class DefaultMaxPoolBackward(BackwardImplementation):
    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[Union[nd.Node, dace.SDFG], BackwardResult]:

        output_shape = butils.forward_out_desc_with_name(
            forward_node, context, "Y").shape

        N, C, H, W = output_shape
        sty, stx = forward_node.strides
        sy, sx = forward_node.kernel_shape

        def maxpool_backward(X, Y_grad, X_grad):
            for b, c, ti, tj in dace.map[0:N, 0:C, 0:H, 0:W]:
                maxv = np.empty([1], dtype=dace.float32)
                maxi = np.empty([1], dtype=dace.int32)
                maxj = np.empty([1], dtype=dace.int32)
                with dace.tasklet:
                    v >> maxv
                    v = -9999999

                # Deterministic argmax (assuming sequential map)
                for i, j in dace.map[0:sy, 0:sx]:
                    with dace.tasklet:
                        o << X[b, c, sty * ti + i, stx * tj + j]
                        vin << maxv
                        v >> maxv(-1)
                        ind_i >> maxi(-1)
                        ind_j >> maxj(-1)
                        if o > vin:
                            v = o
                            ind_i = i
                            ind_j = j
                with dace.tasklet:
                    igrad << Y_grad[b, c, ti, tj]
                    ind_i << maxi
                    ind_j << maxj
                    ograd >> X_grad(1)[b, c, :, :]
                    ograd[ind_i, ind_j] = igrad

        result_node, result = butils.backward_program_for_node(
            maxpool_backward, context, forward_node)

        _find_map_by_param(result_node.sdfg, 'i').schedule = \
            dace.ScheduleType.Sequential

        return result_node, result


@autoregister_params(op="LogSoftmax", name="default")
class DefaultLogSoftmaxBackward(BackwardImplementation):
    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[nd.Node, BackwardResult]:

        dim = forward_node.axis
        output_shape = butils.forward_out_desc_with_name(
            forward_node, context, "output").shape
        output_dtype = butils.forward_out_desc_with_name(
            forward_node, context, "output").dtype

        sums_shape = list(copy.deepcopy(output_shape))
        sums_shape[dim] = 1

        def logsoftmax_backward(output, output_grad, input_grad):
            exp_output = dace.define_local(output_shape, output_dtype)
            donnx.ONNXExp(input=output, output=exp_output)

            grad_output_sum = dace.define_local(sums_shape, output_dtype)
            donnx.ONNXReduceSum(data=output_grad,
                                reduced=grad_output_sum,
                                keepdims=1,
                                axes=[dim])
            # let's not use ONNXMul here; not sure how this inplace op is handled by ORT...
            exp_output[:] = exp_output * grad_output_sum
            donnx.ONNXSub(A=output_grad, B=exp_output, C=input_grad)

        result_node, result = butils.backward_program_for_node(
            logsoftmax_backward, context, forward_node)

        butils.connect_output_from_forward(forward_node, result_node, context,
                                           "output")
        return result_node, result


@autoregister_params(op="Conv", name="cuDNN")
class CuDNNConvBackward(BackwardImplementation):
    """ Conv backward using CUDNN.
        The algorithm implementations can be set using node._data_algorithm and node._filter_algorithm
        Available choices for data algorithm:
            "auto"
            "0"
            "1"
            "fft"
            "fft_tiling"
            "winograd"
            "winograd_nonfused"
        Available choices for filter algorithm:
            "auto"
            "0"
            "1"
            "fft"
            "fft_tiling"
            "3"
            "winograd_nonfused"
    """
    default_data_algorithm = "auto"
    default_filter_algorithm = "auto"

    @staticmethod
    def backward_can_be_applied(node: nd.Node, state: dace.SDFGState,
                                sdfg: dace.SDFG) -> bool:
        return cudnn_implementations.CudnnConvolution.forward_can_be_applied(
            node, state, sdfg)

    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[nd.Node, BackwardResult]:

        nsdfg = dace.SDFG(forward_node.label + "_backward")
        X_desc = butils.forward_in_desc_with_name(forward_node, context, "X")

        T = X_desc.dtype

        # setup gradient arrays
        result = BackwardResult.empty()
        required_grads = set(required_gradients)
        for r in sorted(required_grads):
            result.required_grad_names[
                r] = butils.add_backward_desc_for_connector(nsdfg,
                                                            forward_node,
                                                            context,
                                                            r,
                                                            input=True)
        result.given_grad_names["Y"] = butils.add_backward_desc_for_connector(
            nsdfg, forward_node, context, "Y", input=False)

        # setup non-gradient arrays
        required_forward_inputs = ["W", "X"]
        for i in sorted(required_forward_inputs):
            new_desc = copy.deepcopy(
                butils.forward_in_desc_with_name(forward_node, context, i))
            new_desc.transient = False
            nsdfg.add_datadesc(i, new_desc)

        # setup state
        nstate = nsdfg.add_state()
        unique_id = "{}_{}_{}_{}_bwd".format(
            clean_onnx_name(forward_node.name), context.forward_sdfg.sdfg_id,
            context.forward_sdfg.node_id(context.forward_state),
            context.forward_state.node_id(forward_node))

        init_code = ""
        finalize_code = ""

        #######################
        # add descriptor init code for gradients
        for r in sorted(required_grads):
            is_filter = r == "W"

            if r == "B":
                bias_desc = butils.forward_in_desc_with_name(
                    forward_node, context, "B")
                shape = [1, bias_desc.shape[0], 1, 1]
            else:
                shape = None

            init, exit = cudnn_implementations._cudnn_tensor_descriptor_code(
                nsdfg.arrays[result.required_grad_names[r]],
                f"{unique_id}_d{r}_desc",
                is_filter,
                shape=shape)
            init_code += init
            finalize_code += exit

        for r in sorted(required_forward_inputs):
            desc = butils.forward_in_desc_with_name(forward_node, context, r)
            is_filter = r == "W"
            init, exit = cudnn_implementations._cudnn_tensor_descriptor_code(
                desc, f"{unique_id}_{r}_desc", is_filter)
            init_code += init
            finalize_code += exit

        init, exit = cudnn_implementations._cudnn_tensor_descriptor_code(
            nsdfg.arrays[result.given_grad_names["Y"]], f"{unique_id}_dY_desc",
            False)
        init_code += init
        finalize_code += exit

        #######################
        # setup conv descriptor
        # we know from can_be_applied that the pads are symmetric
        if len(forward_node.strides) == 1:
            pad_h, pad_w = forward_node.pads[0], 0
            stride_h, stride_w = forward_node.strides[0], 1
            dilation_h, dilation_w = forward_node.dilations[0], 1
        else:
            pad_h, pad_w = forward_node.pads[0], forward_node.pads[1]
            stride_h, stride_w = forward_node.strides
            dilation_h, dilation_w = forward_node.dilations
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
            {cudnn_implementations._DACE_DTYPE_TO_CUDNN_DTYPE[T]}));
        """
        if forward_node.group != 1:
            init_code += f"""
            dace::cudnn::CheckCudnnError(cudnnSetConvolutionGroupCount(
                *__state->{unique_id}_conv_desc,
                {forward_node.group}
                ));
            """
        finalize_code += f"""
        dace::cudnn::CheckCudnnError(cudnnDestroyConvolutionDescriptor(*__state->{unique_id}_conv_desc));
        delete __state->{unique_id}_conv_desc;
        """

        #######################
        # setup algorithms

        if hasattr(forward_node, "_data_algorithm"):
            data_algo = forward_node._data_algorithm
        else:
            data_algo = CuDNNConvBackward.default_data_algorithm

        if hasattr(forward_node, "_filter_algorithm"):
            filter_algo = forward_node._filter_algorithm
        else:
            filter_algo = CuDNNConvBackward.default_filter_algorithm

        init_code += f"{donnx.environments.cuDNN.handle_setup_code(forward_node, init_stream=False)}"
        if data_algo == "auto" or filter_algo == "auto":
            # setup fake data
            free_fake_data_code, fake_data_init_code = setup_fake_data(
                forward_node, context.forward_sdfg, context.forward_state,
                True)

            # setup algo
            init_code += f"""
            // setup fake data
            {fake_data_init_code}

            // setup workspace
            void *search_ws; 
            cudaMalloc(&search_ws, {cudnn_implementations.CudnnConvolution.search_ws_size});
            """

        if filter_algo == "auto":
            init_code += f"""
            // run search
            cudnnConvolutionBwdFilterAlgoPerf_t filter_results;
            int filter_algo_count = 1;
            dace::cudnn::CheckCudnnError(cudnnFindConvolutionBackwardFilterAlgorithmEx(
                __dace_cudnn_handle,
                *__state->{unique_id}_X_desc,
                fake_X,
                *__state->{unique_id}_dY_desc,
                fake_dY,
                *__state->{unique_id}_conv_desc,
                *__state->{unique_id}_dW_desc,
                fake_dW,
                1,
                &filter_algo_count,
                &filter_results,
                search_ws,
                {cudnn_implementations.CudnnConvolution.search_ws_size}
            ));
            __state->{unique_id}_filter_algo = new cudnnConvolutionBwdFilterAlgo_t;
            *__state->{unique_id}_filter_algo = filter_results.algo;
            printf("{unique_id} using filter algo %d\\n", *__state->{unique_id}_filter_algo);
            """
        else:
            init_code += f"""
            __state->{unique_id}_filter_algo = new cudnnConvolutionBwdFilterAlgo_t;
            *__state->{unique_id}_filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_{filter_algo.upper()};
            """

        if data_algo == "auto":
            init_code += f"""
            // run search
            cudnnConvolutionBwdDataAlgoPerf_t data_results;
            int data_algo_count = 1;
            dace::cudnn::CheckCudnnError(cudnnFindConvolutionBackwardDataAlgorithmEx(
                __dace_cudnn_handle,
                *__state->{unique_id}_W_desc,
                fake_W,
                *__state->{unique_id}_dY_desc,
                fake_dY,
                *__state->{unique_id}_conv_desc,
                *__state->{unique_id}_dX_desc,
                fake_dX,
                1,
                &data_algo_count,
                &data_results,
                search_ws,
                {cudnn_implementations.CudnnConvolution.search_ws_size}
            ));
            __state->{unique_id}_data_algo = new cudnnConvolutionBwdDataAlgo_t;
            *__state->{unique_id}_data_algo = data_results.algo;
            printf("{unique_id} using data algo %d\\n", *__state->{unique_id}_data_algo);
            """
        else:
            init_code += f"""
            __state->{unique_id}_data_algo = new cudnnConvolutionBwdDataAlgo_t;
            *__state->{unique_id}_data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_{data_algo.upper()};
            """

        if data_algo == "auto" or filter_algo == "auto":
            init_code += f"""
            cudaFree(search_ws);
            {free_fake_data_code}
            """

        finalize_code += f"""
             delete __state->{unique_id}_data_algo;
             delete __state->{unique_id}_filter_algo;
        """

        #######################
        # setup workspace
        init_code += \
            f"""
        // Setup workspace for {unique_id}
        
        size_t data_ws_size;
        dace::cudnn::CheckCudnnError(cudnnGetConvolutionBackwardDataWorkspaceSize(
            __dace_cudnn_handle,
            *__state->{unique_id}_W_desc,
            *__state->{unique_id}_dY_desc,
            *__state->{unique_id}_conv_desc,
            *__state->{unique_id}_dX_desc,
            *__state->{unique_id}_data_algo,
            &data_ws_size));
        size_t filter_ws_size;
        dace::cudnn::CheckCudnnError(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            __dace_cudnn_handle,
            *__state->{unique_id}_X_desc,
            *__state->{unique_id}_dY_desc,
            *__state->{unique_id}_conv_desc,
            *__state->{unique_id}_dW_desc,
            *__state->{unique_id}_filter_algo,
            &filter_ws_size));
        
        size_t ws_size = max(filter_ws_size, data_ws_size);
        __state->{unique_id}_workspace_size = new size_t;
        *__state->{unique_id}_workspace_size = ws_size;
        cudaMalloc(&__state->{unique_id}_workspace, ws_size);
        """
        finalize_code += f"""
        cudaFree(__state->{unique_id}_workspace);
        delete __state->{unique_id}_workspace_size;
        """

        #######################
        # tasklet code

        tasklet_code = f"""
        {donnx.environments.cuDNN.handle_setup_code(forward_node)}
        float alpha = 1.f;
        float beta = 0.f;
        dace::cudnn::CheckCudnnError(cudnnConvolutionBackwardData(
            __dace_cudnn_handle,
            &alpha,
            *__state->{unique_id}_W_desc,
            _W,
            *__state->{unique_id}_dY_desc,
            _dY,
            *__state->{unique_id}_conv_desc,
            *__state->{unique_id}_data_algo,
            __state->{unique_id}_workspace,
            *__state->{unique_id}_workspace_size,
            &beta,
            *__state->{unique_id}_dX_desc,
            _dX));
        dace::cudnn::CheckCudnnError(cudnnConvolutionBackwardFilter(
            __dace_cudnn_handle,
            &alpha,
            *__state->{unique_id}_X_desc,
            _X,
            *__state->{unique_id}_dY_desc,
            _dY,
            *__state->{unique_id}_conv_desc,
            *__state->{unique_id}_filter_algo,
            __state->{unique_id}_workspace,
            *__state->{unique_id}_workspace_size,
            &beta,
            *__state->{unique_id}_dW_desc,
            _dW));
        """

        if "B" in required_gradients:
            tasklet_code += f"""
            dace::cudnn::CheckCudnnError(cudnnConvolutionBackwardBias(
                __dace_cudnn_handle,
                &alpha,
                *__state->{unique_id}_dY_desc,
                _dY,
                &beta,
                *__state->{unique_id}_dB_desc,
                _dB));
            """

        init_code = "{\n" + init_code + "\n}"
        finalize_code = "{\n" + finalize_code + "\n}"
        tasklet = nstate.add_tasklet(
            unique_id, {
                f"_{i}": dace.pointer(T)
                for i in itertools.chain(["dY"], sorted(
                    required_forward_inputs))
            }, {
                f"_d{i}": dace.pointer(T)
                for i in itertools.chain(sorted(required_gradients))
            },
            tasklet_code,
            dace.dtypes.Language.CPP,
            state_fields=[
                f"cudnnTensorDescriptor_t *{unique_id}_X_desc;",
                f"cudnnFilterDescriptor_t *{unique_id}_W_desc;",
                f"cudnnTensorDescriptor_t *{unique_id}_dX_desc;",
                f"cudnnTensorDescriptor_t *{unique_id}_dY_desc;",
                f"cudnnTensorDescriptor_t *{unique_id}_dB_desc;",
                f"cudnnFilterDescriptor_t *{unique_id}_dW_desc;"
                f"cudnnConvolutionBwdDataAlgo_t *{unique_id}_data_algo;"
                f"cudnnConvolutionBwdFilterAlgo_t *{unique_id}_filter_algo;"
                f"cudnnConvolutionDescriptor_t *{unique_id}_conv_desc;",
                f"float *{unique_id}_workspace;",
                f"size_t *{unique_id}_workspace_size;"
            ],
            code_init=init_code,
            code_exit=finalize_code)
        tasklet.environments = {donnx.environments.cuDNN.full_class_path()}

        nstate.add_edge(
            nstate.add_read(result.given_grad_names["Y"]), None, tasklet,
            f"_dY", nsdfg.make_array_memlet((result.given_grad_names["Y"])))
        for name in sorted(required_forward_inputs):
            nstate.add_edge(nstate.add_read(name), None, tasklet, f"_{name}",
                            nsdfg.make_array_memlet(name))

        for name in sorted(required_gradients):
            arr_name = result.required_grad_names[name]
            nstate.add_edge(tasklet, f"_d{name}", nstate.add_write(arr_name),
                            None, nsdfg.make_array_memlet(arr_name))

        inputs = {result.given_grad_names["Y"]}.union(required_forward_inputs)
        outputs = {
            result.required_grad_names[n]
            for n in sorted(required_gradients)
        }
        node = context.backward_state.add_nested_sdfg(nsdfg, None, inputs,
                                                      outputs)

        return node, result


@autoregister_params(op="ConvTranspose", name="cuDNN")
class CuDNNConvTransposeBackward(BackwardImplementation):
    """ ConvTranspose backward using CUDNN.
        The algorithm implementations can be set using node._data_algorithm and node._filter_algorithm
        Available choices for data algorithm (same as Conv forward):
            "auto"
            "0"
            "1"
            "fft"
            "fft_tiling"
            "winograd"
            "winograd_nonfused"
        Available choices for filter algorithm:
            "auto"
            "0"
            "1"
            "fft"
            "fft_tiling"
            "3"
            "winograd_nonfused"
    """
    default_data_algorithm = "auto"
    default_filter_algorithm = "auto"

    @staticmethod
    def backward_can_be_applied(node: nd.Node, state: dace.SDFGState,
                                sdfg: dace.SDFG) -> bool:
        return cudnn_implementations.CudnnConvolution.forward_can_be_applied(
            node, state, sdfg)

    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[nd.Node, BackwardResult]:

        nsdfg = dace.SDFG(forward_node.label + "_backward")
        X_desc = butils.forward_in_desc_with_name(forward_node, context, "X")

        T = X_desc.dtype

        # setup gradient arrays
        result = BackwardResult.empty()
        required_grads = set(required_gradients)
        for r in sorted(required_grads):
            result.required_grad_names[
                r] = butils.add_backward_desc_for_connector(nsdfg,
                                                            forward_node,
                                                            context,
                                                            r,
                                                            input=True)
        result.given_grad_names["Y"] = butils.add_backward_desc_for_connector(
            nsdfg, forward_node, context, "Y", input=False)

        # setup non-gradient arrays
        required_forward_inputs = ["W", "X"]
        for i in sorted(required_forward_inputs):
            new_desc = copy.deepcopy(
                butils.forward_in_desc_with_name(forward_node, context, i))
            new_desc.transient = False
            nsdfg.add_datadesc(i, new_desc)

        # setup state
        nstate = nsdfg.add_state()
        unique_id = "{}_{}_{}_{}_bwd".format(
            clean_onnx_name(forward_node.name), context.forward_sdfg.sdfg_id,
            context.forward_sdfg.node_id(context.forward_state),
            context.forward_state.node_id(forward_node))

        init_code = ""
        finalize_code = ""

        #######################
        # add descriptor init code for gradients
        for r in sorted(required_grads):
            is_filter = r == "W"

            if r == "B":
                bias_desc = butils.forward_in_desc_with_name(
                    forward_node, context, "B")
                shape = [1, bias_desc.shape[0], 1, 1]
            else:
                shape = None

            init, exit = cudnn_implementations._cudnn_tensor_descriptor_code(
                nsdfg.arrays[result.required_grad_names[r]],
                f"{unique_id}_d{r}_desc",
                is_filter,
                shape=shape)
            init_code += init
            finalize_code += exit

        for r in sorted(required_forward_inputs):
            desc = butils.forward_in_desc_with_name(forward_node, context, r)
            is_filter = r == "W"
            init, exit = cudnn_implementations._cudnn_tensor_descriptor_code(
                desc, f"{unique_id}_{r}_desc", is_filter)
            init_code += init
            finalize_code += exit

        init, exit = cudnn_implementations._cudnn_tensor_descriptor_code(
            nsdfg.arrays[result.given_grad_names["Y"]], f"{unique_id}_dY_desc",
            False)
        init_code += init
        finalize_code += exit

        #######################
        # setup conv descriptor
        # we know from can_be_applied that the pads are symmetric
        if len(forward_node.strides) == 1:
            pad_h, pad_w = forward_node.pads[0], 0
            stride_h, stride_w = forward_node.strides[0], 1
            dilation_h, dilation_w = forward_node.dilations[0], 1
        else:
            pad_h, pad_w = forward_node.pads[0], forward_node.pads[1]
            stride_h, stride_w = forward_node.strides
            dilation_h, dilation_w = forward_node.dilations
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
            {cudnn_implementations._DACE_DTYPE_TO_CUDNN_DTYPE[T]}));
        """
        if forward_node.group != 1:
            init_code += f"""
            dace::cudnn::CheckCudnnError(cudnnSetConvolutionGroupCount(
                *__state->{unique_id}_conv_desc,
                {forward_node.group}
                ));
            """
        finalize_code += f"""
        dace::cudnn::CheckCudnnError(cudnnDestroyConvolutionDescriptor(*__state->{unique_id}_conv_desc));
        delete __state->{unique_id}_conv_desc;
        """

        #######################
        # setup algorithms

        if hasattr(forward_node, "_data_algorithm"):
            data_algo = forward_node._data_algorithm
        else:
            data_algo = CuDNNConvTransposeBackward.default_data_algorithm

        if hasattr(forward_node, "_filter_algorithm"):
            filter_algo = forward_node._filter_algorithm
        else:
            filter_algo = CuDNNConvTransposeBackward.default_filter_algorithm

        init_code += f"{donnx.environments.cuDNN.handle_setup_code(forward_node, init_stream=False)}"
        if data_algo == "auto" or filter_algo == "auto":
            # setup fake data
            free_fake_data_code, fake_data_init_code = setup_fake_data(
                forward_node, context.forward_sdfg, context.forward_state,
                True)

            # setup algo
            init_code += f"""
            // setup fake data
            {fake_data_init_code}

            // setup workspace
            void *search_ws; 
            cudaMalloc(&search_ws, {cudnn_implementations.CudnnConvolution.search_ws_size});
            """

        if filter_algo == "auto":
            init_code += f"""
            // run search
            cudnnConvolutionBwdFilterAlgoPerf_t filter_results;
            int filter_algo_count = 1;
            dace::cudnn::CheckCudnnError(cudnnFindConvolutionBackwardFilterAlgorithmEx(
                __dace_cudnn_handle,
                *__state->{unique_id}_dY_desc,
                fake_dY,
                *__state->{unique_id}_X_desc,
                fake_X,
                *__state->{unique_id}_conv_desc,
                *__state->{unique_id}_dW_desc,
                fake_dW,
                1,
                &filter_algo_count,
                &filter_results,
                search_ws,
                {cudnn_implementations.CudnnConvolution.search_ws_size}
            ));
            __state->{unique_id}_filter_algo = new cudnnConvolutionBwdFilterAlgo_t;
            *__state->{unique_id}_filter_algo = filter_results.algo;
            printf("{unique_id} using filter algo %d\\n", *__state->{unique_id}_filter_algo);
            """
        else:
            init_code += f"""
            __state->{unique_id}_filter_algo = new cudnnConvolutionBwdFilterAlgo_t;
            *__state->{unique_id}_filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_{filter_algo.upper()};
            """

        if data_algo == "auto":
            init_code += f"""
            // run search
            cudnnConvolutionFwdAlgoPerf_t data_results;
            int data_algo_count = 1;
            dace::cudnn::CheckCudnnError(cudnnFindConvolutionForwardAlgorithmEx(
                __dace_cudnn_handle,
                *__state->{unique_id}_dY_desc,
                fake_dY,
                *__state->{unique_id}_W_desc,
                fake_W,
                *__state->{unique_id}_conv_desc,
                *__state->{unique_id}_dX_desc,
                fake_dX,
                1,
                &data_algo_count,
                &data_results,
                search_ws,
                {cudnn_implementations.CudnnConvolution.search_ws_size}
            ));
            __state->{unique_id}_data_algo = new cudnnConvolutionFwdAlgo_t;
            *__state->{unique_id}_data_algo = data_results.algo;
            printf("{unique_id} using data algo %d\\n", *__state->{unique_id}_data_algo);
            """
        else:
            init_code += f"""
            __state->{unique_id}_data_algo = new cudnnConvolutionFwdAlgo_t;
            *__state->{unique_id}_data_algo = CUDNN_CONVOLUTION_FWD_ALGO_{data_algo.upper()};
            """

        if data_algo == "auto" or filter_algo == "auto":
            init_code += f"""
            cudaFree(search_ws);
            {free_fake_data_code}
            """

        finalize_code += f"""
             delete __state->{unique_id}_data_algo;
             delete __state->{unique_id}_filter_algo;
        """

        #######################
        # setup workspace
        init_code += \
            f"""
        // Setup workspace for {unique_id}
        
        size_t data_ws_size;
        dace::cudnn::CheckCudnnError(cudnnGetConvolutionForwardWorkspaceSize(
            __dace_cudnn_handle,
            *__state->{unique_id}_dY_desc,
            *__state->{unique_id}_W_desc,
            *__state->{unique_id}_conv_desc,
            *__state->{unique_id}_dX_desc,
            *__state->{unique_id}_data_algo,
            &data_ws_size));
        size_t filter_ws_size;
        dace::cudnn::CheckCudnnError(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            __dace_cudnn_handle,
            *__state->{unique_id}_dY_desc,
            *__state->{unique_id}_X_desc,
            *__state->{unique_id}_conv_desc,
            *__state->{unique_id}_dW_desc,
            *__state->{unique_id}_filter_algo,
            &filter_ws_size));
        
        size_t ws_size = max(filter_ws_size, data_ws_size);
        __state->{unique_id}_workspace_size = new size_t;
        *__state->{unique_id}_workspace_size = ws_size;
        cudaMalloc(&__state->{unique_id}_workspace, ws_size);
        """
        finalize_code += f"""
        cudaFree(__state->{unique_id}_workspace);
        delete __state->{unique_id}_workspace_size;
        """

        #######################
        # tasklet code

        tasklet_code = f"""
        {donnx.environments.cuDNN.handle_setup_code(forward_node)}
        float alpha = 1.f;
        float beta = 0.f;
        dace::cudnn::CheckCudnnError(cudnnConvolutionForward(
            __dace_cudnn_handle,
            &alpha,
            *__state->{unique_id}_dY_desc,
            _dY,
            *__state->{unique_id}_W_desc,
            _W,
            *__state->{unique_id}_conv_desc,
            *__state->{unique_id}_data_algo,
            __state->{unique_id}_workspace,
            *__state->{unique_id}_workspace_size,
            &beta,
            *__state->{unique_id}_dX_desc,
            _dX));
        dace::cudnn::CheckCudnnError(cudnnConvolutionBackwardFilter(
            __dace_cudnn_handle,
            &alpha,
            *__state->{unique_id}_dY_desc,
            _dY,
            *__state->{unique_id}_X_desc,
            _X,
            *__state->{unique_id}_conv_desc,
            *__state->{unique_id}_filter_algo,
            __state->{unique_id}_workspace,
            *__state->{unique_id}_workspace_size,
            &beta,
            *__state->{unique_id}_dW_desc,
            _dW));
        """

        if "B" in required_gradients:
            tasklet_code += f"""
            dace::cudnn::CheckCudnnError(cudnnConvolutionBackwardBias(
                __dace_cudnn_handle,
                &alpha,
                *__state->{unique_id}_dY_desc,
                _dY,
                &beta,
                *__state->{unique_id}_dB_desc,
                _dB));
            """

        init_code = "{\n" + init_code + "\n}"
        finalize_code = "{\n" + finalize_code + "\n}"
        tasklet = nstate.add_tasklet(
            unique_id, {
                f"_{i}": dace.pointer(T)
                for i in itertools.chain(["dY"], sorted(
                    required_forward_inputs))
            }, {
                f"_d{i}": dace.pointer(T)
                for i in itertools.chain(sorted(required_gradients))
            },
            tasklet_code,
            dace.dtypes.Language.CPP,
            state_fields=[
                f"cudnnTensorDescriptor_t *{unique_id}_X_desc;",
                f"cudnnFilterDescriptor_t *{unique_id}_W_desc;",
                f"cudnnTensorDescriptor_t *{unique_id}_dX_desc;",
                f"cudnnTensorDescriptor_t *{unique_id}_dY_desc;",
                f"cudnnTensorDescriptor_t *{unique_id}_dB_desc;",
                f"cudnnFilterDescriptor_t *{unique_id}_dW_desc;"
                f"cudnnConvolutionFwdAlgo_t *{unique_id}_data_algo;"
                f"cudnnConvolutionBwdFilterAlgo_t *{unique_id}_filter_algo;"
                f"cudnnConvolutionDescriptor_t *{unique_id}_conv_desc;",
                f"float *{unique_id}_workspace;",
                f"size_t *{unique_id}_workspace_size;"
            ],
            code_init=init_code,
            code_exit=finalize_code)
        tasklet.environments = {donnx.environments.cuDNN.full_class_path()}

        nstate.add_edge(
            nstate.add_read(result.given_grad_names["Y"]), None, tasklet,
            f"_dY", nsdfg.make_array_memlet((result.given_grad_names["Y"])))
        for name in sorted(required_forward_inputs):
            nstate.add_edge(nstate.add_read(name), None, tasklet, f"_{name}",
                            nsdfg.make_array_memlet(name))

        for name in sorted(required_gradients):
            arr_name = result.required_grad_names[name]
            nstate.add_edge(tasklet, f"_d{name}", nstate.add_write(arr_name),
                            None, nsdfg.make_array_memlet(arr_name))

        inputs = {result.given_grad_names["Y"]}.union(required_forward_inputs)
        outputs = {
            result.required_grad_names[n]
            for n in sorted(required_gradients)
        }
        node = context.backward_state.add_nested_sdfg(nsdfg, None, inputs,
                                                      outputs)

        return node, result


@autoregister_params(op="Conv", name="PyTorch-dwise")
class PyTorchConvBackward(BackwardImplementation):
    """ Conv backward using PyTorch.
    """
    @staticmethod
    def backward_can_be_applied(node: nd.Node, state: dace.SDFGState,
                                sdfg: dace.SDFG) -> bool:
        X_desc = in_desc_with_name(node, state, sdfg, "X")
        return len(X_desc.shape) == 4

    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[nd.Node, BackwardResult]:

        nsdfg = dace.SDFG(forward_node.label + "_backward")
        X_desc = butils.forward_in_desc_with_name(forward_node, context, "X")
        W_desc = butils.forward_in_desc_with_name(forward_node, context, "W")

        T = X_desc.dtype
        if str(T) == 'float':
            pytorch_dtype = 'kFloat'
        elif str(T) == 'double':
            pytorch_dtype = 'kDouble'
        else:
            raise NotImplementedError(
                f"Pytorch backward conv expansion supports only float and double tensors, got {str(T)}"
            )

        # setup gradient arrays
        result = BackwardResult.empty()
        required_grads = set(required_gradients)
        for r in sorted(required_grads):
            result.required_grad_names[
                r] = butils.add_backward_desc_for_connector(nsdfg,
                                                            forward_node,
                                                            context,
                                                            r,
                                                            input=True)
        result.given_grad_names["Y"] = butils.add_backward_desc_for_connector(
            nsdfg, forward_node, context, "Y", input=False)

        # setup non-gradient arrays
        required_forward_inputs = ["W", "X"]
        for i in sorted(required_forward_inputs):
            new_desc = copy.deepcopy(
                butils.forward_in_desc_with_name(forward_node, context, i))
            new_desc.transient = False
            nsdfg.add_datadesc(i, new_desc)

        # setup state
        nstate = nsdfg.add_state()
        unique_id = "{}_{}_{}_{}_bwd".format(
            clean_onnx_name(forward_node.name), context.forward_sdfg.sdfg_id,
            context.forward_sdfg.node_id(context.forward_state),
            context.forward_state.node_id(forward_node))

        init_code = ""
        finalize_code = ""
        code_global = """
            #include <ATen/Tensor.h>
            #include <ATen/Functions.h>
        """
        tasklet_inputs = {
            f"_{i}": dace.pointer(T)
            for i in itertools.chain(["dY"], sorted(required_forward_inputs))
        }
        tasklet_outputs = {
            f"_d{i}": dace.pointer(T)
            for i in itertools.chain(sorted(required_gradients))
        }

        tasklet_code = f"""
            std::vector<int64_t> x_shape = {{ {", ".join(map(str, X_desc.shape))} }};
            std::vector<int64_t> x_strides = {{ {", ".join(map(str, X_desc.strides))} }};
            std::vector<int64_t> w_shape = {{ {", ".join(map(str, W_desc.shape))} }};
            std::vector<int64_t> w_strides = {{ {", ".join(map(str, W_desc.strides))} }};
            at::Tensor x = at::from_blob(_X, x_shape, x_strides, [](void*){{}}, at::TensorOptions().device(at::kCUDA).dtype(at::{pytorch_dtype}).requires_grad(false));
            at::Tensor w = at::from_blob(_W, w_shape, w_strides, [](void*){{}}, at::TensorOptions().device(at::kCUDA).dtype(at::{pytorch_dtype}).requires_grad(false));
            at::Tensor dy = at::from_blob(_dY, x_shape, x_strides, [](void*){{}}, at::TensorOptions().device(at::kCUDA).dtype(at::{pytorch_dtype}).requires_grad(false));
            at::Tensor dw = at::from_blob(_dW, w_shape, w_strides, [](void*){{}}, at::TensorOptions().device(at::kCUDA).dtype(at::{pytorch_dtype}).requires_grad(false));
            at::Tensor dx = at::from_blob(_dX, x_shape, x_strides, [](void*){{}}, at::TensorOptions().device(at::kCUDA).dtype(at::{pytorch_dtype}).requires_grad(false));
            
            std::vector<int64_t> kernel_shape = {{ {", ".join(map(str, forward_node.kernel_shape))} }};
            std::vector<int64_t> conv_strides = {{ {", ".join(map(str, forward_node.strides))} }};
            std::vector<int64_t> padding = {{ {", ".join(map(str, forward_node.pads[::2]))} }};
            std::vector<int64_t> dilation = {{ {", ".join(map(str, forward_node.dilations))} }};
            
            at::thnn_conv_depthwise2d_backward_out(dx, dw, dy, x, w, kernel_shape, conv_strides, padding, dilation);
        """

        tasklet = nstate.add_tasklet(name=unique_id,
                                     inputs=tasklet_inputs,
                                     outputs=tasklet_outputs,
                                     code=tasklet_code,
                                     language=dace.dtypes.Language.CPP,
                                     code_global=code_global,
                                     code_init=init_code,
                                     code_exit=finalize_code)
        tasklet.environments = {
            dace.libraries.torch.environments.PyTorch.full_class_path()
        }

        nstate.add_edge(
            nstate.add_read(result.given_grad_names["Y"]), None, tasklet,
            f"_dY", nsdfg.make_array_memlet((result.given_grad_names["Y"])))
        for name in sorted(required_forward_inputs):
            nstate.add_edge(nstate.add_read(name), None, tasklet, f"_{name}",
                            nsdfg.make_array_memlet(name))

        for name in sorted(required_gradients):
            arr_name = result.required_grad_names[name]
            nstate.add_edge(tasklet, f"_d{name}", nstate.add_write(arr_name),
                            None, nsdfg.make_array_memlet(arr_name))

        inputs = {result.given_grad_names["Y"]}.union(required_forward_inputs)
        outputs = {
            result.required_grad_names[n]
            for n in sorted(required_gradients)
        }
        node = context.backward_state.add_nested_sdfg(nsdfg, None, inputs,
                                                      outputs)

        return node, result


@autoregister_params(op="BatchNormalization", name="cuDNN")
class CuDNNBatchNormBackward(BackwardImplementation):
    @staticmethod
    def backward_can_be_applied(node: nd.Node, state: dace.SDFGState,
                                sdfg: dace.SDFG) -> bool:
        return cudnn_implementations.CudnnBatchNormalizationTraining.forward_can_be_applied(
            node, state, sdfg)

    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[nd.Node, BackwardResult]:

        nsdfg = dace.SDFG(forward_node.label + "_backward")
        X_desc = butils.forward_in_desc_with_name(forward_node, context, "X")
        scale_desc = butils.forward_in_desc_with_name(forward_node, context,
                                                      "scale")
        T = X_desc.dtype

        # setup arrays
        result = BackwardResult.empty()
        result.required_grad_names["X"] = butils.add_backward_desc(
            nsdfg, context.forward_sdfg, X_desc, "X")
        result.given_grad_names["Y"] = butils.add_backward_desc_for_connector(
            nsdfg, forward_node, context, "Y", input=False)
        result.required_grad_names[
            "scale"] = butils.add_backward_desc_for_connector(nsdfg,
                                                              forward_node,
                                                              context,
                                                              "scale",
                                                              input=True)
        result.required_grad_names[
            "B"] = butils.add_backward_desc_for_connector(nsdfg,
                                                          forward_node,
                                                          context,
                                                          "B",
                                                          input=True)

        # input X
        new_X_desc = copy.deepcopy(X_desc)
        new_X_desc.transient = False
        nsdfg.add_datadesc("X", new_X_desc)

        # input scale
        new_scale_desc = copy.deepcopy(scale_desc)
        new_scale_desc.transient = False
        nsdfg.add_datadesc("scale", new_scale_desc)

        # saved vars
        for saved in ["saved_mean", "saved_var"]:
            saved_desc = copy.deepcopy(
                butils.forward_out_desc_with_name(forward_node, context,
                                                  saved))
            saved_desc.transient = False
            nsdfg.add_datadesc(saved, saved_desc)

        # setup state
        nstate = nsdfg.add_state()
        fwd_unique_id = "{}_{}_{}_{}".format(
            clean_onnx_name(forward_node.name), context.forward_sdfg.sdfg_id,
            context.forward_sdfg.node_id(context.forward_state),
            context.forward_state.node_id(forward_node))

        unique_id = f"{fwd_unique_id}_bwd"

        init, exit = cudnn_implementations._cudnn_tensor_descriptor_code(
            new_X_desc, f"{unique_id}_X_desc", False)
        init_code = init
        finalize_code = exit

        dX_desc = nsdfg.arrays[result.required_grad_names["X"]]
        init, exit = cudnn_implementations._cudnn_tensor_descriptor_code(
            dX_desc, f"{unique_id}_dX_desc", False)
        init_code += init
        finalize_code += exit

        dY_desc = nsdfg.arrays[result.given_grad_names["Y"]]
        init, exit = cudnn_implementations._cudnn_tensor_descriptor_code(
            dY_desc, f"{unique_id}_dY_desc", False)
        init_code += init
        finalize_code += exit

        # setup scale descriptor
        init_code += f"""
        __state->{unique_id}_dScale_desc = new cudnnTensorDescriptor_t; 
        dace::cudnn::CheckCudnnError(cudnnCreateTensorDescriptor(__state->{unique_id}_dScale_desc));
        dace::cudnn::CheckCudnnError(cudnnDeriveBNTensorDescriptor(
            *__state->{unique_id}_dScale_desc,
            *__state->{unique_id}_X_desc,
            CUDNN_BATCHNORM_SPATIAL));
        """
        finalize_code += f"""
        dace::cudnn::CheckCudnnError(cudnnDestroyTensorDescriptor(*__state->{unique_id}_dScale_desc));
        delete __state->{unique_id}_dScale_desc;
        """

        # setup workspace
        init_code += \
            f"""
        {donnx.environments.cuDNN.handle_setup_code(forward_node, init_stream=False)}
        // Setup workspace and reserved space for {unique_id}
        size_t ws_size;
        dace::cudnn::CheckCudnnError(cudnnGetBatchNormalizationBackwardExWorkspaceSize(
            __dace_cudnn_handle,
            CUDNN_BATCHNORM_SPATIAL,
            CUDNN_BATCHNORM_OPS_BN,
            *__state->{unique_id}_X_desc,
            nullptr,
            *__state->{unique_id}_dY_desc,
            nullptr,
            *__state->{unique_id}_dX_desc,
            *__state->{unique_id}_dScale_desc,
            nullptr,
            &ws_size));
        __state->{unique_id}_workspace_size = new size_t;
        *__state->{unique_id}_workspace_size = ws_size;
        cudaMalloc(&__state->{unique_id}_workspace, ws_size);
        """
        finalize_code += f"""
        delete __state->{unique_id}_workspace_size;
        cudaFree(__state->{unique_id}_workspace);
        """

        tasklet_code = f"""
        {donnx.environments.cuDNN.handle_setup_code(forward_node)}
        float alpha = 1.f;
        float beta = 0.f;
        dace::cudnn::CheckCudnnError(cudnnBatchNormalizationBackwardEx(
            __dace_cudnn_handle,
            CUDNN_BATCHNORM_SPATIAL,
            CUDNN_BATCHNORM_OPS_BN,
            &alpha,
            &beta,
            &alpha,
            &beta,
            *__state->{unique_id}_X_desc,
            _X,
            nullptr,
            nullptr,
            *__state->{unique_id}_dY_desc,
            _dY,
            nullptr,
            nullptr,
            *__state->{unique_id}_dX_desc,
            _dX,
            *__state->{unique_id}_dScale_desc,
            _scale,
            nullptr,
            _dScale,
            _dBias,
            {forward_node.epsilon},
            _saved_mean,
            _saved_var,
            nullptr,
            __state->{unique_id}_workspace,
            *__state->{unique_id}_workspace_size,
            _reserved_ptr,
            _reserved_size
            ));
        """

        in_connectors = {
            "X": dace.pointer(T),
            "dY": dace.pointer(T),
            "scale": dace.pointer(T),
            "saved_mean": dace.pointer(T),
            "saved_var": dace.pointer(T),
            "reserved_ptr": dace.pointer(dace.typeclass(None)),
            # Here we assume size_t is int64. Unfortunately this is a bad hack and not true
            "reserved_size": dace.int64
        }
        out_connectors = ["dX", "dScale", "dBias"]

        init_code = "{\n" + init_code + "\n}"
        finalize_code = "{\n" + finalize_code + "\n}"
        tasklet = nstate.add_tasklet(
            unique_id, {f"_{i}": t
                        for i, t in in_connectors.items()},
            {f"_{i}": dace.pointer(T)
             for i in out_connectors},
            tasklet_code,
            dace.dtypes.Language.CPP,
            code_init=init_code,
            code_exit=finalize_code,
            state_fields=[
                f"cudnnTensorDescriptor_t *{unique_id}_X_desc;",
                f"cudnnFilterDescriptor_t *{unique_id}_W_desc;",
                f"cudnnTensorDescriptor_t *{unique_id}_dX_desc;",
                f"cudnnTensorDescriptor_t *{unique_id}_dY_desc;",
                f"cudnnTensorDescriptor_t *{unique_id}_dB_desc;",
                f"cudnnFilterDescriptor_t *{unique_id}_dW_desc;",
                f"cudnnTensorDescriptor_t *{unique_id}_dScale_desc;",
                f"cudnnConvolutionDescriptor_t *{unique_id}_conv_desc;",
                f"float *{unique_id}_workspace;",
                f"size_t *{unique_id}_workspace_size;"
            ])
        tasklet.environments = {donnx.environments.cuDNN.full_class_path()}

        # connect inputs
        arr_name = result.given_grad_names["Y"]
        nstate.add_edge(nstate.add_read(arr_name), None, tasklet, f"_dY",
                        nsdfg.make_array_memlet(arr_name))

        for arr_name in ["X", "saved_mean", "scale", "saved_var"]:
            nstate.add_edge(nstate.add_read(arr_name), None, tasklet,
                            f"_{arr_name}", nsdfg.make_array_memlet(arr_name))

        # connect outputs
        arr_name = result.required_grad_names["X"]
        nstate.add_edge(tasklet, "_dX", nstate.add_write(arr_name), None,
                        nsdfg.make_array_memlet(arr_name))
        arr_name = result.required_grad_names["scale"]
        nstate.add_edge(tasklet, "_dScale", nstate.add_write(arr_name), None,
                        nsdfg.make_array_memlet(arr_name))
        arr_name = result.required_grad_names["B"]
        nstate.add_edge(tasklet, "_dBias", nstate.add_write(arr_name), None,
                        nsdfg.make_array_memlet(arr_name))

        # after differentiation, but before validation, we must lower the fwd node,
        # giving the argument that tells it that we need the reserved_ptr output
        def expand(_):
            class Expansion(xf.ExpandTransformation):
                environments = CudnnBatchNormalizationTraining.environments

                @classmethod
                def expansion(cls, node, state, sdfg):
                    return CudnnBatchNormalizationTraining.forward(
                        forward_node,
                        context.forward_state,
                        context.forward_sdfg,
                        reserved_ptr=True)

                @staticmethod
                def annotates_memlets() -> bool:
                    return True

            Expansion._match_node = xf.PatternNode(
                donnx.ONNXBatchNormalization)
            Expansion.apply_to(context.forward_sdfg,
                               verify=False,
                               _match_node=forward_node)

        context.backward_generator.completion_hooks.append(expand)

        # forward the opaque ptr and size
        assert forward_node.add_out_connector("reserved_ptr")
        assert forward_node.add_out_connector("reserved_size")
        reserved_ptr_name, reserved_desc = context.forward_sdfg.add_scalar(
            f"reserved_ptr",
            dace.pointer(dace.typeclass(None)),
            storage=dtypes.StorageType.CPU_Heap,
            transient=True,
            find_new_name=True)

        reserved_size_name, reserved_size_desc = context.forward_sdfg.add_scalar(
            f"reserved_size",
            dace.int64,
            storage=dtypes.StorageType.CPU_Heap,
            transient=True,
            find_new_name=True)

        context.forward_state.add_edge(
            forward_node, "reserved_ptr",
            context.forward_state.add_read(reserved_ptr_name), None,
            dace.Memlet(f"{reserved_ptr_name}[0]"))
        context.forward_state.add_edge(
            forward_node, "reserved_size",
            context.forward_state.add_read(reserved_size_name), None,
            dace.Memlet(f"{reserved_size_name}[0]"))

        bwd_reserved_desc = copy.deepcopy(reserved_desc)
        bwd_reserved_desc.transient = False
        bwd_reserved_size_desc = copy.deepcopy(reserved_size_desc)
        bwd_reserved_size_desc.transient = False
        nsdfg.add_datadesc("reserved_ptr", bwd_reserved_desc)
        nsdfg.add_datadesc("reserved_size", bwd_reserved_size_desc)
        nstate.add_edge(nstate.add_read("reserved_ptr"), None, tasklet,
                        "_reserved_ptr", dace.Memlet("reserved_ptr[0]"))
        nstate.add_edge(nstate.add_read("reserved_size"), None, tasklet,
                        "_reserved_size", dace.Memlet("reserved_size[0]"))

        node = context.backward_state.add_nested_sdfg(
            nsdfg, None, {
                "X", result.given_grad_names["Y"], "scale", "saved_mean",
                "saved_var", "reserved_ptr", "reserved_size"
            }, {result.required_grad_names[a]
                for a in {"X", "scale", "B"}})

        butils.connect_output_from_forward(forward_node, node, context,
                                           "saved_mean")
        butils.connect_output_from_forward(forward_node, node, context,
                                           "saved_var")

        butils.connect_output_from_forward(forward_node, node, context,
                                           "reserved_ptr")
        butils.connect_output_from_forward(forward_node, node, context,
                                           "reserved_size")

        return node, result


@autoregister_params(op="GlobalAveragePool", name="pure")
class PureGlobalAveragePoolingBackward(BackwardImplementation):
    @staticmethod
    def backward_can_be_applied(node: nd.Node, state: dace.SDFGState,
                                sdfg: dace.SDFG) -> bool:
        return len(in_desc_with_name(node, state, sdfg, "X").shape) == 4

    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[nd.Node, BackwardResult]:
        desc = butils.forward_in_desc_with_name(forward_node, context, "X")
        N, C, H, W = desc.shape
        dtype = desc.dtype

        inv = 1.0 / (H * W)

        def bwd(X_grad, Y_grad):
            for n, c, h, w in dace.map[0:N, 0:C, 0:H, 0:W]:
                with dace.tasklet:
                    y_grad << Y_grad[n, c]
                    x_grad >> X_grad[n, c, h, w]
                    x_grad = y_grad * dtype(inv)

        return butils.backward_program_for_node(bwd, context, forward_node)


@autoregister_params(op="Transpose", name="default")
class DefaultTransposeBackward(BackwardImplementation):
    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[nd.Node, BackwardResult]:
        inv_perm = tuple(np.argsort(forward_node.perm))

        node = donnx.ONNXTranspose(forward_node.name + "_backward",
                                   perm=inv_perm)
        context.backward_state.add_node(node)

        result = BackwardResult.empty()
        result.given_grad_names["transposed"] = "data"
        result.required_grad_names["data"] = "transposed"

        return node, result


@autoregister_params(op="Where", name="default")
class WhereBackward(BackwardImplementation):
    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[nd.Node, BackwardResult]:
        # condition, X, Y -> Output
        cdesc = butils.forward_in_desc_with_name(forward_node, context,
                                                 "condition")

        # NOTE: We cannot use ONNX ops for further potential lowering
        # transformations because ONNXMul does not support boolean inputs.
        # notcondition = dace.define_local(condition_shape, condition_dtype)
        # donnx.ONNXMul(A=condition, B=output_grad, C=X_grad)
        # donnx.ONNXNot(X=condition, Y=notcondition)
        # donnx.ONNXMul(A=notcondition, B=output_grad, C=Y_grad)

        if 'X' in required_gradients and 'Y' not in required_gradients:

            def where_backward(condition, output_grad, X_grad):
                X_grad[:] = condition * output_grad
        elif 'Y' in required_gradients and 'X' not in required_gradients:

            def where_backward(condition, output_grad, Y_grad):
                Y_grad[:] = ~condition * output_grad
        elif 'X' in required_gradients and 'Y' in required_gradients:

            def where_backward(condition, output_grad, X_grad, Y_grad):
                X_grad[:] = condition * output_grad
                Y_grad[:] = ~condition * output_grad

        result_node, result = butils.backward_program_for_node(
            where_backward, context, forward_node)

        return result_node, result
