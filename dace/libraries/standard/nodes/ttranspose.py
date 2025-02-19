# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import multiprocessing
from dace import library, nodes, properties
from dace.libraries.blas import blas_helpers
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from numbers import Number
from .. import environments

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
        inp_tensor, out_tensor = node.validate(parent_sdfg, parent_state)
        axes = ','.join([symstr(a) for a in node.axes])
        shape = ','.join([symstr(s) for s in inp_tensor.shape])
        dchar = blas_helpers.to_blastype(inp_tensor.dtype.type).lower()
        if dchar not in ('s', 'd', 'c', 'z'):
            raise TypeError("HPTT supports only single and double (and corresponding complex) FP datatypes")
        alpha = symstr(node.alpha)
        beta = symstr(node.beta)
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
    Implements the TensorDot library node using cuTENSOR for CUDA-compatible GPUs.
    For more information, see https://developer.nvidia.com/cutensor.
    """


    environments = [environments.cuTensor]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        in_tensor, out_tensor = node.validate(parent_sdfg, parent_state)

        dtype = out_tensor.dtype.base_type
        func, cuda_type, _ = blas_helpers.cublas_type_metadata(dtype)
        cuda_dtype = blas_helpers.dtype_to_cudadatatype(dtype)
        compute_type = f"CUTENSOR_COMPUTE_DESC{cuda_dtype[cuda_dtype.rfind('_'):]}"
        func = func + 'getrf'

        alpha = f"({cuda_type})1.0"
        abtext = f"""
            {cuda_type} alpha = {alpha};
        """

        in_modes = list(range(len(in_tensor.shape)))
        out_modes = node.axes

        from dace.codegen.targets import cpp # to avoid import loop

        modes = f"""
            std::vector<int> modeA{{{','.join(cpp.sym2cpp(m) for m in in_modes)}}};
            std::vector<int> modeB{{{','.join(cpp.sym2cpp(m) for m in out_modes)}}};
        """

        extents = "std::unordered_map<int, int64_t> in_extent;\n"
        for i, s in zip(in_modes, in_tensor.shape):
            extents += f"in_extent[{i}] = {s};\n"
        extents += "std::unordered_map<int, int64_t> out_extent;\n"
        for i, s in zip(in_modes, out_tensor.shape):
            extents += f"out_extent[{i}] = {s};\n"
        extents += f"""
            std::vector<int64_t> extentA;
            for (auto mode : modeA) extentA.push_back(in_extent[mode]);
            std::vector<int64_t> extentB;
            for (auto mode : modeA) extentB.push_back(in_extent[mode]);
        """

        extents += f"""
            std::vector<int64_t> stridesA{{{','.join(cpp.sym2cpp(s) for s in in_tensor.strides)}}};
            std::vector<int64_t> stridesB{{{','.join(cpp.sym2cpp(s) for s in out_tensor.strides)}}};
        """

        tdesc = f"""
            //printf("Start permutation op!\\n");
            cutensorTensorDescriptor_t descA;
            cutensorTensorDescriptor_t descB;
            dace::standard::CheckCuTensorError(
                cutensorCreateTensorDescriptor(
                    __dace_cutensor_handle, &descA, modeA.size(), extentA.data(), stridesA.data(), {cuda_dtype.replace("CUDA", "CUTENSOR")}, 256
                )
            );
            dace::standard::CheckCuTensorError(
                cutensorCreateTensorDescriptor(
                    __dace_cutensor_handle, &descB, modeB.size(), extentB.data(), stridesB.data(), {cuda_dtype.replace("CUDA", "CUTENSOR")}, 256
                )
            );
            //printf("Tensor descriptors created!\\n");
        """

        cdesc = f"""
            cutensorPlanPreference_t preference;
            cutensorOperationDescriptor_t op;
            dace::standard::CheckCuTensorError(
                cutensorCreatePlanPreference(__dace_cutensor_handle, &preference, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_DEFAULT)
            );
            dace::standard::CheckCuTensorError(
                cutensorCreatePermutation(__dace_cutensor_handle, &op, descA, modeA.data(), CUTENSOR_OP_IDENTITY, descB, modeB.data(), {compute_type})
            );
            //printf("Plan preference and operation descriptor created!\\n");
        """

        workspace = """
        """

        execute = """
            cutensorPlan_t plan;
            dace::standard::CheckCuTensorError(
                cutensorCreatePlan(__dace_cutensor_handle, &plan, op, preference, 0)
            );
            cutensorStatus_t err = cutensorPermute(__dace_cutensor_handle, plan, &alpha, _inp_tensor, _out_tensor, __dace_current_stream);
            if(err != CUTENSOR_STATUS_SUCCESS) {
                printf("ERROR: %s\\n", cutensorGetErrorString(err));
            }
            cudaStreamSynchronize(__dace_current_stream);
            if(err != CUTENSOR_STATUS_SUCCESS) {
                printf("ERROR: %s\\n", cutensorGetErrorString(err));
            }
            //printf("Transpose executed!\\n");
        """

        code = f"{environments.cuTensor.handle_setup_code(node)}{abtext}{modes}{extents}{tdesc}{cdesc}{workspace}{execute}"

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)

        return tasklet


@library.node
class TensorTranspose(nodes.LibraryNode):
    """ Implements out-of-place tensor transpositions. """

    implementations = {"pure": ExpandPure, "HPTT": ExpandHPTT, "cuTensor": ExpandCuTensor}
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
