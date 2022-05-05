# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import collections
import dace
import multiprocessing
from dace import library, nodes, properties
from dace.data import _prod
from dace.libraries.blas import blas_helpers
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from numbers import Number
import dace.libraries.linalg.environments as environments


@library.expansion
class ExpandPure(ExpandTransformation):
    """ Implements the pure expansion of TensorDot library node. """
    
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        left_tensor, right_tensor, out_tensor = node.validate(parent_sdfg, parent_state)

        sdfg = dace.SDFG(f"{node.label}_sdfg")
        _, left_arr = sdfg.add_array("_left_tensor", left_tensor.shape, left_tensor.dtype, left_tensor.storage, strides=left_tensor.strides)
        _, right_arr = sdfg.add_array("_right_tensor", right_tensor.shape, right_tensor.dtype, right_tensor.storage, strides=right_tensor.strides)
        _, out_arr = sdfg.add_array("_out_tensor", out_tensor.shape, out_tensor.dtype, out_tensor.storage, strides=out_tensor.strides)
        
        init_state = sdfg.add_state(f"{node.label}_init", is_start_state=True)
        init_state.add_mapped_tasklet(f"{node.label}_init_tasklet", 
                                 {f"__i{i}": f"0:{symstr(s)}" for i, s in enumerate(out_tensor.shape)},
                                 {},
                                 '__out = 0',
                                 {'__out': dace.Memlet(expr=f"_out_tensor[{','.join(['__i%d' % i for i in range(len(out_tensor.shape))])}]")},
                                 external_edges=True)

        state = sdfg.add_state(f"{node.label}_state")
        sdfg.add_edge(init_state, state, dace.InterstateEdge())

        outer_map_shape = list([s for i, s in enumerate(left_tensor.shape) if i not in node.left_axes])
        outer_map_shape.extend([s for i, s in enumerate(right_tensor.shape) if i not in node.right_axes])
        outer_map_params = [f"__oi{i}" for i in range(len(outer_map_shape))]
        outer_map_rng = {i: f"0:{symstr(s)}"for i, s in zip(outer_map_params, outer_map_shape)}
        inner_map_shape = list([left_tensor.shape[i] for i in node.left_axes])
        inner_map_params = [f"__ii{i}" for i in range(len(inner_map_shape))]
        inner_map_rng = {i: f"0:{symstr(s)}"for i, s in zip(inner_map_params, inner_map_shape)}

        left_idx = outer_map_params[:len(left_tensor.shape)-len(node.left_axes)]
        left_dict = {j: inner_map_params[i] for i, j in enumerate(node.left_axes)}
        left_sorted_dict = collections.OrderedDict(sorted(left_dict.items()))
        for k, v in left_sorted_dict.items():
            left_idx.insert(k, v)
        right_idx = outer_map_params[len(left_tensor.shape)-len(node.left_axes):]
        right_dict = {j: inner_map_params[i] for i, j in enumerate(node.right_axes)}
        right_sorted_dict = collections.OrderedDict(sorted(right_dict.items()))
        for k, v in right_sorted_dict.items():
            right_idx.insert(k, v)
        out_idx = outer_map_params
        if node.permutation:
            out_idx = [outer_map_params[i] for i in node.permutation]

        left_mem = dace.Memlet(expr=f"_left_tensor[{','.join(left_idx)}]")
        right_mem = dace.Memlet(expr=f"_right_tensor[{','.join(right_idx)}]")
        out_mem = dace.Memlet(expr=f"_out_tensor[{','.join(out_idx)}]", wcr="lambda x, y: x + y")
        inputs = {"_left": left_mem, "_right": right_mem}
        outputs = {"_out": out_mem}
        code = f"_out = _left * _right"
        state.add_mapped_tasklet(f"{node.label}_tasklet", {**outer_map_rng, **inner_map_rng}, inputs, code, outputs, external_edges=True)

        return sdfg


@library.expansion
class ExpandTTGT(ExpandTransformation):
    """
    Expands the TensorDot library node to TensorTranspose + GEMM operations.
    TTGT stands for Transpose-Transpose-GEMM-Transpose.
    """
    
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        left_tensor, right_tensor, out_tensor = node.validate(parent_sdfg, parent_state)

        sdfg = dace.SDFG(f"{node.label}_sdfg")
        _, left_arr = sdfg.add_array("_left_tensor", left_tensor.shape, left_tensor.dtype, left_tensor.storage, strides=left_tensor.strides)
        _, right_arr = sdfg.add_array("_right_tensor", right_tensor.shape, right_tensor.dtype, right_tensor.storage, strides=right_tensor.strides)
        _, out_arr = sdfg.add_array("_out_tensor", out_tensor.shape, out_tensor.dtype, out_tensor.storage, strides=out_tensor.strides)

        from dace.frontend.python.replacements import _transpose
        # NOTE: We use the numpy.transpose replacement because:
        # (1) It will return the tensor itself if transposition is uncessary.
        # (2) It will use matrix transpose operation for 2-mode tensors.
        state = sdfg.add_state(f"{node.label}_inp_transpose_state", is_start_state=True)

        if node.left_axes == list(range(len(node.left_axes))):
            transA = True
        else:
            transA = False
        if node.right_axes == list(range(len(right_arr.shape)-len(node.right_axes), len(right_arr.shape))):
            transB = True
        else:
            transB = False
        
        if transA:
            left_tt = "_left_tensor"
            left_tt_arr = left_arr
        else:
            left_axes = [i for i in range(len(left_arr.shape)) if i not in node.left_axes]
            left_axes.extend(node.left_axes)
            left_tt = _transpose(None, sdfg, state, "_left_tensor", left_axes)
            left_tt_arr = sdfg.arrays[left_tt]

        if transB:
            right_tt = "_right_tensor"
            right_tt_arr = right_arr
        else:
            right_axes = list(node.right_axes)
            right_axes.extend([i for i in range(len(right_arr.shape)) if i not in node.right_axes])
            right_tt = _transpose(None, sdfg, state, "_right_tensor", right_axes)
            right_tt_arr = sdfg.arrays[right_tt]

        from dace.libraries.blas import Gemm
        prv_state = state
        state = sdfg.add_state(f"{node.label}_gemm_state")
        sdfg.add_edge(prv_state, state, dace.InterstateEdge())
        
        if transA:
            left_shape = [_prod(left_tt_arr.shape[:len(node.left_axes)]), _prod(left_tt_arr.shape[len(node.left_axes):])]
            left_strides = [left_tt_arr.strides[len(node.left_axes)-1], left_tt_arr.strides[-1]]
        else:
            left_shape = [_prod(left_tt_arr.shape[:-len(node.left_axes)]), _prod(left_tt_arr.shape[len(left_tt_arr.shape)-len(node.left_axes):])]
            left_strides = [left_tt_arr.strides[-len(node.left_axes)-1], left_tt_arr.strides[-1]]
        left_vname, left_view = sdfg.add_view(left_tt, left_shape, left_tt_arr.dtype, left_tt_arr.storage, strides=left_strides, find_new_name=True)
        left_anode = state.add_read(left_tt)
        left_vnode = state.add_access(left_vname)
        state.add_edge(left_anode, None, left_vnode, 'views', dace.Memlet.from_array(left_tt, left_tt_arr))

        if transB:
            right_shape = [_prod(right_tt_arr.shape[:-len(node.right_axes)]), _prod(right_tt_arr.shape[len(right_tt_arr.shape)-len(node.right_axes):])]
            right_strides = [right_tt_arr.strides[-len(node.right_axes)-1], right_tt_arr.strides[-1]]
        else:
            right_shape = [_prod(right_tt_arr.shape[0:len(node.right_axes)]), _prod(right_tt_arr.shape[len(node.right_axes):])]
            right_strides = [right_tt_arr.strides[len(node.right_axes)-1], right_tt_arr.strides[-1]]
        right_vname, right_view = sdfg.add_view(right_tt, right_shape, right_tt_arr.dtype, right_tt_arr.storage, strides=right_strides, find_new_name=True)
        right_anode = state.add_read(right_tt)
        right_vnode = state.add_access(right_vname)
        state.add_edge(right_anode, None, right_vnode, 'views', dace.Memlet.from_array(right_tt, right_tt_arr))

        tasklet = Gemm('_GEMM_', cin=False, transA=transA, transB=transB)
        state.add_edge(left_vnode, None, tasklet, '_a', dace.Memlet.from_array(left_vname, left_view))
        state.add_edge(right_vnode, None, tasklet, '_b', dace.Memlet.from_array(right_vname, right_view))

        # Output handling
        out_shape = []
        if transA:
            out_shape.append(left_shape[1])
        else:
            out_shape.append(left_shape[0])
        if transB:
            out_shape.append(right_shape[0])
        else:
            out_shape.append(right_shape[1])
        if node.permutation and node.permutation != list(range(len(node.permutation))):
            dot_shape = [s for i, s in enumerate(left_tensor.shape) if i not in node.left_axes]
            dot_shape.extend([s for i, s in enumerate(right_tensor.shape) if i not in node.right_axes])
            dot_name, dot_arr = sdfg.add_temp_transient(dot_shape, out_arr.dtype, out_arr.storage)
            out_strides = [dot_arr.strides[len(left_tt_arr.shape)-len(node.left_axes)-1], dot_arr.strides[-1]]
            dot_vname, dot_view = sdfg.add_view('__gemm_out', out_shape, dot_arr.dtype, dot_arr.storage, strides=out_strides, find_new_name=True)
            dot_anode = state.add_access(dot_name)
            dot_vnode = state.add_access(dot_vname)
            state.add_edge(tasklet, '_c', dot_vnode, None, dace.Memlet.from_array(dot_vname, dot_view))
            state.add_edge(dot_vnode, 'views', dot_anode, None, dace.Memlet.from_array(dot_name, dot_arr))
            out_node = state.add_write('_out_tensor')
            from dace.libraries.ttranspose import TensorTranspose
            tasklet = TensorTranspose('_TensorTranspose', node.permutation)
            state.add_edge(dot_anode, None, tasklet, '_inp_tensor', dace.Memlet.from_array(dot_name, dot_arr))
            state.add_edge(tasklet, '_out_tensor', out_node, None, dace.Memlet.from_array('_out_tensor', out_arr))
        else:
            out_strides = [out_arr.strides[len(left_tt_arr.shape)-len(node.left_axes)-1], out_arr.strides[-1]]
            out_vname, out_view = sdfg.add_view('__gemm_out', out_shape, out_arr.dtype, out_arr.storage, strides=out_strides, find_new_name=True)
            out_anode = state.add_access('_out_tensor')
            out_vnode = state.add_access(out_vname)
            state.add_edge(tasklet, '_c', out_vnode, None, dace.Memlet.from_array(out_vname, out_view))
            state.add_edge(out_vnode, 'views', out_anode, None, dace.Memlet.from_array('_out_tensor', out_arr))

        return sdfg


@library.expansion
class ExpandCuTensor(ExpandTransformation):
    """
    Implements the TensorDot library node using cuTENSOR for CUDA-compatible GPUs.
    For more information, see https://developer.nvidia.com/cutensor.
    """

    environments = [environments.cuTensor]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        left_tensor, right_tensor, out_tensor = node.validate(parent_sdfg, parent_state)

        dtype = out_tensor.dtype.base_type
        func, cuda_type, _ = blas_helpers.cublas_type_metadata(dtype)
        cuda_dtype = blas_helpers.dtype_to_cudadatatype(dtype)
        compute_type = f"CUTENSOR_COMPUTE{cuda_dtype[cuda_dtype.rfind('_'):]}"
        func = func + 'getrf'

        alpha = f"({cuda_type})1.0"
        beta = f"({cuda_type})0.0"
        abtext = f"""
            {cuda_type} alpha = {alpha};
            {cuda_type} beta = {beta};
        """

        left_modes = list(range(len(left_tensor.shape)))
        right_modes = [node.left_axes[node.right_axes.index(i)] if i in node.right_axes else len(left_tensor.shape) + i
                       for i in range(len(right_tensor.shape))]
        out_modes = [i for i in left_modes if i not in node.left_axes]
        out_modes.extend([i for i in right_modes if i not in node.left_axes])
        if node.permutation and node.permutation != list(range(len(node.permutation))):
            out_modes = [node.permutation[i] for i in out_modes]

        modes = f"""
            std::vector<int> modeA{{{','.join(str(m) for m in left_modes)}}};
            std::vector<int> modeB{{{','.join(str(m) for m in right_modes)}}};
            std::vector<int> modeC{{{','.join(str(m) for m in out_modes)}}};
        """

        extents = "std::unordered_map<int, int64_t> extent;\n"
        for i, s in zip(left_modes, left_tensor.shape):
            extents += f"extent[{i}] = {s};\n"
        for i, s in zip(right_modes, left_tensor.shape):
            if i in node.right_axes:
                continue
            extents += f"extent[{i}] = {s};\n"
        extents += f"""
            std::vector<int64_t> extentA;
            for (auto mode : modeA) extentA.push_back(extent[mode]);
            std::vector<int64_t> extentB;
            for (auto mode : modeB) extentB.push_back(extent[mode]);
            std::vector<int64_t> extentC;
            for (auto mode : modeC) extentC.push_back(extent[mode]);
        """

        tdesc = f"""
            cutensorTensorDescriptor_t descA, descB, descC;
            cutensorInitTensorDescriptor(
                &__dace_cutensor_handle, &descA, modeA.size(), extentA.data(), NULL, {cuda_dtype}, CUTENSOR_OP_IDENTITY);
            cutensorInitTensorDescriptor(
                &__dace_cutensor_handle, &descB, modeB.size(), extentB.data(), NULL, {cuda_dtype}, CUTENSOR_OP_IDENTITY);
            cutensorInitTensorDescriptor(
                &__dace_cutensor_handle, &descC, modeA.size(), extentA.data(), NULL, {cuda_dtype}, CUTENSOR_OP_IDENTITY);
        """

        cdesc = f"""
            uint32_t alignmentRequirementA, alignmentRequirementB, alignmentRequirementC;
            cutensorGetAlignmentRequirement(&__dace_cutensor_handle, _left_tensor, &descA, &alignmentRequirementA);
            cutensorGetAlignmentRequirement(&__dace_cutensor_handle, _right_tensor, &descB, &alignmentRequirementB);
            cutensorGetAlignmentRequirement(&__dace_cutensor_handle, _out_tensor, &descC, &alignmentRequirementC);
            cutensorContractionDescriptor_t desc;
            cutensorInitContractionDescriptor(
                &__dace_cutensor_handle, &desc,
                &descA, modeA.data(), alignmentRequirementA,
                &descB, modeB.data(), alignmentRequirementB,
                &descC, modeC.data(), alignmentRequirementC,
                &descC, modeC.data(), alignmentRequirementC,
                {compute_type});
        """

        workspace = """
            cutensorContractionFind_t find;
            cutensorInitContractionFind(&__dace_cutensor_handle, &find, CUTENSOR_ALGO_DEFAULT);
            size_t worksize = 0;
            cutensorContractionGetWorkspace(
                &__dace_cutensor_handle, &desc, &find, CUTENSOR_WORKSPACE_RECOMMENDED, &worksize);
            void *work = nullptr;
            if (worksize > 0) cudaMalloc(&work, worksize);
        """

        execute = """
            cutensorContractionPlan_t plan;
            cutensorInitContractionPlan(&__dace_cutensor_handle, &plan, &desc, &find, worksize);
            cutensorContraction(
                &__dace_cutensor_handle, &plan,
                (void*)&alpha, _left_tensor, _right_tensor, (void*)&beta, _out_tensor, _out_tensor,
                work, worksize, __dace_current_stream);
            if (work) cudaFree(work);
        """

        code = f"{environments.cuTensor.handle_setup_code(node)}{abtext}{modes}{extents}{tdesc}{cdesc}{workspace}{execute}"

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        # conn = tasklet.out_connectors
        # conn = {c: (dace.dtypes.pointer(dace.int32) if c == '_res' else t) for c, t in conn.items()}
        # tasklet.out_connectors = conn

        return tasklet


@library.node
class TensorDot(nodes.LibraryNode):
    """ Implements tensor dot-product. """

    implementations = {
        "pure": ExpandPure,
        "TTGT": ExpandTTGT,
        "cuTENSOR": ExpandCuTensor
    }
    default_implementation = None

    left_axes = properties.ListProperty(element_type=int, default=[], desc="Left tensor's contracting modes")
    right_axes = properties.ListProperty(element_type=int, default=[], desc="Right tensor's contracting modes")
    permutation = properties.ListProperty(element_type=int, allow_none=True, default=None, desc="Permutation of the output tensor")

    def __init__(self, name, left_axes=[], right_axes=[], permutation=None, *args, **kwargs):
        super().__init__(name, *args, inputs={"_left_tensor", "_right_tensor"}, outputs={"_out_tensor"}, **kwargs)
        self.left_axes = left_axes
        self.right_axes = right_axes
        self.permutation = permutation
    
    def validate(self, sdfg, state):
        """
        Validates the tensor dot-product operation.
        :return: A triple (left_tensor, right_tensor, out_tensor) for the data descriptors in the parent SDFG.
        """

        left_tensor, right_tensor, out_tensor = None, None, None
        for e in state.out_edges(self):
            if e.src_conn == "_out_tensor":
                out_tensor = sdfg.arrays[e.data.data]
        for e in state.in_edges(self):
            if e.dst_conn == "_left_tensor":
                left_tensor = sdfg.arrays[e.data.data]
            elif e.dst_conn == "_right_tensor":
                right_tensor = sdfg.arrays[e.data.data]

        if not left_tensor or not right_tensor:
            raise ValueError("Missing the input tensors.")
        if not out_tensor:
            raise ValueError("Missing the output tensor.")

        if left_tensor.dtype != right_tensor.dtype or left_tensor.dtype != out_tensor.dtype:
            raise TypeError("The datatype of the input and output tensors must match.")    
        if left_tensor.storage != right_tensor.storage or left_tensor.storage != out_tensor.storage:
            raise ValueError("The storage of the input and output tensors must match.")

        if any(a >= len(left_tensor.shape) or a < 0 for a in self.left_axes):
            raise ValueError("Axes for left tensor are out-of-bounds.")
        if any(a >= len(right_tensor.shape) or a < 0 for a in self.right_axes):
            raise ValueError("Axes for right tensor are out-of-bounds.")
        if len(self.left_axes) != len(self.right_axes):
            raise ValueError("The input tensors must have the same number of contracting modes.")
        if any(left_tensor.shape[l] != right_tensor.shape[r] for l, r in zip(self.left_axes, self.right_axes)):
            raise ValueError("The input tensors' contracting modes must have the same length.")
        
        dot_shape = [s for i, s in enumerate(left_tensor.shape) if i not in self.left_axes]
        dot_shape.extend([s for i, s in enumerate(right_tensor.shape) if i not in self.right_axes])
        out_shape = list(out_tensor.shape)
        if len(dot_shape) != len(out_shape):
            raise ValueError("The intermediate (dot-product) and output tensors must have the same number of modes..")
        
        # # We check if the output shape is a permutation of a dot-product shape.
        # # NOTE: Since the shapes may be symbolic, we cannot just sort and compare them.
        # for s in out_shape:
        #     try:
        #         idx = dot_shape.index(s)
        #         dot_shape.pop(idx)
        #     except ValueError:
        #         raise ValueError("The output tensor shape is not a permutation of the dot-product shape.")
        # if dot_shape:
        #     raise ValueError("The output tensor shape is not a permutation of the dot-product shape.")


        if not self.permutation:
            if dot_shape != out_shape:
                raise ValueError("The shapes of the intermediate (dot-product) and output tensors must match.")
        else:
            # NOTE: If the output tensor is transposed, then the permutation must be given explicitely. The permutation
            # can only be inferred if each tensor mode has different length, which should never be assumed.
            if len(out_tensor.shape) != len(self.permutation):
                raise ValueError("The permutation list property must have as many elements as the number of output tensor modes.")
            if sorted(self.permutation) != list(range(len(out_tensor.shape))):
                raise ValueError("The permutation list property is not a perimutation of the output tensor's modes.")
            transposed_shape = [dot_shape[p] for p in self.permutation]
            if transposed_shape != list(out_tensor.shape):
                raise ValueError("The permutation of the intermediate (dot-product) shape does not match the output shape.")

        return left_tensor, right_tensor, out_tensor
