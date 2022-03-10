# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import collections
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


@library.expansion
class ExpandCuTensor(ExpandTransformation):
    """
    Implements the TensorDot library node using cuTENSOR for CUDA-compatible GPUs.
    For more information, see https://developer.nvidia.com/cutensor.
    """

    environments = []


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
