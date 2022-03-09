# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import library, nodes, properties, subsets
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
        _, inp_arr = sdfg.add_array("_inp_tensor", inp_tensor.shape, inp_tensor.dtype, inp_tensor.storage, strides=inp_tensor.strides)
        _, out_arr = sdfg.add_array("_out_tensor", out_tensor.shape, out_tensor.dtype, out_tensor.storage, strides=out_tensor.strides)
        
        state = sdfg.add_state(f"{node.label}_state")   
        map_params = [f"__i{i}" for i in range(len(inp_arr.shape))]
        map_rng = {i: f"0:{s}"for i, s in zip(map_params, inp_arr.shape)}
        inp_mem = dace.Memlet(expr=f"_inp_tensor[{','.join(map_params)}]")
        out_mem = dace.Memlet(expr=f"_out_tensor[{','.join([map_params[i] for i in node.axes])}]")
        inputs = {"_inp": inp_mem}
        outputs = {"_out": out_mem}
        code = f"_out = {node.alpha} * _inp"
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



@library.node
class TensorTranspose(nodes.LibraryNode):
    """ Implements out-of-place tensor transpositions. """

    implementations = {
        "pure": ExpandPure,
        "HPTT": ExpandHPTT
    }
    default_implementation = "pure"

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
