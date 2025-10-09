"""
Common utilities for PyTorch-DaCe dispatchers.

This module provides shared functionality for different dispatcher implementations,
including:
- SDFG compilation and initialization
- Argument list extraction and processing
- State management for forward and backward passes
- Integration with PyTorch's autograd system
"""

import dataclasses
from typing import Callable, List, Tuple, Union

import dace
import torch
from dace.codegen.compiled_sdfg import CompiledSDFG
from dace.libraries.onnx.converters import clean_onnx_name
from dace.libraries.onnx.onnx_importer import create_output_array


@dataclasses.dataclass
class DaCeMLTorchFunction:
    """
    An initialized, callable function for a DaceModule and its associated state.

    This dataclass encapsulates a compiled DaCe module with its runtime state,
    providing a callable interface for PyTorch integration.

    Attributes:
        function: The PyTorch callable function that executes the SDFG.
        compiled_sdfgs: The compiled SDFGs holding their runtime states.
        ptr: Pointers to the initialized SDFG state handles. These must be
            passed as the first arguments to the function.
    """
    function: Callable
    compiled_sdfgs: List[CompiledSDFG]
    ptr: List[torch.Tensor]


def get_arglist(module: 'dace.frontend.python.module.DaceModule') -> Tuple[List[str], List[str]]:
    """
    Get the list of forward-pass argument names for a module.

    Args:
        module: The DaCe module to extract argument names from.

    Returns:
        A tuple of (input_names, output_names) where each is a list of cleaned
        argument names suitable for use in generated code.
    """

    arglist = [clean_onnx_name(input_name) for input_name in module.dace_model.inputs]
    outputs = [clean_onnx_name(output_name) for output_name in module.dace_model.outputs]
    return arglist, outputs


def compile_and_init_sdfgs(
    module: 'dace.frontend.python.module.DaceModule', dummy_inputs
) -> Union[Tuple[CompiledSDFG, torch.Tensor], Tuple[CompiledSDFG, torch.Tensor, CompiledSDFG, torch.Tensor]]:
    """
    Compile SDFGs and initialize them using the provided dummy inputs.

    This function compiles the forward pass SDFG and optionally the backward pass
    SDFG if the module has automatic differentiation enabled. It initializes both
    SDFGs with the appropriate tensors and parameters.

    Args:
        module: The DaCe module to compile SDFGs for.
        dummy_inputs: The dummy inputs to use for shape inference and initialization.

    Returns:
        If the module has no backward pass:
            (compiled_sdfg, state_ptr)
        If the module has a backward pass:
            (compiled_fwd_sdfg, fwd_state_ptr, compiled_bwd_sdfg, bwd_state_ptr)

        Where state_ptr is a torch.Tensor containing the pointer to the SDFG state.
    """

    compiled: CompiledSDFG = module.dace_model.compile_and_init()
    # Construct the arguments and initialize the SDFG
    args = tuple(dummy_inputs) + module._call_params()
    args = tuple(arg.detach() for arg in args)
    inputs, symbols, outputs = module.dace_model._call_args(args=args, kwargs={})

    if module.backward:
        forwarded_transients = {
            name:
            create_output_array(symbols, desc, use_torch=True, zeros=True)
            if name not in module.dace_model.initialized_parameters else module.dace_model.initialized_parameters[name]
            for name, desc in module._ad_inp_arrs.items()
        }
    else:
        forwarded_transients = {}

    all_kwargs = {**inputs, **outputs, **symbols, **forwarded_transients, **module.dace_model.initialized_parameters}

    compiled.initialize(**all_kwargs)
    for _, hook in module.post_compile_hooks.items():
        hook(compiled)
    handle_ptr = torch.tensor([compiled._libhandle.value]).squeeze(0)

    if module.backward:
        # Compile and initialize the backward_sdfg
        compiled_bwd: CompiledSDFG = module.backward_sdfg.compile()

        required_grads = {
            bwd_name: create_output_array(symbols, compiled_bwd.sdfg.arrays[bwd_name], use_torch=True, zeros=True)
            for _, bwd_name in module._ad_result.required_grad_names.items()
        }
        given_grads = {
            bwd_name: create_output_array(symbols, compiled_bwd.sdfg.arrays[bwd_name], use_torch=True, zeros=True)
            for _, bwd_name in module._ad_result.given_grad_names.items()
        }

        compiled_bwd.initialize(**required_grads, **given_grads, **forwarded_transients)
        bwd_handle_ptr = torch.tensor([compiled_bwd._libhandle.value]).squeeze(0)
        return compiled, handle_ptr, compiled_bwd, bwd_handle_ptr
    else:
        return compiled, handle_ptr
