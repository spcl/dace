# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
A torch python autograd function that calls the SDFG using ctypes.

This can be as an alternative to the C++ registration for large neural nets to
get around the 64 parameter limit of torch's dispatcher.
"""
import copy
import itertools
from typing import List, Dict, Tuple

from dace import data
import torch
from dace.codegen.compiled_sdfg import CompiledSDFG

import dace
from dace.autodiff import BackwardResult
from dace.frontend.ml.onnx.importer import create_output_array
from dace.libraries.torch.dispatchers import DaceTorchFunction
from dace.libraries.torch.dispatchers.common import compile_and_init_sdfgs, \
    get_arglist


def init_remaining_parameters(module, fwd_arglist, input_names, output_names):
    """Initialize remaining parameters that are not inputs or outputs.

    :param module: The DaCe module containing the weights.
    :param fwd_arglist: Forward pass argument list.
    :param input_names: Names of input tensors.
    :param output_names: Names of output tensors.
    :return: Dictionary of constant parameters.
    :raises ValueError: If a parameter is neither an input/output nor a constant.
    """
    # initialize all remaining parameters
    remaining = set(fwd_arglist).difference(itertools.chain(input_names, output_names))
    constants = {}
    for name in remaining:
        # remaining arguments must be constant
        if name not in module.dace_model.clean_weights:
            raise ValueError(f"Cannot generate ctypes dispatcher: SDFG argument {name} is "
                             f"not an input or output of the PyTorch Module, and not a"
                             f" constant.")
        constants[name] = module.dace_model.clean_weights[name]
        if fwd_arglist[name].storage in dace.dtypes.GPU_STORAGES:
            constants[name] = constants[name].cuda()
    return constants


def callable_for_fwd_module(module: 'dace.frontend.ml.torch.DaceModule', forward_compiled: CompiledSDFG):
    """Create a callable for forward pass execution.

    :param module: The DaCe module containing the model.
    :param forward_compiled: Compiled SDFG for forward pass.
    :return: Function that executes the forward pass.
    """
    assert forward_compiled._initialized

    fwd_arglist = forward_compiled.sdfg.arglist()

    input_names, output_names = get_arglist(module)

    constants = init_remaining_parameters(module, fwd_arglist, input_names, output_names)

    def forward(*inputs):
        kwargs = {}

        # set the inputs
        for i, input_name in enumerate(input_names):
            kwargs[input_name] = inputs[i].contiguous()

        # initialize the outputs
        for name in output_names:
            output_desc = forward_compiled.sdfg.arrays[name]
            kwargs[name] = create_output_array(
                {}, output_desc, use_torch=True, zeros=False
            ) if name not in module.dace_model.initialized_parameters else module.dace_model.initialized_parameters[name]

        # call the SDFG
        return forward_compiled(**kwargs, **constants)

    return forward


def callable_for_bwd_module(module: 'dace.frontend.ml.torch.DaceModule', forward_compiled: CompiledSDFG,
                            backward_compiled: CompiledSDFG, backward_result: BackwardResult,
                            forwarded_arrays: Dict[str, data.Data]):

    assert forward_compiled._initialized
    assert backward_compiled._initialized

    fwd_arglist = forward_compiled.sdfg.arglist()

    input_names, output_names = get_arglist(module)

    # arrays that we will forward to the backward pass using saved_for_backward
    forwarded_io_names: List[str] = [name for name in forwarded_arrays if name in output_names or name in input_names]

    # non input/output arrays that we are forwarding
    forwarded_non_io_names: List[str] = [
        name for name in forwarded_arrays if name not in output_names and name not in input_names
    ]

    # for each gradient array that is required, this contains the:
    # * name of the gradient
    # * whether the array requires zero initialization
    # * the descriptor for the array
    gradient_descriptors: List[Tuple[str, bool, data.Data]] = []

    for _, grad_name in backward_result.required_grad_names.items():
        zero_init = backward_result.zero_init.get(grad_name, True)
        desc = backward_compiled.sdfg.arrays[grad_name]

        gradient_descriptors.append((grad_name, zero_init, desc))

    outputs_with_forwarded_outputs: List[str] = copy.deepcopy(output_names)
    outputs_with_forwarded_outputs.extend(n for n in forwarded_arrays if n not in input_names and n not in output_names)

    output_gradient_names: List[str] = [
        backward_result.given_grad_names[output] if output in backward_result.given_grad_names else None
        for output in output_names
    ]
    input_gradient_names: List[str] = [
        backward_result.required_grad_names[input] if input in backward_result.required_grad_names else None
        for input in input_names
    ]

    constants = init_remaining_parameters(module, fwd_arglist, input_names, outputs_with_forwarded_outputs)

    class DifferentiableFunction(torch.autograd.Function):

        @staticmethod
        def forward(ctx, *inputs):
            kwargs = {}

            # set the inputs
            for i, input_name in enumerate(input_names):
                kwargs[input_name] = inputs[i].contiguous()

            # initialize the outputs
            for name in outputs_with_forwarded_outputs:
                output_desc = forward_compiled.sdfg.arrays[name]
                kwargs[name] = create_output_array(
                    {}, output_desc, use_torch=True, zeros=True
                ) if name not in module.dace_model.initialized_parameters else module.dace_model.initialized_parameters[
                    name]

            # call the SDFG
            outputs = forward_compiled(**kwargs, **constants)

            # save inputs/outputs for backward
            ctx.save_for_backward(*(kwargs[name] for name in forwarded_io_names))

            # save non- input/output values for backward
            for name in forwarded_non_io_names:
                setattr(ctx, f"dace_saved_{name}", kwargs[name])

            return outputs

        @staticmethod
        def backward(ctx, *grad_outputs):
            kwargs = {}

            # recover saved values
            saved = ctx.saved_tensors
            for value_name, saved_value in zip(forwarded_io_names, saved):
                kwargs[value_name] = saved_value

            for value_name in forwarded_non_io_names:
                kwargs[value_name] = getattr(ctx, f"dace_saved_{value_name}")

            # create gradient buffers of inputs
            for grad_name, zero_init, desc in gradient_descriptors:
                kwargs[grad_name] = create_output_array({}, desc, use_torch=True, zeros=zero_init)

            # grab gradient buffers of outputs
            for grad_name, grad_value in zip(output_gradient_names, grad_outputs):
                kwargs[grad_name] = grad_value.contiguous()

            # call bwd sdfg
            backward_compiled(**kwargs)

            # return grads
            grads = tuple(None if name is None else kwargs[name] for name in input_gradient_names)
            if len(grads) == 1:
                return grads[0]
            return grads

    return lambda *args: DifferentiableFunction.apply(*args)


def get_ctypes_dispatcher(module: 'dace.frontend.ml.torch.DaceModule', dummy_inputs) -> DaceTorchFunction:
    """
    Get a torch callable for the module. This will compile the sdfg and create a
    wrapper python callable that can be used with PyTorch.

    :param module: the module.
    :param dummy_inputs: dummy inputs to initialize the model with.
    :return: the callable function for the SDFG.
    """

    # build the SDFG
    # set all states to not-sync
    for state in module.sdfg.nodes():
        state.nosync = True

    if module.backward:
        # TODO we could return the inferred symbols here
        compiled, _, compiled_bwd, _ = compile_and_init_sdfgs(module, dummy_inputs)

        function = callable_for_bwd_module(module, compiled, compiled_bwd, module._ad_result, module._ad_inp_arrs)
        compiled_sdfgs = [compiled, compiled_bwd]
    else:
        compiled, _ = compile_and_init_sdfgs(module, dummy_inputs)
        function = callable_for_fwd_module(module, compiled)
        compiled_sdfgs = [compiled]

    result = DaceTorchFunction(
        function=function,
        compiled_sdfgs=compiled_sdfgs,
        # no pointers required for ctypes dispatcher
        ptr=[])
    return result
