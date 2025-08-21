import dataclasses
from typing import Callable, List, Union, Tuple

import torch
import dace
from dace.codegen.compiled_sdfg import CompiledSDFG
from dace.libraries.onnx.converters import clean_onnx_name
from dace.libraries.onnx.onnx_importer import create_output_array


@dataclasses.dataclass
class DaCeMLTorchFunction:
    """An initialized, callable function for a DaceModule and its associated state"""
    function: Callable  #: the torch callable function
    compiled_sdfgs: List[CompiledSDFG]  #: the compiled SDFGs holding their states
    #: the pointers to the initialized SDFG state handles. Must be passed as the first arguments to function.
    ptr: List[torch.Tensor]


def get_arglist(module: 'dace.frontend.python.module.DaceModule') -> Tuple[List[str], List[str]]:
    """ Get the list of forward-pass argument names for a module

        :param module: the module
        :return: the list of strings that are the argnames to the module, and the list of names of the outputs
    """

    arglist = [clean_onnx_name(i) for i in module.dace_model.inputs]
    outputs = [clean_onnx_name(o) for o in module.dace_model.outputs]
    return arglist, outputs


def compile_and_init_sdfgs(
        module: 'dace.frontend.python.module.DaceModule',
        dummy_inputs) -> (Union[Tuple[CompiledSDFG, int], Tuple[CompiledSDFG, int, CompiledSDFG, int]]):
    """
    Compile SDFGs and initialize them using the provided dummy inputs.
    :param module: the module to compile SDFGs for.
    :param dummy_inputs: the dummy inputs to use
    :return: Tuple of (compiled_sdfg, state_ptr). If the module has a backward
             pass, Tuple of
             (compiled_fwd_sdfg, fwd_state_ptr, compiled_bwd_sdfg, bwd_state_ptr)
    """

    compiled: CompiledSDFG = module.dace_model.compile_and_init()
    # construct the arguments and initialize the SDFG
    args = tuple(dummy_inputs) + module._call_params()
    args = tuple(a.detach() for a in args)
    inputs, symbols, outputs = module.dace_model._call_args(args=args, kwargs={})

    if module.backward:
        forwarded_transients = {
            name: create_output_array(symbols, desc, use_torch=True, zeros=True) if name not in module.dace_model.initialized_parameters else module.dace_model.initialized_parameters[name]
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
        # compile and initialize the backward_sdfg
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
