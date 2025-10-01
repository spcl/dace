"""Code generation for PyTorch C++ dispatched operators."""
import copy
import dataclasses
import itertools
import logging
import operator
import os
from typing import List, Tuple, Callable, Optional, Dict, Union

import dace.library
import numpy as np
import torch
import dace
from dace import dtypes as dt, data
from dace.codegen import targets, compiler
from dace.codegen.codeobject import CodeObject
from dace.codegen.compiled_sdfg import CompiledSDFG
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.common import sym2cpp

from dace.autodiff import BackwardResult
from dace.libraries.torch.environments import PyTorch
from dace.util import is_cuda, platform_library_name

from dace.libraries.torch.dispatchers.common import DaCeMLTorchFunction, compile_and_init_sdfgs, get_arglist

log = logging.getLogger(__name__)

_REPLACED_CTYPES = {dace.int64: "int64_t", dace.uint64: "uint64_t", dace.float16: "at::Half"}


def torch_ctype(dtype: dace.typeclass) -> str:
    """Convert a DaCe type to the corresponding PyTorch C++ type string.

    Args:
        dtype: The DaCe typeclass to convert

    Returns:
        The corresponding C++ type string for PyTorch
    """
    if isinstance(dtype, dace.pointer):
        # assuming pointers are 64 bit
        ctype = "int64_t"
    elif dtype in _REPLACED_CTYPES:
        ctype = _REPLACED_CTYPES[dtype]
    else:
        ctype = dtype.ctype
    return ctype


_TYPECLASS_TO_TORCH_DTYPE_STR = {
    dt.bool: "kBool",
    dt.int8: "kInt8",
    dt.uint8: "kUInt8",
    dt.int16: "kInt16",
    dt.int32: "kInt32",
    dt.int64: "kInt64",
    dt.float16: "kFloat16",
    dt.float32: "kFloat32",
    dt.float64: "kFloat64",
}


def typeclass_to_torch_cpp_type(type: dace.typeclass) -> str:
    """Convert a DaCe typeclass to PyTorch C++ tensor type string.

    Args:
        type: The DaCe typeclass to convert

    Returns:
        The corresponding PyTorch tensor type string (e.g., 'kFloat32')
    """
    if isinstance(type, dace.pointer):
        # assuming pointers are 64 bit
        return "kInt64"
    else:
        return _TYPECLASS_TO_TORCH_DTYPE_STR[type]


def tensor_init_for_desc(name: str, desc: data.Data, clean_weights: Dict[str, torch.Tensor], zeros=True) -> str:
    """Emit the initialization code for a descriptor.

    Args:
        name: The name of the tensor
        desc: The data descriptor
        clean_weights: Dictionary of constant weights
        zeros: Whether to initialize with zeros (True) or empty (False)

    Returns:
        C++ code string for tensor initialization
    """

    # Check if name is in clean_weights
    if name in clean_weights:
        # Get the tensor from clean_weights
        weight_tensor = clean_weights[name]

        # Convert the tensor to a C++ initializer list format
        # Flatten the tensor and convert to list
        values = weight_tensor.flatten().tolist()

        # Format the values based on the data type
        def format_value(v, dtype):
            if dtype in [dt.float32, dt.float16]:
                return f'{v}f'
            elif dtype == dt.float64:
                return str(v)
            elif dtype in [dt.int8, dt.int16, dt.int32, dt.int64, dt.uint8]:
                return str(int(v))
            elif dtype == dt.bool:
                return str(v).lower()
            else:
                return str(v)

        # Format the values as a C++ initializer list
        values_str = ', '.join(format_value(v, desc.dtype) for v in values)

        return f"""\
            Tensor {name} = torch::from_blob(
                new float[{len(values)}]{{{values_str}}},
                {{{', '.join(str(s) for s in desc.shape)}}},
                torch::TensorOptions()
                    .dtype(torch::{typeclass_to_torch_cpp_type(desc.dtype)})
                    .device(torch::{'kCUDA' if is_cuda(desc.storage) else 'kCPU'})
                    .layout(torch::kStrided)).clone();
            """
    else:
        # Initialize with zeros or empty
        return f"""\
            Tensor {name} = torch::{'zeros' if zeros else 'empty'}(
                {{{', '.join(str(s) for s in desc.shape)}}},
                torch::TensorOptions()
                    .dtype(torch::{typeclass_to_torch_cpp_type(desc.dtype)})
                    .device(torch::{'kCUDA' if is_cuda(desc.storage) else 'kCPU'})
                    .layout(torch::kStrided));
            """


def initialize_outputs_code(module: 'dace.frontend.python.DaceModule', output_names: List[str],
                            clean_weights: Dict[str, torch.Tensor]) -> str:
    """Generate the code that initializes the output tensors.

    :param module: The module
    :param output_names: The output names of the SDFG.
    :param clean_weights: Dictionary of constant weights
    :return: The code
    """
    arglist = module.sdfg.arglist()
    code = ""
    for name in sorted(output_names):
        code += tensor_init_for_desc(name, arglist[name], clean_weights)

    return code


def argument_codegen(sdfg: dace.SDFG,
                     clean_weights: Dict[str, torch.Tensor],
                     input_names: List[str],
                     output_names: List[str],
                     guard_contiguous: Optional[List[str]] = None) -> Tuple[str, str, str]:
    """Generate the code that grabs the pointers of inputs and outputs.

    The names of the tensors will match the SDFG tensor names. Tensors that are not created by us (i.e. inputs)
    should be named {sdfg_name}_ first, and then .contiguous() will be called on them to yield the tensor that we
    require. This is the case for all tensors in ``guard_contiguous``.

    :param sdfg: The SDFG to generate code for
    :param clean_weights: The constant weights of the SDFG.
    :param input_names: Names of inputs to the torch function.
    :param output_names: Names of outputs to the torch function.
    :param guard_contiguous: A subset of input_names to call .contiguous on. If None, all input names will be
                             guarded.
    :return: The code for initializing the argument, the SDFG arguments in order, and the init call arguments
    """
    arglist = sdfg.arglist()

    guard_contiguous = set(guard_contiguous or input_names)

    assert set(input_names).issubset(arglist.keys()), \
        f"Input names {set(input_names).difference(arglist.keys())} are not SDFG arguments {arglist.keys()}"

    # Initialize the inputs and outputs
    ptr_init_code = "\n// Setup input and output pointers\n"
    for name in sorted(input_names):
        tctype = torch_ctype(arglist[name].dtype)
        dctype = arglist[name].dtype

        if isinstance(arglist[name], data.Array) or dt.can_access(dt.ScheduleType.GPU_Device, arglist[name].storage):
            if name in guard_contiguous:
                if logging.root.level <= logging.DEBUG:
                    ptr_init_code += f"""
                    if (!{name}_.is_contiguous()) {{
                        fprintf(stderr, "{name} was not contiguous!");
                    }}
                    """
                ptr_init_code += '\n' + f"Tensor {name} = {name}_.contiguous();"

            ptr_init_code += '\n' + f"{dctype} *{name}_ptr = reinterpret_cast<{dctype}*>({name}.data_ptr<{tctype}>());"

        elif isinstance(arglist[name], data.Scalar):
            if name in guard_contiguous:
                ptr_init_code += '\n' + f"{dctype} {name}_ptr = static_cast<{dctype}>({name}_.item().to<{tctype}>());"
            else:
                ptr_init_code += '\n' + f"{dctype} {name}_ptr = static_cast<{dctype}>({name}.item().to<{tctype}>());"
        else:
            raise ValueError(f"Unsupported data type {type(arglist[name])} for descriptor {name}")

    ptr_init_code += '\n'

    # Outputs and backward arrays
    ptr_init_code += '\n'.join(
        f"{arglist[name].dtype.ctype} *{name}_ptr = reinterpret_cast<{arglist[name].dtype.ctype}*>"
        f"({name}.data_ptr<{torch_ctype(arglist[name].dtype)}>());" for name in sorted(output_names))
    ptr_init_code += "\n// Setup constant arguments\n"

    all_access_nodes = set()
    for state in sdfg.nodes():
        all_access_nodes |= set(n.data for n in state.data_nodes())

    # Initialize all remaining parameters
    remaining = set(arglist).difference(itertools.chain(input_names, output_names))
    for name in sorted(remaining):
        # Remaining args must be constants
        if name not in clean_weights:
            raise ValueError(f"Cannot generate PyTorch module C++ code: SDFG argument {name} is not an input or output"
                             f" of the PyTorch Module, and not a constant.")

        value = clean_weights[name]
        ptr_init_code += f"{constant_initializer_code(name, arglist[name], value)}\n"

    arguments = ", ".join(f"{n}_ptr" for n in arglist)
    init_arguments = ", ".join(f"{n}_ptr" for n, desc in arglist.items() if isinstance(desc, data.Scalar))

    return ptr_init_code, arguments, init_arguments


def item_to_cpp_literal(item) -> str:
    """Convert a numpy item to a C++ literal string.

    Args:
        item: The numpy item to convert

    Returns:
        The C++ literal representation as a string
    """
    dtype = str(item.dtype)
    if np.isneginf(item):
        return "-std::numeric_limits<float>::infinity()"
    if np.isposinf(item):
        return "std::numeric_limits<float>::infinity()"
    if dtype == "float32":
        return f"{item}f"
    elif dtype == "bool":
        return f"{str(item).lower()}"
    elif dtype == "int64":
        return f"{item}l"
    elif dtype == "float16":
        ctype = dace.dtypes._CTYPES[item.dtype.type]
        return f"(({ctype}){item})"
    elif dtype in ["float64", "int32", "int16", "int8"]:
        return str(item)
    else:
        raise ValueError(f"Unsupported tensor type {item.dtype}")


def constant_initializer_code(name: str, desc: data.Data, value) -> str:
    """Generate C++ code for initializing a constant value.

    Args:
        name: The name of the constant
        desc: The data descriptor
        value: The constant value

    Returns:
        C++ code string for constant initialization
    """
    gpu_storage = dt.can_access(dt.ScheduleType.GPU_Device, desc.storage)
    gpu_storage = False
    if desc.total_size == 0:
        return f"{desc.dtype.ctype} *{name}_ptr = nullptr;"
    elif isinstance(desc, data.Array) or gpu_storage:
        numpyval = value.cpu().numpy()
        if len(numpyval.shape) == 0:
            numpyval = numpyval.reshape((1, ))
        iterator = np.nditer(numpyval, order="C")
        gpu_copy_code = f"""
        Tensor {name} = torch::from_blob({name}_ptr_cpu, {{{', '.join(sym2cpp(s) for s in desc.shape)}}},
            {{{', '.join(sym2cpp(s) for s in desc.strides)}}}, torch::{typeclass_to_torch_cpp_type(desc.dtype)})
            .to(torch::kCUDA);
        {desc.dtype.ctype} *{name}_ptr = reinterpret_cast<{desc.dtype.ctype}*>({name}.data_ptr<{torch_ctype(desc.dtype)}>());
        """
        return f"""
        {desc.dtype.ctype} {name}_ptr{'_cpu' if gpu_storage else ''}[{sym2cpp(desc.total_size)}] =
            {{{', '.join(item_to_cpp_literal(e) for e in iterator)}}};
        {gpu_copy_code if gpu_storage else ""}
        """
    elif isinstance(desc, data.Scalar):
        if str(value.item()) == "-inf":
            return f"{desc.dtype.ctype} {name}_ptr = -std::numeric_limits<{desc.dtype.ctype}>::infinity();"
        elif str(value.item()) == "inf":
            return f"{desc.dtype.ctype} {name}_ptr = std::numeric_limits<{desc.dtype.ctype}>::infinity();"
        if desc.dtype.ctype == "bool":
            # Special case for bools
            bool_str = "true" if value.item() else "false"
            return f"{desc.dtype.ctype} {name}_ptr = {bool_str};"
        return f"{desc.dtype.ctype} {name}_ptr = {str(value.item())};"
    else:
        raise ValueError("Unsupported data descriptor")


def return_type_str(outputs: List[str]) -> str:
    """Generate the return type string for the given outputs.

    Args:
        outputs: List of output names

    Returns:
        The C++ return type string
    """
    return f"""{"Tensor" if len(outputs) == 1 else f"variable_list"}"""


def save_non_inputs_outputs(names: List[str]):
    """Generate code to save non-input/output tensors for backward pass.

    Args:
        names: List of tensor names to save

    Returns:
        C++ code string for saving tensors
    """
    return "\n".join(f'ctx->saved_data["{n}"] = {n};' for n in names)


def recover_saved_inputs_outputs(saved_inputs_outputs: List[str], other_saved: List[str]):
    """Generate code to recover saved tensors in backward pass.

    Args:
        saved_inputs_outputs: List of saved input/output tensor names
        other_saved: List of other saved tensor names

    Returns:
        C++ code string for recovering saved tensors
    """
    code = ""
    if saved_inputs_outputs:
        code += "auto saved = ctx->get_saved_variables();\n"
        for i, n in enumerate(saved_inputs_outputs):
            code += f"\nauto {n} = saved[{i}];"

    for n in other_saved:
        code += f'\nauto {n} = ctx->saved_data["{n}"].toTensor();'

    return code


def setup_grad_values(backward_result: BackwardResult, sdfg: dace.SDFG, outputs: List[str],
                      clean_weights: Dict[str, torch.Tensor]) -> str:
    """Generate code to setup gradient values for backward pass.

    Args:
        backward_result: The backward pass result containing gradient information
        sdfg: The SDFG
        outputs: List of output names
        clean_weights: Dictionary of constant weights

    Returns:
        C++ code string for gradient setup
    """
    code = "// input grads"
    for param_name, grad_name in sorted(backward_result.required_grad_names.items()):
        zero_init = backward_result.zero_init.get(param_name, True)
        code += "\n" + tensor_init_for_desc(grad_name, sdfg.arrays[grad_name], clean_weights, zeros=zero_init)

    code += "// output grads"
    for i, o in enumerate(outputs):
        grad_name = backward_result.given_grad_names[o]
        code += f'\nauto {grad_name}_ = grad_outputs[{i}];'

    return code


def code_for_backward_function(module: 'dace.frontend.python.module.DaceModule', forward_sdfg: dace.SDFG,
                               backward_sdfg: dace.SDFG, backward_result: BackwardResult,
                               forwarded_arrays: Dict[str, data.Data]) -> str:
    """Generate C++ code for a differentiable PyTorch function.

    Args:
        module: The DaCe module
        forward_sdfg: The forward SDFG
        backward_sdfg: The backward SDFG
        backward_result: The backward pass result
        forwarded_arrays: Arrays forwarded from forward to backward pass

    Returns:
        Complete C++ code string for the differentiable function
    """

    inputs, outputs = get_arglist(module)
    sdfg_name = forward_sdfg.name

    ret_str = return_type_str(outputs)

    outputs_with_forwarded_outputs = copy.deepcopy(outputs)
    outputs_with_forwarded_outputs.extend(n for n in forwarded_arrays if n not in inputs and n not in outputs)

    fwd_ptr_init_code, fwd_sdfg_call_arguments, _ = argument_codegen(forward_sdfg, module.dace_model.clean_weights,
                                                                     inputs, outputs_with_forwarded_outputs)

    # Inputs are given_grads + forwarded_outputs
    bwd_inputs = list(backward_result.given_grad_names.values()) + list(forwarded_arrays)

    # Outputs are required grads
    bwd_outputs = list(backward_result.required_grad_names.values())

    bwd_ptr_init_code, bwd_sdfg_call_arguments, _ = argument_codegen(backward_sdfg,
                                                                     module.dace_model.clean_weights,
                                                                     bwd_inputs,
                                                                     bwd_outputs,
                                                                     guard_contiguous=list(
                                                                         backward_result.given_grad_names.values()))

    # Saved inputs/outputs
    saved_io_for_backward = [n for n in forwarded_arrays if n in inputs or n in outputs]
    other_saved_for_backward = [n for n in forwarded_arrays if n not in inputs and n not in outputs]
    return f"""
{get_header(forward_sdfg, backward_sdfg, inputs, outputs, module.use_cuda)}
class {sdfg_name}Function : public torch::autograd::Function<{sdfg_name}Function> {{
    public:
        static
            {ret_str}
            forward(
            AutogradContext *ctx,
            int64_t fwd_handle_ptr, int64_t bwd_handle_ptr, {", ".join(f"const Tensor& {name}_" for name in inputs)}) {{

            at::AutoDispatchBelowADInplaceOrView g;

            // initialize outputs
            {initialize_outputs_code(module, outputs_with_forwarded_outputs, module.dace_model.clean_weights)}

            {fwd_ptr_init_code}

            // get SDFG state handle
            {forward_sdfg.name}Handle_t handle = reinterpret_cast<{forward_sdfg.name}Handle_t>(fwd_handle_ptr);


            // call SDFG
            __program_{forward_sdfg.name}(handle, {fwd_sdfg_call_arguments});

            // save inputs/outputs for backward
            {
                f"ctx->save_for_backward({{{', '.join(f'{n}' for n in saved_io_for_backward)}}});"
                if saved_io_for_backward else ""
            }

            // save non-inputs/outputs
            {save_non_inputs_outputs(other_saved_for_backward)}

            // save bwd handle
            ctx->saved_data["bwd_handle"] = bwd_handle_ptr;

            // return to torch
            return {f"{outputs[0]}" if len(outputs) == 1
            else f"{{{', '.join(o for o in outputs)}}}"};
        }}

        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {{
            // recover bwd_handle_ptr
            int64_t bwd_handle_ptr = ctx->saved_data.find("bwd_handle")->second.toInt();

            // recover saved values
            {recover_saved_inputs_outputs(saved_io_for_backward, other_saved_for_backward)}

            // create grad values
            // NOTE, it might make sense take these from .grad()
            {setup_grad_values(backward_result, backward_sdfg, outputs, module.dace_model.clean_weights)}

            {bwd_ptr_init_code}

            // get SDFG state handle
            {backward_sdfg.name}Handle_t handle = reinterpret_cast<{backward_sdfg.name}Handle_t>(bwd_handle_ptr);

            // call bwd SDFG
            __program_{backward_sdfg.name}(handle, {bwd_sdfg_call_arguments});

            // return calculated grads in correct order
            // first two grads are None (these are the grads for the handle ptrs)
            return {{
                Tensor(), Tensor(), {', '.join(backward_result.required_grad_names[i] if i in backward_result.required_grad_names else 'Tensor()' for i in inputs )}
    }};
}}
}};

{ret_str}
{sdfg_name}_autograd(int64_t handle_ptr, int64_t bwd_handle_ptr, {",".join(f"const Tensor& {name}_" for name in inputs)}) {{
return {sdfg_name}Function::apply(
handle_ptr, bwd_handle_ptr, {", ".join(f"{name}_" for name in inputs)}
);
}}

TORCH_LIBRARY_IMPL(dace_{sdfg_name}, Autograd{'CUDA' if module.use_cuda else 'CPU'}, m) {{
m.impl("{sdfg_name}", {sdfg_name}_autograd);
}}
"""


def code_for_module(module: 'dace.frontend.python.module.DaceModule', compiled_sdfg: CompiledSDFG) -> str:
    """Generate the code for an operator that calls the SDFGs in the module.

    :param module: The module.
    :param compiled_sdfg: The compiled SDFG.
    """

    inputs, outputs = get_arglist(module)
    sdfg_name = compiled_sdfg.sdfg.name

    ret_str = return_type_str(outputs)
    ptr_init_code, sdfg_call_arguments, init_arguments = argument_codegen(compiled_sdfg.sdfg,
                                                                          module.dace_model.clean_weights, inputs,
                                                                          outputs)
    return f"""
{get_header(compiled_sdfg.sdfg, None, inputs, outputs, module.use_cuda)}

// Function definition
{ret_str}
{sdfg_name}(int64_t handle_ptr, {",".join(f"const Tensor& {name}_" for name in inputs)}) {{

    // Initialize outputs
    {initialize_outputs_code(module, outputs, module.dace_model.clean_weights)}

    {ptr_init_code}

    // Get SDFG state handle
    {sdfg_name}Handle_t handle = reinterpret_cast<{sdfg_name}Handle_t>(handle_ptr);

    // Call SDFG
    __program_{sdfg_name}(handle, {sdfg_call_arguments});

    // Return to torch
    return {f"{outputs[0]}" if len(outputs) == 1
        else f"{{{', '.join(o for o in outputs)}}}"};
}}

TORCH_LIBRARY_IMPL(dace_{sdfg_name}, {'CUDA' if module.use_cuda else 'CPU'}, m) {{
    m.impl("{sdfg_name}", {sdfg_name});
}}
        """


def get_header(fwd_sdfg: dace.SDFG, bwd_sdfg: Optional[dace.SDFG], inputs, outputs, use_cuda: bool) -> str:
    """Generate the C++ header code for the PyTorch extension.

    Args:
        fwd_sdfg: The forward SDFG
        bwd_sdfg: The backward SDFG (optional)
        inputs: List of input names
        outputs: List of output names
        use_cuda: Whether CUDA is used

    Returns:
        C++ header code string
    """
    return f"""
#include <torch/torch.h>
#include <torch/script.h>
#include "{fwd_sdfg.name}.h"
{"" if bwd_sdfg is None else f'#include "{bwd_sdfg.name}.h"'}
using torch::Tensor;
using torch::DeviceType;
using torch::autograd::tensor_list;
using torch::autograd::variable_list;
using torch::autograd::AutogradContext;

TORCH_LIBRARY(dace_{fwd_sdfg.name}, m) {{
    m.def("{fwd_sdfg.name}(int handle_ptr,{"int bwd_handle_ptr," if bwd_sdfg else ""} {", ".join('Tensor ' + arg for arg in inputs)}) -> {'Tensor' if len(outputs) == 1 else 'Tensor[]'}");
}}
"""


def register_and_compile_torch_extension(module: 'dace.frontend.python.module.DaceModule',
                                         dummy_inputs) -> DaCeMLTorchFunction:
    """Get a torch callable for the module. This will compile the SDFG, compile a PyTorch C++ operator, register it
    with PyTorch and return the function that calls it.

    This function handles code generation for both the forward and backward pass.

    :param module: The module.
    :param dummy_inputs: Dummy inputs to initialize the model with.
    :return: The callable function for the SDFG.
    """

    # Build the SDFG
    # Set all states to not-sync
    for state in module.sdfg.nodes():
        state.nosync = True

    environments = {
        PyTorch.full_class_path(),
    }
    if module.backward:
        compiled, handle_ptr, compiled_bwd, bwd_handle_ptr = compile_and_init_sdfgs(module, dummy_inputs)

        compiled_sdfgs = [compiled, compiled_bwd]
        ptrs = [handle_ptr, bwd_handle_ptr]
        if compiled_bwd is not None:
            environments.add(get_env_for_sdfg(compiled_bwd).full_class_path())
            bwd_sdfg = compiled_bwd.sdfg
            code = code_for_backward_function(module, compiled.sdfg, bwd_sdfg, module._ad_result, module._ad_inp_arrs)
        else:
            bwd_sdfg = module.backward_sdfg
            compiled_sdfgs = [compiled]
            ptrs = [handle_ptr]
            code = code_for_module(module, compiled)
    else:
        compiled, handle_ptr = compile_and_init_sdfgs(module, dummy_inputs)
        ptrs = [handle_ptr]
        code = code_for_module(module, compiled)
        compiled_sdfgs = [compiled]
    environments.add(get_env_for_sdfg(compiled).full_class_path())
    code = indent_code(code)

    # Build the PyTorch module
    libname = f"torch_{compiled.sdfg.name}"
    program = CodeObject(libname,
                         code,
                         "cpp",
                         targets.cpu.CPUCodeGen,
                         f"Torch{module.sdfg_name}",
                         environments=environments)
    torch_module_build_path = os.path.join('.dacecache', f"torch_{compiled.sdfg.name}")

    parts = os.path.normpath(compiled.filename).split(os.sep)
    sdfg_folder_name = parts[parts.index('.dacecache') + 1]
    backward_sdfg_folder_name = f"{compiled.sdfg.name}_backward_{sdfg_folder_name.removeprefix(compiled.sdfg.name + '_')}"
    compiler.generate_program_folder(None, [program], torch_module_build_path)
    include_path = os.path.abspath(os.path.join('.dacecache', sdfg_folder_name, "include"))
    include_path_bwd = os.path.abspath(os.path.join('.dacecache', backward_sdfg_folder_name, "include"))
    dace_include_path = os.path.abspath(os.path.join(os.path.dirname(dace.__file__), "runtime", "include"))
    code_path = os.path.join('.dacecache', sdfg_folder_name, "src", "cpu", f"{compiled.sdfg.name}.cpp")
    code_path_bwd = os.path.join('.dacecache', backward_sdfg_folder_name, "src", "cpu",
                                 f"{compiled.sdfg.name}_backward.cpp")
    torch_code_path = os.path.join('.dacecache', f"torch_{compiled.sdfg.name}", "src", "cpu",
                                   f"torch_{compiled.sdfg.name}.cpp")
    sources = [code_path, torch_code_path]
    if os.path.exists(code_path_bwd):
        sources.append(code_path_bwd)
    dace_include_path_onnx = os.path.abspath(
        os.path.join(os.path.dirname(dace.__file__), "libraries", "onnx", "include"))
    dace_include_path_blas = os.path.abspath(
        os.path.join(os.path.dirname(dace.__file__), "libraries", "blas", "include"))
    conda_lib_path = os.path.abspath(os.getenv("CONDA_PREFIX") + "/lib")
    torch.utils.cpp_extension.load(
        name=libname,
        sources=sources,
        extra_cflags=["-g"],
        extra_include_paths=[
            include_path,
            include_path_bwd,
            dace_include_path,
            dace_include_path_blas,
            dace_include_path_onnx,
        ],
        extra_ldflags=[
            f'-L{conda_lib_path}',
        ],
        is_python_module=False,
    )

    torch_function = operator.attrgetter(f"dace_{compiled.sdfg.name}.{compiled.sdfg.name}")(torch.ops)

    result = DaCeMLTorchFunction(function=torch_function, compiled_sdfgs=compiled_sdfgs, ptr=ptrs)
    return result


def get_env_for_sdfg(compiled: CompiledSDFG):
    """Create an environment for the given compiled SDFG.

    Args:
        compiled: The compiled SDFG

    Returns:
        The environment class for the SDFG
    """
    sdfg_build_path = os.path.abspath(compiled.sdfg.build_folder)

    class SDFGEnvironment:
        """Environment for the SDFG."""

        cmake_minimum_version = None
        cmake_packages = []
        cmake_variables = {}
        cmake_includes = [os.path.join(sdfg_build_path, "include")]
        cmake_compile_flags = []
        cmake_link_flags = []
        cmake_files = []
        cmake_libraries = [os.path.join(sdfg_build_path, "build", platform_library_name(compiled.sdfg.name))]
        state_fields = []
        dependencies = []
        headers = []
        init_code = ""
        finalize_code = ""

    SDFGEnvironment.__name__ = compiled.sdfg.name
    dace.library.environment(SDFGEnvironment)
    return SDFGEnvironment


def indent_code(code: str) -> str:
    """Indent the given code string properly.

    Args:
        code: The code string to indent

    Returns:
        The indented code string
    """
    stream = CodeIOStream()
    stream.write(code)
    return stream.getvalue()
