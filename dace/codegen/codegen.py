# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import functools
import os
from typing import List

import dace
from dace import dtypes
from dace import data
from dace.sdfg import SDFG
from dace.codegen.targets import framecode, target
from dace.codegen.codeobject import CodeObject
from dace.config import Config
from dace.sdfg import infer_types

# Import CPU code generator. TODO: Remove when refactored
from dace.codegen.targets import cpp, cpu

from dace.codegen.instrumentation import InstrumentationProvider


def generate_headers(sdfg: SDFG) -> str:
    """ Generate a header file for the SDFG """
    proto = ""
    proto += "#include <dace/dace.h>\n"
    init_params = (sdfg.name, sdfg.name, sdfg.signature(with_types=True, for_call=False, with_arrays=False))
    call_params = sdfg.signature(with_types=True, for_call=False)
    if len(call_params) > 0:
        call_params = ', ' + call_params
    params = (sdfg.name, sdfg.name, call_params)
    exit_params = (sdfg.name, sdfg.name)
    proto += 'typedef void * %sHandle_t;\n' % sdfg.name
    proto += 'extern "C" %sHandle_t __dace_init_%s(%s);\n' % init_params
    proto += 'extern "C" void __dace_exit_%s(%sHandle_t handle);\n' % exit_params
    proto += 'extern "C" void __program_%s(%sHandle_t handle%s);\n' % params
    return proto


def generate_dummy(sdfg: SDFG) -> str:
    """ Generates a C program calling this SDFG. Since we do not
        know the purpose/semantics of the program, we allocate
        the right types and and guess values for scalars.
    """
    al = sdfg.arglist()
    init_params = sdfg.signature(with_types=False, for_call=True, with_arrays=False)
    params = sdfg.signature(with_types=False, for_call=True)
    if len(params) > 0:
        params = ', ' + params

    allocations = ''
    deallocations = ''

    # first find all scalars and set them to 42
    for argname, arg in al.items():
        if isinstance(arg, data.Scalar):
            allocations += ("    " + str(arg.as_arg(name=argname, with_types=True)) + " = 42;\n")

    # allocate the array args using calloc
    for argname, arg in al.items():
        if isinstance(arg, data.Array):
            dims_mul = cpp.sym2cpp(functools.reduce(lambda a, b: a * b, arg.shape, 1))
            basetype = str(arg.dtype)
            allocations += ("    " + str(arg.as_arg(name=argname, with_types=True)) + " = (" + basetype + "*) calloc(" +
                            dims_mul + ", sizeof(" + basetype + ")" + ");\n")
            deallocations += "    free(" + argname + ");\n"

    return f'''#include <cstdlib>
#include "../include/{sdfg.name}.h"

int main(int argc, char **argv) {{
    {sdfg.name}Handle_t handle;
{allocations}

    handle = __dace_init_{sdfg.name}({init_params});
    __program_{sdfg.name}(handle{params});
    __dace_exit_{sdfg.name}(handle);

{deallocations}

    return 0;
}}
'''


def generate_code(sdfg, validate=True) -> List[CodeObject]:
    """ Generates code as a list of code objects for a given SDFG.
        :param sdfg: The SDFG to use
        :param validate: If True, validates the SDFG before generating the code.
        :return: List of code objects that correspond to files to compile.
    """
    # Before compiling, validate SDFG correctness
    if validate:
        sdfg.validate()

    if Config.get_bool('testing', 'serialization'):
        from dace.sdfg import SDFG
        import filecmp
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            sdfg.save(f'{tmp_dir}/test.sdfg')
            sdfg2 = SDFG.from_file(f'{tmp_dir}/test.sdfg')
            sdfg2.save(f'{tmp_dir}/test2.sdfg')
            print('Testing SDFG serialization...')
            if not filecmp.cmp(f'{tmp_dir}/test.sdfg', f'{tmp_dir}/test2.sdfg'):
                shutil.move(f"{tmp_dir}/test.sdfg", "test.sdfg")
                shutil.move(f"{tmp_dir}/test2.sdfg", "test2.sdfg")
                raise RuntimeError('SDFG serialization failed - files do not match')

        # Run with the deserialized version
        # NOTE: This means that all subsequent modifications to `sdfg`
        # are not reflected outside of this function (e.g., library
        # node expansion).
        sdfg = sdfg2

    # Before generating the code, run type inference on the SDFG connectors
    infer_types.infer_connector_types(sdfg)

    # Set default storage/schedule types in SDFG
    infer_types.set_default_schedule_and_storage_types(sdfg, None)

    # Recursively expand library nodes that have not yet been expanded
    sdfg.expand_library_nodes()

    # After expansion, run another pass of connector/type inference
    infer_types.infer_connector_types(sdfg)
    infer_types.set_default_schedule_and_storage_types(sdfg, None)

    frame = framecode.DaCeCodeGenerator()

    # Instantiate CPU first (as it is used by the other code generators)
    # TODO: Refactor the parts used by other code generators out of CPU
    default_target = cpu.CPUCodeGen
    for k, v in target.TargetCodeGenerator.extensions().items():
        # If another target has already been registered as CPU, use it instead
        if v['name'] == 'cpu':
            default_target = k
    targets = {'cpu': default_target(frame, sdfg)}

    # Instantiate the rest of the targets
    targets.update({
        v['name']: k(frame, sdfg)
        for k, v in target.TargetCodeGenerator.extensions().items() if v['name'] not in targets
    })

    # Instantiate all instrumentation providers in SDFG
    provider_mapping = InstrumentationProvider.get_provider_mapping()
    frame._dispatcher.instrumentation[dtypes.InstrumentationType.No_Instrumentation] = None
    for node, _ in sdfg.all_nodes_recursive():
        if hasattr(node, 'instrument'):
            frame._dispatcher.instrumentation[node.instrument] = \
                provider_mapping[node.instrument]
        elif hasattr(node, 'consume'):
            frame._dispatcher.instrumentation[node.consume.instrument] = \
                provider_mapping[node.consume.instrument]
        elif hasattr(node, 'map'):
            frame._dispatcher.instrumentation[node.map.instrument] = \
                provider_mapping[node.map.instrument]
    if sdfg.instrument != dtypes.InstrumentationType.No_Instrumentation:
        frame._dispatcher.instrumentation[sdfg.instrument] = \
            provider_mapping[sdfg.instrument]
    frame._dispatcher.instrumentation = {
        k: v() if v is not None else None
        for k, v in frame._dispatcher.instrumentation.items()
    }

    # Generate frame code (and the rest of the code)
    (global_code, frame_code, used_targets, used_environments) = frame.generate_code(sdfg, None)
    target_objects = [
        CodeObject(sdfg.name,
                   global_code + frame_code,
                   'cpp',
                   cpu.CPUCodeGen,
                   'Frame',
                   environments=used_environments,
                   sdfg=sdfg)
    ]

    # Create code objects for each target
    for tgt in used_targets:
        target_objects.extend(tgt.get_generated_codeobjects())

    # add a header file for calling the SDFG
    dummy = CodeObject(sdfg.name,
                       generate_headers(sdfg),
                       'h',
                       cpu.CPUCodeGen,
                       'CallHeader',
                       target_type='../../include',
                       linkable=False)
    target_objects.append(dummy)

    for env in dace.library.get_environments_and_dependencies(used_environments):
        if hasattr(env, "codeobjects"):
            target_objects.extend(env.codeobjects)

    # add a dummy main function to show how to call the SDFG
    dummy = CodeObject(sdfg.name + "_main",
                       generate_dummy(sdfg),
                       'cpp',
                       cpu.CPUCodeGen,
                       'SampleMain',
                       target_type='../../sample',
                       linkable=False)
    target_objects.append(dummy)

    return target_objects


##################################################################
