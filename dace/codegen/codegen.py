# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import functools
import os
from typing import List, Set

import dace
from dace import dtypes
from dace import data
from dace.sdfg import SDFG
from dace.codegen.targets import framecode
from dace.codegen.codeobject import CodeObject
from dace.config import Config
from dace.sdfg import infer_types

# Import CPU code generator. TODO: Remove when refactored
from dace.codegen.targets import cpp, cpu

from dace.codegen.instrumentation import InstrumentationProvider
from dace.sdfg.state import SDFGState


def generate_headers(sdfg: SDFG, frame: framecode.DaCeCodeGenerator) -> str:
    """ Generate a header file for the SDFG """
    proto = ""
    proto += "#include <dace/dace.h>\n"
    init_params = (sdfg.name, sdfg.name, sdfg.init_signature(free_symbols=frame.free_symbols(sdfg)))
    call_params = sdfg.signature(with_types=True, for_call=False, arglist=frame.arglist)
    if len(call_params) > 0:
        call_params = ', ' + call_params
    params = (sdfg.name, sdfg.name, call_params)
    exit_params = (sdfg.name, sdfg.name)
    proto += 'typedef void * %sHandle_t;\n' % sdfg.name
    proto += 'extern "C" %sHandle_t __dace_init_%s(%s);\n' % init_params
    proto += 'extern "C" void __dace_exit_%s(%sHandle_t handle);\n' % exit_params
    proto += 'extern "C" void __program_%s(%sHandle_t handle%s);\n' % params
    return proto


def generate_dummy(sdfg: SDFG, frame: framecode.DaCeCodeGenerator) -> str:
    """ Generates a C program calling this SDFG. Since we do not
        know the purpose/semantics of the program, we allocate
        the right types and and guess values for scalars.
    """
    al = frame.arglist
    init_params = sdfg.init_signature(for_call=True, free_symbols=frame.free_symbols(sdfg))
    params = sdfg.signature(with_types=False, for_call=True, arglist=frame.arglist)
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


def _get_codegen_targets(sdfg: SDFG, frame: framecode.DaCeCodeGenerator):
    """
    Queries all code generation targets in this SDFG and all nested SDFGs,
    as well as instrumentation providers, and stores them in the frame code generator.
    """
    disp = frame._dispatcher
    provider_mapping = InstrumentationProvider.get_provider_mapping()
    disp.instrumentation[dtypes.InstrumentationType.No_Instrumentation] = None
    for node, parent in sdfg.all_nodes_recursive():
        # Query nodes and scopes
        if isinstance(node, SDFGState):
            frame.targets.add(disp.get_state_dispatcher(parent, node))
        elif isinstance(node, dace.nodes.EntryNode):
            frame.targets.add(disp.get_scope_dispatcher(node.schedule))
        elif isinstance(node, dace.nodes.Node):
            state: SDFGState = parent
            nsdfg = state.parent
            frame.targets.add(disp.get_node_dispatcher(nsdfg, state, node))

        # Array allocation
        if isinstance(node, dace.nodes.AccessNode):
            state: SDFGState = parent
            nsdfg = state.parent
            desc = node.desc(nsdfg)
            frame.targets.add(disp.get_array_dispatcher(desc.storage))

        # Copies and memlets - via access nodes and tasklets
        # To avoid duplicate checks, only look at outgoing edges of access nodes and tasklets
        if isinstance(node, (dace.nodes.AccessNode, dace.nodes.Tasklet)):
            state: SDFGState = parent
            for e in state.out_edges(node):
                if e.data.is_empty():
                    continue
                mtree = state.memlet_tree(e)
                if mtree.downwards:
                    # Rooted at src_node
                    for leaf_e in mtree.leaves():
                        dst_node = leaf_e.dst
                        if leaf_e.data.is_empty():
                            continue
                        tgt = disp.get_copy_dispatcher(node, dst_node, leaf_e, state.parent, state)
                        if tgt is not None:
                            frame.targets.add(tgt)
                else:
                    # Rooted at dst_node
                    dst_node = mtree.root().edge.dst
                    tgt = disp.get_copy_dispatcher(node, dst_node, e, state.parent, state)
                    if tgt is not None:
                        frame.targets.add(tgt)

        # Instrumentation-related query
        if hasattr(node, 'instrument'):
            disp.instrumentation[node.instrument] = provider_mapping[node.instrument]
        elif hasattr(node, 'consume'):
            disp.instrumentation[node.consume.instrument] = provider_mapping[node.consume.instrument]
        elif hasattr(node, 'map'):
            disp.instrumentation[node.map.instrument] = provider_mapping[node.map.instrument]

    # Query instrumentation provider of SDFG
    if sdfg.instrument != dtypes.InstrumentationType.No_Instrumentation:
        disp.instrumentation[sdfg.instrument] = provider_mapping[sdfg.instrument]


def generate_code(sdfg, validate=True) -> List[CodeObject]:
    """ Generates code as a list of code objects for a given SDFG.
        :param sdfg: The SDFG to use
        :param validate: If True, validates the SDFG before generating the code.
        :return: List of code objects that correspond to files to compile.
    """
    from dace.codegen.targets.target import TargetCodeGenerator  # Avoid import loop

    # Before compiling, validate SDFG correctness
    if validate:
        sdfg.validate()

    if Config.get_bool('testing', 'serialization'):
        from dace.sdfg import SDFG
        import filecmp
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            sdfg.simplify_expr()
            sdfg.save(f'{tmp_dir}/test.sdfg')
            sdfg2 = SDFG.from_file(f'{tmp_dir}/test.sdfg')
            sdfg2.simplify_expr()
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

    frame = framecode.DaCeCodeGenerator(sdfg)

    # Instantiate CPU first (as it is used by the other code generators)
    # TODO: Refactor the parts used by other code generators out of CPU
    default_target = cpu.CPUCodeGen
    for k, v in TargetCodeGenerator.extensions().items():
        # If another target has already been registered as CPU, use it instead
        if v['name'] == 'cpu':
            default_target = k
    targets = {'cpu': default_target(frame, sdfg)}

    # Instantiate the rest of the targets
    targets.update(
        {v['name']: k(frame, sdfg)
         for k, v in TargetCodeGenerator.extensions().items() if v['name'] not in targets})

    # Query all code generation targets and instrumentation providers in SDFG
    _get_codegen_targets(sdfg, frame)

    # Preprocess SDFG
    for target in frame.targets:
        target.preprocess(sdfg)

    # Instantiate instrumentation providers
    frame._dispatcher.instrumentation = {
        k: v() if v is not None else None
        for k, v in frame._dispatcher.instrumentation.items()
    }

    # NOTE: THE SDFG IS ASSUMED TO BE FROZEN (not change) FROM THIS POINT ONWARDS

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

    # Ensure that no new targets were dynamically added
    assert frame._dispatcher.used_targets == (frame.targets - {frame})

    # add a header file for calling the SDFG
    dummy = CodeObject(sdfg.name,
                       generate_headers(sdfg, frame),
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
                       generate_dummy(sdfg, frame),
                       'cpp',
                       cpu.CPUCodeGen,
                       'SampleMain',
                       target_type='../../sample',
                       linkable=False)
    target_objects.append(dummy)

    return target_objects


##################################################################
