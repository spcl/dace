# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import functools
import os
from typing import List

from dace import dtypes
from dace import data
from dace.codegen.targets import framecode, target
from dace.codegen.codeobject import CodeObject
from dace.config import Config
from dace.sdfg.infer_types import infer_connector_types

# Import CPU code generator. TODO: Remove when refactored
from dace.codegen.targets import cpp, cpu

from dace.codegen.instrumentation import InstrumentationProvider


class CodegenError(Exception):
    pass


def generate_headers(sdfg) -> str:
    """ Generate a header file for the SDFG """
    proto = ""
    params = (sdfg.name, sdfg.signature(with_types=True, for_call=False))
    proto += 'extern "C" int __dace_init_%s(%s);\n' % params
    proto += 'extern "C" int __dace_exit_%s(%s);\n' % params
    proto += 'extern "C" void __program_%s(%s);\n' % params
    return proto


def generate_dummy(sdfg) -> str:
    """ Generates a C program calling this SDFG. Since we do not
        know the purpose/semantics of the program, we allocate
        the right types and and guess values for scalars.
    """
    includes = "#include <stdlib.h>\n"
    includes += "#include \"" + sdfg.name + ".h\"\n\n"
    header = "int main(int argc, char** argv) {\n"
    allocations = ""
    deallocations = ""
    sdfg_call = ""
    footer = "  return 0;\n}\n"

    al = sdfg.arglist()

    # first find all scalars and set them to 42
    for argname, arg in al.items():
        if isinstance(arg, data.Scalar):
            allocations += "  " + str(arg.as_arg(name=argname,
                                                 with_types=True)) + " = 42;\n"

    # allocate the array args using calloc
    for argname, arg in al.items():
        if isinstance(arg, data.Array):
            dims_mul = cpp.sym2cpp(
                functools.reduce(lambda a, b: a * b, arg.shape, 1))
            basetype = str(arg.dtype)
            allocations += "  " + str(arg.as_arg(name=argname, with_types=True)) + \
                           " = (" + basetype + "*) calloc(" + dims_mul + ", sizeof("+ basetype +")" + ");\n"
            deallocations += "  free(" + argname + ");\n"

    sdfg_call = '''
  __dace_init_{name}({params});
  __program_{name}({params});
  __dace_exit_{name}({params});\n\n'''.format(name=sdfg.name,
                                              params=sdfg.signature(
                                                  with_types=False,
                                                  for_call=True))

    res = includes
    res += header
    res += allocations
    res += sdfg_call
    res += deallocations
    res += footer
    return res


def generate_code(sdfg) -> List[CodeObject]:
    """ Generates code as a list of code objects for a given SDFG.
        :param sdfg: The SDFG to use
        :return: List of code objects that correspond to files to compile.
    """
    # Before compiling, validate SDFG correctness
    sdfg.validate()

    if Config.get_bool('testing', 'serialization'):
        from dace.sdfg import SDFG
        import filecmp
        sdfg.save('test.sdfg')
        sdfg2 = SDFG.from_file('test.sdfg')
        sdfg2.save('test2.sdfg')
        print('Testing SDFG serialization...')
        if not filecmp.cmp('test.sdfg', 'test2.sdfg'):
            raise RuntimeError('SDFG serialization failed - files do not match')
        os.remove('test.sdfg')
        os.remove('test2.sdfg')

        # Run with the deserialized version
        sdfg = sdfg2

    # Before generating the code, run type inference on the SDFG connectors
    infer_connector_types(sdfg)

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
        for k, v in target.TargetCodeGenerator.extensions().items()
        if v['name'] not in targets
    })

    # Instantiate all instrumentation providers in SDFG
    provider_mapping = InstrumentationProvider.get_provider_mapping()
    frame._dispatcher.instrumentation[
        dtypes.InstrumentationType.No_Instrumentation] = None
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
    frame._dispatcher.instrumentation = {
        k: v() if v is not None else None
        for k, v in frame._dispatcher.instrumentation.items()
    }

    # Generate frame code (and the rest of the code)
    (global_code, frame_code, used_targets,
     used_environments) = frame.generate_code(sdfg, None)
    target_objects = [
        CodeObject(sdfg.name,
                   global_code + frame_code,
                   'cpp',
                   cpu.CPUCodeGen,
                   'Frame',
                   environments=used_environments)
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
                       linkable=False)
    target_objects.append(dummy)

    # add a dummy main function to show how to call the SDFG
    dummy = CodeObject(sdfg.name + "_main",
                       generate_dummy(sdfg),
                       'cpp',
                       cpu.CPUCodeGen,
                       'DummyMain',
                       linkable=False)
    target_objects.append(dummy)

    return target_objects


##################################################################
