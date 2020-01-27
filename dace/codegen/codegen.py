import os
from typing import List

from dace import dtypes
from dace import data
from dace.codegen.targets import framecode
from dace.codegen.codeobject import CodeObject
from dace.config import Config

# Import all code generation targets
from dace.codegen.targets import cpu, cuda, immaterial, mpi, xilinx, intel_fpga
from dace.codegen.instrumentation import INSTRUMENTATION_PROVIDERS


class CodegenError(Exception):
    pass


STRING_TO_TARGET = {
    "cpu": cpu.CPUCodeGen,
    "cuda": cuda.CUDACodeGen,
    "immaterial": immaterial.ImmaterialCodeGen,
    "mpi": mpi.MPICodeGen,
    "intel_fpga": intel_fpga.IntelFPGACodeGen,
    "xilinx": xilinx.XilinxCodeGen,
}

_TARGET_REGISTER_ORDER = [
    'cpu', 'cuda', 'immaterial', 'mpi', 'intel_fpga', 'xilinx'
]


def generate_headers(sdfg) -> str:
    """ Generate a header file for the SDFG """
    proto = ""
    proto += "int __dace_init(" + sdfg.signature(
        with_types=True, for_call=False) + ");\n"
    proto += "int __dace_exit(" + sdfg.signature(
        with_types=True, for_call=False) + ");\n"
    proto += "void __program_" + sdfg.name + "(" + sdfg.signature(
        with_types=True, for_call=False) + ");\n\n"
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
    for arg in al:
        if isinstance(al[arg], data.Scalar):
            allocations += "  " + str(al[arg].signature(
                name=arg, with_types=True)) + " = 42;\n"

    # allocate the array args using calloc
    for arg in al:
        if isinstance(al[arg], data.Array):
            dims_mul = "*".join(map(str, al[arg].shape))
            basetype = str(al[arg].dtype)
            allocations += "  " + str(al[arg].signature(name=arg, with_types=True)) + \
                           " = (" + basetype + "*) calloc(" + dims_mul + ", sizeof("+ basetype +")" + ");\n"
            deallocations += "  free(" + str(arg) + ");\n"

    sdfg_call = "\n  __dace_init(" + sdfg.signature(
        with_types=False, for_call=True) + ");\n"
    sdfg_call += "  __program_" + sdfg.name + "(" + sdfg.signature(
        with_types=False, for_call=True) + ");\n"
    sdfg_call += "  __dace_exit(" + sdfg.signature(
        with_types=False, for_call=True) + ");\n\n"

    res = ""
    res += includes
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

    if Config.get_bool('experimental', 'test_serialization'):
        from dace.sdfg import SDFG
        import filecmp
        sdfg.save('test.sdfg')
        sdfg2 = SDFG.from_file('test.sdfg')
        sdfg2.save('test2.sdfg')
        print('Testing SDFG serialization...')
        if not filecmp.cmp('test.sdfg', 'test2.sdfg'):
            raise RuntimeError(
                'SDFG serialization failed - files do not match')
        os.remove('test.sdfg')
        os.remove('test2.sdfg')

        # Run with the deserialized version
        sdfg = sdfg2

    frame = framecode.DaCeCodeGenerator()
    # Instantiate all targets (who register themselves with framecodegen)
    targets = {
        name: STRING_TO_TARGET[name](frame, sdfg)
        for name in _TARGET_REGISTER_ORDER
    }

    # Instantiate all instrumentation providers in SDFG
    frame._dispatcher.instrumentation[
        dtypes.InstrumentationType.No_Instrumentation] = None
    for node, _ in sdfg.all_nodes_recursive():
        if hasattr(node, 'instrument'):
            frame._dispatcher.instrumentation[node.instrument] = \
                INSTRUMENTATION_PROVIDERS[node.instrument]
        elif hasattr(node, 'consume'):
            frame._dispatcher.instrumentation[node.consume.instrument] = \
                INSTRUMENTATION_PROVIDERS[node.consume.instrument]
        elif hasattr(node, 'map'):
            frame._dispatcher.instrumentation[node.map.instrument] = \
                INSTRUMENTATION_PROVIDERS[node.map.instrument]
    frame._dispatcher.instrumentation = {
        k: v() if v is not None else None
        for k, v in frame._dispatcher.instrumentation.items()
    }

    # Generate frame code (and the rest of the code)
    global_code, frame_code, used_targets = frame.generate_code(sdfg, None)
    target_objects = [
        CodeObject(sdfg.name, global_code + frame_code, 'cpp', cpu.CPUCodeGen,
                   'Frame')
    ]

    # Create code objects for each target
    for tgt in used_targets:
        target_objects.extend(tgt.get_generated_codeobjects())

    # add a header file for calling the SDFG
    dummy = CodeObject(
        sdfg.name,
        generate_headers(sdfg),
        'h',
        cpu.CPUCodeGen,
        'CallHeader',
        linkable=False)
    target_objects.append(dummy)

    # add a dummy main function to show how to call the SDFG
    dummy = CodeObject(
        sdfg.name + "_main",
        generate_dummy(sdfg),
        'cpp',
        cpu.CPUCodeGen,
        'DummyMain',
        linkable=False)
    target_objects.append(dummy)

    return target_objects


##################################################################
