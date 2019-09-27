import numpy as np

from typing import List

from dace import dtypes
from dace.codegen.targets import framecode
from dace.codegen.codeobject import CodeObject

# Import all code generation targets
from dace.codegen.targets import cpu, cuda, immaterial, mpi, xilinx
from dace.codegen.instrumentation import INSTRUMENTATION_PROVIDERS


class CodegenError(Exception):
    pass


STRING_TO_TARGET = {
    "cpu": cpu.CPUCodeGen,
    "cuda": cuda.CUDACodeGen,
    "immaterial": immaterial.ImmaterialCodeGen,
    "mpi": mpi.MPICodeGen,
    "xilinx": xilinx.XilinxCodeGen,
}

_TARGET_REGISTER_ORDER = ['cpu', 'cuda', 'immaterial', 'mpi', 'xilinx']


def generate_code(sdfg) -> List[CodeObject]:
    """ Generates code as a list of code objects for a given SDFG.
        @param sdfg: The SDFG to use
        @return: List of code objects that correspond to files to compile.
    """
    # Before compiling, validate SDFG correctness
    sdfg.validate()

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

    return target_objects


##################################################################
