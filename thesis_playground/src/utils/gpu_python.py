"""
General helper functions to deal with running python DaCe programs in GPU
"""
from typing import Dict
import dace

from utils.gpu_general import copy_to_device
from utils.python import gen_arguments, generate_sdfg


def run_function_dace(f: dace.frontend.python.parser.DaceProgram, symbols: Dict[str, int], save_graphs: bool = False,
                      define_symbols: bool = False) -> Dict:

    csdfg = generate_sdfg(f, symbols, save_graphs, define_symbols).compile()
    arguments = gen_arguments(f, symbols)
    arguments_device = copy_to_device(arguments)
    csdfg(**arguments_device, **symbols)
    return arguments_device
