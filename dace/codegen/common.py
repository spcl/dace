# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import ast
from copy import deepcopy
import ctypes.util
from dace import config, data, dtypes, sdfg as sd, symbolic
from dace.sdfg import SDFG
from dace.properties import CodeBlock
from dace.codegen import cppunparse
from functools import lru_cache
from io import StringIO
import os
import subprocess
from typing import List, Optional, Set, Union
import warnings


def find_incoming_edges(node, dfg):
    # If it's an entire SDFG, look in each state
    if isinstance(dfg, SDFG):
        result = []
        for state in dfg.nodes():
            result.extend(list(state.in_edges(node)))
        return result
    else:  # If it's one state
        return list(dfg.in_edges(node))


def find_outgoing_edges(node, dfg):
    # If it's an entire SDFG, look in each state
    if isinstance(dfg, SDFG):
        result = []
        for state in dfg.nodes():
            result.extend(list(state.out_edges(node)))
        return result
    else:  # If it's one state
        return list(dfg.out_edges(node))


@lru_cache(maxsize=16384)
def _sym2cpp(s, arrayexprs):
    return cppunparse.pyexpr2cpp(symbolic.symstr(s, arrayexprs, cpp_mode=True))


def sym2cpp(s, arrayexprs: Optional[Set[str]] = None) -> Union[str, List[str]]:
    """ 
    Converts an array of symbolic variables (or one) to C++ strings. 
    
    :param s: Symbolic expression to convert.
    :param arrayexprs: Set of names of arrays, used to convert SymPy 
                       user-functions back to array expressions.
    :return: C++-compilable expression or list thereof.
    """
    if not isinstance(s, list):
        return _sym2cpp(s, None if arrayexprs is None else frozenset(arrayexprs))
    return [sym2cpp(d, arrayexprs) for d in s]


def codeblock_to_cpp(cb: CodeBlock):
    """
    Converts a CodeBlock object to a C++ string.
    """
    if cb.language == dtypes.Language.CPP:
        return cb.as_string
    elif cb.language == dtypes.Language.Python:
        return cppunparse.py2cpp(cb.code)
    else:
        warnings.warn('Unrecognized language %s in codeblock' % cb.language)
        return cb.as_string


def update_persistent_desc(desc: data.Data, sdfg: SDFG):
    """
    Replaces the symbols used in a persistent data descriptor according to NestedSDFG's symbol mapping.
    The replacement happens recursively up to the top-level SDFG.
    """
    if (desc.lifetime == dtypes.AllocationLifetime.Persistent and sdfg.parent
            and any(str(s) in sdfg.parent_nsdfg_node.symbol_mapping for s in desc.free_symbols)):
        newdesc = deepcopy(desc)
        csdfg = sdfg
        while csdfg.parent_sdfg:
            if any(str(s) not in csdfg.parent_nsdfg_node.symbol_mapping for s in newdesc.free_symbols):
                raise ValueError("Persistent data descriptor depends on symbols defined in NestedSDFG scope.")
            symbolic.safe_replace(csdfg.parent_nsdfg_node.symbol_mapping,
                                  lambda m: sd.replace_properties_dict(newdesc, m))
            csdfg = csdfg.parent_sdfg
        return newdesc
    return desc


def unparse_interstate_edge(code_ast: Union[ast.AST, str], sdfg: SDFG, symbols=None, codegen=None) -> str:
    from dace.codegen.targets.cpp import InterstateEdgeUnparser  # Avoid import loop

    # Convert from code to AST as necessary
    if isinstance(code_ast, str):
        code_ast = ast.parse(code_ast).body[0]

    strio = StringIO()
    InterstateEdgeUnparser(sdfg, code_ast, strio, symbols, codegen)
    return strio.getvalue().strip()


@lru_cache()
def get_gpu_backend() -> str:
    """
    Returns the currently-selected GPU backend. If automatic,
    will perform a series of checks to see if an NVIDIA device exists,
    then if an AMD device exists, or fail.
    Otherwise, chooses the configured backend in ``compiler.cuda.backend``.
    """
    backend: str = config.Config.get('compiler', 'cuda', 'backend')
    if backend and backend != 'auto':
        return backend

    def _try_execute(cmd: str) -> bool:
        process = subprocess.Popen(cmd.split(' '), stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
        errcode = process.wait()
        return errcode == 0

    # Test 1: Test for existence of *-smi
    if _try_execute('nvidia-smi'):
        return 'cuda'
    if _try_execute('rocm-smi'):
        return 'hip'

    # Test 2: Attempt to check with CMake
    if _try_execute('cmake --find-package -DNAME=CUDA -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=EXIST'):
        return 'cuda'
    if _try_execute('cmake --find-package -DNAME=HIP -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=EXIST'):
        return 'hip'

    # Test 3: Environment variables
    if os.getenv('HIP_PLATFORM') == 'amd':
        return 'hip'
    elif os.getenv('CUDA_HOME'):
        return 'cuda'

    # Test 4: Runtime libraries
    if ctypes.util.find_library('amdhip64') and not ctypes.util.find_library('cudart'):
        return 'hip'
    elif ctypes.util.find_library('cudart') and not ctypes.util.find_library('amdhip64'):
        return 'cuda'

    raise RuntimeError('Cannot autodetect existence of NVIDIA or AMD GPU, please '
                       'set the DaCe configuration entry ``compiler.cuda.backend`` '
                       'or the ``DACE_compiler_cuda_backend`` environment variable '
                       'to either "cuda" or "hip".')
