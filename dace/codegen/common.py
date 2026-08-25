# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import ast
from copy import deepcopy
import ctypes.util
from dace import config, data, dtypes, mpr_lowering, sdfg as sd, symbolic
from dace.sdfg import SDFG
from dace.properties import CodeBlock
from dace.codegen import cppunparse
from dace.codegen.tools import gpu_runtime
from functools import lru_cache
from io import StringIO
import numpy as np
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


@lru_cache(maxsize=16384, typed=True)
def _sym2cpp(s, arrayexprs, dialect, fp_ctype):
    return cppunparse.pyexpr2cpp(symbolic.symstr(s, arrayexprs, cpp_mode=True, dialect=dialect, fp_ctype=fp_ctype))


def sym2cpp(s,
            arrayexprs: Optional[Set[str]] = None,
            dialect: Optional[mpr_lowering.Dialect] = None,
            fp_ctype: Optional[str] = None) -> Union[str, List[str]]:
    """
    Converts an array of symbolic variables (or one) to C++ strings.

    :param s: Symbolic expression to convert.
    :param arrayexprs: Set of names of arrays, used to convert SymPy
                       user-functions back to array expressions.
    :param dialect: which C++ vocabulary may be emitted. ``None`` (the default) takes the ambient
                    dialect (:func:`~dace.mpr_lowering.active_dialect`), which is ``RUNTIME``
                    unless an MPR rendering is in progress -- so the several hundred call sites in
                    the code generators need no change. Resolved HERE and passed down as an
                    argument, so it still reaches the ``_sym2cpp`` memoization key; nothing inside
                    the cached function reads the ambient value. See
                    :class:`~dace.mpr_lowering.Dialect`.
    :param fp_ctype: C++ floating type the expression evaluates in, so a sympy ``Rational``
                     becomes a division of THAT type instead of a truncating integer division.
                     ``None`` (the default) keeps integer division, which index arithmetic
                     needs. In the memoization key for the same reason ``dialect`` is.
    :return: C++-compilable expression or list thereof.
    """
    if dialect is None:
        dialect = mpr_lowering.active_dialect()
    if isinstance(s, list):
        return [sym2cpp(d, arrayexprs, dialect, fp_ctype) for d in s]
    # Two literal kinds symstr cannot carry: a bool round-trips as Python 'True' (or as the
    # integer 1, which loses the type), and a complex loses its width -- a complex64 constant
    # would be emitted as dace::complex128.
    if isinstance(s, (bool, np.bool_)):
        return 'true' if s else 'false'
    if isinstance(s, (complex, np.complexfloating)):
        ctype = str(dtypes.dtype_to_typeclass(type(s)))
        if dialect is mpr_lowering.Dialect.STANDALONE_C:
            # ``a + b*I`` is not the same literal: it evaluates, so a NaN or an infinite component
            # propagates through the multiplication. ``CMPLX`` builds the value component-wise.
            builder = 'CMPLXF' if ctype == 'dace::complex64' else 'CMPLX'
            return f'{builder}({s.real}, {s.imag})'
        if dialect is mpr_lowering.Dialect.STANDALONE:
            ctype = mpr_lowering.ctype_for(ctype, dialect)
        return f'{ctype}({s.real}, {s.imag})'
    return _sym2cpp(s, None if arrayexprs is None else frozenset(arrayexprs), dialect, fp_ctype)


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
    if (desc.lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External) and sdfg.parent
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


def gpu_stream_expr(stream: Union[int, str]) -> str:
    """Renders a ``_cuda_stream`` annotation as the C expression naming that stream.

    The annotation indexes the context's stream array, except for ``'nullptr'``: the legacy default
    stream lives outside it. Going through here keeps that stream an ordinary one, passed to work
    and synchronized like any other.
    """
    if stream == 'nullptr':
        return 'nullptr'
    return f'__state->gpu_context->streams[{stream}]'


def cpp_standard() -> str:
    """The C++ standard version to build with, per ``compiler.cpp_standard`` -- clamped to a minimum
    of 20. DaCe assumes C++20 or newer everywhere (aligned ``operator new``, ``consteval``, ...), so a
    lower configured value is raised to 20 rather than passed through to the compiler invocation."""
    try:
        standard = int(str(config.Config.get('compiler', 'cpp_standard')).strip())
    except ValueError:
        standard = 20
    return str(max(standard, 20))


#: GCC releases whose auto-vectorizer mislowers a select whose operands are loaded inline. It builds
#: the lane mask for one operand by sign-extending the low half of an int32 compare and then reuses
#: that same mask for the high-half ``vmaskmovpd``, which zeroes the lanes it believes are masked
#: off. A ``mask[i] ? t[i] : f[i]`` over int32/float64 therefore returns zeros for the upper half of
#: every vector -- a silent wrong answer, not a crash. Reproduced on 13.3.0 at ``-O3`` with any
#: vectorizing ``-march``; ``-fno-tree-vectorize`` avoids it and GCC 14 fixed it.
_MISCOMPILING_GCC_MAJORS = (13, )


def warn_if_cxx_miscompiles_inline_selects() -> None:
    """Warn when the configured host compiler is a release known to miscompile the inlined select
    the readable code generator emits.

    Checked here, at emission time, rather than left to the build: the pattern is emitted for every
    ``a if c else b`` tasklet, the wrong answer is silent, and by the time a test compares numbers
    there is nothing left pointing at the toolchain. The lowering itself is correct C++ and stays as
    it is -- what is reported is the compiler.
    """
    from dace.codegen import compiler_family  # Avoid import loop
    if config.Config.get('compiler', 'cpu', 'implementation') != 'experimental_readable':
        return  # classic codegen keeps the operands in connector locals, so the pattern never forms
    executable = compiler_family.host_compiler()
    if compiler_family.detect(executable) != 'gnu':
        return
    version = compiler_family.detect_version(executable)
    if version is None or version[0] not in _MISCOMPILING_GCC_MAJORS:
        return
    warnings.warn(f'Host C++ compiler {executable} is GCC {".".join(str(v) for v in version)}, whose '
                  'auto-vectorizer mislowers a masked select and silently returns zeros for the upper '
                  'half of each vector. Build with GCC 14 or newer, or add -fno-tree-vectorize to '
                  'compiler.cpu.args.')


def emits_tree_reductions(experimental: bool) -> bool:
    """Whether a WCR accumulator folds into a tree reduction rather than a per-thread atomic.

    Always on for the experimental targets -- the fold IS how they lower a reduction, so there is
    nothing to opt into. The legacy targets keep it behind ``compiler.emit_tree_reductions``.
    """
    return experimental or config.Config.get_bool('compiler', 'emit_tree_reductions')


def cuda_emits_tree_reductions() -> bool:
    """:func:`emits_tree_reductions` for whichever CUDA codegen ``compiler.cuda.implementation`` picks.

    For callers outside codegen (a pass sizing thread blocks for a fold that codegen may or may not
    emit) that have no code generator to ask.
    """
    return emits_tree_reductions(config.Config.get('compiler', 'cuda', 'implementation') == 'experimental')


def get_gpu_backend() -> str:
    """Returns the currently-selected GPU backend in ``compiler.cuda.backend``.

    If automatic, will perform a series of checks to see if an NVIDIA device exists,
    then if an AMD device exists, or fail. Note that the automatically detected case
    will never be revisited.

    NOT cached as a whole: that would freeze the first answer, so a later
    ``set_temporary('compiler', 'cuda', 'backend', ...)`` could never take effect. Only the
    probing is expensive, and it carries its own cache.
    """
    backend: str = config.Config.get('compiler', 'cuda', 'backend')
    if backend and backend != 'auto':
        return backend

    return _probing_for_gpu_backend()


@lru_cache(maxsize=None, typed=True)
def _probing_for_gpu_backend() -> str:
    # Probe the system for the GPU backend. Called by ``get_gpu_backend()`` when
    # the backend is unset, not directly; the cached result never changes.
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


def get_gpu_runtime() -> gpu_runtime.GPURuntime:
    """
    Returns the GPU runtime library (CUDA / HIP) if exists. The result is cached for performance.
    """
    backend = get_gpu_backend()
    return _look_for_runtime_file(backend)


@lru_cache(maxsize=None, typed=True)
def _look_for_runtime_file(backend: str) -> gpu_runtime.GPURuntime:
    # Locate a GPU backend's runtime. Called indirectly by ``get_gpu_runtime()``,
    # not directly.

    if backend == 'cuda':
        libpath = ctypes.util.find_library('cudart')
        if os.name == 'nt' and not libpath:  # Windows-based search
            for version in (12, 11, 10, 9):
                libpath = ctypes.util.find_library(f'cudart64_{version}0')
                if libpath:
                    break
    elif backend == 'hip':
        libpath = ctypes.util.find_library('amdhip64')
    else:
        raise RuntimeError(f'Cannot obtain GPU runtime library for backend {backend}')

    if not libpath:
        envname = 'PATH' if os.name == 'nt' else 'LD_LIBRARY_PATH'
        raise RuntimeError(f'GPU runtime library for {backend} not found. Please set the {envname} '
                           'environment variable to point to the libraries.')

    return gpu_runtime.GPURuntime(backend, libpath)


def platform_library_name(libname: str) -> str:
    """ Get the filename of a library.

        :param libname: the name of the library.
        :return: the filename of the library.
    """
    prefix = config.Config.get('compiler', 'library_prefix')
    suffix = config.Config.get('compiler', 'library_extension')
    return f"{prefix}{libname}.{suffix}"
