import functools  
import sympy      
from typing import List

from dace import Config, symbolic
from dace.codegen import cppunparse
from dace.codegen.prettycode import CodeIOStream    
from dace.sdfg import nodes


def symbolic_to_cpp(arr):
    """ Converts an array of symbolic variables (or one) to C++ strings. """
    if not isinstance(arr, list):
        return cppunparse.pyexpr2cpp(symbolic.symstr(arr, cpp_mode=True))
    return [cppunparse.pyexpr2cpp(symbolic.symstr(d, cpp_mode=True)) for d in arr]

def get_cuda_dim(idx):
    """ Converts 0 to x, 1 to y, 2 to z, or raises an exception. """
    if idx < 0 or idx > 2:
        raise ValueError(f'idx must be between 0 and 2, got {idx}')
    return ('x', 'y', 'z')[idx]

def product(iterable):
    """
    Computes the symbolic product of elements in the iterable using sympy.Mul.

    This is equivalent to: ```functools.reduce(sympy.Mul, iterable, 1)```.

    Purpose: This function is used to improve readability of the codeGen.
    """
    return functools.reduce(sympy.Mul, iterable, 1)

def to_3d_dims(dim_sizes: List) -> List:
    """
    Converts a list of dimension sizes to a 3D format.

    If the list has more than three dimensions, all dimensions beyond the second are 
    collapsed into the third (via multiplication). If the list has fewer than three 
    entries, it is padded with 1s to ensure a fixed length of three.

    Examples:
        [x]             → [x, 1, 1]
        [x, y]          → [x, y, 1]
        [x, y, z]       → [x, y, z]
        [x, y, z, u, v] → [x, y, z * u * v]
    """
    
    if len(dim_sizes) > 3:
        # multiply everything from the 3rd onward into d[2]
        dim_sizes[2] = product(dim_sizes[2:])
        dim_sizes = dim_sizes[:3]

    # pad with 1s if necessary
    dim_sizes += [1] * (3 - len(dim_sizes))

    return dim_sizes

def validate_block_size_limits(kernel_map_entry: nodes.MapEntry, block_size: List):
    """
    Validates that the given block size for a kernel does not exceed typical CUDA hardware limits.

    These limits are not enforced by the CUDA compiler itself, but are configurable checks
    performed by DaCe during GPU code generation. They are based on common hardware
    restrictions and can be adjusted via the configuration system.

    Specifically, this function checks:
    - That the total number of threads in the block does not exceed `compiler.cuda.block_size_limit`.
    - That the number of threads in the last (z) dimension does not exceed
      `compiler.cuda.block_size_lastdim_limit`.

    Raises:
        ValueError: If either limit is exceeded.
    """

    kernel_map_label = kernel_map_entry.map.label

    total_block_size = product(block_size)
    limit = Config.get('compiler', 'cuda', 'block_size_limit')
    lastdim_limit = Config.get('compiler', 'cuda', 'block_size_lastdim_limit')
    
    if (total_block_size > limit) == True:
        raise ValueError(f'Block size for kernel "{kernel_map_label}" ({block_size}) '
                            f'is larger than the possible number of threads per block ({limit}). '
                            'The kernel will potentially not run, please reduce the thread-block size. '
                            'To increase this limit, modify the `compiler.cuda.block_size_limit` '
                            'configuration entry.')
    
    if (block_size[-1] > lastdim_limit) == True:
        raise ValueError(f'Last block size dimension for kernel "{kernel_map_label}" ({block_size}) '
                            'is larger than the possible number of threads in the last block dimension '
                            f'({lastdim_limit}). The kernel will potentially not run, please reduce the '
                            'thread-block size. To increase this limit, modify the '
                            '`compiler.cuda.block_size_lastdim_limit` configuration entry.')

def emit_sync_debug_checks(backend: str, codestream: CodeIOStream):
    """
    Emit backend sync and error-check calls if synchronous debugging is enabled.
    
    Args:
        backend (str): Backend API prefix (e.g., 'cuda').
        codestream (CodeIOStream): Stream to write code to.
    """
    if Config.get_bool('compiler', 'cuda', 'syncdebug'):
        codestream.write(
            f"DACE_GPU_CHECK({backend}GetLastError());\n"
            f"DACE_GPU_CHECK({backend}DeviceSynchronize());\n"
        ) 