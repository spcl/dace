# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Where a vendor BLAS call's alpha/beta live decides the handle's pointer mode.

``cublasSetPointerMode`` / ``rocblas_set_pointer_mode`` is handle-wide, and both handles are
created in device mode (``dace_cublas.h`` / ``dace_rocblas.h``). So:

* a compile-time 1.0 / 0.0 uses the preallocated device constants and must not touch the mode;
* a runtime coefficient read on the host is passed by host address, which needs host mode for the
  call and device mode restored after, or the GPU dereferences a host pointer;
* a device-resident coefficient is passed as a device pointer and stays in device mode.

The constants are handed to rocBLAS as ``rocblas_*_complex``, the type its C++ API declares. The
hip vector types are a distinct type in C++, so quatrex_rgf's complex GEMM failed to compile when the
constants used them.
"""
import pathlib

import pytest

import dace
from dace import dtypes
from dace.libraries.blas.nodes.gemm import Gemm

N = 8
#: The rocBLAS header that defines the preallocated alpha/beta constants.
ROCBLAS_HEADER = pathlib.Path(dace.__file__).parent / 'libraries' / 'blas' / 'include' / 'dace_rocblas.h'


def gemm_code(alpha: float,
              storage: dtypes.StorageType = dtypes.StorageType.GPU_Global,
              dtype: dace.typeclass = dace.float64) -> str:
    """The tasklet code a rocBLAS GEMM with this ``alpha`` expands to."""
    sdfg = dace.SDFG(f'gemm_alpha_{str(alpha).replace(".", "_").replace("-", "m")}_{dtype.to_string()}')
    for name in ('A', 'B', 'C'):
        sdfg.add_array(name, [N, N], dtype, storage=storage)
    state = sdfg.add_state()
    node = Gemm('gemm', alpha=alpha)
    node.implementation = 'rocBLAS'
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_a', dace.Memlet(f'A[0:{N}, 0:{N}]'))
    state.add_edge(state.add_read('B'), None, node, '_b', dace.Memlet(f'B[0:{N}, 0:{N}]'))
    state.add_edge(node, '_c', state.add_write('C'), None, dace.Memlet(f'C[0:{N}, 0:{N}]'))
    node.expand(state)
    return '\n'.join(n.code.as_string for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))


def test_a_constant_coefficient_uses_the_device_constants():
    """alpha=1 is the preallocated device constant, so the mode is left alone."""
    code = gemm_code(1.0)
    assert 'Constants()' in code, code[:400]
    assert 'rocblas_pointer_mode_host' not in code, code[:400]


def test_a_host_coefficient_switches_the_mode_and_restores_it():
    """A value passed by host address must be read under host mode, and device mode restored."""
    code = gemm_code(2.5)
    assert 'rocblas_pointer_mode_host' in code, code[:400]
    assert code.index('rocblas_pointer_mode_host') < code.index('rocblas_zgemm' if 'zgemm' in code else 'gemm')
    assert 'rocblas_pointer_mode_device' in code, code[:400]
    assert code.rindex('rocblas_pointer_mode_device') > code.index('rocblas_pointer_mode_host'), code


def test_the_mode_is_restored_after_every_host_coefficient_call():
    """Leaving the handle in host mode would make the next call misread its device constant."""
    code = gemm_code(2.5)
    assert code.rstrip().endswith(';'), code[-200:]
    assert code.count('rocblas_pointer_mode_device') >= 1, code


@pytest.mark.parametrize('spelling', ['rocblas_float_complex', 'rocblas_double_complex'])
def test_the_constants_are_declared_in_the_rocblas_complex_types(spelling: str):
    """rocBLAS declares its complex parameters as rocblas_complex_num<T> in C++.

    The constants are built with the hip vector types (``make_hipDoubleComplex``) and handed over as
    the rocBLAS ones; declaring the accessors as the hip types made every complex rocBLAS call fail
    to compile with ``cannot convert 'const hipDoubleComplex*' to 'const rocblas_double_complex*'``.
    """
    header = ROCBLAS_HEADER.read_text()
    assert f'{spelling} const*' in header or f'{spelling}*' in header, spelling
    accessors = [line for line in header.splitlines() if 'Complex' in line and 'const*' in line and '()' in line]
    assert accessors, header[:400]
    assert not any('hipComplex const*' in line or 'hipDoubleComplex const*' in line for line in accessors), accessors
