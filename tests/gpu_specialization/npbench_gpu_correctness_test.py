# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""NPBench kernels through the new GPU stream pipeline compared element-wise against the CPU SDFG."""
import importlib
from dataclasses import dataclass, field

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.new_gpu_codegen_only]

TSTEPS = 3


@dataclass(frozen=True)
class Case:
    """One kernel: ``init(*init_args)`` returns the data, named by ``names`` (``None`` drops an item)."""
    subdir: str
    module: str
    kernel: str
    init: str
    init_args: tuple
    names: tuple
    symbols: dict
    constants: dict = field(default_factory=dict)
    rtol: float = 1e-10
    atol: float = 1e-12


CASES = {
    'atax':
    Case('polybench', 'atax_test', 'kernel', 'init_data', (12, 16), ('A', 'x', None), dict(M=12, N=16), {}, 1e-5, 1e-6),
    'bicg':
    Case('polybench', 'bicg_test', 'bicg_kernel', 'initialize', (12, 16), ('A', 'p', 'r'), dict(M=12, N=16)),
    'gemm':
    Case('polybench', 'gemm_npbench_test', 'gemm_kernel', 'initialize', (12, 14, 16), ('alpha', 'beta', 'C', 'A', 'B'),
         dict(NI=12, NJ=14, NK=16)),
    'k2mm':
    Case('polybench', 'k2mm_test', 'k2mm_kernel', 'initialize', (8, 10, 12, 14), ('alpha', 'beta', 'A', 'B', 'C', 'D'),
         dict(NI=8, NJ=10, NK=12, NL=14)),
    'k3mm':
    Case('polybench', 'k3mm_test', 'k3mm_kernel', 'initialize', (6, 8, 10, 12, 14), ('A', 'B', 'C', 'D'),
         dict(NI=6, NJ=8, NK=10, NL=12, NM=14)),
    'mvt':
    Case('polybench', 'mvt_test', 'mvt_kernel', 'initialize', (16, ), ('x1', 'x2', 'y_1', 'y_2', 'A'), dict(N=16)),
    'gesummv':
    Case('polybench', 'gesummv_test', 'gesummv_kernel', 'initialize', (16, ), ('alpha', 'beta', 'A', 'B', 'x'),
         dict(N=16)),
    'gemver':
    Case('polybench', 'gemver_test', 'gemver_kernel', 'initialize', (16, ),
         ('alpha', 'beta', 'A', 'u1', 'v1', 'u2', 'v2', 'w', 'x', 'y', 'z'), dict(N=16)),
    'syrk':
    Case('polybench', 'syrk_test', 'kernel', 'init_data', (12, 16), ('alpha', 'beta', 'C', 'A'), dict(M=16, N=12), {},
         1e-5, 1e-6),
    'syr2k':
    Case('polybench', 'syr2k_test', 'syr2k_kernel', 'initialize', (12, 16), ('alpha', 'beta', 'C', 'A', 'B'),
         dict(M=16, N=12)),
    'symm':
    Case('polybench', 'symm_test', 'symm_kernel', 'initialize', (12, 16), ('alpha', 'beta', 'C', 'A', 'B'),
         dict(M=12, N=16)),
    'trmm':
    Case('polybench', 'trmm_test', 'trmm_kernel', 'initialize', (12, 16), ('alpha', 'A', 'B'), dict(M=12, N=16)),
    'trisolv':
    Case('polybench', 'trisolv_test', 'trisolv_kernel', 'initialize', (16, ), ('L', 'x', 'b'), dict(N=16)),
    'durbin':
    Case('polybench', 'durbin_test', 'durbin_kernel', 'initialize', (16, ), ('r', ), dict(N=16)),
    'lu':
    Case('polybench', 'lu_test', 'lu_kernel', 'init_data', (16, ), ('A', ), dict(N=16), {}, 1e-4, 1e-5),
    'ludcmp':
    Case('polybench', 'ludcmp_test', 'ludcmp_kernel', 'initialize', (16, ), ('A', 'b'), dict(N=16)),
    'correlation':
    Case('polybench', 'correlation_test', 'correlation_kernel', 'initialize', (12, 16), ('float_n', 'data'),
         dict(M=12, N=16)),
    'covariance':
    Case('polybench', 'covariance_test', 'covariance_kernel', 'init_data', (12, 16), ('float_n', 'data'),
         dict(M=12, N=16), {}, 1e-4, 1e-5),
    'gramschmidt':
    Case('polybench', 'gramschmidt_test', 'gramschmidt_kernel', 'initialize', (14, 10), ('A', ), dict(M=14, N=10), {},
         1e-6, 1e-8),
    'doitgen':
    Case('polybench', 'doitgen_test', 'doitgen_kernel', 'initialize', (4, 6, 8), ('A', 'C4'), dict(NR=4, NQ=6, NP=8)),
    'deriche':
    Case('polybench', 'deriche_test', 'deriche_kernel', 'initialize', (16, 20), ('alpha', 'imgIn'), dict(W=16, H=20)),
    'floyd_warshall':
    Case('polybench', 'floyd_warshall_test', 'kernel', 'init_data', (16, ), ('path', ), dict(N=16)),
    'nussinov':
    Case('polybench', 'nussinov_test', 'kernel', 'init_data', (16, ), ('seq', None), dict(N=16)),
    'jacobi_1d':
    Case('polybench', 'jacobi_1d_test', 'jacobi_1d_kernel', 'initialize', (16, ), ('A', 'B'), dict(N=16),
         dict(TSTEPS=TSTEPS)),
    'jacobi_2d':
    Case('polybench', 'jacobi_2d_test', 'kernel', 'init_data', (16, ), ('A', 'B'), dict(N=16), dict(TSTEPS=TSTEPS),
         1e-5, 1e-6),
    'seidel_2d':
    Case('polybench', 'seidel_2d_test', 'seidel_2d_kernel', 'initialize', (16, ), ('A', ), dict(N=16),
         dict(TSTEPS=TSTEPS)),
    'heat_3d':
    Case('polybench', 'heat_3d_test', 'heat_3d_kernel', 'initialize', (10, ), ('A', 'B'), dict(N=10),
         dict(TSTEPS=TSTEPS)),
    'adi':
    Case('polybench', 'adi_test', 'adi_kernel', 'initialize', (16, ), ('u', ), dict(N=16), dict(TSTEPS=TSTEPS)),
    'fdtd_2d':
    Case('polybench', 'fdtd_2d_test', 'kernel', 'init_data', (TSTEPS, 12, 16), ('ex', 'ey', 'hz', '_fict_'),
         dict(TMAX=TSTEPS, NX=12, NY=16), {}, 1e-5, 1e-6),
    'cavity_flow':
    Case('misc', 'cavity_flow_test', 'dace_cavity_flow', 'initialize', (21, 21), ('u', 'v', 'p', 'dx', 'dy', 'dt'),
         dict(ny=21, nx=21), dict(nt=4, nit=5, rho=1.0, nu=0.1), 1e-6, 1e-8),
    'channel_flow':
    Case('misc', 'channel_flow_test', 'dace_channel_flow', 'initialize', (21, 21), ('u', 'v', 'p', 'dx', 'dy', 'dt'),
         dict(ny=21, nx=21), dict(nit=5, rho=1.0, nu=0.1, F=1.0), 1e-6, 1e-8),
    'hdiff':
    Case('weather_stencils', 'hdiff_test', 'hdiff_kernel', 'initialize', (16, 16, 8),
         ('in_field', 'out_field', 'coeff'), dict(I=16, J=16, K=8)),
    'vadv':
    Case('weather_stencils', 'vadv_test', 'vadv_kernel', 'initialize', (16, 16, 8),
         ('dtr_stage', 'utens_stage', 'u_stage', 'wcon', 'u_pos', 'utens'), dict(I=16, J=16, K=8)),
}


def load_kernel_module(case: Case):
    """The kernel-test module, imported as ``tests.npbench.<subdir>.<module>`` from the repository root."""
    return importlib.import_module(f"tests.npbench.{case.subdir}.{case.module}")


def build_arguments(case: Case, module):
    """A factory of fresh keyword arguments: arrays are copied per call, scalars shared."""
    data = getattr(module, case.init)(*case.init_args)
    data = data if isinstance(data, tuple) else (data, )
    named = {name: value for name, value in zip(case.names, data, strict=True) if name is not None}
    return lambda: {
        **{
            name: value.copy() if isinstance(value, np.ndarray) else value
            for name, value in named.items()
        },
        **case.constants
    }


def compare(cpu, gpu, rtol: float, atol: float, what: str):
    if cpu is None:
        return
    np.testing.assert_allclose(gpu, cpu, rtol=rtol, atol=atol, err_msg=what)


@pytest.mark.parametrize('name', CASES)
def test_the_gpu_sdfg_matches_the_cpu_sdfg_elementwise(name):
    case = CASES[name]
    kernel = getattr(load_kernel_module(case), case.kernel)
    arguments = build_arguments(case, load_kernel_module(case))

    cpu_arguments = arguments()
    cpu_result = kernel.to_sdfg(simplify=True)(**cpu_arguments, **case.symbols)

    # ``ExperimentalCUDACodeGen.preprocess`` runs the stream pipeline itself; applying it here too would wire twice.
    gpu_sdfg = kernel.to_sdfg(simplify=True)
    gpu_sdfg.apply_gpu_transformations()
    gpu_arguments = arguments()
    gpu_result = gpu_sdfg(**gpu_arguments, **case.symbols)

    for argument, expected in cpu_arguments.items():
        if isinstance(expected, np.ndarray):
            compare(expected, gpu_arguments[argument], case.rtol, case.atol, f'argument "{argument}"')
    cpu_results = cpu_result if isinstance(cpu_result, tuple) else (cpu_result, )
    gpu_results = gpu_result if isinstance(gpu_result, tuple) else (gpu_result, )
    for index, (expected, got) in enumerate(zip(cpu_results, gpu_results, strict=True)):
        compare(expected, got, case.rtol, case.atol, f'return[{index}]')


if __name__ == '__main__':
    for name in CASES:
        test_the_gpu_sdfg_matches_the_cpu_sdfg_elementwise(name)
