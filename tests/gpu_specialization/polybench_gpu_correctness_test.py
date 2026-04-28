# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU-offloading correctness tests for npbench polybench kernels.

Each test builds a CPU SDFG (reference) and a GPU-transformed SDFG from the same
``@dc.program``, runs both on independent copies of the inputs at a small size
(no dimension equals 1), and compares the results element-wise.  Kernels are
imported from ``tests/npbench/polybench`` so the canonical source is exercised.
"""
import os
import sys
from typing import Callable, Dict

import numpy as np
import pytest

pytestmark = pytest.mark.new_gpu_codegen_only

_NPBENCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'npbench', 'polybench'))
if _NPBENCH_DIR not in sys.path:
    sys.path.insert(0, _NPBENCH_DIR)

import adi_test  # noqa: E402
import atax_test  # noqa: E402
import bicg_test  # noqa: E402
import cholesky_test  # noqa: E402
import cholesky2_test  # noqa: E402
import correlation_test  # noqa: E402
import covariance_test  # noqa: E402
import deriche_test  # noqa: E402
import doitgen_test  # noqa: E402
import durbin_test  # noqa: E402
import fdtd_2d_test  # noqa: E402
import floyd_warshall_test  # noqa: E402
import gemm_npbench_test  # noqa: E402
import gemver_test  # noqa: E402
import gesummv_test  # noqa: E402
import gramschmidt_test  # noqa: E402
import heat_3d_test  # noqa: E402
import jacobi_1d_test  # noqa: E402
import jacobi_2d_test  # noqa: E402
import k2mm_test  # noqa: E402
import k3mm_test  # noqa: E402
import lu_test  # noqa: E402
import ludcmp_test  # noqa: E402
import mvt_test  # noqa: E402
import nussinov_test  # noqa: E402
import seidel_2d_test  # noqa: E402
import symm_test  # noqa: E402
import syr2k_test  # noqa: E402
import syrk_test  # noqa: E402
import trisolv_test  # noqa: E402
import trmm_test  # noqa: E402


def _compare_arrays(cpu_args: Dict[str, np.ndarray], gpu_args: Dict[str, np.ndarray], rtol: float, atol: float) -> None:
    for name, cpu_val in cpu_args.items():
        if not isinstance(cpu_val, np.ndarray):
            continue
        np.testing.assert_allclose(gpu_args[name], cpu_val, rtol=rtol, atol=atol, err_msg=f'arg "{name}" mismatch')


def _compare_returns(cpu_ret, gpu_ret, rtol: float, atol: float) -> None:
    if cpu_ret is None:
        return
    if isinstance(cpu_ret, tuple):
        for i, (c, g) in enumerate(zip(cpu_ret, gpu_ret)):
            np.testing.assert_allclose(g, c, rtol=rtol, atol=atol, err_msg=f'return[{i}] mismatch')
    else:
        np.testing.assert_allclose(gpu_ret, cpu_ret, rtol=rtol, atol=atol, err_msg='return mismatch')


def _run_gpu_vs_cpu(kernel,
                    build_args: Callable[[], Dict[str, np.ndarray]],
                    symbols: Dict[str, int],
                    *,
                    rtol: float = 1e-10,
                    atol: float = 1e-12) -> None:
    """Run ``kernel`` on CPU (reference) and on a GPU-transformed SDFG, compare outputs.

    :param kernel: the ``@dc.program`` to exercise.
    :param build_args: callable returning a fresh ``{name: array}`` argument dict per call.
    :param symbols: symbol values passed to both SDFGs.
    :param rtol: relative tolerance for ``np.testing.assert_allclose``.
    :param atol: absolute tolerance for ``np.testing.assert_allclose``.
    """
    cpu_sdfg = kernel.to_sdfg(simplify=True)
    cpu_args = build_args()
    cpu_ret = cpu_sdfg(**cpu_args, **symbols)

    gpu_sdfg = kernel.to_sdfg(simplify=True)
    gpu_sdfg.apply_gpu_transformations()
    gpu_args = build_args()
    gpu_ret = gpu_sdfg(**gpu_args, **symbols)

    _compare_arrays(cpu_args, gpu_args, rtol, atol)
    _compare_returns(cpu_ret, gpu_ret, rtol, atol)


_TSTEPS_SMALL = 3


@pytest.mark.gpu
def test_atax_gpu_matches_cpu():
    M, N = 12, 16
    A, x, _y = atax_test.init_data(M, N)
    _run_gpu_vs_cpu(atax_test.kernel, lambda: dict(A=A.copy(), x=x.copy()), dict(M=M, N=N), rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_bicg_gpu_matches_cpu():
    M, N = 12, 16
    A, p, r = bicg_test.initialize(M, N)
    _run_gpu_vs_cpu(bicg_test.bicg_kernel, lambda: dict(A=A.copy(), p=p.copy(), r=r.copy()), dict(M=M, N=N))


@pytest.mark.gpu
def test_gemm_gpu_matches_cpu():
    NI, NJ, NK = 12, 14, 16
    alpha, beta, C, A, B = gemm_npbench_test.initialize(NI, NJ, NK)
    _run_gpu_vs_cpu(gemm_npbench_test.gemm_kernel,
                    lambda: dict(alpha=alpha, beta=beta, C=C.copy(), A=A.copy(), B=B.copy()), dict(NI=NI, NJ=NJ, NK=NK))


@pytest.mark.gpu
def test_k2mm_gpu_matches_cpu():
    NI, NJ, NK, NL = 8, 10, 12, 14
    alpha, beta, A, B, C, D = k2mm_test.initialize(NI, NJ, NK, NL)
    _run_gpu_vs_cpu(k2mm_test.k2mm_kernel,
                    lambda: dict(alpha=alpha, beta=beta, A=A.copy(), B=B.copy(), C=C.copy(), D=D.copy()),
                    dict(NI=NI, NJ=NJ, NK=NK, NL=NL))


@pytest.mark.gpu
def test_k3mm_gpu_matches_cpu():
    NI, NJ, NK, NL, NM = 6, 8, 10, 12, 14
    A, B, C, D = k3mm_test.initialize(NI, NJ, NK, NL, NM)
    _run_gpu_vs_cpu(k3mm_test.k3mm_kernel, lambda: dict(A=A.copy(), B=B.copy(), C=C.copy(), D=D.copy()),
                    dict(NI=NI, NJ=NJ, NK=NK, NL=NL, NM=NM))


@pytest.mark.gpu
def test_mvt_gpu_matches_cpu():
    N = 16
    x1, x2, y_1, y_2, A = mvt_test.initialize(N)
    _run_gpu_vs_cpu(mvt_test.mvt_kernel,
                    lambda: dict(x1=x1.copy(), x2=x2.copy(), y_1=y_1.copy(), y_2=y_2.copy(), A=A.copy()), dict(N=N))


@pytest.mark.gpu
def test_gesummv_gpu_matches_cpu():
    N = 16
    alpha, beta, A, B, x = gesummv_test.initialize(N)
    _run_gpu_vs_cpu(gesummv_test.gesummv_kernel,
                    lambda: dict(alpha=alpha, beta=beta, A=A.copy(), B=B.copy(), x=x.copy()), dict(N=N))


@pytest.mark.gpu
def test_gemver_gpu_matches_cpu():
    N = 16
    alpha, beta, A, u1, v1, u2, v2, w, x, y, z = gemver_test.initialize(N)
    _run_gpu_vs_cpu(
        gemver_test.gemver_kernel, lambda: dict(alpha=alpha,
                                                beta=beta,
                                                A=A.copy(),
                                                u1=u1.copy(),
                                                v1=v1.copy(),
                                                u2=u2.copy(),
                                                v2=v2.copy(),
                                                w=w.copy(),
                                                x=x.copy(),
                                                y=y.copy(),
                                                z=z.copy()), dict(N=N))


@pytest.mark.gpu
def test_syrk_gpu_matches_cpu():
    N, M = 12, 16
    alpha, beta, C, A = syrk_test.init_data(N, M)
    _run_gpu_vs_cpu(syrk_test.kernel,
                    lambda: dict(alpha=alpha, beta=beta, C=C.copy(), A=A.copy()),
                    dict(M=M, N=N),
                    rtol=1e-5,
                    atol=1e-6)


@pytest.mark.gpu
def test_syr2k_gpu_matches_cpu():
    N, M = 12, 16
    alpha, beta, C, A, B = syr2k_test.initialize(N, M)
    _run_gpu_vs_cpu(syr2k_test.syr2k_kernel, lambda: dict(alpha=alpha, beta=beta, C=C.copy(), A=A.copy(), B=B.copy()),
                    dict(M=M, N=N))


@pytest.mark.gpu
def test_symm_gpu_matches_cpu():
    M, N = 12, 16
    alpha, beta, C, A, B = symm_test.initialize(M, N)
    _run_gpu_vs_cpu(symm_test.symm_kernel, lambda: dict(alpha=alpha, beta=beta, C=C.copy(), A=A.copy(), B=B.copy()),
                    dict(M=M, N=N))


@pytest.mark.gpu
def test_trmm_gpu_matches_cpu():
    M, N = 12, 16
    alpha, A, B = trmm_test.initialize(M, N)
    _run_gpu_vs_cpu(trmm_test.trmm_kernel, lambda: dict(alpha=alpha, A=A.copy(), B=B.copy()), dict(M=M, N=N))


@pytest.mark.gpu
def test_trisolv_gpu_matches_cpu():
    N = 16
    L, x, b = trisolv_test.initialize(N)
    _run_gpu_vs_cpu(trisolv_test.trisolv_kernel, lambda: dict(L=L.copy(), x=x.copy(), b=b.copy()), dict(N=N))


@pytest.mark.gpu
def test_durbin_gpu_matches_cpu():
    N = 16
    r = durbin_test.initialize(N)
    _run_gpu_vs_cpu(durbin_test.durbin_kernel, lambda: dict(r=r.copy()), dict(N=N))


@pytest.mark.gpu
def test_cholesky_gpu_matches_cpu():
    N = 16
    A = cholesky_test.init_data(N)
    _run_gpu_vs_cpu(cholesky_test.kernel, lambda: dict(A=A.copy()), dict(N=N), rtol=1e-4, atol=1e-5)


@pytest.mark.gpu
def test_cholesky2_gpu_matches_cpu():
    N = 16
    A = cholesky2_test.init_data(N)
    _run_gpu_vs_cpu(cholesky2_test.cholesky2_kernel, lambda: dict(A=A.copy()), dict(N=N))


@pytest.mark.gpu
def test_lu_gpu_matches_cpu():
    N = 16
    A = lu_test.init_data(N)
    _run_gpu_vs_cpu(lu_test.lu_kernel, lambda: dict(A=A.copy()), dict(N=N), rtol=1e-4, atol=1e-5)


@pytest.mark.gpu
def test_ludcmp_gpu_matches_cpu():
    N = 16
    A, b = ludcmp_test.initialize(N)
    _run_gpu_vs_cpu(ludcmp_test.ludcmp_kernel, lambda: dict(A=A.copy(), b=b.copy()), dict(N=N))


@pytest.mark.gpu
def test_correlation_gpu_matches_cpu():
    M, N = 12, 16
    float_n, data = correlation_test.initialize(M, N)
    _run_gpu_vs_cpu(correlation_test.correlation_kernel, lambda: dict(float_n=float_n, data=data.copy()), dict(M=M,
                                                                                                               N=N))


@pytest.mark.gpu
def test_covariance_gpu_matches_cpu():
    M, N = 12, 16
    float_n, data = covariance_test.init_data(M, N)
    _run_gpu_vs_cpu(covariance_test.covariance_kernel,
                    lambda: dict(float_n=float_n, data=data.copy()),
                    dict(M=M, N=N),
                    rtol=1e-4,
                    atol=1e-5)


@pytest.mark.gpu
def test_gramschmidt_gpu_matches_cpu():
    M, N = 14, 10
    A = gramschmidt_test.initialize(M, N)
    _run_gpu_vs_cpu(gramschmidt_test.gramschmidt_kernel, lambda: dict(A=A.copy()), dict(M=M, N=N), rtol=1e-6, atol=1e-8)


@pytest.mark.gpu
def test_doitgen_gpu_matches_cpu():
    NR, NQ, NP = 4, 6, 8
    A, C4 = doitgen_test.initialize(NR, NQ, NP)
    _run_gpu_vs_cpu(doitgen_test.doitgen_kernel, lambda: dict(A=A.copy(), C4=C4.copy()), dict(NR=NR, NQ=NQ, NP=NP))


@pytest.mark.gpu
def test_deriche_gpu_matches_cpu():
    W, H = 16, 20
    alpha, imgIn = deriche_test.initialize(W, H)
    _run_gpu_vs_cpu(deriche_test.deriche_kernel, lambda: dict(alpha=alpha, imgIn=imgIn.copy()), dict(W=W, H=H))


@pytest.mark.gpu
def test_floyd_warshall_gpu_matches_cpu():
    N = 16
    path = floyd_warshall_test.init_data(N)
    _run_gpu_vs_cpu(floyd_warshall_test.kernel, lambda: dict(path=path.copy()), dict(N=N))


@pytest.mark.gpu
def test_nussinov_gpu_matches_cpu():
    N = 16
    seq, _table = nussinov_test.init_data(N)
    _run_gpu_vs_cpu(nussinov_test.kernel, lambda: dict(seq=seq.copy()), dict(N=N))


@pytest.mark.gpu
def test_jacobi_1d_gpu_matches_cpu():
    N = 16
    A, B = jacobi_1d_test.initialize(N)
    _run_gpu_vs_cpu(jacobi_1d_test.jacobi_1d_kernel, lambda: dict(TSTEPS=_TSTEPS_SMALL, A=A.copy(), B=B.copy()),
                    dict(N=N))


@pytest.mark.gpu
def test_jacobi_2d_gpu_matches_cpu():
    N = 16
    A, B = jacobi_2d_test.init_data(N)
    _run_gpu_vs_cpu(jacobi_2d_test.kernel,
                    lambda: dict(TSTEPS=_TSTEPS_SMALL, A=A.copy(), B=B.copy()),
                    dict(N=N),
                    rtol=1e-5,
                    atol=1e-6)


@pytest.mark.gpu
def test_seidel_2d_gpu_matches_cpu():
    N = 16
    A = seidel_2d_test.initialize(N)
    _run_gpu_vs_cpu(seidel_2d_test.seidel_2d_kernel, lambda: dict(TSTEPS=_TSTEPS_SMALL, A=A.copy()), dict(N=N))


@pytest.mark.gpu
def test_heat_3d_gpu_matches_cpu():
    N = 10
    A, B = heat_3d_test.initialize(N)
    _run_gpu_vs_cpu(heat_3d_test.heat_3d_kernel, lambda: dict(TSTEPS=_TSTEPS_SMALL, A=A.copy(), B=B.copy()), dict(N=N))


@pytest.mark.gpu
def test_adi_gpu_matches_cpu():
    N = 16
    u = adi_test.initialize(N)
    _run_gpu_vs_cpu(adi_test.adi_kernel, lambda: dict(TSTEPS=_TSTEPS_SMALL, u=u.copy()), dict(N=N))


@pytest.mark.gpu
def test_fdtd_2d_gpu_matches_cpu():
    NX, NY = 12, 16
    TMAX = _TSTEPS_SMALL
    ex, ey, hz, _fict_ = fdtd_2d_test.init_data(TMAX, NX, NY)
    _run_gpu_vs_cpu(fdtd_2d_test.kernel,
                    lambda: dict(ex=ex.copy(), ey=ey.copy(), hz=hz.copy(), _fict_=_fict_.copy()),
                    dict(TMAX=TMAX, NX=NX, NY=NY),
                    rtol=1e-5,
                    atol=1e-6)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
