# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Numerical-validation matrix for NPBench kernels through the new GPU pipeline.

For each kernel:
  1. Build a CPU SDFG and run it (reference).
  2. Build a fresh SDFG, apply ``apply_gpu_transformations()``, then run the
     explicit GPU stream-management pipeline (``InsertExplicitGPUGlobalMemoryCopies``
     -> stream scheduling/insertion/connection -> sync tasklets), and compile
     with the new GPU codegen.
  3. Run the GPU SDFG on independent copies of the inputs and compare to the
     CPU result element-wise.

Failure modes are surfaced as test failures with the kernel name, plus the
type of failure (compile vs numerical) so the report is easy to read.
"""
import os
import sys
from typing import Callable, Dict

import numpy as np
import pytest

from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import NaiveGPUStreamScheduler
from dace.transformation.passes.gpu_specialization.insert_explicit_gpu_global_memory_copies import (
    InsertExplicitGPUGlobalMemoryCopies, )
from dace.transformation.passes.gpu_specialization.insert_gpu_streams import InsertGPUStreams
from dace.transformation.passes.gpu_specialization.connect_gpu_streams_to_nodes import ConnectGPUStreamsToNodes
from dace.transformation.passes.gpu_specialization.insert_gpu_stream_sync_tasklets import InsertGPUStreamSyncTasklets

# Make the existing polybench / NPBench kernel modules importable.
_TESTS_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_DIR = os.path.abspath(os.path.join(_TESTS_DIR, os.pardir))
for sub in ('npbench/polybench', 'npbench/misc', 'npbench/weather_stencils'):
    p = os.path.join(_REPO_DIR, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

# Polybench (already in this branch).
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

# NPBench misc / weather (added as part of this PR).
import cavity_flow_test  # noqa: E402
import channel_flow_test  # noqa: E402
import hdiff_test  # noqa: E402
import vadv_test  # noqa: E402

_GPU_STREAM_PIPELINE = Pipeline([
    InsertExplicitGPUGlobalMemoryCopies(),
    NaiveGPUStreamScheduler(),
    InsertGPUStreams(),
    ConnectGPUStreamsToNodes(),
    InsertGPUStreamSyncTasklets(),
])

_TSTEPS_SMALL = 3


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


def _run_through_new_gpu_pipeline(kernel,
                                  build_args: Callable[[], Dict[str, np.ndarray]],
                                  symbols: Dict[str, int],
                                  *,
                                  rtol: float = 1e-10,
                                  atol: float = 1e-12) -> None:
    """Build CPU and new-pipeline GPU SDFGs, run both, compare. Compilation
    failures and numerical mismatches surface as ``pytest.fail`` with a tag
    so the matrix output classifies them at a glance."""
    cpu_sdfg = kernel.to_sdfg(simplify=True)
    cpu_args = build_args()
    cpu_ret = cpu_sdfg(**cpu_args, **symbols)

    gpu_sdfg = kernel.to_sdfg(simplify=True)
    gpu_sdfg.apply_gpu_transformations()

    # The codegen refuses to re-run the stream pipeline on an already-lowered
    # SDFG, so we must expand library nodes first -- otherwise kernels created
    # by later expansion would never get stream wiring.
    gpu_sdfg.expand_library_nodes()

    # New explicit-stream-management pipeline.
    _GPU_STREAM_PIPELINE.apply_pass(gpu_sdfg, {})

    try:
        compiled = gpu_sdfg.compile()
    except Exception as e:  # pragma: no cover - expected to fail on some kernels
        pytest.fail(f'COMPILE_FAIL: {type(e).__name__}: {e}', pytrace=False)

    gpu_args = build_args()
    try:
        gpu_ret = compiled(**gpu_args, **symbols)
    except Exception as e:  # pragma: no cover
        pytest.fail(f'RUNTIME_FAIL: {type(e).__name__}: {e}', pytrace=False)

    try:
        _compare_arrays(cpu_args, gpu_args, rtol, atol)
        _compare_returns(cpu_ret, gpu_ret, rtol, atol)
    except AssertionError as e:
        pytest.fail(f'NUMERICAL_FAIL: {e}', pytrace=False)


# --- Polybench kernel cases -----------------------------------------------------


@pytest.mark.gpu
def test_atax():
    M, N = 12, 16
    A, x, _y = atax_test.init_data(M, N)
    _run_through_new_gpu_pipeline(atax_test.kernel,
                                  lambda: dict(A=A.copy(), x=x.copy()),
                                  dict(M=M, N=N),
                                  rtol=1e-5,
                                  atol=1e-6)


@pytest.mark.gpu
def test_bicg():
    M, N = 12, 16
    A, p, r = bicg_test.initialize(M, N)
    _run_through_new_gpu_pipeline(bicg_test.bicg_kernel, lambda: dict(A=A.copy(), p=p.copy(), r=r.copy()), dict(M=M,
                                                                                                                N=N))


@pytest.mark.gpu
def test_gemm():
    NI, NJ, NK = 12, 14, 16
    alpha, beta, C, A, B = gemm_npbench_test.initialize(NI, NJ, NK)
    _run_through_new_gpu_pipeline(gemm_npbench_test.gemm_kernel,
                                  lambda: dict(alpha=alpha, beta=beta, C=C.copy(), A=A.copy(), B=B.copy()),
                                  dict(NI=NI, NJ=NJ, NK=NK))


@pytest.mark.gpu
def test_k2mm():
    NI, NJ, NK, NL = 8, 10, 12, 14
    alpha, beta, A, B, C, D = k2mm_test.initialize(NI, NJ, NK, NL)
    _run_through_new_gpu_pipeline(k2mm_test.k2mm_kernel,
                                  lambda: dict(alpha=alpha, beta=beta, A=A.copy(), B=B.copy(), C=C.copy(), D=D.copy()),
                                  dict(NI=NI, NJ=NJ, NK=NK, NL=NL))


@pytest.mark.gpu
def test_k3mm():
    NI, NJ, NK, NL, NM = 6, 8, 10, 12, 14
    A, B, C, D = k3mm_test.initialize(NI, NJ, NK, NL, NM)
    _run_through_new_gpu_pipeline(k3mm_test.k3mm_kernel, lambda: dict(A=A.copy(), B=B.copy(), C=C.copy(), D=D.copy()),
                                  dict(NI=NI, NJ=NJ, NK=NK, NL=NL, NM=NM))


@pytest.mark.gpu
def test_mvt():
    N = 16
    x1, x2, y_1, y_2, A = mvt_test.initialize(N)
    _run_through_new_gpu_pipeline(mvt_test.mvt_kernel,
                                  lambda: dict(x1=x1.copy(), x2=x2.copy(), y_1=y_1.copy(), y_2=y_2.copy(), A=A.copy()),
                                  dict(N=N))


@pytest.mark.gpu
def test_gesummv():
    N = 16
    alpha, beta, A, B, x = gesummv_test.initialize(N)
    _run_through_new_gpu_pipeline(gesummv_test.gesummv_kernel,
                                  lambda: dict(alpha=alpha, beta=beta, A=A.copy(), B=B.copy(), x=x.copy()), dict(N=N))


@pytest.mark.gpu
def test_gemver():
    N = 16
    alpha, beta, A, u1, v1, u2, v2, w, x, y, z = gemver_test.initialize(N)
    _run_through_new_gpu_pipeline(
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
def test_syrk():
    N, M = 12, 16
    alpha, beta, C, A = syrk_test.init_data(N, M)
    _run_through_new_gpu_pipeline(syrk_test.kernel,
                                  lambda: dict(alpha=alpha, beta=beta, C=C.copy(), A=A.copy()),
                                  dict(M=M, N=N),
                                  rtol=1e-5,
                                  atol=1e-6)


@pytest.mark.gpu
def test_syr2k():
    N, M = 12, 16
    alpha, beta, C, A, B = syr2k_test.initialize(N, M)
    _run_through_new_gpu_pipeline(syr2k_test.syr2k_kernel,
                                  lambda: dict(alpha=alpha, beta=beta, C=C.copy(), A=A.copy(), B=B.copy()),
                                  dict(M=M, N=N))


@pytest.mark.gpu
def test_symm():
    M, N = 12, 16
    alpha, beta, C, A, B = symm_test.initialize(M, N)
    _run_through_new_gpu_pipeline(symm_test.symm_kernel,
                                  lambda: dict(alpha=alpha, beta=beta, C=C.copy(), A=A.copy(), B=B.copy()),
                                  dict(M=M, N=N))


@pytest.mark.gpu
def test_trmm():
    M, N = 12, 16
    alpha, A, B = trmm_test.initialize(M, N)
    _run_through_new_gpu_pipeline(trmm_test.trmm_kernel, lambda: dict(alpha=alpha, A=A.copy(), B=B.copy()),
                                  dict(M=M, N=N))


@pytest.mark.gpu
def test_trisolv():
    N = 16
    L, x, b = trisolv_test.initialize(N)
    _run_through_new_gpu_pipeline(trisolv_test.trisolv_kernel, lambda: dict(L=L.copy(), x=x.copy(), b=b.copy()),
                                  dict(N=N))


@pytest.mark.gpu
def test_durbin():
    N = 16
    r = durbin_test.initialize(N)
    _run_through_new_gpu_pipeline(durbin_test.durbin_kernel, lambda: dict(r=r.copy()), dict(N=N))


@pytest.mark.gpu
def test_cholesky():
    N = 16
    A = cholesky_test.init_data(N)
    _run_through_new_gpu_pipeline(cholesky_test.kernel, lambda: dict(A=A.copy()), dict(N=N), rtol=1e-4, atol=1e-5)


@pytest.mark.gpu
def test_cholesky2():
    N = 16
    A = cholesky2_test.init_data(N)
    _run_through_new_gpu_pipeline(cholesky2_test.cholesky2_kernel, lambda: dict(A=A.copy()), dict(N=N))


@pytest.mark.gpu
def test_lu():
    N = 16
    A = lu_test.init_data(N)
    _run_through_new_gpu_pipeline(lu_test.lu_kernel, lambda: dict(A=A.copy()), dict(N=N), rtol=1e-4, atol=1e-5)


@pytest.mark.gpu
def test_ludcmp():
    N = 16
    A, b = ludcmp_test.initialize(N)
    _run_through_new_gpu_pipeline(ludcmp_test.ludcmp_kernel, lambda: dict(A=A.copy(), b=b.copy()), dict(N=N))


@pytest.mark.gpu
def test_correlation():
    M, N = 12, 16
    float_n, data = correlation_test.initialize(M, N)
    _run_through_new_gpu_pipeline(correlation_test.correlation_kernel, lambda: dict(float_n=float_n, data=data.copy()),
                                  dict(M=M, N=N))


@pytest.mark.gpu
def test_covariance():
    M, N = 12, 16
    float_n, data = covariance_test.init_data(M, N)
    _run_through_new_gpu_pipeline(covariance_test.covariance_kernel,
                                  lambda: dict(float_n=float_n, data=data.copy()),
                                  dict(M=M, N=N),
                                  rtol=1e-4,
                                  atol=1e-5)


@pytest.mark.gpu
def test_gramschmidt():
    M, N = 14, 10
    A = gramschmidt_test.initialize(M, N)
    _run_through_new_gpu_pipeline(gramschmidt_test.gramschmidt_kernel,
                                  lambda: dict(A=A.copy()),
                                  dict(M=M, N=N),
                                  rtol=1e-6,
                                  atol=1e-8)


@pytest.mark.gpu
def test_doitgen():
    NR, NQ, NP = 4, 6, 8
    A, C4 = doitgen_test.initialize(NR, NQ, NP)
    _run_through_new_gpu_pipeline(doitgen_test.doitgen_kernel, lambda: dict(A=A.copy(), C4=C4.copy()),
                                  dict(NR=NR, NQ=NQ, NP=NP))


@pytest.mark.gpu
def test_deriche():
    W, H = 16, 20
    alpha, imgIn = deriche_test.initialize(W, H)
    _run_through_new_gpu_pipeline(deriche_test.deriche_kernel, lambda: dict(alpha=alpha, imgIn=imgIn.copy()),
                                  dict(W=W, H=H))


@pytest.mark.gpu
def test_floyd_warshall():
    N = 16
    path = floyd_warshall_test.init_data(N)
    _run_through_new_gpu_pipeline(floyd_warshall_test.kernel, lambda: dict(path=path.copy()), dict(N=N))


@pytest.mark.gpu
def test_nussinov():
    N = 16
    seq, _table = nussinov_test.init_data(N)
    _run_through_new_gpu_pipeline(nussinov_test.kernel, lambda: dict(seq=seq.copy()), dict(N=N))


@pytest.mark.gpu
def test_jacobi_1d():
    N = 16
    A, B = jacobi_1d_test.initialize(N)
    _run_through_new_gpu_pipeline(jacobi_1d_test.jacobi_1d_kernel,
                                  lambda: dict(TSTEPS=_TSTEPS_SMALL, A=A.copy(), B=B.copy()), dict(N=N))


@pytest.mark.gpu
def test_jacobi_2d():
    N = 16
    A, B = jacobi_2d_test.init_data(N)
    _run_through_new_gpu_pipeline(jacobi_2d_test.kernel,
                                  lambda: dict(TSTEPS=_TSTEPS_SMALL, A=A.copy(), B=B.copy()),
                                  dict(N=N),
                                  rtol=1e-5,
                                  atol=1e-6)


@pytest.mark.gpu
def test_seidel_2d():
    N = 16
    A = seidel_2d_test.initialize(N)
    _run_through_new_gpu_pipeline(seidel_2d_test.seidel_2d_kernel, lambda: dict(TSTEPS=_TSTEPS_SMALL, A=A.copy()),
                                  dict(N=N))


@pytest.mark.gpu
def test_heat_3d():
    N = 10
    A, B = heat_3d_test.initialize(N)
    _run_through_new_gpu_pipeline(heat_3d_test.heat_3d_kernel,
                                  lambda: dict(TSTEPS=_TSTEPS_SMALL, A=A.copy(), B=B.copy()), dict(N=N))


@pytest.mark.gpu
def test_adi():
    N = 16
    u = adi_test.initialize(N)
    _run_through_new_gpu_pipeline(adi_test.adi_kernel, lambda: dict(TSTEPS=_TSTEPS_SMALL, u=u.copy()), dict(N=N))


@pytest.mark.gpu
def test_fdtd_2d():
    NX, NY = 12, 16
    TMAX = _TSTEPS_SMALL
    ex, ey, hz, _fict_ = fdtd_2d_test.init_data(TMAX, NX, NY)
    _run_through_new_gpu_pipeline(fdtd_2d_test.kernel,
                                  lambda: dict(ex=ex.copy(), ey=ey.copy(), hz=hz.copy(), _fict_=_fict_.copy()),
                                  dict(TMAX=TMAX, NX=NX, NY=NY),
                                  rtol=1e-5,
                                  atol=1e-6)


# --- NPBench misc / weather kernels (newly ported) -----------------------------


@pytest.mark.gpu
def test_cavity_flow():
    """Lid-driven cavity flow (NPBench misc), a small Navier-Stokes solver."""
    ny, nx, nt, nit, rho, nu = 21, 21, 4, 5, 1.0, 0.1
    u, v, p, dx, dy, dt = cavity_flow_test.initialize(ny, nx)
    build_args = lambda: dict(nt=nt, nit=nit, u=u.copy(), v=v.copy(), dt=dt, dx=dx, dy=dy, p=p.copy(), rho=rho, nu=nu)
    _run_through_new_gpu_pipeline(cavity_flow_test.dace_cavity_flow,
                                  build_args,
                                  dict(ny=ny, nx=nx),
                                  rtol=1e-6,
                                  atol=1e-8)


@pytest.mark.gpu
def test_channel_flow():
    """Channel flow with periodic BC (NPBench misc)."""
    ny, nx, nit, rho, nu, F = 21, 21, 5, 1.0, 0.1, 1.0
    u, v, p, dx, dy, dt = channel_flow_test.initialize(ny, nx)
    build_args = lambda: dict(nit=nit, u=u.copy(), v=v.copy(), dt=dt, dx=dx, dy=dy, p=p.copy(), rho=rho, nu=nu, F=F)
    _run_through_new_gpu_pipeline(channel_flow_test.dace_channel_flow,
                                  build_args,
                                  dict(ny=ny, nx=nx),
                                  rtol=1e-6,
                                  atol=1e-8)


@pytest.mark.gpu
def test_hdiff():
    """Horizontal diffusion stencil (NPBench weather)."""
    I, J, K = 16, 16, 8
    in_field, out_field, coeff = hdiff_test.initialize(I, J, K)
    build_args = lambda: dict(in_field=in_field.copy(), out_field=out_field.copy(), coeff=coeff.copy())
    _run_through_new_gpu_pipeline(hdiff_test.hdiff_kernel, build_args, dict(I=I, J=J, K=K), rtol=1e-10, atol=1e-12)


@pytest.mark.gpu
def test_vadv():
    """Vertical advection stencil (NPBench weather)."""
    I, J, K = 16, 16, 8
    dtr_stage, utens_stage, u_stage, wcon, u_pos, utens = vadv_test.initialize(I, J, K)
    build_args = lambda: dict(utens_stage=utens_stage.copy(),
                              u_stage=u_stage.copy(),
                              wcon=wcon.copy(),
                              u_pos=u_pos.copy(),
                              utens=utens.copy(),
                              dtr_stage=dtr_stage)
    _run_through_new_gpu_pipeline(vadv_test.vadv_kernel, build_args, dict(I=I, J=J, K=K), rtol=1e-10, atol=1e-12)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
