# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""numpy references for the polybench corpus -- the *parallel numpy* baseline.

One ``ref_<kernel>`` per polybench kernel, keyed in :data:`REFERENCES` by the kernel's
module name (``k2mm``/``k3mm`` are polybench's ``2mm``/``3mm``). Each takes the corpus
kernel's own parameter names plus the dataset symbols it needs, writes its results into
the arrays it is handed (the polybench kernels are all in-place), and returns nothing --
:func:`tests.corpus.polybench.polybench.numpy_reference` hands it copies and reads the
arrays back out, exactly as the untransformed-SDFG reference does.

Provenance: the bodies are the numpy implementations from the npbench suite
(``npbench/benchmarks/polybench/<kernel>/<kernel>_numpy.py``, ETH Zurich / npbench
authors, BSD-3), COPIED IN and adapted to the in-repo signatures -- npbench passes
scalars where this corpus passes 1-element arrays (``alpha[0]``), returns freshly
allocated outputs where this corpus writes into caller arrays, and takes array shapes
where this corpus is parametrized on dataset symbols. There is deliberately NO import of
an external npbench checkout.

These kernels needed more than a signature adaptation, because npbench's numpy file does
not compute what the in-repo kernel computes:

* ``adi``      -- npbench uses ``b = 1 + mul2``; PolyBench/C and the in-repo kernel use
  ``b = 1 + mul1``. The in-repo constants are reproduced here.
* ``deriche``  -- npbench drops the factor 2 from ``k``'s denominator
  (``1 + alpha*exp(-alpha)`` instead of ``1 + 2*alpha*exp(-alpha)``). The filter constants
  are imported from the kernel module so the two can never drift.
* ``cholesky`` -- the in-repo kernel *is* ``np.linalg.cholesky`` (a POTRF library node), so
  the numpy correspondent is LAPACK, not npbench's hand-rolled Crout loop.
* ``nussinov`` -- npbench's is a scalar python triple loop (O(N^3) interpreter iterations,
  ~10^7 at the ``paper`` preset). The inner ``k`` split is a max-reduction over a vector,
  so it is expressed as one; the ``i``/``j`` walk stays sequential (it is a DP).
* ``correlation`` / ``covariance`` / ``gemver`` / ``ludcmp`` / ``gramschmidt`` / ``bicg`` /
  ``durbin`` -- npbench returns newly allocated outputs; here they are written into the
  corpus's own output arrays (``mean``, ``stddev``, ``R``, ``Q``, ``x``, ``y``, ...).

WARNING: Vectorization status is NOT uniform, and a figure captioned "vs parallel numpy" must
say so. :data:`VECTORIZATION` records, per kernel, whether the reference is array-level
throughout, array-level inside a python loop whose trip count grows with the problem, or a
genuine SCALAR python loop. ``seidel_2d`` is the last kind: its in-row Gauss-Seidel scan is
a first-order recurrence with no closed numpy form that does not overflow, so its numpy
reference interprets ~10^7 scalar updates at the ``paper`` preset and is NOT a defensible
timing denominator.
"""
from typing import Callable, Dict

import numpy as np

# The Deriche filter coefficients are part of the problem definition, not of the algorithm
# the reference re-implements; importing them keeps the two from drifting apart silently.
from tests.corpus.polybench.medley.deriche import a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, c1, c2

#: Per-kernel vectorization class of the numpy reference below. This is the honesty record
#: for the "vs parallel numpy" figure:
#:
#: * ``'array'``  -- array-level throughout; any python loop is a time-step / sweep loop whose
#:   body touches the whole domain, so interpreter overhead is negligible next to BLAS/ufunc work.
#: * ``'blocked'`` -- array-level *operations* driven by a python loop whose trip count grows with
#:   the dataset (O(N) or O(N^2) interpreter iterations). Real numpy, but interpreter-bound at
#:   small extents; report the trip count next to any speedup taken against it.
#: * ``'scalar'`` -- SCALAR python loop over elements. NOT a defensible timing denominator; it is
#:   the trap that disqualified the tsvc oracles. Must not enter a "vs numpy" figure unlabelled.
VECTORIZATION: Dict[str, str] = {
    'adi': 'blocked',
    'atax': 'array',
    'bicg': 'array',
    'cholesky': 'array',
    'correlation': 'blocked',
    'covariance': 'blocked',
    'deriche': 'blocked',
    'doitgen': 'array',
    'durbin': 'blocked',
    'fdtd_2d': 'array',
    'floyd_warshall': 'array',
    'gemm': 'array',
    'gemver': 'array',
    'gesummv': 'array',
    'gramschmidt': 'blocked',
    'heat_3d': 'array',
    'jacobi_1d': 'array',
    'jacobi_2d': 'array',
    'k2mm': 'array',
    'k3mm': 'array',
    'lu': 'blocked',
    'ludcmp': 'blocked',
    'mvt': 'array',
    'nussinov': 'blocked',
    'seidel_2d': 'scalar',
    'symm': 'blocked',
    'syr2k': 'blocked',
    'syrk': 'blocked',
    'trisolv': 'blocked',
    'trmm': 'blocked',
}


def ref_adi(u: np.ndarray, N: int, tsteps: int) -> None:
    """Alternating-direction implicit solver. The in-repo kernel's constants are
    ``b = 1 + mul1`` / ``e = 1 + mul2`` (PolyBench/C); npbench's numpy file uses
    ``b = 1 + mul2``, which is a different problem -- these are the in-repo ones."""
    v = np.zeros_like(u)
    p = np.zeros_like(u)
    q = np.zeros_like(u)
    a = -(2.0 * (1.0 / tsteps) / (1.0 / (N * N))) / 2.0
    b = 1.0 + (2.0 * (1.0 / tsteps) / (1.0 / (N * N)))
    c = a
    d = -(1.0 * (1.0 / tsteps) / (1.0 / (N * N))) / 2.0
    e = 1.0 + (1.0 * (1.0 / tsteps) / (1.0 / (N * N)))
    f = d
    for _ in range(tsteps):
        v[0, 1:N - 1] = 1.0
        p[1:N - 1, 0] = 0.0
        q[1:N - 1, 0] = 1.0
        for j in range(1, N - 1):
            p[1:N - 1, j] = -c / (a * p[1:N - 1, j - 1] + b)
            q[1:N - 1, j] = (-d * u[j, 0:N - 2] + (1.0 + 2.0 * d) * u[j, 1:N - 1] - f * u[j, 2:N] -
                             a * q[1:N - 1, j - 1]) / (a * p[1:N - 1, j - 1] + b)
        v[N - 1, 1:N - 1] = 1.0
        for j in range(N - 2, 0, -1):
            v[j, 1:N - 1] = p[1:N - 1, j] * v[j + 1, 1:N - 1] + q[1:N - 1, j]
        u[1:N - 1, 0] = 1.0
        p[1:N - 1, 0] = 0.0
        q[1:N - 1, 0] = 1.0
        for j in range(1, N - 1):
            p[1:N - 1, j] = -f / (d * p[1:N - 1, j - 1] + e)
            q[1:N - 1, j] = (-a * v[0:N - 2, j] + (1.0 + 2.0 * a) * v[1:N - 1, j] - c * v[2:N, j] -
                             d * q[1:N - 1, j - 1]) / (d * p[1:N - 1, j - 1] + e)
        u[1:N - 1, N - 1] = 1.0
        for j in range(N - 2, 0, -1):
            u[1:N - 1, j] = p[1:N - 1, j] * u[1:N - 1, j + 1] + q[1:N - 1, j]


def ref_atax(A: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    y[:] = (A @ x) @ A


def ref_bicg(A: np.ndarray, s: np.ndarray, q: np.ndarray, p: np.ndarray, r: np.ndarray) -> None:
    s[:] = r @ A
    q[:] = A @ p


def ref_cholesky(A: np.ndarray, N: int) -> None:
    """The in-repo kernel is ``np.linalg.cholesky`` (POTRF) writing only the lower triangle
    back, leaving the strictly-upper (symmetric input) triangle alone; mirror that exactly."""
    lower = np.linalg.cholesky(A)
    tri = np.tril_indices(N)
    A[tri] = lower[tri]


def ref_correlation(data: np.ndarray, corr: np.ndarray, mean: np.ndarray, stddev: np.ndarray, M: int, N: int) -> None:
    mean[:] = np.mean(data, axis=0)
    stddev[:] = np.std(data, axis=0)
    stddev[stddev <= 0.1] = 1.0
    data -= mean
    data /= np.sqrt(np.float64(N)) * stddev
    corr[:] = np.eye(M, dtype=data.dtype)
    for i in range(M - 1):
        corr[i + 1:M, i] = corr[i, i + 1:M] = data[:, i] @ data[:, i + 1:M]


def ref_covariance(data: np.ndarray, cov: np.ndarray, mean: np.ndarray, M: int, N: int) -> None:
    mean[:] = np.mean(data, axis=0)
    data -= mean
    for i in range(M):
        cov[i:M, i] = cov[i, i:M] = data[:, i] @ data[:, i:M] / (np.float64(N) - 1.0)


def ref_deriche(imgIn: np.ndarray, imgOut: np.ndarray, W: int, H: int) -> None:
    """Deriche edge detector: four IIR scans. Each scan step is a full-length vector update,
    but the scan axis is a loop-carried recurrence, so it stays a python loop."""
    y1 = np.zeros_like(imgIn)
    y1[:, 0] = a1 * imgIn[:, 0]
    y1[:, 1] = a1 * imgIn[:, 1] + a2 * imgIn[:, 0] + b1 * y1[:, 0]
    for j in range(2, H):
        y1[:, j] = a1 * imgIn[:, j] + a2 * imgIn[:, j - 1] + b1 * y1[:, j - 1] + b2 * y1[:, j - 2]
    y2 = np.zeros_like(imgIn)
    y2[:, H - 1] = 0.0
    y2[:, H - 2] = a3 * imgIn[:, H - 1]
    for j in range(H - 3, -1, -1):
        y2[:, j] = a3 * imgIn[:, j + 1] + a4 * imgIn[:, j + 2] + b1 * y2[:, j + 1] + b2 * y2[:, j + 2]
    imgOut[:] = c1 * (y1 + y2)
    y1[0, :] = a5 * imgOut[0, :]
    y1[1, :] = a5 * imgOut[1, :] + a6 * imgOut[0, :] + b1 * y1[0, :]
    for i in range(2, W):
        y1[i, :] = a5 * imgOut[i, :] + a6 * imgOut[i - 1, :] + b1 * y1[i - 1, :] + b2 * y1[i - 2, :]
    y2[W - 1, :] = 0.0
    y2[W - 2, :] = a7 * imgOut[W - 1, :]
    for i in range(W - 3, -1, -1):
        y2[i, :] = a7 * imgOut[i + 1, :] + a8 * imgOut[i + 2, :] + b1 * y2[i + 1, :] + b2 * y2[i + 2, :]
    imgOut[:] = c2 * (y1 + y2)


def ref_doitgen(A: np.ndarray, C4: np.ndarray) -> None:
    # (NR, NQ, NP) @ (NP, NP) is the batched form of the in-repo kernel's per-``r`` matmul.
    A[:] = A @ C4


def ref_durbin(r: np.ndarray, y: np.ndarray) -> None:
    alpha = -r[0]
    beta = 1.0
    y[0] = -r[0]
    for k in range(1, r.shape[0]):
        beta *= 1.0 - alpha * alpha
        alpha = -(r[k] + np.dot(np.flip(r[:k]), y[:k])) / beta
        y[:k] += alpha * np.flip(y[:k])
        y[k] = alpha


def ref_fdtd_2d(ex: np.ndarray, ey: np.ndarray, hz: np.ndarray, _fict_: np.ndarray, TMAX: int) -> None:
    for t in range(TMAX):
        ey[0, :] = _fict_[t]
        ey[1:, :] -= 0.5 * (hz[1:, :] - hz[:-1, :])
        ex[:, 1:] -= 0.5 * (hz[:, 1:] - hz[:, :-1])
        hz[:-1, :-1] -= 0.7 * (ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] - ey[:-1, :-1])


def ref_floyd_warshall(path: np.ndarray, N: int) -> None:
    for k in range(N):
        path[:] = np.minimum(path, np.add.outer(path[:, k], path[k, :]))


def ref_gemm(C: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> None:
    C[:] = alpha[0] * A @ B + beta[0] * C


def ref_gemver(A: np.ndarray, u1: np.ndarray, v1: np.ndarray, u2: np.ndarray, v2: np.ndarray, w: np.ndarray,
               x: np.ndarray, y: np.ndarray, z: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> None:
    A += np.outer(u1, v1) + np.outer(u2, v2)
    x += beta[0] * y @ A + z
    w += alpha[0] * A @ x


def ref_gesummv(A: np.ndarray, B: np.ndarray, tmp: np.ndarray, x: np.ndarray, y: np.ndarray, alpha: np.ndarray,
                beta: np.ndarray) -> None:
    # ``tmp`` is dead in the npbench formulation the in-repo kernel adopted; it stays in the
    # signature (and untouched) because the corpus compares every argument array.
    y[:] = alpha[0] * A @ x + beta[0] * B @ x


def ref_gramschmidt(A: np.ndarray, R: np.ndarray, Q: np.ndarray, N: int) -> None:
    for k in range(N):
        nrm = np.dot(A[:, k], A[:, k])
        R[k, k] = np.sqrt(nrm)
        Q[:, k] = A[:, k] / R[k, k]
        for j in range(k + 1, N):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] -= Q[:, k] * R[k, j]


def ref_heat_3d(A: np.ndarray, B: np.ndarray, tsteps: int) -> None:
    for _ in range(1, tsteps):
        B[1:-1, 1:-1,
          1:-1] = (0.125 * (A[2:, 1:-1, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[:-2, 1:-1, 1:-1]) + 0.125 *
                   (A[1:-1, 2:, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, :-2, 1:-1]) + 0.125 *
                   (A[1:-1, 1:-1, 2:] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, 1:-1, 0:-2]) + A[1:-1, 1:-1, 1:-1])
        A[1:-1, 1:-1,
          1:-1] = (0.125 * (B[2:, 1:-1, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[:-2, 1:-1, 1:-1]) + 0.125 *
                   (B[1:-1, 2:, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, :-2, 1:-1]) + 0.125 *
                   (B[1:-1, 1:-1, 2:] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, 1:-1, 0:-2]) + B[1:-1, 1:-1, 1:-1])


def ref_jacobi_1d(A: np.ndarray, B: np.ndarray, tsteps: int) -> None:
    for _ in range(1, tsteps):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])


def ref_jacobi_2d(A: np.ndarray, B: np.ndarray, tsteps: int) -> None:
    for _ in range(1, tsteps):
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] + A[2:, 1:-1] + A[:-2, 1:-1])
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] + B[2:, 1:-1] + B[:-2, 1:-1])


def ref_k2mm(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> None:
    D[:] = alpha[0] * A @ B @ C + beta[0] * D


def ref_k3mm(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, G: np.ndarray) -> None:
    G[:] = A @ B @ C @ D


def ref_lu(A: np.ndarray, N: int) -> None:
    for i in range(N):
        for j in range(i):
            A[i, j] -= A[i, :j] @ A[:j, j]
            A[i, j] /= A[j, j]
        for j in range(i, N):
            A[i, j] -= A[i, :i] @ A[:i, j]


def ref_ludcmp(A: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray, N: int) -> None:
    for i in range(N):
        for j in range(i):
            A[i, j] -= A[i, :j] @ A[:j, j]
            A[i, j] /= A[j, j]
        for j in range(i, N):
            A[i, j] -= A[i, :i] @ A[:i, j]
    for i in range(N):
        y[i] = b[i] - A[i, :i] @ y[:i]
    for i in range(N - 1, -1, -1):
        x[i] = (y[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]


def ref_mvt(x1: np.ndarray, x2: np.ndarray, y_1: np.ndarray, y_2: np.ndarray, A: np.ndarray) -> None:
    x1 += A @ y_1
    x2 += y_2 @ A


def ref_nussinov(seq: np.ndarray, table: np.ndarray, N: int) -> None:
    """RNA folding DP. The ``i``/``j`` walk is a genuine dependence and stays sequential; the
    inner ``k`` split is a max-reduction, so it is one vector op instead of npbench's third
    python loop (which would interpret ~10^7 iterations at the ``paper`` preset)."""
    for i in range(N - 1, -1, -1):
        for j in range(i + 1, N):
            best = table[i, j]
            if j - 1 >= 0:
                best = max(best, table[i, j - 1])
            if i + 1 < N:
                best = max(best, table[i + 1, j])
            if j - 1 >= 0 and i + 1 < N:
                paired = table[i + 1, j - 1]
                if i < j - 1 and seq[i] + seq[j] == 3:
                    paired = paired + 1
                best = max(best, paired)
            if j > i + 1:
                best = max(best, (table[i, i + 1:j] + table[i + 2:j + 1, j]).max())
            table[i, j] = best


def ref_seidel_2d(A: np.ndarray, N: int, tsteps: int) -> None:
    """WARNING: SCALAR python loop. The in-row update ``A[i, j] = (A[i, j] + A[i, j-1]) / 9`` is a
    first-order recurrence; its closed form scales by ``9**-j``, which underflows/overflows
    fp64 well before the ``paper`` extent, so there is no numpy formulation. This reference
    interprets O(tsteps * N^2) scalar updates and is NOT a defensible timing denominator."""
    for _ in range(0, tsteps - 1):
        for i in range(1, N - 1):
            A[i, 1:-1] += (A[i - 1, :-2] + A[i - 1, 1:-1] + A[i - 1, 2:] + A[i, 2:] + A[i + 1, :-2] + A[i + 1, 1:-1] +
                           A[i + 1, 2:])
            for j in range(1, N - 1):
                A[i, j] += A[i, j - 1]
                A[i, j] /= 9.0


def ref_symm(C: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: np.ndarray, beta: np.ndarray, M: int, N: int) -> None:
    temp2 = np.zeros((N, ), dtype=C.dtype)
    C *= beta[0]
    for i in range(M):
        for j in range(N):
            C[:i, j] += alpha[0] * B[i, j] * A[i, :i]
            temp2[j] = B[:i, j] @ A[i, :i]
        C[i, :] += alpha[0] * B[i, :] * A[i, i] + alpha[0] * temp2


def ref_syr2k(C: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: np.ndarray, beta: np.ndarray, M: int, N: int) -> None:
    for i in range(N):
        C[i, :i + 1] *= beta[0]
        for k in range(M):
            C[i, :i + 1] += A[:i + 1, k] * alpha[0] * B[i, k] + B[:i + 1, k] * alpha[0] * A[i, k]


def ref_syrk(C: np.ndarray, A: np.ndarray, alpha: np.ndarray, beta: np.ndarray, M: int, N: int) -> None:
    for i in range(N):
        C[i, :i + 1] *= beta[0]
        for k in range(M):
            C[i, :i + 1] += alpha[0] * A[i, k] * A[:i + 1, k]


def ref_trisolv(L: np.ndarray, x: np.ndarray, b: np.ndarray, N: int) -> None:
    for i in range(N):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]


def ref_trmm(A: np.ndarray, B: np.ndarray, alpha: np.ndarray, M: int, N: int) -> None:
    for i in range(M):
        for j in range(N):
            B[i, j] += np.dot(A[i + 1:, i], B[i + 1:, j])
    B *= alpha[0]


#: kernel module name -> numpy reference. Parameters are resolved BY NAME against the corpus's
#: ``call_arrays`` and dataset symbols, so a signature here must use the in-repo kernel's own
#: argument names (``_fict_``, ``y_1``, ``imgIn``, ...) and symbol names (``tsteps``, ``TMAX``).
REFERENCES: Dict[str, Callable[..., None]] = {
    'adi': ref_adi,
    'atax': ref_atax,
    'bicg': ref_bicg,
    'cholesky': ref_cholesky,
    'correlation': ref_correlation,
    'covariance': ref_covariance,
    'deriche': ref_deriche,
    'doitgen': ref_doitgen,
    'durbin': ref_durbin,
    'fdtd_2d': ref_fdtd_2d,
    'floyd_warshall': ref_floyd_warshall,
    'gemm': ref_gemm,
    'gemver': ref_gemver,
    'gesummv': ref_gesummv,
    'gramschmidt': ref_gramschmidt,
    'heat_3d': ref_heat_3d,
    'jacobi_1d': ref_jacobi_1d,
    'jacobi_2d': ref_jacobi_2d,
    'k2mm': ref_k2mm,
    'k3mm': ref_k3mm,
    'lu': ref_lu,
    'ludcmp': ref_ludcmp,
    'mvt': ref_mvt,
    'nussinov': ref_nussinov,
    'seidel_2d': ref_seidel_2d,
    'symm': ref_symm,
    'syr2k': ref_syr2k,
    'syrk': ref_syrk,
    'trisolv': ref_trisolv,
    'trmm': ref_trmm,
}
