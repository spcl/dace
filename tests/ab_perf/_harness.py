"""Shared timing / dispatch helpers for A/B perf tests.

Two device kinds: ``'cpu'`` and ``'gpu'``. CPU timings use
``time.perf_counter`` around a compiled SDFG call. GPU timings use
``cupy.cuda.Event`` to bracket the call so async kernel time is captured.
Arrays are passed PRE-PLACED on the target device (CPU = numpy, GPU =
cupy) so the timed window measures kernel execution only, not host->device
transfer.
"""
import statistics
import time
from typing import Callable, Dict, List


def _gpu_sync():
    import cupy
    cupy.cuda.runtime.deviceSynchronize()


def ensure_relaxed_constexpr_nvcc():
    """Append ``--expt-relaxed-constexpr`` to the nvcc args if not already
    present. Required for several library-node expansions (e.g. ``Scan(pure)``)
    that use ``std::plus`` / ``std::min`` style ``constexpr`` host functors
    inside ``__global__`` device code; without the flag nvcc rejects them.

    Idempotent -- safe to call multiple times per session."""
    import dace
    cur = dace.Config.get('compiler', 'cuda', 'args')
    if '--expt-relaxed-constexpr' in cur:
        return
    dace.Config.set('compiler', 'cuda', 'args', value=cur + ' --expt-relaxed-constexpr')


def ensure_gpu_heap(min_bytes: int = 256 * 1024 * 1024):
    """Raise the CUDA device-side heap size to at least ``min_bytes`` (default
    256 MB). DaCe-generated kernels that allocate per-thread VLA transients
    via device-side ``new[]`` go through this heap. The CUDA default is
    only 8 MB, which overflows for any large parallel-Map kernel.

    Idempotent: only raises, never lowers. Requires cupy."""
    import cupy
    # cuda.runtime.getLimit/setLimit semantics: 0x02 = cudaLimitMallocHeapSize.
    try:
        cur = cupy.cuda.runtime.deviceGetLimit(2)
    except Exception:
        cur = 0
    if cur >= min_bytes:
        return
    cupy.cuda.runtime.deviceSetLimit(2, min_bytes)


def time_cpu(fn: Callable[[], None], iters: int, warmup: int = 1) -> Dict[str, float]:
    """Time ``fn`` on the CPU. Returns a dict with median / min / max / mean
    in microseconds. ``fn`` must be a zero-argument callable; pre-bind your
    SDFG with its arguments via ``lambda`` or ``functools.partial`` before
    calling. Numpy arrays are assumed to live in host memory."""
    for _ in range(warmup):
        fn()
    samples: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1e6)
    return _stats(samples)


def time_gpu(fn: Callable[[], None], iters: int, warmup: int = 1) -> Dict[str, float]:
    """Time ``fn`` on the GPU using ``cupy.cuda.Event``. The kernel's input
    AccessNodes must already point at device-resident arrays (e.g. cupy
    ndarrays); the timed window measures kernel execution and any
    intra-kernel D2D copies, but NOT H2D / D2H transfers."""
    import cupy
    for _ in range(warmup):
        fn()
    _gpu_sync()
    start_ev = [cupy.cuda.Event() for _ in range(iters)]
    end_ev = [cupy.cuda.Event() for _ in range(iters)]
    for i in range(iters):
        start_ev[i].record()
        fn()
        end_ev[i].record()
    _gpu_sync()
    # cupy.cuda.get_elapsed_time returns ms.
    samples = [cupy.cuda.get_elapsed_time(start_ev[i], end_ev[i]) * 1000.0 for i in range(iters)]
    return _stats(samples)


def _stats(samples_us: List[float]) -> Dict[str, float]:
    return {
        'median_us': statistics.median(samples_us),
        'min_us': min(samples_us),
        'max_us': max(samples_us),
        'mean_us': statistics.mean(samples_us),
        'n': len(samples_us),
    }


def format_ab(label_a: str, stats_a: Dict[str, float], label_b: str, stats_b: Dict[str, float]) -> str:
    """One-line tabular A/B summary."""
    med_a, med_b = stats_a['median_us'], stats_b['median_us']
    speedup = med_a / med_b if med_b > 0 else float('nan')
    return (f'  {label_a:<30s} median {med_a:>10.1f} us  (min {stats_a["min_us"]:>10.1f}, '
            f'max {stats_a["max_us"]:>10.1f}, n={stats_a["n"]})\n'
            f'  {label_b:<30s} median {med_b:>10.1f} us  (min {stats_b["min_us"]:>10.1f}, '
            f'max {stats_b["max_us"]:>10.1f}, n={stats_b["n"]})\n'
            f'  speedup ({label_a} / {label_b}) = {speedup:.3f}x')


def to_gpu(np_array):
    """Move a numpy array onto the GPU (as a cupy ndarray). Asserts CUDA
    + cupy are available; tests should gate the GPU branch on
    ``ab_gpu_enabled`` first."""
    import cupy
    return cupy.asarray(np_array)


def to_cpu(cp_array):
    """Bring a cupy ndarray back to host."""
    import cupy
    return cupy.asnumpy(cp_array)
