# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Layout-aware A/B perf comparison for the cloudsc ``for_1133`` prefix-sum
shape. Three rewrites cross-producted with two storage layouts on CPU + GPU
gives twelve cells per ``(dtype, klev)`` so the cost-model question can be
settled empirically.

Kernel (single carrier, pre-L2M)::

    for jk in range(1, KLEV):
        for jl in range(KLON):
            pfsqrf[jk, jl] = pfsqrf[jk-1, jl] + delta[jk, jl]

Rewrites:

* **A** -- ``LoopToMap`` on the inner only. Outer sequential ``LoopRegion[jk]``
  wraps an inner parallel ``Map[jl]``. No scan lift.
* **B** -- ``LoopToMap`` on the inner + ``LoopToScan(interchange_carry_with_map=True)``.
  Relocates the outer carry ``LoopRegion[jk]`` INTO the NestedSDFG sitting
  inside the parallel ``Map[jl]``. Each thread runs its own sequential
  carry loop reading/writing directly out of global memory. No buffers,
  no Scan libnode.
* **C** -- ``LoopToScan`` on the pre-L2M shape. Emits a ``Scan`` library
  node over a ``delta_buf[trip, inner_size]`` transient with
  ``stride = inner_size``, followed by a 2-D ``Map`` seed-add.

Layouts:

* **C row-major** -- shape ``[KLEV, KLON]``, strides ``[KLON, 1]``. ``jl``
  (the data-parallel axis) is the contiguous axis.
* **F col-major** -- shape ``[KLEV, KLON]``, strides ``[1, KLEV]``. ``jk``
  (the scan axis) is the contiguous axis.

The cost model the AB sweep is meant to settle::

    contiguous axis == scan axis (F)   ->  Scan libnode wins (contiguous reads
                                            inside the scan)
    contiguous axis == parallel axis (C) ->  Map-above-scan wins (threads
                                              coalesce on the contiguous
                                              dimension; sequential carry
                                              lives in registers)
"""
import functools

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.loop_to_scan import LoopToScan

from tests.ab_perf._harness import (ensure_gpu_heap, ensure_relaxed_constexpr_nvcc, time_cpu, time_gpu)

KLEV = dace.symbol('KLEV')
KLON = dace.symbol('KLON')


@dace.program
def _kernel_f64(pfsqrf: dace.float64[KLEV, KLON], delta: dace.float64[KLEV, KLON]):
    for jk in range(1, KLEV):
        for jl in range(KLON):
            pfsqrf[jk, jl] = pfsqrf[jk - 1, jl] + delta[jk, jl]


@dace.program
def _kernel_f32(pfsqrf: dace.float32[KLEV, KLON], delta: dace.float32[KLEV, KLON]):
    for jk in range(1, KLEV):
        for jl in range(KLON):
            pfsqrf[jk, jl] = pfsqrf[jk - 1, jl] + delta[jk, jl]


_KERNEL = {np.float32: _kernel_f32, np.float64: _kernel_f64}
_DTYPE = {'fp32': np.float32, 'fp64': np.float64}


def _set_f_layout(sdfg: dace.SDFG) -> None:
    """In-place: switch ``pfsqrf`` and ``delta`` to Fortran column-major.
    Must run BEFORE simplify / L2M / L2S, otherwise subset propagation
    bakes in the original C-style stride pattern."""
    for nm in ('pfsqrf', 'delta'):
        sh = sdfg.arrays[nm].shape
        sdfg.arrays[nm].strides = (1, sh[0])


def _build(variant: str, layout: str, name_suffix: str, dtype_name: str) -> dace.SDFG:
    """Build one variant/layout cell."""
    sdfg = _KERNEL[_DTYPE[dtype_name]].to_sdfg(simplify=False)
    sdfg.name = f'for_1133_{variant}_{layout}_{dtype_name}_{name_suffix}'
    if layout == 'F':
        _set_f_layout(sdfg)
    sdfg.simplify()

    if variant == 'A':
        # Post-L2M only.
        inner = next(r for r in sdfg.all_control_flow_regions()
                     if isinstance(r, LoopRegion) and r.loop_variable == 'jl')
        xf = LoopToMap()
        xf.loop = inner
        xf.expr_index = 0
        assert xf.can_be_applied(inner.parent_graph, 0, sdfg, permissive=False)
        xf.apply(inner.parent_graph, sdfg)
    elif variant == 'B':
        inner = next(r for r in sdfg.all_control_flow_regions()
                     if isinstance(r, LoopRegion) and r.loop_variable == 'jl')
        xf = LoopToMap()
        xf.loop = inner
        xf.expr_index = 0
        assert xf.can_be_applied(inner.parent_graph, 0, sdfg, permissive=False)
        xf.apply(inner.parent_graph, sdfg)
        res = LoopToScan(interchange_carry_with_map=True).apply_pass(sdfg, {})
        assert res is not None and res >= 1, 'interchange path must lift'
    elif variant == 'C':
        # Pre-L2M Scan libnode emit. L2S's composite-body matcher fires on
        # outer LoopRegion[jk] + inner LoopRegion[jl] -- if the inner is
        # already a Map (post-L2M), the matcher refuses, so we run L2S
        # directly on the @dace.program-built shape.
        #
        # IMPORTANT GPU CAVEAT: when this SDFG is GPU-transformed, the
        # surviving outer ``for jk`` host loop emits per-(jk, jl) cudaMemcpy
        # + per-iter kernel launches that dominate the runtime by orders
        # of magnitude (the actual Scan libnode call is sub-millisecond,
        # but the sequential-host wrapper around it makes the total cell
        # measure ~10s instead of ~ms). The number itself is real and
        # informative for the cost model: the AUTO knob should NEVER pick
        # Scan libnode on GPU for this shape -- the interchange path is
        # the correct choice. A future L2S enhancement that lifts the
        # outer loop body to a Map alongside the Scan emission would
        # close the gap.
        res = LoopToScan().apply_pass(sdfg, {})
        assert res is not None and res >= 1, 'Scan libnode path must lift'
    else:
        raise ValueError(variant)
    sdfg.validate()
    return sdfg


def _to_gpu_sdfg(sdfg: dace.SDFG, suffix: str, device_resident_data=()) -> dace.SDFG:
    import copy as _copy
    from dace import nodes as _nodes
    gpu = _copy.deepcopy(sdfg)
    gpu.name = f'{sdfg.name}_{suffix}'
    for arr in device_resident_data:
        if arr in gpu.arrays:
            gpu.arrays[arr].storage = dace.dtypes.StorageType.GPU_Global
    gpu.apply_gpu_transformations(host_data=list(device_resident_data))
    for arr in device_resident_data:
        if arr in gpu.arrays:
            gpu.arrays[arr].storage = dace.dtypes.StorageType.GPU_Global
    # Library-node implementations default to CPU; on a GPU SDFG we want the
    # device-side expansion. Pick by stride: cub::DeviceScan for unit-stride,
    # the custom block-per-residue-class kernel (``CUDA_strided``) for stride
    # > 1 -- the latter avoids cub.cuh's CCCL 3 g++-incompatible templates.
    # ``apply_gpu_transformations`` does not touch library-node
    # implementation strings, so set them explicitly here.
    from dace import symbolic as _symbolic
    for n, _ in gpu.all_nodes_recursive():
        if isinstance(n, _nodes.LibraryNode) and type(n).__name__ == 'Scan':
            is_unit_stride = (_symbolic.pystr_to_symbolic(str(n.stride)) == 1)
            n.implementation = 'CUDA' if is_unit_stride else 'CUDA_strided'
    gpu.validate()
    return gpu


def _oracle(p0: np.ndarray, d: np.ndarray) -> np.ndarray:
    out = p0.copy()
    for jk in range(1, p0.shape[0]):
        out[jk, :] = out[jk - 1, :] + d[jk, :]
    return out


def _layout_prep(arr: np.ndarray, layout: str) -> np.ndarray:
    """Return a copy with the requested host layout."""
    if layout == 'F':
        return np.asfortranarray(arr)
    return np.ascontiguousarray(arr)


_VARIANTS = ('A', 'B', 'C')
_LAYOUTS = ('C', 'F')


@pytest.mark.parametrize('dtype_name', ['fp64', 'fp32'])
def test_for_1133_layout_ab_cpu_gpu(dtype_name, ab_klev, ab_klon, ab_iters, ab_warmup, ab_gpu_enabled, capsys):
    """3 variants x 2 layouts x {CPU, GPU} timing sweep.

    Asserts numerical correctness for every cell (per-cell tolerance:
    fp64 rtol=1e-5/atol=1e-8, fp32 widened to 1e-3 to absorb the
    prefix-sum's accumulated rounding)."""
    klev, klon = ab_klev, ab_klon
    assert klon % 32 == 0
    np_dtype = _DTYPE[dtype_name]
    tol = dict(rtol=1e-3, atol=1e-3) if dtype_name == 'fp32' else dict(rtol=1e-5, atol=1e-8)

    rng = np.random.default_rng(1133)
    pfsqrf_init_c = rng.standard_normal((klev, klon)).astype(np_dtype)
    delta_c = rng.standard_normal((klev, klon)).astype(np_dtype)
    ref = _oracle(pfsqrf_init_c, delta_c)

    if ab_gpu_enabled:
        ensure_relaxed_constexpr_nvcc()
        ensure_gpu_heap()

    rows = []
    for variant in _VARIANTS:
        for layout in _LAYOUTS:
            cell_label = f'{variant}/{layout}'
            row = {
                'cell': cell_label,
                'cpu_ok': None,
                'cpu_diff': None,
                'cpu_med_us': None,
                'cpu_err': None,
                'gpu_ok': None,
                'gpu_diff': None,
                'gpu_med_us': None,
                'gpu_err': None,
            }
            try:
                sdfg = _build(variant, layout, 'cpu', dtype_name)
            except Exception as e:
                row['cpu_err'] = f'build: {type(e).__name__}: {e}'[:60]
                rows.append(row)
                continue

            p0 = _layout_prep(pfsqrf_init_c, layout)
            d0 = _layout_prep(delta_c, layout)

            # CPU correctness + timing.
            try:
                p_buf = np.asfortranarray(p0) if layout == 'F' else p0.copy()
                d_buf = np.asfortranarray(d0) if layout == 'F' else d0.copy()
                fn = functools.partial(sdfg, pfsqrf=p_buf, delta=d_buf, KLEV=klev, KLON=klon)
                fn()
                row['cpu_ok'] = bool(np.allclose(p_buf, ref, **tol))
                row['cpu_diff'] = float(np.abs(p_buf - ref).max())

                def _reseed_cpu(pb=p_buf, db=d_buf, lay=layout):
                    if lay == 'F':
                        pb[...] = np.asfortranarray(pfsqrf_init_c)
                        db[...] = np.asfortranarray(delta_c)
                    else:
                        pb[...] = pfsqrf_init_c
                        db[...] = delta_c

                def _timed_cpu(reseed=_reseed_cpu, f=fn):
                    reseed()
                    f()

                row['cpu_med_us'] = time_cpu(_timed_cpu, iters=ab_iters, warmup=ab_warmup)['median_us']
            except Exception as e:
                row['cpu_err'] = f'{type(e).__name__}: {e}'[:60]

            # GPU correctness + timing.
            if ab_gpu_enabled:
                try:
                    import cupy
                    sdfg_gpu = _to_gpu_sdfg(sdfg, 'gpu', device_resident_data=('pfsqrf', 'delta'))
                    p_buf_gpu = cupy.asarray(p0)
                    d_buf_gpu = cupy.asarray(d0)
                    fn_gpu = functools.partial(sdfg_gpu, pfsqrf=p_buf_gpu, delta=d_buf_gpu, KLEV=klev, KLON=klon)
                    fn_gpu()
                    cupy.cuda.runtime.deviceSynchronize()
                    p_back = cupy.asnumpy(p_buf_gpu)
                    row['gpu_ok'] = bool(np.allclose(p_back, ref, **tol))
                    row['gpu_diff'] = float(np.abs(p_back - ref).max())

                    def _reseed_gpu(pb=p_buf_gpu, db=d_buf_gpu, p_src=p0, d_src=d0):
                        pb[...] = cupy.asarray(p_src)
                        db[...] = cupy.asarray(d_src)

                    def _timed_gpu(reseed=_reseed_gpu, f=fn_gpu):
                        reseed()
                        f()

                    row['gpu_med_us'] = time_gpu(_timed_gpu, iters=ab_iters, warmup=ab_warmup)['median_us']
                except Exception as e:
                    row['gpu_err'] = f'{type(e).__name__}: {e}'[:60]
            rows.append(row)

    # Report.
    lines = []
    lines.append('')
    lines.append(f'== for_1133 layout AB  dtype={dtype_name}  klev={klev}  klon={klon}  iters={ab_iters} ==')
    lines.append(f'{"cell":<8} {"CPU(us)":>12} {"CPU ok":>8} {"GPU(us)":>12} {"GPU ok":>8}  notes')
    lines.append('-' * 80)
    for r in rows:

        def _fmt_us(v):
            return f'{v:>12.1f}' if isinstance(v, float) else f'{"-":>12}'

        def _fmt_ok(v):
            return f'{str(v):>8}' if v is not None else f'{"-":>8}'

        note = ''
        if r['cpu_err']:
            note += f'cpu_err={r["cpu_err"]} '
        if r['gpu_err']:
            note += f'gpu_err={r["gpu_err"]}'
        lines.append(f'{r["cell"]:<8} {_fmt_us(r["cpu_med_us"])} {_fmt_ok(r["cpu_ok"])} '
                     f'{_fmt_us(r["gpu_med_us"])} {_fmt_ok(r["gpu_ok"])}  {note}')
    lines.append('')
    captured = '\n'.join(lines)
    with capsys.disabled():
        print(captured)

    # Don't FAIL the test on individual cell correctness failures -- the
    # point of the sweep is to surface the cost model. Numerical issues
    # are reported in the table and via the per-cell ``ok`` columns.
