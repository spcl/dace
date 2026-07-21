# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Numeric tests for :mod:`tests.corpus.cloudsc.offload_cloudsc_to_gpu` -- the offloaded graph is RUN
ON THE GPU and compared against the same graph run on the host.

The sibling ``cloudsc_offload_to_gpu_test`` pins the offload's *structure* without a GPU; this pins
that the structure computes the right thing on the device. Both fixtures keep CLOUDSC's shape -- an
outer per-block map with the real work nested inside it, so the block loop stays on the host and each
inner map becomes a kernel -- and both are called with ORDINARY HOST NUMPY ARRAYS: the offload mirrors
every kernel-side array to a ``gpu_<name>`` transient with copy-in/copy-out states, so the graph does
its own transfers and no ``gpu_*`` name ever reaches the argument list
(``test_offloaded_graph_is_called_with_host_arrays`` pins exactly that).

Two levels of strictness, on purpose:

* :func:`test_pure_arithmetic_offload_is_bit_exact_on_device` uses a graph of nothing but multiplies,
  where host and device MUST agree bit-for-bit. No tolerance at all -- a mis-indexed mirror, a
  dropped copy-out or a kernel writing the wrong element cannot hide under a fuzzy compare.
* :func:`test_microphysics_offload_matches_the_host_to_last_bits` adds the transcendentals and the
  vertical recurrence that CLOUDSC actually runs, where host and device libm differ in the last bits
  by construction, and bounds the disagreement in ULPs (see :data:`LIBM_ULP_BOUND`).

    pytest tests/corpus/cloudsc/cloudsc_offload_numeric_test.py -v -m gpu
"""
import copy
import math

import numpy as np
import pytest

import dace
from dace.config import set_temporary
from dace.sdfg import nodes

from tests.corpus.cloudsc.cloudsc_offload_to_gpu_test import blocked_sdfg
from tests.corpus.cloudsc.offload_cloudsc_to_gpu import offload_cloudsc_to_gpu
from tests.corpus.cloudsc.pipelines import STRICT_FP_CUDA_ARGS, check_offload_phase, gpu_is_runnable

pytestmark = pytest.mark.gpu

#: Strict-FP host flags, the same regime ``IEEE_CPU_ARGS`` uses: the point of these tests is a
#: host/device comparison, so neither side may be left on a fast-math build that reassociates.
STRICT_FP_CPU_ARGS: str = '-std=c++14 -fPIC -O0 -fopenmp -fno-fast-math -ffp-contract=off'

#: How far the transcendental fixture's device result may sit from the host result, in ULPs of the
#: host value. CUDA's ``exp``/``log`` are not glibc's, so a few last bits differ no matter how strictly
#: both sides are built -- this bounds that, it does not tolerate reassociation. Measured on an RTX
#: 4050 / CUDA 13.1 / glibc build: 6 ULP worst over both outputs; the bound is set an order of
#: magnitude above so a toolkit or driver bump does not turn into a spurious failure, and is still
#: ~1e-13 relative -- far tighter than any real logic error would land.
LIBM_ULP_BOUND: int = 64

nblocks = dace.symbol('nblocks')
klev = dace.symbol('klev')
klon = dace.symbol('klon')

SIZES = {'klon': 8, 'klev': 16, 'nblocks': 4}


@dace.program
def microphysics(pt: dace.float64[klon, klev, nblocks], pq: dace.float64[klon, klev, nblocks],
                 pflux: dace.float64[klon, klev, nblocks], tend: dace.float64[klon, klev, nblocks]):
    """CLOUDSC's compute shape in miniature: per-block outer map, horizontal map inside it, and a
    SEQUENTIAL vertical sweep carrying a running flux -- plus the saturation-pressure exponential the
    real kernel spends most of its time in. Enough arithmetic that a wrong schedule, a lost carry or a
    mirrored-but-not-copied-back array changes the answer visibly."""
    for ibl in dace.map[0:nblocks]:
        for jl in dace.map[0:klon]:
            carry = 0.0
            for jk in range(klev):
                sat = 611.21 * math.exp(17.502 * (pt[jl, jk, ibl] - 273.16) / (pt[jl, jk, ibl] - 32.19))
                excess = pq[jl, jk, ibl] - sat * 1e-5
                carry = carry + excess * 0.5
                pflux[jl, jk, ibl] = carry
                tend[jl, jk, ibl] = excess / (1.0 + math.log(1.0 + sat))


def microphysics_buffers(seed: int = 7):
    """Seeded inputs in the physical window (``pt`` well clear of the ``pt - 32.19`` pole) and zeroed
    outputs. A fresh set per leg, so neither run can see the other's buffers."""
    rng = np.random.default_rng(seed)
    shape = (SIZES['klon'], SIZES['klev'], SIZES['nblocks'])
    return {
        'pt': 250.0 + 50.0 * rng.random(shape),
        'pq': 1e-3 * rng.random(shape),
        'pflux': np.zeros(shape),
        'tend': np.zeros(shape),
    }


def blocked_buffers(seed: int = 3):
    rng = np.random.default_rng(seed)
    shape = (SIZES['klev'], SIZES['klon'], SIZES['nblocks'])  # blocked_sdfg is [klev, klon, nblocks]
    return {'pin': rng.random(shape), 'pout': np.zeros(shape)}


def run_on_host(sdfg: dace.SDFG, buffers: dict, name: str) -> dict:
    sdfg = copy.deepcopy(sdfg)
    sdfg.name = name
    with set_temporary('compiler', 'cpu', 'args', value=STRICT_FP_CPU_ARGS):
        sdfg(**buffers, **SIZES)
    return buffers


def run_on_device(sdfg: dace.SDFG, buffers: dict, name: str) -> dict:
    """Offload ``sdfg`` and run it on the GPU, under the same strict FP rules as the host leg."""
    sdfg = copy.deepcopy(sdfg)
    sdfg.name = name
    offload_cloudsc_to_gpu(sdfg)
    assert any(node.map.schedule == dace.ScheduleType.GPU_Device
               for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry)), \
        'nothing was scheduled onto the device -- the comparison would be host-vs-host'
    cuda_args = f'{STRICT_FP_CUDA_ARGS} {dace.Config.get("compiler", "cuda", "args")}'
    with set_temporary('compiler', 'cuda', 'implementation', value='experimental'):
        with set_temporary('compiler', 'cuda', 'args', value=cuda_args):
            sdfg(**buffers, **SIZES)
    return buffers


def ulp_distance(host: np.ndarray, device: np.ndarray) -> np.ndarray:
    """Host/device disagreement per element, in ULPs of the host value."""
    delta = np.abs(host - device)
    spacing = np.spacing(np.abs(host))
    return np.where(spacing > 0, delta / spacing, delta)


def test_gpu_is_runnable_agrees_with_the_gpu_mark():
    """These tests are selected by ``-m gpu``, so the box running them has a GPU. If the pipeline's own
    detector disagreed, the offload phase would quietly take the structural fallback and every claim of
    a device-verified run in this file -- and in the pipeline log -- would be vacuous."""
    assert gpu_is_runnable() is True


def test_offloaded_graph_is_called_with_host_arrays():
    """The mirroring assumption, verified rather than trusted: after offload the argument list still
    names only the ORIGINAL host arrays, and every ``gpu_*`` mirror is an internal transient. That is
    what lets both legs be driven from the identical plain-numpy dict."""
    sdfg = copy.deepcopy(microphysics.to_sdfg(simplify=True))
    offload_cloudsc_to_gpu(sdfg)
    mirrors = sorted(name for name in sdfg.arrays if name.startswith('gpu_'))
    assert mirrors, 'nothing was mirrored to the device'
    arglist = sdfg.arglist()
    assert not [name for name in mirrors if name in arglist], f'mirrors leaked into the arglist: {mirrors}'
    assert all(sdfg.arrays[name].transient for name in mirrors)
    for name in ('pt', 'pq', 'pflux', 'tend'):
        assert name in arglist, f'{name} vanished from the arglist'


def test_pure_arithmetic_offload_is_bit_exact_on_device():
    """No transcendentals, so there is nothing for host and device libm to disagree about: the
    offloaded graph must reproduce the host result BIT FOR BIT. Asserted with ``array_equal`` -- no
    tolerance to hide behind."""
    base = blocked_sdfg()
    host = run_on_host(base, blocked_buffers(), 'offload_numeric_blocked_host')
    device = run_on_device(base, blocked_buffers(), 'offload_numeric_blocked_device')
    assert np.array_equal(host['pout'], device['pout']), (
        f'device result is not bit-exact: {int((host["pout"] != device["pout"]).sum())} of '
        f'{host["pout"].size} elements differ, max |delta|={np.abs(host["pout"] - device["pout"]).max():.3e}')
    assert np.array_equal(host['pin'], device['pin']), 'the copy-in/copy-out round trip corrupted an input'


def test_microphysics_offload_matches_the_host_to_last_bits():
    """The CLOUDSC-shaped fixture: same inputs, host leg and device leg, both strict-FP. Bounded in
    ULPs so what is being allowed is exactly "the last bits of exp/log differ" and nothing looser."""
    base = microphysics.to_sdfg(simplify=True)
    host = run_on_host(base, microphysics_buffers(), 'offload_numeric_micro_host')
    device = run_on_device(base, microphysics_buffers(), 'offload_numeric_micro_device')
    for name in ('pflux', 'tend'):
        ulps = ulp_distance(host[name], device[name])
        worst = float(ulps.max())
        assert worst <= LIBM_ULP_BOUND, (f'{name}: device result is {worst:.1f} ULP from the host result '
                                         f'(bound {LIBM_ULP_BOUND}); max |delta|='
                                         f'{np.abs(host[name] - device[name]).max():.3e}. A gap this wide is '
                                         f'not a libm last-bit difference -- suspect the offload.')
    assert np.array_equal(host['pt'], device['pt']), 'the copy-in/copy-out round trip corrupted an input'


def test_check_offload_phase_runs_the_numeric_check_on_device():
    """The pipeline gate itself, end to end: on a GPU box :func:`check_offload_phase` must report a
    NUMERIC check, actually call the wired ``numeric_check`` on the offloaded graph, and let that
    check's failure through rather than swallowing it."""
    sdfg = blocked_sdfg()
    offload_cloudsc_to_gpu(sdfg)
    sdfg.name = 'offload_numeric_gate'
    seen = []
    assert check_offload_phase(sdfg, lambda graph, phase: seen.append((graph.name, phase))) is True
    assert seen == [('offload_numeric_gate', 'offload')], seen

    def failing_check(_graph, _phase):
        raise AssertionError('wired check failed')

    with pytest.raises(AssertionError, match='wired check failed'):
        check_offload_phase(sdfg, failing_check)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
