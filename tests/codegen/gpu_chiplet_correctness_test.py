# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the results of a kernel whose thread-blocks are distributed over the chiplets of a GPU. """

import numpy as np
import pytest

import dace
from dace.transformation.auto.auto_optimize import auto_optimize

KLEV, KLON = 137, 1024

# Constants of the CLOUDSC routine the kernel below is taken from
_RLMIN = 1e-8
_RAMIN = 1e-6
_ZQTMST = 1.0 / 300.0
_RALVDCP = 2489.0
_RALSDCP = 2821.0

CHIPLETS = 6

# The arrays the kernel modifies in place, and therefore the ones that are compared
ARRAYS = ('zqx_l', 'zqx_i', 'zqx_v', 'za', 'ptend_q', 'ptend_t')


@dace.program
def cloudsc_tidy_branch(zqx_l: dace.float64[KLEV, KLON], zqx_i: dace.float64[KLEV, KLON],
                        zqx_v: dace.float64[KLEV, KLON], za: dace.float64[KLEV, KLON],
                        ptend_q: dace.float64[KLEV, KLON], ptend_t: dace.float64[KLEV, KLON]):
    # cloudsc_bottom_lower.F90: "Tidy up very small cloud cover or total
    # cloud water" — guarded read-modify-write over several arrays (the
    # CLOUDSC-characteristic conditional accumulation pattern).
    for jk in range(KLEV):
        for jl in range(KLON):
            if zqx_l[jk, jl] + zqx_i[jk, jl] < _RLMIN or za[jk, jl] < _RAMIN:
                zqadj_l = zqx_l[jk, jl] * _ZQTMST
                ptend_q[jk, jl] = ptend_q[jk, jl] + zqadj_l
                ptend_t[jk, jl] = ptend_t[jk, jl] - _RALVDCP * zqadj_l
                zqx_v[jk, jl] = zqx_v[jk, jl] + zqx_l[jk, jl]
                zqx_l[jk, jl] = 0.0
                zqadj_i = zqx_i[jk, jl] * _ZQTMST
                ptend_q[jk, jl] = ptend_q[jk, jl] + zqadj_i
                ptend_t[jk, jl] = ptend_t[jk, jl] - _RALSDCP * zqadj_i
                zqx_v[jk, jl] = zqx_v[jk, jl] + zqx_i[jk, jl]
                zqx_i[jk, jl] = 0.0
                za[jk, jl] = 0.0


def _random_inputs():
    """
    Returns random inputs for ``cloudsc_tidy_branch``.

    ``zqx_l + zqx_i`` and ``za`` are drawn around the thresholds of the guard, so that both of its
    branches are taken for a substantial part of the points. Drawing every array from the unit
    interval instead would leave the guard false everywhere, and the kernel a no-op.
    """
    rng = np.random.default_rng(42)
    return {
        'zqx_l': rng.random((KLEV, KLON)) * 2 * _RLMIN,
        'zqx_i': rng.random((KLEV, KLON)) * 2 * _RLMIN,
        'zqx_v': rng.random((KLEV, KLON)),
        'za': rng.random((KLEV, KLON)) * 2 * _RAMIN,
        'ptend_q': rng.random((KLEV, KLON)),
        'ptend_t': rng.random((KLEV, KLON)),
    }


def _reference(inputs):
    """
    Computes the expected results with NumPy, following the order in which the kernel accumulates
    into each array, and returns them together with the mask of the points entering the branch.
    """
    zqx_l, zqx_i, zqx_v, za, ptend_q, ptend_t = (inputs[name] for name in ARRAYS)

    taken = (zqx_l + zqx_i < _RLMIN) | (za < _RAMIN)
    zqadj_l = zqx_l * _ZQTMST
    zqadj_i = zqx_i * _ZQTMST

    expected = {
        'zqx_l': np.where(taken, 0.0, zqx_l),
        'zqx_i': np.where(taken, 0.0, zqx_i),
        'zqx_v': np.where(taken, (zqx_v + zqx_l) + zqx_i, zqx_v),
        'za': np.where(taken, 0.0, za),
        'ptend_q': np.where(taken, (ptend_q + zqadj_l) + zqadj_i, ptend_q),
        'ptend_t': np.where(taken, (ptend_t - _RALVDCP * zqadj_l) - _RALSDCP * zqadj_i, ptend_t),
    }
    return expected, taken


def _setup():
    """ Returns the inputs, the expected results, and a fresh copy of the inputs to compute into. """
    inputs = _random_inputs()
    expected, taken = _reference(inputs)

    # Neither branch of the guard may be dead, otherwise the kernel is trivially correct
    assert 0.2 < taken.mean() < 0.8, f'the guard is taken for {taken.mean():.1%} of the points'

    return {name: inputs[name].copy() for name in ARRAYS}, expected


def _assert_results(actual, expected, context):
    for name in ARRAYS:
        assert np.allclose(actual[name], expected[name]), f'{name} differs {context}'


def test_cloudsc_tidy_branch_cpu():
    args, expected = _setup()
    cloudsc_tidy_branch(**args)
    _assert_results(args, expected, 'on the CPU')


@pytest.mark.gpu
def test_cloudsc_tidy_branch_gpu_chiplet_distribution():
    # The distribution is only of use on a multi-chiplet AMD GPU, but it is a pure remapping of
    # thread-blocks to work items, so the results have to be identical on any GPU. It is therefore
    # also run on a non-AMD GPU here, where it is pointless for performance, so that the arithmetic
    # of the reshaped grid is exercised by a GPU CI that does not run on an AMD GPU. Running the
    # kernel without the distribution as well tells a failure of the distribution apart from a
    # failure of the kernel itself.
    for chiplets in (1, CHIPLETS):
        args, expected = _setup()

        with dace.config.temporary_config():
            dace.config.Config.set('compiler', 'cuda', 'chiplet_number', value=chiplets)

            sdfg = cloudsc_tidy_branch.to_sdfg(simplify=True)
            # Give every configuration its own build folder, so that they cannot share a binary
            sdfg.name = f'cloudsc_tidy_branch_{chiplets}_chiplets'
            auto_optimize(sdfg, dace.DeviceType.GPU)

            # The kernel has to remain one that the distribution applies to, otherwise the results
            # below are those of a kernel that was never distributed
            if chiplets > 1:
                assert f'dim3({chiplets}, ' in sdfg.generate_code()[1].code

            sdfg(**args)

        _assert_results(args, expected, f'on the GPU over {chiplets} chiplet(s)')


if __name__ == '__main__':
    test_cloudsc_tidy_branch_cpu()
    test_cloudsc_tidy_branch_gpu_chiplet_distribution()
