# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``std::numeric_limits`` must be specialized for ``dace::half`` and ``dace::bfloat16``. Unspecialized
class types get the primary template, whose ``max()``/``lowest()``/``infinity()`` are all ``T()`` --
zero -- so identity-seeded min/max reductions silently produce zeros instead of failing to build."""
import os
import shutil
import subprocess

import ml_dtypes
import numpy as np
import pytest

import dace
from dace.config import Config

INCLUDE = os.path.join(os.path.dirname(os.path.abspath(dace.__file__)), 'runtime', 'include')

#: Every check would pass trivially against the primary template if it compared to zero, so each one
#: is a value the primary template cannot produce.
PROBE_SOURCE = r'''
#include "dace/types.h"
template <typename T>
static bool ok() {
    using L = std::numeric_limits<T>;
    return L::is_specialized && (float)L::max() > 0.0f && (float)L::lowest() < 0.0f &&
           (float)L::infinity() > (float)L::max() && (float)L::denorm_min() > 0.0f;
}
int main() {
    if (!ok<dace::half>() || !ok<dace::bfloat16>()) return 1;
    // The exact finite bounds of IEEE binary16 and bfloat16.
    if ((float)std::numeric_limits<dace::half>::max() != 65504.0f) return 2;
    if ((float)std::numeric_limits<dace::bfloat16>::lowest() != -3.38953139e+38f) return 3;
    return 0;
}
'''


def test_numeric_limits_is_specialized(tmp_path):
    executable = Config.get('compiler', 'cpu', 'executable') or 'c++'
    assert shutil.which(executable), f'configured compiler {executable!r} is not on PATH'
    source, binary = tmp_path / 'probe.cpp', tmp_path / 'probe'
    source.write_text(PROBE_SOURCE)
    build = subprocess.run([
        executable, f'-std=c++{Config.get("compiler", "cpp_standard")}', '-I', INCLUDE,
        str(source), '-o',
        str(binary)
    ],
                           capture_output=True,
                           text=True,
                           timeout=300)
    assert build.returncode == 0, f'probe did not compile:\n{build.stderr}'
    assert subprocess.run([str(binary)], timeout=60).returncode == 0, \
        'std::numeric_limits is unspecialized for a low-precision type, so its identities are zero'


# A min-reduce seeded from a wrongly-zero identity over all-positive data gets stuck at 0 (0 is less
# than every candidate, so the compare-and-swap never fires); the mirror bug for max-reduce needs
# all-negative data. Reproduced by seeding the exact ``dace::wcr_fixed`` primitive DaCe-generated
# reduce code calls (see the ``dace::wcr_fixed<...>::reduce_atomic`` calls in the C++ generated for
# ``numpy.min``/``numpy.max`` over a low-precision array) with the identity expressions
# dace/libraries/tileops/nodes/tile_reduce.py:34-37 on the `extended` branch uses for min/max
# reduction accumulators: ``std::numeric_limits<T>::max()`` and ``::lowest()``. tileops does not exist
# on `main`, so it is not imported -- just mirrored, with no optional dependency, so it runs in
# general-ci.
REGRESSION_SOURCE = r'''
#include "dace/reduction.h"
#include <cstdio>

dace::half min_id_half = std::numeric_limits<dace::half>::max();
dace::half max_id_half = std::numeric_limits<dace::half>::lowest();
dace::bfloat16 min_id_bf16 = std::numeric_limits<dace::bfloat16>::max();
dace::bfloat16 max_id_bf16 = std::numeric_limits<dace::bfloat16>::lowest();

template <typename T, dace::ReductionType RT>
static T reduce_seeded(T seed, const T *data, int n) {
    T acc = seed;
    for (int i = 0; i < n; ++i) dace::wcr_fixed<RT, T>::reduce_atomic(&acc, data[i]);
    return acc;
}

// Exactly representable in both binary16 (10 mantissa bits) and bfloat16 (7 mantissa bits): each is a
// sum of one or two powers of two.
template <typename T>
static bool run(T min_id, T max_id) {
    const float positive[5] = {3.5f, 1.25f, 9.0f, 2.0f, 4.5f};
    const float negative[5] = {-3.5f, -1.25f, -9.0f, -2.0f, -4.5f};
    T pos[5], neg[5];
    for (int i = 0; i < 5; ++i) { pos[i] = T(positive[i]); neg[i] = T(negative[i]); }
    T mn = reduce_seeded<T, dace::ReductionType::Min>(min_id, pos, 5);
    T mx = reduce_seeded<T, dace::ReductionType::Max>(max_id, neg, 5);
    // Structural, not just numeric: 0 is exactly the value the unfixed header silently produces.
    if ((float)mn == 0.0f || (float)mx == 0.0f) return false;
    return (float)mn == 1.25f && (float)mx == -1.25f;
}

int main() {
    if (!run<dace::half>(min_id_half, max_id_half)) return 1;
    if (!run<dace::bfloat16>(min_id_bf16, max_id_bf16)) return 2;
    return 0;
}
'''


def test_numeric_limits_max_lowest_seed_correct_reduction(tmp_path):
    """Both ``max()`` and ``lowest()`` are 0 on the unspecialized primary template, so a min-reduce
    seeded from ``max()`` over all-positive data, and a max-reduce seeded from ``lowest()`` over
    all-negative data, both wrongly get stuck at 0 instead of reaching the true 1.25 / -1.25.
    """
    executable = Config.get('compiler', 'cpu', 'executable') or 'c++'
    assert shutil.which(executable), f'configured compiler {executable!r} is not on PATH'
    source, binary = tmp_path / 'regression_probe.cpp', tmp_path / 'regression_probe'
    source.write_text(REGRESSION_SOURCE)
    build = subprocess.run([
        executable, f'-std=c++{Config.get("compiler", "cpp_standard")}', '-I', INCLUDE,
        str(source), '-o',
        str(binary)
    ],
                           capture_output=True,
                           text=True,
                           timeout=300)
    assert build.returncode == 0, f'regression probe did not compile:\n{build.stderr}'
    run = subprocess.run([str(binary)], timeout=60)
    assert run.returncode == 0, \
        f'min/max reduction identity is wrong (exit code {run.returncode}: 1 = half, 2 = bfloat16)'


# NOT a regression test for this fix -- an end-to-end sanity check that the low-precision reduction
# path this header exists for actually computes the right numbers. Checked by grepping
# sdfg.generate_code() for ``numeric_limits``: it is absent for numpy.min/numpy.max, dace.reduce with
# an explicit lambda, WCR-min/max memlets on a plain map, and the Reduce library node's CPU expansions
# (pure, OpenMP) -- every reduction identity DaCe's frontend and CPU codegen produce on `main` is a
# Python/numpy literal, never a call into this header. It passes with the specialization block removed
# too (checked). The header path itself is exercised by
# test_numeric_limits_max_lowest_seed_correct_reduction above; dace/libraries/tileops/nodes/tile_reduce.py
# on the `extended` branch is the SDFG-level consumer that does reach it, but tileops does not exist here.
def test_e2e_low_precision_reduction_sanity():
    """fp32 in, an fp16/bfloat16 transient in between, fp32 out: a map casts down, dace.reduce computes
    min and max over the transient, a tasklet casts back up. Values are exactly representable in both
    binary16 and bfloat16 (round-trip checked below, not assumed) so the comparison is exact.
    """
    positive = np.array([3.5, 1.25, 9.0, 2.0, 4.5], dtype=np.float32)
    negative = -positive

    for lp_dtype, np_dtype in [(dace.float16, np.float16), (dace.bfloat16, ml_dtypes.bfloat16)]:
        for values in (positive, negative):
            assert np.all(np_dtype(values).astype(np.float32) == values), \
                f'{lp_dtype}: test data is not exactly representable, pick different literals'

        @dace.program
        def lowp_reduce(A: dace.float32[5], out_min: dace.float32[1], out_max: dace.float32[1]):
            lp = dace.define_local([5], lp_dtype)
            for i in dace.map[0:5]:
                with dace.tasklet:
                    a << A[i]
                    l >> lp[i]
                    l = a
            mn = dace.reduce(lambda a, b: min(a, b), lp, identity=1e30)
            mx = dace.reduce(lambda a, b: max(a, b), lp, identity=-1e30)
            with dace.tasklet:
                m << mn
                o >> out_min[0]
                o = m
            with dace.tasklet:
                m << mx
                o >> out_max[0]
                o = m

        sdfg = lowp_reduce.to_sdfg(simplify=True)

        out_min, out_max = np.zeros([1], dtype=np.float32), np.zeros([1], dtype=np.float32)
        sdfg(A=positive.copy(), out_min=out_min, out_max=out_max)
        assert out_min[0] == 1.25, f'{lp_dtype}: min over positive data = {out_min[0]}, expected 1.25'

        out_min2, out_max2 = np.zeros([1], dtype=np.float32), np.zeros([1], dtype=np.float32)
        sdfg(A=negative.copy(), out_min=out_min2, out_max=out_max2)
        assert out_max2[0] == -1.25, f'{lp_dtype}: max over negative data = {out_max2[0]}, expected -1.25'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
