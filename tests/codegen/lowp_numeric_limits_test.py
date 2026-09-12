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


# ctype names, matching dace.dtypes.TYPECLASS_TO_STRING for float16/bfloat16.
_CASES = [(dace.float16, 'dace::half', np.float16), (dace.bfloat16, 'dace::bfloat16', ml_dtypes.bfloat16)]


def _build_sdfg(dace_dtype: dace.typeclass, ctype: str) -> dace.SDFG:
    """SDFG with a CPP tasklet seeding a wcr_fixed min/max reduction from numeric_limits, mirroring
    what ``dace/libraries/tileops/nodes/tile_reduce.py`` emits."""
    sdfg = dace.SDFG(f'lowp_minmax_{dace_dtype.type.__name__}')
    for name in ('data', 'out_min', 'out_max'):
        sdfg.add_array(name, [5 if name == 'data' else 1], dace_dtype)
    state = sdfg.add_state()
    code = f'''
{ctype} amn = std::numeric_limits<{ctype}>::max();
{ctype} amx = std::numeric_limits<{ctype}>::lowest();
for (int i = 0; i < 5; i++) {{
    dace::wcr_fixed<dace::ReductionType::Min, {ctype}>::reduce_atomic(&amn, d[i]);
    dace::wcr_fixed<dace::ReductionType::Max, {ctype}>::reduce_atomic(&amx, d[i]);
}}
mn = amn;
mx = amx;
'''
    tasklet = state.add_tasklet('lowp_minmax', {'d': None}, {'mn': None, 'mx': None}, code, language=dace.Language.CPP)
    state.add_edge(state.add_read('data'), None, tasklet, 'd', dace.Memlet.from_array('data', sdfg.arrays['data']))
    state.add_edge(tasklet, 'mn', state.add_write('out_min'), None,
                   dace.Memlet.from_array('out_min', sdfg.arrays['out_min']))
    state.add_edge(tasklet, 'mx', state.add_write('out_max'), None,
                   dace.Memlet.from_array('out_max', sdfg.arrays['out_max']))
    sdfg.validate()
    return sdfg


def test_numeric_limits_seeds_correct_lowp_reduction():
    """A min-reduce seeded from max() over all-positive data, and a max-reduce seeded from lowest()
    over all-negative data, both get stuck at 0 if numeric_limits is unspecialized (0 is never beaten:
    it's below every positive candidate, above every negative one)."""
    positive = np.array([3.5, 1.25, 9.0, 2.0, 4.5], dtype=np.float32)
    negative = -positive

    for dace_dtype, ctype, np_dtype in _CASES:
        sdfg = _build_sdfg(dace_dtype, ctype)
        code = '\n'.join(c.clean_code for c in sdfg.generate_code())
        assert 'numeric_limits' in code, f'{ctype}: generated code does not reach the header under test'
        csdfg = sdfg.compile()

        out_min, out_max = np.zeros([1], dtype=np_dtype), np.zeros([1], dtype=np_dtype)
        csdfg(data=positive.astype(np_dtype), out_min=out_min, out_max=out_max)
        assert out_min[0] != 0, f'{ctype}: min-reduce over positive data stuck at the unspecialized identity'
        assert out_min[0] == 1.25, f'{ctype}: min over positive data = {out_min[0]}, expected 1.25'

        out_min2, out_max2 = np.zeros([1], dtype=np_dtype), np.zeros([1], dtype=np_dtype)
        csdfg(data=negative.astype(np_dtype), out_min=out_min2, out_max=out_max2)
        assert out_max2[0] != 0, f'{ctype}: max-reduce over negative data stuck at the unspecialized identity'
        assert out_max2[0] == -1.25, f'{ctype}: max over negative data = {out_max2[0]}, expected -1.25'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
