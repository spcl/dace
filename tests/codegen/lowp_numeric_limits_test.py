# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression test for std::numeric_limits<dace::half>/<dace::bfloat16>. Unspecialized, the primary
template makes max()/lowest() both zero, so a min/max reduction seeded from them gets stuck at 0.
"""
import ml_dtypes
import numpy as np

import dace

# ctype names, matching dace.dtypes.TYPECLASS_TO_STRING for float16/bfloat16.
_CASES = [(dace.float16, 'dace::half', np.float16), (dace.bfloat16, 'dace::bfloat16', ml_dtypes.bfloat16)]


def _build_sdfg(dace_dtype: dace.typeclass, ctype: str) -> dace.SDFG:
    """SDFG with a CPP tasklet seeding a wcr_fixed min/max reduction from numeric_limits, mirroring
    dace/libraries/tileops/nodes/tile_reduce.py on the `extended` branch (not present on `main`)."""
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
    test_numeric_limits_seeds_correct_lowp_reduction()
