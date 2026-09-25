# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``canonicalize(specialize_constants=...)`` naming a scalar argument, not a symbol.

CloudSC selects its supersaturation formula with ``yrecldp_nssopt``, a run-time ``int32`` argument.
Its if/elif chain has no ``else``, so for an unlisted value the temporary keeps the previous column's
value and the column loop cannot become a map. Baking the flag in collapses the chain.
"""
import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')


@dace.program
def flag_selected_temporary(a: dace.float64[N], b: dace.float64[N], flag: dace.int32):
    t = 0.0
    for i in range(N):
        if flag == 0:
            t = a[i] * 2.0
        elif flag == 1:
            t = a[i] * 3.0
        b[i] = t + 1.0


def canonicalized(constants: dict) -> dace.SDFG:
    sdfg = flag_selected_temporary.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, validate_all=False, target='cpu', specialize_constants=constants)
    sdfg.validate()
    return sdfg


def loops_left(sdfg: dace.SDFG) -> list:
    return sorted(r.label for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion))


def test_a_specialized_flag_argument_lets_its_guarded_loop_become_a_map():
    sdfg = canonicalized({'flag': 1})
    assert loops_left(sdfg) == [], loops_left(sdfg)
    a = np.random.default_rng(0).random(17)
    b = np.zeros(17)
    sdfg(a=a, b=b, flag=1, N=17)
    np.testing.assert_allclose(b, a * 3.0 + 1.0, rtol=1e-15, atol=0)


def test_a_specialized_flag_argument_stays_in_the_signature():
    """Callers pass every argument of the unspecialized program, so the ABI must not change."""
    assert 'flag' in canonicalized({'flag': 1}).arglist()
