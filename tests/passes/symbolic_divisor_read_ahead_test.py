# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A read-ahead over a disjoint region must stay a Map when the offset's divisor is SYMBOLIC.

``a[i] = a[i + LEN_1D // M] + b[i]`` over ``range(LEN_1D // M)`` writes ``[0, D)`` and reads
``[D, 2D)`` for ``D = LEN_1D // M`` -- disjoint for every value of ``M``, so it is a parallel Map
and not a recurrence. The constant-divisor spelling (``// 2``, TSVC ``s1421``) was already handled;
the symbolic one was not, because the canonicalize contract builds free symbols ``nonnegative`` and
that admits ``M == 0``, leaving SymPy unable to decide the sign of ``-floor(LEN_1D/M)``.
``_admissible_scan_stride`` read UNKNOWN as admissible and lifted a scan, which cost the Map and
measured 3.1x slower on CPU and 84x slower on GPU than the same kernel with a constant divisor.
"""
import dace
import pytest

from dace.sdfg import nodes as dnodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol('LEN_1D', dtype=dace.int64)
M = dace.symbol('M', dtype=dace.int64)


def _maps(sdfg):
    return [n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, dnodes.MapEntry)]


def _loops(sdfg):
    return [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]


@dace.program
def read_ahead_symbolic_divisor(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N // M]:
        a[i] = a[i + N // M] + b[i]


@dace.program
def read_ahead_constant_divisor(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N // 2]:
        a[i] = a[i + N // 2] + b[i]


@pytest.mark.parametrize('program', [read_ahead_symbolic_divisor, read_ahead_constant_divisor],
                         ids=['symbolic_divisor', 'constant_divisor'])
def test_read_ahead_keeps_its_map(program):
    """Both spellings canonicalize to one Map: no scan, no sequential loop left behind."""
    sdfg = program.to_sdfg(simplify=False)
    assert len(_maps(sdfg)) == 1, 'the frontend should give one Map to start from'

    canonicalize(sdfg, target='cpu', validate_all=False)

    assert len(_maps(sdfg)) == 1, f'expected exactly one Map, got {[m.map.label for m in _maps(sdfg)]}'
    assert not _loops(sdfg), f'a sequential loop survived: {[r.label for r in _loops(sdfg)]}'


def test_symbolic_divisor_matches_constant_divisor():
    """The two spellings differ only in the divisor, so they must canonicalize to the same shape."""
    shapes = []
    for program in (read_ahead_symbolic_divisor, read_ahead_constant_divisor):
        sdfg = program.to_sdfg(simplify=False)
        canonicalize(sdfg, target='cpu', validate_all=False)
        shapes.append((len(_maps(sdfg)), len(_loops(sdfg))))
    assert shapes[0] == shapes[1], f'symbolic {shapes[0]} != constant {shapes[1]}'


@dace.program
def prefix_scan(a: dace.float64[N], b: dace.float64[N]):
    for i in range(1, N):
        b[i] = b[i - 1] + a[i]


def test_a_real_recurrence_is_still_lifted():
    """The guard must not cost us actual scans: a backward carry is still a recurrence.

    Checked on the ``Scan`` library node, which is what a lift produces -- not on Map labels or a
    leftover loop, neither of which a successful lift leaves behind.
    """
    sdfg = prefix_scan.to_sdfg(simplify=False)
    canonicalize(sdfg, target='cpu', validate_all=False)
    scans = [n for st in sdfg.all_states() for n in st.nodes() if type(n).__name__ == 'Scan']
    assert scans, 'a genuine prefix scan must still lift to a Scan library node'


if __name__ == '__main__':
    test_read_ahead_keeps_its_map(read_ahead_symbolic_divisor)
    test_read_ahead_keeps_its_map(read_ahead_constant_divisor)
    test_symbolic_divisor_matches_constant_divisor()
    test_a_real_recurrence_is_still_lifted()
