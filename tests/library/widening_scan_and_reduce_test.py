# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A scan and a reduction may read a NARROWER input than they write, and must not fold at its width.

The case that forced this: stream compaction carries its predicate mask as ``int8`` -- one byte per
element instead of eight, which is most of the auxiliary traffic the three-phase lift spends -- and
prefix-sums it into ``int64`` ranks, which stay int64 because a rank is a write cursor, i.e. an
index. Every test here uses more than 127 set elements, so an accumulator at the INPUT's width wraps
and the assertion fails rather than passing on a lucky size.

The contract is one line: **the accumulator is the OUTPUT element type**. Two of ``Reduce``'s three
host expansions already held it -- ``ExpandReduceOpenMP`` through ``dace::reduce::sum<T, U>``
(``T`` seed/output, ``U`` input, cast per element) and ``ExpandReducePure`` by writing its WCR
straight into ``_out``. ``ExpandReducePureSequentialDim`` (``pure-seq``) did NOT: it staged an
accumulator at the INPUT's dtype, and that is the expansion ``ExpandReduceAuto`` dispatches to for
any ``Sequential`` reduction carrying an identity -- so the wrap was reachable, not theoretical.
``Scan`` refused a mismatched pair outright. Refusals are asserted here too: a widening that is not
value-preserving, and the shapes that have no widening implementation.
"""
import numpy as np
import pytest

import dace
from dace import memlet as mm
from dace.libraries.standard.nodes.scan import (INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME, ExpandCUDA, Scan, ScanOp,
                                                widening_is_value_preserving)

N = dace.symbol('N')

#: Well past int8's 127, so a fold at the input's width wraps.
_N = 4096


def scan_sdfg(in_dtype, out_dtype, implementation, exclusive=True, stride=1, chains=1):
    sdfg = dace.SDFG('widening_scan_%s_%s_%s' % (in_dtype.to_string(), out_dtype.to_string(), implementation))
    sdfg.add_array('A', [N], in_dtype)
    sdfg.add_array('B', [N], out_dtype)
    state = sdfg.add_state()
    scan = Scan(name='scan', op=ScanOp.SUM, exclusive=exclusive, identity=0, chains=chains)
    # ``stride`` is a Property, not a constructor argument.
    scan.stride = stride
    scan.implementation = implementation
    scan.schedule = dace.ScheduleType.CPU_Multicore
    state.add_node(scan)
    state.add_edge(state.add_read('A'), None, scan, INPUT_CONNECTOR_NAME, mm.Memlet('A[0:N]'))
    state.add_edge(scan, OUTPUT_CONNECTOR_NAME, state.add_write('B'), None, mm.Memlet('B[0:N]'))
    return sdfg, state, scan


def reduce_sdfg(in_dtype, out_dtype, implementation):
    sdfg = dace.SDFG('widening_reduce_%s_%s' % (in_dtype.to_string(), implementation.replace('-', '_')))
    sdfg.add_array('A', [N], in_dtype)
    sdfg.add_array('B', [1], out_dtype)
    state = sdfg.add_state()
    red = state.add_reduce('lambda x, y: x + y', None, 0)
    red.implementation = implementation
    red.schedule = dace.ScheduleType.CPU_Multicore
    state.add_edge(state.add_read('A'), None, red, '_in', mm.Memlet('A[0:N]'))
    state.add_edge(red, '_out', state.add_write('B'), None, mm.Memlet('B[0]'))
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize('implementation', ['pure', 'CPU'])
def test_exclusive_scan_widens_int8_ranks_to_int64(implementation):
    """The compaction shape: an all-ones mask exclusive-scanned gives rank[k] == k, up to 4095."""
    sdfg, _state, _scan = scan_sdfg(dace.int8, dace.int64, implementation)
    sdfg.validate()
    mask = np.ones(_N, np.int8)
    rank = np.zeros(_N, np.int64)
    sdfg(A=mask, B=rank, N=_N)
    assert np.array_equal(rank, np.arange(_N, dtype=np.int64))


@pytest.mark.parametrize('implementation', ['pure', 'CPU'])
def test_inclusive_scan_widens_int8_to_int64(implementation):
    sdfg, _state, _scan = scan_sdfg(dace.int8, dace.int64, implementation, exclusive=False)
    sdfg.validate()
    mask = np.ones(_N, np.int8)
    out = np.zeros(_N, np.int64)
    sdfg(A=mask, B=out, N=_N)
    assert np.array_equal(out, np.arange(1, _N + 1, dtype=np.int64))


@pytest.mark.parametrize('implementation', ['pure', 'pure-seq', 'OpenMP'])
def test_reduce_widens_int8_into_an_int64_total(implementation):
    """The other half of the compaction phase. ``pure-seq`` staged the INPUT's accumulator and wrapped."""
    sdfg = reduce_sdfg(dace.int8, dace.int64, implementation)
    total = np.zeros(1, np.int64)
    sdfg(A=np.ones(_N, np.int8), B=total, N=_N)
    assert int(total[0]) == _N


def test_the_widening_rule_is_value_preserving_only():
    """Integers only, strictly wider only, and never signed -> unsigned."""
    assert widening_is_value_preserving(dace.int8, dace.int64)
    assert widening_is_value_preserving(dace.uint8, dace.int64)
    assert widening_is_value_preserving(dace.uint16, dace.uint64)
    assert not widening_is_value_preserving(dace.int64, dace.int8), 'narrowing'
    assert not widening_is_value_preserving(dace.int32, dace.int32), 'same width is not a widening'
    assert not widening_is_value_preserving(dace.int8, dace.uint64), 'signed -> unsigned drops sign'
    assert not widening_is_value_preserving(dace.float32, dace.float64), 'reals are not covered'
    assert not widening_is_value_preserving(dace.int8, dace.float64), 'mixed kinds are not covered'


@pytest.mark.parametrize('pair', [(dace.int64, dace.int8), (dace.float32, dace.float64), (dace.int8, dace.uint64),
                                  (dace.int8, dace.float64)])
def test_a_non_widening_dtype_pair_is_refused(pair):
    sdfg, state, scan = scan_sdfg(pair[0], pair[1], 'pure')
    with pytest.raises(ValueError, match='dtype mismatch'):
        scan.validate(sdfg, state)


def test_widening_with_a_stride_is_refused():
    """One accumulator per residue class, each seeded from the input -- no widening design there."""
    sdfg, state, scan = scan_sdfg(dace.int8, dace.int64, 'pure', exclusive=False, stride=4)
    with pytest.raises(NotImplementedError, match='stride > 1'):
        scan.expand(sdfg, state)


def test_the_cuda_expansion_widens_through_its_exclusive_seed():
    """``cub::DeviceScan::ExclusiveScan`` deduces ``AccumT`` from the INIT VALUE, not the input.

    So the accumulator's width is an argument this expansion controls: seeding at the output's type
    makes the fold happen there. The seed has to carry that type explicitly -- a bare ``0`` literal
    is an ``int``, and an int8 -> int64 scan would then accumulate in 32 bits.
    """
    sdfg, state, scan = scan_sdfg(dace.int8, dace.int64, 'CUDA')
    tasklet = ExpandCUDA.expansion(scan, state, sdfg)
    wrapper = next(code for code in sdfg.global_code.values() if 'ExclusiveScan' in code.code)
    assert 'static_cast<long long>(0)' in wrapper.code or 'static_cast<int64_t>(0)' in wrapper.code, \
        f'the seed does not name the accumulator type:\n{wrapper.code}'
    assert 'cub::' not in tasklet.code.as_string, \
        f'the CUB call must stay in the CUDA unit, not the host tasklet:\n{tasklet.code.as_string}'


def test_an_inclusive_widening_scan_on_cuda_is_still_refused():
    """Nothing pins the accumulator there.

    A plain inclusive scan deduces from the input iterator, and a device-resident seed reaches cub as
    a ``FutureValue`` of the SEED's type -- neither is the output's. Refuse rather than fold narrow.
    """
    sdfg, state, scan = scan_sdfg(dace.int8, dace.int64, 'CUDA', exclusive=False)
    with pytest.raises(NotImplementedError, match='CUDA expansion without an exclusive seed'):
        ExpandCUDA.expansion(scan, state, sdfg)


if __name__ == '__main__':
    for impl in ('pure', 'CPU'):
        test_exclusive_scan_widens_int8_ranks_to_int64(impl)
        test_inclusive_scan_widens_int8_to_int64(impl)
    for impl in ('pure', 'pure-seq', 'OpenMP'):
        test_reduce_widens_int8_into_an_int64_total(impl)
    test_the_widening_rule_is_value_preserving_only()
