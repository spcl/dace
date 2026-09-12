# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Every corpus kernel's numpy oracle must actually compute something.

A gate entry whose oracle leaves its inputs untouched certifies nothing: the comparison is then
``inputs == inputs``, which an SDFG that computes NOTHING also satisfies. That is not hypothetical
-- it is how a tile constant larger than the array extent (``range(1, LEN_2D - 1 - 64, 64)`` with
``LEN_2D = 32`` is empty) and a threshold no sample ever clears (``standard_normal > 3``) sat in
the suite looking green.

Catching it needs no compilation: run the oracle on a copy and require that it changed at least one
compared array. Cheap enough to run over the whole corpus.
"""
import numpy as np
import pytest

import tests.corpus.measure_parallelization as mp
from tests.corpus.polybench import polybench as _PB
from tests.corpus.tsvc_2_5 import tsvc_2_5 as _T25

#: A paper-preset tsvc_2_5 dataset extent, and a per-thread block of the blocked scan.
_PAPER_N = 589824
_BLOCK = 4096


def _changed(arrays, ref) -> bool:
    """Did the oracle move any compared array off its input value?"""
    return any(not np.array_equal(np.asarray(arrays[n]), np.asarray(ref[n])) for n in arrays)


@pytest.mark.parametrize('name', mp.CORPORA['tsvc'][0]())
def test_tsvc_oracle_writes_something(name):
    arrays, _, ref = mp.tsvc_reference(name)
    assert _changed(arrays, ref), f'{name}: the numpy oracle left every array at its input value'


@pytest.mark.parametrize('name', mp.CORPORA['tsvc25'][0]())
def test_tsvc25_oracle_writes_something(name):
    program = [p for p in _T25.collect() if p.name == name][0]
    arrays, _, ref = mp.tsvc25_reference(program)
    assert _changed(arrays, ref), f'{name}: the numpy oracle left every array at its input value'


def _two_summation_orders(seed: int):
    """``(sequential, blocked)`` prefix sums of one paper-scale fp64 dataset.

    Both are correct inclusive scans of the same data; they differ only in association,
    which is what every parallelized scan and reduction in the corpus does to its
    reference. The blocked order is the one ``dace/scan.hpp`` runs.
    """
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(_PAPER_N)
    blocks = d.reshape(-1, _BLOCK)
    offsets = np.concatenate(([0.0], np.cumsum(blocks.sum(axis=1))[:-1]))
    return np.cumsum(d), (np.cumsum(blocks, axis=1) + offsets[:, None]).ravel()


def test_gate_accepts_reassociation_at_paper_scale():
    """The gate must not call a DIFFERENT SUMMATION ORDER a miscompile.

    ``_tol_for``'s absolute term is a constant, so it stops covering the reference's own
    rounding once the dataset is large enough -- at the paper preset a 589824-long fp64
    prefix sum carries ~3e-11 of it, past the 1e-11 constant, and elements that cancelled
    to near zero have no relative tolerance left to absorb it. Held to the constant the
    gate reported tsvc_2_5 ``scan_multi_5carry`` as wrong under canonicalize while the
    canonicalize answer was the more accurate of the two (2.8e-12 against a long-double
    reference, versus the scalar oracle's 2.7e-11). ``REASSOC_SCALE`` is what closes that.
    """
    seq, blocked = _two_summation_orders(11)
    assert not np.array_equal(seq, blocked), 'the two summation orders must actually differ'
    assert np.max(np.abs(seq - blocked)) > _PB._tol_for(np.float64)[1], \
        'this dataset no longer exceeds the constant floor, so it gates nothing'
    assert _PB.outputs_match({'acc': seq}, {'acc': blocked}), \
        (f'the gate rejected a legal reassociation: max|diff|={np.max(np.abs(seq - blocked)):.3e} '
         f'against a floor of {_PB.atol_for(seq, _PB._tol_for(np.float64)[1]):.3e}')


def test_gate_still_rejects_a_real_error_at_paper_scale():
    """The floor is scale-relative, not a blanket pass: one element off by more than it
    must still be caught -- including at an element that cancelled to near zero, which is
    exactly where the floor does its work and so exactly where it could blind the gate."""
    seq, blocked = _two_summation_orders(11)
    floor = _PB.atol_for(seq, _PB._tol_for(np.float64)[1])
    victim = int(np.argmin(np.abs(seq)))
    wrong = blocked.copy()
    wrong[victim] += 1000.0 * floor
    assert not _PB.outputs_match({'acc': seq}, {'acc': wrong}), \
        f'an error of {1000.0 * floor:.3e} at acc[{victim}] (|value|={abs(seq[victim]):.3e}) went unnoticed'


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
