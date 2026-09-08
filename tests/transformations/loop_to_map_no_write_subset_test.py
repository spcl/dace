# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LoopToMap``'s SMT fallback must refuse a write it has no subset for, not crash on it.

``Memlet.get_dst_subset`` returns ``None`` whenever the memlet names its edge's SOURCE rather
than its destination -- an ``A -> B`` copy carrying ``A``'s subset resolves ``dst_subset`` to the
unset ``other_subset``. ``can_be_applied`` handles that for its own affine test (``bool(dst_subset)
and _check_range(...)``), but that guard is exactly what routes such an edge on to the SMT oracle,
which then dereferenced the ``None`` it was handed. On CloudSC that surfaced as
``AttributeError: 'NoneType' object has no attribute 'ndrange'`` raised out of ``ParallelizeLoops``
-- a crash in the middle of canonicalization, not a refusal.
"""
import pytest

import dace
from dace.transformation.interstate.loop_to_map import _smt_proves_injective_write

N = dace.symbol('N')


def test_the_oracle_refuses_a_write_with_no_subset():
    """The documented contract is conservative refusal; ``None`` must take that path.

    Asserted directly on the oracle because that is where the contract lives: every caller reaches
    it through the branch where the subset test already failed, so any of them can hand it a
    ``None``, and a guard added at one call site would leave the others exposed.
    """
    assert _smt_proves_injective_write(None, dace.symbolic.pystr_to_symbolic('i'), 0, N - 1, 1) is False


def test_a_copy_carrying_the_source_subset_has_no_destination_subset():
    """The shape behind the ``None``: pin that this really is how such an edge resolves.

    Without this the test above is a statement about ``None`` in the abstract. Here a plain
    ``A -> B`` copy whose memlet names ``A`` is built and asked for its destination subset, which
    is what ``can_be_applied`` calls before deciding whether to consult the oracle.
    """
    sdfg = dace.SDFG('copy_named_after_source')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    state = sdfg.add_state('copy', is_start_block=True)
    src, dst = state.add_access('A'), state.add_access('B')
    edge = state.add_edge(src, None, dst, None, dace.Memlet(data='A', subset='0:N'))

    assert edge.data.get_dst_subset(edge, state) is None
    assert _smt_proves_injective_write(edge.data.get_dst_subset(edge, state), dace.symbolic.pystr_to_symbolic('i'), 0,
                                       N - 1, 1) is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
