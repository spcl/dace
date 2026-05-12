"""Same array+subset read multiple times in one RHS -> one input connector.

The bridge used to mint one connector per *occurrence* — so
``B(i) = A(i) * A(i) * A(i)`` produced three identical ``_in_A_0/1/2``
connectors plus three identical memlets, all loading the same scalar.
After the dedup pass in ``emit_tasklet.py``, occurrences that share an
``(array_name, index_exprs)`` key share one connector + one memlet.

The check below confirms by inspecting the tasklet directly: count
of input connectors == count of DISTINCT (array, index) reads in
the RHS, not the total occurrence count.
"""
from __future__ import annotations

import numpy as np
import pytest

from dace.sdfg import nodes as nd

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _tasklet_for(sdfg, code_fragment: str):
    """Find the (single) tasklet whose code contains ``code_fragment``."""
    for state in sdfg.all_states():
        for n in state.nodes():
            if isinstance(n, nd.Tasklet) and code_fragment in n.code.as_string:
                return n
    return None


def test_dedup_same_index_three_times(tmp_path):
    test_string = """
                    SUBROUTINE cube(a, b)
                    double precision a(5), b(5)
                    integer i
                    DO i = 1, 5
                        b(i) = a(i) * a(i) * a(i)
                    ENDDO
                    END SUBROUTINE cube
                    """
    sdfg = build_sdfg(test_string, tmp_path, name='cube', entry='_QPcube').build()
    t = _tasklet_for(sdfg, '_in_a')
    assert t is not None, "couldn't find the cube tasklet"
    in_conns = [c for c in t.in_connectors if c.startswith('_in_a')]
    assert len(in_conns) == 1, f"expected one _in_a* connector, got {in_conns}"

    a = np.arange(1, 6, dtype=np.float64)
    b = np.zeros(5, dtype=np.float64)
    sdfg(a=a, b=b)
    assert np.allclose(b, a**3)


def test_no_dedup_when_index_differs(tmp_path):
    """Distinct subsets stay separate: ``A(i) + A(j)`` needs two
    connectors + two memlets even though the array name is the same.
    """
    test_string = """
                    SUBROUTINE pair_sum(a, b)
                    double precision a(5), b(5)
                    integer i, j
                    DO i = 1, 5
                        j = 6 - i
                        b(i) = a(i) + a(j)
                    ENDDO
                    END SUBROUTINE pair_sum
                    """
    sdfg = build_sdfg(test_string, tmp_path, name='pair_sum', entry='_QPpair_sum').build()
    t = _tasklet_for(sdfg, '_in_a')
    assert t is not None
    in_conns = sorted(c for c in t.in_connectors if c.startswith('_in_a'))
    assert len(in_conns) == 2, f"expected two _in_a* connectors, got {in_conns}"

    a = np.arange(1.0, 6.0, dtype=np.float64)
    b = np.zeros(5, dtype=np.float64)
    sdfg(a=a, b=b)
    # a[i-1] + a[5-i-1+1-1=5-i-1] = a[i-1] + a[5-i]
    expected = np.array([a[i] + a[4 - i] for i in range(5)], dtype=np.float64)
    assert np.allclose(b, expected)
