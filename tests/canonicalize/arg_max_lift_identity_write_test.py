# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An argmax whose carrier write carries an arithmetic identity still lifts to ``ArgReduce``.

``x = y`` on a rank-0 value makes ``x`` a second NAME for ``y``'s container rather than a copy, so
a producer that needs a copy spells it ``x = y + 0``. The numpy-to-DaCe emitter does exactly that
for every scalar alias it lowers, which is how TSVC ``s318`` reaches the arg-reduce match as
``maxv := abs(a[k]) + 0.0``. Matching the surface syntax refused it and left a textbook argmax as
a sequential loop.
"""
import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import pipeline as canon

N = dace.symbol('N', dtype=dace.int64)


def lifted(sdfg: dace.SDFG) -> bool:
    """An ``ArgReduce`` is present and no sequential loop survives."""
    libs = [
        type(n).__name__ for g in sdfg.all_sdfgs_recursive() for st in g.all_states() for n in st.nodes()
        if isinstance(n, nodes.LibraryNode)
    ]
    loops = sum(1 for g in sdfg.all_sdfgs_recursive() for b in g.all_control_flow_regions(recursive=True)
                if isinstance(b, LoopRegion))
    return 'ArgReduce' in libs and loops == 0


def test_plain_carrier_write_lifts():
    """The form with no identity -- the control, so a failure here is not about the fold."""

    @dace.program
    def argmax_plain(a: dace.float64[N], result: dace.float64[1]):
        maxv = a[0]
        index = 0
        for i in range(1, N):
            if a[i] > maxv:
                index = i
                maxv = a[i]
        result[0] = maxv + float(index)

    sdfg = argmax_plain.to_sdfg(simplify=False)
    canon.canonicalize(sdfg)
    assert lifted(sdfg), 'the plain argmax did not lift'


def test_identity_on_the_carrier_write_still_lifts():
    """``maxv = v + 0.0`` is the same value write as ``maxv = v``."""

    @dace.program
    def argmax_identity(a: dace.float64[N], result: dace.float64[1]):
        maxv = abs(a[0])
        index = 0
        for i in range(1, N):
            v = abs(a[i])
            if v > maxv:
                index = i + 0
                maxv = v + 0.0
        result[0] = maxv + float(index)

    sdfg = argmax_identity.to_sdfg(simplify=False)
    canon.canonicalize(sdfg)
    assert lifted(sdfg), 'the identity-spelled carrier write blocked the lift'


def test_a_real_offset_is_not_folded_away():
    """Only the NEUTRAL constant is peeled -- ``maxv = v + 1.0`` writes a different value."""

    @dace.program
    def argmax_offset(a: dace.float64[N], result: dace.float64[1]):
        maxv = a[0]
        index = 0
        for i in range(1, N):
            if a[i] > maxv:
                index = i
                maxv = a[i] + 1.0
        result[0] = maxv + float(index)

    sdfg = argmax_offset.to_sdfg(simplify=False)
    canon.canonicalize(sdfg)
    libs = [
        type(n).__name__ for g in sdfg.all_sdfgs_recursive() for st in g.all_states() for n in st.nodes()
        if isinstance(n, nodes.LibraryNode)
    ]
    assert 'ArgReduce' not in libs, 'a carrier write that adds 1.0 must not be read as an argmax'


if __name__ == '__main__':
    test_plain_carrier_write_lifts()
    test_identity_on_the_carrier_write_still_lifts()
    test_a_real_offset_is_not_folded_away()
