# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SDFGState.defined_symbols`` types interstate assignments from one environment kept per call."""
import dace
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import LoopRegion


def assigning_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG('assigning')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [dace.symbol('N')], dace.float32)
    sdfg.add_scalar('k', dace.float64, transient=True)
    first = sdfg.add_state('first', is_start_block=True)
    loop = LoopRegion('loop', 'i < N', 'i', 'i = 0', 'i = i + 1')
    loop.add_state('body', is_start_block=True)
    sdfg.add_node(loop)
    last = sdfg.add_state('last')
    # ``m`` is typed from ``N``, ``w`` from the array element, ``k`` shadows the scalar descriptor and
    # ``z`` is typed from ``k`` -- which must resolve to the DESCRIPTOR's dtype, not the symbol's.
    sdfg.add_edge(first, loop, InterstateEdge(assignments={'m': 'N + 1', 'w': 'A[0]'}))
    sdfg.add_edge(loop, last, InterstateEdge(assignments={'k': 'm * 2', 'z': 'k + 1'}))
    return sdfg


def reference_defined_symbols(state: dace.SDFGState) -> dict:
    """The per-edge environment the method used to rebuild, kept verbatim as the oracle."""
    sdfg = state.sdfg
    defined = dict(sdfg.symbols)
    for desc in sdfg.arrays.values():
        for sym in desc.free_symbols:
            if sym.dtype is not None:
                defined[str(sym)] = sym.dtype
    for edge in sdfg.all_interstate_edges():
        defined.update({k: v for k, v in edge.data.new_symbols(sdfg, defined).items() if v is not None})
        if isinstance(edge.dst, LoopRegion):
            defined.update({k: v for k, v in edge.dst.new_symbols(defined).items() if v is not None})
    return defined


def test_defined_symbols_types_every_assignment_as_before():
    sdfg = assigning_sdfg()
    state = next(s for s in sdfg.states() if s.label == 'last')
    assert state.defined_symbols() == reference_defined_symbols(state)
    assert state.defined_symbols()['z'] == dace.float64


def test_defined_symbols_builds_no_type_environment_per_edge(monkeypatch):
    """``new_symbols(sdfg, ...)`` copies the symbols and every array's dtype per assigning edge:
    2.4 s per call on warpx_field_gather, once per LoopToMap lift that nests a body."""
    sdfg = assigning_sdfg()
    state = next(s for s in sdfg.states() if s.label == 'last')
    rebuilt = []
    original = InterstateEdge.new_symbols

    def recorded(self, sd, symbols):
        if sd is not None:
            rebuilt.append(self)
        return original(self, sd, symbols)

    monkeypatch.setattr(InterstateEdge, 'new_symbols', recorded)
    state.defined_symbols()
    assert rebuilt == []
