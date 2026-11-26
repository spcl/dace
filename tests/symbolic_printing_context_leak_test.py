# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The memoized printers' cache key must not keep generated SDFGs alive."""

import gc

import dace
from dace import symbolic


def scaled_sdfg(name: str) -> dace.SDFG:
    N = dace.symbol("N")
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "scale",
        {"i": "0:N"},
        {"x": dace.Memlet("a[N - i - 1]")},
        "y = x * 2.0 + 1.0",
        {"y": dace.Memlet("b[i]")},
        external_edges=True,
    )
    return sdfg


# Reproducer: every node's facts provider in the symstr/sym2cpp cache key kept the frame, the state and the SDFG alive,
# so an Alloy sweep grew by the whole program per candidate render and ran out of memory (warpx_esirkepov_deposition,
# ault 923577_1, 924037, 925386)
def test_no_copy_of_an_sdfg_outlives_its_code_generation() -> None:
    sdfg = scaled_sdfg("printing_context_leak")
    sdfg.generate_code()  # generates a copy of the SDFG
    del sdfg
    gc.collect()

    live = [o for o in gc.get_objects() if isinstance(o, dace.SDFG) and o.name == "printing_context_leak"]
    assert live == []


def test_printing_facts_match_their_own_cache_entries_while_they_last() -> None:
    def provider() -> symbolic.Facts:
        return symbolic.Facts.none()

    with symbolic.printing_facts(provider):  # a memoized printer hashes its key while the facts are set
        first, second = symbolic.printing_context(), symbolic.printing_context()
        hashes = hash(first), hash(second)
    with symbolic.printing_facts(provider):
        other = symbolic.printing_context()

        assert first == second and hashes[0] == hashes[1]
        assert other != first
