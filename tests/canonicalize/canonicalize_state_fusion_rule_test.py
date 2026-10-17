# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalization fuses a demoted accumulator's load state with ``StateFusionExtended``, the rule ``FuseStates``
drives everywhere else in it, not with plain ``StateFusion``.

``DemoteNonStructuralSymbols`` puts the load of a demoted symbol accumulator in a state of its own and then fuses it
into the state that stores it (``fuse_into_successor``). Plain ``StateFusion`` refuses what the extended rule orders
and decides hazards differently, so one canonicalization judged two states by two rules.
"""

import pytest

import dace
from dace.transformation.interstate.state_fusion import StateFusion
from dace.transformation.passes.demote_non_structural_symbols import fuse_into_successor


@pytest.fixture
def plain_state_fusion_forbidden(monkeypatch):
    """Fail the test the moment plain ``StateFusion`` is consulted."""

    def forbidden(*args, **kwargs):
        raise AssertionError("asked plain StateFusion; canonicalization fuses with StateFusionExtended")

    monkeypatch.setattr(StateFusion, "can_be_applied", forbidden)
    monkeypatch.setattr(StateFusion, "apply", forbidden)


def load_then_store() -> tuple[dace.SDFG, dace.SDFGState]:
    """``c = o[0]`` in one state, ``o[0] = c + 1`` in the next: the demoted shape of ``o[0] = o[0] + 1``."""
    sdfg = dace.SDFG("load_then_store")
    sdfg.add_array("o", [1], dace.int64)
    sdfg.add_scalar("c", dace.int64, transient=True)
    load = sdfg.add_state("load")
    copy_in = load.add_tasklet("load", {"i"}, {"v"}, "v = i")
    load.add_edge(load.add_read("o"), None, copy_in, "i", dace.Memlet("o[0]"))
    load.add_edge(copy_in, "v", load.add_write("c"), None, dace.Memlet("c[0]"))
    store = sdfg.add_state_after(load, "store")
    bump = store.add_tasklet("store", {"v"}, {"o_out"}, "o_out = v + 1")
    store.add_edge(store.add_read("c"), None, bump, "v", dace.Memlet("c[0]"))
    store.add_edge(bump, "o_out", store.add_write("o"), None, dace.Memlet("o[0]"))
    return sdfg, load


def test_a_demoted_load_is_fused_by_the_extended_rule(plain_state_fusion_forbidden):
    sdfg, load = load_then_store()
    fuse_into_successor([load])
    assert len(list(sdfg.states())) == 1, "the load state must be fused into its store"
    sdfg.validate()


if __name__ == "__main__":
    with pytest.MonkeyPatch.context() as patch:
        plain_state_fusion_forbidden.__wrapped__(patch)
        test_a_demoted_load_is_fused_by_the_extended_rule(None)
