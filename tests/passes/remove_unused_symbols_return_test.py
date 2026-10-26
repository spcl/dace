# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``RemoveUnusedSymbols`` returns ``None`` only when it left the SDFG untouched: deleting a dead interstate
assignment is a modification even when no declared symbol goes away."""

import dace
from dace.transformation.passes.prune_symbols import RemoveUnusedSymbols


def test_deleting_an_assignment_is_a_modification():
    sdfg = dace.SDFG("dead_assignment")
    first = sdfg.add_state("first", is_start_block=True)
    sdfg.add_state_after(first, "second", assignments={"dead": "1"})
    if "dead" in sdfg.symbols:
        sdfg.remove_symbol("dead")

    assert RemoveUnusedSymbols().apply_pass(sdfg, {}) is not None
    assert not any(e.data.assignments for e in sdfg.all_interstate_edges())
