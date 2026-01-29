# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import pytest

from dace import SDFG
from dace.sdfg import InvalidSDFGError, InterstateEdge


def test_validation_no_state():
    """SDFGs require a start block."""
    sdfg = SDFG("empty_sdfg")

    with pytest.raises(InvalidSDFGError, match="SDFGs are required to contain at least one state."):
        sdfg.validate()


def test_validation_ambiguous_start_block():
    """SDFGs require an unambiguous start block."""
    sdfg = SDFG("ambiguous_start_block")
    state_1 = sdfg.add_state("state_1")
    state_2 = sdfg.add_state("state_2")

    with pytest.raises(InvalidSDFGError, match="Starting block is ambiguous or undefined."):
        sdfg.validate()

    # Disambiguate by adding an edge between state_1 and state_2.
    sdfg.add_edge(state_1, state_2, InterstateEdge())
    assert sdfg.is_valid()


if __name__ == "__main__":
    test_validation_no_state()
    test_validation_ambiguous_start_block()
