# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :class:`ClearPerLaneIndexSymbols` (G10; design section 10.6)."""
import pytest

import dace
from dace.transformation.passes.vectorization.clear_per_lane_index_symbols import (ClearPerLaneIndexSymbols)


def test_clean_sdfg_passes_audit():
    """An SDFG with no per-lane symbol names triggers no failure."""
    sdfg = dace.SDFG("clean")
    sdfg.add_array("A", (16, ), dace.float64, transient=False)
    sdfg.add_state("s")
    result = ClearPerLaneIndexSymbols().apply_pass(sdfg, {})
    assert result is None


def test_audit_refuses_legacy_laneid_in_array_name():
    """Legacy 1D ``<base>_laneid_<n>`` array name in the SDFG triggers AssertionError."""
    sdfg = dace.SDFG("leaked_legacy")
    sdfg.add_array("x_laneid_3", (1, ), dace.float64, transient=True)
    sdfg.add_state("s")
    with pytest.raises(AssertionError, match=r"per-lane scalar"):
        ClearPerLaneIndexSymbols().apply_pass(sdfg, {})


def test_audit_refuses_canonical_lane_d_id_in_symbol_name():
    """Canonical multi-dim ``<base>_lane<d>id_<n>`` symbol triggers AssertionError."""
    sdfg = dace.SDFG("leaked_canonical")
    sdfg.add_symbol("sym_lane0id_2", dace.int64)
    sdfg.add_state("s")
    with pytest.raises(AssertionError, match=r"per-lane scalar"):
        ClearPerLaneIndexSymbols().apply_pass(sdfg, {})


def test_audit_walks_into_nested_sdfgs():
    """A leak inside a nested SDFG is detected (recursive walk)."""
    sdfg = dace.SDFG("outer")
    sdfg.add_array("A", (16, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    inner = dace.SDFG("inner")
    inner.add_array("A", (16, ), dace.float64, transient=False)
    inner.add_array("y_laneid_0", (1, ), dace.float64, transient=True)
    inner.add_state("body")
    state.add_nested_sdfg(inner, {"A"}, set())
    with pytest.raises(AssertionError, match=r"per-lane scalar"):
        ClearPerLaneIndexSymbols().apply_pass(sdfg, {})


def test_audit_message_lists_offending_names():
    """The failure message names at least one of the leaked symbols + the count."""
    sdfg = dace.SDFG("leaked_multiple")
    sdfg.add_array("p_laneid_0", (1, ), dace.float64, transient=True)
    sdfg.add_array("q_laneid_1", (1, ), dace.float64, transient=True)
    sdfg.add_state("s")
    with pytest.raises(AssertionError) as ei:
        ClearPerLaneIndexSymbols().apply_pass(sdfg, {})
    msg = str(ei.value)
    assert "p_laneid_0" in msg or "q_laneid_1" in msg
    assert "2" in msg  # leak count
