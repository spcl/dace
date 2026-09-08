# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``collect_element_write_subsets`` must report the DESTINATION region of a copy.

An AN-to-AN copy edge carries one endpoint's region in ``memlet.subset`` and the other's in
``memlet.other_subset``, and which end ``subset`` names is the memlet's orientation -- for a copy
built from the source it is the region READ. Reading ``edge.data.subset`` therefore answered
``i`` for the copy ``delta[i] -> buf[i - 1]``.

``SameWriteSetIfElseToITECFG`` sizes its ITE memlet from this query, so the shifted answer made a
masked prefix scan (TSVC-2.5 ``scan_conditional``) write ``buf[i]`` where the arm wrote
``buf[i - 1]``: every value landed one cell late, the first cell kept its initializer, and the last
iteration ran one element past a buffer of ``N - 1``. The overrun surfaced as
``malloc(): invalid size (unsorted)`` or a SIGSEGV, which read as flakiness rather than as the
off-by-one it was.

``BypassTrivialAssignTasklets`` is what produces the bare copy: the arm is written with a tasklet
in between, and bypassing it leaves the source-oriented memlet this query then misread.
"""
import dace
import pytest

from dace.transformation.passes.vectorization.utils.queries import collect_element_write_subsets

N = dace.symbol("N")


def _copy_state(write_offset: str):
    """``delta[i] -> buf[<write_offset>]`` as a bare AN-to-AN copy, oriented on the SOURCE."""
    sdfg = dace.SDFG("copy_orientation")
    sdfg.add_array("delta", (N, ), dace.float64)
    sdfg.add_array("buf", (N, ), dace.float64)
    sdfg.add_symbol("i", dace.int64)
    state = sdfg.add_state("s", is_start_block=True)
    src = state.add_access("delta")
    dst = state.add_access("buf")
    # Source-oriented: ``data`` names ``delta``, so ``subset`` is the READ region and the write
    # region is in ``other_subset`` -- the orientation a bypassed assign tasklet leaves behind.
    state.add_edge(src, None, dst, None, dace.Memlet(data="delta", subset="i", other_subset=write_offset))
    return sdfg, state


@pytest.mark.parametrize("write_offset", ["i - 1", "i", "i + 2"])
def test_copy_write_subset_is_the_destination_side(write_offset):
    sdfg, state = _copy_state(write_offset)
    subsets = collect_element_write_subsets(state)

    assert subsets is not None, "an element-wise copy must not be reported as non-element-wise"
    assert set(subsets) == {"buf"}, f"only the written array is a write, got {sorted(subsets)}"
    assert str(subsets["buf"]) == write_offset, (
        f"reported the source region instead of the destination: {subsets['buf']} != {write_offset}")


def test_tasklet_write_subset_is_unchanged():
    """The tasklet-written case is destination-oriented already and must keep answering the same."""
    sdfg = dace.SDFG("tasklet_orientation")
    sdfg.add_array("buf", (N, ), dace.float64)
    sdfg.add_symbol("i", dace.int64)
    state = sdfg.add_state("s", is_start_block=True)
    t = state.add_tasklet("w", {}, {"_o"}, "_o = 1.0")
    state.add_edge(t, "_o", state.add_access("buf"), None, dace.Memlet("buf[i - 1]"))

    subsets = collect_element_write_subsets(state)
    assert subsets is not None
    assert str(subsets["buf"]) == "i - 1"


def test_multi_element_copy_is_refused():
    """A non-element-wise WRITE must refuse, even when the read side happens to be one element."""
    sdfg = dace.SDFG("wide_write")
    sdfg.add_array("delta", (N, ), dace.float64)
    sdfg.add_array("buf", (N, ), dace.float64)
    sdfg.add_symbol("i", dace.int64)
    state = sdfg.add_state("s", is_start_block=True)
    state.add_edge(state.add_access("delta"), None, state.add_access("buf"), None,
                   dace.Memlet(data="delta", subset="0:N", other_subset="0:N"))

    assert collect_element_write_subsets(state) is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
