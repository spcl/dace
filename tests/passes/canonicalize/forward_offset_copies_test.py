# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ForwardOffsetCopies``: a shifted copy read in later states is read from its source at the shifted index, and
the copy goes; a write to the source between the copy and a read keeps it."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.forward_offset_copies import ForwardOffsetCopies

N = dace.symbol("N")


def shifted_copy_sdfg(write_between: bool) -> dace.SDFG:
    """``t = q[1:N+1]`` in one state, ``out[i] = t[i] * 2`` in the next; with ``write_between`` a state between them
    sets ``q[:] = 0``."""
    sdfg = dace.SDFG("shifted_copy_" + ("written" if write_between else "clean"))
    sdfg.add_array("q", [N + 1], dace.float64)
    sdfg.add_array("out", [N], dace.float64)
    sdfg.add_transient("t", [N], dace.float64)
    copy = sdfg.add_state("copy")
    copy.add_nedge(copy.add_read("q"), copy.add_write("t"), dace.Memlet("q[1:N + 1] -> [0:N]"))
    last = copy
    if write_between:
        clobber = sdfg.add_state_after(copy, "clobber")
        clobber.add_mapped_tasklet(
            "zero", {"i": "0:N + 1"}, {}, "o = 0.0", {"o": dace.Memlet("q[i]")}, external_edges=True
        )
        last = clobber
    use = sdfg.add_state_after(last, "use")
    use.add_mapped_tasklet(
        "double",
        {"i": "0:N"},
        {"x": dace.Memlet("t[i]")},
        "o = x * 2.0",
        {"o": dace.Memlet("out[i]")},
        external_edges=True,
    )
    sdfg.validate()
    return sdfg


def test_a_shifted_copy_read_in_a_later_state_reads_its_source():
    sdfg = shifted_copy_sdfg(write_between=False)
    assert ForwardOffsetCopies().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert "t" not in sdfg.arrays
    reads = [str(e.data) for st in sdfg.states() for e in st.edges() if isinstance(e.dst, nodes.Tasklet)]
    assert reads == ["q[i + 1]"], reads
    q, out = np.random.default_rng(0).random(9), np.zeros(8)
    sdfg(q=q, out=out, N=8)
    assert np.allclose(out, 2.0 * q[1:])


def test_a_write_to_the_source_between_keeps_the_copy():
    sdfg = shifted_copy_sdfg(write_between=True)
    assert ForwardOffsetCopies().apply_pass(sdfg, {}) is None
    q, out = np.random.default_rng(1).random(9), np.zeros(8)
    expected = 2.0 * q[1:]
    sdfg(q=q, out=out, N=8)
    assert np.allclose(out, expected)


if __name__ == "__main__":
    test_a_shifted_copy_read_in_a_later_state_reads_its_source()
    test_a_write_to_the_source_between_keeps_the_copy()
