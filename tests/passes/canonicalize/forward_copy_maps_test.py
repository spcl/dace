# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ForwardCopyMaps``: a transposed column copy (nbody's ``x.T``) read element by element is read from the column; a
reader that also writes the column keeps the copy."""

import copy

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.forward_copy_maps import ForwardCopyMaps

N = dace.symbol("N")


def column_copy_sdfg(name: str, reader_writes_source: bool = False) -> dace.SDFG:
    """``t[0, i] = pos[i, 2]``; ``out[i, j] = t[0, j] - pos[i, 2]``; then ``pos[i, 2] = out[i, 0]`` (``2 * t[0, i]``
    with ``reader_writes_source``)."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("pos", [N, 3], dace.float64)
    sdfg.add_array("out", [N, N], dace.float64)
    sdfg.add_transient("t", [1, N], dace.float64)
    state = sdfg.add_state("main")
    pos, t, out = state.add_read("pos"), state.add_access("t"), state.add_access("out")
    state.add_mapped_tasklet(
        "copy",
        {"i": "0:N"},
        {"v": dace.Memlet("pos[i, 2]")},
        "o = v",
        {"o": dace.Memlet("t[0, i]")},
        external_edges=True,
        input_nodes={"pos": pos},
        output_nodes={"t": t},
    )
    state.add_mapped_tasklet(
        "diff",
        {"i": "0:N", "j": "0:N"},
        {"a": dace.Memlet("t[0, j]"), "b": dace.Memlet("pos[i, 2]")},
        "o = a - b",
        {"o": dace.Memlet("out[i, j]")},
        external_edges=True,
        input_nodes={"t": t, "pos": pos},
        output_nodes={"out": out},
    )
    updated = state.add_write("pos")
    source, code, memlet = (t, "o = 2.0 * v", "t[0, i]") if reader_writes_source else (out, "o = v", "out[i, 0]")
    state.add_mapped_tasklet(
        "update",
        {"i": "0:N"},
        {"v": dace.Memlet(memlet)},
        code,
        {"o": dace.Memlet("pos[i, 2]")},
        external_edges=True,
        input_nodes={source.data: source},
        output_nodes={"pos": updated},
    )
    sdfg.validate()
    return sdfg


def run(sdfg: dace.SDFG, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos, out = pos.copy(), np.zeros((pos.shape[0], pos.shape[0]))
    sdfg(pos=pos, out=out, N=pos.shape[0])
    return pos, out


def test_a_transposed_column_copy_is_read_from_the_column():
    sdfg = column_copy_sdfg("column_copy")
    assert ForwardCopyMaps().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert "t" not in sdfg.arrays
    state = sdfg.start_block
    assert [n.map.label for n in state.nodes() if isinstance(n, nodes.MapEntry)] == ["diff_map", "update_map"]
    pos = np.random.default_rng(3).random((5, 3))
    new_pos, out = run(sdfg, pos)
    assert np.allclose(out, pos[:, 2][None, :] - pos[:, 2][:, None])
    assert np.allclose(new_pos[:, 2], pos[0, 2] - pos[:, 2]) and np.allclose(new_pos[:, :2], pos[:, :2])


def test_a_reader_that_writes_the_column_keeps_the_copy():
    sdfg = column_copy_sdfg("column_copy_rewritten", reader_writes_source=True)
    reference = copy.deepcopy(sdfg)
    reference.name = "column_copy_rewritten_reference"
    assert ForwardCopyMaps().apply_pass(sdfg, {}) is None
    assert "t" in sdfg.arrays
    pos = np.random.default_rng(4).random((5, 3))
    new_pos, out = run(sdfg, pos)
    assert np.allclose(out, pos[:, 2][None, :] - pos[:, 2][:, None])
    assert np.allclose(new_pos[:, 2], 2.0 * pos[:, 2])
    ref_pos, ref_out = run(reference, pos)
    assert np.allclose(new_pos, ref_pos) and np.allclose(out, ref_out)


if __name__ == "__main__":
    test_a_transposed_column_copy_is_read_from_the_column()
    test_a_reader_that_writes_the_column_keeps_the_copy()
