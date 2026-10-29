# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace
from dace.transformation.passes.minimize_transients import MinimizeTransients


def scaled_rows_sdfg(name: str, copy_out: bool, clobber: bool = False) -> dace.SDFG:
    """``t = 2 * a`` (a map) then ``b[1, :] = t`` (copy out), or ``t = b[1, :]`` then ``a = 2 * t`` (copy in);
    ``clobber`` writes ``b[1, :]`` again where the copy-in's readers may still run."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [8], dace.float64)
    sdfg.add_array("b", [3, 8], dace.float64)
    sdfg.add_array("t", [8], dace.float64, transient=True)
    state = sdfg.add_state()
    t = state.add_access("t")
    if copy_out:
        state.add_mapped_tasklet(
            "scale",
            {"i": "0:8"},
            {"x": dace.Memlet("a[i]")},
            "y = 2 * x",
            {"y": dace.Memlet("t[i]")},
            external_edges=True,
            input_nodes={"a": state.add_read("a")},
            output_nodes={"t": t},
        )
        state.add_nedge(t, state.add_write("b"), dace.Memlet("t[0:8] -> [1, 0:8]"))
    else:
        b = state.add_read("b")
        state.add_nedge(b, t, dace.Memlet("b[1, 0:8] -> [0:8]"))
        state.add_mapped_tasklet(
            "scale",
            {"i": "0:8"},
            {"x": dace.Memlet("t[i]")},
            "y = 2 * x",
            {"y": dace.Memlet("a[i]")},
            external_edges=True,
            input_nodes={"t": t},
            output_nodes={"a": state.add_write("a")},
        )
        if clobber:
            state.add_mapped_tasklet(
                "zero", {"i": "0:8"}, {}, "y = 0", {"y": dace.Memlet("b[1, i]")}, external_edges=True
            )
    return sdfg


def run(sdfg: dace.SDFG) -> tuple[np.ndarray, np.ndarray]:
    a = np.arange(8, dtype=np.float64)
    b = np.arange(24, dtype=np.float64).reshape(3, 8)
    sdfg(a=a, b=b)
    return a, b


def test_copied_out_transient_is_its_target_region():
    sdfg = scaled_rows_sdfg("minimize_transients_copy_out", copy_out=True)
    expected = run(sdfg)
    assert MinimizeTransients().apply_pass(sdfg, {}) == 1
    assert "t" not in sdfg.arrays
    sdfg.validate()
    for got, want in zip(run(sdfg), expected):
        np.testing.assert_array_equal(got, want)


def test_copied_in_transient_reads_its_source_region():
    sdfg = scaled_rows_sdfg("minimize_transients_copy_in", copy_out=False)
    expected = run(sdfg)
    assert MinimizeTransients().apply_pass(sdfg, {}) == 1
    assert "t" not in sdfg.arrays
    sdfg.validate()
    for got, want in zip(run(sdfg), expected):
        np.testing.assert_array_equal(got, want)


def test_snapshot_of_a_region_written_concurrently_stays():
    sdfg = scaled_rows_sdfg("minimize_transients_clobber", copy_out=False, clobber=True)
    assert MinimizeTransients().apply_pass(sdfg, {}) is None
    assert "t" in sdfg.arrays


if __name__ == "__main__":
    test_copied_out_transient_is_its_target_region()
    test_copied_in_transient_reads_its_source_region()
    test_snapshot_of_a_region_written_concurrently_stays()
