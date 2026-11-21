# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""SDFG.frontend_metadata survives a save/load round-trip and is not shared between SDFGs."""

import dace


def test_frontend_metadata_round_trips_through_a_file(tmp_path):
    sdfg = dace.SDFG("frontend_metadata")
    sdfg.add_state()
    sdfg.frontend_metadata = {"signature": {"args": ["a", "b"], "ranks": [1, 2]}}
    path = tmp_path / "meta.sdfgz"
    sdfg.save(path, compress=True)
    assert dace.SDFG.from_file(path).frontend_metadata == {"signature": {"args": ["a", "b"], "ranks": [1, 2]}}


def test_frontend_metadata_is_per_sdfg():
    first, second = dace.SDFG("first"), dace.SDFG("second")
    first.frontend_metadata["key"] = 1
    assert second.frontend_metadata == {}


if __name__ == "__main__":
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_frontend_metadata_round_trips_through_a_file(pathlib.Path(tmp))
    test_frontend_metadata_is_per_sdfg()
