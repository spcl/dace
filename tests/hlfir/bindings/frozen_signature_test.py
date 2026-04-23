"""``FrozenSignature`` self-contained tests — JSON round-trip, drift
detection.  These don't need the full HLFIR pipeline or flang-new on
PATH; they exercise the binding-side plumbing in isolation.
"""
from __future__ import annotations

from pathlib import Path

import dace
import pytest

from dace.frontend.hlfir.bindings import (
    FrozenArg,
    FrozenSignature,
    SignatureDriftError,
)


def _demo_signature() -> FrozenSignature:
    return FrozenSignature(
        entry="compute",
        mangled="_QPcompute",
        args=(
            FrozenArg(fortran_name="a",
                      sdfg_name="a",
                      kind="array",
                      dtype="float64",
                      rank=2,
                      shape=("n", "m"),
                      intent="in"),
            FrozenArg(fortran_name="b",
                      sdfg_name="b",
                      kind="array",
                      dtype="float64",
                      rank=2,
                      shape=("n", "m"),
                      intent="inout"),
        ),
        free_symbols=("m", "n"),
    )


def test_json_roundtrip(tmp_path: Path):
    fs = _demo_signature()
    p = tmp_path / "sig.json"
    fs.to_json(str(p))
    loaded = FrozenSignature.from_json(str(p))
    assert loaded == fs


def test_verify_against_happy_path():
    sdfg = dace.SDFG("compute")
    sdfg.add_symbol("m", dace.int64)
    sdfg.add_symbol("n", dace.int64)
    sdfg.add_array("a", shape=(dace.symbol("n"), dace.symbol("m")), dtype=dace.float64, transient=False)
    sdfg.add_array("b", shape=(dace.symbol("n"), dace.symbol("m")), dtype=dace.float64, transient=False)

    fs = _demo_signature()
    # No raise.
    fs.verify_against(sdfg)


def test_drift_detection_arg_reordering():
    sdfg = dace.SDFG("compute")
    sdfg.add_symbol("m", dace.int64)
    sdfg.add_symbol("n", dace.int64)
    # Swap order vs the snapshot.
    sdfg.add_array("b", shape=(dace.symbol("n"), dace.symbol("m")), dtype=dace.float64, transient=False)
    sdfg.add_array("a", shape=(dace.symbol("n"), dace.symbol("m")), dtype=dace.float64, transient=False)

    fs = _demo_signature()
    # arglist sorts alphabetically, so (a, b) in live — the snapshot
    # order is also (a, b), so this case is actually fine.  Mutate the
    # snapshot to flip the order and assert drift.
    swapped = FrozenSignature(
        entry=fs.entry,
        mangled=fs.mangled,
        args=(fs.args[1], fs.args[0]),
        free_symbols=fs.free_symbols,
    )
    with pytest.raises(SignatureDriftError):
        swapped.verify_against(sdfg)


def test_drift_detection_dtype_change():
    sdfg = dace.SDFG("compute")
    sdfg.add_symbol("m", dace.int64)
    sdfg.add_symbol("n", dace.int64)
    # Use float32 on the live SDFG; frozen says float64.
    sdfg.add_array("a", shape=(dace.symbol("n"), dace.symbol("m")), dtype=dace.float32, transient=False)
    sdfg.add_array("b", shape=(dace.symbol("n"), dace.symbol("m")), dtype=dace.float64, transient=False)

    fs = _demo_signature()
    with pytest.raises(SignatureDriftError) as exc:
        fs.verify_against(sdfg)
    assert "dtype" in str(exc.value)


def test_drift_detection_extra_free_symbol():
    sdfg = dace.SDFG("compute")
    sdfg.add_symbol("m", dace.int64)
    sdfg.add_symbol("n", dace.int64)
    sdfg.add_symbol("extra", dace.int64)  # not in snapshot
    sdfg.add_array("a", shape=(dace.symbol("n"), dace.symbol("m")), dtype=dace.float64, transient=False)
    sdfg.add_array("b", shape=(dace.symbol("n"), dace.symbol("m")), dtype=dace.float64, transient=False)

    # Use `extra` somewhere so it counts as used.
    s0 = sdfg.add_state("s0")
    s1 = sdfg.add_state("s1")
    sdfg.add_edge(s0, s1, dace.InterstateEdge(condition="extra > 0"))

    fs = _demo_signature()
    with pytest.raises(SignatureDriftError):
        fs.verify_against(sdfg)
