"""Codegen-time drift check — ``codegen.generate_code`` must refuse
to emit C++ headers when the SDFG's live arglist has drifted from
its attached ``FrozenSignature``.

This is the contract that keeps an already-generated Fortran
binding honest: mutating the SDFG after ``build()`` can still be
useful for other purposes, but you can't SHIP a compiled library
whose signature disagrees with the wrapper that calls it.
"""
from __future__ import annotations

import dace
import pytest

from dace.codegen import codegen
from dace.frontend.hlfir.bindings import (
    FrozenArg,
    FrozenSignature,
    SignatureDriftError,
)


def _demo_sdfg() -> dace.SDFG:
    """Small SDFG: ``a`` and ``b`` as two non-transient float64
    arrays + one free symbol ``n``."""
    sdfg = dace.SDFG("demo")
    sdfg.add_symbol("n", dace.int64)
    sdfg.add_array("a", shape=(dace.symbol("n"), ), dtype=dace.float64, transient=False)
    sdfg.add_array("b", shape=(dace.symbol("n"), ), dtype=dace.float64, transient=False)
    return sdfg


def _pin(sdfg: dace.SDFG) -> FrozenSignature:
    fs = FrozenSignature(
        entry="demo",
        mangled="_QPdemo",
        args=(
            FrozenArg(fortran_name="a",
                      sdfg_name="a",
                      kind="array",
                      dtype="float64",
                      rank=1,
                      shape=("n", ),
                      intent="in"),
            FrozenArg(fortran_name="b",
                      sdfg_name="b",
                      kind="array",
                      dtype="float64",
                      rank=1,
                      shape=("n", ),
                      intent="inout"),
        ),
        free_symbols=("n", ),
    )
    sdfg._frozen_signature = fs
    return fs


def test_codegen_honours_frozen_signature_happy_path():
    """When the SDFG hasn't drifted, codegen generates code normally."""
    sdfg = _demo_sdfg()
    _pin(sdfg)
    # Needs at least one state to be a legal SDFG.
    sdfg.add_state("s0", is_start_block=True)
    # Should NOT raise.
    codegen.generate_code(sdfg, validate=False)


def test_codegen_raises_on_arg_removal():
    """Transformation dropped array ``b`` → drift → raise."""
    sdfg = _demo_sdfg()
    _pin(sdfg)
    sdfg.add_state("s0", is_start_block=True)
    del sdfg.arrays["b"]
    with pytest.raises(SignatureDriftError):
        codegen.generate_code(sdfg, validate=False)


def test_codegen_raises_on_dtype_change():
    """Transformation changed ``a`` to float32 → drift → raise."""
    sdfg = _demo_sdfg()
    _pin(sdfg)
    sdfg.add_state("s0", is_start_block=True)
    sdfg.arrays["a"].dtype = dace.float32
    with pytest.raises(SignatureDriftError):
        codegen.generate_code(sdfg, validate=False)


def test_codegen_unpinned_sdfg_unaffected():
    """A plain SDFG without a frozen signature goes through codegen
    without any drift check — the pinning is purely opt-in."""
    sdfg = _demo_sdfg()
    sdfg.add_state("s0", is_start_block=True)
    # No frozen signature set.
    assert not hasattr(sdfg, "_frozen_signature") or sdfg._frozen_signature is None
    codegen.generate_code(sdfg, validate=False)
