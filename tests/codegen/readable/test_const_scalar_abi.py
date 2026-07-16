# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``compiler.cpu.const_scalar_abi``: how the readable generator binds a READ-ONLY scalar.

``by_ref`` (default) emits ``const T& x`` -- the legacy convention; ``by_value`` emits ``const T x``.
The two are semantically identical, so both must produce bit-identical results; which one is FASTER
is a backend artifact, not a general rule (measured: on Neoverse-V2 clang vectorizes the by_ref form
of a masked reduction to an unpredicated ``fadd`` but the by_value form to a merging-predicated one,
which issues on fewer FP pipelines). Hence a knob, defaulting to the legacy ABI.

The binding is emitted by ``cpp.emit_memlet_reference`` for a nested-SDFG argument, which is the only
caller that passes an authoritative ``is_write=False`` (``vconn in node.out_connectors``); a plain
tasklet copy-in is a non-const local. So the fixture below is a nested SDFG with a read-only scalar
in-connector. The legacy generator ignores the flag entirely (its output stays byte-identical).
"""
import numpy as np
import pytest

import dace
from dace.config import set_temporary

from conftest import LEGACY, EXPERIMENTAL, use_implementation, generated_code, run_isolated, experimental_available

N = dace.symbol("N")
#: The scalar's declaration in the generated nested-SDFG signature.
BY_REF, BY_VALUE = "const double& ", "const double "


def _nsdfg_scalar_arg_sdfg(name="abi"):
    """out[j] *= s, with `s` a READ-ONLY scalar argument of a (non-inlined) nested SDFG."""
    inner = dace.SDFG("inner")
    inner.add_scalar("sc", dace.float64)
    inner.add_array("io", [N], dace.float64)
    ist = inner.add_state("n")
    ime, imx = ist.add_map("im", dict(j="0:N"))
    it = ist.add_tasklet("it", {"v", "s"}, {"w"}, "w = v * s")
    ist.add_memlet_path(ist.add_read("io"), ime, it, dst_conn="v", memlet=dace.Memlet("io[j]"))
    ist.add_memlet_path(ist.add_read("sc"), ime, it, dst_conn="s", memlet=dace.Memlet("sc[0]"))
    ist.add_memlet_path(it, imx, ist.add_write("io"), src_conn="w", memlet=dace.Memlet("io[j]"))

    sdfg = dace.SDFG(name)
    sdfg.add_array("o", [N], dace.float64)
    sdfg.add_transient("tmp", [1], dace.float64)
    st = sdfg.add_state("init")
    tk = st.add_tasklet("mk", {}, {"r"}, "r = 3.0")
    st.add_edge(tk, "r", st.add_access("tmp"), None, dace.Memlet("tmp[0]"))
    st2 = sdfg.add_state_after(st, "use")
    nn = st2.add_nested_sdfg(inner, {"sc", "io"}, {"io"}, symbol_mapping={"N": N})
    nn.no_inline = True  # keep the nested SDFG as a real function so its args are emitted
    st2.add_edge(st2.add_read("tmp"), None, nn, "sc", dace.Memlet("tmp[0]"))
    st2.add_edge(st2.add_read("o"), None, nn, "io", dace.Memlet("o[0:N]"))
    st2.add_edge(nn, "io", st2.add_write("o"), None, dace.Memlet("o[0:N]"))
    sdfg.validate()
    return sdfg


def _scalar_arg_decl(code):
    """The nested-SDFG signature's `sc` parameter declaration."""
    for line in code.splitlines():
        if "inner" in line and "sc" in line and "const double" in line:
            return line.strip()
    return ""


def _generate(impl, abi, sdfg=None):
    """Generated C++ for the fixture under (impl, abi). Pass an existing `sdfg` to generate the SAME
    object twice -- the SDFG name is baked into the output, so two differently-named builds are never
    byte-comparable."""
    sdfg = sdfg if sdfg is not None else _nsdfg_scalar_arg_sdfg(f"abi_{impl}_{abi}")
    with use_implementation(impl), set_temporary("compiler", "cpu", "codegen_params", "const_scalar_abi", value=abi):
        return generated_code(sdfg)


def test_readable_default_is_by_ref():
    """The readable generator's DEFAULT binds by const reference -- the legacy ABI."""
    if not experimental_available():
        pytest.skip("experimental readable codegen not ready")
    assert dace.Config.get("compiler", "cpu", "codegen_params", "const_scalar_abi") == "by_ref", "default must be by_ref"
    decl = _scalar_arg_decl(_generate(EXPERIMENTAL, "by_ref"))
    assert BY_REF in decl, decl


def test_readable_by_value():
    """``by_value`` binds by const value (a copy) -- no reference."""
    if not experimental_available():
        pytest.skip("experimental readable codegen not ready")
    decl = _scalar_arg_decl(_generate(EXPERIMENTAL, "by_value"))
    assert BY_VALUE in decl and BY_REF not in decl, decl


def test_legacy_ignores_the_flag():
    """Legacy always binds by const reference; the flag must not change its output at all."""
    if not experimental_available():
        pytest.skip("experimental readable codegen not ready")
    sdfg = _nsdfg_scalar_arg_sdfg("abi_legacy_noop")  # ONE object -> the two outputs are comparable
    by_ref = _generate(LEGACY, "by_ref", sdfg)
    by_value = _generate(LEGACY, "by_value", sdfg)
    assert by_ref == by_value, "the flag must be a no-op for the legacy generator"
    assert BY_REF in _scalar_arg_decl(by_ref), _scalar_arg_decl(by_ref)


def test_readable_matches_legacy_abi_by_default():
    """The readable default reproduces legacy's scalar binding exactly."""
    if not experimental_available():
        pytest.skip("experimental readable codegen not ready")
    assert _scalar_arg_decl(_generate(EXPERIMENTAL, "by_ref")).count(BY_REF) == \
        _scalar_arg_decl(_generate(LEGACY, "by_ref")).count(BY_REF)


@pytest.mark.parametrize("abi", ["by_ref", "by_value"])
def test_both_abis_are_bit_exact(abi):
    """Both bindings are semantically identical -> bit-identical results vs legacy."""
    if not experimental_available():
        pytest.skip("experimental readable codegen not ready")

    def run(impl, abi_value):

        def build_and_run():
            with use_implementation(impl), set_temporary("compiler", "cpu", "codegen_params", "const_scalar_abi", value=abi_value):
                sdfg = _nsdfg_scalar_arg_sdfg(f"abirun_{impl}_{abi_value}")
                csdfg = sdfg.compile()
            o = np.arange(16, dtype=np.float64)
            csdfg(o=o, N=16)
            return {"o": o}

        return run_isolated(build_and_run)

    legacy = run(LEGACY, "by_ref")
    experimental = run(EXPERIMENTAL, abi)
    assert np.array_equal(legacy["o"], experimental["o"]), f"{abi} is not bit-exact vs legacy"
    assert np.array_equal(experimental["o"], np.arange(16, dtype=np.float64) * 3.0)


if __name__ == "__main__":
    test_readable_default_is_by_ref()
    test_readable_by_value()
    test_legacy_ignores_the_flag()
    test_readable_matches_legacy_abi_by_default()
    for a in ("by_ref", "by_value"):
        test_both_abis_are_bit_exact(a)
    print("ok")
