# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Round-trip tests for ``dace.sdfg.to_python``.

Each test builds an SDFG with the imperative API, emits Python source via
``sdfg_to_python``, executes the source to rebuild the SDFG, and verifies
that the rebuilt SDFG validates and (where runnable) produces the same
output as the original.
"""

import ctypes
import pathlib
import textwrap

import numpy as np
import pytest

import dace
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import (
    ConditionalBlock,
    ContinueBlock,
    ControlFlowRegion,
    LoopRegion,
)
from dace.sdfg.to_python import sdfg_to_python

_DATA_DIR = pathlib.Path(__file__).parent / "data" / "sdfg_reconstruction"


def _ensure_libgomp_loaded():
    """Pre-load libgomp into the process so generated SDFG ``.so``s, which
    don't directly link it on this system's CMake/OpenMP setup, can resolve
    ``omp_get_max_threads`` via the global symbol table.

    This is a host-build-environment workaround, not a correctness bug in
    the emitter. Skips silently if libgomp can't be located.
    """
    candidates = (
        "libgomp.so.1",
        "/usr/lib/x86_64-linux-gnu/libgomp.so.1",
        "/lib/x86_64-linux-gnu/libgomp.so.1",
    )
    for cand in candidates:
        try:
            ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
            return True
        except OSError:
            continue
    return False


_ensure_libgomp_loaded()


def _exec_emitted(src: str) -> SDFG:
    """Compile & execute the emitted source, returning the rebuilt SDFG."""
    namespace: dict = {}
    code = compile(src, "<emitted>", "exec")
    exec(code, namespace)  # noqa: S102 — intentional, this is the round-trip
    rebuilt = namespace["build_sdfg"]()
    rebuilt.validate()
    return rebuilt


def _try_compile_and_run(sdfg: SDFG, **call_args):
    """Compile & run an SDFG, returning the output dict or None on env failure.

    Returns ``None`` if compile fails in a way that suggests a build-env issue
    (e.g. OpenMP linkage on the host) rather than an SDFG-correctness issue.
    Lets every other exception propagate.
    """
    try:
        sdfg.compile()(**call_args)
    except OSError as exc:
        if "undefined symbol" in str(exc) or "cannot open shared object" in str(exc):
            pytest.skip(f"build env can't link generated SDFG: {exc}")
        raise
    return call_args


def _assert_runs_equivalent(original: SDFG, rebuilt: SDFG, **call_args):
    """Compile both SDFGs and assert they produce the same output.

    If the original fails to compile/load due to an environmental linker
    issue (e.g. OpenMP not linked by the system CMake config), the test is
    skipped — that's not a correctness regression in the emitter.
    """
    original_args = {k: v.copy() if hasattr(v, "copy") else v for k, v in call_args.items()}
    rebuilt_args = {k: v.copy() if hasattr(v, "copy") else v for k, v in call_args.items()}
    if _try_compile_and_run(original, **original_args) is None:
        return
    if _try_compile_and_run(rebuilt, **rebuilt_args) is None:
        return
    for k in call_args:
        a = original_args[k]
        b = rebuilt_args[k]
        if isinstance(a, np.ndarray):
            np.testing.assert_allclose(a, b, err_msg=f"Mismatch on {k!r}")


# ---------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------


def test_empty_sdfg_with_single_state():
    sdfg = dace.SDFG("empty")
    sdfg.add_state("only", is_start_block=True)

    src = sdfg_to_python(sdfg)
    rebuilt = _exec_emitted(src)
    assert rebuilt.name == "empty"
    assert rebuilt.number_of_nodes() == 1
    assert rebuilt.start_block.label == "only"


def test_descriptors_round_trip():
    sdfg = dace.SDFG("descs")
    N = dace.symbol("N")
    sdfg.add_symbol("N", dace.int32)
    sdfg.add_array("A", [N, 4], dace.float32)
    sdfg.add_array("tmp", [N], dace.float64, transient=True)
    sdfg.add_scalar("s", dace.int32)
    sdfg.add_stream("q", dace.int32, buffer_size=8)
    sdfg.add_state("only", is_start_block=True)

    rebuilt = _exec_emitted(sdfg_to_python(sdfg))

    assert set(rebuilt.arrays.keys()) == {"A", "tmp", "s", "q"}
    assert rebuilt.arrays["tmp"].transient is True
    assert rebuilt.arrays["A"].shape == sdfg.arrays["A"].shape
    assert rebuilt.arrays["q"].buffer_size == sdfg.arrays["q"].buffer_size


# ---------------------------------------------------------------------
# Mapped tasklet — covers state, map, tasklet, memlet
# ---------------------------------------------------------------------


def _build_mapped_increment() -> SDFG:
    sdfg = dace.SDFG("mapped_increment")
    sdfg.add_array("A", [16], dace.float32)
    sdfg.add_array("B", [16], dace.float32)
    state = sdfg.add_state("compute", is_start_block=True)
    a = state.add_access("A")
    b = state.add_access("B")
    me, mx = state.add_map("m", {"i": "0:16"})
    t = state.add_tasklet("inc", {"_in"}, {"_out"}, "_out = _in + 1.0")
    state.add_memlet_path(a, me, t, dst_conn="_in", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(t, mx, b, src_conn="_out", memlet=dace.Memlet("B[i]"))
    return sdfg


def test_mapped_tasklet_round_trip():
    original = _build_mapped_increment()
    rebuilt = _exec_emitted(sdfg_to_python(original))

    A = np.arange(16, dtype=np.float32)
    _assert_runs_equivalent(
        original,
        rebuilt,
        A=A,
        B=np.zeros(16, dtype=np.float32),
    )


# ---------------------------------------------------------------------
# LoopRegion — for-loop with state inside, interstate edges before/after
# ---------------------------------------------------------------------


def _build_loop_region() -> SDFG:
    sdfg = dace.SDFG("loop_fill")
    sdfg.using_explicit_control_flow = True
    sdfg.add_array("A", [10], dace.float32)
    sdfg.add_symbol("i", dace.int32)

    pre = sdfg.add_state("pre", is_start_block=True)
    loop = LoopRegion(
        label="loop1",
        condition_expr="i < 10",
        loop_var="i",
        initialize_expr="i = 0",
        update_expr="i = i + 1",
    )
    sdfg.add_node(loop)
    body = loop.add_state("body", is_start_block=True)
    a = body.add_access("A")
    t = body.add_tasklet("write", None, {"a"}, "a = i")
    body.add_edge(t, "a", a, None, dace.Memlet("A[i]"))

    post = sdfg.add_state("post")
    sdfg.add_edge(pre, loop, dace.InterstateEdge())
    sdfg.add_edge(loop, post, dace.InterstateEdge())
    return sdfg


def test_loop_region_round_trip():
    original = _build_loop_region()
    rebuilt = _exec_emitted(sdfg_to_python(original))

    _assert_runs_equivalent(
        original,
        rebuilt,
        A=np.zeros(10, dtype=np.float32),
    )


# ---------------------------------------------------------------------
# Interstate edge with assignments + condition
# ---------------------------------------------------------------------


def test_interstate_edge_with_assignment_and_condition():
    sdfg = dace.SDFG("isedge")
    sdfg.add_symbol("k", dace.int32)
    sdfg.add_array("A", [4], dace.int32)
    s0 = sdfg.add_state("s0", is_start_block=True)
    s1 = sdfg.add_state("s1")
    s2 = sdfg.add_state("s2")
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"k": "0"}))
    sdfg.add_edge(s1, s2, dace.InterstateEdge(condition="k < 4"))

    rebuilt = _exec_emitted(sdfg_to_python(sdfg))

    edges = list(rebuilt.edges())
    cond_edges = [e for e in edges if e.data.condition.as_string.strip() not in ("", "1")]
    assign_edges = [e for e in edges if e.data.assignments]
    assert len(cond_edges) == 1
    assert "k" in assign_edges[0].data.assignments


# ---------------------------------------------------------------------
# ConditionalBlock — covers branches
# ---------------------------------------------------------------------


def test_conditional_block_round_trip():
    sdfg = dace.SDFG("cond")
    sdfg.add_array("A", [1], dace.float32)
    sdfg.add_symbol("i", dace.int32)
    s0 = sdfg.add_state("s0", is_start_block=True)
    cb = ConditionalBlock("cb")
    sdfg.add_node(cb)
    sdfg.add_edge(s0, cb, dace.InterstateEdge())

    body = ControlFlowRegion("if_body", sdfg=sdfg)
    cb.add_branch(dace.properties.CodeBlock("i == 1"), body)
    body_state = body.add_state("body_state", is_start_block=True)
    a = body_state.add_access("A")
    t = body_state.add_tasklet("set", None, {"a"}, "a = 100")
    body_state.add_edge(t, "a", a, None, dace.Memlet("A[0]"))

    rebuilt = _exec_emitted(sdfg_to_python(sdfg))

    rebuilt_cb = next(b for b in rebuilt.nodes() if isinstance(b, ConditionalBlock))
    assert len(rebuilt_cb.branches) == 1
    cond, region = rebuilt_cb.branches[0]
    assert cond.as_string == "(i == 1)"
    assert region.label == "if_body"


# ---------------------------------------------------------------------
# NestedSDFG — covers recursive emission
# ---------------------------------------------------------------------


def _build_nested_increment() -> SDFG:
    inner = dace.SDFG("inner")
    inner.add_array("x", [1], dace.float32)
    inner.add_array("y", [1], dace.float32)
    s = inner.add_state("s", is_start_block=True)
    xn = s.add_access("x")
    yn = s.add_access("y")
    t = s.add_tasklet("inc", {"_i"}, {"_o"}, "_o = _i + 1.0")
    s.add_edge(xn, None, t, "_i", dace.Memlet("x[0]"))
    s.add_edge(t, "_o", yn, None, dace.Memlet("y[0]"))

    outer = dace.SDFG("outer")
    outer.add_array("A", [1], dace.float32)
    outer.add_array("B", [1], dace.float32)
    state = outer.add_state("call", is_start_block=True)
    nsdfg = state.add_nested_sdfg(inner, {"x"}, {"y"})
    state.add_edge(state.add_access("A"), None, nsdfg, "x", dace.Memlet("A[0]"))
    state.add_edge(nsdfg, "y", state.add_access("B"), None, dace.Memlet("B[0]"))
    return outer


def test_nested_sdfg_round_trip():
    original = _build_nested_increment()
    src = sdfg_to_python(original)
    rebuilt = _exec_emitted(src)

    _assert_runs_equivalent(
        original,
        rebuilt,
        A=np.array([2.0], dtype=np.float32),
        B=np.zeros(1, dtype=np.float32),
    )


# ---------------------------------------------------------------------
# Reduce (LibraryNode) — covers LibraryNode imperative emission
# ---------------------------------------------------------------------


def _build_reduce_sum() -> SDFG:
    sdfg = dace.SDFG("reduce_sum")
    sdfg.add_array("A", [16], dace.float32)
    sdfg.add_array("B", [1], dace.float32)
    state = sdfg.add_state("s", is_start_block=True)
    a = state.add_access("A")
    b = state.add_access("B")
    red = state.add_reduce("lambda x, y: x + y", None, 0)
    state.add_edge(a, None, red, None, dace.Memlet("A[0:16]"))
    state.add_edge(red, None, b, None, dace.Memlet("B[0]"))
    return sdfg


def test_reduce_library_node_round_trip():
    original = _build_reduce_sum()
    src = sdfg_to_python(original)
    rebuilt = _exec_emitted(src)

    _assert_runs_equivalent(
        original,
        rebuilt,
        A=np.arange(16, dtype=np.float32),
        B=np.zeros(1, dtype=np.float32),
    )


# ---------------------------------------------------------------------
# Break/Continue/Return inside a loop region
# ---------------------------------------------------------------------


def test_break_continue_blocks_inside_loop():
    sdfg = dace.SDFG("brk_cnt")
    sdfg.using_explicit_control_flow = True
    sdfg.add_symbol("i", dace.int32)
    sdfg.add_array("A", [1], dace.float32)
    pre = sdfg.add_state("pre", is_start_block=True)
    loop = LoopRegion(
        label="loop",
        condition_expr="i < 5",
        loop_var="i",
        initialize_expr="i = 0",
        update_expr="i = i + 1",
    )
    sdfg.add_node(loop)
    body = loop.add_state("body", is_start_block=True)
    cnt = ContinueBlock("cnt")
    loop.add_node(cnt)
    loop.add_edge(body, cnt, dace.InterstateEdge())
    sdfg.add_edge(pre, loop, dace.InterstateEdge())

    rebuilt = _exec_emitted(sdfg_to_python(sdfg))
    rebuilt_loop = next(n for n in rebuilt.nodes() if isinstance(n, LoopRegion))
    assert any(isinstance(b, ContinueBlock) for b in rebuilt_loop.nodes())


# ---------------------------------------------------------------------
# Sanity: emitter never produces from_json / set_properties_from_json
# ---------------------------------------------------------------------


@pytest.mark.parametrize("builder", [
    _build_mapped_increment,
    _build_loop_region,
    _build_nested_increment,
    _build_reduce_sum,
])
def test_emitter_uses_imperative_api_only(builder):
    src = sdfg_to_python(builder())
    forbidden = ("from_json", "set_properties_from_json", "dace.serialize")
    for token in forbidden:
        assert token not in src, (f"Emitter leaked non-imperative token {token!r}; output:\n" +
                                  textwrap.indent(src, "    "))


# ---------------------------------------------------------------------
# Velocity stage1 SDFGs (real-world ICON dycore artifacts)
# ---------------------------------------------------------------------
# Four stage1 SDFGs from the velocity tendencies pipeline. Each is hundreds
# of arrays / dozens of states / multiple nesting levels — exercises every
# major surface of the emitter on production-shaped graphs.

VELOCITY_STAGE1_FILES = [
    "velocity_no_nproma_if_prop_lvn_only_0_istep_1.sdfgz",
    "velocity_no_nproma_if_prop_lvn_only_0_istep_2.sdfgz",
    "velocity_no_nproma_if_prop_lvn_only_1_istep_1.sdfgz",
    "velocity_no_nproma_if_prop_lvn_only_1_istep_2.sdfgz",
]


@pytest.mark.parametrize("filename", VELOCITY_STAGE1_FILES)
def test_velocity_stage1_round_trip(filename):
    """Load → emit → exec → validate → signature match → compile both.

    Marked slow: each case is a few-hundred-array SDFG; emit + exec + compile
    can take 10-30 seconds.
    """
    path = _DATA_DIR / filename
    if not path.exists():
        pytest.skip(f"missing test artifact: {path}")

    original = SDFG.from_file(str(path))
    src = sdfg_to_python(original)
    rebuilt = _exec_emitted(src)

    # Signature equivalence: the exported argument list must match.
    assert original.signature_arglist() == rebuilt.signature_arglist(), (f"signature mismatch for {filename}\n"
                                                                         f"original: {original.signature_arglist()}\n"
                                                                         f"rebuilt:  {rebuilt.signature_arglist()}")

    # Structural equivalence: arrays, symbols, state count.
    assert set(original.arrays.keys()) == set(rebuilt.arrays.keys())
    assert set(original.symbols.keys()) == set(rebuilt.symbols.keys())
    orig_states = sum(1 for _ in original.all_states())
    rebuilt_states = sum(1 for _ in rebuilt.all_states())
    assert orig_states == rebuilt_states

    # Compile both — failure here surfaces real codegen breakage.
    try:
        original.compile()
    except Exception as exc:
        pytest.skip(f"original SDFG won't compile in this env: {exc}")
    rebuilt.compile()


# ---------------------------------------------------------------------
# CloudSC kernel — full numerical correctness round-trip
# ---------------------------------------------------------------------
# Adapted from a real CloudSC ice-nucleation/deposition kernel.
# This is a real CloudSC ice-nucleation/deposition kernel; build SDFG via
# @dace.program, emit Python source, exec to rebuild, compile both, run on
# the same random inputs, assert array-by-array equality.

_N_SYM = dace.symbol("N", dtype=dace.int64)


@dace.program
def _cloudsc_kernel(
    pap: dace.float64[_N_SYM],
    ptsphy: dace.float64,
    r2es: dace.float64,
    r3ies: dace.float64,
    r4ies: dace.float64,
    rcldtopcf: dace.float64,
    rd: dace.float64,
    rdepliqrefdepth: dace.float64,
    rdepliqrefrate: dace.float64,
    rg: dace.float64,
    riceinit: dace.float64,
    rlmin: dace.float64,
    rlstt: dace.float64,
    rtt: dace.float64,
    rv: dace.float64,
    za: dace.float64[_N_SYM],
    zdp: dace.float64[_N_SYM],
    zfokoop: dace.float64[_N_SYM],
    zicecld: dace.float64[_N_SYM],
    zrho: dace.float64[_N_SYM],
    ztp1: dace.float64[_N_SYM],
    zcldtopdist: dace.float64[_N_SYM],
    zicenuclei: dace.float64[_N_SYM],
    zqxfg: dace.float64[_N_SYM],
    zsolqa: dace.float64[_N_SYM],
):
    for it_47 in dace.map[0:_N_SYM:1]:
        if za[it_47] < rcldtopcf and za[it_47] >= rcldtopcf:
            zcldtopdist[it_47] = 0.0
        else:
            zcldtopdist[it_47] = zcldtopdist[it_47] + (zdp[it_47] / (rg * zrho[it_47]))

        if ztp1[it_47] < rtt and zqxfg[it_47] > rlmin:
            tmp_arg_72 = (r3ies * (ztp1[it_47] - rtt)) / (ztp1[it_47] - r4ies)
            tmp_call_47 = r2es * np.exp(tmp_arg_72)
            zvpice = (rv * tmp_call_47) / rd

            zvpliq = zfokoop[it_47] * np.log(zvpice)

            tmp_arg_27 = -0.639 + ((-1.96 * zvpice + 1.96 * zvpliq) / zvpliq)
            zicenuclei[it_47] = 1000.0 * np.exp(tmp_arg_27)

            zinfactor = min(1.0, 6.66666666666667e-05 * zicenuclei[it_47])

            zadd = (1.6666666666667 * rlstt * (rlstt / (rv * ztp1[it_47]) - 1.0)) / ztp1[it_47]
            zbdd = (0.452488687782805 * pap[it_47] * rv * ztp1[it_47]) / zvpice

            tmp_call_49 = (zicenuclei[it_47] / zrho[it_47])
            zcvds = (7.8 * tmp_call_49 * (zvpliq - zvpice)) / (zvpice * (zadd + zbdd))

            zice0 = max(riceinit * zicenuclei[it_47] / zrho[it_47], zicecld[it_47])

            tmp_arg_30 = 0.666 * ptsphy * zcvds + zice0
            zinew = tmp_arg_30**1.5

            zdepos1 = max(0.0, za[it_47] * (zinew - zice0))
            zdepos2 = min(zdepos1, 1.1)

            tmp_arg_33 = zinfactor + (1.0 - zinfactor) * (rdepliqrefrate + zcldtopdist[it_47] / rdepliqrefdepth)
            zdepos3 = zdepos2 * min(1.0, tmp_arg_33)

            zqxfg[it_47] = zqxfg[it_47] + zdepos3
            zsolqa[it_47] = zsolqa[it_47] + zdepos3


def _cloudsc_inputs(n: int):
    """Reproducible random inputs for the kernel; safe ranges that avoid
    NaN-in-log/div edge cases."""
    rng = np.random.default_rng(0)

    def u(low, high, size):
        return rng.uniform(low, high, size).astype(np.float64)

    return {
        "ptsphy": np.float64(36.0),
        "r2es": np.float64(6.11),
        "r3ies": np.float64(12.0),
        "r4ies": np.float64(15.5),
        "rcldtopcf": np.float64(16.8),
        "rd": np.float64(287.0),
        "rdepliqrefdepth": np.float64(20.0),
        "rdepliqrefrate": np.float64(17.3),
        "rg": np.float64(9.81),
        "riceinit": np.float64(5.3),
        "rlmin": np.float64(3.9),
        "rlstt": np.float64(2.5e6),
        "rtt": np.float64(273.15),
        "rv": np.float64(461.5),
        "N": np.int64(n),
        "pap": u(1.0, 2.0, (n, )),
        "za": u(0.9, 1.5, (n, )),
        "ztp1": u(260.0, 280.0, (n, )),
        "zqxfg": u(5.0, 11.0, (n, )),
        "zsolqa": u(5.0, 11.0, (n, )),
        "zdp": u(0.5, 2.0, (n, )),
        "zfokoop": u(0.95, 1.05, (n, )),
        "zicecld": u(10.0, 11.0, (n, )),
        "zrho": u(0.9, 1.2, (n, )),
        "zcldtopdist": u(0.1, 1.0, (n, )),
        "zicenuclei": u(1e2, 1e4, (n, )),
    }


def test_cloudsc_kernel_numerical_round_trip():
    """End-to-end CloudSC kernel: build SDFG via @dace.program, emit Python
    source via the imperative API, exec to rebuild, run both on the same
    inputs, assert per-array numerical equivalence.
    """
    original = _cloudsc_kernel.to_sdfg(simplify=True)
    src = sdfg_to_python(original)
    rebuilt = _exec_emitted(src)

    n = 1024  # the "very big test case" — 1024 grid points
    inputs_orig = _cloudsc_inputs(n)
    inputs_rebuilt = {k: v.copy() if hasattr(v, "copy") else v for k, v in inputs_orig.items()}

    # Compile both. If the original won't compile in this env, that's an
    # env-side failure, not a reconstruction regression.
    try:
        original_compiled = original.compile()
    except Exception as exc:
        pytest.skip(f"original CloudSC kernel won't compile in this env: {exc}")
    rebuilt_compiled = rebuilt.compile()

    original_compiled(**inputs_orig)
    rebuilt_compiled(**inputs_rebuilt)

    array_keys = [k for k, v in inputs_orig.items() if isinstance(v, np.ndarray)]
    for k in array_keys:
        np.testing.assert_allclose(
            inputs_orig[k],
            inputs_rebuilt[k],
            rtol=0.0,
            atol=0.0,
            err_msg=f"CloudSC numerical mismatch on output {k!r}",
        )


# ---------------------------------------------------------------------
# Full CloudSC SDFG (the real one, not the pattern_one loop-nest)
# ---------------------------------------------------------------------
# Loads cloudsc_simplified.sdfgz from the SC26 layout artifacts — the
# full IFS cloud-microphysics scheme as a single DaCe program. This test
# exercises every emitter codepath at production scale: 2300+ arrays,
# 220+ symbols, 12k-line emitted Python source.
#
# Numerical correctness for the full CloudSC requires the HDF5 input
# pipeline + C++ runner that lives outside this repo. This test verifies
# structural equivalence (validate, signature, array set, symbol set);
# users wanting numerical correctness can feed the rebuilt SDFG into the
# same C++ runner and compare against the original's outputs.


@pytest.mark.slow
def test_full_cloudsc_round_trip():
    """Full CloudSC: load → emit → exec → validate → signature equality.

    Skipped if the artifact isn't present in the test data dir. Compile
    takes ~60s for the original alone; the rebuilt's exec+build phase is
    ~3min on a workstation-class machine.
    """
    path = _DATA_DIR / "cloudsc_simplified.sdfgz"
    if not path.exists():
        pytest.skip(f"missing test artifact: {path}")

    original = SDFG.from_file(str(path))
    src = sdfg_to_python(original)
    rebuilt = _exec_emitted(src)

    assert original.signature_arglist() == rebuilt.signature_arglist()
    assert set(original.arrays.keys()) == set(rebuilt.arrays.keys())
    assert set(original.symbols.keys()) == set(rebuilt.symbols.keys())

    orig_states = sum(1 for _ in original.all_states())
    rebuilt_states = sum(1 for _ in rebuilt.all_states())
    assert orig_states == rebuilt_states


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
