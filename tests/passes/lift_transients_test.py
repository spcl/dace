# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``LiftTransients`` / ``lift_transients``.

Each test:
 1. builds an SDFG with a transient array inside a nested SDFG;
 2. declares the expected result as a pure numpy computation;
 3. runs a fresh copy of the *baseline* SDFG (before the pass) and
    asserts it matches the numpy reference -- this validates the
    input SDFG and the reference simultaneously;
 4. applies :class:`LiftTransients` to the original SDFG;
 5. checks structural post-conditions (shape, layout, transient
    flags, connector plumbing);
 6. runs the transformed SDFG and asserts it still matches the
    numpy reference.

Both packed-Fortran (column-major, default) and packed-C (row-major)
layouts are exercised. Not every test contains a *computation* in
numpy -- many of the simpler cases fill with constants and check
uniformity; the richer cases (``2*A``, row reduction) exercise real
numpy reference computations.
"""
import copy

import numpy as np
import pytest

import dace
from dace import dtypes, memlet as mm, nodes
from dace.sdfg import SDFG
from dace.transformation.passes.lift_transients import LiftTransients, lift_transients


def _force_sequential_maps(sdfg: dace.SDFG):
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.MapEntry):
            n.map.schedule = dtypes.ScheduleType.Sequential


def _fortran_strides(shape):
    s = [1]
    for d in shape[:-1]:
        s.append(s[-1] * d)
    return s


def _c_strides(shape):
    s = [1]
    for d in reversed(shape[1:]):
        s.insert(0, s[0] * d)
    return s


def _run_check(sdfg: dace.SDFG, expected: np.ndarray, *,
               out_kw: str = "B", order: str = 'F', call_kwargs=None):
    _force_sequential_maps(sdfg)
    out = np.zeros(expected.shape, dtype=expected.dtype, order=order)
    kwargs = dict(call_kwargs or {})
    kwargs[out_kw] = out
    sdfg(**kwargs)
    assert np.allclose(out, expected), (
        f"SDFG output disagreed with numpy reference: got {out} vs {expected}")


def _inner_1d(size: int, name: str = "t") -> SDFG:
    inner = SDFG(f"inner_1d_{name}")
    inner.add_array(name, [size], dace.float64, transient=True)
    inner.add_array("B", [size], dace.float64, transient=False)
    st = inner.add_state()
    me, mx = st.add_map("fill", {"k": f"0:{size}"})
    task = st.add_tasklet("set", {}, {"o"}, "o = 1.0")
    t_acc = st.add_access(name)
    b_acc = st.add_write("B")
    st.add_edge(me, None, task, None, mm.Memlet())
    mx.add_in_connector("IN_" + name)
    mx.add_out_connector("OUT_" + name)
    st.add_edge(task, "o", mx, "IN_" + name, mm.Memlet(data=name, subset="k"))
    st.add_edge(mx, "OUT_" + name, t_acc, None,
                mm.Memlet.from_array(name, inner.arrays[name]))
    st.add_edge(t_acc, None, b_acc, None,
                mm.Memlet.from_array("B", inner.arrays["B"]))
    return inner


def _inner_2d(M: int, N: int, layout: str) -> SDFG:
    strides = _fortran_strides((M, N)) if layout == "fortran" else _c_strides((M, N))
    inner = SDFG(f"inner_2d_{layout}")
    inner.add_array("t", [M, N], dace.float64, transient=True, strides=strides)
    inner.add_array("B", [M, N], dace.float64, transient=False, strides=strides)
    st = inner.add_state()
    me, mx = st.add_map("fill", {"i": f"0:{M}", "j": f"0:{N}"})
    task = st.add_tasklet("set", {}, {"o"}, "o = 100 * i + j")
    t_acc = st.add_access("t")
    b_acc = st.add_write("B")
    st.add_edge(me, None, task, None, mm.Memlet())
    mx.add_in_connector("IN_t")
    mx.add_out_connector("OUT_t")
    st.add_edge(task, "o", mx, "IN_t", mm.Memlet(data="t", subset="i, j"))
    st.add_edge(mx, "OUT_t", t_acc, None,
                mm.Memlet.from_array("t", inner.arrays["t"]))
    st.add_edge(t_acc, None, b_acc, None,
                mm.Memlet.from_array("B", inner.arrays["B"]))
    return inner


def _inner_2d_with_input(M: int, N: int, layout: str) -> SDFG:
    strides = _fortran_strides((M, N)) if layout == "fortran" else _c_strides((M, N))
    inner = SDFG(f"inner_2d_input_{layout}")
    inner.add_array("A", [M, N], dace.float64, transient=False, strides=strides)
    inner.add_array("t", [M, N], dace.float64, transient=True, strides=strides)
    inner.add_array("B", [M, N], dace.float64, transient=False, strides=strides)
    st = inner.add_state()
    a_r = st.add_read("A")
    t_w = st.add_access("t")
    b_w = st.add_write("B")
    me, mx = st.add_map("scale", {"i": f"0:{M}", "j": f"0:{N}"})
    task = st.add_tasklet("mul2", {"inp"}, {"o"}, "o = 2.0 * inp")
    me.add_in_connector("IN_A")
    me.add_out_connector("OUT_A")
    mx.add_in_connector("IN_t")
    mx.add_out_connector("OUT_t")
    st.add_edge(a_r, None, me, "IN_A", mm.Memlet.from_array("A", inner.arrays["A"]))
    st.add_edge(me, "OUT_A", task, "inp", mm.Memlet(data="A", subset="i, j"))
    st.add_edge(task, "o", mx, "IN_t", mm.Memlet(data="t", subset="i, j"))
    st.add_edge(mx, "OUT_t", t_w, None, mm.Memlet.from_array("t", inner.arrays["t"]))
    st.add_edge(t_w, None, b_w, None, mm.Memlet.from_array("B", inner.arrays["B"]))
    return inner


def _inner_row_reduce(M: int, N: int, layout: str) -> SDFG:
    strides = _fortran_strides((M, N)) if layout == "fortran" else _c_strides((M, N))
    inner = SDFG(f"inner_row_reduce_{layout}")
    inner.add_array("A", [M, N], dace.float64, transient=False, strides=strides)
    inner.add_array("t", [M, N], dace.float64, transient=True, strides=strides)
    inner.add_array("S", [M], dace.float64, transient=False)

    # State 1: t[i, j] = A[i, j].
    st1 = inner.add_state()
    a_r = st1.add_read("A")
    t_w = st1.add_access("t")
    me, mx = st1.add_map("copy", {"i": f"0:{M}", "j": f"0:{N}"})
    task = st1.add_tasklet("id", {"inp"}, {"o"}, "o = inp")
    me.add_in_connector("IN_A")
    me.add_out_connector("OUT_A")
    mx.add_in_connector("IN_t")
    mx.add_out_connector("OUT_t")
    st1.add_edge(a_r, None, me, "IN_A", mm.Memlet.from_array("A", inner.arrays["A"]))
    st1.add_edge(me, "OUT_A", task, "inp", mm.Memlet(data="A", subset="i, j"))
    st1.add_edge(task, "o", mx, "IN_t", mm.Memlet(data="t", subset="i, j"))
    st1.add_edge(mx, "OUT_t", t_w, None, mm.Memlet.from_array("t", inner.arrays["t"]))

    # State 2: zero S[:] so the WCR accumulator starts from 0.
    st_init = inner.add_state_after(st1, label="init_S")
    s_init = st_init.add_write("S")
    me_z, mx_z = st_init.add_map("zero_S", {"i": f"0:{M}"})
    z_task = st_init.add_tasklet("zero", {}, {"o"}, "o = 0.0")
    mx_z.add_in_connector("IN_S")
    mx_z.add_out_connector("OUT_S")
    st_init.add_edge(me_z, None, z_task, None, mm.Memlet())
    st_init.add_edge(z_task, "o", mx_z, "IN_S", mm.Memlet(data="S", subset="i"))
    st_init.add_edge(mx_z, "OUT_S", s_init, None,
                     mm.Memlet.from_array("S", inner.arrays["S"]))

    # State 3: row reduction via map + WCR on S. DaCe handles the
    # strided access correctly for both Fortran and C layouts.
    st2 = inner.add_state_after(st_init, label="reduce_rows")
    t_r = st2.add_read("t")
    s_w = st2.add_write("S")
    me2, mx2 = st2.add_map("reduce", {"i": f"0:{M}", "j": f"0:{N}"})
    rtask = st2.add_tasklet("sum", {"inp"}, {"out"}, "out = inp")
    me2.add_in_connector("IN_t")
    me2.add_out_connector("OUT_t")
    mx2.add_in_connector("IN_S")
    mx2.add_out_connector("OUT_S")
    st2.add_edge(t_r, None, me2, "IN_t", mm.Memlet.from_array("t", inner.arrays["t"]))
    st2.add_edge(me2, "OUT_t", rtask, "inp", mm.Memlet(data="t", subset="i, j"))
    st2.add_edge(rtask, "out", mx2, "IN_S",
                 mm.Memlet(data="S", subset="i", wcr="lambda a, b: a + b"))
    st2.add_edge(mx2, "OUT_S", s_w, None, mm.Memlet.from_array("S", inner.arrays["S"]))
    return inner


def test_lift_out_of_plain_nested_sdfg():
    inner = _inner_1d(8)
    top = SDFG("top_plain")
    top.add_array("B", [8], dace.float64, transient=False)
    st = top.add_state()
    n = st.add_nested_sdfg(inner, {}, {"B"})
    bw = st.add_write("B")
    st.add_edge(n, "B", bw, None, mm.Memlet.from_array("B", top.arrays["B"]))

    expected = np.ones(8, dtype=np.float64)
    _run_check(copy.deepcopy(top), expected, order='C')

    lifted = lift_transients(top)
    top.validate()
    assert lifted == 1
    assert "t" in top.arrays
    assert top.arrays["t"].transient is True
    assert tuple(top.arrays["t"].shape) == (8, )
    assert inner.arrays["t"].transient is False
    assert "t" in n.out_connectors

    _run_check(top, expected, order='C')


def test_lift_1d_through_map_fortran_default_appends():
    size, K = 8, 4
    inner = _inner_1d(size)
    top = SDFG("top_1d_map_fortran")
    top.add_array("B", [size, K], dace.float64,
                  transient=False, strides=_fortran_strides((size, K)))
    st = top.add_state()
    me, mx = st.add_map("outer", {"jb": f"0:{K}"})
    n = st.add_nested_sdfg(inner, {}, {"B"})
    bw = st.add_write("B")
    mx.add_in_connector("IN_B")
    mx.add_out_connector("OUT_B")
    st.add_edge(me, None, n, None, mm.Memlet())
    st.add_edge(n, "B", mx, "IN_B", mm.Memlet(data="B", subset=f"0:{size}, jb"))
    st.add_edge(mx, "OUT_B", bw, None, mm.Memlet.from_array("B", top.arrays["B"]))

    expected = np.ones((size, K), dtype=np.float64)
    _run_check(copy.deepcopy(top), expected, order='F')

    lifted = lift_transients(top)
    top.validate()
    assert lifted == 1
    assert tuple(top.arrays["t"].shape) == (size, K)
    assert top.arrays["t"].is_packed_fortran_strides()

    _run_check(top, expected, order='F')


def test_lift_through_map_with_nonzero_begin_uses_param_minus_begin():
    size, K = 8, 4
    inner = _inner_1d(size)
    top = SDFG("top_1d_map_offset")
    top.add_symbol("B_start", dace.int32)
    top.add_array("B", [size, K], dace.float64,
                  transient=False, strides=_fortran_strides((size, K)))
    st = top.add_state()
    me, mx = st.add_map("outer", {"jb": f"B_start : B_start + {K}"})
    n = st.add_nested_sdfg(inner, {}, {"B"})
    bw = st.add_write("B")
    mx.add_in_connector("IN_B")
    mx.add_out_connector("OUT_B")
    st.add_edge(me, None, n, None, mm.Memlet())
    st.add_edge(n, "B", mx, "IN_B",
                mm.Memlet(data="B", subset=f"0:{size}, jb - B_start"))
    st.add_edge(mx, "OUT_B", bw, None, mm.Memlet.from_array("B", top.arrays["B"]))

    expected = np.ones((size, K), dtype=np.float64)
    _run_check(copy.deepcopy(top), expected, order='F',
               call_kwargs={"B_start": 2})

    lifted = lift_transients(top)
    top.validate()
    assert lifted == 1
    st = top.start_state
    nsdfg = next(x for x in st.nodes() if isinstance(x, nodes.NestedSDFG))
    expected_idx = dace.symbolic.pystr_to_symbolic("jb - B_start")
    for e in st.in_edges(nsdfg):
        if e.dst_conn != "t":
            continue
        begin, end, _ = list(e.data.subset.ndrange())[-1]
        assert dace.symbolic.simplify(begin - expected_idx) == 0
        assert dace.symbolic.simplify(end - expected_idx) == 0
        break
    else:
        raise AssertionError("no incoming memlet on 't' connector found")

    _run_check(top, expected, order='F', call_kwargs={"B_start": 2})


def test_lift_with_suggestion_overrides_shape_and_indexes_from_begin():
    size = 8
    inner = _inner_1d(size)
    top = SDFG("top_suggestion")
    top.add_symbol("i_startblk", dace.int32)
    top.add_symbol("i_endblk", dace.int32)
    top.add_symbol("nblks_c", dace.int32)
    top.add_array("B", [size, "nblks_c"], dace.float64,
                  transient=False,
                  strides=_fortran_strides((size, dace.symbol("nblks_c"))))
    st = top.add_state()
    me, mx = st.add_map("outer", {"jb": "i_startblk : i_endblk"})
    n = st.add_nested_sdfg(inner, {}, {"B"})
    bw = st.add_write("B")
    mx.add_in_connector("IN_B")
    mx.add_out_connector("OUT_B")
    st.add_edge(me, None, n, None, mm.Memlet())
    st.add_edge(n, "B", mx, "IN_B",
                mm.Memlet(data="B", subset=f"0:{size}, jb - i_startblk"))
    st.add_edge(mx, "OUT_B", bw, None, mm.Memlet.from_array("B", top.arrays["B"]))

    nblks = 6
    expected = np.zeros((size, nblks), dtype=np.float64)
    expected[:, 0:3] = 1.0

    _run_check(copy.deepcopy(top), expected, order='F',
               call_kwargs={"i_startblk": 1, "i_endblk": 4, "nblks_c": nblks})

    lifted = lift_transients(top, map_range_suggestions={
        ("i_startblk", "i_endblk"): "nblks_c",
    })
    top.validate()
    assert lifted == 1
    assert str(top.arrays["t"].shape[-1]) == "nblks_c"
    assert tuple(top.arrays["t"].shape)[:-1] == (size, )

    st = top.start_state
    nsdfg = next(x for x in st.nodes() if isinstance(x, nodes.NestedSDFG))
    expected_idx = dace.symbolic.pystr_to_symbolic("jb - i_startblk")
    for e in st.in_edges(nsdfg):
        if e.dst_conn != "t":
            continue
        begin, end, _ = list(e.data.subset.ndrange())[-1]
        assert dace.symbolic.simplify(begin - expected_idx) == 0
        assert dace.symbolic.simplify(end - expected_idx) == 0
        break
    else:
        raise AssertionError("no incoming memlet on 't' connector found")

    _run_check(top, expected, order='F',
               call_kwargs={"i_startblk": 1, "i_endblk": 4, "nblks_c": nblks})


def test_lift_2d_fortran_through_map():
    M, N, K = 3, 4, 5
    inner = _inner_2d(M, N, "fortran")
    top = SDFG("top_2d_fortran")
    top.add_array("B", [M, N, K], dace.float64,
                  transient=False, strides=_fortran_strides((M, N, K)))
    st = top.add_state()
    me, mx = st.add_map("outer", {"jb": f"0:{K}"})
    n = st.add_nested_sdfg(inner, {}, {"B"})
    bw = st.add_write("B")
    mx.add_in_connector("IN_B")
    mx.add_out_connector("OUT_B")
    st.add_edge(me, None, n, None, mm.Memlet())
    st.add_edge(n, "B", mx, "IN_B",
                mm.Memlet(data="B", subset=f"0:{M}, 0:{N}, jb"))
    st.add_edge(mx, "OUT_B", bw, None, mm.Memlet.from_array("B", top.arrays["B"]))

    expected = np.fromfunction(
        lambda i, j, k: 100 * i + j, (M, N, K), dtype=np.float64)

    _run_check(copy.deepcopy(top), expected, order='F')

    lifted = lift_transients(top)
    top.validate()
    assert lifted == 1
    assert tuple(top.arrays["t"].shape) == (M, N, K)
    assert top.arrays["t"].is_packed_fortran_strides()

    _run_check(top, expected, order='F')


def test_lift_2d_c_through_map():
    M, N, K = 3, 4, 5
    inner = _inner_2d(M, N, "c")
    top = SDFG("top_2d_c")
    top.add_array("B", [K, M, N], dace.float64,
                  transient=False, strides=_c_strides((K, M, N)))
    st = top.add_state()
    me, mx = st.add_map("outer", {"jb": f"0:{K}"})
    n = st.add_nested_sdfg(inner, {}, {"B"})
    bw = st.add_write("B")
    mx.add_in_connector("IN_B")
    mx.add_out_connector("OUT_B")
    st.add_edge(me, None, n, None, mm.Memlet())
    st.add_edge(n, "B", mx, "IN_B",
                mm.Memlet(data="B", subset=f"jb, 0:{M}, 0:{N}"))
    st.add_edge(mx, "OUT_B", bw, None, mm.Memlet.from_array("B", top.arrays["B"]))

    expected = np.fromfunction(
        lambda k, i, j: 100 * i + j, (K, M, N), dtype=np.float64)

    _run_check(copy.deepcopy(top), expected, order='C')

    lifted = lift_transients(top)
    top.validate()
    assert lifted == 1
    assert tuple(top.arrays["t"].shape) == (K, M, N)
    assert top.arrays["t"].is_packed_c_strides()

    _run_check(top, expected, order='C')


def test_lift_2d_fortran_with_input_through_map():
    M, N, K = 3, 4, 5
    inner = _inner_2d_with_input(M, N, "fortran")
    top = SDFG("top_2d_input_fortran")
    top.add_array("A", [M, N, K], dace.float64,
                  transient=False, strides=_fortran_strides((M, N, K)))
    top.add_array("B", [M, N, K], dace.float64,
                  transient=False, strides=_fortran_strides((M, N, K)))
    st = top.add_state()
    me, mx = st.add_map("outer", {"jb": f"0:{K}"})
    n = st.add_nested_sdfg(inner, {"A"}, {"B"})
    ar = st.add_read("A")
    bw = st.add_write("B")
    me.add_in_connector("IN_A")
    me.add_out_connector("OUT_A")
    mx.add_in_connector("IN_B")
    mx.add_out_connector("OUT_B")
    st.add_edge(ar, None, me, "IN_A", mm.Memlet.from_array("A", top.arrays["A"]))
    st.add_edge(me, "OUT_A", n, "A", mm.Memlet(data="A", subset=f"0:{M}, 0:{N}, jb"))
    st.add_edge(n, "B", mx, "IN_B", mm.Memlet(data="B", subset=f"0:{M}, 0:{N}, jb"))
    st.add_edge(mx, "OUT_B", bw, None, mm.Memlet.from_array("B", top.arrays["B"]))

    a = np.arange(M * N * K, dtype=np.float64).reshape((M, N, K)).copy(order='F')
    expected = 2.0 * a

    _run_check(copy.deepcopy(top), expected, order='F', call_kwargs={"A": a})

    lifted = lift_transients(top)
    top.validate()
    assert lifted == 1
    assert tuple(top.arrays["t"].shape) == (M, N, K)
    assert top.arrays["t"].is_packed_fortran_strides()

    _run_check(top, expected, order='F', call_kwargs={"A": a})


def test_lift_2d_c_with_input_through_map():
    M, N, K = 3, 4, 5
    inner = _inner_2d_with_input(M, N, "c")
    top = SDFG("top_2d_input_c")
    top.add_array("A", [K, M, N], dace.float64,
                  transient=False, strides=_c_strides((K, M, N)))
    top.add_array("B", [K, M, N], dace.float64,
                  transient=False, strides=_c_strides((K, M, N)))
    st = top.add_state()
    me, mx = st.add_map("outer", {"jb": f"0:{K}"})
    n = st.add_nested_sdfg(inner, {"A"}, {"B"})
    ar = st.add_read("A")
    bw = st.add_write("B")
    me.add_in_connector("IN_A")
    me.add_out_connector("OUT_A")
    mx.add_in_connector("IN_B")
    mx.add_out_connector("OUT_B")
    st.add_edge(ar, None, me, "IN_A", mm.Memlet.from_array("A", top.arrays["A"]))
    st.add_edge(me, "OUT_A", n, "A", mm.Memlet(data="A", subset=f"jb, 0:{M}, 0:{N}"))
    st.add_edge(n, "B", mx, "IN_B", mm.Memlet(data="B", subset=f"jb, 0:{M}, 0:{N}"))
    st.add_edge(mx, "OUT_B", bw, None, mm.Memlet.from_array("B", top.arrays["B"]))

    a = np.arange(K * M * N, dtype=np.float64).reshape((K, M, N)).copy(order='C')
    expected = 2.0 * a

    _run_check(copy.deepcopy(top), expected, order='C', call_kwargs={"A": a})

    lifted = lift_transients(top)
    top.validate()
    assert lifted == 1
    assert tuple(top.arrays["t"].shape) == (K, M, N)
    assert top.arrays["t"].is_packed_c_strides()

    _run_check(top, expected, order='C', call_kwargs={"A": a})


def test_lift_2d_with_row_reduction_fortran():
    M, N, K = 3, 4, 5
    inner = _inner_row_reduce(M, N, "fortran")
    top = SDFG("top_row_reduce")
    top.add_array("A", [M, N, K], dace.float64,
                  transient=False, strides=_fortran_strides((M, N, K)))
    top.add_array("B", [M, K], dace.float64,
                  transient=False, strides=_fortran_strides((M, K)))
    st = top.add_state()
    me, mx = st.add_map("outer", {"jb": f"0:{K}"})
    n = st.add_nested_sdfg(inner, {"A"}, {"S"})
    ar = st.add_read("A")
    bw = st.add_write("B")
    me.add_in_connector("IN_A")
    me.add_out_connector("OUT_A")
    mx.add_in_connector("IN_B")
    mx.add_out_connector("OUT_B")
    st.add_edge(ar, None, me, "IN_A", mm.Memlet.from_array("A", top.arrays["A"]))
    st.add_edge(me, "OUT_A", n, "A", mm.Memlet(data="A", subset=f"0:{M}, 0:{N}, jb"))
    st.add_edge(n, "S", mx, "IN_B", mm.Memlet(data="B", subset=f"0:{M}, jb"))
    st.add_edge(mx, "OUT_B", bw, None, mm.Memlet.from_array("B", top.arrays["B"]))

    a = np.arange(M * N * K, dtype=np.float64).reshape((M, N, K)).copy(order='F')
    expected = a.sum(axis=1).copy(order='F')

    _run_check(copy.deepcopy(top), expected, order='F', call_kwargs={"A": a})

    lifted = lift_transients(top)
    top.validate()
    assert lifted == 1
    assert tuple(top.arrays["t"].shape) == (M, N, K)
    assert top.arrays["t"].is_packed_fortran_strides()

    _run_check(top, expected, order='F', call_kwargs={"A": a})


def test_lift_through_two_levels_of_nested_sdfg_with_map():
    size, K_outer, K_inner = 6, 3, 4
    deep = _inner_1d(size)

    mid = SDFG("mid_two_level")
    mid.add_array("B", [size, K_inner], dace.float64,
                  transient=False, strides=_fortran_strides((size, K_inner)))
    mst = mid.add_state()
    me_m, mx_m = mst.add_map("mid_map", {"ji": f"0:{K_inner}"})
    dn = mst.add_nested_sdfg(deep, {}, {"B"})
    bm = mst.add_write("B")
    mx_m.add_in_connector("IN_B")
    mx_m.add_out_connector("OUT_B")
    mst.add_edge(me_m, None, dn, None, mm.Memlet())
    mst.add_edge(dn, "B", mx_m, "IN_B", mm.Memlet(data="B", subset=f"0:{size}, ji"))
    mst.add_edge(mx_m, "OUT_B", bm, None, mm.Memlet.from_array("B", mid.arrays["B"]))

    top = SDFG("top_two_level_map")
    top.add_array("B", [size, K_inner, K_outer], dace.float64,
                  transient=False, strides=_fortran_strides((size, K_inner, K_outer)))
    st = top.add_state()
    me_t, mx_t = st.add_map("top_map", {"jo": f"0:{K_outer}"})
    mn = st.add_nested_sdfg(mid, {}, {"B"})
    bw = st.add_write("B")
    mx_t.add_in_connector("IN_B")
    mx_t.add_out_connector("OUT_B")
    st.add_edge(me_t, None, mn, None, mm.Memlet())
    st.add_edge(mn, "B", mx_t, "IN_B",
                mm.Memlet(data="B", subset=f"0:{size}, 0:{K_inner}, jo"))
    st.add_edge(mx_t, "OUT_B", bw, None, mm.Memlet.from_array("B", top.arrays["B"]))

    expected = np.ones((size, K_inner, K_outer), dtype=np.float64)

    _run_check(copy.deepcopy(top), expected, order='F')

    lifted = lift_transients(top)
    top.validate()
    assert lifted >= 2
    assert "t" in top.arrays
    assert top.arrays["t"].transient is True
    assert tuple(top.arrays["t"].shape) == (size, K_inner, K_outer)
    assert top.arrays["t"].is_packed_fortran_strides()
    assert deep.arrays["t"].transient is False
    assert mid.arrays["t"].transient is False

    _run_check(top, expected, order='F')


def test_scalar_transient_left_alone():
    inner = SDFG("inner_scalar")
    inner.add_array("s", [1], dace.float64, transient=True)
    inner.add_array("B", [1], dace.float64, transient=False)
    st = inner.add_state()
    task = st.add_tasklet("w", {}, {"o"}, "o = 3.14")
    s_acc = st.add_access("s")
    b_acc = st.add_write("B")
    st.add_edge(task, "o", s_acc, None, mm.Memlet(data="s", subset="0"))
    st.add_edge(s_acc, None, b_acc, None, mm.Memlet(data="B", subset="0"))

    top = SDFG("top_scalar")
    top.add_array("B", [1], dace.float64, transient=False)
    tst = top.add_state()
    n = tst.add_nested_sdfg(inner, {}, {"B"})
    bw = tst.add_write("B")
    tst.add_edge(n, "B", bw, None, mm.Memlet.from_array("B", top.arrays["B"]))

    expected = np.array([3.14], dtype=np.float64)
    _run_check(copy.deepcopy(top), expected, order='C')

    lifted = lift_transients(top)
    top.validate()
    assert lifted == 0
    assert "s" not in top.arrays
    assert inner.arrays["s"].transient is True

    _run_check(top, expected, order='C')


def test_non_transient_left_alone():
    inner = SDFG("inner_pass_through")
    inner.add_array("u", [8], dace.float64, transient=False)
    inner.add_array("B", [8], dace.float64, transient=False)
    st = inner.add_state()
    u_acc = st.add_read("u")
    b_acc = st.add_write("B")
    st.add_edge(u_acc, None, b_acc, None,
                mm.Memlet.from_array("B", inner.arrays["B"]))

    top = SDFG("top_pass_through")
    top.add_array("u", [8], dace.float64, transient=False)
    top.add_array("B", [8], dace.float64, transient=False)
    tst = top.add_state()
    n = tst.add_nested_sdfg(inner, {"u"}, {"B"})
    ur = tst.add_read("u")
    bw = tst.add_write("B")
    tst.add_edge(ur, None, n, "u", mm.Memlet.from_array("u", top.arrays["u"]))
    tst.add_edge(n, "B", bw, None, mm.Memlet.from_array("B", top.arrays["B"]))

    u = np.arange(8, dtype=np.float64)
    expected = u.copy()
    _run_check(copy.deepcopy(top), expected, order='C', call_kwargs={"u": u})

    lifted = lift_transients(top)
    top.validate()
    assert lifted == 0

    _run_check(top, expected, order='C', call_kwargs={"u": u})


def test_idempotent():
    inner = _inner_1d(8)
    top = SDFG("top_idempotent")
    top.add_array("B", [8], dace.float64, transient=False)
    st = top.add_state()
    n = st.add_nested_sdfg(inner, {}, {"B"})
    bw = st.add_write("B")
    st.add_edge(n, "B", bw, None, mm.Memlet.from_array("B", top.arrays["B"]))

    expected = np.ones(8, dtype=np.float64)
    _run_check(copy.deepcopy(top), expected, order='C')

    first = lift_transients(top)
    second = lift_transients(top)
    top.validate()
    assert first >= 1
    assert second == 0

    _run_check(top, expected, order='C')


def test_two_level_nesting_both_lift():
    deep = _inner_1d(8)

    mid = SDFG("mid_plain")
    mid.add_array("B", [8], dace.float64, transient=False)
    mst = mid.add_state()
    mn = mst.add_nested_sdfg(deep, {}, {"B"})
    mbw = mst.add_write("B")
    mst.add_edge(mn, "B", mbw, None, mm.Memlet.from_array("B", mid.arrays["B"]))

    top = SDFG("top_two_level_plain")
    top.add_array("B", [8], dace.float64, transient=False)
    tst = top.add_state()
    tn = tst.add_nested_sdfg(mid, {}, {"B"})
    tbw = tst.add_write("B")
    tst.add_edge(tn, "B", tbw, None, mm.Memlet.from_array("B", top.arrays["B"]))

    expected = np.ones(8, dtype=np.float64)
    _run_check(copy.deepcopy(top), expected, order='C')

    lifted = lift_transients(top)
    top.validate()
    assert lifted >= 2
    assert "t" in top.arrays
    assert top.arrays["t"].transient is True
    assert deep.arrays["t"].transient is False
    assert mid.arrays["t"].transient is False

    _run_check(top, expected, order='C')


def test_pass_class_apply_pass_returns_count():
    """Smoke check for the DaCe-style ``Pass`` entrypoint."""
    inner = _inner_1d(8)
    top = SDFG("top_pass_class")
    top.add_array("B", [8], dace.float64, transient=False)
    st = top.add_state()
    n = st.add_nested_sdfg(inner, {}, {"B"})
    bw = st.add_write("B")
    st.add_edge(n, "B", bw, None, mm.Memlet.from_array("B", top.arrays["B"]))

    result = LiftTransients().apply_pass(top, {})
    top.validate()
    assert result == 1


def test_post_conditions_raise_on_wrong_lifetime():
    inner = _inner_1d(8)
    top = SDFG("top_bad_lifetime")
    top.add_array("B", [8], dace.float64, transient=False)
    st = top.add_state()
    n = st.add_nested_sdfg(inner, {}, {"B"})
    bw = st.add_write("B")
    st.add_edge(n, "B", bw, None, mm.Memlet.from_array("B", top.arrays["B"]))

    lift_transients(top)
    top.arrays["t"].lifetime = dtypes.AllocationLifetime.Scope
    from dace.transformation.passes.lift_transients import _verify_postconditions
    with pytest.raises(ValueError, match="lifetime"):
        _verify_postconditions(top)


def test_post_conditions_raise_on_internally_assigned_shape_symbol():
    top = SDFG("top_internal_sym")
    top.add_symbol("N", dace.int32)
    entry = top.add_state("entry")
    mid = top.add_state("mid")
    top.add_edge(entry, mid, dace.InterstateEdge(assignments={"N": "5"}))
    top.add_array("bad", ["N"], dace.float64, transient=True,
                  lifetime=dtypes.AllocationLifetime.SDFG)
    from dace.transformation.passes.lift_transients import _verify_postconditions
    with pytest.raises(ValueError, match="internally-assigned"):
        _verify_postconditions(top)


if __name__ == "__main__":
    test_lift_out_of_plain_nested_sdfg()
    test_lift_1d_through_map_fortran_default_appends()
    test_lift_through_map_with_nonzero_begin_uses_param_minus_begin()
    test_lift_with_suggestion_overrides_shape_and_indexes_from_begin()
    test_lift_2d_fortran_through_map()
    test_lift_2d_c_through_map()
    test_lift_2d_fortran_with_input_through_map()
    test_lift_2d_c_with_input_through_map()
    test_lift_2d_with_row_reduction_fortran()
    test_lift_through_two_levels_of_nested_sdfg_with_map()
    test_scalar_transient_left_alone()
    test_non_transient_left_alone()
    test_idempotent()
    test_two_level_nesting_both_lift()
    test_pass_class_apply_pass_returns_count()
    test_post_conditions_raise_on_wrong_lifetime()
    test_post_conditions_raise_on_internally_assigned_shape_symbol()
    print("all LiftTransients tests passed")
