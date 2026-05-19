# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared imports, symbols, and reusable SDFG fixtures for the topical
vectorization test files in this directory.

All `_get_*_sdfg` fixtures and the `run_vectorization_test` harness live here
so each topical `test_*.py` file imports them rather than redefining.
"""

import dace

import copy

import numpy

from dace import InterstateEdge

from dace import Union

from dace.properties import CodeBlock

from dace.sdfg import ControlFlowRegion

from dace.sdfg.state import ConditionalBlock

from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU

N = dace.symbol('N')
S1 = dace.symbol("S1")
S2 = dace.symbol("S2")
S = dace.symbol("S")
klev = dace.symbol("klev")
kidia = dace.symbol("kidia")
kfdia = dace.symbol("kfdia")
n = dace.symbol('n')  # number of rows
m = dace.symbol('m')  # number of columns
nnz = dace.symbol('nnz')  # number of nonzeros
C = 32
ssym = dace.symbolic.symbol("ssym")
Y = dace.symbolic.symbol("Y")
X = dace.symbolic.symbol("X")
KLON = dace.symbol('KLON')
KLEV = dace.symbol('KLEV')
NCLDQL = dace.symbol('NCLDQL')
NCLDQI = dace.symbol('NCLDQI')


def run_vectorization_test(dace_func: Union[dace.SDFG, callable],
                           arrays,
                           params,
                           vector_width=8,
                           simplify=True,
                           skip_simplify=None,
                           sdfg_name=None,
                           fuse_overlapping_loads=False,
                           insert_copies=True,
                           filter_map=-1,
                           cleanup=False,
                           from_sdfg=False,
                           no_inline=False,
                           exact=None,
                           branch_mode: str = "merge",
                           remainder_strategy: str = "scalar",
                           param_tag: str = None,
                           lower_to_intrinsics: bool = False,
                           collapse_laneid_index_loads: bool = False,
                           loop_to_map_permissive: bool = False):

    # K1=fp_factor + K2=masked is rejected by VectorizeCPU per the locked
    # plan decision (the masked path emits merge tasklets / iter_mask blends
    # that fp-factor lowering can't combine cleanly). Skip rather than
    # propagate a hard error through every (fp_factor, masked, *) parametrize.
    import pytest as _pytest
    if branch_mode == "fp_factor" and remainder_strategy == "masked":
        _pytest.skip("fp_factor is incompatible with masked remainder (locked plan rule)")

    # Create copies for comparison
    arrays_orig = {k: copy.deepcopy(v) for k, v in arrays.items()}
    arrays_vec = {k: copy.deepcopy(v) for k, v in arrays.items()}

    # Suffix the sdfg name with the parametrization keys so each combination
    # of (branch_mode, remainder_strategy) gets its own ``.dacecache/<name>/``
    # build directory. Without this, parallel pytest workers building the
    # same kernel under different parametrizations race on shared cmake state
    # and the loser worker crashes with a stale ``.so`` load error.
    #
    # ``param_tag`` is the convention for tests that ALSO parametrise over an
    # ``opt_parameters`` (or similar) tuple beyond the conftest fixtures.
    # Pass ``param_tag=f"param{idx}"`` from the test body so each tuple gets
    # a unique cache dir. The verbose alternative is to encode the tuple
    # into the sdfg.name directly, but parametrize indices keep names short.
    if sdfg_name is not None:
        sdfg_name = f"{sdfg_name}_{branch_mode}_{remainder_strategy}"
        if param_tag is not None:
            sdfg_name = f"{sdfg_name}_{param_tag}"

    # Original SDFG
    if not from_sdfg:
        sdfg: dace.SDFG = dace_func.to_sdfg(simplify=False)
        sdfg.name = sdfg_name
        if simplify:
            sdfg.simplify(validate=True, validate_all=True, skip=skip_simplify or set())
    else:
        sdfg: dace.SDFG = dace_func
        # When the caller pre-built the SDFG (from_sdfg=True) and supplied
        # a base sdfg_name, also rename the reference SDFG itself so the
        # unvectorized compile lands in its own .dacecache directory. Without
        # this, every (branch_mode, remainder_strategy, opt_parameters)
        # parametrization shares the same reference build dir and parallel
        # pytest workers race on CMake state — surfaces as FileExistsError /
        # CompilerConfigurationError / OSError loading the .so.
        if sdfg_name is not None:
            sdfg.name = sdfg_name

    c_sdfg = sdfg.compile()

    # Vectorized SDFG
    copy_sdfg: dace.SDFG = copy.deepcopy(sdfg)
    copy_sdfg.name = copy_sdfg.name + "_vectorized"

    if cleanup:
        for e, g in copy_sdfg.all_edges_recursive():
            if isinstance(g, dace.SDFGState):
                if (isinstance(e.src, dace.nodes.AccessNode) and isinstance(e.dst, dace.nodes.AccessNode)
                        and isinstance(g.sdfg.arrays[e.dst.data], dace.data.Scalar)
                        and e.data.other_subset is not None):
                    # Add assignment taskelt
                    src_data = e.src.data
                    src_subset = e.data.subset if e.data.data == src_data else e.data.other_subset
                    dst_data = e.dst.data
                    dst_subset = e.data.subset if e.data.data == dst_data else e.data.other_subset
                    g.remove_edge(e)
                    t = g.add_tasklet(name=f"assign_dst_{dst_data}_from_{src_data}",
                                      code="_out = _in",
                                      inputs={"_in"},
                                      outputs={"_out"})
                    g.add_edge(e.src, e.src_conn, t, "_in",
                               dace.memlet.Memlet(data=src_data, subset=copy.deepcopy(src_subset)))
                    g.add_edge(t, "_out", e.dst, e.dst_conn,
                               dace.memlet.Memlet(data=dst_data, subset=copy.deepcopy(dst_subset)))
        copy_sdfg.validate()

    if filter_map != -1:
        map_labels = [n.map.label for (n, g) in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]
        filter_map_labels = map_labels[0:filter_map]
        filter_map = filter_map_labels
    else:
        filter_map = None

    if branch_mode == "fp_factor":
        branch_kwargs = dict(use_fp_factor=True, branch_normalization=False)
    elif branch_mode == "merge":
        branch_kwargs = dict(use_fp_factor=False, branch_normalization=True)
    else:
        raise ValueError(f"branch_mode must be 'fp_factor' or 'merge', got {branch_mode!r}")

    VectorizeCPU(vector_width=vector_width,
                 fuse_overlapping_loads=fuse_overlapping_loads,
                 insert_copies=insert_copies,
                 apply_on_maps=filter_map,
                 no_inline=no_inline,
                 fail_on_unvectorizable=True,
                 remainder_strategy=remainder_strategy,
                 lower_to_intrinsics=lower_to_intrinsics,
                 collapse_laneid_index_loads=collapse_laneid_index_loads,
                 loop_to_map_permissive=loop_to_map_permissive,
                 **branch_kwargs).apply_pass(copy_sdfg, {})
    copy_sdfg.validate()

    c_copy_sdfg = copy_sdfg.compile()

    # Run both
    c_sdfg(**arrays_orig, **params)

    c_copy_sdfg(**arrays_vec, **params)

    # Compare results
    for name in arrays.keys():
        assert numpy.allclose(arrays_orig[name], arrays_vec[name], rtol=1e-32), \
            f"{name} Diff: {arrays_orig[name] - arrays_vec[name]}"
        if exact is not None:
            diff = arrays_vec[name] - exact
            assert numpy.allclose(arrays_vec[name], exact, rtol=0, atol=1e-300), \
                f"{name} Diff: max abs diff = {numpy.max(numpy.abs(diff))}"
    return copy_sdfg


def assert_fused_nsdfg_structure(sdfg: dace.SDFG, bases):
    """Verify the baked ``fuse_overlapping_loads`` collapsed the staging buffers.

    Two invariants are checked per fused array base name:

    1. **Collapse**: no legacy per-subset ``<base>_vec_0 .. <base>_vec_n``
       buffer survives anywhere — the knob must have replaced the N
       independent ``vector_copy`` staging buffers with a single shared
       union-window buffer ``<base>_vec``.
    2. **Wiring**: at least one body NSDFG holds a ``<base>_vec`` access
       node that is *both* produced (the union staging copy-in writes it)
       *and* consumed (the inner map body reads it). This is the genuine
       fused-read buffer; it proves the union copy is connected to the map
       body and did not orphan it.

    The shared name ``<base>_vec`` is also used for unrelated *movable*
    boundary buffers (e.g. a written output) which are produced-only inside
    their NSDFG — those are not fusion products, so the wiring check is
    satisfied by *any* NSDFG (existential), not all. Graph well-formedness
    and numerical correctness are already covered by ``sdfg.validate()`` and
    the e2e compare inside :func:`run_vectorization_test`; this adds the
    fusion-specific structural guarantee on top so a test cannot silently
    pass e2e while the fusion is structurally broken.

    :param sdfg: The vectorized SDFG returned by :func:`run_vectorization_test`.
    :param bases: Array base names expected to be load-fused (e.g. ``("A",)``).
    """
    for base in bases:
        prefix = f"{base}_vec_"
        fully_wired = False
        for nsdfg in (n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)):
            inner = nsdfg.sdfg
            indexed = [a for a in inner.arrays if a.startswith(prefix) and a[len(prefix):].isdigit()]
            assert not indexed, (f"fuse_overlapping_loads on, but per-subset buffers {indexed} survived for "
                                 f"{base!r} in NSDFG {inner.label}: fusion did not collapse them")
            shared = f"{base}_vec"
            if shared not in inner.arrays:
                continue
            produced = any(st.in_degree(an) >= 1 for st in inner.all_states() for an in st.data_nodes()
                           if an.data == shared)
            consumed = any(st.out_degree(an) >= 1 for st in inner.all_states() for an in st.data_nodes()
                           if an.data == shared)
            if produced and consumed:
                fully_wired = True
        assert fully_wired, (f"no NSDFG holds a fully-wired shared fused buffer {base}_vec (produced by the union "
                             f"staging copy-in and consumed by the map body) — fusion did not connect the buffer")


def _get_disjoint_chain_sdfg(trivial_if: bool, fortran_layout: bool = False) -> dace.SDFG:
    sd1 = dace.SDFG("disjoint_chain")
    cb1 = ConditionalBlock("cond_if_cond_58", sdfg=sd1, parent=sd1)
    ss1 = sd1.add_state(label="pre", is_start_block=True)
    sd1.add_node(cb1, is_start_block=False)

    cfg1 = dace.ControlFlowRegion(label="cond_58_true", sdfg=sd1, parent=cb1)
    s1 = cfg1.add_state("main_1", is_start_block=True)
    cfg2 = dace.ControlFlowRegion(label="cond_58_false", sdfg=sd1, parent=cb1)
    s2 = cfg2.add_state("main_2", is_start_block=True)

    cb1.add_branch(
        condition=CodeBlock("_if_cond_58 == 1" if not trivial_if else "1 == 1"),
        branch=cfg1,
    )
    cb1.add_branch(
        condition=None,
        branch=cfg2,
    )
    for arr_name, shape in [
        ("zsolqa", (5, 5, C)),
        ("zrainaut", (C, )),
        ("zrainacc", (C, )),
        ("ztp1", (C, )),
    ]:
        sd1.add_array(arr_name, shape, dace.float64)
    sd1.add_scalar("rtt", dace.float64)
    sd1.add_symbol("_if_cond_58", dace.float64)
    sd1.add_symbol("_for_it_52", dace.int64)
    sd1.add_edge(src=ss1, dst=cb1, data=dace.InterstateEdge(assignments={
        "_if_cond_58": "ztp1[_for_it_52] <= rtt",
    }, ))

    for state, d1_access_str, zsolqa_access_str, zsolqa_access_str_rev in [
        (s1, "_for_it_52", "0,3,_for_it_52", "3,0,_for_it_52"), (s2, "_for_it_52", "0,2,_for_it_52", "2,0,_for_it_52")
    ]:
        zrainaut = state.add_access("zrainaut")
        zrainacc = state.add_access("zrainacc")
        zsolqa1 = state.add_access("zsolqa")
        zsolqa2 = state.add_access("zsolqa")
        zsolqa3 = state.add_access("zsolqa")
        zsolqa4 = state.add_access("zsolqa")
        zsolqa5 = state.add_access("zsolqa")
        for i, (tasklet_code, in1, instr1, in2, instr2, out, outstr) in enumerate([
            ("_out = _in1 + _in2", zrainaut, d1_access_str, zsolqa1, zsolqa_access_str, zsolqa2, zsolqa_access_str),
            ("_out = _in1 + _in2", zrainacc, d1_access_str, zsolqa2, zsolqa_access_str, zsolqa3, zsolqa_access_str),
            ("_out = (-_in1) + _in2", zrainaut, d1_access_str, zsolqa3, zsolqa_access_str_rev, zsolqa4,
             zsolqa_access_str_rev),
            ("_out = (-_in1) + _in2", zrainacc, d1_access_str, zsolqa4, zsolqa_access_str_rev, zsolqa5,
             zsolqa_access_str_rev),
        ]):
            t1 = state.add_tasklet("t1", {"_in1", "_in2"}, {"_out"}, tasklet_code)
            state.add_edge(in1, None, t1, "_in1", dace.memlet.Memlet(f"{in1.data}[{instr1}]"))
            state.add_edge(in2, None, t1, "_in2", dace.memlet.Memlet(f"{in2.data}[{instr2}]"))
            state.add_edge(t1, "_out", out, None, dace.memlet.Memlet(f"{out.data}[{outstr}]"))

    sd1.validate()

    sd2 = dace.SDFG("disjoin_chain_sdfg")
    p_s1 = sd2.add_state("p_s1", is_start_block=True)

    map_entry, map_exit = p_s1.add_map(name="map1", ndrange={"_for_it_52": dace.subsets.Range([(0, C - 1, 1)])})
    nsdfg = p_s1.add_nested_sdfg(sdfg=sd1,
                                 inputs={"zsolqa", "ztp1", "zrainaut", "zrainacc", "rtt"},
                                 outputs={"zsolqa"},
                                 symbol_mapping={"_for_it_52": "_for_it_52"})
    for arr_name, shape in [("zsolqa", (5, 5, C)), ("zrainaut", (C, )), ("zrainacc", (C, )), ("ztp1", (C, ))]:
        sd2.add_array(arr_name, shape, dace.float64)
    sd2.add_scalar("rtt", dace.float64)
    for input_name in {"zsolqa", "ztp1", "zrainaut", "zrainacc", "rtt"}:
        a = p_s1.add_access(input_name)
        p_s1.add_edge(a, None, map_entry, f"IN_{input_name}",
                      dace.memlet.Memlet.from_array(input_name, sd2.arrays[input_name]))
        p_s1.add_edge(map_entry, f"OUT_{input_name}", nsdfg, input_name,
                      dace.memlet.Memlet.from_array(input_name, sd2.arrays[input_name]))
        map_entry.add_in_connector(f"IN_{input_name}")
        map_entry.add_out_connector(f"OUT_{input_name}")
    for output_name in {"zsolqa"}:
        a = p_s1.add_access(output_name)
        p_s1.add_edge(map_exit, f"OUT_{output_name}", a, None,
                      dace.memlet.Memlet.from_array(output_name, sd2.arrays[output_name]))
        p_s1.add_edge(nsdfg, output_name, map_exit, f"IN_{output_name}",
                      dace.memlet.Memlet.from_array(output_name, sd2.arrays[output_name]))
        map_exit.add_in_connector(f"IN_{output_name}")
        map_exit.add_out_connector(f"OUT_{output_name}")

    nsdfg.sdfg.parent_nsdfg_node = nsdfg

    sd1.validate()
    sd2.validate()
    return sd2, p_s1


def _get_disjoint_chain_sdfg_two() -> dace.SDFG:
    sd1 = dace.SDFG("disjoint_chain_two")
    cb1 = ConditionalBlock("cond_if_cond_58", sdfg=sd1, parent=sd1)
    ss1 = sd1.add_state(label="pre", is_start_block=True)
    sd1.add_node(cb1, is_start_block=False)
    sd1.add_symbol("N", dace.int64)

    cfg1 = ControlFlowRegion(label="cond_58_true", sdfg=sd1, parent=cb1)
    s1 = cfg1.add_state("main_1", is_start_block=True)
    cfg2 = ControlFlowRegion(label="cond_58_false", sdfg=sd1, parent=cb1)
    s2 = cfg2.add_state("main_2", is_start_block=True)

    cb1.add_branch(
        condition=CodeBlock("_if_cond_58 == 1"),
        branch=cfg1,
    )
    cb1.add_branch(
        condition=None,
        branch=cfg2,
    )
    for arr_name, shape in [
        ("zsolqa", (5, 5, N)),
        ("zrainaut", (N, )),
        ("zrainacc", (N, )),
        ("ztp1", (N, )),
    ]:
        sd1.add_array(arr_name, shape, dace.float64)
    sd1.add_scalar("rtt", dace.float64)
    sd1.add_symbol("_if_cond_58", dace.float64)
    sd1.add_symbol("_for_it_52", dace.int64)
    sd1.add_edge(src=ss1, dst=cb1, data=InterstateEdge(assignments={
        "_if_cond_58": "ztp1[_for_it_52] <= rtt",
    }, ))

    for state, d1_access_str, zsolqa_access_str, zsolqa_access_str_rev in [
        (s1, "_for_it_52", "3,0,_for_it_52", "0,3,_for_it_52"), (s2, "_for_it_52", "2,0,_for_it_52", "0,2,_for_it_52")
    ]:
        zrainaut = state.add_access("zrainaut")
        zrainacc = state.add_access("zrainacc")
        zsolqa1 = state.add_access("zsolqa")
        zsolqa2 = state.add_access("zsolqa")
        zsolqa3 = state.add_access("zsolqa")
        zsolqa4 = state.add_access("zsolqa")
        zsolqa5 = state.add_access("zsolqa")
        for i, (tasklet_code, in1, instr1, in2, instr2, out, outstr) in enumerate([
            ("_out = _in1 + _in2", zrainaut, d1_access_str, zsolqa1, zsolqa_access_str, zsolqa2, zsolqa_access_str),
            ("_out = _in1 + _in2", zrainacc, d1_access_str, zsolqa2, zsolqa_access_str, zsolqa3, zsolqa_access_str),
            ("_out = (-_in1) + _in2", zrainaut, d1_access_str, zsolqa3, zsolqa_access_str_rev, zsolqa4,
             zsolqa_access_str_rev),
            ("_out = (-_in1) + _in2", zrainacc, d1_access_str, zsolqa4, zsolqa_access_str_rev, zsolqa5,
             zsolqa_access_str_rev),
        ]):
            t1 = state.add_tasklet("t1", {"_in1", "_in2"}, {"_out"}, tasklet_code)
            state.add_edge(in1, None, t1, "_in1", dace.memlet.Memlet(f"{in1.data}[{instr1}]"))
            state.add_edge(in2, None, t1, "_in2", dace.memlet.Memlet(f"{in2.data}[{instr2}]"))
            state.add_edge(t1, "_out", out, None, dace.memlet.Memlet(f"{out.data}[{outstr}]"))

    sd1_s2 = sd1.add_state_after(cb1, label="extra")

    z1 = sd1_s2.add_access("zrainacc")
    z2 = sd1_s2.add_access("zrainacc")
    t1 = sd1_s2.add_tasklet("increment", {"_in"}, {"_out"}, "_out = _in + 1")
    sd1_s2.add_edge(z1, None, t1, "_in", dace.memlet.Memlet("zrainacc[_for_it_52]"))
    sd1_s2.add_edge(t1, "_out", z2, None, dace.memlet.Memlet("zrainacc[_for_it_52]"))
    sd1.validate()

    sd2 = dace.SDFG("sd2")
    sd2.add_symbol("N", dace.int64)
    p_s1 = sd2.add_state("p_s1", is_start_block=True)

    map_entry, map_exit = p_s1.add_map(name="map1", ndrange={"_for_it_52": dace.subsets.Range([(0, N - 1, 1)])})
    nsdfg = p_s1.add_nested_sdfg(sdfg=sd1,
                                 inputs={"zsolqa", "ztp1", "zrainaut", "zrainacc", "rtt"},
                                 outputs={"zsolqa", "zrainacc"},
                                 symbol_mapping={"_for_it_52": "_for_it_52"})
    for arr_name, shape in [("zsolqa", (5, 5, N)), ("zrainaut", (N, )), ("zrainacc", (N, )), ("ztp1", (N, ))]:
        sd2.add_array(arr_name, shape, dace.float64)
    sd2.add_scalar("rtt", dace.float64)

    for input_name in {"zsolqa", "ztp1", "zrainaut", "zrainacc", "rtt"}:
        a = p_s1.add_access(input_name)
        p_s1.add_edge(a, None, map_entry, f"IN_{input_name}",
                      dace.memlet.Memlet.from_array(input_name, sd2.arrays[input_name]))
        p_s1.add_edge(map_entry, f"OUT_{input_name}", nsdfg, input_name,
                      dace.memlet.Memlet.from_array(input_name, sd2.arrays[input_name]))
        map_entry.add_in_connector(f"IN_{input_name}")
        map_entry.add_out_connector(f"OUT_{input_name}")
    for output_name in {"zsolqa", "zrainacc"}:
        a = p_s1.add_access(output_name)
        p_s1.add_edge(map_exit, f"OUT_{output_name}", a, None,
                      dace.memlet.Memlet.from_array(output_name, sd2.arrays[output_name]))
        p_s1.add_edge(nsdfg, output_name, map_exit, f"IN_{output_name}",
                      dace.memlet.Memlet.from_array(output_name, sd2.arrays[output_name]))
        map_exit.add_in_connector(f"IN_{output_name}")
        map_exit.add_out_connector(f"OUT_{output_name}")

    nsdfg.sdfg.parent_nsdfg_node = nsdfg

    sd1.validate()
    sd2.validate()
    return sd2, p_s1


def _get_cloudsc_snippet_three(add_scalar: bool, map_range_dependent_subset: bool = False):
    klon = dace.symbolic.symbol("klon")
    klev = dace.symbolic.symbol("klev")
    # Add all arrays to the SDFGs
    in_arrays = {"tendency_tmp_q", "pa", "pq", "tendency_tmp_t", "tendency_tmp_a", "pt"}
    in_scalars = {"kfdia", "kidia", "ptsphy"}
    if add_scalar:
        in_scalars = in_scalars.union({"ralvdcp"})
    out_arrays = {"zqx0", "zqx", "ztp1", "zaorig", "za"}
    arr_shapes = {
        "tendency_tmp_q": ((klon, klev), (1, klon), dace.float64),
        "pa": ((klon, klev), (1, klon), dace.float64),
        "pq": ((klon, klev), (1, klon), dace.float64),
        "tendency_tmp_t": ((klon, klev), (1, klon), dace.float64),
        "tendency_tmp_a": ((klon, klev), (1, klon), dace.float64),
        "pt": ((klon, klev), (1, klon), dace.float64),
        "zqx0": ((klon, klev, 5), (1, klon, klon * klev), dace.float64),
        "zqx": ((klon, klev, 5), (1, klon, klon * klev), dace.float64),
        "ztp1": ((klon, klev), (1, klon), dace.float64),
        "zaorig": ((klon, klev), (1, klon), dace.float64),
        "za": ((klon, klev), (1, klon), dace.float64),
    }
    scalar_dtypes = {
        "kfdia": dace.int64,
        "kidia": dace.int64,
        "ptsphy": dace.float64,
    }
    if add_scalar:
        scalar_dtypes["ralvdcp"] = dace.float64
    outer_sdfg = dace.SDFG("outer")
    inner_sdfg = dace.SDFG("inner")
    inner_state = inner_sdfg.add_state("inner_compute_state", is_start_block=True)
    outer_state = outer_sdfg.add_state("outer_compute_state", is_start_block=True)

    symbols = {"i", "j", "klev", "klon"}
    for sym in symbols:
        inner_sdfg.add_symbol(sym, dace.int64)
        outer_sdfg.add_symbol(sym, dace.int64)

    for sdfg in [inner_sdfg, outer_sdfg]:
        for arr_name in in_arrays.union(out_arrays):
            shape, strides, dtype = arr_shapes[arr_name]
            sdfg.add_array(arr_name, shape, dtype, strides=strides, transient=False)
        for scalar_name in in_scalars:
            if sdfg == inner_sdfg and scalar_name in {"kfdia", "kidia"}:
                continue
            sdfg.add_scalar(scalar_name, scalar_dtypes[scalar_name], transient=False)

    for transient_scl in {"t0_s1", "t0_s2", "t0_s3", "t0_s4", "t0_s5"}:
        inner_sdfg.add_scalar(transient_scl, dace.float64, dace.dtypes.StorageType.Register, True)

    # All tasklets for the inner SDFG
    if add_scalar:
        tasklets = {
            ("ralvdcp", "0", None, None, "_out = - _in1", "t0_s1", "0"),
            ("ptsphy", "0", "tendency_tmp_q", "i, j", "_out = _in2 * _in1", "t0_s2", "0"),
            ("ptsphy", "0", "tendency_tmp_q", "i, j", "_out = _in2 * _in1", "t0_s3", "0"),
            ("ptsphy", "0", "tendency_tmp_a", "i, j", "_out = _in2 * _in1", "t0_s4", "0"),
            ("ptsphy", "0", "tendency_tmp_a", "i, j", "_out = _in2 * _in1", "t0_s5", "0"),
            ("t0_s1", "0", "pt", "i, j", "_out = _in1 + _in2", "ztp1", "i, j"),
            ("t0_s2", "0", "pq", "i, j", "_out = _in2 + _in1", "zqx", "i, j, 4"),
            ("t0_s3", "0", "pq", "i, j", "_out = _in2 + _in1", "zqx0", "i, j, 4"),
            ("t0_s4", "0", "pa", "i, j", "_out = _in2 + _in1", "za", "i, j"),
            ("t0_s5", "0", "pa", "i, j", "_out = _in2 + _in1", "zaorig", "i, j"),
        }
    else:
        tasklets = {
            ("ptsphy", "0", "tendency_tmp_t", "i, j", "_out = _in2 * _in1", "t0_s1", "0"),
            ("ptsphy", "0", "tendency_tmp_q", "i, j", "_out = _in2 * _in1", "t0_s2", "0"),
            ("ptsphy", "0", "tendency_tmp_q", "i, j", "_out = _in2 * _in1", "t0_s3", "0"),
            ("ptsphy", "0", "tendency_tmp_a", "i, j", "_out = _in2 * _in1", "t0_s4", "0"),
            ("ptsphy", "0", "tendency_tmp_a", "i, j", "_out = _in2 * _in1", "t0_s5", "0"),
            ("t0_s1", "0", "pt", "i, j", "_out = _in1 + _in2", "ztp1", "i, j"),
            ("t0_s2", "0", "pq", "i, j", "_out = _in2 + _in1", "zqx", "i, j, 4"),
            ("t0_s3", "0", "pq", "i, j", "_out = _in2 + _in1", "zqx0", "i, j, 4"),
            ("t0_s4", "0", "pa", "i, j", "_out = _in2 + _in1", "za", "i, j"),
            ("t0_s5", "0", "pa", "i, j", "_out = _in2 + _in1", "zaorig", "i, j"),
        }
    access_nodes = dict()
    for in1_arr, in1_subset, in2_arr, in2_subset, tasklet_code, out_arr, out_subset in tasklets:
        in1_an = inner_state.add_access(in1_arr) if in1_arr not in access_nodes else access_nodes[in1_arr]
        if in2_arr is not None:
            in2_an = inner_state.add_access(in2_arr) if in2_arr not in access_nodes else access_nodes[in2_arr]
        out_an = inner_state.add_access(out_arr) if out_arr not in access_nodes else access_nodes[out_arr]
        access_nodes[in1_arr] = in1_an
        if in2_arr is not None:
            access_nodes[in2_arr] = in2_an
        access_nodes[out_arr] = out_an

        t = inner_state.add_tasklet("t_" + out_arr, {"_in1", "_in2"} if in2_arr is not None else {"_in1"}, {"_out"},
                                    tasklet_code)
        access_str1 = f"{in1_arr}[{in1_subset}]" if in1_subset != "0" else in1_arr
        if in2_arr is not None:
            access_str2 = f"{in2_arr}[{in2_subset}]" if in2_subset != "0" else in2_arr
        inner_state.add_edge(in1_an, None, t, "_in1", dace.memlet.Memlet(access_str1))
        if in2_arr is not None:
            inner_state.add_edge(in2_an, None, t, "_in2", dace.memlet.Memlet(access_str2))

        access_str3 = f"{out_arr}[{out_subset}]" if out_subset != "0" else out_arr
        inner_state.add_edge(t, "_out", out_an, None, dace.memlet.Memlet(access_str3))

    inner_symbol_mapping = {sym: sym for sym in symbols}
    in_args = in_arrays.union({"ptsphy"})
    if add_scalar:
        in_args.add("ralvdcp")
    nsdfg = outer_state.add_nested_sdfg(inner_sdfg, in_args, out_arrays, inner_symbol_mapping)

    m1_entry, m1_exit = outer_state.add_map(name="m1", ndrange={"j": "0:klev:1"})
    m2_entry, m2_exit = outer_state.add_map(name="m2", ndrange={"i": "kidia-1:kfdia:1"})

    inner_sdfg.validate()

    # Access nodes to map entry
    for arr in in_arrays.union(in_scalars):
        outer_state.add_edge(outer_state.add_access(arr), None, m1_entry, f"IN_{arr}",
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_entry.add_in_connector(f"IN_{arr}")
    # Access nodes to map entry1 to map entry 2 and nsdfg
    for arr in in_arrays.union(in_scalars):
        if map_range_dependent_subset:
            if arr in in_arrays:
                if arr in {"zqx", "zqx0"}:
                    mem = dace.memlet.Memlet(f"{arr}[0:i+1, 0:j+1, 0:5]")
                else:
                    mem = dace.memlet.Memlet(f"{arr}[0:i+1, 0:j+1]")
            else:
                mem = dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr])
        else:
            mem = dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr])

        outer_state.add_edge(m1_entry, f"OUT_{arr}", m2_entry, f"IN_{arr}" if arr not in {"kidia", "kfdia"} else arr,
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_entry.add_out_connector(f"OUT_{arr}", force=True)
        m2_entry.add_in_connector(f"IN_{arr}" if arr not in {"kidia", "kfdia"} else arr, force=True)
        if arr not in {"kidia", "kfdia"}:
            outer_state.add_edge(m2_entry, f"OUT_{arr}", nsdfg, arr, copy.deepcopy(mem))
            m2_entry.add_out_connector(f"OUT_{arr}", force=True)
    # Same for exit nodes
    for arr in out_arrays:
        if map_range_dependent_subset:
            if arr in in_arrays:
                if arr in {"zqx", "zqx0"}:
                    mem = dace.memlet.Memlet(f"{arr}[0:i+1, 0:j+1, 0:5]")
                else:
                    mem = dace.memlet.Memlet(f"{arr}[0:i+1, 0:j+1]")
            else:
                mem = dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr])
        else:
            mem = dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr])
        outer_state.add_edge(nsdfg, arr, m2_exit, f"IN_{arr}", copy.deepcopy(mem))
        m2_exit.add_in_connector(f"IN_{arr}", force=True)
        outer_state.add_edge(m2_exit, f"OUT_{arr}", m1_exit, f"IN_{arr}",
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_exit.add_in_connector(f"IN_{arr}", force=True)
        m2_exit.add_out_connector(f"OUT_{arr}", force=True)

    for arr in out_arrays:
        outer_state.add_edge(m1_exit, f"OUT_{arr}", outer_state.add_access(arr), None,
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_exit.add_out_connector(f"OUT_{arr}", force=True)
    sdfg.validate()
    return sdfg


def _get_cloudsc_snippet_four():
    klon = dace.symbolic.symbol("klon")
    klev = dace.symbolic.symbol("klev")
    kidia = dace.symbolic.symbol("kidia")
    kfdia = dace.symbolic.symbol("kfdia")
    # Add all arrays to the SDFGs
    in_arrays = {"zsolqb", "zfallsink", "zqlhs"}
    out_arrays = {"zqlhs"}
    arr_shapes = {
        "zqlhs": ((klon, 5, 5), (1, klon, 5 * klon), dace.float64),
        "zfallsink": ((
            klon,
            5,
        ), (
            1,
            klon,
        ), dace.float64),
        "zsolqb": ((klon, 5, 5), (1, klon, 5 * klon), dace.float64),
    }
    outer_sdfg = dace.SDFG("outer")
    inner_sdfg = dace.SDFG("inner")
    inner_state = inner_sdfg.add_state("inner_compute_state", is_start_block=True)
    outer_state = outer_sdfg.add_state("outer_compute_state", is_start_block=True)

    symbols = {"_for_it_93", "_for_it_91", "_for_it_92", "for_i", "klev", "klon", "kfdia", "kidia"}
    for sym in symbols:
        inner_sdfg.add_symbol(sym, dace.int64)
        outer_sdfg.add_symbol(sym, dace.int64)

    for sdfg in [inner_sdfg, outer_sdfg]:
        for arr_name in in_arrays.union(out_arrays):
            shape, strides, dtype = arr_shapes[arr_name]
            sdfg.add_array(arr_name, shape, dtype, strides=strides, transient=False)

    for transient_scl in {"t0_s1", "t0_s2", "t0_s3", "t0_s4", "t0_s5"}:
        inner_sdfg.add_scalar(transient_scl, dace.float64, dace.dtypes.StorageType.Register, True)

    tasklets = {
        ("zfallsink", "_for_it_93, _for_it_91", None, None, "_out = _in1 + 1.0", "zqlhs@1",
         "_for_it_93, _for_it_92, _for_it_91"),
        ("zqlhs@1", "_for_it_93, _for_it_92, _for_it_91", "zsolqb", "_for_it_93, 0, _for_it_91", "_out = _in1 + _in2",
         "zqlhs@2", "_for_it_93, _for_it_92, _for_it_91"),
        ("zqlhs@2", "_for_it_93, _for_it_92, _for_it_91", "zsolqb", "_for_it_93, 1, _for_it_91", "_out = _in1 + _in2",
         "zqlhs@3", "_for_it_93, _for_it_92, _for_it_91"),
        ("zqlhs@3", "_for_it_93, _for_it_92, _for_it_91", "zsolqb", "_for_it_93, 2, _for_it_91", "_out = _in1 + _in2",
         "zqlhs@4", "_for_it_93, _for_it_92, _for_it_91"),
        ("zqlhs@4", "_for_it_93, _for_it_92, _for_it_91", "zsolqb", "_for_it_93, 3, _for_it_91", "_out = _in1 + _in2",
         "zqlhs@5", "_for_it_93, _for_it_92, _for_it_91"),
        ("zqlhs@5", "_for_it_93, _for_it_92, _for_it_91", "zsolqb", "_for_it_93, 4, _for_it_91", "_out = _in1 + _in2",
         "zqlhs@6", "_for_it_93, _for_it_92, _for_it_91"),
    }
    zqlhs_ans = list()
    for i in range(6):
        zqlhs_an = inner_state.add_access("zqlhs")
        zqlhs_ans.append(zqlhs_an)

    access_nodes = dict()
    for in1_arr, in1_subset, in2_arr, in2_subset, tasklet_code, out_arr, out_subset in tasklets:
        if in1_arr.startswith("zqlhs@"):
            in1_an = zqlhs_ans[int(in1_arr.split("@")[1]) - 1]
            in1_arr = "zqlhs"
        else:
            in1_an = inner_state.add_access(in1_arr) if in1_arr not in access_nodes else access_nodes[in1_arr]

        if in2_arr is not None:
            if in2_arr.startswith("zqlhs@"):
                in2_an = zqlhs_ans[int(in2_arr.split("@")[1]) - 1]
                in2_arr = "zqlhs"
            else:
                in2_an = inner_state.add_access(in2_arr) if in2_arr not in access_nodes else access_nodes[in2_arr]

        if out_arr.startswith("zqlhs@"):
            out_an = zqlhs_ans[int(out_arr.split("@")[1]) - 1]
            out_arr = "zqlhs"
        else:
            out_an = inner_state.add_access(out_arr) if out_arr not in access_nodes else access_nodes[out_arr]

        access_nodes[in1_arr] = in1_an
        if in2_arr is not None:
            access_nodes[in2_arr] = in2_an
        access_nodes[out_arr] = out_an

        t = inner_state.add_tasklet("t_" + out_arr, {"_in1", "_in2"} if in2_arr is not None else {"_in1"}, {"_out"},
                                    tasklet_code)
        access_str1 = f"{in1_arr}[{in1_subset}]" if in1_subset != "0" else in1_arr
        if in2_arr is not None:
            access_str2 = f"{in2_arr}[{in2_subset}]" if in2_subset != "0" else in2_arr
        inner_state.add_edge(in1_an, None, t, "_in1", dace.memlet.Memlet(access_str1))
        if in2_arr is not None:
            inner_state.add_edge(in2_an, None, t, "_in2", dace.memlet.Memlet(access_str2))

        access_str3 = f"{out_arr}[{out_subset}]" if out_subset != "0" else out_arr
        inner_state.add_edge(t, "_out", out_an, None, dace.memlet.Memlet(access_str3))

    inner_symbol_mapping = {sym: sym for sym in symbols}
    in_args = in_arrays
    nsdfg = outer_state.add_nested_sdfg(inner_sdfg, in_args, out_arrays, inner_symbol_mapping)

    m1_entry, m1_exit = outer_state.add_map(name="m1", ndrange={"_for_it_93": "kidia-1:kfdia:1"})

    inner_sdfg.validate()

    # Access nodes to map entry
    for arr in in_arrays:
        outer_state.add_edge(outer_state.add_access(arr), None, m1_entry, f"IN_{arr}",
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_entry.add_in_connector(f"IN_{arr}")
        mem = dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr])
        m1_entry.add_out_connector(f"OUT_{arr}", force=True)
        outer_state.add_edge(m1_entry, f"OUT_{arr}", nsdfg, arr, copy.deepcopy(mem))
    # Same for exit nodes
    for arr in out_arrays:
        mem = dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr])
        outer_state.add_edge(nsdfg, arr, m1_exit, f"IN_{arr}", copy.deepcopy(mem))
        outer_state.add_edge(m1_exit, f"OUT_{arr}", outer_state.add_access(arr), None,
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_exit.add_out_connector(f"OUT_{arr}", force=True)
        m1_exit.add_in_connector(f"IN_{arr}", force=True)
    sdfg.validate()
    return sdfg


def _get_map_inside_nested_map():
    klon = dace.symbolic.symbol("klon")
    # Add all arrays to the SDFGs
    in_arrays = set()
    in_scalars = {
        "kfdia",
        "kidia",
    }
    out_arrays = {"int_array", "int_array2"}
    arr_shapes = {
        "int_array": ((klon, 5, 5), (1, klon, klon * 5), dace.int64),
        "int_array2": ((klon, 5, 5), (1, klon, klon * 5), dace.int64),
    }
    scalar_dtypes = {
        "kfdia": dace.int64,
        "kidia": dace.int64,
    }
    outer_sdfg = dace.SDFG("outer")
    inner_sdfg = dace.SDFG("inner")
    inner_state = inner_sdfg.add_state("inner_compute_state", is_start_block=True)
    outer_state = outer_sdfg.add_state("outer_compute_state", is_start_block=True)

    symbols = {"i", "j", "k", "klev", "klon"}
    for sym in symbols:
        inner_sdfg.add_symbol(sym, dace.int64)
        outer_sdfg.add_symbol(sym, dace.int64)

    for sdfg in [inner_sdfg, outer_sdfg]:
        for arr_name in in_arrays.union(out_arrays):
            shape, strides, dtype = arr_shapes[arr_name]
            sdfg.add_array(arr_name, shape, dtype, strides=strides, transient=False)
        for scalar_name in in_scalars:
            sdfg.add_scalar(scalar_name, scalar_dtypes[scalar_name], transient=False)

    t, map_entry, map_exit = inner_state.add_mapped_tasklet(name="assign",
                                                            map_ranges={
                                                                "j": dace.subsets.Range([(0, 4, 1)]),
                                                                "k": dace.subsets.Range([(kfdia - 1, kidia - 1, 1)])
                                                            },
                                                            inputs=dict(),
                                                            code="_out = 0",
                                                            outputs={
                                                                "_out": dace.memlet.Memlet("int_array[k, j, i]"),
                                                            },
                                                            external_edges=True,
                                                            input_nodes=dict(),
                                                            output_nodes={
                                                                "int_array": inner_state.add_access("int_array"),
                                                            })
    for scl_name in {"kfdia", "kidia"}:
        an = inner_state.add_access(scl_name)
        inner_state.add_edge(an, None, map_entry, scl_name, dace.memlet.Memlet(scl_name))
        map_entry.add_in_connector(scl_name)

    inner_symbol_mapping = {sym: sym for sym in symbols}
    in_args = in_scalars
    nsdfg = outer_state.add_nested_sdfg(inner_sdfg, in_args, out_arrays, inner_symbol_mapping)

    m1_entry, m1_exit = outer_state.add_map(name="m1", ndrange={"i": "0:5:1"})

    inner_sdfg.validate()

    t2 = inner_state.add_tasklet("t2", set(), {"_out"}, "_out = 1")
    inner_state.add_edge(t2, "_out", inner_state.add_access("int_array2"), None,
                         dace.memlet.Memlet("int_array2[1,1,1]"))

    # Access nodes to map entry
    for arr in in_arrays.union(in_scalars):
        outer_state.add_edge(outer_state.add_access(arr), None, m1_entry, f"IN_{arr}",
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_entry.add_in_connector(f"IN_{arr}")
        outer_state.add_edge(m1_entry, f"OUT_{arr}", nsdfg, arr,
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_entry.add_out_connector(f"OUT_{arr}")
    # Same for exit nodes
    for arr in out_arrays:
        outer_state.add_edge(nsdfg, arr, m1_exit, f"IN_{arr}",
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_exit.add_in_connector(f"IN_{arr}", force=True)
        outer_state.add_edge(m1_exit, f"OUT_{arr}", outer_state.add_access(arr), None,
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_exit.add_out_connector(f"OUT_{arr}", force=True)
    sdfg.validate()
    return sdfg


def _get_dependency_edge_to_unary_symbol_sdfg():
    klon = dace.symbolic.symbol("klon")
    # Add all arrays to the SDFGs
    in_arrays = {"int_array"}
    in_scalars = {}
    out_arrays = {"int_array2"}
    arr_shapes = {
        "int_array": ((klon, ), (1, ), dace.int64),
        "int_array2": ((klon, ), (1, ), dace.int64),
    }
    scalar_dtypes = {}
    outer_sdfg = dace.SDFG("outer")
    inner_sdfg = dace.SDFG("inner")
    inner_state = inner_sdfg.add_state("inner_compute_state", is_start_block=True)
    outer_state = outer_sdfg.add_state("outer_compute_state", is_start_block=True)

    an1_src = inner_state.add_access("int_array")
    an2_dst = inner_state.add_access("int_array2")
    an_tmp = inner_state.add_access("tmp0")
    inner_state.sdfg.add_scalar("tmp0", dace.int64, dace.dtypes.StorageType.Register, True)
    t1 = inner_state.add_tasklet("t1", set(), {"_out"}, "_out = i + 1")
    t2 = inner_state.add_tasklet("t2", {"_in1", "_in2"}, {"_out"}, "_out = _in1 > _in2")

    inner_state.add_edge(an1_src, None, t1, None, dace.memlet.Memlet(None))
    inner_state.add_edge(t1, "_out", an_tmp, None, dace.memlet.Memlet("tmp0"))
    inner_state.add_edge(an_tmp, None, t2, "_in1", dace.memlet.Memlet("tmp0"))
    inner_state.add_edge(an1_src, None, t2, "_in2", dace.memlet.Memlet("int_array[i]"))
    inner_state.add_edge(t2, "_out", an2_dst, None, dace.memlet.Memlet("int_array2[i]"))

    symbols = {"i", "klon"}
    for sym in symbols:
        inner_sdfg.add_symbol(sym, dace.int64)
        outer_sdfg.add_symbol(sym, dace.int64)

    for sdfg in [inner_sdfg, outer_sdfg]:
        for arr_name in in_arrays.union(out_arrays):
            shape, strides, dtype = arr_shapes[arr_name]
            sdfg.add_array(arr_name, shape, dtype, strides=strides, transient=False)
        for scalar_name in in_scalars:
            sdfg.add_scalar(scalar_name, scalar_dtypes[scalar_name], transient=False)

    inner_symbol_mapping = {sym: sym for sym in symbols}
    in_args = in_arrays.union(in_scalars)
    nsdfg = outer_state.add_nested_sdfg(inner_sdfg, in_args, out_arrays, inner_symbol_mapping)

    m1_entry, m1_exit = outer_state.add_map(name="m1", ndrange={"i": "0:klon:1"})

    inner_sdfg.validate()

    # Access nodes to map entry
    for arr in in_arrays.union(in_scalars):
        outer_state.add_edge(outer_state.add_access(arr), None, m1_entry, f"IN_{arr}",
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_entry.add_in_connector(f"IN_{arr}")
        outer_state.add_edge(m1_entry, f"OUT_{arr}", nsdfg, arr,
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_entry.add_out_connector(f"OUT_{arr}")
    # Same for exit nodes
    for arr in out_arrays:
        outer_state.add_edge(nsdfg, arr, m1_exit, f"IN_{arr}",
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_exit.add_in_connector(f"IN_{arr}", force=True)
        outer_state.add_edge(m1_exit, f"OUT_{arr}", outer_state.add_access(arr), None,
                             dace.memlet.Memlet.from_array(arr, outer_state.sdfg.arrays[arr]))
        m1_exit.add_out_connector(f"OUT_{arr}", force=True)
    sdfg.validate()
    return sdfg


def _get_unstructured_access_cloudsc_sdfg(layout: str = "C") -> dace.SDFG:
    klon = dace.symbolic.symbol("klon")
    klev = dace.symbolic.symbol("klev")
    sdfg_outer = dace.SDFG(f"unstructured_access_cloudsc_sdfg_{layout.lower()}")
    sdfg_inner = dace.SDFG("inner")

    outer_symbols = {("klon", dace.int64), ("klev", dace.int64)}
    inner_symbols = {
        ("jo", dace.int64),
        ("_for_it_88", dace.int64),
    }  #  ("_for_it_85", dace.int64)
    # Add inner symbols to inner SDFG
    for sname, stype in inner_symbols:
        sdfg_inner.add_symbol(sname, stype)

    # Add outer symbols to both
    for sdfg in [sdfg_outer, sdfg_inner]:
        for sname, stype in outer_symbols:
            sdfg.add_symbol(sname, stype)

    if layout == "Fortran":
        arrays = {("iorder", dace.int64, (klon, 5), (1, klon)),
                  ("zqx", dace.float64, (klon, klev, 5), (1, klon, klon * klev)),
                  ("zsinksum", dace.float64, (klon, 5), (1, klon)), ("zratio", dace.float64, (klon, 5), (1, klon))}
    else:
        assert layout == "C"
        arrays = {("iorder", dace.int64, (5, klon), (klon, 1)),
                  ("zqx", dace.float64, (5, klev, klon), (klon * klev, klon, 1)),
                  ("zsinksum", dace.float64, (5, klon), (klon, 1)), ("zratio", dace.float64, (5, klon), (klon, 1))}
    scalars = {("zmm", dace.float64), ("zrr", dace.float64)}

    # Add arrays
    for sdfg in [sdfg_outer, sdfg_inner]:
        for arr_name, dtype, shape, stride in arrays:
            sdfg.add_array(arr_name, shape, dtype, strides=stride)

    # Add scalars to inner SDFG
    for sname, dtype in scalars:
        sdfg.add_scalar(sname, dtype, dace.dtypes.StorageType.Register, True)

    # Add states
    state_outer = sdfg_outer.add_state("outer_s1", is_start_block=True)
    state_inner1 = sdfg_inner.add_state("inner_s1", is_start_block=True)
    state_inner2 = sdfg_inner.add_state("inner_s2")

    # Populate inner SDFG
    # ==============================
    if layout == "Fortran":
        sdfg_inner.add_edge(state_inner1, state_inner2, InterstateEdge(assignments={"jo": "iorder[_for_it_88, 0]"}))
    else:
        sdfg_inner.add_edge(state_inner1, state_inner2, InterstateEdge(assignments={"jo": "iorder[0, _for_it_88]"}))

    zqx = state_inner2.add_access("zqx")
    zsinksum = state_inner2.add_access("zsinksum")
    zratio = state_inner2.add_access("zratio")
    zrr = state_inner2.add_access("zrr")
    zmm = state_inner2.add_access("zmm")

    t1 = state_inner2.add_tasklet("t1", {"_in1"}, {"_out"}, "_out = max(1e-14, _in1)")
    t2 = state_inner2.add_tasklet("t2", {"_in1", "_in2"}, {"_out"}, "_out = max(_in1, _in2)")
    t3 = state_inner2.add_tasklet("t3", {"_in1", "_in2"}, {"_out"}, "_out = _in1 / _in2")

    if layout == "Fortran":
        state_inner2.add_edge(zqx, None, t1, "_in1", dace.memlet.Memlet("zqx[_for_it_88, 0, jo - 1]"))
    else:
        state_inner2.add_edge(zqx, None, t1, "_in1", dace.memlet.Memlet("zqx[ jo - 1, 0, _for_it_88]"))

    state_inner2.add_edge(t1, "_out", zmm, None, dace.memlet.Memlet("zmm"))

    if layout == "Fortran":
        state_inner2.add_edge(zsinksum, None, t2, "_in2", dace.memlet.Memlet("zsinksum[_for_it_88, jo-1]"))
    else:
        state_inner2.add_edge(zsinksum, None, t2, "_in2", dace.memlet.Memlet("zsinksum[jo-1, _for_it_88]"))

    state_inner2.add_edge(zmm, None, t2, "_in1", dace.memlet.Memlet("zmm"))
    state_inner2.add_edge(t2, "_out", zrr, None, dace.memlet.Memlet("zrr"))

    state_inner2.add_edge(zrr, None, t3, "_in2", dace.memlet.Memlet("zrr"))
    state_inner2.add_edge(zmm, None, t3, "_in1", dace.memlet.Memlet("zmm"))

    if layout == "Fortran":
        state_inner2.add_edge(t3, "_out", zratio, None, dace.memlet.Memlet("zratio[_for_it_88, jo -1]"))
    else:
        state_inner2.add_edge(t3, "_out", zratio, None, dace.memlet.Memlet("zratio[jo -1, _for_it_88]"))

    # ==============================

    map_entry, map_exit = state_outer.add_map("m1", {"_for_it_88": "0:klon:1"})

    nsdfg = state_outer.add_nested_sdfg(sdfg_inner,
                                        inputs={"zqx", "zsinksum", "iorder"},
                                        outputs={"zratio"},
                                        symbol_mapping={"_for_it_88": "_for_it_88"})

    # Add in arrays
    for arr_name in {"zqx", "zsinksum", "iorder"}:
        an = state_outer.add_access(arr_name)
        state_outer.add_edge(an, None, map_entry, f"IN_{arr_name}",
                             dace.memlet.Memlet.from_array(arr_name, state_outer.sdfg.arrays[arr_name]))
        state_outer.add_edge(map_entry, f"OUT_{arr_name}", nsdfg, arr_name,
                             dace.memlet.Memlet.from_array(arr_name, state_outer.sdfg.arrays[arr_name]))
        map_entry.add_in_connector(f"IN_{arr_name}")
        map_entry.add_out_connector(f"OUT_{arr_name}")

    # Add out arrays
    for arr_name in {"zratio"}:
        an = state_outer.add_access(arr_name)
        state_outer.add_edge(nsdfg, arr_name, map_exit, f"IN_{arr_name}",
                             dace.memlet.Memlet.from_array(arr_name, state_outer.sdfg.arrays[arr_name]))
        state_outer.add_edge(map_exit, f"OUT_{arr_name}", an, None,
                             dace.memlet.Memlet.from_array(arr_name, state_outer.sdfg.arrays[arr_name]))
        map_exit.add_in_connector(f"IN_{arr_name}")
        map_exit.add_out_connector(f"OUT_{arr_name}")

    sdfg_outer.validate()
    return sdfg_outer
