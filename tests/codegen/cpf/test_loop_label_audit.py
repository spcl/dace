# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Loop labels state only what is proven, in the names of the rendered code (corpus audit of 11-23).

Each test reproduces one root cause of a wrong label found in the HPCAgent-Bench renders: a label that names a loop
counter the rendering never declares, calls a block of thread 0 a loop, or calls a loop independent that accumulates
or shares a buffer; a proof built on an access that may not happen or on sizes no input can have.
"""

import re

import dace
from dace.codegen.cpf import render
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_to_map import LoopRefusal, Refusal
from dace.transformation.passes.canonicalize.annotate_loop_kinds import AnnotateLoopKinds, unproven_reason
from dace.transformation.passes.mark_simd_maps import MarkSIMDMaps
from dace.transformation.passes.parallelization.wavefront.skew import WavefrontSkew
from tests.codegen.cpf.test_specialization_hints import (
    comment_lines,
    loops_of,
    named,
    skewed_sdfg,
)
from tests.codegen.cpf.test_token_cuts import team_one_trip

N = dace.symbol("N")
M = dace.symbol("M")
LDA = dace.symbol("lda")
T = dace.symbol("T")


def annotated(program, name: str) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    AnnotateLoopKinds().apply_pass(sdfg, {})
    return sdfg


def loop_hint(sdfg: dace.SDFG) -> str:
    (loop,) = [loop for loop in loops_of(sdfg) if loop.specialization_hint]
    return loop.specialization_hint


def map_hint_of(sdfg: dace.SDFG, param: str) -> str:
    (entry,) = [
        node
        for node, parent in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.MapEntry) and node.map.params[-1] == param
    ]
    return entry.specialization_hint or ""


def code_lines(code: str) -> str:
    return "\n".join(line for line in code.splitlines() if not line.strip().startswith("//"))


def test_a_tiled_wavefront_names_only_counters_the_rendering_declares():
    """needleman_wunsch, wf_triangular: the tile labels named ``_loop_it_0`` and the placeholder ``w``."""
    code = named(skewed_sdfg("cpf_audit_wavefront_tiles", tile=4))
    labels = [line for line in comment_lines(code) if "wavefront" in line]
    named_counters = [
        name for line in labels for name in re.findall(r"\bstep (\w+) runs\b|(\w+) is the tile column", line)
    ]
    counters = [name for pair in named_counters for name in pair if name]
    assert len(labels) >= 2 and counters, labels
    assert all(re.search(rf"\bfor \([^;]*\b{name}\b", code_lines(code)) for name in counters), (counters, code)
    assert not any("_loop_it_" in line or "+ w)" in line for line in labels), labels


def test_a_one_trip_map_in_a_team_is_labelled_thread_zeros_block():
    """tsvc s118, bfs, gmres: ``if (omp_get_thread_num() == 0)`` carried ``parallel -- independent``."""
    sdfg = team_one_trip.to_sdfg(simplify=True)
    sdfg.name = "cpf_audit_thread_zero"
    lines = render(sdfg, language="c").code.splitlines()
    (block,) = [n for n, line in enumerate(lines) if "if (omp_get_thread_num() == 0)" in line]
    label = next(line.strip() for line in reversed(lines[:block]) if " -- " in line)
    assert label.startswith("// sequential -- not a loop: the map's one iteration runs on thread 0"), lines


@dace.program
def row_sums(a: dace.float64[N, M], out: dace.float64[N]):
    for i, j in dace.map[0:N, 0:M]:
        out[i] += a[i, j]


def test_each_map_of_an_expanded_nest_says_what_it_is():
    """conv_2d, lavamd, nbody: the vector pass split a map in two after the labels, and the inner map, which
    accumulates into ``out[i]``, fell back to ``parallel -- independent``."""
    sdfg = annotated(row_sums, "cpf_audit_expanded")
    (entry,) = [node for node, parent in sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry)]
    entry.map.schedule = dace.ScheduleType.CPU_Multicore  # as codegen's schedule inference leaves it
    assert MarkSIMDMaps().apply_pass(sdfg, {}), "the vector pass did not split the map"
    assert map_hint_of(sdfg, "i").startswith("parallel -- independent"), map_hint_of(sdfg, "i")
    assert map_hint_of(sdfg, "j").startswith("parallel -- reduction: + into out"), map_hint_of(sdfg, "j")


def nested_scatter_sdfg() -> dace.SDFG:
    """ilu0's ``f1``: a map whose nested body adds into ``out[idx[k]]``; only the inner edge carries the WCR."""
    body = dace.SDFG("scatter_body")
    body.add_symbol("k", dace.int64)
    body.add_array("x", [N], dace.float64)
    body.add_array("idx", [N], dace.int64)
    body.add_array("y", [M], dace.float64)
    inner = body.add_state()
    add = inner.add_tasklet("add", {"v", "at"}, {"w"}, "w[at] = v")
    inner.add_edge(inner.add_read("x"), None, add, "v", Memlet("x[k]"))
    inner.add_edge(inner.add_read("idx"), None, add, "at", Memlet("idx[k]"))
    inner.add_edge(add, "w", inner.add_write("y"), None, Memlet("y[0:M]", wcr="lambda p, q: p + q", dynamic=True))

    sdfg = dace.SDFG("cpf_audit_nested_wcr")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("idx", [N], dace.int64)
    sdfg.add_array("out", [M], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("scatter", {"k": "0:N"})
    call = state.add_nested_sdfg(body, {"x", "idx"}, {"y"}, {"N": "N", "M": "M", "k": "k"})
    state.add_memlet_path(state.add_read("a"), entry, call, dst_conn="x", memlet=Memlet("a[0:N]"))
    state.add_memlet_path(state.add_read("idx"), entry, call, dst_conn="idx", memlet=Memlet("idx[0:N]"))
    state.add_memlet_path(call, exit_node, state.add_write("out"), src_conn="y", memlet=Memlet("out[0:M]"))
    sdfg.validate()
    return sdfg


def test_a_map_accumulating_inside_a_nested_body_is_a_reduction():
    """ilu0, cloudsc_subsidence_forcing: the accumulation sat in a nested SDFG, and the map was labelled
    independent."""
    sdfg = nested_scatter_sdfg()
    AnnotateLoopKinds().apply_pass(sdfg, {})
    assert map_hint_of(sdfg, "k").startswith("parallel -- reduction: + into out"), map_hint_of(sdfg, "k")


def shared_scratch_sdfg() -> dace.SDFG:
    """A map whose nested body writes ``buf[0]`` every iteration: a scratch lifted out of the body, one for all."""
    body = dace.SDFG("scratch_body")
    body.add_symbol("row", dace.int64)
    body.add_array("x", [N], dace.float64)
    body.add_array("buf", [M], dace.float64)
    body.add_array("y", [N], dace.float64)
    inner = body.add_state()
    fill = inner.add_tasklet("fill", {"v"}, {"w"}, "w = v * 2")
    use = inner.add_tasklet("use", {"w2"}, {"r"}, "r = w2 + 1")
    scratch = inner.add_access("buf")
    inner.add_edge(inner.add_read("x"), None, fill, "v", Memlet("x[row]"))
    inner.add_edge(fill, "w", scratch, None, Memlet("buf[0]"))
    inner.add_edge(scratch, None, use, "w2", Memlet("buf[0]"))
    inner.add_edge(use, "r", inner.add_write("y"), None, Memlet("y[row]"))

    sdfg = dace.SDFG("cpf_audit_shared_scratch")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("out", [N], dace.float64)
    sdfg.add_transient("buf", [M], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("rows", {"i": "0:N"})
    call = state.add_nested_sdfg(body, {"x"}, {"buf", "y"}, {"N": "N", "M": "M", "row": "i"})
    state.add_memlet_path(state.add_read("a"), entry, call, dst_conn="x", memlet=Memlet("a[0:N]"))
    state.add_memlet_path(call, exit_node, state.add_write("buf"), src_conn="buf", memlet=Memlet("buf[0:M]"))
    state.add_memlet_path(call, exit_node, state.add_write("out"), src_conn="y", memlet=Memlet("out[0:N]"))
    sdfg.validate()
    return sdfg


def test_a_map_sharing_one_buffer_across_iterations_names_it():
    """cloudsc_subsidence_forcing: every iteration wrote ``zevap[0:KLON]``, and the map read plain independent."""
    sdfg = shared_scratch_sdfg()
    AnnotateLoopKinds().apply_pass(sdfg, {})
    hint = map_hint_of(sdfg, "i")
    assert hint.startswith("parallel -- independent") and hint.endswith("; private per iteration: buf"), hint


def row_through_two_bodies_sdfg() -> dace.SDFG:
    """mandelbrot: the map passes all of ``rows`` to a body, whose own call writes ``rows[row]``."""
    deeper = dace.SDFG("row_writer")
    deeper.add_symbol("r", dace.int64)
    deeper.add_array("x", [N], dace.float64)
    deeper.add_array("rows", [N], dace.float64)
    leaf = deeper.add_state()
    scale = leaf.add_tasklet("scale", {"v"}, {"w"}, "w = v * 2")
    leaf.add_edge(leaf.add_read("x"), None, scale, "v", Memlet("x[r]"))
    leaf.add_edge(scale, "w", leaf.add_write("rows"), None, Memlet("rows[r]"))

    body = dace.SDFG("row_body")
    body.add_symbol("row", dace.int64)
    body.add_array("x", [N], dace.float64)
    body.add_array("rows", [N], dace.float64)
    inner = body.add_state()
    call = inner.add_nested_sdfg(deeper, {"x"}, {"rows"}, {"N": "N", "r": "row"})
    inner.add_edge(inner.add_read("x"), None, call, "x", Memlet("x[0:N]"))
    inner.add_edge(call, "rows", inner.add_write("rows"), None, Memlet("rows[0:N]"))

    sdfg = dace.SDFG("cpf_audit_row_through_two_bodies")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_transient("rows", [N], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("tiles", {"i": "0:N"})
    outer = state.add_nested_sdfg(body, {"x"}, {"rows"}, {"N": "N", "row": "i"})
    state.add_memlet_path(state.add_read("a"), entry, outer, dst_conn="x", memlet=Memlet("a[0:N]"))
    state.add_memlet_path(outer, exit_node, state.add_write("rows"), src_conn="rows", memlet=Memlet("rows[0:N]"))
    sdfg.validate()
    return sdfg


def test_a_buffer_a_deeper_body_writes_per_iteration_is_not_private():
    """mandelbrot: ``private per iteration: Z`` on a loop whose iterations write different rows of ``Z``."""
    sdfg = row_through_two_bodies_sdfg()
    AnnotateLoopKinds().apply_pass(sdfg, {})
    assert (
        map_hint_of(sdfg, "i")
        == "parallel -- independent: no iteration writes an element another iteration reads or writes"
    )


@dace.program
def masked_rows(a: dace.float64[N, 4], mask: dace.bool_[N, 4], out: dace.float64[4]):
    for i in range(N):
        out[mask[i]] = a[i]


def test_a_write_that_may_not_happen_proves_no_dependence():
    """Review F15: a masked store fires only where the mask holds, so two iterations need not write the same
    element; the solver's pair of iterations was printed as a proof."""
    hint = loop_hint(annotated(masked_rows, "cpf_audit_dynamic"))
    assert hint.startswith("unknown -- "), hint


@dace.program
def strided_store(a: dace.float64[M * LDA], b: dace.float64[M]):
    for i in range(M):
        a[i * LDA] = b[i]


def test_a_dependence_that_needs_a_size_of_zero_is_not_proven():
    """Review F17: ``a[i * lda]`` collides only for ``lda = 0``, where ``a`` has no element at all."""
    hint = loop_hint(annotated(strided_store, "cpf_audit_degenerate"))
    assert hint.startswith("unknown -- "), hint


def strided_copy_sdfg() -> dace.SDFG:
    """``for i: a[i, 0:M:2] = a[i - 1, 0:M:2]``, an array-to-array copy with a stride."""
    sdfg = dace.SDFG("cpf_audit_copy")
    sdfg.add_array("a", [N, M], dace.float64)
    rows = LoopRegion("rows", "i < N", "i", "i = 1", "i = i + 1")
    sdfg.add_node(rows, is_start_block=True)
    body = rows.add_state("body", is_start_block=True)
    body.add_nedge(
        body.add_read("a"), body.add_write("a"), Memlet(data="a", subset="i - 1, 0:M:2", other_subset="i, 0:M:2")
    )
    sdfg.validate()
    return sdfg


def test_a_copy_dependence_names_the_range_not_an_internal_iterator():
    """Review F31: the copy's iterator ``__copy<id>`` leaked into the label."""
    sdfg = strided_copy_sdfg()
    AnnotateLoopKinds().apply_pass(sdfg, {})
    hint = loop_hint(sdfg)
    assert hint.startswith("sequential -- proven dependence on a[") and "__copy" not in hint, hint


@dace.program
def seidel_steps(a: dace.float64[N, N]):
    for t in range(T):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                a[i, j] = (a[i - 1, j] + a[i, j - 1] + a[i, j] + a[i + 1, j] + a[i, j + 1]) * 0.2


def test_a_time_loop_around_a_skewed_nest_keeps_its_proven_recurrence():
    """seidel_2d: after the skew the accesses sit under tile guards, and the time loop read ``unknown`` with a dump of
    a subset; its body proved the recurrence before the skew."""
    sdfg = seidel_steps.to_sdfg(simplify=True)
    sdfg.name = "cpf_audit_seidel"
    assert WavefrontSkew(target="cpu").apply_pass(sdfg, {}), "the nest was not skewed"
    AnnotateLoopKinds().apply_pass(sdfg, {})
    (steps,) = [loop for loop in loops_of(sdfg) if loop.loop_variable == "t"]
    hint = steps.specialization_hint
    assert hint.startswith("sequential -- proven dependence on a: an iteration reads what an earlier one wrote"), hint
    assert "subset" not in hint and len(hint) < 200, hint


def test_a_skewed_seidel_renders_unique_counters_and_a_proven_header():
    """seidel_2d: the time loop in ``body`` and the tile interior in ``loop_body`` both rendered as ``i``, so the
    header's ``open: i`` named two loops; the time loop read ``unknown``."""
    sdfg = seidel_steps.to_sdfg(simplify=True)
    sdfg.name = "cpf_audit_seidel_render"
    assert WavefrontSkew(target="cpu").apply_pass(sdfg, {}), "the nest was not skewed"
    code = render(sdfg, language="c").code
    counters = re.findall(r"\bfor \(int64_t (\w+) =", code)
    assert len(counters) >= 3 and len(counters) == len(set(counters)), counters
    (proven,) = [line for line in code.splitlines() if line.startswith("// cpf: proven: ")]
    assert re.fullmatch(r"// cpf: proven: (all|partial; open: [^,]+(, [^,]+)*)", proven), proven
    lines = code.splitlines()
    (steps,) = [n for n, line in enumerate(lines) if re.search(r"\bfor \(int64_t (\w+) = 0; \(?\1 < T\b", line)]
    label = next(line.strip() for line in reversed(lines[:steps]) if " -- " in line)
    assert label.startswith("// sequential -- proven dependence on a: "), label


def test_every_unknown_reason_says_what_failed_when_it_is_parallel_and_what_goes_wrong_otherwise():
    """seidel_2d (review #8): a refusal the labels had no sentence for printed its message, a 300-token subset. Every
    kind now reads ``<what was not shown>; parallel if (or only as) <condition>; otherwise <what goes wrong>``."""
    for kind in Refusal:
        refusal = LoopRefusal(kind, "dst_subset=64*j - 64*Min(j, int_ceil(2*N - 5, 64) - 1) + 1", container="A")
        reason = unproven_reason(refusal)
        assert re.fullmatch(r"[^;]+(; [^;]+)*; parallel (if|only as) [^;]+; otherwise [^;]+", reason), (kind, reason)
        assert "dst_subset" not in reason and "Min(" not in reason, (kind, reason)


if __name__ == "__main__":
    test_a_tiled_wavefront_names_only_counters_the_rendering_declares()
    test_a_one_trip_map_in_a_team_is_labelled_thread_zeros_block()
    test_each_map_of_an_expanded_nest_says_what_it_is()
    test_a_map_accumulating_inside_a_nested_body_is_a_reduction()
    test_a_map_sharing_one_buffer_across_iterations_names_it()
    test_a_buffer_a_deeper_body_writes_per_iteration_is_not_private()
    test_a_write_that_may_not_happen_proves_no_dependence()
    test_a_dependence_that_needs_a_size_of_zero_is_not_proven()
    test_a_copy_dependence_names_the_range_not_an_internal_iterator()
    test_a_time_loop_around_a_skewed_nest_keeps_its_proven_recurrence()
    test_a_skewed_seidel_renders_unique_counters_and_a_proven_header()
    test_every_unknown_reason_says_what_failed_when_it_is_parallel_and_what_goes_wrong_otherwise()
