# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPF states the device trade a canonicalizing pass declined to make.

Canonicalization always picks the most parallel form. That form is not the fastest one on every
device -- a parallel scan does more work than the sequential loop it replaces, and snapshotting a
read window costs a copy a CPU would rather not pay -- so the pass records what it could have
traded for instead of silently discarding the choice. The record is a ``specialization_hint`` on
the node or loop it applies to, and the standalone (CPF) rendering is the one place it is written
out: CPF output is read by a specializing pass or by a person, and both need to know which parts
of the shape were a decision.

A second kind of hint names what a loop IS. The rendering flattens a parallel Map and a sequential
LoopRegion to the same ``for``, so ``AnnotateLoopKinds`` writes the distinction back down before it
is lost -- and splits the sequential case in two, because "a carried dependence was proven" and
"the dependence test declined to answer" are different facts and a reader specializing this code
has to tell them apart.

These tests hold the ends of both contracts: a hint reaches the rendering, on a map, on a loop and
through a library node that the expansion consumes; every loop kind is named and named correctly;
the naming adds comments and nothing else; and nothing reaches ordinary code generation, which
already has a target and would only be given noise.
"""

import re

import pytest

import dace
from dace import cpf_lowering
from dace.codegen.cpf import CONTRACT_LINES, render
from dace.codegen.targets.experimental_cpu import map_hint
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from dace.libraries.standard.nodes.scan import PARALLEL_SCAN_HINT, SEQUENTIAL_SCAN_HINT, Scan, ScanOp
from dace.properties import CodeBlock
from dace.sdfg import nodes, tasklet_utils
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.canonicalize.annotate_loop_kinds import (
    CONFLICT_MEANING,
    EARLY_EXIT,
    MAP_DIMENSIONS,
    PARALLEL,
    PRIVATE,
    PROVEN_PREFIX,
    REDUCTION,
    SEQUENTIAL_PINNED,
    SEQUENTIAL_PROVEN,
    TEAM,
    WAVEFRONT_FRONT,
    WHILE_LOOP,
    AnnotateLoopKinds,
    is_scratch,
    loop_header_names,
    parallel_hint,
    proven_carrying_access,
    witnessed_carrying_access,
)
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.parallelization.wavefront.skew import WavefrontSkew

N = dace.symbol("N")
K = dace.symbol("K")

#: Two lines, because a hint names a trade per device and the emitter has to keep them apart.
MAP_HINT = "traded a tiled map for a flat one.\nCPU: the tiles are worth trying."


def scaling_sdfg(name: str) -> dace.SDFG:
    """A single parallel map, the smallest graph with a ``MapEntry`` to hang a hint on."""

    @dace.program
    def scaler(a: dace.float64[N], b: dace.float64[N]):
        for i in dace.map[0:N]:
            b[i] = a[i] * 2.0

    sdfg = scaler.to_sdfg(simplify=True)
    sdfg.name = name
    return sdfg


def map_entries(sdfg: dace.SDFG):
    return [node for state in sdfg.states() for node in state.nodes() if isinstance(node, nodes.MapEntry)]


def comment_lines(code: str):
    return [line.strip() for line in code.splitlines() if line.strip().startswith("//")]


def test_the_rendering_states_which_verdicts_are_settled_and_which_are_open():
    """Proven verdicts need no more reasoning, so the head of the unit says so once, before any loop."""
    lines = render(scaling_sdfg("cpf_contract")).code.splitlines()
    # A unit of parallel loops explains the parallel class only: no line for a class it does not use
    assert lines[1:5] == [CONTRACT_LINES[0], *CONTRACT_LINES[1:3], CONTRACT_LINES[-1]], lines[:8]


def test_a_map_hint_is_rendered_one_comment_line_per_line():
    """Each line of a hint gets its own ``//``.

    The alternatives a hint describes are per-device (``CPU: ... / GPU: ...``) and do not read as
    one sentence, so joining them onto one line would be a worse comment than none.
    """
    sdfg = scaling_sdfg("cpf_hint_map")
    entries = map_entries(sdfg)
    assert entries, "the program lost its map, so this test would pass without emitting anything"
    for entry in entries:
        entry.specialization_hint = MAP_HINT

    rendered = comment_lines(render(sdfg).code)
    assert "// traded a tiled map for a flat one." in rendered
    assert "// CPU: the tiles are worth trying." in rendered


def test_a_while_loop_header_reads_only_its_condition():
    assert loop_header_names(LoopRegion("while_loop", "i < N"), {}) == {"i", "N"}


def test_a_loop_hint_is_rendered():
    """A hint on a ``LoopRegion``, which is where ``BreakAntiDependence`` records its trade."""

    @dace.program
    def prefix(a: dace.float64[N], b: dace.float64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]

    sdfg = prefix.to_sdfg(simplify=True)
    sdfg.name = "cpf_hint_loop"
    loops = [region for region in sdfg.all_control_flow_regions(recursive=True) if isinstance(region, LoopRegion)]
    assert loops, "the sequential loop was rewritten away, so there is nothing to hang a hint on"
    for loop in loops:
        loop.specialization_hint = "kept the loop sequential.\nGPU: a scan is worth trying."

    rendered = comment_lines(render(sdfg).code)
    assert "// kept the loop sequential." in rendered
    assert "// GPU: a scan is worth trying." in rendered


def test_a_scan_states_its_trade_although_the_expansion_consumes_the_node():
    """The hint has to be captured before expansion, for the same reason the description is.

    A ``Scan`` renders as several maps and a loop, and none of them remembers what chose their
    shape -- by the time code is emitted the library node is gone.
    """
    sdfg = dace.SDFG("cpf_hint_scan")
    # Past the small-extent bound: a scan over a few literal elements runs single-core.
    sdfg.add_array("arr_in", [256], dace.float64)
    sdfg.add_array("arr_out", [256], dace.float64)
    state = sdfg.add_state("scan")
    node = Scan("Scan", op=ScanOp.SUM)
    state.add_node(node)
    state.add_edge(state.add_read("arr_in"), None, node, "_scan_in", dace.Memlet("arr_in[0:256]"))
    state.add_edge(node, "_scan_out", state.add_write("arr_out"), None, dace.Memlet("arr_out[0:256]"))
    sdfg.validate()

    rendered = comment_lines(render(sdfg).code)
    # CPF renders a host Scan through its parallel CPU expansion, so the trade is stated from that side.
    assert "// " + PARALLEL_SCAN_HINT.splitlines()[0] in rendered
    assert "// " + PARALLEL_SCAN_HINT.splitlines()[1] in rendered
    # Folded into the node's description so one expansion is one trade, not one per map it leaves.
    assert rendered.count("// " + PARALLEL_SCAN_HINT.splitlines()[1]) == 1
    assert not [line for line in rendered if line.startswith("// sequential -- scan")]


@pytest.mark.parametrize("implementation, hint", [("pure", SEQUENTIAL_SCAN_HINT), ("CPU", PARALLEL_SCAN_HINT)])
def test_a_scan_hint_follows_the_expansion_that_ran(implementation, hint):
    """scan_affine_decay: "parallel scan" was printed over the sequential loop ``pure`` emits."""
    sdfg = dace.SDFG(f"cpf_hint_scan_{implementation.lower()}")
    sdfg.add_array("arr_in", [N], dace.float64)
    sdfg.add_array("arr_out", [N], dace.float64)
    state = sdfg.add_state("scan")
    node = Scan("Scan", op=ScanOp.SUM)
    state.add_node(node)
    state.add_edge(state.add_read("arr_in"), None, node, "_scan_in", dace.Memlet("arr_in[0:N]"))
    state.add_edge(node, "_scan_out", state.add_write("arr_out"), None, dace.Memlet("arr_out[0:N]"))
    assert node.specialization_hint == PARALLEL_SCAN_HINT
    node.expand(state, implementation)
    assert node.specialization_hint == hint


def test_no_hint_and_an_empty_hint_render_identically():
    """An empty hint is not a hint. A pass that clears one must not leave a bare ``//`` behind."""
    without = render(scaling_sdfg("cpf_hint_absent")).code

    sdfg = scaling_sdfg("cpf_hint_empty")
    for entry in map_entries(sdfg):
        entry.specialization_hint = ""
    empty = render(sdfg).code

    blank = scaling_sdfg("cpf_hint_blank")
    for entry in map_entries(blank):
        entry.specialization_hint = "   \n\t"
    whitespace = render(blank).code

    assert comment_lines(empty) == comment_lines(without)
    assert comment_lines(whitespace) == comment_lines(without)
    assert "//" not in [line.strip() for line in empty.splitlines()]


@pytest.mark.parametrize("hint", [None, "", "   \n\t"])
def test_hint_comment_is_empty_for_an_empty_hint(hint):
    """The renderer answers for itself, so a caller cannot produce a dangling comment marker."""
    with cpf_lowering.dialect_scope(cpf_lowering.Dialect.STANDALONE):
        assert cpf_lowering.hint_comment(hint) == ""


def test_hint_comment_is_empty_outside_a_standalone_rendering():
    """An ordinary build already has a target; the alternatives are noise in code nobody reads."""
    with cpf_lowering.dialect_scope(cpf_lowering.Dialect.RUNTIME):
        assert cpf_lowering.hint_comment(MAP_HINT) == ""
    with cpf_lowering.dialect_scope(cpf_lowering.Dialect.STANDALONE_C):
        assert cpf_lowering.hint_comment(MAP_HINT) != ""


def test_ordinary_codegen_carries_no_hint():
    """The whole feature is scoped to the standalone rendering."""
    sdfg = scaling_sdfg("cpf_hint_leak")
    for entry in map_entries(sdfg):
        entry.specialization_hint = MAP_HINT

    with dace.config.set_temporary("compiler", "cpu", "implementation", value="experimental_readable"):
        ordinary = "\n".join(obj.clean_code for obj in sdfg.generate_code())
    assert "traded a tiled map for a flat one." not in ordinary


# Naming what every loop is: parallel, sequential (carried or pinned), unknown, and the wavefront roles.


def loops_of(sdfg: dace.SDFG):
    return [region for region in sdfg.all_control_flow_regions(recursive=True) if isinstance(region, LoopRegion)]


def named(sdfg: dace.SDFG) -> str:
    """``sdfg`` rendered after :class:`AnnotateLoopKinds` has named its loops."""
    AnnotateLoopKinds().apply_pass(sdfg, {})
    return render(sdfg).code


def carried_sdfg(name: str) -> dace.SDFG:
    """A row recurrence: the ``i`` loop carries a dependence, the ``j`` axis does not."""

    @dace.program
    def rows(a: dace.float64[N, N], b: dace.float64[N, N]):
        for i in range(1, N):
            for j in range(N):
                b[i, j] = b[i - 1, j] + a[i, j]

    sdfg = rows.to_sdfg(simplify=True)
    sdfg.name = name
    return sdfg


def breaking_sdfg(name: str) -> dace.SDFG:
    """A loop with an early exit. ``LoopToMap`` stops at the ``break`` -- before any dependence
    question is asked -- which is the case that must not be reported as a proof."""

    @dace.program
    def until_negative(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N):
            if a[i] < 0.0:
                break
            b[i] = a[i]

    sdfg = until_negative.to_sdfg(simplify=True)
    sdfg.name = name
    return sdfg


def unexamined_sdfg(name: str) -> dace.SDFG:
    """A while loop: no iteration variable, so ``LoopToMap`` has nothing to match on and the
    dependence question is never put to it at all.

    Built by hand rather than written as a ``dace.program``, so the test states the one property it
    is about -- a ``LoopRegion`` with no ``loop_variable`` -- and does not ride on the Python
    frontend's while-loop support.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [16], dace.float64)
    sdfg.add_array("b", [1], dace.float64)
    sdfg.add_symbol("it", dace.int64)
    entry = sdfg.add_state("entry", is_start_block=True)
    loop = LoopRegion("spin", "it < 16")
    sdfg.add_node(loop)
    sdfg.add_edge(entry, loop, dace.InterstateEdge(assignments={"it": "0"}))
    body = loop.add_state("body", is_start_block=True)
    loop.add_edge(body, loop.add_state("step"), dace.InterstateEdge(assignments={"it": "it + 1"}))
    tasklet = body.add_tasklet("accumulate", {"inp"}, {"out"}, "out = inp")
    body.add_edge(body.add_read("a"), None, tasklet, "inp", dace.Memlet("a[it]"))
    body.add_edge(tasklet, "out", body.add_write("b"), None, dace.Memlet("b[0]"))
    sdfg.validate()
    assert not loop.loop_variable, "the loop grew an iteration variable, so it is no longer unexamined"
    return sdfg


def skewed_sdfg(name: str, tile: int) -> dace.SDFG:
    """``wf_north_west`` after an isolated skew. ``tile=0`` disables tiling."""

    @dace.program
    def wf_north_west(a: dace.float64[N, N]):
        for i in range(1, N):
            for j in range(1, N):
                a[i, j] = a[i, j] + a[i - 1, j] + a[i, j - 1]

    sdfg = wf_north_west.to_sdfg(simplify=True)
    sdfg.name = name
    skew = WavefrontSkew(target="cpu")
    skew.tile_i, skew.tile_j = tile, tile
    assert skew.apply_pass(sdfg, {}), "the nest was not skewed, so there is no wavefront to name"
    sdfg.validate()
    return sdfg


def test_a_parallel_map_says_it_is_parallel():
    """A Map renders as a plain ``for``. Without the comment nothing in the text says the order
    is free, which is the first thing a reader specializing the code needs to know."""
    rendered = comment_lines(named(scaling_sdfg("cpf_kind_parallel")))
    assert "// " + PARALLEL.splitlines()[0] in rendered


def test_a_reduction_map_names_its_operator_and_accumulator_not_independence():
    """A map that accumulates into ``s`` through a WCR edge has iterations that all update ``s``: they are not
    independent, they reduce. The class line must say so and name the operator and the accumulator, or an agent that
    rewrites the loop under "the iterations are independent" can drop the reduction (CPF paper review, Figure 4)."""
    sdfg = dace.SDFG("cpf_kind_reduction")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("s", [1], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "accumulate",
        {"j": "0:N"},
        {"inp": dace.Memlet("a[j]")},
        "out = inp",
        {"out": dace.Memlet("s[0]", wcr="lambda x, y: x + y")},
        external_edges=True,
    )
    entry = next(n for n in state.nodes() if isinstance(n, nodes.MapEntry))
    first = parallel_hint(state, entry, []).splitlines()[0]
    assert first == REDUCTION.format(reductions="+ into s")
    assert "independent" not in first.split(";")[0]


def test_a_witness_skips_a_nested_argument_that_is_the_callers_scratch():
    """TSVC ``s118`` after the CPU team is hoisted: the outer loop sits in a nested SDFG, and the accumulator ``s`` the
    loop privatized is the caller's transient, passed in as an argument. Every iteration writes ``s`` before it reads
    it, so a conflict on ``s`` is no dependence; the witness must name the real one, through ``a``."""

    @dace.program
    def body(a: dace.float64[N], s: dace.float64[1]):
        for i in range(1, N):
            s[0] = a[i]
            s[0] = s[0] + a[i - 1]
            a[i] = s[0]

    inner = body.to_sdfg(simplify=True)
    outer = dace.SDFG("cpf_witness_scratch")
    outer.add_array("a", [N], dace.float64)
    outer.add_array("s", [1], dace.float64, transient=True)
    init = outer.add_state("init", is_start_block=True)
    zero = init.add_tasklet("zero", {}, {"out"}, "out = 0.0")
    init.add_edge(zero, "out", init.add_write("s"), None, dace.Memlet("s[0]"))
    state = outer.add_state_after(init)
    call = state.add_nested_sdfg(inner, {"a": None, "s": None}, {"a": None, "s": None}, {"N": "N"})
    state.add_edge(state.add_read("a"), None, call, "a", dace.Memlet("a[0:N]"))
    state.add_edge(state.add_read("s"), None, call, "s", dace.Memlet("s[0]"))
    state.add_edge(call, "a", state.add_write("a"), None, dace.Memlet("a[0:N]"))
    state.add_edge(call, "s", state.add_write("s"), None, dace.Memlet("s[0]"))
    outer.validate()
    (loop,) = loops_of(inner)
    assert is_scratch(inner, "s") and not is_scratch(inner, "a")
    found = witnessed_carrying_access(loop)
    assert found is not None and found.startswith("RAW on a["), found


def test_a_read_its_own_iteration_rewrites_is_no_raw():
    """``t`` is an argument the loop reuses: each iteration stores ``t[0]`` and then loads it. Two iterations conflict
    on ``t[0]``, but the load sees its own iteration's store, so the carried dependence is storage reuse (WAR or
    WAW), not a RAW."""

    @dace.program
    def reuse(a: dace.float64[N], b: dace.float64[N], t: dace.float64[1]):
        for i in range(N):
            t[0] = a[i]
            b[i] = t[0]

    (loop,) = loops_of(reuse.to_sdfg(simplify=False))
    found = witnessed_carrying_access(loop)
    assert found is not None and not found.startswith("RAW") and "t[0]" in found, found


def test_a_multi_dimensional_map_names_every_loop_it_renders_as():
    """One map becomes one nested ``for`` per dimension, with the comment above the outermost only (addusxx_g's
    reduction nest); the comment names the inner ones too, or they read as loops nobody classified."""
    entry, _ = dace.SDFG("cpf_kind_map_dims").add_state().add_map("plane", {"i": "0:8", "j": "0:8"})
    lines = map_hint(entry).splitlines()
    assert lines[0] == PARALLEL.splitlines()[0]
    assert lines[-1] == MAP_DIMENSIONS.format(count=2, params="i, j")


def test_a_proven_carried_dependence_is_rendered_as_a_proof_and_names_it():
    """The strong half of the sequential case: ``LoopToMap`` proved the dependence, so the order
    is required, and the hint carries the analysis' own words rather than restating the loop."""
    rendered = comment_lines(named(carried_sdfg("cpf_kind_carried")))
    proofs = [line for line in rendered if line.startswith(f"// {PROVEN_PREFIX}")]
    assert proofs, rendered
    # The hint carries the ACCESS that carries the dependence, which is what a reader acts on.
    assert "b[" in proofs[0], proofs
    # The row axis is proven; nothing here is merely unknown.
    assert not [line for line in rendered if line.startswith("// unknown")]


def test_an_early_exit_is_sequential_with_its_cause_and_never_a_proven_dependence():
    """``LoopToMap`` refuses a loop with a ``break`` at the control-flow check, long before it asks anything about
    data. The order is still required -- no iteration after the exit may run -- so the loop is ``sequential`` and the
    comment names the exit. Calling it a proven dependence would send a reader looking for a recurrence that is not
    there."""
    rendered = comment_lines(named(breaking_sdfg("cpf_kind_unknown")))
    assert "// " + EARLY_EXIT.format(statement="break") in rendered, rendered
    assert not [line for line in rendered if line.startswith("// unknown")], rendered
    assert not [line for line in rendered if line.startswith(f"// {PROVEN_PREFIX}")], rendered


def branch_carried_sdfg(name: str) -> dace.SDFG:
    """TSVC s161: the write ``c[i + 1]`` and the read ``c[i]`` each sit under a data-dependent branch, so no pair of
    iterations is known to run both."""

    @dace.program
    def s161(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N], e: dace.float64[N]):
        for i in range(N - 1):
            if b[i] < 0.0:
                c[i + 1] = a[i] + d[i] * d[i]
            else:
                a[i] = c[i] + d[i] * e[i]

    sdfg = s161.to_sdfg(simplify=True)
    sdfg.name = name
    return sdfg


def test_an_access_under_a_data_dependent_branch_proves_no_carried_dependence():
    """``sequential -- carried`` is a proof, and accesses a branch may skip prove nothing: s161 was labelled
    ``carried: RAW on c[i]`` from its two branch bodies. The solver witness skips branch bodies; so must the
    constant-distance proof."""
    rendered = comment_lines(named(branch_carried_sdfg("cpf_kind_branch_carried")))
    assert not [line for line in rendered if line.startswith(f"// {PROVEN_PREFIX}")], rendered


def carried_symbol_sdfg(name: str, branch: bool) -> dace.SDFG:
    """``a[i] = s`` then ``s = i``: the symbol ``s`` carries the previous iteration's value, assigned on every
    iteration, or only when ``b[i] > 0`` if ``branch``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.int64)
    sdfg.add_symbol("s", dace.int64)
    init = sdfg.add_state("init", is_start_block=True)
    loop = LoopRegion("carry", condition_expr="i < N", loop_var="i", initialize_expr="i = 0", update_expr="i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={"s": "0"}))
    use = loop.add_state("use", is_start_block=True)
    put = use.add_tasklet("put", {}, {"out"}, "out = s")
    use.add_edge(put, "out", use.add_write("a"), None, dace.Memlet("a[i]"))
    after = loop.add_state("after")
    if branch:
        pick = ConditionalBlock("pick")
        loop.add_node(pick)
        taken = ControlFlowRegion("taken", sdfg=sdfg)
        pick.add_branch(CodeBlock("b[i] > 0"), taken)
        first = taken.add_state("first", is_start_block=True)
        taken.add_edge(first, taken.add_state("second"), dace.InterstateEdge(assignments={"s": "i"}))
        loop.add_edge(use, pick, dace.InterstateEdge())
        loop.add_edge(pick, after, dace.InterstateEdge())
    else:
        loop.add_edge(use, after, dace.InterstateEdge(assignments={"s": "i"}))
    return sdfg


def test_a_symbol_carried_on_every_iteration_is_rendered_as_a_proof():
    rendered = comment_lines(named(carried_symbol_sdfg("cpf_kind_symbol_carried", branch=False)))
    assert [line for line in rendered if line.startswith(f"// {PROVEN_PREFIX}")], rendered


def test_a_symbol_assigned_only_under_a_branch_proves_no_carried_dependence():
    """A branch that never runs hands nothing to the next iteration, so the carried symbol proves nothing there."""
    rendered = comment_lines(named(carried_symbol_sdfg("cpf_kind_symbol_branch", branch=True)))
    assert not [line for line in rendered if line.startswith(f"// {PROVEN_PREFIX}")], rendered


def test_a_while_loop_is_sequential_and_says_why():
    """A while loop has no iteration variable to split the work by, and each test of its condition reads what the
    previous iteration computed, so it is ``sequential`` with that reason, never an open ``unknown``."""
    rendered = comment_lines(named(unexamined_sdfg("cpf_kind_unexamined")))
    assert "// " + WHILE_LOOP in rendered, rendered
    assert not [line for line in rendered if line.startswith(f"// {PROVEN_PREFIX}")]


def test_a_wavefront_names_its_diagonal_and_its_front():
    """The skew is the only thing that knows an axis is a wavefront: afterwards it is an ordinary
    sequential loop over an ordinary map, and the generic classifier would say only that."""
    rendered = comment_lines(named(skewed_sdfg("cpf_kind_wavefront", tile=0)))
    # The skew that produced the wavefront is named, since the rewrite renames the axes to t/p and
    # a reader cannot recover it from the loop.
    diagonal = [line for line in rendered if line.startswith("// sequential -- wavefront step: ")]
    assert diagonal and diagonal[0].endswith("; alt: the unskewed nest"), rendered
    assert any(line.startswith("// parallel -- one wavefront step: ") for line in rendered)


def test_a_tiled_wavefront_names_all_three_of_its_axes():
    """The tiled lowering is four loops, and a reader has to be able to tell which of them is the
    parallel one -- the tile column, not the diagonal above it or the interior below."""
    rendered = comment_lines(named(skewed_sdfg("cpf_kind_wavefront_tiled", tile=32)))
    # Each axis names the tile it belongs to; the diagonal names the skew as well.
    assert any(line.startswith("// sequential -- wavefront of tiles [32x32]: ") for line in rendered)
    assert any(line.startswith("// parallel -- the tiles [32x32] of one wavefront step") for line in rendered)
    assert any(line.startswith("// sequential -- inside one wavefront tile [32x32]") for line in rendered)
    # The tiled form is a different trade from the untiled one and must not borrow its wording.
    assert not any(line.startswith("// sequential -- wavefront step") for line in rendered)


def test_naming_the_loops_does_not_move_a_line_of_code():
    """The whole feature is comments. Anything else -- an ``#pragma omp`` gained or lost, a bound
    rewritten -- would mean a note had turned into a decision.
    """

    def without_comments(code: str):
        return [line for line in code.splitlines() if not line.strip().startswith("//")]

    # The same SDFG rendered twice, so the entry point keeps its name and the only variable left
    # between the two texts is the annotation. The rendering names loops itself, so the baseline
    # pre-empts it: a hint already set is kept, and ``x`` names no kind.
    for build in (scaling_sdfg, carried_sdfg, breaking_sdfg):
        placeholder = build(f"cpf_kind_stable_{build.__name__}")
        for node, _ in placeholder.all_nodes_recursive():
            if isinstance(node, (nodes.MapEntry, LoopRegion)):
                node.specialization_hint = "x"
        before = render(placeholder).code
        after = named(build(f"cpf_kind_stable_{build.__name__}"))
        assert without_comments(before) == without_comments(after), build.__name__
        assert "// x" in comment_lines(before) and "// x" not in comment_lines(after), build.__name__
        assert any(" -- " in line for line in comment_lines(after)), build.__name__


def test_a_recorded_note_keeps_its_place_below_the_verdict():
    """A pass's note says what it did (an alternative it declined), not what the loop is: the classifier keeps it
    and puts the class first, so the reader still learns the verdict of every loop. A hint that already opens with
    a class is the pass's own verdict and stays as it is."""
    sdfg = carried_sdfg("cpf_kind_note")
    loops = loops_of(sdfg)
    loops[0].specialization_hint = MAP_HINT
    AnnotateLoopKinds().apply_pass(sdfg, {})
    lines = loops[0].specialization_hint.splitlines()
    assert lines[0].startswith(PROVEN_PREFIX), lines
    assert MAP_HINT.splitlines()[0] in lines, lines

    loops[0].specialization_hint = WAVEFRONT_FRONT.format(skew="(t = i + j)")
    AnnotateLoopKinds().apply_pass(sdfg, {})
    assert loops[0].specialization_hint == WAVEFRONT_FRONT.format(skew="(t = i + j)")


def test_the_canonicalize_pipeline_names_the_loops_it_leaves_behind():
    """The pass is only worth anything if the recipe runs it: an unwired pass leaves every test
    above green while no real rendering ever gains a comment.

    Also the end-to-end shape of the feature -- one graph, both answers. Canonicalization maps the
    column axis and cannot map the row axis, and the rendering now says which is which.
    """
    sdfg = carried_sdfg("cpf_kind_pipeline")
    canonicalize(sdfg)
    rendered = comment_lines(render(sdfg).code)
    assert "// " + PARALLEL.splitlines()[0] in rendered, rendered
    assert any(line.startswith(f"// {PROVEN_PREFIX}") for line in rendered)


# Loops born after canonicalize: specialization, library expansion, codegen copy lowering.


def test_a_map_a_library_expansion_leaves_is_labelled():
    """s311: a map a library expansion left carried the library description and no kind, unlike every other map.

    A copy of unknown size expands (``Auto``) to a parallel mapped tasklet; a Reduce now renders through its OpenMP
    reduction call and leaves no map (``test_reduce_lowering``)."""
    sdfg = dace.SDFG("cpf_kind_library_map")
    sdfg.add_array("src", [N], dace.float64)
    sdfg.add_array("dst", [N], dace.float64)
    state = sdfg.add_state("copy")
    copy = CopyLibraryNode("copy")
    state.add_node(copy)
    state.add_edge(state.add_read("src"), None, copy, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet("src[0:N]"))
    state.add_edge(copy, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, state.add_write("dst"), None, dace.Memlet("dst[0:N]"))
    sdfg.validate()
    rendered = comment_lines(render(sdfg).code)
    assert "// copy: write one buffer into another" in rendered, rendered
    assert "// " + PARALLEL.splitlines()[0] in rendered, rendered


def test_a_map_codegen_lowers_a_copy_into_is_labelled():
    """ext_war_unit's seam copy: codegen lowers the copy into a map after every pass has run."""
    sdfg = dace.SDFG("cpf_kind_copy")
    sdfg.add_array("src", [N], dace.float64)
    sdfg.add_array("dst", [N], dace.float64)
    state = sdfg.add_state("copy")
    state.add_nedge(state.add_read("src"), state.add_write("dst"), dace.Memlet("src[0:N:2] -> [0:N/2]"))
    sdfg.validate()
    code = render(sdfg).code
    lines = [line.strip() for line in code.splitlines()]
    pragma = next(index for index, line in enumerate(lines) if line.startswith("#pragma omp parallel for"))
    assert lines[pragma - 1] == "// " + PARALLEL, code


def test_a_loop_left_unlabelled_by_canonicalize_is_labelled_by_the_rendering():
    """StripMineInterchange and seam splitting add loops after ``AnnotateLoopKinds`` ran."""
    rendered = comment_lines(render(carried_sdfg("cpf_kind_late")).code)
    assert any(line.startswith(f"// {PROVEN_PREFIX} on b[") for line in rendered), rendered


def test_a_guarded_recurrence_is_not_proven_through_its_guard():
    """s275: a guarded column recurrence. Canonicalize moves the guard ``aa[0, i] > 0`` inside the sequential loop,
    so the recurrence's accesses sit under a branch the data decides, and an access a branch may skip proves no
    dependence: the loop stays unknown, never ``sequential -- carried``."""

    @dace.program
    def guarded_columns(aa: dace.float64[N, N], bb: dace.float64[N, N]):
        for i in range(N):
            if aa[0, i] > 0.0:
                for j in range(1, N):
                    aa[j, i] = aa[j - 1, i] + bb[j, i]

    sdfg = guarded_columns.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, validate_all=False, target="cpu")
    AnnotateLoopKinds().apply_pass(sdfg, {})
    hints = [loop.specialization_hint.splitlines() for loop in loops_of(sdfg)]
    assert len(hints) == 1, hints
    assert hints[0][0].startswith("unknown -- "), hints


def test_an_inner_loop_carried_dependence_is_proven_in_one_outer_iteration():
    """nussinov's ``j`` loop: ``t[i, j]`` reads ``t[i, j - 1]``. Every access names the outer ``i`` as well; the witness
    picks one outer iteration both accesses run in, instead of refusing ``i`` as an unknown symbol."""

    @dace.program
    def row_recurrence(t: dace.float64[N, N]):
        for i in range(N - 1, -1, -1):
            for j in range(i + 1, N):
                t[i, j] = t[i, j - 1] + t[i + 1, j]

    # Not canonicalized: canonicalization skews this nest into a wavefront, and the j loop is what is tested here.
    sdfg = row_recurrence.to_sdfg(simplify=True)
    AnnotateLoopKinds().apply_pass(sdfg, {})
    inner = [loop.specialization_hint.splitlines()[0] for loop in loops_of(sdfg) if loop.loop_variable == "j"]
    assert len(inner) == 1 and inner[0].startswith(f"{PROVEN_PREFIX} on t["), inner
    assert CONFLICT_MEANING["RAW"] in inner[0], inner
    # The solver's example, with the outer iteration it chose; the constant-distance proof names only j.
    assert "(example the solver found: i=" in inner[0], inner


def test_a_time_loop_over_a_stencil_is_proven_and_names_the_skew_it_is_not_given():
    """heat_3d: ``TSTEPS`` only bounds the time loop, so a witness may choose it like an array size. The loop carries
    a dependence, but a skew with the parallel nest inside would parallelize it, so the label says so."""
    T = dace.symbol("T")

    @dace.program
    def heat(A: dace.float64[N], B: dace.float64[N]):
        for t in range(1, T + 1):
            B[1:-1] = 0.25 * (A[:-2] + 2.0 * A[1:-1] + A[2:])
            A[1:-1] = 0.25 * (B[:-2] + 2.0 * B[1:-1] + B[2:])

    sdfg = heat.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, validate_all=False, target="cpu")
    AnnotateLoopKinds().apply_pass(sdfg, {})
    hints = [loop.specialization_hint.splitlines() for loop in loops_of(sdfg)]
    assert len(hints) == 1, hints
    assert hints[0][0].startswith(f"{PROVEN_PREFIX} on ") and "T=" in hints[0][0], hints
    # The alternative says what was not done and why, and does not contradict itself (review #17)
    assert CONFLICT_MEANING["RAW"] in hints[0][0], hints
    assert hints[0][0].endswith(
        "alt: a wavefront that skews this loop with the parallel loops inside it, not applied because those loops "
        "are parallel already at each iteration of this one"
    ), hints


def test_a_read_ahead_dependence_is_named_war_not_raw():
    """ext_war_unit: ``a[i] = a[i + 1] + ...`` reads what a LATER iteration writes."""

    @dace.program
    def ahead(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N - 1):
            a[i] = a[i + 1] + b[i]

    sdfg = ahead.to_sdfg(simplify=True)
    AnnotateLoopKinds().apply_pass(sdfg, {})
    (hint,) = [loop.specialization_hint for loop in loops_of(sdfg)]
    first_line = hint.splitlines()[0]
    reason = re.fullmatch(
        r"(on a\[i \+ 1\]: (.*) \(example the solver found: N=(\d+), iterations i=(\d+) and i=(\d+)\))",
        first_line.removeprefix(PROVEN_PREFIX + " "),
    )
    assert reason, hint
    assert reason.group(2) == CONFLICT_MEANING["WAR"], hint
    assert first_line == SEQUENTIAL_PROVEN.format(reason=reason.group(1))
    # The example is an execution: the earlier iteration reads the element the next one overwrites
    size, first, second = (int(g) for g in reason.groups()[2:])
    assert second == first + 1 and second < size - 1, hint


def test_an_offset_that_bounds_the_loop_is_named_in_the_witness():
    """An offset ``K`` that sizes nothing may not be chosen to make a conflict, but here every ``K`` under which the
    loop runs twice conflicts, so the solver proves the carry. ``K`` bounds the loop, so the example names its value."""
    K = dace.symbol("K", dtype=dace.int64)

    @dace.program
    def offset_recurrence(a: dace.float64[N], b: dace.float64[N]):
        for i in range(1, N - K):
            a[i + K] = a[i + K - 1] + b[i]

    sdfg = offset_recurrence.to_sdfg(simplify=True)
    AnnotateLoopKinds().apply_pass(sdfg, {})
    (hint,) = [loop.specialization_hint for loop in loops_of(sdfg)]
    first_line = hint.splitlines()[0]
    reason = re.fullmatch(
        r"(on a\[(.*)\]: (.*) \(example the solver found: N=(\d+), K=(-?\d+), iterations i=(\d+) and i=(\d+)\))",
        first_line.removeprefix(PROVEN_PREFIX + " "),
    )
    assert reason, hint
    assert reason.group(3) == CONFLICT_MEANING["RAW"]
    assert first_line == SEQUENTIAL_PROVEN.format(reason=reason.group(1))
    # The example is an execution: both iterations run, one after the other, for that K
    size, offset, first, second = (int(g) for g in reason.groups()[3:])
    assert 1 <= first and second == first + 1 < size - offset, hint


def test_no_carried_dependence_is_claimed_for_independent_elements():
    """The element proof needs one write and one read ``d != 0`` iterations apart, nothing less."""

    @dace.program
    def offset_copy(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N):
            b[i] = a[i] + b[i]

    loops = loops_of(offset_copy.to_sdfg(simplify=True))
    assert loops and all(proven_carrying_access(loop) is None for loop in loops)


def test_a_pinned_independent_loop_does_not_read_as_parallel():
    """versioned_distance_update: the K == 0 arm is pinned, rendered without a pragma, and was
    labelled parallel."""

    @dace.program
    def independent(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N):
            b[i] = a[i] * 2.0

    sdfg = independent.to_sdfg(simplify=True)
    loops = loops_of(sdfg)
    assert loops, "the frontend mapped the loop, so there is no pinned loop to label"
    for loop in loops:
        loop.pinned_sequential = True
    code = named(sdfg)
    assert "// " + SEQUENTIAL_PINNED.splitlines()[0] in [line.strip() for line in code.splitlines()], code
    assert "#pragma omp" not in code
    assert not any(line.startswith("// parallel --") for line in comment_lines(code))


def overlapping_writes_sdfg(name: str) -> dace.SDFG:
    """Writes ``a[5*i]`` and ``a[3*i]``: ``LoopToMap`` cannot separate the iterations, and no constant distance
    relates them, but ``i = 3`` and ``i = 5`` write the same element once ``N >= 6``."""

    @dace.program
    def two_strides(a: dace.float64[5 * N], b: dace.float64[N]):
        for i in range(N):
            a[5 * i] = b[i]
            a[3 * i] = b[i] + 1.0

    sdfg = two_strides.to_sdfg(simplify=True)
    sdfg.name = name
    return sdfg


def test_writes_the_test_cannot_separate_are_proven_by_a_witness():
    """``LoopToMap`` refuses ``a[5*i]`` against ``a[3*i]`` because they MAY overlap. The solver finds two iterations
    that DO write one element, for an array size it chooses, so the loop is sequential and the hint says why."""
    rendered = comment_lines(named(overlapping_writes_sdfg("cpf_kind_overlap")))
    carried = [line for line in rendered if line.startswith(f"// {PROVEN_PREFIX} on a[")]
    assert len(carried) == 1 and CONFLICT_MEANING["WAW"] in carried[0], rendered
    witness = re.search(r"\(example the solver found: N=(\d+), iterations i=(\d+) and i=(\d+)\)$", carried[0])
    assert witness, carried
    size, first, second = (int(g) for g in witness.groups())
    assert first < second < size, carried
    assert {5 * first, 3 * first} & {5 * second, 3 * second}, carried


def test_a_distance_set_by_an_argument_stays_unknown():
    """``a[i] = a[i - k]`` with ``k`` an argument: ``k`` is fixed but unknown, and ``k = 0`` carries nothing, so a
    witness that picks ``k = 1`` proves nothing about the program."""

    @dace.program
    def shift_by(a: dace.float64[N], k: dace.int64):
        for i in range(N):
            a[i] = a[i - k] + 1.0

    rendered = comment_lines(named(shift_by.to_sdfg(simplify=True)))
    assert [line for line in rendered if line.startswith("// unknown -- ")], rendered
    assert not [line for line in rendered if line.startswith(f"// {PROVEN_PREFIX}")], rendered


def test_a_conflict_only_under_a_branch_is_not_witnessed():
    """A branch body may never run, so an access inside one cannot be half of a witness."""

    @dace.program
    def guarded_stride(a: dace.float64[5 * N], b: dace.float64[N]):
        for i in range(N):
            a[5 * i] = b[i]
            if b[i] > 0.0:
                a[3 * i] = b[i] + 1.0

    rendered = comment_lines(named(guarded_stride.to_sdfg(simplify=True)))
    assert not [line for line in rendered if line.startswith(f"// {PROVEN_PREFIX}")], rendered


def test_a_triangular_solve_is_proven_through_its_row_copy():
    """trisolv: each row reads ``x[0:i]``, which the earlier iterations wrote. The read is the inner product's operand
    subset, every element of which the product reads, and the witness names that range."""

    @dace.program
    def trisolv(L: dace.float64[N, N], x: dace.float64[N], b: dace.float64[N]):
        for i in range(N):
            x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]

    sdfg = trisolv.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, validate_all=False, target="cpu")
    AnnotateLoopKinds().apply_pass(sdfg, {})
    hints = [loop.specialization_hint.splitlines()[0] for loop in loops_of(sdfg)]
    assert len(hints) == 1, hints
    assert re.match(
        rf"{PROVEN_PREFIX} on x\[0:\w+\]: .* \(example the solver found: N=\d+, iterations \w+=0 and \w+=1\)$", hints[0]
    ), hints


def test_a_guarded_loop_names_its_guard_on_both_arms():
    """s171: ``a[i*inc]`` is injective, so the loop is parallel, exactly when ``inc != 0``. The parallel arm says it
    is parallel under that guard, and the sequential arm says it is the fallback for when the guard fails."""
    inc = dace.symbol("inc")

    @dace.program
    def s171(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N):
            a[i * inc] = a[i * inc] + b[i]

    sdfg = s171.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, validate_all=False, target="cpu")
    maps = [
        n.specialization_hint.splitlines()[0]
        for n, parent in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry)
    ]
    loops = [loop.specialization_hint.splitlines()[0] for loop in loops_of(sdfg)]
    assert any(
        re.match(r"parallel -- the runtime check found .*inc.*, so no iteration writes", line) for line in maps
    ), maps
    assert any(
        re.match(r"sequential -- the runtime check found .*inc.*: .*, so this arm keeps the order", line)
        for line in loops
    ), loops
    # The fallback names the failing guard itself, never a negation of a negation
    assert not any("not (not" in line for line in loops), loops


def test_a_map_resting_on_an_unknown_sign_is_versioned_on_it():
    """``a[i] = a[i + K] + b[i]`` is parallel after the snapshot only for ``K >= 0``. Nothing declares ``K``
    nonnegative, so canonicalization versions the loop on the sign instead of assuming it: ``K < 0`` keeps a version
    of its own, and the parallel map runs in the arm after it, whose guard facts prove ``0 <= K``, so no sequential
    fallback is left."""

    @dace.program
    def shift_sym(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N - K):
            a[i] = a[i + K] + b[i]

    sdfg = shift_sym.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, validate_all=False, target="cpu")
    AnnotateLoopKinds().apply_pass(sdfg, {})
    maps = [
        n.specialization_hint.splitlines()[0]
        for n, parent in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry)
    ]
    assert PARALLEL in maps, maps
    branches = [
        cond.as_string
        for region in sdfg.all_control_flow_regions()
        if isinstance(region, ConditionalBlock)
        for cond, _ in region.branches
        if cond is not None
    ]
    assert branches == ["((- K) >= 1)"], branches
    assert not loops_of(sdfg), [loop.label for loop in loops_of(sdfg)]


def test_a_map_resting_on_no_trap_assumes_nothing():
    """A trap on a symbol the map never names is not part of its evidence."""
    rendered = comment_lines(named(scaling_sdfg("cpf_kind_no_assumption")))
    assert "// " + PARALLEL.splitlines()[0] in rendered


def accumulating_loop_sdfg(name: str) -> dace.SDFG:
    """vexx_k's shape: each ``i`` writes the scalar ``s``, then a map over ``j`` adds into ``tmp[j]`` through WCR."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [K], dace.float64)
    sdfg.add_array("tmp", [K], dace.float64)
    sdfg.add_scalar("s", dace.float64, transient=True)
    loop = LoopRegion("rows", "i < N", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    scale = body.add_tasklet("scale", {"x"}, {"y"}, "y = 2.0 * x")
    s = body.add_access("s")
    body.add_edge(body.add_read("a"), None, scale, "x", dace.Memlet("a[i]"))
    body.add_edge(scale, "y", s, None, dace.Memlet("s[0]"))
    body.add_mapped_tasklet(
        "acc",
        {"j": "0:K"},
        {"u": dace.Memlet("s[0]"), "v": dace.Memlet("b[j]")},
        "w = u * v",
        {"w": dace.Memlet("tmp[j]", wcr="lambda p, q: p + q")},
        input_nodes={"s": s},
        external_edges=True,
    )
    sdfg.validate()
    return sdfg


def test_a_loop_whose_iterations_accumulate_into_one_element_is_a_reduction_with_its_private_scalar():
    """vexx_k: every ``i`` adds into the same ``tmp[j]`` and writes the function-scope scalar ``s``. "independent"
    tells an agent a plain ``omp parallel for`` is safe; it races on both."""
    sdfg = accumulating_loop_sdfg("cpf_kind_loop_reduction")
    AnnotateLoopKinds().apply_pass(sdfg, {})
    (loop,) = loops_of(sdfg)
    assert loop.specialization_hint == REDUCTION.format(reductions="+ into tmp") + PRIVATE.format(names="s")


def test_a_map_whose_wcr_writes_distinct_elements_is_independent():
    """vexx_k's inner map writes ``tmp[j]``, one element per ``j``: the WCR is for the enclosing loop, the map's own
    iterations do not reduce."""
    sdfg = accumulating_loop_sdfg("cpf_kind_map_distinct_wcr")
    AnnotateLoopKinds().apply_pass(sdfg, {})
    (entry,) = map_entries(sdfg)
    assert entry.specialization_hint == PARALLEL


def test_a_map_whose_wcr_index_mixes_two_parameters_stays_a_reduction():
    """vexx_k's ``hpsi[j, i*npwx + k]``: ``i`` and ``k`` share one index, so two iterations can meet when ``k``
    runs past ``npwx``; only an index of its own per parameter proves the elements distinct."""
    sdfg = dace.SDFG("cpf_kind_map_mixed_wcr")
    sdfg.add_array("a", [N, K], dace.float64)
    sdfg.add_array("out", [N * K], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "acc",
        {"i": "0:N", "k": "0:K"},
        {"x": dace.Memlet("a[i, k]")},
        "y = x",
        {"y": dace.Memlet("out[i * N + k]", wcr="lambda p, q: p + q")},
        external_edges=True,
    )
    (entry,) = map_entries(sdfg)
    assert parallel_hint(state, entry, []) == REDUCTION.format(reductions="+ into out")


def lifted_buffer_sdfg(name: str) -> dace.SDFG:
    """A loop calling a nested SDFG whose persistent ``buf`` each iteration fills and reads."""
    inner = dace.SDFG("inner")
    inner.add_symbol("row", dace.int64)
    inner.add_array("A", [N, K], dace.float64)
    inner.add_array("B", [N, K], dace.float64)
    inner.add_array("buf", [K], dace.float64, transient=True, lifetime=dace.AllocationLifetime.Persistent)
    fill = inner.add_state("fill", is_start_block=True)
    fill.add_mapped_tasklet(
        "w",
        {"k": "0:K"},
        {"x": dace.Memlet("A[row, k]")},
        "y = 2.0 * x",
        {"y": dace.Memlet("buf[k]")},
        external_edges=True,
    )
    use = inner.add_state_after(fill)
    use.add_mapped_tasklet(
        "r",
        {"k": "0:K"},
        {"x": dace.Memlet("buf[k]")},
        "y = x + 1.0",
        {"y": dace.Memlet("B[row, k]")},
        external_edges=True,
    )
    sdfg = dace.SDFG(name)
    sdfg.add_array("x", [N, K], dace.float64)
    sdfg.add_array("y", [N, K], dace.float64)
    loop = LoopRegion("rows", "i < N", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    call = body.add_nested_sdfg(inner, {"A"}, {"B"}, {"N": "N", "K": "K", "row": "i"})
    body.add_edge(body.add_read("x"), None, call, "A", dace.Memlet("x[0:N, 0:K]"))
    body.add_edge(call, "B", body.add_write("y"), None, dace.Memlet("y[0:N, 0:K]"))
    sdfg.validate()
    return sdfg


def test_a_loop_whose_buffer_the_rendering_lifts_names_it_private():
    """The rendering lifts a nested function's persistent buffer before the loop, where every iteration shares it:
    the loop is labelled after the lift, and says the buffer needs a copy per iteration."""
    rendered = comment_lines(render(lifted_buffer_sdfg("cpf_kind_lifted")).code)
    assert "// " + PARALLEL + PRIVATE.format(names="buf") in rendered, rendered


def trapped_sdfg(name: str, inner_symbol: str, outer_value: str) -> dace.SDFG:
    """An entry trap ``nkb > 0`` and a nested SDFG whose map reads ``d[<inner_symbol>]``, bound to ``outer_value``."""
    inner = dace.SDFG("inner_trap")
    inner.add_symbol(inner_symbol, dace.int64)
    inner.add_array("c", [K], dace.float64)
    inner.add_array("d", [K], dace.float64)
    inner.add_state().add_mapped_tasklet(
        "w",
        {"j": "0:K"},
        {"x": dace.Memlet(f"d[{inner_symbol}]")},
        "y = x",
        {"y": dace.Memlet("c[j]")},
        external_edges=True,
    )
    sdfg = dace.SDFG(name)
    sdfg.add_symbol("nkb", dace.int64)
    sdfg.add_array("c", [K], dace.float64)
    sdfg.add_array("d", [K], dace.float64)
    guard = sdfg.add_state("guard", is_start_block=True)
    tasklet_utils.add_abort_guard(guard, "check", "nkb <= 0")
    state = sdfg.add_state_after(guard)
    call = state.add_nested_sdfg(inner, {"d"}, {"c"}, {inner_symbol: outer_value, "K": "K"})
    state.add_edge(state.add_read("d"), None, call, "d", dace.Memlet("d[0:K]"))
    state.add_edge(call, "c", state.add_write("c"), None, dace.Memlet("c[0:K]"))
    sdfg.validate()
    return sdfg


def test_an_entry_trap_reaches_a_nested_function_in_its_own_names():
    """vexx_k: "independent only if nkb > 0" inside a function that has no ``nkb``. A trap enters a nested SDFG
    through its symbol mapping, renamed, and is dropped where the inner name means something else."""
    for inner_symbol, outer_value, expected in (("n", "nkb", "n > 0"), ("nkb", "K", None)):
        sdfg = trapped_sdfg(f"cpf_kind_trap_{inner_symbol}", inner_symbol, outer_value)
        AnnotateLoopKinds().apply_pass(sdfg, {})
        (hint,) = [n.specialization_hint for n, parent in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
        if expected is None:
            assert hint == PARALLEL, hint
        else:
            assert hint.endswith(f"; independent only if {expected} (the program aborts at entry otherwise)"), hint


def test_a_solver_example_writes_an_inner_iterator_as_its_range():
    """fdtd: the example named ``ey[j, __i1]``, an inner map's parameter the rendering renamed; the range it covers
    does not depend on the name."""

    @dace.program
    def ping(a: dace.float64[N], b: dace.float64[N]):
        for t in range(K):
            for i in dace.map[0:N]:
                a[i] = b[i] + 1.0
            for i in dace.map[1:N]:
                b[i] = a[i - 1]

    (loop,) = loops_of(ping.to_sdfg(simplify=True))
    found = witnessed_carrying_access(loop)
    assert found is not None and re.match(r"(RAW|WAR|WAW) on [ab]\[\d+:N( - 1)?\] ", found), found


def test_a_wavefront_names_its_skew_on_the_diagonal_and_the_front():
    """wf_north_west: the skew labels named ``_loop_it_0`` and a placeholder ``w`` the rendering never declares; each
    counter a wavefront label names is a loop counter of the rendered code."""
    rendered_code = named(skewed_sdfg("cpf_kind_wavefront_skew", tile=0))
    rendered = comment_lines(rendered_code)
    diagonal = [
        re.search(r"step (\w+) runs the points with row \+ column = \1 of the original nest", line) for line in rendered
    ]
    front = [re.search(r"(\w+) walks the (?:column|row) of step (\w+)", line) for line in rendered]
    counters = [m.group(1) for m in diagonal if m] + [name for m in front if m for name in m.groups()]
    assert len([m for m in diagonal if m]) >= 1 and len([m for m in front if m]) >= 1, rendered
    code = "\n".join(line for line in rendered_code.splitlines() if not line.strip().startswith("//"))
    assert all(re.search(rf"\bfor \([^;]*\b{name}\b", code) for name in counters), (counters, rendered_code)


def test_a_thread_team_is_not_labelled_a_loop_of_independent_iterations():
    """A ``CPU_Persistent`` map is one ``#pragma omp parallel`` around a sequential time loop, not a loop."""
    state = dace.SDFG("cpf_kind_team").add_state()
    entry, team_exit = state.add_map("team", {"t": "0:1"})
    entry.map.schedule = dace.ScheduleType.CPU_Persistent
    assert parallel_hint(state, entry, []) == TEAM


def kinds_and_proofs(name: str) -> tuple[str, str]:
    """One canonicalized graph -- a parallel axis, a carried one, an open one -- rendered with ``hints='kinds'`` and
    with ``hints='proofs'``."""

    @dace.program
    def three_kinds(a: dace.float64[N, N], c: dace.float64[N], k: dace.int64):
        for i in range(1, N):
            for j in range(N):
                a[i, j] = a[i - 1, j] + 1.0
        for i in range(N):
            c[i] = c[i - k] + 1.0

    sdfg = three_kinds.to_sdfg(simplify=True)
    sdfg.name = name
    canonicalize(sdfg)
    return render(sdfg, hints="kinds").code, render(sdfg, hints="proofs").code


def comment_blocks(code: str) -> list[list[str]]:
    """The runs of consecutive comment lines of ``code``, each without its ``// ``."""
    blocks: list[list[str]] = [[]]
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            blocks[-1].append(stripped[2:].strip())
        elif blocks[-1]:
            blocks.append([])
    return [block for block in blocks if block]


def test_the_kinds_rendering_says_its_class_words_and_nothing_else():
    """The light variant: per loop and map its class -- a ``sequential`` or ``unknown`` loop with its reason, a
    ``parallel`` one with the word alone -- and no witness, no alternative, no header contract, no library-node
    provenance."""
    kinds, proofs = (comment_lines(code)[1:] for code in kinds_and_proofs("cpf_kinds_only"))
    known = set(dict(cpf_lowering.KIND_WORDS).values())
    words = [line for line in kinds if line.removeprefix("// ").split(":", 1)[0] in known]
    classes = {line.removeprefix("// ").split(":", 1)[0] for line in words}
    assert {"parallel", "sequential"} <= classes, kinds
    assert all(":" not in line for line in words if line.removeprefix("// ").startswith("parallel")), kinds
    assert all(": " in line for line in words if line.removeprefix("// ").startswith("sequential")), kinds
    # What is left is the unit's own section labels, which the proofs rendering carries word for word.
    assert all(line in proofs for line in kinds if line not in words), kinds
    extras = ("--", "witness", "settled", "open:", "example", "alt:", "CPU:", "GPU:")
    assert not [line for line in kinds if any(term in line for term in extras)], kinds
    assert any("example" in line for line in proofs), proofs


def test_no_rendering_carries_a_tasklet_or_scope_label():
    """A tasklet's label (``_Add_``, ``for_696_scan_op``) is a DaCe name, not a verdict: neither rendering writes it
    after the statement or the brace."""
    kinds, proofs = kinds_and_proofs("cpf_kinds_no_label")
    trailing = lambda code: [
        line
        for line in code.splitlines()
        if "//" in line and line.split("//")[0].strip() and not line.lstrip().startswith("#")
    ]
    assert not trailing(kinds), trailing(kinds)
    assert not trailing(proofs), trailing(proofs)


def test_a_rendering_asked_for_tasklet_labels_names_each_statements_tasklet_and_keeps_the_code():
    """``tasklet_labels=True`` is for a tool that maps the unit's lines back to tasklets: each statement ends with
    its tasklet's label, and the code is the same."""

    @dace.program
    def scaled(a: dace.float64[N], b: dace.float64[N]):
        for i in dace.map[0:N]:
            b[i] = a[i] * 2.0

    sdfg = scaled.to_sdfg(simplify=True)
    sdfg.name = "cpf_tasklet_labels"
    canonicalize(sdfg)
    plain, labeled = (render(sdfg, tasklet_labels=on).code for on in (False, True))

    labels = {n.label for n, parent in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)}
    assert any(line.rstrip().endswith(f"// {label}") for line in labeled.splitlines() for label in labels), labeled
    strip = lambda code: [
        line.split("  // ")[0].rstrip() if line.split("//")[0].strip() else line for line in code.splitlines()
    ]
    assert strip(labeled) == strip(plain)


def test_the_kinds_rendering_names_the_loops_the_proofs_rendering_classes():
    """Same graph, same verdicts: each classed comment block of the proofs rendering is one word of the kinds
    rendering, the class of the block's first classed line, in the same order."""
    kinds, proofs = kinds_and_proofs("cpf_kinds_match")
    classes = [next(filter(None, map(cpf_lowering.kind_word, block)), None) for block in comment_blocks(proofs)]
    known = set(dict(cpf_lowering.KIND_WORDS).values())
    words = [
        word.split(":", 1)[0] for block in comment_blocks(kinds) for word in block if word.split(":", 1)[0] in known
    ]
    assert words == [word for word in classes if word], (words, classes)


def test_the_two_variants_render_the_same_code():
    sdfg = carried_sdfg("cpf_kinds_code")
    AnnotateLoopKinds().apply_pass(sdfg, {})
    # Comments aside, whole-line and trailing alike
    strip = lambda code: [line.split("  //")[0] for line in code.splitlines() if not line.strip().startswith("//")]
    assert strip(render(sdfg, hints="kinds").code) == strip(render(sdfg, hints="proofs").code)


def test_the_none_rendering_has_no_loop_comment_and_the_same_code():
    """The comments-ablation variant: every comment line of the ``none`` rendering is a section label the kinds
    rendering carries too (no class word, no reasoning, no tasklet label), and the code is that of ``proofs``."""

    @dace.program
    def three_kinds(a: dace.float64[N, N], c: dace.float64[N], k: dace.int64):
        for i in range(1, N):
            for j in range(N):
                a[i, j] = a[i - 1, j] + 1.0
        for i in range(N):
            c[i] = c[i - k] + 1.0

    sdfg = three_kinds.to_sdfg(simplify=True)
    sdfg.name = "cpf_hints_none"
    canonicalize(sdfg)
    none, kinds, proofs = (render(sdfg, hints=detail).code for detail in ("none", "kinds", "proofs"))
    known = set(dict(cpf_lowering.KIND_WORDS).values())
    is_class = lambda line: line.removeprefix("// ").split(":", 1)[0] in known
    assert not [line for line in comment_lines(none) if is_class(line)], none
    assert [line for line in comment_lines(kinds) if not is_class(line)] == comment_lines(none), none
    assert not [line for line in none.splitlines() if "//" in line and line.split("//")[0].strip()], none
    strip = lambda code: [line.split("  //")[0] for line in code.splitlines() if not line.strip().startswith("//")]
    assert strip(none) == strip(proofs)


def test_an_unknown_hint_detail_is_refused():
    with pytest.raises(ValueError):
        render(scaling_sdfg("cpf_kinds_refused"), hints="some")


if __name__ == "__main__":
    test_the_rendering_states_which_verdicts_are_settled_and_which_are_open()
    test_a_map_hint_is_rendered_one_comment_line_per_line()
    test_a_while_loop_header_reads_only_its_condition()
    test_a_loop_hint_is_rendered()
    test_a_scan_states_its_trade_although_the_expansion_consumes_the_node()
    test_a_scan_hint_follows_the_expansion_that_ran("pure", SEQUENTIAL_SCAN_HINT)
    test_a_scan_hint_follows_the_expansion_that_ran("CPU", PARALLEL_SCAN_HINT)
    test_no_hint_and_an_empty_hint_render_identically()
    for hint in (None, "", "   \n\t"):
        test_hint_comment_is_empty_for_an_empty_hint(hint)
    test_hint_comment_is_empty_outside_a_standalone_rendering()
    test_ordinary_codegen_carries_no_hint()
    test_a_parallel_map_says_it_is_parallel()
    test_a_multi_dimensional_map_names_every_loop_it_renders_as()
    test_a_proven_carried_dependence_is_rendered_as_a_proof_and_names_it()
    test_an_early_exit_is_sequential_with_its_cause_and_never_a_proven_dependence()
    test_an_access_under_a_data_dependent_branch_proves_no_carried_dependence()
    test_a_while_loop_is_sequential_and_says_why()
    test_a_wavefront_names_its_diagonal_and_its_front()
    test_a_tiled_wavefront_names_all_three_of_its_axes()
    test_naming_the_loops_does_not_move_a_line_of_code()
    test_a_recorded_note_keeps_its_place_below_the_verdict()
    test_the_canonicalize_pipeline_names_the_loops_it_leaves_behind()
    test_a_map_a_library_expansion_leaves_is_labelled()
    test_a_map_codegen_lowers_a_copy_into_is_labelled()
    test_a_loop_left_unlabelled_by_canonicalize_is_labelled_by_the_rendering()
    test_a_guarded_recurrence_is_not_proven_through_its_guard()
    test_an_inner_loop_carried_dependence_is_proven_in_one_outer_iteration()
    test_a_time_loop_over_a_stencil_is_proven_and_names_the_skew_it_is_not_given()
    test_a_read_ahead_dependence_is_named_war_not_raw()
    test_an_offset_that_bounds_the_loop_is_named_in_the_witness()
    test_no_carried_dependence_is_claimed_for_independent_elements()
    test_a_pinned_independent_loop_does_not_read_as_parallel()
    test_writes_the_test_cannot_separate_are_proven_by_a_witness()
    test_a_distance_set_by_an_argument_stays_unknown()
    test_a_conflict_only_under_a_branch_is_not_witnessed()
    test_a_triangular_solve_is_proven_through_its_row_copy()
    test_a_guarded_loop_names_its_guard_on_both_arms()
    test_a_map_resting_on_an_unknown_sign_is_versioned_on_it()
    test_a_map_resting_on_no_trap_assumes_nothing()
    test_the_kinds_rendering_says_its_class_words_and_nothing_else()
    test_no_rendering_carries_a_tasklet_or_scope_label()
    test_a_rendering_asked_for_tasklet_labels_names_each_statements_tasklet_and_keeps_the_code()
    test_the_kinds_rendering_names_the_loops_the_proofs_rendering_classes()
    test_the_two_variants_render_the_same_code()
    test_an_unknown_hint_detail_is_refused()
    test_a_loop_whose_iterations_accumulate_into_one_element_is_a_reduction_with_its_private_scalar()
    test_a_map_whose_wcr_writes_distinct_elements_is_independent()
    test_a_map_whose_wcr_index_mixes_two_parameters_stays_a_reduction()
    test_a_loop_whose_buffer_the_rendering_lifts_names_it_private()
    test_an_entry_trap_reaches_a_nested_function_in_its_own_names()
    test_a_solver_example_writes_an_inner_iterator_as_its_range()
    test_a_wavefront_names_its_skew_on_the_diagonal_and_the_front()
    test_a_thread_team_is_not_labelled_a_loop_of_independent_iterations()
