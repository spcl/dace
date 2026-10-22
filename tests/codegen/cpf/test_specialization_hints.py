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
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.passes.canonicalize.annotate_loop_kinds import (
    CONFLICT_MEANING,
    MAP_DIMENSIONS,
    NO_COUNTER,
    PARALLEL,
    SEQUENTIAL_PINNED,
    SEQUENTIAL_PROVEN,
    UNSURE,
    WAVEFRONT_FRONT,
    AnnotateLoopKinds,
    proven_carrying_access,
)
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.canonicalize.wavefront_skew import WavefrontSkew

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
    assert lines[1 : 1 + len(CONTRACT_LINES)] == CONTRACT_LINES


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
    assert "// parallel scan; canonicalization takes the parallel form." in rendered
    assert "// Alternative: a sequential loop over parallel maps." in rendered
    # Folded into the node's description so one expansion is one trade, not one per map it leaves.
    assert rendered.count("// Alternative: a sequential loop over parallel maps.") == 1
    assert not [line for line in rendered if line.startswith("// sequential scan")]


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


# Naming what every loop is: parallel, sequential (carried or pinned), unsure, and the wavefront roles.


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
    proofs = [line for line in rendered if line.startswith("// sequential -- carried:")]
    assert proofs, rendered
    # The hint carries the ACCESS that carries the dependence, which is what a reader acts on.
    assert "b[" in proofs[0], proofs
    # The row axis is proven; nothing here is merely unsure.
    assert not [line for line in rendered if line.startswith("// unsure")]


def test_a_declined_dependence_test_is_not_rendered_as_a_proof():
    """The weak half, and the reason the two are separate hints at all.

    ``LoopToMap`` refuses a loop with a ``break`` at the control-flow check, long before it asks
    anything about data. Calling that a carried dependence would be a claim the pipeline never
    made, and would send a reader looking for a recurrence that is not there.
    """
    rendered = comment_lines(named(breaking_sdfg("cpf_kind_unsure")))
    unsure = [line for line in rendered if line.startswith("// unsure -- ")]
    assert unsure, rendered
    # The refusal names WHERE the test stopped, which is what separates unsure from proven.
    assert "Break" in unsure[0], unsure
    # "carried" is the proven wording, and an unsure loop must never borrow it.
    assert not [line for line in rendered if line.startswith("// sequential -- carried:")]


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
    assert not [line for line in rendered if line.startswith("// sequential -- carried:")], rendered


def test_a_loop_that_was_never_examined_says_exactly_that():
    """A while loop has no iteration variable, so nothing ever asked whether it carries a dependence.
    That is as open as a declined test, so it is ``unsure`` too, with the reason saying why."""
    rendered = comment_lines(named(unexamined_sdfg("cpf_kind_unexamined")))
    assert "// " + UNSURE.format(reason=NO_COUNTER).splitlines()[0] in rendered, rendered
    assert not [line for line in rendered if line.startswith("// sequential -- carried:")]


def test_a_wavefront_names_its_diagonal_and_its_front():
    """The skew is the only thing that knows an axis is a wavefront: afterwards it is an ordinary
    sequential loop over an ordinary map, and the generic classifier would say only that."""
    rendered = comment_lines(named(skewed_sdfg("cpf_kind_wavefront", tile=0)))
    # The skew that produced the wavefront is named, since the rewrite renames the axes to t/p and
    # a reader cannot recover it from the loop.
    diagonal = [line for line in rendered if line.startswith("// sequential -- wavefront diagonal (t = ")]
    assert diagonal and diagonal[0].endswith(": the skew put every dependence on this axis"), rendered
    assert any(line.startswith("// parallel -- wavefront front (t = ") for line in rendered)
    # A wavefront is a trade like any other, so it states the alternative and the device it pays on.
    assert any(line.startswith("// alternative: the unskewed nest") for line in rendered)


def test_a_tiled_wavefront_names_all_three_of_its_axes():
    """The tiled lowering is four loops, and a reader has to be able to tell which of them is the
    parallel one -- the tile column, not the diagonal above it or the interior below."""
    rendered = comment_lines(named(skewed_sdfg("cpf_kind_wavefront_tiled", tile=32)))
    # Each axis names the tile it belongs to; the diagonal names the skew as well.
    assert any(
        line.startswith("// sequential -- wavefront tile diagonal (t = ") and "[32x32]" in line for line in rendered
    )
    assert any(line.startswith("// parallel -- wavefront tile column [32x32]") for line in rendered)
    assert any(line.startswith("// sequential -- inner tile of a wavefront [32x32]") for line in rendered)
    # The tiled form is a different trade from the untiled one and must not borrow its wording.
    assert not any(line.startswith("// sequential -- wavefront diagonal") for line in rendered)


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
    assert lines[0].startswith("sequential -- carried:"), lines
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
    assert any(line.startswith("// sequential -- carried:") for line in rendered)


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
    assert lines[pragma - 2 : pragma] == ["// " + line for line in PARALLEL.splitlines()], code


def test_a_loop_left_unlabelled_by_canonicalize_is_labelled_by_the_rendering():
    """StripMineInterchange and seam splitting add loops after ``AnnotateLoopKinds`` ran."""
    rendered = comment_lines(render(carried_sdfg("cpf_kind_late")).code)
    assert any(line.startswith("// sequential -- carried: RAW on b[") for line in rendered), rendered


def test_a_guarded_recurrence_is_not_proven_through_its_guard():
    """s275: a guarded column recurrence. Canonicalize moves the guard ``aa[0, i] > 0`` inside the sequential loop,
    so the recurrence's accesses sit under a branch the data decides, and an access a branch may skip proves no
    dependence: the loop stays unsure, never ``sequential -- carried``."""

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
    assert hints[0][0].startswith("unsure -- "), hints


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
    assert len(inner) == 1 and inner[0].startswith("sequential -- carried: RAW on t["), inner
    # The solver's witness, with the outer iteration it chose; the constant-distance proof names only j.
    assert "(witness: i=" in inner[0], inner


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
    assert hints[0][0].startswith("sequential -- carried: RAW on ") and "T=" in hints[0][0], hints
    assert "a skew of this loop with the parallel nest inside it" in hints[0][1], hints


def test_a_read_ahead_dependence_is_named_war_not_raw():
    """ext_war_unit: ``a[i] = a[i + 1] + ...`` reads what a LATER iteration writes."""

    @dace.program
    def ahead(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N - 1):
            a[i] = a[i + 1] + b[i]

    sdfg = ahead.to_sdfg(simplify=True)
    AnnotateLoopKinds().apply_pass(sdfg, {})
    (hint,) = [loop.specialization_hint for loop in loops_of(sdfg)]
    reason = re.search(r"carried: (WAR on a\[i \+ 1\] \(witness: N=(\d+), i=(\d+) and i=(\d+)\))\n", hint)
    assert reason, hint
    assert hint == SEQUENTIAL_PROVEN.format(reason=reason.group(1), why=CONFLICT_MEANING["WAR"])
    # The witness is an execution: the earlier iteration reads the element the next one overwrites
    size, first, second = (int(g) for g in reason.groups()[1:])
    assert second == first + 1 and second < size - 1, hint


def test_a_pattern_matched_dependence_names_its_witness_too():
    """An offset ``K`` that sizes nothing keeps the solver out (a conflict for one chosen ``K`` proves nothing), so
    the constant distance proves the carry. Its hint still names an execution: the first iteration and the next."""
    K = dace.symbol("K", dtype=dace.int64)

    @dace.program
    def offset_recurrence(a: dace.float64[N], b: dace.float64[N]):
        for i in range(1, N - K):
            a[i + K] = a[i + K - 1] + b[i]

    sdfg = offset_recurrence.to_sdfg(simplify=True)
    AnnotateLoopKinds().apply_pass(sdfg, {})
    (hint,) = [loop.specialization_hint for loop in loops_of(sdfg)]
    reason = re.search(r"carried: (RAW on a\[(.*)\] \(witness: (\w+)=1 and (\w+)=2\))\n", hint)
    assert reason, hint
    assert reason.group(3) == reason.group(4)
    assert hint == SEQUENTIAL_PROVEN.format(reason=reason.group(1), why=CONFLICT_MEANING["RAW"])


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
    carried = [line for line in rendered if line.startswith("// sequential -- carried: WAW on a[")]
    assert len(carried) == 1, rendered
    witness = re.search(r"\(witness: N=(\d+), i=(\d+) and i=(\d+)\)$", carried[0])
    assert witness, carried
    size, first, second = (int(g) for g in witness.groups())
    assert first < second < size, carried
    assert {5 * first, 3 * first} & {5 * second, 3 * second}, carried


def test_a_distance_set_by_an_argument_stays_unsure():
    """``a[i] = a[i - k]`` with ``k`` an argument: ``k`` is fixed but unknown, and ``k = 0`` carries nothing, so a
    witness that picks ``k = 1`` proves nothing about the program."""

    @dace.program
    def shift_by(a: dace.float64[N], k: dace.int64):
        for i in range(N):
            a[i] = a[i - k] + 1.0

    rendered = comment_lines(named(shift_by.to_sdfg(simplify=True)))
    assert [line for line in rendered if line.startswith("// unsure -- ")], rendered
    assert not [line for line in rendered if line.startswith("// sequential -- carried:")], rendered


def test_a_conflict_only_under_a_branch_is_not_witnessed():
    """A branch body may never run, so an access inside one cannot be half of a witness."""

    @dace.program
    def guarded_stride(a: dace.float64[5 * N], b: dace.float64[N]):
        for i in range(N):
            a[5 * i] = b[i]
            if b[i] > 0.0:
                a[3 * i] = b[i] + 1.0

    rendered = comment_lines(named(guarded_stride.to_sdfg(simplify=True)))
    assert not [line for line in rendered if line.startswith("// sequential -- carried:")], rendered


def test_a_triangular_solve_is_proven_through_its_row_copy():
    """trisolv: each row reads ``x[0:i]``, which the earlier iterations wrote. The read is an array copy, so every
    element of the range is read, and the witness names the copied range."""

    @dace.program
    def trisolv(L: dace.float64[N, N], x: dace.float64[N], b: dace.float64[N]):
        for i in range(N):
            x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]

    sdfg = trisolv.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, validate_all=False, target="cpu")
    AnnotateLoopKinds().apply_pass(sdfg, {})
    hints = [loop.specialization_hint.splitlines()[0] for loop in loops_of(sdfg)]
    assert len(hints) == 1, hints
    assert re.match(r"sequential -- carried: RAW on x\[0:\w+\] \(witness: N=\d+, \w+=0 and \w+=1\)$", hints[0]), hints


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
        n.specialization_hint.splitlines()[0] for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)
    ]
    loops = [loop.specialization_hint.splitlines()[0] for loop in loops_of(sdfg)]
    assert any(re.match(r"parallel -- under guard: .*inc.*", line) for line in maps), maps
    assert any(re.match(r"sequential -- fallback arm, runs when not \(.*inc.*\)$", line) for line in loops), loops


def test_a_map_resting_on_an_unknown_sign_is_versioned_on_it():
    """``a[i] = a[i + K] + b[i]`` is parallel after the snapshot only for ``K >= 0``. Nothing declares ``K``
    nonnegative, so canonicalization versions the loop on the sign instead of assuming it: the parallel map runs
    under a ``0 <= K`` branch, and the other signs keep a version of their own."""

    @dace.program
    def shift_sym(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N - K):
            a[i] = a[i + K] + b[i]

    sdfg = shift_sym.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, validate_all=False, target="cpu")
    AnnotateLoopKinds().apply_pass(sdfg, {})
    maps = [
        n.specialization_hint.splitlines()[0] for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)
    ]
    assert "parallel -- the iterations are independent" in maps, maps
    branches = [
        cond.as_string
        for region in sdfg.all_control_flow_regions()
        if isinstance(region, ConditionalBlock)
        for cond, _ in region.branches
        if cond is not None
    ]
    assert "(0 <= K)" in branches, branches


def test_a_map_resting_on_no_trap_assumes_nothing():
    """A trap on a symbol the map never names is not part of its evidence."""
    rendered = comment_lines(named(scaling_sdfg("cpf_kind_no_assumption")))
    assert "// " + PARALLEL.splitlines()[0] in rendered


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


def test_the_kinds_rendering_says_par_seq_or_unknown_and_nothing_else():
    """The bare variant: one word per loop and map -- no proof, no witness, no advice, no header contract, no
    library-node provenance."""
    kinds, proofs = (comment_lines(code)[1:] for code in kinds_and_proofs("cpf_kinds_only"))
    words = [line for line in kinds if line in ("// par", "// seq", "// unknown")]
    assert {"// par", "// seq"} <= set(words), kinds
    # What is left is the unit's own section labels, which the proofs rendering carries word for word.
    assert all(line in proofs for line in kinds if line not in words), kinds
    reasoning = ("--", "witness", "settled", "open:", "proven", "carried", "assumes", "CPU:", "GPU:")
    assert not [line for line in kinds if any(term in line for term in reasoning)], kinds
    assert any("witness" in line for line in proofs), proofs


def test_the_kinds_rendering_carries_no_tasklet_or_scope_label():
    """A tasklet's label (``_Add_``, ``for_696_scan_op``) is a DaCe name, not a verdict: the kinds rendering drops
    it from the statement and from the brace, the proofs rendering keeps it."""
    kinds, proofs = kinds_and_proofs("cpf_kinds_no_label")
    trailing = lambda code: [
        line
        for line in code.splitlines()
        if "//" in line and line.split("//")[0].strip() and not line.lstrip().startswith("#")
    ]
    assert not trailing(kinds), trailing(kinds)
    assert any("// _Add_" in line for line in trailing(proofs)), proofs


def test_the_kinds_rendering_names_the_loops_the_proofs_rendering_classes():
    """Same graph, same verdicts: each classed comment block of the proofs rendering is one word of the kinds
    rendering, the class of the block's first classed line, in the same order."""
    kinds, proofs = kinds_and_proofs("cpf_kinds_match")
    classes = [next(filter(None, map(cpf_lowering.kind_word, block)), None) for block in comment_blocks(proofs)]
    words = [word for block in comment_blocks(kinds) for word in block if word in ("par", "seq", "unknown")]
    assert words == [word for word in classes if word], (words, classes)


def test_the_two_variants_render_the_same_code():
    sdfg = carried_sdfg("cpf_kinds_code")
    AnnotateLoopKinds().apply_pass(sdfg, {})
    # Comments aside, whole-line and trailing alike
    strip = lambda code: [line.split("  //")[0] for line in code.splitlines() if not line.strip().startswith("//")]
    assert strip(render(sdfg, hints="kinds").code) == strip(render(sdfg, hints="proofs").code)


def test_an_unknown_hint_detail_is_refused():
    with pytest.raises(ValueError):
        render(scaling_sdfg("cpf_kinds_refused"), hints="some")


if __name__ == "__main__":
    test_the_rendering_states_which_verdicts_are_settled_and_which_are_open()
    test_a_map_hint_is_rendered_one_comment_line_per_line()
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
    test_a_declined_dependence_test_is_not_rendered_as_a_proof()
    test_an_access_under_a_data_dependent_branch_proves_no_carried_dependence()
    test_a_loop_that_was_never_examined_says_exactly_that()
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
    test_a_pattern_matched_dependence_names_its_witness_too()
    test_no_carried_dependence_is_claimed_for_independent_elements()
    test_a_pinned_independent_loop_does_not_read_as_parallel()
    test_writes_the_test_cannot_separate_are_proven_by_a_witness()
    test_a_distance_set_by_an_argument_stays_unsure()
    test_a_conflict_only_under_a_branch_is_not_witnessed()
    test_a_triangular_solve_is_proven_through_its_row_copy()
    test_a_guarded_loop_names_its_guard_on_both_arms()
    test_a_map_resting_on_an_unknown_sign_is_versioned_on_it()
    test_a_map_resting_on_no_trap_assumes_nothing()
    test_the_kinds_rendering_says_par_seq_or_unknown_and_nothing_else()
    test_the_kinds_rendering_carries_no_tasklet_or_scope_label()
    test_the_kinds_rendering_names_the_loops_the_proofs_rendering_classes()
    test_the_two_variants_render_the_same_code()
    test_an_unknown_hint_detail_is_refused()
