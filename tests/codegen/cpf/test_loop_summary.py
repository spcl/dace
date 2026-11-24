# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The header of a rendered unit sums up its loop comments and says whether every loop is proven.

It is read back from the comments, the audited facts, and adds no analysis: the parallel loops, the reductions and
scans, each sequential loop with its dependence, each open loop with what failed, and the machine-readable
``// cpf: proven:`` line."""

import re

import dace
from dace import cpf_lowering
from dace.codegen import cpf_summary
from dace.codegen.cpf import CONTRACT_LINES, render
from dace.transformation.passes.canonicalize.annotate_loop_kinds import AnnotateLoopKinds
from tests.codegen.cpf.test_specialization_hints import carried_sdfg, named, scaling_sdfg

N = dace.symbol("N")
M = dace.symbol("M")


INNER_PRIVATE = """\
    // parallel -- independent: no iteration writes an element another iteration reads or writes; private per iteration: t
    for (int64_t j = 0; (j < n); j = (j + 1)) {
        for (int64_t k = 0; (k < m); k = (k + 1)) {
            {
                double t;
                t = a[k];
                b[j] = t;
            }
        }
    }
"""


def test_a_private_name_the_loop_body_declares_is_dropped():
    """cloudsc_solver_carry: ``private per iteration: arr60`` named a scalar declared inside the loop body, not a
    buffer before the loop that every iteration shares."""
    first = cpf_summary.drop_inner_privates(INNER_PRIVATE).splitlines()[0]
    assert first.endswith("reads or writes"), first


def test_a_private_buffer_declared_before_the_loop_stays_named():
    """eigh: ``arr18`` is declared before the loop and assigned inside it; the note is the one fact the code hides."""
    code = "double t;\n" + INNER_PRIVATE.replace("double t;", "")
    assert cpf_summary.drop_inner_privates(code).splitlines()[1].endswith("private per iteration: t")


INNER_SUM = """\
    // parallel -- reduction: + into s and + into out; the other accesses are independent
    for (j = 0; (j < 6); j = (j + 1)) {
        for (int64_t k = 0; k < n; k += 1) {
            double s;
            s = 0.0;
            for (int64_t l = 0; l < 6; l += 1) {
                s = s + a[l];
            }
            out[0] = out[0] + s;
        }
    }
"""


def test_a_reduction_into_a_name_the_loop_body_declares_is_dropped():
    """icon_aes_graupel: the ``j`` loop said ``reduction: + into arr109`` of a scalar each of its iterations declares."""
    first = cpf_summary.drop_inner_reductions(INNER_SUM).splitlines()[0]
    assert first == "    // parallel -- reduction: + into out; the other accesses are independent", first
    only_local = INNER_SUM.replace(" and + into out", "")
    first = cpf_summary.drop_inner_reductions(only_local).splitlines()[0]
    assert (
        first == "    // parallel -- independent: no iteration writes an element another iteration reads or writes"
    ), first


DATA_RANGE = """\
    // parallel -- independent: no iteration writes an element another iteration reads or writes
    {
        int64_t __map_17_b0 = ptr[i];
        int64_t __map_17_e1 = ptr[(i + 1)];
        #pragma omp parallel for
        for (int64_t k = __map_17_b0; k < __map_17_e1; k += 1) {
            y[k] = 0.0;
        }
    }
"""


def test_a_label_over_the_bounds_of_a_range_read_from_data_names_its_loop():
    """A map over ``ptr[i]:ptr[i + 1]`` loads its bounds between label and loop: the summary said ``statement`` and
    listed ``k`` open, uncommented."""
    (loop,) = cpf_summary.loops_of(DATA_RANGE)
    assert (loop.name, loop.kind) == ("k", cpf_lowering.LoopClass.PARALLEL), loop


def summary(code: str) -> list[str]:
    lines = code.splitlines()
    start = lines.index("// Summary of the loop comments below.")
    end = next(n for n in range(start, len(lines)) if lines[n].startswith("// cpf: proven: "))
    return lines[start : end + 1]


CLOSING = "// No reordering is left to prove: every loop is either parallel or a proven recurrence."


def test_a_unit_of_parallel_loops_is_summed_up_as_proven():
    header = summary(named(scaling_sdfg("cpf_audit_summary_parallel")))
    assert re.fullmatch(r"//   parallel: i", header[1]), header
    assert header[-2].startswith(CLOSING) and header[-1] == "// cpf: proven: all", header


def test_a_proven_recurrence_is_summed_up_with_its_dependence():
    header = summary(named(carried_sdfg("cpf_audit_summary_recurrence")))
    sequential = [line for line in header if line.startswith("//   sequential: ")]
    assert len(sequential) == 1 and re.fullmatch(
        r"//   sequential: i, proven dependence on b\[i - 1, 0:N\]: an iteration reads what an earlier one wrote",
        sequential[0],
    ), header
    assert header[-2].startswith(CLOSING) and header[-1] == "// cpf: proven: all", header


@dace.program
def gather_store(idx: dace.int64[N], b: dace.float64[N], a: dace.float64[M]):
    for i in range(N):
        a[idx[i]] = b[i]


def test_a_unit_with_an_open_loop_names_it_and_does_not_close():
    sdfg = gather_store.to_sdfg(simplify=True)
    sdfg.name = "cpf_audit_summary_open"
    header = summary(named(sdfg))
    assert [line for line in header if line.startswith("//   unknown: i, ")], header
    assert re.fullmatch(r"// cpf: proven: partial; open: i", header[-1]), header
    assert not any(line.startswith(CLOSING) for line in header), header


def test_the_kinds_rendering_has_no_summary():
    sdfg = scaling_sdfg("cpf_audit_summary_kinds")
    AnnotateLoopKinds().apply_pass(sdfg, {})
    code = render(sdfg, hints=cpf_lowering.HintDetail.KINDS.value).code
    assert "// cpf: proven" not in code


NESTS = """\
static inline void body(double* a, int64_t n) {
    // unknown -- a conditional write to a at an index the loop counter does not fix
    for (int64_t t = 0; (t < n); t = (t + 1)) {
        // unknown -- a conditional write to a at an index the loop counter does not fix
        for (int64_t i = 0; (i < n); i = (i + 1)) {
            a[i] = 0.0;
        }
        // unknown -- a conditional write to a at an index the loop counter does not fix
        for (int64_t i = 0; (i < n); i = (i + 1)) {
            a[i] = 1.0;
        }
    }
}
static inline void other(double* a, int64_t n) {
    // unknown -- a conditional write to a at an index the loop counter does not fix
    for (int64_t i = 0; (i < n); i = (i + 1)) {
        a[i] = 2.0;
    }
}
"""

#: The bench's parser: one line, ``all`` or ``partial; open: `` and a list of unique names with no comma in them.
PROVEN_LINE = re.compile(r"^// cpf: proven: (all|partial; open: ([^,\n]+(?:, [^,\n]+)*))$")


def test_a_counter_left_repeated_is_numbered_in_the_header():
    """A counter ``ReadableNames`` keeps (one read after its loop) may repeat; the header numbers the repeats."""
    (proven,) = [line for line in cpf_summary.summary_lines(cpf_summary.loops_of(NESTS)) if "cpf: proven" in line]
    assert proven == "// cpf: proven: partial; open: t, i, i#2, i#3", proven


def test_the_proven_line_keeps_the_format_the_bench_parses():
    for code in (NESTS, NESTS.replace("// unknown -- a conditional write", "// parallel -- independent: a write")):
        summary = cpf_summary.summary_lines(cpf_summary.loops_of(code))
        (proven,) = [line for line in summary if line.startswith("// cpf: proven")]
        found = PROVEN_LINE.match(proven)
        assert found is not None, proven
        names = (found.group(2) or "").split(", ")
        assert len(names) == len(set(names)), names


def test_the_legend_explains_only_the_classes_the_unit_uses():
    """s2233: the header explained ``unknown`` loops in a unit that has none."""
    code = named(carried_sdfg("cpf_audit_legend"))
    assert CONTRACT_LINES[3] in code and CONTRACT_LINES[4] not in code, code


def test_only_unknown_loops_are_open():
    """Owner 11-24: a sequential loop carries a proven dependence, so only ``unknown`` loops keep a form partial."""
    code = (
        "// sequential -- proven dependence on a[k + 1]: an iteration reads what an earlier one wrote\n"
        "for (int64_t k = 0; (k < 3); k = (k + 1)) {\n}\n"
        "// unknown -- the solver could not show that each iteration writes a different element of b; parallel if no "
        "two iterations write the same element; otherwise a later iteration's write must win, so keep the order\n"
        "for (int64_t j = 0; (j < n); j = (j + 1)) {\n}\n"
    )
    summary = cpf_summary.summary_lines(cpf_summary.loops_of(code))
    assert summary[-1] == "// cpf: proven: partial; open: j", summary


@dace.program
def small_fixed_loops(a: dace.float64[4], b: dace.float64[4], c: dace.float64[4]):
    for k in range(3):
        a[k + 1] = a[k + 1] + a[k]
    for m in range(3):
        c[m] = b[m] * 2.0


def test_a_loop_over_a_few_fixed_iterations_gets_a_real_class():
    """Owner 11-24: a short loop is analyzed like any other; its trip count only decides whether to fork."""
    sdfg = small_fixed_loops.to_sdfg(simplify=True)
    sdfg.name = "cpf_audit_summary_small"
    code = named(sdfg)
    assert "not analyzed" not in code and "only a few iterations" not in code, code
    header = summary(code)
    assert any(re.match(r"//   sequential: k, proven dependence on a\[", line) for line in header), header
    assert header[-1] == "// cpf: proven: all", header


@dace.program
def scattered_species(a: dace.float64[N], order: dace.int64[5], b: dace.float64[N]):
    for s in range(5):
        a[order[s]] = a[order[s]] + 1.0
    for i in range(N):
        b[order[i % 5]] = b[order[i % 5]] + 1.0


def test_an_unproven_loop_over_a_few_fixed_iterations_is_unknown_and_open():
    """CloudSC's nclv = 5 species loops: too few iterations to share out, so no check is worth it, but the dependence
    test did not settle them, so they are open like the same body over ``N`` iterations."""
    sdfg = scattered_species.to_sdfg(simplify=True)
    sdfg.name = "cpf_audit_summary_small_extent"
    header = summary(named(sdfg))
    assert any(re.match(r"//   unknown: s, small extent \(5 iterations\), not proven: ", line) for line in header), (
        header
    )
    assert header[-1] == "// cpf: proven: partial; open: s, i", header


VERSIONED_LOOP = """\
if ((inc != 0)) {
    // parallel -- the runtime check found inc != 0, so no iteration writes an element another iteration reads or writes
    for (int64_t i = 0; i < n; i += 1) {
        a[i * inc] = a[i * inc] + b[i];
    }
} else {
    // sequential -- the runtime check found inc == 0: two iterations may then touch the same element, so this arm keeps the order
    for (int64_t i2 = 0; (i2 < n); i2 = (i2 + 1)) {
        a[i2 * inc] = a[i2 * inc] + b[i2];
    }
}
"""


def test_a_versioned_loop_counts_as_proven_and_lists_its_guarded_arm():
    """Owner ruling 11-23: ``if (check) parallel else in order`` is proven; the parallel arms are named apart."""
    summary = cpf_summary.summary_lines(cpf_summary.loops_of(VERSIONED_LOOP))
    assert summary[-2:] == ["// cpf: proven: all", "// cpf: guarded: i"], summary


def test_a_unit_whose_only_loop_is_a_helper_is_proven():
    """argmax_with_index: the one loop is the argmax helper's, so nothing is open and the bench's v2 note needs the
    proven line."""
    code = (
        "// head\nvoid f(const double* a, int64_t n) {\n"
        "    // parallel -- implementation loop of argmax: the team's threads split these iterations (omp for)\n"
        "    for (int64_t i = 0; i < n; ++i) {\n    }\n}\n"
    )
    header = cpf_summary.with_summary(code, ("// head",)).splitlines()[1:3]
    assert header[0].startswith(CLOSING) and header[1] == "// cpf: proven: all", header


def test_a_unit_without_a_versioned_loop_has_no_guarded_line():
    summary = cpf_summary.summary_lines(cpf_summary.loops_of(NESTS))
    assert not any(line.startswith("// cpf: guarded") for line in summary), summary


FUSED = """\
// parallel -- independent: no iteration writes an element another iteration reads or writes
// fusion: shares (j) with the earlier loop that writes mean and gone, at an inner axis; alt: reorder this loop to (j, i), fuse the two, and keep mean and gone per iteration
for (int64_t i = 0; i < n; i += 1) {
    // fusion: shares (j) with the earlier loop that writes folded, at an inner axis; alt: reorder this loop to (j, i), fuse the two, and keep folded per iteration
    for (int64_t j = 0; j < n; j += 1) {
        out[i] = mean[j];
    }
}
"""


def test_a_fusion_note_names_only_intermediates_the_code_holds():
    """qe_vexx_k, correlation: a later pass folded the intermediate away, and the note still named it."""
    notes = [line.strip() for line in cpf_summary.drop_missing_fusion_names(FUSED).splitlines() if "fusion:" in line]
    assert notes == [
        "// fusion: shares (j) with the earlier loop that writes mean, at an inner axis; alt: reorder this loop to "
        "(j, i), fuse the two, and keep mean per iteration"
    ], notes


READABLE_ACCUMULATOR = """\
    // parallel -- reduction: + into arr14; the other accesses are independent
    {
        double arr14_acc0 = arr14[i];
        #pragma omp simd reduction(+:arr14_acc0)
        for (int64_t j = 0; j < 4; j += 1) {
            arr14_acc0 = arr14_acc0 + a[j];
        }
    }
"""


def test_a_respelled_accumulator_between_comment_and_loop_keeps_the_comment_on_the_loop():
    """cfd, lavamd: ``cpf_compact`` respells ``__acc_1_0_arr14`` to ``arr14_acc0``, the label landed on that
    declaration (``statement``), and the header listed the reduction loop ``j`` as unknown with no comment."""
    (loop,) = cpf_summary.loops_of(READABLE_ACCUMULATOR)
    assert (loop.name, loop.kind) == ("j", cpf_lowering.LoopClass.PARALLEL), loop


UNCOMMENTED = """\
void f0(double *a, double *b, int64_t N, int64_t M) {
    // parallel -- independent: no iteration writes an element another iteration reads or writes
    // the 2 nested loops over i, j are all parallel
    #pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < N; i += 1) {
        for (int64_t j = 0; j < M; j += 1) {
            a[i * M + j] = b[i * M + j];
        }
    }
    // scan: running (prefix) fold along an axis
    {
        #pragma omp parallel
        for (long t = 0; t < 4; t += 1) {
            // pass 1: each chunk's total
            #pragma omp for schedule(static)
            for (long c = t; c < 4; ++c) {
                for (long k = 0; k < 8; ++k) b[c] += a[k];
            }
        }
    }
    for (int64_t c2 = 0; c2 < 256; c2 += 1) {
        // unknown -- not proven; parallel if no two iterations write one element; otherwise keep the order
        for (int64_t i2 = c2 * 16; i2 < (c2 + 1) * 16; i2 += 1) {
            for (int64_t k2 = i2 * 2; k2 < i2 * 2 + 2; k2 += 1) a[k2] = b[i2];
        }
    }
}
static inline double cpf_sum(const double *in, long count) {
    double acc = 0;
    #pragma omp simd reduction(+:acc)
    for (long i = 0; i < count; ++i) acc = acc + in[i];
    return acc;
}
"""


def class_lines(code: str) -> list[tuple[str, str]]:
    """Each loop head of ``code`` with the class line above it, skipping the pragma, comment, and brace lines between
    them; ``("", head)`` for a loop without one."""
    lines = code.splitlines()
    found = []
    for n, line in enumerate(lines):
        if cpf_summary.ANY_LOOP_HEAD.match(line):
            k = n - 1
            while k >= 0 and cpf_summary.holds_label(lines[k].strip()) and not cpf_summary.CLASS_LINE.match(lines[k]):
                k -= 1
            found.append((lines[k].strip() if k >= 0 and cpf_summary.CLASS_LINE.match(lines[k]) else "", line.strip()))
    return found


def test_every_loop_gets_a_comment_naming_its_role():
    """A map's other dimensions, a lowering's hand-written loops, a tile loop and a helper's loop had no comment, so
    a reader could not tell a proven loop from an unread one."""
    comments = [comment for comment, head in class_lines(cpf_summary.comment_every_loop(UNCOMMENTED))]
    assert comments == [
        "// parallel -- independent: no iteration writes an element another iteration reads or writes",
        "// parallel -- dimension 2 of the 2 nested loops from i; the comment above proves them all",
        "// sequential -- implementation loop of scan: runs in order as written",
        "// parallel -- implementation loop of scan: the team's threads split these iterations (omp for)",
        "// sequential -- implementation loop of pass 1: runs in order as written",
        "// sequential -- tile loop: its tiles run in order; the loops inside class the iterations of each tile",
        "// unknown -- not proven; parallel if no two iterations write one element; otherwise keep the order",
        "// sequential -- inner tile of loop i2: runs one tile of its range in order",
        "// parallel -- implementation loop of helper cpf_sum: vectorized (omp simd)",
    ], comments


def test_an_added_comment_sits_above_the_loop_pragmas_and_the_header_lists_only_program_loops():
    """The added roles (a dimension, a tile, an implementation loop) belong to a loop the header already accounts
    for, so only the classed program loops stay listed and the open one keeps the unit partial."""
    code = cpf_summary.comment_every_loop(UNCOMMENTED)
    lines = code.splitlines()
    for n, line in enumerate(lines[1:], 1):
        if cpf_summary.CLASS_LINE.match(line):
            assert not lines[n - 1].strip().startswith("#pragma omp"), (lines[n - 1], line)
    header = cpf_summary.summary_lines(cpf_summary.loops_of(code))
    assert header[1] == "//   parallel: i", header
    assert header[-1] == "// cpf: proven: partial; open: i2", header


def test_a_program_loop_no_pass_classed_stays_open_with_its_guard():
    """An uncommented ``int64_t`` loop of the program was open in the header; its added comment keeps it open and
    says what would make it parallel, so the unit cannot read as proven."""
    code = """\
void f0(double *a, int64_t N) {
    for (int64_t i = 0; i < N; i += 1) {
        a[i] = 0;
    }
}
"""
    ((comment, head),) = class_lines(cpf_summary.comment_every_loop(code))
    assert comment.startswith("// unknown -- not analyzed for dependences") and "parallel if " in comment, comment
    assert "; otherwise " in comment, comment
    header = cpf_summary.summary_lines(cpf_summary.loops_of(cpf_summary.comment_every_loop(code)))
    assert header[-1] == "// cpf: proven: partial; open: i", header


def test_a_triangular_loop_is_not_a_tile():
    """Only a loop whose start and bound both use the parent's counter walks one tile of it: ``j = i; j < n`` does
    not."""
    code = """\
for (long i = 0; i < n; ++i) {
    for (long j = i; j < n; ++j) a[j] += b[i];
}
"""
    comments = [comment for comment, head in class_lines(cpf_summary.comment_every_loop(code))]
    assert all("tile" not in comment for comment in comments), comments


@dace.program
def two_dims(a: dace.float64[N, M], b: dace.float64[N, M]):
    for i, j in dace.map[0:N, 0:M]:
        b[i, j] = 2 * a[i, j]


def test_a_rendered_unit_comments_every_loop_above_its_pragmas():
    sdfg = two_dims.to_sdfg(simplify=True)
    sdfg.name = "cpf_audit_every_loop"
    code = named(sdfg)
    found = class_lines(code)
    assert found and all(comment for comment, head in found), found
    lines = code.splitlines()
    for n, line in enumerate(lines[1:], 1):
        if cpf_summary.CLASS_LINE.match(line):
            assert (
                not lines[n - 1].strip().startswith(("#pragma omp parallel", "#pragma omp for", "#pragma omp simd"))
            ), (
                lines[n - 1],
                line,
            )


def test_an_added_comment_skips_a_barrier_above_the_loop():
    """A barrier is a statement of its own, not a pragma of the loop under it: the added comment goes between the
    two, so the barrier does not read as part of the loop."""
    code = """\
static inline void cpf_helper(double *a, long n) {
    #pragma omp parallel
    {
        #pragma omp barrier
        for (long q = 0; q < n; ++q) {
            a[q] = 0;
        }
    }
}
"""
    lines = cpf_summary.comment_every_loop(code).splitlines()
    barrier = next(n for n, line in enumerate(lines) if "#pragma omp barrier" in line)
    assert cpf_summary.CLASS_LINE.match(lines[barrier + 1]), lines
    assert cpf_summary.ANY_LOOP_HEAD.match(lines[barrier + 2]), lines


def test_the_kinds_rendering_gets_a_class_word_on_every_loop():
    """The kinds rendering carries the word alone: a map's other dimensions follow its parallel head, an
    implementation loop of a helper says its class and role, and no line has a proof's dash."""
    code = """\
void f0(double *a, int64_t N, int64_t M) {
    // parallel
    #pragma omp parallel for
    for (int64_t i = 0; i < N; i += 1) {
        for (int64_t j = 0; j < M; j += 1) {
            a[i * M + j] = 0;
        }
    }
}
static inline void cpf_fill(double *a, long n) {
    for (long q = 0; q < n; ++q) a[q] = 0;
}
"""
    lines = cpf_summary.comment_every_loop(code, kinds=True).splitlines()
    above = [lines[n - 1].strip() for n, line in enumerate(lines) if cpf_summary.ANY_LOOP_HEAD.match(line)]
    assert above[1] == "// parallel", lines
    assert above[2] == "// sequential: implementation loop of helper cpf_fill: runs in order as written", lines
    assert not [line for line in lines if " -- " in line], lines


if __name__ == "__main__":
    test_a_private_name_the_loop_body_declares_is_dropped()
    test_a_private_buffer_declared_before_the_loop_stays_named()
    test_a_reduction_into_a_name_the_loop_body_declares_is_dropped()
    test_a_label_over_the_bounds_of_a_range_read_from_data_names_its_loop()
    test_a_unit_of_parallel_loops_is_summed_up_as_proven()
    test_a_proven_recurrence_is_summed_up_with_its_dependence()
    test_a_unit_with_an_open_loop_names_it_and_does_not_close()
    test_the_kinds_rendering_has_no_summary()
    test_a_counter_left_repeated_is_numbered_in_the_header()
    test_the_proven_line_keeps_the_format_the_bench_parses()
    test_the_legend_explains_only_the_classes_the_unit_uses()
    test_only_unknown_loops_are_open()
    test_a_loop_over_a_few_fixed_iterations_gets_a_real_class()
    test_an_unproven_loop_over_a_few_fixed_iterations_is_sequential_and_settled()
    test_a_versioned_loop_counts_as_proven_and_lists_its_guarded_arm()
    test_a_unit_whose_only_loop_is_a_helper_is_proven()
    test_a_unit_without_a_versioned_loop_has_no_guarded_line()
    test_a_fusion_note_names_only_intermediates_the_code_holds()
    test_a_respelled_accumulator_between_comment_and_loop_keeps_the_comment_on_the_loop()
    test_every_loop_gets_a_comment_naming_its_role()
    test_an_added_comment_sits_above_the_loop_pragmas_and_the_header_lists_only_program_loops()
    test_a_program_loop_no_pass_classed_stays_open_with_its_guard()
    test_a_triangular_loop_is_not_a_tile()
    test_a_rendered_unit_comments_every_loop_above_its_pragmas()
    test_an_added_comment_skips_a_barrier_above_the_loop()
    test_the_kinds_rendering_gets_a_class_word_on_every_loop()
