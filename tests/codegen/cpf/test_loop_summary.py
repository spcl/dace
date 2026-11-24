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
    assert (loop.name, loop.kind) == ("k", "parallel"), loop


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
    test_a_versioned_loop_counts_as_proven_and_lists_its_guarded_arm()
    test_a_unit_without_a_versioned_loop_has_no_guarded_line()
    test_a_fusion_note_names_only_intermediates_the_code_holds()
