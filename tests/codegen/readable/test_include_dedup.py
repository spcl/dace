# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Duplicate ``#include`` elimination in the experimental (readable) CPU code generator.

The global code of the frame translation unit is written by streams that cannot see one another: the
file header (target/environment headers, then every SDFG's ``global_code``) and one write per tasklet
for its ``code_global``. A header named by two tasklets, or by a tasklet and its SDFG, therefore
arrives once per writer. :func:`~dace.codegen.targets.experimental_cpu.deduplicate_includes` drops
the repeats at the single point that sees all of them.
"""
import collections

import numpy as np
import pytest

import dace
from dace.codegen.targets.experimental_cpu import deduplicate_includes
from tests.codegen.readable.conftest import EXPERIMENTAL, use_implementation

#: Two headers every tasklet below asks for, so N tasklets produce N copies without the dedupe.
TASKLET_GLOBAL_CODE = "#include <cmath>\n#include <cstdio>"
#: Config path giving each top-level map nest its own translation unit.
SPLIT_KEY = ("compiler", "cpu", "codegen_params", "split_nsdfg_translation_units")


def include_lines(code_object):
    """The ``#include`` lines of a generated code object, provenance annotations stripped."""
    return [line.strip() for line in code_object.clean_code.splitlines() if line.strip().startswith("#include")]


def sqrt_chain_sdfg(name, num_tasklets, sdfg_global_code=None):
    """``b = sqrt(a)`` repeated over ``num_tasklets`` sequential states, every tasklet carrying the
    same ``code_global``. Optionally the SDFG itself names one of the same headers too."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [8], dace.float64)
    sdfg.add_array("B", [8], dace.float64)
    if sdfg_global_code:
        sdfg.append_global_code(sdfg_global_code)
    previous = None
    for index in range(num_tasklets):
        state = sdfg.add_state(f"s{index}")
        if previous is not None:
            sdfg.add_edge(previous, state, dace.InterstateEdge())
        previous = state
        tasklet = state.add_tasklet(f"t{index}", {"a"}, {"b"},
                                    "b = std::sqrt(a);",
                                    language=dace.Language.CPP,
                                    code_global=TASKLET_GLOBAL_CODE)
        entry, exit_node = state.add_map(f"m{index}", dict(i="0:8"))
        state.add_memlet_path(state.add_read("A"), entry, tasklet, dst_conn="a", memlet=dace.Memlet("A[i]"))
        state.add_memlet_path(tasklet, exit_node, state.add_write("B"), src_conn="b", memlet=dace.Memlet("B[i]"))
    return sdfg


def generate(sdfg):
    with use_implementation(EXPERIMENTAL):
        return sdfg.generate_code()


def frame_object(objects, sdfg):
    """The frame ``.cpp`` among the generated code objects."""
    return next(obj for obj in objects if obj.name == sdfg.name and obj.language == "cpp")


def test_repeated_tasklet_code_global_included_once():
    """Three tasklets naming the same two headers: each header appears exactly once."""
    sdfg = sqrt_chain_sdfg("incdedup_tasklets", 3)
    lines = include_lines(frame_object(generate(sdfg), sdfg))
    counts = collections.Counter(lines)
    assert counts["#include <cmath>"] == 1
    assert counts["#include <cstdio>"] == 1
    assert not [line for line, count in counts.items() if count > 1]


def test_sdfg_global_code_and_tasklet_agree_on_one_copy():
    """A header named by BOTH the SDFG's ``global_code`` and a tasklet's ``code_global`` -- the two
    streams cannot see each other, so this is the cross-source case."""
    sdfg = sqrt_chain_sdfg("incdedup_mixed", 2, sdfg_global_code="#include <cmath>\n#include <vector>")
    lines = include_lines(frame_object(generate(sdfg), sdfg))
    counts = collections.Counter(lines)
    assert counts["#include <cmath>"] == 1
    assert counts["#include <vector>"] == 1
    assert counts["#include <cstdio>"] == 1


def test_first_occurrence_order_is_preserved():
    """Order matters for config-dependent headers, so the surviving copy is the FIRST one."""
    sdfg = sqrt_chain_sdfg("incdedup_order", 3)
    lines = include_lines(frame_object(generate(sdfg), sdfg))
    assert lines == list(dict.fromkeys(lines))
    assert lines.index("#include <cmath>") < lines.index("#include <cstdio>")


def test_generated_code_still_compiles_and_runs():
    """The dedupe is a text edit on the emitted global code -- the program must still build and run."""
    sdfg = sqrt_chain_sdfg("incdedup_run", 3)
    a = np.arange(1.0, 9.0)
    b = np.zeros(8)
    with use_implementation(EXPERIMENTAL):
        sdfg(A=a, B=b)
    assert np.allclose(b, np.sqrt(a))


def test_split_translation_unit_includes_once():
    """Two tasklets of ONE map nest, emitted into their own translation unit by
    ``split_nsdfg_translation_units``: that file re-emits the shared header itself, so it is
    deduplicated where the target hands its code objects over."""
    sdfg = dace.SDFG("incdedup_split")
    for name in ("A", "B", "C"):
        sdfg.add_array(name, [8], dace.float64)
    state = sdfg.add_state("s")
    entry, exit_node = state.add_map("m", dict(i="0:8"))
    read = state.add_read("A")
    for index, out in enumerate(("B", "C")):
        tasklet = state.add_tasklet(f"t{index}", {"a"}, {"b"},
                                    "b = std::sqrt(a);",
                                    language=dace.Language.CPP,
                                    code_global=TASKLET_GLOBAL_CODE)
        state.add_memlet_path(read, entry, tasklet, dst_conn="a", memlet=dace.Memlet("A[i]"))
        state.add_memlet_path(tasklet, exit_node, state.add_write(out), src_conn="b", memlet=dace.Memlet(f"{out}[i]"))
    split = dace.config.set_temporary(*SPLIT_KEY, value=True)
    with use_implementation(EXPERIMENTAL), split:
        objects = sdfg.generate_code()
    nests = [obj for obj in objects if obj.target_type == "nsdfg"]
    assert nests, "split_nsdfg_translation_units emitted no nest translation unit"
    for nest in nests:
        counts = collections.Counter(include_lines(nest))
        assert not [line for line, count in counts.items() if count > 1], counts


def test_only_include_lines_are_touched():
    """Non-include lines are left alone, repeats included."""
    code = "int x = 1;\n#include <a.h>\nint x = 1;\n#include <a.h>\nint y = 2;\n"
    assert deduplicate_includes(code) == "int x = 1;\n#include <a.h>\nint x = 1;\nint y = 2;\n"


def test_annotated_lines_compare_without_their_provenance_tag():
    """``CodeIOStream`` pads a per-node ``////__DACE:`` tag onto each line, so two copies of one
    include are never textually equal; the comparison must look past the tag."""
    code = ("#include <a.h>    ////__DACE:0:0:0\n"
            "#include <a.h>    ////__DACE:0:1:0\n")
    assert deduplicate_includes(code) == "#include <a.h>    ////__DACE:0:0:0\n"


def test_conditional_includes_are_left_alone():
    """An include inside ``#ifdef`` is selected by its branch, so it neither gets dropped nor
    suppresses the unconditional copy that follows it."""
    code = ("#ifdef USE_FAST\n"
            "#include <a.h>\n"
            "#else\n"
            "#include <a.h>\n"
            "#endif\n"
            "#include <a.h>\n"
            "#include <a.h>\n")
    assert deduplicate_includes(code) == ("#ifdef USE_FAST\n"
                                          "#include <a.h>\n"
                                          "#else\n"
                                          "#include <a.h>\n"
                                          "#endif\n"
                                          "#include <a.h>\n")


def test_nested_conditionals_restore_the_unconditional_depth():
    """Nesting is counted, so the include after the outermost ``#endif`` is deduped again."""
    code = ("#if A\n"
            "#ifdef B\n"
            "#include <a.h>\n"
            "#endif\n"
            "#endif\n"
            "#include <b.h>\n"
            "#include <b.h>\n")
    assert deduplicate_includes(code) == ("#if A\n"
                                          "#ifdef B\n"
                                          "#include <a.h>\n"
                                          "#endif\n"
                                          "#endif\n"
                                          "#include <b.h>\n")


if __name__ == "__main__":
    pytest.main([__file__])
