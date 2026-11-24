# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The header that sums up a rendered unit's loop comments, read back from the comments themselves.

The loop comments are the audited facts; the header restates them per loop and adds no analysis of its own, so the
two cannot disagree. It names each loop by its counter in the rendered code.
"""

import re
from dataclasses import dataclass

from dace.cpf_lowering import KIND_EXTRAS, LoopClass, Reason, kind_line, reason_of
from dace.ordered import OrderedSet
from dace.transformation.passes.canonicalize.annotate_loop_kinds import PARALLEL

CLASSES = "|".join(cls.value for cls in LoopClass if cls is not LoopClass.LIBRARY)
CLASS_LINE = re.compile(rf"^\s*// ({CLASSES}) -- (.*)$")
#: The comment of a library lowering's loop: its construct and role in parentheses (:func:`library_comment`).
LIBRARY_LINE = re.compile(rf"^\s*// ({LoopClass.LIBRARY.value}) \((.*)\): not a tuning target$")
#: A class line of the kinds rendering (:func:`dace.cpf_lowering.kind_line`): the word, and a reason after ``: ``.
#: A class line anywhere in a unit, with its `` -- `` after the class.
CLASS_LINE_ANYWHERE = re.compile(rf"// ({CLASSES}) -- |// ({LoopClass.LIBRARY.value}) \(")
KIND_LINE = re.compile(
    rf"^\s*// ({LoopClass.PARALLEL.value}(?: {Reason.REDUCTION.value}| {Reason.SCAN.value})?|{LoopClass.SEQUENTIAL.value}|{LoopClass.UNKNOWN.value})(?:: (.*))?$"
)
#: The pragmas that belong to the loop under them; a barrier or a flush is a statement of its own.
LOOP_PRAGMA = re.compile(
    r"^\s*#pragma\s+(?:omp\s+(?:parallel\b|for\b|simd\b|distribute\b|taskloop\b|teams\b|target\b)|GCC\s+(?:ivdep|unroll)\b|unroll\b|clang\s+loop\b)"
)
LOOP_HEAD = re.compile(r"^\s*for \((?:[\w:<>]+\s+)*\**(\w+)\s*=")
WHILE_HEAD = re.compile(r"^\s*(?:while|do)\b")
CALL_HEAD = re.compile(r"^\s*(?:[\w:<>*]+\s+)*(\w+)\s*\(")
FUNCTION_HEAD = re.compile(r"^[A-Za-z_][\w\s:<>*,&]*?\b(\w+)\s*\([^;]*$")
NESTED_LOOPS = re.compile(r"the (\d+) nested loops over .* are all parallel")
#: One ``<op> into <target>`` of a reduction comment, which lists one per accumulator.
FOLD = re.compile(r"(\S+) into (\w+)")
PRIVATE = re.compile(r"private per iteration: ([\w, ]+)")
EXAMPLE = re.compile(r" \(example[^)]*\)")
#: Lines codegen puts between a loop's comment and its head: an accumulator load, the bounds of a range read from
#: data, a step check, a lone brace.
#: A line between a loop's comment and its head: an opening brace, an assert, or the declaration of a reduction
#: accumulator or map buffer, spelled ``__acc_*`` / ``__map_*`` by the generator and ``<container>_acc<n>`` once
#: ``cpf_compact`` respells it (``__acc_1_0_out`` -> ``out_acc0``).
BEFORE_LOOP = re.compile(r"^(?:[\w\s]+ (?:__(?:acc|map)_\w*|\w+_acc\d+) = .*;|assert\(.*\);|\{)$")
#: A loop head: ``for (``, ``while (`` or ``do`` opening a statement; a ``} while (c);`` tail starts with ``}``.
ANY_LOOP_HEAD = re.compile(r"^\s*(?:for|while)\s*\(|^\s*do\b")
#: The counter a ``for`` head declares or steps: ``for (long i = 0; ...``, or ``for (; i < n; ...`` (first name in
#: the condition).
FOR_COUNTER = re.compile(r"^\s*for\s*\(\s*(?:(?:[\w:<>]+\s+)*\**(\w+)\s*=|;\s*(\w+))")
#: The reasons :func:`comment_every_loop` writes. They describe a loop's role, not a proof of its own, so the header
#: does not list them: a dimension belongs to the nest the comment above proves, the others to a lowering's
#: implementation (a scan's passes, a BLAS call's zero fill, a helper function).
ADDED_REASONS = (Reason.DIMENSION, Reason.TILE, Reason.INNER_TILE, Reason.IMPLEMENTATION)
#: Labels on a construct that is not a loop: a thread team, a one-trip block on thread 0.
NOT_A_LOOP = (Reason.THREAD_TEAM, Reason.NOT_A_LOOP)

CLOSING = (
    "// No reordering is left to prove: every loop is either parallel or a proven recurrence. Gains beyond this form "
    "come from memory traffic (layout, fusion, blocking), not from more parallelism."
)


@dataclass(slots=True)
class Loop:
    name: str
    kind: LoopClass
    reason: str


def loops_of(code: str) -> list[Loop]:
    """Every loop of ``code`` outside the ``cpf_`` helpers, with the class its comment gives; an uncommented one is
    ``unknown``. Each is named by its counter, which ``ReadableNames`` makes unique in the unit. shortcut: an
    uncommented loop without an ``int64_t`` counter, or with one that starts with ``_``, is a library expansion's own
    loop (gemv, reduce, scan) and is left out; mark those loops apart if one needs a verdict."""
    found: list[Loop] = []
    label: tuple[LoopClass, str] | None = None
    covered, function = 0, ""
    names = Names()
    for line in code.splitlines():
        stripped = line.strip()
        head = FUNCTION_HEAD.match(line)
        if head is not None:
            function = head.group(1)
        classed = CLASS_LINE.match(line) or LIBRARY_LINE.match(line)
        if classed is not None:
            label = label or (LoopClass(classed.group(1)), classed.group(2))
            continue
        nested = NESTED_LOOPS.search(line)
        if nested is not None:
            covered = int(nested.group(1)) - 1
        if not stripped or stripped.startswith(("//", "#")) or BEFORE_LOOP.match(stripped):
            continue
        name = loop_name(stripped)
        if label is not None:
            if reason_of(label[1]) is Reason.DIMENSION and name:
                names.unique(name)
            elif label[0] is not LoopClass.LIBRARY and reason_of(label[1]) not in (*NOT_A_LOOP, *ADDED_REASONS):
                found.append(Loop(names.unique(name or call_name(stripped)), label[0], label[1]))
            label = None
        elif name and covered:
            covered -= 1
            names.unique(name)
        elif stripped.startswith("for (int64_t ") and not function.startswith("cpf_") and not name.startswith("_"):
            found.append(Loop(names.unique(name), LoopClass.UNKNOWN, "no comment: not analyzed"))
    return found


class Names:
    """The loop names given so far: a counter ``ReadableNames`` could not make unique (one read after its loop) gets
    ``#2``, ``#3`` on a repeat, so every name in the header is unique."""

    __slots__ = ("seen",)

    def __init__(self) -> None:
        self.seen: dict[str, int] = {}

    def unique(self, name: str) -> str:
        self.seen[name] = self.seen.get(name, 0) + 1
        return name if self.seen[name] == 1 else f"{name}#{self.seen[name]}"


def holds_label(stripped: str) -> bool:
    """Whether a line between a loop's comment and its head keeps the comment on that loop."""
    return not stripped or stripped.startswith(("//", "#")) or BEFORE_LOOP.match(stripped) is not None


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def loop_counter(line: str) -> str:
    counter = FOR_COUNTER.match(line)
    return (counter.group(1) or counter.group(2)) if counter is not None else ""


def construct_of(lines: list[str], index: int, function: str) -> str:
    """What an implementation loop belongs to: the title of the nearest comment block above it at a smaller
    indentation, the text before the ``:`` of its first line (``// BLAS gemv: y = ...`` -> ``BLAS gemv``, ``// pass
    1: ...`` -> ``pass 1``), searched back to the enclosing function's head; else that function."""
    own = indent_of(lines[index])
    for n in range(index - 1, -1, -1):
        line = lines[n]
        stripped = line.strip()
        if stripped and not stripped.startswith(("//", "#")) and indent_of(line) == 0:
            break  # the enclosing function's head, or code before it
        if not stripped.startswith("// ") or indent_of(line) >= own:
            continue
        if CLASS_LINE.match(line) or LIBRARY_LINE.match(line) or NESTED_LOOPS.search(line):
            continue
        above = lines[n - 1] if n > 0 else ""
        if above.strip().startswith("//") and indent_of(above) == indent_of(line):
            continue  # a continuation line of a longer comment
        title, colon, text = stripped[3:].partition(":")
        title = title.strip()
        if colon and title and ";" not in title and not title.startswith(("alt", "cpf")):
            return title
    if not function:
        return "the code above"
    return f"{'helper' if function.startswith('cpf_') else 'function'} {function}"


def bounds_of(line: str) -> tuple[str, str]:
    """The start and the condition of a ``for`` head (``for (long i = a; i < b; ...)`` -> ``("long i = a", "i < b")``),
    empty for any other loop."""
    parts = line.split("(", 1)[1].split(";") if FOR_COUNTER.match(line) else []
    return (parts[0], parts[1]) if len(parts) >= 3 else ("", "")


def uses(name: str, text: str) -> bool:
    return bool(name) and re.search(rf"\b{re.escape(name)}\b", text) is not None


def tiles(inner: str, counter: str) -> bool:
    """Whether the ``for`` head ``inner`` walks one tile of loop ``counter``: both its start and its condition use
    ``counter`` (a triangular ``j = i; j < n`` uses only the start)."""
    start, condition = bounds_of(inner)
    return uses(counter, start.split("=", 1)[-1]) and uses(counter, condition)


def added_comment(lines: list[str], index: int, parent: str, function: str) -> str:
    """The class and role of the uncommented loop at ``lines[index]`` inside loop ``parent`` ("" at the top of a
    function; see :data:`ADDED_REASONS`)."""
    line = lines[index]
    child = next((lines[n] for n in range(index + 1, len(lines)) if ANY_LOOP_HEAD.match(lines[n])), "")
    if indent_of(child) > indent_of(line) and tiles(child, loop_counter(line)):
        return f"{LoopClass.SEQUENTIAL.prefix}{Reason.TILE.value}: its tiles run in order; the loops inside class the iterations of each tile"
    if tiles(line, parent):
        return f"{LoopClass.SEQUENTIAL.prefix}{Reason.INNER_TILE.value}{parent}: runs one tile of its range in order"
    if (
        line.strip().startswith("for (int64_t ")
        and not function.startswith("cpf_")
        and not loop_counter(line).startswith("_")
    ):
        # A program loop no pass labelled: open, as the header has always counted it.
        return (
            f"{LoopClass.UNKNOWN.prefix}not analyzed for dependences: no pass classed this loop; parallel if no iteration writes an "
            "element another iteration reads or writes; otherwise keep the order"
        )
    construct = construct_of(lines, index, function)
    pragmas = " ".join(lines[n] for n in range(pragma_start(lines, index), index))
    if re.search(r"#pragma omp\b[^\n]*\bfor\b", pragmas):
        return f"{LoopClass.PARALLEL.prefix}{Reason.IMPLEMENTATION.value}{construct}: the team's threads split these iterations (omp for)"
    if re.search(r"#pragma omp\b[^\n]*\bsimd\b", pragmas):
        return f"{LoopClass.PARALLEL.prefix}{Reason.IMPLEMENTATION.value}{construct}: vectorized (omp simd)"
    return library_comment(lines, index, construct)


ZERO_FILL = re.compile(r"=\s*0(?:\.0*f?)?\s*;")


def library_comment(lines: list[str], index: int, construct: str) -> str:
    """The comment of a loop a library call's lowering writes without a pragma: it is no program loop and no tuning
    target, and its order is the lowering's own, so it carries no class word. A loop whose body stores zero is the
    zero fill."""
    body = " ".join(lines[index : index + 2])
    role = " zero fill" if ZERO_FILL.search(body) else ""
    return f"{LoopClass.LIBRARY.value} ({construct}{role}): not a tuning target"


def pragma_start(lines: list[str], index: int) -> int:
    """The first line of the run of ``#pragma`` lines right above ``lines[index]`` (``index`` when there is none)."""
    start = index
    while start > 0 and LOOP_PRAGMA.match(lines[start - 1]) is not None:
        start -= 1
    return start


def comment_every_loop(code: str, kinds: bool = False) -> str:
    """``code`` with a comment on every loop that has none. The CPU generator labels a map's or loop's first ``for``;
    the other dimensions of a multi-dimensional map, and the loops a lowering writes out by hand (a scan's passes, a
    BLAS call's zero fill, a helper function), had none, so a reader could not tell a proven loop from an unread
    one.

    ``kinds``: ``code`` is the kinds rendering, whose class lines carry the word alone (:data:`KIND_LINE`), and the
    added comments are written that way too. Its map comment does not count the nested loops, so a map's other
    dimensions are the unlabelled loops whose heads follow the labelled parallel head with only pragmas between."""
    lines = code.splitlines()
    out: list[str] = []
    labelled, nest, covered, dimension, total, first = False, 0, 0, 0, 0, ""
    # parallel: the last class line names a parallel loop; chain: the line above (pragmas aside) is the head of a
    # parallel loop, so in the kinds rendering an unlabelled head here is another dimension of its map.
    parallel, chain = False, False
    function = ""
    parents: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        head = FUNCTION_HEAD.match(line)
        if head is not None:
            function, parents = head.group(1), []
        classed = CLASS_LINE.match(line) or LIBRARY_LINE.match(line) or (KIND_LINE.match(line) if kinds else None)
        if classed is not None:
            # A thread team's or a one-trip block's comment is not a loop's: the loop inside gets its own.
            labelled = reason_of(classed.group(2) or "") not in NOT_A_LOOP
            parallel = classed.group(1).startswith(LoopClass.PARALLEL.value)
        nested = NESTED_LOOPS.search(line)
        if nested is not None:
            nest = int(nested.group(1))
        if stripped and not stripped.startswith(("//", "#")):
            # A code line at or left of an open loop's head ends that loop's body.
            while parents and parents[-1][0] >= indent_of(line):
                parents.pop()
        if ANY_LOOP_HEAD.match(line):
            indent = line[: indent_of(line)]
            dimension_of_chain = kinds and chain and not labelled
            if labelled:
                covered, dimension, total, first = (
                    (nest - 1, 1, nest, loop_counter(line)) if nest > 1 else (0, 0, 0, "")
                )
            else:
                if covered:
                    dimension += 1
                    covered -= 1
                    comment = (
                        f"{LoopClass.PARALLEL.prefix}{Reason.DIMENSION.value}{dimension} of the {total} nested loops from {first}; the comment "
                        "above proves them all"
                    )
                elif dimension_of_chain:
                    comment = LoopClass.PARALLEL.value
                else:
                    comment = added_comment(lines, index, parents[-1][1] if parents else "", function)
                    comment = (kind_line(comment) or comment) if kinds else comment
                # Above the loop's pragmas, as every generated comment is, so each pragma stays next to its loop.
                pragmas = index - pragma_start(lines, index)
                out[len(out) - pragmas : len(out) - pragmas] = [f"{indent}// {comment}"]
            parents.append((indent_of(line), loop_counter(line)))
            chain = kinds and ((labelled and parallel) or dimension_of_chain)
            labelled, nest = False, 0
        else:
            if stripped and not stripped.startswith("#pragma"):
                chain = False
            if not holds_label(stripped):
                labelled, nest, covered = False, 0, 0
        out.append(line)
    return "\n".join(out) + ("\n" if code.endswith("\n") else "")


def loop_name(line: str) -> str:
    counter = LOOP_HEAD.match(line)
    if counter is not None:
        return counter.group(1)
    return "while" if WHILE_HEAD.match(line) else ""


def call_name(line: str) -> str:
    call = CALL_HEAD.match(line)
    return f"{call.group(1)}()" if call is not None else "statement"


def reason_head(reason: str) -> str:
    """The reason up to its first ``;``, without the example and the alternative."""
    return KIND_EXTRAS.sub("", reason).split(";")[0].strip()


def summary_lines(loops: list[Loop]) -> list[str]:
    """The header: per class, the loops by name; the ``unknown`` loops with what failed and what would settle them;
    the closing line and ``cpf: proven: all`` only when every loop is proven parallel or a proven recurrence. A unit
    without a program loop (its loops are a lowering's, such as a find-first or argmax helper) has nothing open: just
    the closing line and ``cpf: proven: all``."""
    parallel = [loop for loop in loops if loop.kind is LoopClass.PARALLEL]
    folds = OrderedSet(
        f"{target} ({op})"
        for loop in parallel
        if reason_of(loop.reason) is Reason.REDUCTION
        for op, target in FOLD.findall(loop.reason.split(";")[0])
    )
    scans = [loop.name for loop in parallel if reason_of(loop.reason) is Reason.SCAN]
    private = [m.group(1) for loop in loops for m in [PRIVATE.search(loop.reason)] if m]
    lines = ["// Summary of the loop comments below."] if loops else []
    if parallel:
        names = ", ".join(dict.fromkeys(loop.name for loop in parallel))
        lines.append(f"//   parallel: {names}")
    if folds:
        lines.append(f"//     reductions into {', '.join(folds)}")
    if scans:
        lines.append(f"//     scans: {', '.join(dict.fromkeys(scans))}")
    if private:
        lines.append(f"//     private per iteration: {', '.join(dict.fromkeys(private))}")
    lines += [
        f"//   sequential: {loop.name}, {reason_head(loop.reason)}"
        for loop in loops
        if loop.kind is LoopClass.SEQUENTIAL
    ]
    lines += [
        f"//   unknown: {loop.name}, {EXAMPLE.sub('', loop.reason)}" for loop in loops if loop.kind is LoopClass.UNKNOWN
    ]
    # Every loop is analyzed (a short one too; its trip count only decides whether to fork): only unknown loops are open.
    open_loops = [loop for loop in loops if loop.kind is LoopClass.UNKNOWN]
    if not open_loops:
        lines += [CLOSING, "// cpf: proven: all"]
    else:
        lines.append(f"// cpf: proven: partial; open: {', '.join(dict.fromkeys(loop.name for loop in open_loops))}")
    # A versioned loop (``if (check) parallel else in order``) counts as proven; its parallel arms are listed apart
    guarded = [loop.name for loop in parallel if reason_of(loop.reason) is Reason.GUARDED]
    if guarded:
        lines.append(f"// cpf: guarded: {', '.join(dict.fromkeys(guarded))}")
    return lines


#: A declaration of ``{name}``: a type (no statement keyword), then the name, then ``[``, ``;`` or ``=``.
DECLARATION = r"^\s*(?!return\b|else\b|case\b|goto\b)[A-Za-z_][\w\s*]*?\s\**{name}\s*(\[|;|=)"
PRIVATE_NOTE = re.compile(r"; private per iteration: ([\w, ]+)")


def drop_inner_privates(code: str) -> str:
    """``code`` with each ``private per iteration`` name dropped from a loop comment when the loop body declares it:
    the note names a buffer declared before the loop that every iteration shares (cloudsc_solver_carry named ``arr60``,
    declared in an inner block)."""
    lines = code.splitlines()
    for number, line in enumerate(lines):
        note = PRIVATE_NOTE.search(line) if CLASS_LINE.match(line) else None
        if note is None:
            continue
        body = loop_body(lines, number + 1)
        kept = [n.strip() for n in note.group(1).split(",") if not declared(n.strip(), body)]
        replacement = f"; private per iteration: {', '.join(kept)}" if kept else ""
        lines[number] = line[: note.start()] + replacement + line[note.end() :]
    return "\n".join(lines) + ("\n" if code.endswith("\n") else "")


REDUCTION_CLAUSE = re.compile(r"reduction: (.+?); the other accesses are independent")


def drop_inner_reductions(code: str) -> str:
    """``code`` with each ``reduction: <op> into <name>`` dropped from a loop comment when the loop body declares
    ``name``: every iteration then accumulates into its own copy, so the loop does not reduce into it
    (icon_aes_graupel's ``j`` loop named ``arr109``, declared inside it); a loop left with none is independent."""
    lines = code.splitlines()
    for number, line in enumerate(lines):
        clause = REDUCTION_CLAUSE.search(line) if CLASS_LINE.match(line) else None
        if clause is None:
            continue
        body = loop_body(lines, number + 1)
        kept = [part for part in clause.group(1).split(" and ") if not declared(part.split(" into ")[-1], body)]
        joined = " and ".join(kept)
        replacement = clause.group(0).replace(clause.group(1), joined) if kept else PARALLEL.split(" -- ", 1)[1]
        lines[number] = line[: clause.start()] + replacement + line[clause.end() :]
    return "\n".join(lines) + ("\n" if code.endswith("\n") else "")


def declared(name: str, body: str) -> bool:
    return re.search(DECLARATION.format(name=re.escape(name)), body, re.MULTILINE) is not None


def loop_body(lines: list[str], start: int) -> str:
    """The text from the first ``for``/``while`` head at or after line ``start`` to the brace that closes its body."""
    head = next((n for n in range(start, len(lines)) if loop_name(lines[n].strip())), None)
    if head is None:
        return ""
    depth, opened = 0, False
    for number in range(head, len(lines)):
        depth += lines[number].count("{") - lines[number].count("}")
        opened = opened or "{" in lines[number]
        if opened and depth <= 0:
            return "\n".join(lines[head + 1 : number + 1])
    return "\n".join(lines[head + 1 :])


FUSION_NOTE = re.compile(r"(loop that (?:writes|reads) )(.+?)(, at an inner axis.*?keep )(.+?)( per iteration)")
IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


def drop_missing_fusion_names(code: str) -> str:
    """``code`` with each fusion note naming only intermediates the code still holds, and without a note none of
    whose intermediates is left: a later pass removed or folded it (qe_vexx_k's ``arange_nh``, folded into an index),
    so there is nothing left to fuse, and a comment must not name what the code does not show."""
    lines = code.splitlines()
    present = OrderedSet(IDENTIFIER.findall("\n".join(line for line in lines if not line.lstrip().startswith("//"))))
    kept_lines = []
    for line in lines:
        found = FUSION_NOTE.search(line) if line.lstrip().startswith("// fusion:") else None
        if found is None:
            kept_lines.append(line)
            continue
        kept = " and ".join(name for name in found.group(2).split(" and ") if name in present)
        if kept:
            head, gap, tail = found.group(1), found.group(3), found.group(5)
            kept_lines.append(line[: found.start()] + f"{head}{kept}{gap}{kept}{tail}" + line[found.end() :])
    return "\n".join(kept_lines) + ("\n" if code.endswith("\n") else "")


def with_summary(code: str, anchors: tuple[str, ...]) -> str:
    """``code`` with the summary of its loops inserted after the first line of ``anchors`` it holds; unchanged when
    ``code`` has none of those lines."""
    after = next((line for line in anchors if line in code), None)
    if after is None:
        return code
    head, tail = code.split(after, 1)
    return head + after + "\n" + "\n".join(summary_lines(loops_of(code))) + tail
