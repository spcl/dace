# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The header that sums up a rendered unit's loop comments, read back from the comments themselves.

The loop comments are the audited facts; the header restates them per loop and adds no analysis of its own, so the
two cannot disagree. It names each loop by its counter in the rendered code.
"""

import re
from dataclasses import dataclass

from dace.cpf_lowering import KIND_EXTRAS
from dace.ordered import OrderedSet
from dace.transformation.passes.canonicalize.annotate_loop_kinds import PARALLEL
from dace.transformation.passes.loop_specialization import VERSIONED

CLASS_LINE = re.compile(r"^\s*// (parallel|sequential|unknown) -- (.*)$")
LOOP_HEAD = re.compile(r"^\s*for \((?:[\w:<>]+\s+)*\**(\w+)\s*=")
WHILE_HEAD = re.compile(r"^\s*(?:while|do)\b")
CALL_HEAD = re.compile(r"^\s*(?:[\w:<>*]+\s+)*(\w+)\s*\(")
FUNCTION_HEAD = re.compile(r"^[A-Za-z_][\w\s:<>*,&]*?\b(\w+)\s*\([^;]*$")
NESTED_LOOPS = re.compile(r"the (\d+) nested loops over .* are all parallel")
REDUCTION = re.compile(r"^reduction: (\S+) into (\w+)")
PRIVATE = re.compile(r"private per iteration: ([\w, ]+)")
EXAMPLE = re.compile(r" \(example[^)]*\)")
#: Lines codegen puts between a loop's comment and its head: an accumulator load, the bounds of a range read from
#: data, a step check, a lone brace.
BEFORE_LOOP = re.compile(r"^(?:[\w\s]+ __(?:acc|map)_\w* = .*;|assert\(.*\);|\{)$")
#: Labels on a construct that is not a loop: a thread team, a one-trip block on thread 0.
NOT_A_LOOP = ("thread team", "not a loop")

CLOSING = (
    "// No reordering is left to prove: every loop is either parallel or a proven recurrence. Gains beyond this form "
    "come from memory traffic (layout, fusion, blocking), not from more parallelism."
)


@dataclass(slots=True)
class Loop:
    name: str
    kind: str
    reason: str


def loops_of(code: str) -> list[Loop]:
    """Every loop of ``code`` outside the ``cpf_`` helpers, with the class its comment gives; an uncommented one is
    ``unknown``. Each is named by its counter, which ``ReadableNames`` makes unique in the unit. shortcut: an
    uncommented loop without an ``int64_t`` counter, or with one that starts with ``_``, is a library expansion's own
    loop (gemv, reduce, scan) and is left out; mark those loops apart if one needs a verdict."""
    found: list[Loop] = []
    label: tuple[str, str] | None = None
    covered, function = 0, ""
    names = Names()
    for line in code.splitlines():
        stripped = line.strip()
        head = FUNCTION_HEAD.match(line)
        if head is not None:
            function = head.group(1)
        classed = CLASS_LINE.match(line)
        if classed is not None:
            label = label or (classed.group(1), classed.group(2))
            continue
        nested = NESTED_LOOPS.search(line)
        if nested is not None:
            covered = int(nested.group(1)) - 1
        if not stripped or stripped.startswith(("//", "#")) or BEFORE_LOOP.match(stripped):
            continue
        name = loop_name(stripped)
        if label is not None:
            if not label[1].startswith(NOT_A_LOOP):
                found.append(Loop(names.unique(name or call_name(stripped)), label[0], label[1]))
            label = None
        elif name and covered:
            covered -= 1
            names.unique(name)
        elif stripped.startswith("for (int64_t ") and not function.startswith("cpf_") and not name.startswith("_"):
            found.append(Loop(names.unique(name), "unknown", "no comment: not analyzed"))
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
    the closing line and ``cpf: proven: all`` only when every loop is proven parallel or a proven recurrence."""
    parallel = [loop for loop in loops if loop.kind == "parallel"]
    reductions = [REDUCTION.match(loop.reason) for loop in parallel]
    scans = [loop.name for loop in parallel if loop.reason.startswith("scan")]
    private = [m.group(1) for loop in loops for m in [PRIVATE.search(loop.reason)] if m]
    lines = ["// Summary of the loop comments below."]
    if parallel:
        names = ", ".join(dict.fromkeys(loop.name for loop in parallel))
        lines.append(f"//   parallel: {names}")
    if any(reductions):
        folds = ", ".join(dict.fromkeys(f"{m.group(2)} ({m.group(1)})" for m in reductions if m))
        lines.append(f"//     reductions into {folds}")
    if scans:
        lines.append(f"//     scans: {', '.join(dict.fromkeys(scans))}")
    if private:
        lines.append(f"//     private per iteration: {', '.join(dict.fromkeys(private))}")
    lines += [
        f"//   sequential: {loop.name}, {reason_head(loop.reason)}" for loop in loops if loop.kind == "sequential"
    ]
    lines += [f"//   unknown: {loop.name}, {EXAMPLE.sub('', loop.reason)}" for loop in loops if loop.kind == "unknown"]
    # Every loop is analyzed (a short one too; its trip count only decides whether to fork): only unknown loops are open.
    open_loops = [loop for loop in loops if loop.kind == "unknown"]
    if not open_loops:
        lines += [CLOSING, "// cpf: proven: all"]
    else:
        lines.append(f"// cpf: proven: partial; open: {', '.join(dict.fromkeys(loop.name for loop in open_loops))}")
    # A versioned loop (``if (check) parallel else in order``) counts as proven; its parallel arms are listed apart
    guarded = [loop.name for loop in parallel if loop.reason.startswith(VERSIONED)]
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
    ``code`` has no loop or none of those lines."""
    loops = loops_of(code)
    after = next((line for line in anchors if line in code), None)
    if not loops or after is None:
        return code
    head, tail = code.split(after, 1)
    return head + after + "\n" + "\n".join(summary_lines(loops)) + tail
