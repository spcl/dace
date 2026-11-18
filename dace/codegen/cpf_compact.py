# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The LLM-friendly spelling of a CPF rendering: the same translation unit in fewer tokens.

A CPF form is read by an agent, which pays for every token. :func:`compact` respells a finished
:class:`~dace.codegen.cpf.Rendering` without changing what it computes:

1. every generated name gets a short one: a container ``arr0, arr1, ...``, a symbol, map or loop variable
   longer than four characters ``sym0, ...``, a codegen-made ``_``-prefixed local ``t0, ...``, a generated
   function longer than eight characters ``f0, ...`` (the ``cpf_`` runtime helpers keep their names, which
   say what they compute, and so do index and size helpers). The entry point's parameters keep their names.
   Names derived from a renamed one follow it (``x_idx`` -> ``arr0_idx``; the persistent buffer, its count and
   its release function ``arr0_buf``, ``arr0_cap``, ``arr0_free``);
2. comments keep their text, with the names inside them respelled the same way, so a loop hint still
   names the container it is about;
3. ``static`` definitions nothing references go, to a fixpoint;
4. a parenthesized plain variable between operators loses its parentheses (``(i)*(n)`` -> ``i*n``);
5. blank-line runs collapse.

String literals are never touched. :class:`Compacted` carries the name map, so a short name can be read
back as the SDFG's (:meth:`Compacted.revert`).
"""

import re
from collections.abc import Iterator
from typing import TYPE_CHECKING, NamedTuple

from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion

if TYPE_CHECKING:
    from dace.codegen.cpf import Rendering

__all__ = ["Compacted", "compact", "tokens"]

#: One lexical piece of a C/C++ unit; the named group says which.
TOKEN = re.compile(
    r"(?P<comment>/\*.*?\*/|//[^\n]*)"
    r"|(?P<string>\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*')"
    r"|(?P<ident>[A-Za-z_]\w*)"
    r"|(?P<number>\.?\d(?:[\w.]|[eEpP][+-])*)"
    r"|(?P<other>\s+|.)",
    re.DOTALL,
)
#: Words a name never becomes and parentheses never come off.
C_WORDS = frozenset(
    "auto break case char const continue default do double else enum extern float for goto if inline int long "
    "register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while "
    "bool true false nullptr NULL alignas constexpr template typename class namespace using new delete".split()
)
#: Characters after which ``(name)`` is an operand, never a call, a cast or a ``sizeof`` argument.
OPERAND_BEFORE = frozenset("*+-/%<>=&|^!~?:,([")
#: Characters before which ``(name)`` is an operand, never a cast.
OPERAND_AFTER = frozenset("*+-/%<>=&|^?:,;)]")
#: The persistent-buffer spellings of CPF's lifted transients, and their short forms.
PERSISTENT_PREFIX = "__cpf_persistent_"
#: A persistent buffer's names (pointer, element count, release function) after the prefix and container, and the
#: endings of their short forms.
PERSISTENT_PARTS = {"": "_buf", "_count": "_cap", "_release": "_free"}
#: Suffixes of names derived from a container's (its index and size helpers).
DERIVED_SUFFIXES = ("_idx", "_size")


class Compacted(NamedTuple):
    """A compacted translation unit and the names it changed."""

    #: The LLM-friendly translation unit.
    code: str
    #: Short name -> the rendering's name, for every identifier :func:`compact` respelled.
    names: dict[str, str]

    def revert(self, text: str) -> str:
        """``text`` (the compacted unit, or an edit of it) with every short name spelled as the rendering did."""
        return respell(text, self.names, in_comments=True)


def tokens(text: str) -> Iterator[tuple[str, str]]:
    """``(kind, text)`` per lexical piece of ``text``; the pieces concatenate back to ``text``."""
    for match in TOKEN.finditer(text):
        kind = match.lastgroup
        if kind is None:
            raise AssertionError("every TOKEN alternative is named")
        yield kind, match.group()


def respell(text: str, mapping: dict[str, str], in_comments: bool) -> str:
    """``text`` with every identifier in ``mapping`` respelled; strings untouched, comments when ``in_comments``."""
    if not mapping:
        return text
    word = re.compile(r"\b(" + "|".join(re.escape(name) for name in sorted(mapping, key=len, reverse=True)) + r")\b")
    out = []
    for kind, piece in tokens(text):
        if kind == "ident":
            out.append(mapping.get(piece, piece))
        elif kind == "comment" and in_comments:
            out.append(word.sub(lambda m: mapping[m.group(1)], piece))
        else:
            out.append(piece)
    return "".join(out)


def generated_names(rendering: "Rendering") -> tuple[list[str], list[str]]:
    """The rendering's containers and long symbols that are not entry parameters: (containers, symbols)."""
    keep = set(rendering.arguments)
    containers: dict[str, None] = {}
    symbols: dict[str, None] = {}
    for sdfg in rendering.sdfg.all_sdfgs_recursive():
        for name in sdfg.arrays:
            if name not in keep:
                containers[name] = None
        for name in sdfg.symbols:
            if name not in keep and len(name) > 4:
                symbols[name] = None
        for node, unused_parent in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.MapEntry):
                symbols.update((p, None) for p in node.map.params if p not in keep and len(p) > 4)
        for region in sdfg.all_control_flow_regions():
            if isinstance(region, LoopRegion) and region.loop_variable:
                variable = region.loop_variable
                if variable not in keep and len(variable) > 4:
                    symbols[variable] = None
    return list(containers), [s for s in symbols if s not in containers]


def static_functions(text: str) -> list[str]:
    """Names of the ``static`` functions ``text`` defines, in order."""
    found = re.findall(r"^static\s+(?:inline\s+)?[\w\s\*]+?\b(\w+)\s*\(", text, re.MULTILINE)
    return list(dict.fromkeys(found))


def name_map(text: str, rendering: "Rendering") -> dict[str, str]:
    """Rendering name -> short name, for every identifier of ``text`` :func:`compact` respells."""
    present = {piece for kind, piece in tokens(text) if kind == "ident"}
    taken = set(present) | C_WORDS
    mapping: dict[str, str] = {}

    def assign(name: str, prefix: str, counter: list[int]) -> str:
        while f"{prefix}{counter[0]}" in taken:
            counter[0] += 1
        short = f"{prefix}{counter[0]}"
        counter[0] += 1
        taken.add(short)
        mapping[name] = short
        return short

    containers, symbols = generated_names(rendering)
    arr, sym, tmp, fn = [0], [0], [0], [0]
    for name in containers:
        persistent = {PERSISTENT_PREFIX + name + end: short_end for end, short_end in PERSISTENT_PARTS.items()}
        related = [name + s for s in DERIVED_SUFFIXES] + list(persistent)
        if name not in present and not any(r in present for r in related):
            continue
        short = assign(name, "arr", arr)
        for suffix in DERIVED_SUFFIXES:
            if name + suffix in present:
                mapping[name + suffix] = short + suffix
        for long_name, short_end in persistent.items():
            if long_name in present:
                mapping[long_name] = short + short_end
    for name in symbols:
        if name in present and name not in mapping:
            assign(name, "sym", sym)
    keep = set(rendering.arguments)
    for name in static_functions(text):
        if (
            name not in mapping
            and name not in keep
            and len(name) > 8
            and not name.startswith("cpf_")
            and not name.endswith(DERIVED_SUFFIXES)
        ):
            assign(name, "f", fn)
    for name in sorted(present):
        if (
            name not in mapping
            and name not in keep
            and re.match(r"_[a-z_]", name)
            and not name.endswith("__")
            and not name.startswith(("__builtin", PERSISTENT_PREFIX))
        ):
            assign(name, "t", tmp)
    return mapping


def split_top_level(text: str) -> list[str]:
    """``text`` as top-level items: a preprocessor ``#if`` group, another directive, a function or declaration
    through its closing ``}`` or ``;``, or a run of blanks and comments. The items concatenate back to ``text``."""
    items: list[str] = []
    current: list[str] = []
    depth = 0
    pp_depth = 0
    line_start = True
    directive = ""
    for kind, piece in tokens(text):
        current.append(piece)
        if directive:
            if directive == "#" and kind == "ident":
                directive = piece
                pp_depth += piece in ("if", "ifdef", "ifndef")
                pp_depth -= piece == "endif"
            if kind == "other" and "\n" in piece and current[-2:-1] != ["\\"]:
                directive = ""
                line_start = True
                if pp_depth == 0 and depth == 0:
                    items.append("".join(current))
                    current = []
            continue
        if kind == "other" and piece == "#" and line_start and depth == 0:
            directive = "#"
            continue
        if kind == "other" and piece.isspace():
            line_start = line_start or "\n" in piece
            continue
        line_start = False
        if kind != "other" or pp_depth:
            continue
        if piece == "{":
            depth += 1
        elif piece == "}":
            depth -= 1
            if depth == 0:
                items.append("".join(current))
                current = []
        elif piece == ";" and depth == 0:
            items.append("".join(current))
            current = []
    if current:
        items.append("".join(current))
    return items


def static_definition(item: str) -> str | None:
    """The name ``item`` defines when it is a ``static`` function or variable, else ``None``."""
    code = "".join(piece for kind, piece in tokens(item) if kind != "comment").strip()
    if not code.startswith("static"):
        return None
    if code.endswith("}"):
        head = code[: code.index("{")]
        call = re.search(r"(\w+)\s*\([^()]*\)\s*$", head)
        return call.group(1) if call else None
    declared = re.search(r"(\w+)\s*(?:\[[^\]]*\]\s*)*(?:=[^;]*)?;$", code)
    return declared.group(1) if declared else None


def drop_unused(text: str) -> str:
    """``text`` without the ``static`` definitions nothing else references, to a fixpoint."""
    items = split_top_level(text)
    while True:
        uses = [{piece for kind, piece in tokens(item) if kind == "ident"} for item in items]
        everywhere: dict[str, int] = {}
        for used in uses:
            for name in used:
                everywhere[name] = everywhere.get(name, 0) + 1
        keep: list[str] = []
        for item, used in zip(items, uses):
            name = static_definition(item)
            if name is not None and everywhere.get(name, 0) - (name in used) == 0:
                continue
            keep.append(item)
        if len(keep) == len(items):
            return "".join(items)
        items = keep


def unparenthesize(text: str) -> str:
    """``text`` with ``(name)`` -> ``name`` where the parentheses group a plain variable between operators; macro
    lines keep theirs."""
    pieces = list(tokens(text))
    out: list[str] = []
    index = 0
    directive = False
    while index < len(pieces):
        kind, piece = pieces[index]
        if kind == "other" and "\n" in piece:
            directive = False
        elif kind == "other" and piece == "#" and not "".join(out).rsplit("\n", 1)[-1].strip():
            directive = True
        window = pieces[index : index + 3]
        if (
            not directive
            and len(window) == 3
            and window[0] == ("other", "(")
            and window[1][0] == "ident"
            and window[1][1] not in C_WORDS
            and not window[1][1].endswith("_t")
            and window[2] == ("other", ")")
        ):
            before = "".join(out).rstrip()
            after = "".join(text_ for unused_kind, text_ in pieces[index + 3 : index + 6]).lstrip()
            attribute = before.endswith("__attribute__(")  # GNU attributes need their double parentheses
            if before and before[-1] in OPERAND_BEFORE and after and after[0] in OPERAND_AFTER and not attribute:
                out.append(window[1][1])
                index += 3
                continue
        out.append(piece)
        index += 1
    return "".join(out)


def compact(rendering: "Rendering") -> Compacted:
    """``rendering`` (a :class:`~dace.codegen.cpf.Rendering` of a host dialect) in the LLM-friendly spelling."""
    text = drop_unused(rendering.code)
    mapping = name_map(text, rendering)
    text = respell(text, mapping, in_comments=True)
    text = unparenthesize(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return Compacted(text, {short: name for name, short in mapping.items()})
