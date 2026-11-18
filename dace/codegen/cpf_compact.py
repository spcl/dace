# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The LLM-friendly spelling of a CPF unit: the same translation unit under shorter generated names.

A CPF form is read by an agent, which pays for every token. :func:`compact` respells a host unit that
:func:`~dace.codegen.cpf.render` produced (``render(..., llm_friendly=True)`` calls it before its own checks)
without changing what it computes or exports. It is conservative by construction: it renames only names the
SDFG says are generated, and it drops only definitions nothing can reach.

1. A generated name gets a short one: a container ``arr0, arr1, ...``, a symbol, map or loop variable ``sym0,
   ...``, a ``static`` function ``f0, ...``. A candidate is a name of the prepared SDFG (or a ``static`` function
   of the unit) longer than four characters that contains ``_`` or a digit, so a plain word is never respelled.
   Names derived from a renamed one follow it (``x_idx`` -> ``arr0_idx``; a persistent buffer, its count and its
   release function ``arr0_buf``, ``arr0_cap``, ``arr0_free``). Every short name, derived ones included, is new
   to the unit.
2. Never renamed: the names the caller keeps (the entry point's name and parameters), every non-``static``
   file-scope name, the ``cpf_`` runtime helpers, and any name that occurs in a preprocessor line other than
   ``#pragma`` (``#define``, ``#if``, ``#include``) or next to ``##``. A ``#pragma`` line is respelled with the
   code, since an OpenMP clause names the code's variables.
3. Comments keep their text with the respelled names in it; string literals are never touched.
4. A ``static`` definition nothing references goes, to a fixpoint, when it is exactly one plain ``static``
   function or variable: no attribute, no directive inside, balanced braces.

:func:`compact` refuses (``ValueError``) a unit whose exported names the respelling would change. The name map is
returned, so a short name can be read back as the SDFG's (:meth:`Compacted.revert`).
"""

import itertools
import re
from collections.abc import Iterable, Iterator
from typing import NamedTuple

from dace.sdfg import SDFG, nodes
from dace.sdfg.state import LoopRegion

__all__ = ["Compacted", "compact", "exported_names", "respell", "tokens"]

#: One lexical piece of a C/C++ unit; the named group says which. A preprocessor line, continuations included,
#: is one piece, so a name in it is never mistaken for code.
TOKEN = re.compile(
    r"(?P<directive>^[ \t]*#(?:[^\n\\]|\\.)*)"
    r"|(?P<comment>/\*.*?\*/|//[^\n]*)"
    r"|(?P<string>\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*')"
    r"|(?P<ident>[A-Za-z_]\w*)"
    r"|(?P<number>\.?\d(?:[\w.]|[eEpP][+-])*)"
    r"|(?P<other>\s+|.)",
    re.DOTALL | re.MULTILINE,
)
#: A ``#pragma`` line: respelled with the code it annotates.
PRAGMA = re.compile(r"^[ \t]*#[ \t]*pragma\b")
#: The persistent-buffer spellings of CPF's lifted transients after the prefix and container, and the endings of
#: their short forms.
PERSISTENT_PREFIX = "__cpf_persistent_"
PERSISTENT_PARTS = {"": "_buf", "_count": "_cap", "_release": "_free"}
#: Suffixes of names derived from a container's (its index and size helpers).
DERIVED_SUFFIXES = ("_idx", "_size")
#: The name a non-``static`` file-scope definition or declaration introduces.
EXPORTED = re.compile(r"(?:extern\s+\"C\"\s+)?[A-Za-z_][\w\s\*]*?\b(\w+)\s*[\(\[=;]")


class Compacted(NamedTuple):
    """A compacted translation unit and the names it changed."""

    #: The LLM-friendly translation unit.
    code: str
    #: Short name -> the rendering's name, for every identifier :func:`compact` respelled.
    names: dict[str, str]

    def revert(self, text: str) -> str:
        """``text`` (the compacted unit, or an edit of it) with every short name spelled as the rendering did."""
        return respell(text, self.names)


def tokens(text: str) -> Iterator[tuple[str, str]]:
    """``(kind, text)`` per lexical piece of ``text``; the pieces concatenate back to ``text``."""
    for match in TOKEN.finditer(text):
        kind = match.lastgroup
        if kind is None:
            raise AssertionError("every TOKEN alternative is named")
        yield kind, match.group()


def respell(text: str, mapping: dict[str, str]) -> str:
    """``text`` with every identifier in ``mapping`` respelled in code, comments and ``#pragma`` lines; strings and
    other preprocessor lines untouched."""
    if not mapping:
        return text
    word = re.compile(r"\b(" + "|".join(re.escape(name) for name in sorted(mapping, key=len, reverse=True)) + r")\b")
    out = []
    for kind, piece in tokens(text):
        if kind == "ident":
            out.append(mapping.get(piece, piece))
        elif kind == "comment" or (kind == "directive" and PRAGMA.match(piece)):
            out.append(word.sub(lambda m: mapping[m.group(1)], piece))
        else:
            out.append(piece)
    return "".join(out)


def split_top_level(text: str) -> list[str]:
    """``text`` as top-level items: a directive line, a function or declaration through its closing ``}`` or ``;``,
    or a run of blanks and comments. The items concatenate back to ``text``."""
    items: list[str] = []
    current: list[str] = []
    depth = 0
    for kind, piece in tokens(text):
        if kind == "directive" and depth == 0:
            if current:
                items.append("".join(current))
                current = []
            items.append(piece)
            continue
        current.append(piece)
        if kind != "other":
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


def code_of(item: str) -> str:
    """``item`` without comments and directive lines, stripped."""
    return "".join(piece for kind, piece in tokens(item) if kind not in ("comment", "directive")).strip()


def exported_names(text: str) -> set[str]:
    """Names of the non-``static`` file-scope definitions and declarations of ``text`` (what it exports)."""
    names: set[str] = set()
    for item in split_top_level(text):
        code = code_of(item)
        if not code or re.match(r"(static|typedef|struct|union|enum)\b", code):
            continue
        match = EXPORTED.match(code)
        if match is not None:
            names.add(match.group(1))
    return names


def is_candidate(name: str) -> bool:
    """Whether ``name`` may be respelled at all: long enough to gain, and a generated spelling, not a word."""
    return len(name) > 4 and re.search(r"[_\d]", name) is not None


def protected(text: str, keep: Iterable[str]) -> set[str]:
    """The names :func:`compact` must keep: ``keep``, the exports, any name a non-``#pragma`` preprocessor line
    mentions, and any name next to a token paste."""
    names = set(keep) | exported_names(text)
    for kind, piece in tokens(text):
        if kind == "directive" and not PRAGMA.match(piece):
            names |= set(re.findall(r"[A-Za-z_]\w*", piece))
    for left, right in re.findall(r"([A-Za-z_]\w*)\s*##|##\s*([A-Za-z_]\w*)", text):
        names |= {left, right} - {""}
    return names


def generated_names(sdfg: SDFG) -> tuple[list[str], list[str]]:
    """``sdfg``'s containers, and its symbols, map parameters and loop variables, that :func:`is_candidate`."""
    containers: dict[str, None] = {}
    symbols: dict[str, None] = {}
    for nested in sdfg.all_sdfgs_recursive():
        containers.update((name, None) for name in nested.arrays if is_candidate(name))
        symbols.update((name, None) for name in nested.symbols if is_candidate(name))
        for node, unused_parent in nested.all_nodes_recursive():
            if isinstance(node, nodes.MapEntry):
                symbols.update((p, None) for p in node.map.params if is_candidate(p))
        for region in nested.all_control_flow_regions():
            if isinstance(region, LoopRegion) and region.loop_variable and is_candidate(region.loop_variable):
                symbols[region.loop_variable] = None
    return list(containers), [name for name in symbols if name not in containers]


def static_functions(text: str) -> list[str]:
    """Names of the ``static`` functions ``text`` defines, in order."""
    return list(dict.fromkeys(re.findall(r"^static\s+(?:inline\s+)?[\w\s\*]+?\b(\w+)\s*\(", text, re.MULTILINE)))


def name_map(text: str, sdfg: SDFG, keep: Iterable[str]) -> dict[str, str]:
    """Rendering name -> short name, for every identifier of ``text`` :func:`compact` respells."""
    present = {piece for kind, piece in tokens(text) if kind == "ident"}
    fixed = protected(text, keep)
    taken = present | fixed
    mapping: dict[str, str] = {}
    counters = {prefix: itertools.count() for prefix in ("arr", "sym", "f")}

    def fresh(prefix: str, endings: Iterable[str]) -> str:
        """A short name with ``prefix`` that is new to the unit with each of ``endings`` appended too."""
        spellings = ["", *endings]
        while True:
            short = f"{prefix}{next(counters[prefix])}"
            if all(short + end not in taken for end in spellings):
                taken.update(short + end for end in spellings)
                return short

    def free(name: str) -> bool:
        return name not in fixed and name not in mapping

    containers, symbols = generated_names(sdfg)
    for name in containers:
        derived = {name + suffix: suffix for suffix in DERIVED_SUFFIXES}
        derived |= {PERSISTENT_PREFIX + name + end: short_end for end, short_end in PERSISTENT_PARTS.items()}
        here = {long: end for long, end in derived.items() if long in present}
        if not free(name) or not all(free(long) for long in here) or (name not in present and not here):
            continue
        short = fresh("arr", here.values())
        mapping[name] = short
        mapping.update((long, short + end) for long, end in here.items())
    for name in symbols:
        if name in present and free(name):
            mapping[name] = fresh("sym", ())
    for name in static_functions(text):
        if free(name) and is_candidate(name) and not name.startswith("cpf_") and not name.endswith(DERIVED_SUFFIXES):
            mapping[name] = fresh("f", ())
    return mapping


def droppable(item: str) -> str | None:
    """The name ``item`` defines when it is exactly one plain ``static`` function or variable definition (no
    attribute, no directive inside, balanced braces), else ``None``: only such an item may be dropped."""
    if any(kind == "directive" for kind, unused_piece in tokens(item)):
        return None
    code = code_of(item)
    if not re.match(r"static\b", code) or re.search(r"__attribute__|\[\[|\basm\b|__asm__", code):
        return None
    if code.count("{") != code.count("}"):
        return None
    if code.endswith("}"):
        call = re.search(r"(\w+)\s*\([^()]*\)\s*$", code[: code.index("{")])
        return call.group(1) if call else None
    declared = re.fullmatch(r"static\s+[\w\s\*]+?\b(\w+)\s*(?:\[[^\]]*\]\s*)*(?:=[^;{}]*)?;", code)
    return declared.group(1) if declared else None


def drop_unused(text: str) -> str:
    """``text`` without the :func:`droppable` definitions nothing else references, to a fixpoint."""
    items = split_top_level(text)
    while True:
        words = [set(re.findall(r"[A-Za-z_]\w*", item)) for item in items]
        everywhere: dict[str, int] = {}
        for used in words:
            for name in used:
                everywhere[name] = everywhere.get(name, 0) + 1
        keep = []
        for item, used in zip(items, words):
            name = droppable(item)
            if name is None or everywhere.get(name, 0) - (name in used) > 0:
                keep.append(item)
        if len(keep) == len(items):
            return "".join(items)
        items = keep


def compact(code: str, sdfg: SDFG, keep: Iterable[str]) -> Compacted:
    """``code``, a host unit rendered from ``sdfg``, in the LLM-friendly spelling.

    :param keep: names that must not change besides the exports: the entry point's name and parameters.
    :raises ValueError: if the respelling would change a name the unit exports.
    """
    text = drop_unused(code)
    mapping = name_map(text, sdfg, keep)
    text = respell(text, mapping)
    text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+\n", "\n", text))
    if exported_names(text) != exported_names(code):
        changed = sorted(exported_names(code) ^ exported_names(text))
        raise ValueError(f"compacting {sdfg.name} would change what it exports: {changed}")
    return Compacted(text, {short: name for name, short in mapping.items()})
