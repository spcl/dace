"""Flatten plan — the single source of truth for AoS→SoA unpacking.

Produced by ``hlfir-flatten-structs`` (as an MLIR module attribute)
and consumed by the binding emitter.  One ``FlattenRecipe`` per
outer storage path that was unpacked; the recipe carries both the
forward element expressions (outer → flat) and the inverse
expression (flats → outer) so the binding emitter can emit copy-in
and copy-out without knowing which specific flattening scheme fired.

Arbitrary struct hierarchies fall out naturally — the outer
expression in a recipe is a free-form Fortran expression (e.g.
``st%a%b%c``), threaded verbatim into the generated wrapper's
``c_loc(...)`` or ``do``-loop body.

Index convention inside recipe expressions:
    ``$i1``, ``$i2``, ..., ``$iN`` are placeholders for the N loop
    indices the copy-in / copy-out nest will declare.  The helper
    ``substitute_indices`` replaces them with concrete names at
    template time.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class FlattenRecipe:
    """One recorded unpacking.

    Args:
        flat_names:
            SDFG-visible flat names in argument order.  Examples:
                plain real member:     ("fld_a",)
                complex-split member:  ("st_z_re", "st_z_im")
        read_exprs:
            Parallel to ``flat_names``.  For each flat, the Fortran
            expression computing that flat's element at position
            ``($i1, $i2, ...)`` from the outer source.  Examples:
                plain real:       "st%u($i1, $i2)"
                complex split re: "real(st%z($i1, $i2), kind=c_double)"
                complex split im: "aimag(st%z($i1, $i2))"
        write_expr:
            Fortran expression reconstructing the outer's element
            at ``($i1, ...)`` from the flats.  Empty when the outer
            is read-only or the recipe is aliased (no copy-out
            needed).  Example:
                complex split: "cmplx(st_z_re($i1,$i2), st_z_im($i1,$i2), kind=c_double)"
        rank:
            Number of loop indices used by the expressions.  0 for
            scalar unpacks (unusual but supported).
        shape_exprs:
            Per-rank Fortran extent expression; length == rank.
            Typically ``("size(<outer>, dim=1)", ...)``.
        aliasable:
            ``True`` iff the recipe is pure element identity with
            matching storage layout — the emitter can skip
            allocate/copy and emit one ``c_f_pointer`` per
            ``flat_names[i]`` aliasing ``read_exprs[i]`` (with
            index placeholders stripped).  The pass sets this
            based on rank + element-type match.
        scratch_dtype:
            SDFG element dtype the emitter declares for flat
            scratch buffers (``float64`` / ``int32`` / ...).
            Today all flats of one recipe share a dtype.
    """
    flat_names: Tuple[str, ...]
    read_exprs: Tuple[str, ...]
    write_expr: str = ''
    rank: int = 0
    shape_exprs: Tuple[str, ...] = field(default_factory=tuple)
    aliasable: bool = False
    scratch_dtype: str = 'float64'

    # ----- JSON I/O ---------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d['flat_names'] = list(self.flat_names)
        d['read_exprs'] = list(self.read_exprs)
        d['shape_exprs'] = list(self.shape_exprs)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'FlattenRecipe':
        d = dict(d)
        d['flat_names'] = tuple(d.get('flat_names', []))
        d['read_exprs'] = tuple(d.get('read_exprs', []))
        d['shape_exprs'] = tuple(d.get('shape_exprs', []))
        return cls(**d)


@dataclass(frozen=True)
class FlattenEntry:
    """One outer dummy / storage path that was unpacked.

    Args:
        outer_expr:
            Fortran expression the user passes — ``st`` or
            ``st%a%b%c`` for arbitrary hierarchy depth.  Threaded
            verbatim into generated ``c_loc`` / loop bodies.
        outer_type:
            Fortran type of ``outer_expr`` — ``type(t_state)`` or
            ``real(c_double), dimension(:,:)``.  Used in
            auto-generated comments.
        writeback_intent:
            ``'out'`` / ``'inout'`` / ``''`` (= ``in`` or no copy
            back).  When non-empty and ``recipe.write_expr`` is set,
            the emitter generates a copy-out loop.
        recipe:
            The ``FlattenRecipe`` describing the unpack.
    """
    outer_expr: str
    outer_type: str
    writeback_intent: str
    recipe: FlattenRecipe

    def to_dict(self) -> dict:
        return {
            'outer_expr': self.outer_expr,
            'outer_type': self.outer_type,
            'writeback_intent': self.writeback_intent,
            'recipe': self.recipe.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'FlattenEntry':
        return cls(
            outer_expr=d['outer_expr'],
            outer_type=d['outer_type'],
            writeback_intent=d.get('writeback_intent', ''),
            recipe=FlattenRecipe.from_dict(d['recipe']),
        )


@dataclass(frozen=True)
class FlattenPlan:
    """All unpacks performed by ``hlfir-flatten-structs`` for one
    entry subroutine.  One entry per outer dummy that got
    flattened; untouched scalars / plain-array dummies don't
    appear.

    Args:
        entries:
            Tuple of ``FlattenEntry`` in argument order.  The
            emitter walks them sequentially.
    """
    entries: Tuple[FlattenEntry, ...] = field(default_factory=tuple)

    def to_json(self, path: str) -> None:
        with open(path, 'w') as fh:
            json.dump({'entries': [e.to_dict() for e in self.entries]}, fh, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'FlattenPlan':
        with open(path) as fh:
            d = json.load(fh)
        return cls(entries=tuple(FlattenEntry.from_dict(e) for e in d.get('entries', [])))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INDEX_RE = re.compile(r'\$i(\d+)')


def substitute_indices(expr: str, names: Tuple[str, ...]) -> str:
    """Replace ``$i1``, ``$i2``, ... placeholders with concrete loop
    variable names.

    Args:
        expr:
            Source expression with ``$iN`` placeholders.
        names:
            Tuple of concrete loop-index names.  ``$i1`` → ``names[0]``,
            ``$i2`` → ``names[1]``, etc.

    Returns:
        The expression with every placeholder substituted.  Raises
        ``IndexError`` if a placeholder references a name past the
        end of ``names``.

    Example:
        >>> substitute_indices("st%a%v($i1, $i2)", ("i1", "i2"))
        'st%a%v(i1, i2)'
    """

    def repl(m: re.Match) -> str:
        idx = int(m.group(1)) - 1
        if idx >= len(names):
            raise IndexError(f"placeholder $i{idx + 1} referenced but only {len(names)} names supplied")
        return names[idx]

    return _INDEX_RE.sub(repl, expr)


def strip_index_args(expr: str) -> str:
    """Strip the ``($i1, ...)`` suffix from an expression so it
    names the base storage path alone.

    Used by the alias emitter: ``c_loc`` takes the array, not an
    element.  Given ``"st%a%v($i1, $i2)"``, returns ``"st%a%v"``.

    Falls back to returning the input unchanged if the expression
    doesn't end in a parenthesised placeholder list.
    """
    m = re.match(r'^(.+?)\(\s*\$i\d+(?:\s*,\s*\$i\d+)*\s*\)\s*$', expr)
    return m.group(1) if m else expr
