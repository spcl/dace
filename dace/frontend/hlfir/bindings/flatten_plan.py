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

    Three emitter shapes are encoded by combinations of the boolean
    flags:

      * ``aliasable=True``: zero-copy ``c_f_pointer`` alias.
      * ``aos_alloc=False, aliasable=False``: explicit allocate +
        deep ``do``-loop copy.
      * ``aos_alloc=True``:  padding-to-max pack/unpack for an
        AoS dummy with allocatable / pointer array members
        (Phase 5c-B).  See ``aos_alloc`` field below.

    Args:
        flat_names:
            SDFG-visible flat names in argument order.  Examples:
                plain real member:     ("fld_a",)
                complex-split member:  ("st_z_re", "st_z_im")
                aos_alloc:             ("a_w",)            (single flat)
        read_exprs:
            Parallel to ``flat_names``.  For each flat, the Fortran
            expression computing that flat's element at position
            ``($i1, $i2, ...)`` from the outer source.  Examples:
                plain real:       "st%u($i1, $i2)"
                complex split re: "real(st%z($i1, $i2), kind=c_double)"
                complex split im: "aimag(st%z($i1, $i2))"
                aos_alloc:        "a($i1)%w($i2)"
        write_expr:
            Fortran expression reconstructing the outer's element
            at ``($i1, ...)`` from the flats.  Empty when the outer
            is read-only, the recipe is aliased (no copy-out needed),
            or the recipe is ``aos_alloc`` (the bindings layer uses
            bespoke pack-out code instead of an element-wise template).
            Example:
                complex split: "cmplx(st_z_re($i1,$i2), st_z_im($i1,$i2), kind=c_double)"
        rank:
            Number of loop indices used by the expressions.  0 for
            scalar unpacks (unusual but supported).  For ``aos_alloc``
            this is ``outer_rank + 1``.
        shape_exprs:
            Per-rank Fortran extent expression; length == rank.
            Typically ``("size(<outer>, dim=1)", ...)``.  For an
            ``aos_alloc`` recipe the inner dim is the cap symbol
            verbatim (not a ``size()`` call), so ``shape_exprs[-1]
            == cap_symbol``.
        aliasable:
            ``True`` iff the recipe is pure element identity with
            matching storage layout — the emitter can skip
            allocate/copy and emit one ``c_f_pointer`` per
            ``flat_names[i]`` aliasing ``read_exprs[i]`` (with
            index placeholders stripped).  The pass sets this
            based on rank + element-type match.  Mutually exclusive
            with ``aos_alloc``.
        scratch_dtype:
            SDFG element dtype the emitter declares for flat
            scratch buffers (``float64`` / ``int32`` / ...).
            Today all flats of one recipe share a dtype.
        aos_alloc:
            Phase 5c-B (AoS + allocatable / pointer array member at
            the SDFG-boundary dummy).  When ``True`` the emitter
            switches to the padding-to-max pack/unpack path:

                cap = max_i(merge(size(A(i)%w), 0, allocated(A(i)%w)))
                allocate(A_w(N, cap)); A_w = 0
                do i = 1, N; if (allocated(A(i)%w)) A_w(i, 1:size(A(i)%w)) = A(i)%w
                <SDFG call>
                do i = 1, N; if (allocated(A(i)%w)) A(i)%w = A_w(i, 1:size(A(i)%w))   ! intent(out)/(inout)
                deallocate(A_w)

            The companion buffer is always ``A_<member>(N, cap)``
            (single flat per recipe — multi-flat layouts like
            complex-split don't combine with ``aos_alloc``).  Mixed
            structs (one allocatable + one plain member) split
            across two recipes: one ``aos_alloc=True`` per
            allocatable member, one regular ``aliasable=True``
            covering the rest.
        cap_symbol:
            Name of the SDFG runtime symbol carrying the cap.
            Empty unless ``aos_alloc=True``; otherwise
            ``cap_<base>_<member>``.  ``_build_symbol_assigns``
            skips this symbol because the pack-in code computes it
            directly.
    """
    flat_names: Tuple[str, ...]
    read_exprs: Tuple[str, ...]
    write_expr: str = ''
    rank: int = 0
    shape_exprs: Tuple[str, ...] = field(default_factory=tuple)
    aliasable: bool = False
    scratch_dtype: str = 'float64'
    aos_alloc: bool = False
    cap_symbol: str = ''

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

    def to_dict(self) -> dict:
        return {'entries': [e.to_dict() for e in self.entries]}

    @classmethod
    def from_dict(cls, d: dict) -> 'FlattenPlan':
        """Rehydrate a plan from a plain dict — used by the bridge,
        which returns the MLIR-side ``hlfir.flatten_plan`` attribute as
        a nested dict of the same shape."""
        return cls(entries=tuple(FlattenEntry.from_dict(e) for e in d.get('entries', [])))

    def to_json(self, path: str) -> None:
        with open(path, 'w') as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'FlattenPlan':
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


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
