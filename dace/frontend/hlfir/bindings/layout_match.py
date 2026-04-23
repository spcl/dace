"""Per-argument strategy picker: given one ``FrozenArg`` (the inner
view — what the SDFG expects) and the matching ``OriginalArg`` /
``Member`` from the outer interface, decide how the wrapper should
bridge between them.

Three strategies today; easy to add more.

- ``AliasStrategy`` — same rank, same element type, caller memory is
  contiguous.  Emit ``c_loc`` + ``c_f_pointer`` — zero copy.
- ``ComplexSplitStrategy`` — outer is ``complex(kind)``, inner is two
  ``real(kind)`` arrays.  Emit a Fortran ``do`` loop that splits the
  real/imag parts; reverse loop after the SDFG call for
  intent(out/inout).
- ``ExplicitCopyStrategy`` — fallback: rank change, layout transpose,
  anything that needs a real copy but isn't complex-split.  Emit a
  generic ``do`` loop with user-supplied element expressions.

The rule the user asked for: **only deep-copy when the
dimensionality / element type actually differs**.  If both views see
the same shape + dtype, always alias.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

from dace.frontend.hlfir.bindings.fortran_interface import Member, OriginalArg
from dace.frontend.hlfir.bindings.frozen_signature import FrozenArg

# ---------------------------------------------------------------------------
# Strategy tags
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AliasStrategy:
    """Zero-copy: ``call c_f_pointer(c_loc(<outer_expr>), <inner_name>,
    shape=[<dim_exprs>])``."""
    outer_expr: str  # 'st%u' / 'a'
    inner_name: str  # 'st_u' / 'a'
    shape_exprs: Tuple[str, ...]  # ['size(st%u, dim=1)', 'size(st%u, dim=2)']


@dataclass(frozen=True)
class ComplexSplitStrategy:
    """Fortran complex → two real arrays (re + im).  Emits a copy-in
    loop before the SDFG call; copy-out loop after if the intent is
    writeable."""
    outer_expr: str  # 'st%z'
    re_name: str  # 'st_z_re'
    im_name: str  # 'st_z_im'
    shape_exprs: Tuple[str, ...]
    writeback: bool = False  # True for intent(out|inout)


@dataclass(frozen=True)
class ExplicitCopyStrategy:
    """Generic deep-copy fallback.  ``element_expr`` is the Fortran
    expression computing one element from the outer source; reversed
    for writeback (intent out|inout)."""
    outer_expr: str
    inner_name: str
    shape_exprs: Tuple[str, ...]
    element_expr: str  # 'real(<outer_expr>(i,j), kind=c_double)'
    writeback: bool = False
    writeback_expr: str = ''


Strategy = Union[AliasStrategy, ComplexSplitStrategy, ExplicitCopyStrategy]

# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def _is_complex(fortran_type: str) -> bool:
    return fortran_type.strip().lower().startswith('complex')


def _element_real_kind(fortran_type: str) -> str:
    """For ``complex(c_double)`` return ``c_double``; for real kinds
    pass through."""
    t = fortran_type.strip().lower()
    if t.startswith('complex('):
        return t[len('complex('):-1]
    if t.startswith('real('):
        return t[len('real('):-1]
    return 'c_double'


def decide_strategy(frozen: FrozenArg, outer: Union[OriginalArg, Member]) -> Strategy:
    """Pick a strategy for one ``(FrozenArg, OriginalArg-or-Member)``
    pair.

    Inputs:
        frozen: what the SDFG expects (inner view).
        outer:  what the caller passes (outer view, possibly a struct
                member).

    Returns one of the three ``*Strategy`` dataclasses.
    """
    outer_name = getattr(outer, 'name', None)
    outer_expr = frozen.from_struct_member or outer_name or frozen.sdfg_name
    # Shape expressions the wrapper can compute at runtime from the
    # caller's storage — ``size(<outer>, dim=d)`` is always valid.
    shape_exprs = tuple(f"size({outer_expr}, dim={d + 1})" for d in range(frozen.rank))

    # Complex-split — outer is complex, inner is two real arrays.
    if _is_complex(outer.fortran_type) and frozen.layout == 'complex_split':
        return ComplexSplitStrategy(
            outer_expr=outer_expr,
            re_name=frozen.sdfg_name,  # convention: the base name is _re
            im_name=f"{frozen.sdfg_name}_im" if not frozen.sdfg_name.endswith('_re') else f"{frozen.sdfg_name[:-3]}_im",
            shape_exprs=shape_exprs,
            writeback=frozen.intent in ('out', 'inout'),
        )

    # Same rank + same element type → zero-copy alias.
    same_rank = (outer.rank == frozen.rank)
    if same_rank and not _is_complex(outer.fortran_type):
        return AliasStrategy(
            outer_expr=outer_expr,
            inner_name=frozen.sdfg_name,
            shape_exprs=shape_exprs,
        )

    # Fallback — rank mismatch, etc.
    return ExplicitCopyStrategy(
        outer_expr=outer_expr,
        inner_name=frozen.sdfg_name,
        shape_exprs=shape_exprs,
        element_expr=f"real({outer_expr}(<idx>), kind={_element_real_kind(outer.fortran_type)})",
        writeback=frozen.intent in ('out', 'inout'),
        writeback_expr=f"{outer_expr}(<idx>) = {frozen.sdfg_name}(<idx>)",
    )
