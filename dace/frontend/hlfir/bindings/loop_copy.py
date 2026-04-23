"""Recipe renderers — the ONLY code that knows how to turn a
``FlattenRecipe`` into Fortran.

Three functions, one per code shape the emitter needs:

- ``render_alias_calls`` — zero-copy path: one ``c_f_pointer`` per
  flat name, aliasing the outer storage path.
- ``render_copy_in_loop`` — forward path: allocate flats, rank-N
  nested ``do`` loop computing each flat's element from the outer.
- ``render_copy_out_loop`` — reverse path: rank-N nested ``do`` loop
  reconstructing the outer's element from the flats, followed by
  matching deallocates.

All three return ``List[str]`` — individual source lines already
indented to wrapper-body level (four spaces).  Callers concatenate
and hand to ``assemble_module``.
"""
from __future__ import annotations

from typing import List, Tuple

from dace.frontend.hlfir.bindings.flatten_plan import (
    FlattenRecipe,
    strip_index_args,
    substitute_indices,
)

_DTYPE_TO_F = {
    'float64': 'real(c_double)',
    'float32': 'real(c_float)',
    'int32': 'integer(c_int)',
    'int64': 'integer(c_long)',
    'bool': 'logical(c_bool)',
}


def _fortran_type(dtype: str) -> str:
    """Map a DaCe dtype string to its Fortran iso_c_binding form."""
    return _DTYPE_TO_F.get(dtype, 'real(c_double)')


def _loop_index_names(rank: int) -> Tuple[str, ...]:
    """Return the loop-index names the emitter declares at the
    wrapper head: ``('i1', 'i2', ..., 'iN')``."""
    return tuple(f"i{d + 1}" for d in range(rank))


# ----------------------------------------------------------------------------
# Alias path
# ----------------------------------------------------------------------------


def render_alias_calls(recipe: FlattenRecipe) -> List[str]:
    """Zero-copy alias emission for an ``aliasable=True`` recipe.

    Args:
        recipe:
            The recipe.  ``aliasable`` must be True; behaviour is
            undefined otherwise.

    Returns:
        One-line-per-flat ``call c_f_pointer(c_loc(<outer>), <flat>,
        [<shape>])`` statements, indented four spaces.  Empty list
        if the recipe has no flats (defensive).

    Example:
        Recipe for ``fld%a`` with flat name ``fld_a``:
            call c_f_pointer(c_loc(fld%a), fld_a,  &
                             [size(fld%a, dim=1), size(fld%a, dim=2)])
    """
    if not recipe.aliasable:
        raise ValueError("render_alias_calls called on non-aliasable recipe")
    shape_list = ", ".join(recipe.shape_exprs)
    out: List[str] = []
    for flat, read_expr in zip(recipe.flat_names, recipe.read_exprs):
        base = strip_index_args(read_expr)
        out.append(f"    call c_f_pointer(c_loc({base}), {flat}, [{shape_list}])")
    return out


# ----------------------------------------------------------------------------
# Forward copy (outer → flats)
# ----------------------------------------------------------------------------


def render_copy_in_loop(recipe: FlattenRecipe) -> List[str]:
    """Generic forward copy: allocate flats, nested do-loops assign
    each flat from the corresponding ``read_expr`` with loop-index
    placeholders substituted.

    Args:
        recipe:
            The recipe.  ``aliasable`` must be False; ``rank`` ≥ 1.

    Returns:
        Indented Fortran lines — allocates followed by the nested
        do-nest followed by the closing ``end do`` markers.

    Example:
        Recipe for complex ``st%z``, flat names ``st_z_re`` /
        ``st_z_im``, rank 2:
            allocate(st_z_re(size(st%z, dim=1), size(st%z, dim=2)))
            allocate(st_z_im(size(st%z, dim=1), size(st%z, dim=2)))
            ! Copy-in for st%z → st_z_re, st_z_im
            do i2 = 1, size(st%z, dim=2)
              do i1 = 1, size(st%z, dim=1)
                st_z_re(i1, i2) = real(st%z(i1, i2), kind=c_double)
                st_z_im(i1, i2) = aimag(st%z(i1, i2))
              end do
            end do
    """
    if recipe.aliasable:
        raise ValueError("render_copy_in_loop called on aliasable recipe — use render_alias_calls")
    out: List[str] = []
    # Allocate every flat using the per-rank shape expressions.
    for flat in recipe.flat_names:
        out.append(f"    allocate({flat}({', '.join(recipe.shape_exprs)}))")

    # Loop nest — outermost rank first (column-major).
    idx_names = _loop_index_names(recipe.rank)
    # Comment mentions the outer source with substituted indices so
    # placeholders never leak out.
    summary = substitute_indices(recipe.read_exprs[0], idx_names)
    out.append(f"    ! Copy-in: {', '.join(recipe.flat_names)} ← {summary}")
    for d in reversed(range(recipe.rank)):
        indent = ' ' * ((recipe.rank - 1 - d) * 2)
        out.append(f"    {indent}do {idx_names[d]} = 1, {recipe.shape_exprs[d]}")

    # Body — one assignment per flat with placeholders substituted.
    body_indent = ' ' * (recipe.rank * 2)
    idx_tuple = ", ".join(idx_names)
    for flat, read_expr in zip(recipe.flat_names, recipe.read_exprs):
        rhs = substitute_indices(read_expr, idx_names)
        out.append(f"    {body_indent}{flat}({idx_tuple}) = {rhs}")

    # Closing markers, innermost-first.
    for d in range(recipe.rank):
        indent = ' ' * ((recipe.rank - 1 - d) * 2)
        out.append(f"    {indent}end do")
    return out


# ----------------------------------------------------------------------------
# Reverse copy (flats → outer) + dealloc
# ----------------------------------------------------------------------------


def render_copy_out_loop(recipe: FlattenRecipe, outer_expr: str) -> List[str]:
    """Inverse of ``render_copy_in_loop``: pack the flat buffers
    back into the outer storage at each position, then deallocate.

    Args:
        recipe:
            The recipe.  Requires ``write_expr`` to be non-empty;
            empty write_expr means the caller shouldn't be invoking
            copy-out.
        outer_expr:
            The Fortran expression for the outer destination
            (``st%z``).  Same as the ``FlattenEntry.outer_expr`` the
            recipe was created under.  Passed separately so the
            renderer doesn't have to reach back up to the entry.

    Returns:
        Indented Fortran lines — the reverse do-nest followed by
        the matching ``deallocate`` statements.

    Example:
        Complex-split recipe for ``st%z``:
            ! Copy-out: st%z ← st_z_re, st_z_im
            do i2 = 1, size(st%z, dim=2)
              do i1 = 1, size(st%z, dim=1)
                st%z(i1, i2) = cmplx(st_z_re(i1,i2), st_z_im(i1,i2), kind=c_double)
              end do
            end do
            deallocate(st_z_re)
            deallocate(st_z_im)
    """
    if not recipe.write_expr:
        raise ValueError("render_copy_out_loop called on recipe with empty write_expr")
    out: List[str] = [f"    ! Copy-out: {outer_expr} ← {', '.join(recipe.flat_names)}"]

    idx_names = _loop_index_names(recipe.rank)
    for d in reversed(range(recipe.rank)):
        indent = ' ' * ((recipe.rank - 1 - d) * 2)
        out.append(f"    {indent}do {idx_names[d]} = 1, {recipe.shape_exprs[d]}")

    body_indent = ' ' * (recipe.rank * 2)
    idx_tuple = ", ".join(idx_names)
    outer_lhs = f"{outer_expr}({idx_tuple})"
    rhs = substitute_indices(recipe.write_expr, idx_names)
    out.append(f"    {body_indent}{outer_lhs} = {rhs}")

    for d in range(recipe.rank):
        indent = ' ' * ((recipe.rank - 1 - d) * 2)
        out.append(f"    {indent}end do")

    for flat in recipe.flat_names:
        out.append(f"    deallocate({flat})")
    return out
