"""Named block builders — one function per Fortran section of the
generated wrapper module.  Each takes the canonical bundle
``(frozen, iface, plan)`` (or a subset) and returns one string
representing that block.

The builders are deliberately thin: they consume a template from
``templates/*.f90.in``, substitute the section-specific variables,
and return the rendered text.  All Fortran-construction logic that
depends on the flattening plan lives in ``loop_copy.py`` and is
called from ``build_wrapper_body`` / ``build_wrapper_tail``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from dace.frontend.hlfir.bindings.flatten_plan import (
    FlattenPlan, )
from dace.frontend.hlfir.bindings.fortran_interface import OriginalInterface
from dace.frontend.hlfir.bindings.frozen_signature import FrozenSignature
from dace.frontend.hlfir.bindings.loop_copy import (
    _fortran_type,
    render_alias_calls,
    render_copy_in_loop,
    render_copy_out_loop,
)

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _load(name: str) -> str:
    return (_TEMPLATE_DIR / name).read_text()


# ---------------------------------------------------------------------------
# bind(c) interface block
# ---------------------------------------------------------------------------


def build_c_interface(frozen: FrozenSignature, iface: OriginalInterface) -> str:
    """Render the ``interface ... end interface`` block declaring the
    three C entry points that the compiled SDFG exports.

    Args:
        frozen:  Frozen signature — drives the per-arg declarations.
        iface:   Outer interface — only ``iface.entry`` is read,
                 used in the ``bind(c, name='...')`` attribute.

    Returns:
        One rendered string containing the full ``interface`` block.

    Template:
        ``templates/c_interface.f90.in``

    Example fragment:
        interface
          function dace_init_kernel() bind(c, name='__dace_init_kernel') result(h)
            type(c_ptr) :: h
          end function
          subroutine dace_program_kernel(h, fld_a, fld_b) bind(c, name='__program_kernel')
            type(c_ptr), value :: h
            type(c_ptr), value :: fld_a
            type(c_ptr), value :: fld_b
          end subroutine
          function dace_exit_kernel(h) bind(c, name='__dace_exit_kernel') result(err)
            ...
          end function
        end interface
    """
    tpl = _load("c_interface.f90.in")
    header_lines: List[str] = []
    body_lines: List[str] = []
    for a in frozen.args:
        header_lines.append(f"      {a.sdfg_name}")
        if a.rank > 0:
            body_lines.append(f"      type(c_ptr), value :: {a.sdfg_name}")
        elif a.kind == 'symbol':
            body_lines.append(f"      integer(c_int), value :: {a.sdfg_name}")
        else:
            body_lines.append(f"      type(c_ptr), value :: {a.sdfg_name}")
    return tpl.format(entry=iface.entry,
                      c_arg_decls=",  &\n".join(header_lines),
                      c_arg_decls_body="\n".join(body_lines))


# ---------------------------------------------------------------------------
# Ref-counted handle state
# ---------------------------------------------------------------------------


def build_handle_state(iface: OriginalInterface) -> str:
    """Render the module-level ``dace_handle`` + ``init_count``
    declarations.

    Args:
        iface:  Only ``iface.entry`` is read (for comment text).

    Returns:
        Rendered block; the two ``save``-scoped variables that
        ``<entry>_dace`` and ``<entry>_dace_finalize`` share.

    Template:
        ``templates/handle_state.f90.in``
    """
    return _load("handle_state.f90.in").format(entry=iface.entry)


# ---------------------------------------------------------------------------
# Wrapper head — dummy decls, flat pointer / scratch decls, symbol / iter locals
# ---------------------------------------------------------------------------


def build_wrapper_head(frozen: FrozenSignature, iface: OriginalInterface, plan: FlattenPlan) -> str:
    """Render the ``<entry>_dace`` subroutine header + declaration
    section.

    Walks ``plan.entries``:
        * aliasable recipes → ``<type>, pointer :: <flat>(:,:,...)``
        * non-aliasable     → ``<type>, allocatable, target :: <flat>(:,:)``
    Free symbols that aren't already outer dummies become local
    ``integer(c_int)`` scalars; ``i1..iN`` loop iters are declared
    whenever any non-aliasable recipe exists.

    Args:
        frozen:  For free-symbol list.
        iface:   Outer dummies drive the subroutine signature.
        plan:    Drives pointer-vs-scratch decisions per flat.

    Returns:
        Everything from ``subroutine <entry>_dace(...)`` through the
        final declaration line.  Does NOT include the body.

    Template:
        ``templates/wrapper_head.f90.in``

    Example fragment:
        subroutine kernel_dace(fld, n, m)
          type(t_fields), intent(inout), target :: fld
          integer(c_int), intent(in),    target :: n
          integer(c_int), intent(in),    target :: m
          real(c_double), pointer :: fld_a(:,:)
          real(c_double), pointer :: fld_b(:,:)
          integer(c_int) :: dace_err
    """
    tpl = _load("wrapper_head.f90.in")
    outer_dummy_names = [a.name for a in iface.args]
    outer_dummy_set = set(outer_dummy_names)
    outer_dummy_decls = "\n".join(f"    {a.fortran_type}, intent({a.intent or 'inout'}), target :: {a.name}"
                                  f"{_dim_spec(a.shape)}" for a in iface.args)

    flat_ptr_lines: List[str] = []
    scratch_lines: List[str] = []
    max_loop_rank = 0
    for entry in plan.entries:
        r = entry.recipe
        ftype = _fortran_type(r.scratch_dtype)
        shape_dims = "(" + ", ".join(":" for _ in range(r.rank)) + ")"
        if r.aliasable:
            for flat in r.flat_names:
                flat_ptr_lines.append(f"    {ftype}, pointer :: {flat}{shape_dims}")
        else:
            max_loop_rank = max(max_loop_rank, r.rank)
            for flat in r.flat_names:
                scratch_lines.append(f"    {ftype}, allocatable, target :: {flat}{shape_dims}")

    symbol_decls = "\n".join(f"    integer(c_int) :: {s}" for s in frozen.free_symbols if s not in outer_dummy_set)
    if max_loop_rank:
        iter_decl = "    integer(c_int) :: " + ", ".join(f"i{d + 1}" for d in range(max_loop_rank))
        symbol_decls = (symbol_decls + "\n" + iter_decl) if symbol_decls else iter_decl

    return tpl.format(
        entry=iface.entry,
        outer_dummy_list=", ".join(outer_dummy_names),
        outer_dummy_decls=outer_dummy_decls or "    ! (no dummies)",
        flat_ptr_decls="\n".join(flat_ptr_lines) or "    ! (no flat pointers)",
        scratch_decls="\n".join(scratch_lines) or "    ! (no scratch)",
        symbol_decls=symbol_decls or "    ! (no free symbols)",
    )


# ---------------------------------------------------------------------------
# Wrapper body — per-entry alias calls / copy-in loops, symbol population
# ---------------------------------------------------------------------------


def build_wrapper_body(frozen: FrozenSignature, iface: OriginalInterface, plan: FlattenPlan) -> str:
    """Render the between-declaration-and-SDFG-call block — for each
    ``FlattenEntry`` either alias it (zero-copy) or allocate + copy
    in, then populate SDFG free symbols from ``size(...)`` on the
    outer storage.

    Args:
        frozen:  For the free-symbol set.
        iface:   To skip symbols that are already outer dummies.
        plan:    Drives per-entry alias-vs-copy emission.

    Returns:
        Indented Fortran lines ready to slot between the wrapper
        head's declarations and the SDFG call.
    """
    outer_dummy_set = {a.name for a in iface.args}
    body: List[str] = ["    ! ----- Copy-in / alias per flatten entry -----"]
    for entry in plan.entries:
        r = entry.recipe
        if r.aliasable:
            body.extend(render_alias_calls(r))
        else:
            body.extend(render_copy_in_loop(r))

    sym_lines = _build_symbol_assigns(frozen, plan, outer_dummy_set)
    if sym_lines:
        body.append("")
        body.append("    ! ----- Symbol population -----")
        body.extend(sym_lines)
    return "\n".join(body)


# ---------------------------------------------------------------------------
# Wrapper tail — init-count bump, SDFG call, copy-back, deallocate, end sub
# ---------------------------------------------------------------------------


def build_wrapper_tail(frozen: FrozenSignature, iface: OriginalInterface, plan: FlattenPlan) -> str:
    """Render the tail of the wrapper: init-count bump + ``call
    dace_program_<entry>`` + copy-back for every non-aliased
    writeable entry, then deallocate scratch, then close the
    subroutine.

    Args:
        frozen:  For the SDFG call argument list.
        iface:   For the entry name.
        plan:    For per-entry writeback decisions.

    Returns:
        Everything after ``build_wrapper_body`` through the final
        ``end subroutine <entry>_dace``.

    Template:
        ``templates/wrapper_call.f90.in`` supplies the init bump,
        the SDFG call, and the ``end subroutine`` + finalize marker.
        We splice the copy-back block in before the end marker.
    """
    tpl = _load("wrapper_call.f90.in")
    call_args = ",  &\n".join(f"      {a.sdfg_name}" for a in frozen.args)
    call_block = tpl.format(entry=iface.entry, call_arg_list=call_args)

    copy_out_lines: List[str] = []
    for entry in plan.entries:
        r = entry.recipe
        if r.aliasable:
            continue
        if not r.write_expr:
            continue
        if entry.writeback_intent not in ('out', 'inout'):
            continue
        copy_out_lines.extend(render_copy_out_loop(r, entry.outer_expr))

    if not copy_out_lines:
        return call_block

    copy_out_block = "\n    ! ----- Copy-out for writeable deep-copy entries -----\n" + "\n".join(copy_out_lines)
    marker = f"  end subroutine {iface.entry}_dace"
    pre, post = call_block.split(marker, 1)
    return pre + copy_out_block + "\n" + marker + post


# ---------------------------------------------------------------------------
# Finalize subroutine
# ---------------------------------------------------------------------------


def build_finalize(iface: OriginalInterface) -> str:
    """Placeholder — the finalize subroutine is baked into
    ``templates/wrapper_call.f90.in`` and emitted together with the
    main wrapper tail.  Kept as a named function so the coordinator
    has a uniform shape.

    Args:
        iface:  Unused today — kept for API symmetry.

    Returns:
        Empty string.  Reserved for a future split that moves the
        finalize body out of ``wrapper_call.f90.in``.
    """
    del iface  # unused
    return ""


# ---------------------------------------------------------------------------
# Module assembler
# ---------------------------------------------------------------------------


def assemble_module(iface: OriginalInterface, frozen: FrozenSignature, blocks: dict) -> str:
    """Stitch the rendered blocks into the complete Fortran module.

    Args:
        iface:   For ``iface.used_modules`` (use-only statements).
        frozen:  For the schema_version stamped in the header.
        blocks:  Dict of ``'c_interface' / 'handle_state' / 'wrapper_head'
                 / 'wrapper_body' / 'wrapper_tail' / 'finalize'`` → str.

    Returns:
        Complete Fortran module source.

    Template:
        ``templates/module.f90.in`` — three placeholders (use
        statements, c-interface, handle state, wrapper body,
        finalize body) plus the entry name + schema version.
    """
    use_statements = "\n".join(f"  use {mod}, only: {', '.join(syms)}"
                               for mod, syms in sorted(iface.used_modules.items()))
    wrapper_body = (blocks['wrapper_head'] + "\n" + blocks['wrapper_body'] + "\n" + blocks['wrapper_tail'])
    return _load("module.f90.in").format(
        entry=iface.entry,
        schema_version=frozen.schema_version,
        use_statements=use_statements,
        c_interface=blocks['c_interface'],
        handle_state=blocks['handle_state'],
        wrapper_body=wrapper_body,
        finalize_body=blocks['finalize'],
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dim_spec(shape) -> str:
    """Render the dimension spec suffix for an outer dummy's
    declaration.  Assumed-shape dummies use ``:``; explicit extents
    pass through."""
    if not shape:
        return ""
    return f", dimension({','.join(s if s != '?' else ':' for s in shape)})"


def _build_symbol_assigns(frozen: FrozenSignature, plan: FlattenPlan, outer_dummy_set: set) -> List[str]:
    """Emit ``sym = int(size(<outer>, dim=d), c_int)`` for every
    free symbol that isn't already an outer dummy.  Walks the plan
    to find the first recipe whose ``shape_exprs`` mention the
    symbol; falls back to a TODO comment if none does.
    """
    out: List[str] = []
    for sym in frozen.free_symbols:
        if sym in outer_dummy_set:
            continue
        found = False
        for entry in plan.entries:
            r = entry.recipe
            for d, shape in enumerate(r.shape_exprs):
                # Cheap substring check — ``size(st%a, dim=1)`` doesn't
                # mention the symbol directly; we'd need the shape to be
                # recorded symbolically ("n") for this to match.  Left as
                # a simple heuristic for now; future work: extend recipes
                # with symbolic shape metadata alongside the Fortran
                # ``size(...)`` expressions.
                if f"size({entry.outer_expr}, dim={d + 1})" in shape:
                    out.append(f"    {sym} = int({shape}, c_int)")
                    found = True
                    break
            if found:
                break
        if not found:
            out.append(f"    ! TODO: no plan entry gives size for free symbol {sym!r}")
    return out
