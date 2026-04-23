"""String-template Fortran emitter — turns ``(FrozenSignature,
OriginalInterface)`` into a ``<entry>_bindings.f90`` file.

Templates live in ``templates/*.f90.in`` and are applied with plain
``str.format``.  No Jinja / no fparser dependency — the generated code
is linear and human-readable.

This is the v1 skeleton: it emits the module / `bind(c)` interface
block / ref-counted handle / wrapper subroutine shell.  The per-arg
body (alias vs copy loops, symbol population, SDFG call) is built up
from the layout-match strategies in ``layout_match.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from dace.frontend.hlfir.bindings.fortran_interface import OriginalInterface
from dace.frontend.hlfir.bindings.frozen_signature import FrozenArg, FrozenSignature
from dace.frontend.hlfir.bindings.layout_match import (
    AliasStrategy,
    ComplexSplitStrategy,
    ExplicitCopyStrategy,
    decide_strategy,
)

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _load(name: str) -> str:
    return (_TEMPLATE_DIR / name).read_text()


# ----------------------------------------------------------------------------
# Dtype → Fortran iso_c_binding kind
# ----------------------------------------------------------------------------

_DTYPE_TO_F = {
    'float64': ('real', 'c_double'),
    'float32': ('real', 'c_float'),
    'int32': ('integer', 'c_int'),
    'int64': ('integer', 'c_long'),
    'bool': ('logical', 'c_bool'),
}


def _f_type(dtype: str) -> str:
    """``'float64'`` → ``'real(c_double)'``."""
    base, kind = _DTYPE_TO_F.get(dtype, ('real', 'c_double'))
    return f"{base}({kind})"


# ----------------------------------------------------------------------------
# C-interface argument declarations
# ----------------------------------------------------------------------------


def _c_arg_decls(frozen: FrozenSignature) -> Tuple[str, str]:
    """Return ``(header_continuation, body_decls)`` for the
    ``bind(c)`` interface subroutine.  Header continuation is the
    ``, arg1, arg2, &``-style list; body_decls are the
    ``type(c_ptr), value :: arg1`` lines inside the interface.
    """
    header: List[str] = []
    body: List[str] = []
    for a in frozen.args:
        header.append(f"      {a.sdfg_name}")
        if a.rank > 0:
            body.append(f"      type(c_ptr), value :: {a.sdfg_name}")
        elif a.kind == 'symbol':
            # Symbols are passed by value as ints.
            body.append(f"      integer({_DTYPE_TO_F[a.dtype][1]}), value :: {a.sdfg_name}")
        else:
            # Size-1 array scalar — DaCe passes by reference.
            body.append(f"      type(c_ptr), value :: {a.sdfg_name}")
    return ",  &\n".join(header), "\n".join(body)


# ----------------------------------------------------------------------------
# Main emitter
# ----------------------------------------------------------------------------


def emit_bindings(frozen: FrozenSignature, iface: OriginalInterface, out_path: str) -> Path:
    """Write ``<out_path>`` (typically ``<entry>_bindings.f90``) from
    the frozen signature + the original Fortran interface.

    V1 scope: module header + bind(c) interface + ref-counted handle
    + wrapper subroutine shell with one-pass-through per arg using
    ``decide_strategy``.  Symbol population and call list are
    populated; the per-member copy-in/out loop bodies are emitted
    for ``ComplexSplitStrategy`` / ``ExplicitCopyStrategy`` as
    TODO-flagged stubs (rendered as comments with the exact Fortran
    shape the binding consumer can fill in).  Subsequent revisions
    will generate real loop bodies once we grow a per-dtype kernel
    library.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    module_tpl = _load("module.f90.in")
    c_iface_tpl = _load("c_interface.f90.in")
    handle_tpl = _load("handle_state.f90.in")
    wrapper_head_tpl = _load("wrapper_head.f90.in")
    wrapper_call_tpl = _load("wrapper_call.f90.in")
    alias_tpl = _load("alias.f90.in")

    # --- bind(c) interface block ---
    c_header_cont, c_body = _c_arg_decls(frozen)
    c_interface = c_iface_tpl.format(
        entry=iface.entry,
        c_arg_decls=c_header_cont,
        c_arg_decls_body=c_body,
    )

    handle_state = handle_tpl.format(entry=iface.entry)

    # --- wrapper head: dummy arg decls + flat ptr + scratch + symbols ---
    outer_dummy_names = [a.name for a in iface.args]
    outer_dummy_decls = "\n".join(f"    {a.fortran_type}, intent({a.intent or 'inout'}), target :: {a.name}"
                                  f"{_dim_spec(a.shape)}" for a in iface.args)

    # For each frozen array arg, decide its strategy once — the
    # emitted body + the per-arg declarations (pointer vs allocatable
    # scratch) both depend on it.
    outer_dummy_set = {a.name for a in iface.args}
    strategies = _strategies_by_arg(frozen, iface)

    flat_ptr_lines: List[str] = []
    scratch_lines: List[str] = []
    for fa in frozen.args:
        if fa.kind != 'array':
            continue
        ftype = _f_type(fa.dtype)
        shape = "(:" + ", :" * (fa.rank - 1) + ")"
        strat = strategies.get(fa.sdfg_name)
        if isinstance(strat, AliasStrategy):
            # c_f_pointer aliases the caller's storage → pointer.
            flat_ptr_lines.append(f"    {ftype}, pointer :: {fa.sdfg_name}{shape}")
        else:
            # Deep-copy target: allocate real scratch memory.
            dim_decl = ",".join([":" for _ in range(fa.rank)])
            scratch_lines.append(f"    {ftype}, allocatable, target :: {fa.sdfg_name}({dim_decl})")

    # Free-symbol declarations — skip any that are already outer dummies
    # (Fortran won't let you redeclare a dummy as a local).
    symbol_decls = "\n".join(f"    integer(c_int) :: {s}" for s in frozen.free_symbols if s not in outer_dummy_set)

    # Loop-iter locals for emitted copy-in / copy-out nests.  Pick the
    # max rank among deep-copy strategies and declare ``i1..iN``.
    max_copy_rank = 0
    for strat in strategies.values():
        if isinstance(strat, (ComplexSplitStrategy, ExplicitCopyStrategy)):
            max_copy_rank = max(max_copy_rank, len(strat.shape_exprs))
    loop_iter_decls = ""
    if max_copy_rank:
        loop_iter_decls = "    integer(c_int) :: " + ", ".join(f"i{d + 1}" for d in range(max_copy_rank))

    sym_and_loop_decls = symbol_decls
    if loop_iter_decls:
        sym_and_loop_decls = (sym_and_loop_decls + "\n" + loop_iter_decls) if sym_and_loop_decls else loop_iter_decls

    wrapper_head = wrapper_head_tpl.format(
        entry=iface.entry,
        outer_dummy_list=", ".join(outer_dummy_names),
        outer_dummy_decls=outer_dummy_decls or "    ! (no dummies)",
        flat_ptr_decls="\n".join(flat_ptr_lines) or "    ! (no flat pointers)",
        scratch_decls="\n".join(scratch_lines) or "    ! (no scratch)",
        symbol_decls=sym_and_loop_decls or "    ! (no free symbols)",
    )

    # --- per-arg body ---
    # Track complex-split pairs so we emit the copy loop once per
    # source (not once per _re + _im frozen arg).
    emitted_complex_sources: set = set()

    copy_in_lines: List[str] = []
    copy_out_lines: List[str] = []
    alloc_lines: List[str] = []
    dealloc_lines: List[str] = []
    for fa in frozen.args:
        if fa.kind != 'array':
            continue
        strat = strategies.get(fa.sdfg_name)
        if strat is None:
            copy_in_lines.append(f"    ! TODO: no outer match for frozen arg {fa.sdfg_name!r}")
            continue
        if isinstance(strat, AliasStrategy):
            copy_in_lines.append(
                alias_tpl.format(outer_expr=strat.outer_expr,
                                 inner_name=strat.inner_name,
                                 shape_list=", ".join(strat.shape_exprs)).rstrip())
            continue
        if isinstance(strat, ComplexSplitStrategy):
            if strat.outer_expr in emitted_complex_sources:
                continue  # pair already emitted from its _re partner
            emitted_complex_sources.add(strat.outer_expr)
            alloc_lines.extend(_allocate_pair_lines(strat))
            copy_in_lines.extend(_complex_copy_in(strat))
            if strat.writeback:
                copy_out_lines.extend(_complex_copy_out(strat))
            dealloc_lines.extend([
                f"    deallocate({strat.re_name})",
                f"    deallocate({strat.im_name})",
            ])
            continue
        if isinstance(strat, ExplicitCopyStrategy):
            copy_in_lines.append(f"    ! TODO: explicit copy-in for {strat.outer_expr} → "
                                 f"{strat.inner_name} (shape={strat.shape_exprs})")

    # Symbol population — use size() on whichever outer arg carries
    # the symbol in its shape.  Skip symbols that are already outer
    # dummies; the caller-supplied value wins.
    sym_assigns = [
        line for line in _symbol_assigns(frozen, iface)
        if not any(line.lstrip().startswith(f"{s} =") for s in outer_dummy_set)
    ]

    body_lines: List[str] = []
    if alloc_lines:
        body_lines.append("    ! ----- Scratch allocations for deep-copy members -----")
        body_lines.extend(alloc_lines)
    body_lines.append("    ! ----- Per-argument copy-in / alias -----")
    body_lines.extend(copy_in_lines)
    if sym_assigns:
        body_lines.append("\n    ! ----- Symbol population -----")
        body_lines.extend(sym_assigns)

    # --- SDFG call list ---
    call_args = ",  &\n".join(f"      {a.sdfg_name}" for a in frozen.args)
    wrapper_call = wrapper_call_tpl.format(
        entry=iface.entry,
        call_arg_list=call_args,
    )

    # --- copy-out (intent out/inout, deep-copy strategies) -----------
    copy_out_block = ""
    if copy_out_lines or dealloc_lines:
        copy_out_block = "\n    ! ----- Per-argument copy-back -----\n" + "\n".join(copy_out_lines + dealloc_lines)

    # --- Assemble the module ---
    # wrapper_call_tpl has ``end subroutine <entry>_dace`` baked in;
    # insert the copy-back block BEFORE that end-subroutine marker.
    call_pieces = wrapper_call.split(f"  end subroutine {iface.entry}_dace", 1)
    wrapper_call = call_pieces[0] + copy_out_block + f"\n  end subroutine {iface.entry}_dace" + call_pieces[1]

    wrapper_body = wrapper_head + "\n" + "\n".join(body_lines) + "\n" + wrapper_call

    use_statements = "\n".join(f"  use {mod}, only: {', '.join(syms)}"
                               for mod, syms in sorted(iface.used_modules.items()))

    out = module_tpl.format(
        entry=iface.entry,
        schema_version=frozen.schema_version,
        use_statements=use_statements,
        c_interface=c_interface,
        handle_state=handle_state,
        wrapper_body=wrapper_body,
        finalize_body="",  # baked into wrapper_call_tpl already
    )

    out_path.write_text(out)
    return out_path


# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------


def _dim_spec(shape: Tuple[str, ...]) -> str:
    """Return ``'(:,:)'`` / ``''`` / ``'(n,m)'`` for an argument's
    shape list (Fortran dimension spec).  Assumed-shape is the
    default when shapes come from the bridge with ``?``."""
    if not shape:
        return ""
    return f", dimension({','.join(s if s != '?' else ':' for s in shape)})"


def _resolve_outer(frozen_arg: FrozenArg, iface: OriginalInterface):
    """Find the outer arg / struct member that matches a frozen arg.

    If ``frozen_arg.from_struct_member`` is set (e.g. ``'st%u'``), we
    look up the member on the enclosing struct; otherwise the
    ``fortran_name`` should match an outer dummy directly.
    """
    if frozen_arg.from_struct_member and '%' in frozen_arg.from_struct_member:
        outer_name, member_name = frozen_arg.from_struct_member.split('%', 1)
        outer_arg = next((a for a in iface.args if a.name == outer_name), None)
        if outer_arg is None or outer_arg.struct_type is None:
            return None
        st = iface.struct_types.get(outer_arg.struct_type)
        if st is None:
            return None
        return next((m for m in st.members if m.name == member_name), None)
    return next((a for a in iface.args if a.name == frozen_arg.fortran_name), None)


def _strategies_by_arg(frozen: FrozenSignature, iface: OriginalInterface) -> dict:
    """Run ``decide_strategy`` once per frozen array arg, keyed by
    ``sdfg_name``.  Returns ``{name: Strategy | None}``."""
    out: dict = {}
    for fa in frozen.args:
        if fa.kind != 'array':
            continue
        outer = _resolve_outer(fa, iface)
        out[fa.sdfg_name] = decide_strategy(fa, outer) if outer is not None else None
    return out


def _allocate_pair_lines(strat) -> List[str]:
    """Per-pair allocation for ComplexSplitStrategy.  The _re and _im
    arrays share the same shape, computed from the outer complex
    source via ``size(<outer>, dim=d)``."""
    shape = ", ".join(strat.shape_exprs)
    return [
        f"    allocate({strat.re_name}({shape}))",
        f"    allocate({strat.im_name}({shape}))",
    ]


def _complex_copy_in(strat) -> List[str]:
    """Fortran nested-loop that splits complex source into _re / _im.
    Uses one index variable per rank (i1, i2, ...).  Lines are
    already indented with 4 spaces (wrapper-body indent)."""
    rank = len(strat.shape_exprs)
    idx_vars = [f"i{d + 1}" for d in range(rank)]
    out: List[str] = [f"    ! Complex-split copy-in: {strat.outer_expr} → "
                      f"{strat.re_name} / {strat.im_name}"]
    # Loop headers — outer first so memory accesses are column-major.
    for d in reversed(range(rank)):
        out.append(f"    {' ' * ((rank - 1 - d) * 2)}do {idx_vars[d]} = 1, {strat.shape_exprs[d]}")
    body_indent = ' ' * (rank * 2)
    idx_expr = ", ".join(idx_vars)
    out.append(f"    {body_indent}{strat.re_name}({idx_expr}) = "
               f" real({strat.outer_expr}({idx_expr}), kind=c_double)")
    out.append(f"    {body_indent}{strat.im_name}({idx_expr}) = "
               f"aimag({strat.outer_expr}({idx_expr}))")
    for d in range(rank):
        out.append(f"    {' ' * ((rank - 1 - d) * 2)}end do")
    return out


def _complex_copy_out(strat) -> List[str]:
    """Reverse of _complex_copy_in for intent(out|inout)."""
    rank = len(strat.shape_exprs)
    idx_vars = [f"i{d + 1}" for d in range(rank)]
    out: List[str] = [f"    ! Complex-split copy-out: "
                      f"{strat.re_name} / {strat.im_name} → {strat.outer_expr}"]
    for d in reversed(range(rank)):
        out.append(f"    {' ' * ((rank - 1 - d) * 2)}do {idx_vars[d]} = 1, {strat.shape_exprs[d]}")
    body_indent = ' ' * (rank * 2)
    idx_expr = ", ".join(idx_vars)
    out.append(f"    {body_indent}{strat.outer_expr}({idx_expr}) = cmplx("
               f"{strat.re_name}({idx_expr}), {strat.im_name}({idx_expr}), kind=c_double)")
    for d in range(rank):
        out.append(f"    {' ' * ((rank - 1 - d) * 2)}end do")
    return out


def _render_strategy(strat, alias_tpl: str, fa: FrozenArg) -> str:
    if isinstance(strat, AliasStrategy):
        return alias_tpl.format(
            outer_expr=strat.outer_expr,
            inner_name=strat.inner_name,
            shape_list=", ".join(strat.shape_exprs),
        )
    if isinstance(strat, ComplexSplitStrategy):
        return (f"    ! TODO: complex-split copy-in for {strat.outer_expr} → "
                f"{strat.re_name}/{strat.im_name} (shape={strat.shape_exprs})")
    if isinstance(strat, ExplicitCopyStrategy):
        return (f"    ! TODO: explicit copy-in for {strat.outer_expr} → "
                f"{strat.inner_name} (shape={strat.shape_exprs})")
    return "    ! unknown strategy"


def _symbol_assigns(frozen: FrozenSignature, iface: OriginalInterface) -> List[str]:
    """Pick a source dummy for each free symbol and emit
    ``sym = size(<outer>, dim=d)``.  Best-effort for v1: first outer
    arg whose frozen-side shape mentions the symbol.  Unresolved
    symbols emit a TODO comment."""
    out: List[str] = []
    for sym in frozen.free_symbols:
        picked = False
        for fa in frozen.args:
            if fa.kind != 'array':
                continue
            outer = _resolve_outer(fa, iface)
            if outer is None:
                continue
            for dim, s in enumerate(fa.shape):
                if s == sym:
                    outer_expr = fa.from_struct_member or getattr(outer, 'name', fa.fortran_name)
                    out.append(f"    {sym} = int(size({outer_expr}, dim={dim + 1}), c_int)")
                    picked = True
                    break
            if picked:
                break
        if not picked:
            out.append(f"    ! TODO: no outer source found for free symbol {sym!r}")
    return out
