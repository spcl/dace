"""Named block builders  --  one function per Fortran section of the
generated wrapper module.  Each takes the canonical bundle
``(frozen, iface, plan)`` (or a subset) and returns one string
representing that block.

The builders are deliberately thin: they consume a template from
``templates/*.f90.in``, substitute the section-specific variables,
and return the rendered text.  All Fortran-construction logic that
depends on the flattening plan lives in ``loop_copy.py`` and is
called from ``build_wrapper_body`` / ``build_wrapper_tail``.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

from dace.frontend.hlfir.bindings.flatten_plan import (
    FlattenPlan, )
from dace.frontend.hlfir.bindings.fortran_interface import OriginalInterface
from dace.frontend.hlfir.bindings.frozen_signature import FrozenSignature
from dace.frontend.hlfir.bindings.loop_copy import (
    _fortran_type,
    render_alias_calls,
    render_aos_alloc_pack_in,
    render_aos_alloc_pack_out,
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
        frozen:  Frozen signature  --  drives the per-arg declarations.
        iface:   Outer interface  --  only ``iface.entry`` is read,
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
            # Real array, or length-1 wrapper for a scalar OUTPUT
            # (``intent(out)`` / ``intent(inout)``).  Either way DaCe
            # passes a pointer.
            body_lines.append(f"      type(c_ptr), value :: {a.sdfg_name}")
        elif a.kind == 'symbol':
            # Free symbol -- pass-by-value integer.
            body_lines.append(f"      integer(c_int), value :: {a.sdfg_name}")
        elif a.kind == 'scalar':
            # Scalar INPUT (``intent(in)`` or ``REAL(8), VALUE``) lives
            # as a non-transient ``Scalar`` on the SDFG -- DaCe codegen
            # emits a pass-by-value parameter, so the Fortran interface
            # must also bind by value (not via ``c_ptr``).
            body_lines.append(f"      {_fortran_c_value_type(a.dtype)}, value :: {a.sdfg_name}")
        else:
            body_lines.append(f"      type(c_ptr), value :: {a.sdfg_name}")
    return tpl.format(entry=iface.entry,
                      c_arg_decls=",  &\n".join(header_lines),
                      c_arg_decls_body="\n".join(body_lines))


def _fortran_c_value_type(dtype: str) -> str:
    """Map a frozen-arg ``dtype`` string to its ``iso_c_binding`` form
    for a pass-by-value Fortran dummy."""
    table = {
        'int32': 'integer(c_int)',
        'int64': 'integer(c_long_long)',
        'float32': 'real(c_float)',
        'float64': 'real(c_double)',
        'bool': 'logical(c_bool)',
    }
    if dtype not in table:
        raise ValueError(f"_fortran_c_value_type: unsupported scalar dtype {dtype!r} -- "
                         "extend the dtype map for new pass-by-value scalar shapes.")
    return table[dtype]


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
# Wrapper head  --  dummy decls, flat pointer / scratch decls, symbol / iter locals
# ---------------------------------------------------------------------------


def build_wrapper_head(frozen: FrozenSignature, iface: OriginalInterface, plan: FlattenPlan) -> str:
    """Render the ``<entry>_dace`` subroutine header + declaration
    section.

    Walks ``plan.entries``:
        * aliasable recipes -> ``<type>, pointer :: <flat>(:,:,...)``
        * non-aliasable     -> ``<type>, allocatable, target :: <flat>(:,:)``
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

    bridge_decls, _, _, _ = _build_logical_bridges(frozen, iface)
    if bridge_decls:
        scratch_lines = scratch_lines + bridge_decls

    return tpl.format(
        entry=iface.entry,
        outer_dummy_list=", ".join(outer_dummy_names),
        outer_dummy_decls=outer_dummy_decls or "    ! (no dummies)",
        flat_ptr_decls="\n".join(flat_ptr_lines) or "    ! (no flat pointers)",
        scratch_decls="\n".join(scratch_lines) or "    ! (no scratch)",
        symbol_decls=symbol_decls or "    ! (no free symbols)",
    )


# ---------------------------------------------------------------------------
# Wrapper body  --  per-entry alias calls / copy-in loops, symbol population
# ---------------------------------------------------------------------------


def build_wrapper_body(frozen: FrozenSignature, iface: OriginalInterface, plan: FlattenPlan) -> str:
    """Render the between-declaration-and-SDFG-call block  --  for each
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
        # Three mutually exclusive emitter shapes  --  see FlattenRecipe
        # for the flag matrix.
        if r.aos_alloc:
            body.extend(render_aos_alloc_pack_in(r, entry.outer_expr))
        elif r.aliasable:
            body.extend(render_alias_calls(r))
        else:
            body.extend(render_copy_in_loop(r))

    _, copy_in_lines, _, _ = _build_logical_bridges(frozen, iface)
    if copy_in_lines:
        body.append("")
        body.append("    ! ----- LOGICAL -> logical(c_bool) bridge (copy-in) -----")
        body.extend(copy_in_lines)

    sym_lines = _build_symbol_assigns(frozen, plan, outer_dummy_set)
    if sym_lines:
        body.append("")
        body.append("    ! ----- Symbol population -----")
        body.extend(sym_lines)
    return "\n".join(body)


# ---------------------------------------------------------------------------
# Wrapper tail  --  init-count bump, SDFG call, copy-back, deallocate, end sub
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
    _, _, bridge_copy_out, name_override = _build_logical_bridges(frozen, iface)

    # The C interface declares every array arg as ``type(c_ptr), value``
    # (see ``build_c_interface``).  Wrap each array actual with
    # ``c_loc(...)`` so Fortran's bind(c) conversion is explicit -- the
    # implicit conversion from a Fortran pointer / target array to
    # ``c_ptr`` only kicks in for ``intent``-typed dummies, NOT for
    # ``type(c_ptr), value`` dummies (gfortran rejects with "type
    # mismatch ... passed REAL/LOGICAL to TYPE(c_ptr)").  Scalars and
    # symbols pass by value -- no wrapping.
    def _call_actual(a) -> str:
        actual = name_override.get(a.sdfg_name, a.sdfg_name)
        if a.kind == 'array' or a.rank > 0:
            return f"c_loc({actual})"
        return actual

    call_args = ",  &\n".join(f"      {_call_actual(a)}" for a in frozen.args)
    call_block = tpl.format(entry=iface.entry, call_arg_list=call_args)

    copy_out_lines: List[str] = []
    for entry in plan.entries:
        r = entry.recipe
        if r.aos_alloc:
            if entry.writeback_intent in ('out', 'inout'):
                copy_out_lines.extend(render_aos_alloc_pack_out(r, entry.outer_expr))
            else:
                # intent(in): no copy-back, but the scratch buffer
                # was allocated unconditionally in pack-in and still
                # needs releasing.
                copy_out_lines.append(f"    deallocate({r.flat_names[0]})")
            continue
        if r.aliasable:
            continue
        if not r.write_expr:
            continue
        if entry.writeback_intent not in ('out', 'inout'):
            continue
        copy_out_lines.extend(render_copy_out_loop(r, entry.outer_expr))

    bridge_block = ""
    if bridge_copy_out:
        bridge_block = "\n    ! ----- logical(c_bool) -> LOGICAL bridge (copy-out + dealloc) -----\n" + "\n".join(
            bridge_copy_out)

    if not copy_out_lines and not bridge_copy_out:
        return call_block

    copy_out_block = ""
    if copy_out_lines:
        copy_out_block = "\n    ! ----- Copy-out for writeable deep-copy entries -----\n" + "\n".join(copy_out_lines)
    marker = f"  end subroutine {iface.entry}_dace"
    pre, post = call_block.split(marker, 1)
    return pre + copy_out_block + bridge_block + "\n" + marker + post


# ---------------------------------------------------------------------------
# Finalize subroutine
# ---------------------------------------------------------------------------


def build_finalize(iface: OriginalInterface) -> str:
    """Placeholder  --  the finalize subroutine is baked into
    ``templates/wrapper_call.f90.in`` and emitted together with the
    main wrapper tail.  Kept as a named function so the coordinator
    has a uniform shape.

    Args:
        iface:  Unused today  --  kept for API symmetry.

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
                 / 'wrapper_body' / 'wrapper_tail' / 'finalize'`` -> str.

    Returns:
        Complete Fortran module source.

    Template:
        ``templates/module.f90.in``  --  three placeholders (use
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
    declaration as a *postfix* shape (``name(d1,d2)``), not a
    ``dimension(...)`` attribute.  Postfix is the only form that
    works when the suffix lands after the ``::`` -- a leading comma
    plus ``dimension(...)`` after the ``::`` is read by Fortran as
    a SECOND variable declaration (so ``:: mask, dimension(n)``
    silently declares ``mask`` AND ``dimension`` of unknown rank).

    Assumed-shape dummies render the surviving ``?`` placeholders as
    ``:``; explicit extents pass through.  An empty shape leaves the
    declaration as a scalar (no suffix).
    """
    if not shape:
        return ""
    return f"({','.join(s if s != '?' else ':' for s in shape)})"


def _is_default_logical(fortran_type: str) -> bool:
    """Recognise a caller-visible Fortran LOGICAL declaration whose
    storage layout differs from ``logical(c_bool)``.

    Default ``logical`` is 4 bytes (``LOGICAL(KIND=4)``); ``logical(1)``
    /  ``logical(8)`` are different sizes again.  Only ``logical(c_bool)``
    matches the SDFG's bool storage directly  --  every other LOGICAL kind
    needs a copy-via-Fortran-intrinsic-cast at the wrapper boundary so
    the SDFG sees the correct 1-byte ``bool`` layout.
    """
    s = fortran_type.strip().lower()
    if s == 'logical':
        return True
    if s.startswith('logical(') and 'c_bool' not in s:
        return True
    return False


def _build_logical_bridges(frozen: FrozenSignature, iface: OriginalInterface):
    """Emit scratch buffers + entry/exit copies for any LOGICAL outer
    dummy that the SDFG sees as ``bool``.

    The C ABI binds the wrapper's outer ``logical`` (default 4-byte)
    dummy to a ``T*`` whose elements are 4 bytes wide; the SDFG expects
    1-byte ``bool*``.  Passing the outer dummy's address straight
    through corrupts every other element's read.  The fix is a
    ``logical(c_bool)`` scratch buffer with the same shape  --  Fortran's
    intrinsic LOGICAL-kind-conversion (``cbool_buf = outer``) handles
    the bit-fiddling, and ``c_loc(cbool_buf)`` is then safely passed
    to the SDFG.

    Returns:
        ``(decl_lines, copy_in_lines, copy_out_lines, name_override)``:
            * ``decl_lines``: scratch buffer declarations (one per
              affected dummy).  Empty if no dummy needs bridging.
            * ``copy_in_lines``: Fortran-intrinsic cast assignments
              run before the SDFG call.
            * ``copy_out_lines``: reverse-direction casts for
              ``intent(out)/inout`` dummies, run after the SDFG call.
            * ``name_override``: ``{sdfg_name: scratch_name}`` mapping
              the SDFG-call-side name should pass instead of the
              outer dummy when this dummy needs bridging.

    Bool dummies whose outer Fortran declaration is already
    ``logical(c_bool)`` need no bridge  --  pass-through is correct.
    Bool ``intent(in)`` scalars are pass-by-value; the C interface
    builder takes a ``logical(c_bool), value`` so the Fortran
    intrinsic cast happens at the call site instead of through a
    scratch buffer (handled separately in ``build_wrapper_tail``).
    """
    decl_lines: List[str] = []
    copy_in_lines: List[str] = []
    copy_out_lines: List[str] = []
    name_override: dict = {}

    iface_by_name = {a.name: a for a in iface.args}
    for fa in frozen.args:
        if fa.dtype != 'bool':
            continue
        oa = iface_by_name.get(fa.fortran_name)
        if oa is None:
            continue
        if not _is_default_logical(oa.fortran_type):
            continue
        # Array dummy  --  explicit scratch buffer + element-wise cast.
        if fa.rank > 0:
            scratch = f"{fa.fortran_name}_cbool"
            shape_dim = "(" + ",".join(":" for _ in range(fa.rank)) + ")"
            decl_lines.append(f"    logical(c_bool), allocatable, target :: {scratch}{shape_dim}")
            shape_args = ", ".join(f"size({oa.name}, dim={d + 1})" for d in range(fa.rank))
            copy_in_lines.append(f"    allocate({scratch}({shape_args}))")
            copy_in_lines.append(f"    {scratch} = {oa.name}")
            if oa.intent in ('out', 'inout', ''):
                copy_out_lines.append(f"    {oa.name} = {scratch}")
            copy_out_lines.append(f"    deallocate({scratch})")
            name_override[fa.sdfg_name] = scratch
        # Scalar bool dummy (fa.rank == 0): the SDFG declares the C
        # interface as ``logical(c_bool), value :: <name>``, but the
        # outer Fortran dummy is default ``logical`` (4 bytes).  A
        # direct call ``dace_program_X(flag, ...)`` makes gfortran
        # reject with ``Type mismatch ... passed LOGICAL(4) to
        # LOGICAL(1)`` -- there is no implicit kind conversion at the
        # call expression for a pass-by-value bind(c) dummy.
        #
        # The fix mirrors the array path: declare a local
        # ``logical(c_bool)`` temporary, run the Fortran-intrinsic
        # LOGICAL-kind cast on it (``flag_cbool = flag``), then pass
        # the temp.  ``name_override`` redirects the SDFG-call name to
        # the temp; the call-arg renderer in ``build_wrapper_tail``
        # already knows ``kind == 'scalar'`` means pass-by-value, so it
        # won't wrap with ``c_loc`` (correct: the interface wants the
        # value itself, not a c_ptr).
        else:
            scratch = f"{fa.fortran_name}_cbool"
            decl_lines.append(f"    logical(c_bool) :: {scratch}")
            copy_in_lines.append(f"    {scratch} = {oa.name}")
            if oa.intent in ('out', 'inout', ''):
                # Symmetric copy-back for intent(out)/inout scalars.
                # No deallocate -- this is a stack temporary, not
                # allocatable.
                copy_out_lines.append(f"    {oa.name} = {scratch}")
            name_override[fa.sdfg_name] = scratch
            continue

    return decl_lines, copy_in_lines, copy_out_lines, name_override


_OFFSET_SYM_RE = re.compile(r"^offset_(.+)_d(\d+)$")
_EXTENT_SYM_RE = re.compile(r"^(.+)_d(\d+)$")


def _sym_from_intrinsic(sym: str, frozen: FrozenSignature) -> Optional[Tuple[str, str, int]]:
    """Map a free SDFG symbol to the Fortran intrinsic that populates
    it from the caller's actual storage.

    ``offset_<arr>_d<i>``  -> ``("lbound", <fortran-expr>, i+1)``
    ``<arr>_d<i>`` (extent) -> ``("size",   <fortran-expr>, i+1)``

    ``<arr>`` is matched to a ``FrozenArg`` by ``sdfg_name``; the
    Fortran expression is the original dummy (or, for a flattened
    struct member, the ``st%u`` outer expression) so ``lbound`` /
    ``size`` query the array the caller actually passed.

    :param sym: a free symbol name.
    :param frozen: the frozen signature (arg metadata).
    :returns: ``(intrinsic, fortran_expr, dim)`` or ``None`` when the
        symbol isn't an offset/extent of a known array arg.
    """
    by_sdfg = {a.sdfg_name: a for a in frozen.args}

    def _expr(arr: str) -> Optional[str]:
        a = by_sdfg.get(arr)
        if a is None or a.kind != "array":
            return None
        return a.from_struct_member or a.fortran_name

    m = _OFFSET_SYM_RE.match(sym)
    if m:
        e = _expr(m.group(1))
        return ("lbound", e, int(m.group(2)) + 1) if e else None
    m = _EXTENT_SYM_RE.match(sym)
    if m:
        e = _expr(m.group(1))
        return ("size", e, int(m.group(2)) + 1) if e else None
    return None


def _build_symbol_assigns(frozen: FrozenSignature, plan: FlattenPlan, outer_dummy_set: set) -> List[str]:
    """Emit one assignment per free SDFG symbol from the caller's
    actual Fortran storage.

    A free symbol is either an array's per-dim lower bound
    (``offset_<arr>_d<i>`` -> ``lbound``) or extent
    (``<arr>_d<i>`` -> ``size``).  The struct-flatten plan supplies a
    precise ``size(st%a, dim=d)`` expression where it has one;
    otherwise we fall back to ``lbound``/``size`` on the arg's own
    Fortran expression (covers plain assumed-shape and non-default
    lower-bound dummies, which have no flatten entry).  Symbols that
    are themselves outer dummies are left for the caller to pass.
    """
    # Cap symbols of aos_alloc recipes are populated by the pack-in
    # code (``render_aos_alloc_pack_in`` writes ``cap_<m> = max_i(...)``)
    # before the SDFG call  --  skip them here so we don't emit a stray
    # TODO line or duplicate assignment.
    aos_cap_syms = {
        entry.recipe.cap_symbol
        for entry in plan.entries if entry.recipe.aos_alloc and entry.recipe.cap_symbol
    }
    out: List[str] = []
    for sym in frozen.free_symbols:
        if sym in outer_dummy_set:
            continue
        if sym in aos_cap_syms:
            continue
        found = False
        for entry in plan.entries:
            r = entry.recipe
            for d, shape in enumerate(r.shape_exprs):
                # Cheap substring check  --  ``size(st%a, dim=1)`` doesn't
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
        if found:
            continue
        # No flatten-plan size expr: derive the value directly from the
        # caller's array via lbound/size (closes the gap for plain
        # assumed-shape / non-default-lower-bound dummies, and is the
        # ONLY path that ever populates an ``offset_<arr>_d<i>``).
        intr = _sym_from_intrinsic(sym, frozen)
        if intr is not None:
            fn, expr, dim = intr
            out.append(f"    {sym} = int({fn}({expr}, dim={dim}), c_int)")
        else:
            out.append(f"    ! TODO: no plan entry gives size for free symbol {sym!r}")
    return out
