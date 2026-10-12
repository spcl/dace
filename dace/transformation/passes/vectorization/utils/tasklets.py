# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tasklet creation / classification / vectorized-code-emission helpers.

``materialise_lane_id_index_tile`` mints per-lane index tiles for the tile-op
path; the ``EmitCtx`` / ``_generate_code`` helpers pick per-template C++ from an
operator classification, falling back to a scalar lane loop.
"""

import ast
from dataclasses import dataclass

import dace
from dace import typeclass
from dace.memlet import Memlet

#: Label prefix of every tasklet :func:`materialise_lane_id_index_tile` mints. The tile pipeline
#: emits these itself, already at tile shape, so the tile-candidate gates recognise them by this
#: prefix and stay transparent to them the way they do to the vectorizer's own tile lib nodes.
LANE_ID_MATERIALISER_PREFIX = "lane_id_mat_"


def is_python_tasklet(node: "dace.nodes.Tasklet") -> bool:
    """Whether ``node``'s body is a Python expression, i.e. whether parsing it is even defined.

    Every pass that reads ``code.as_string`` as an expression raises on a non-Python body rather
    than declining. The tile remainder guard (a C++ ``abort()`` tasklet) is the one that
    reaches the vectorizer.
    """
    return node.language == dace.dtypes.Language.Python


def stripped_tasklet_body(node: "dace.nodes.Tasklet") -> str:
    """``node``'s code with surrounding whitespace and trailing ``;`` removed."""
    return node.code.as_string.strip().rstrip(";").strip()


def single_assignment(code: str) -> ast.Assign | None:
    """Parse ``code`` and return its body iff it is one bare ``Assign``."""
    try:
        tree = ast.parse((code or "").strip())
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return None
    return tree.body[0]


def is_vectorizable_tasklet(state: "dace.SDFGState", node: "dace.nodes.Tasklet") -> bool:
    """Whether the vectorizer may rewrite ``node`` INTO LANES.

    Adds the scope condition to :func:`is_python_tasklet`: a TOP-LEVEL tasklet -- one with no
    enclosing map scope -- has no lane dimension to be widened along, so widening it is a rewrite
    of scalar program-level code. A pass that only READS a tasklet's symbols, and rewrites the
    SDFG rather than the lane structure, wants :func:`is_python_tasklet` instead.
    """
    if state.entry_node(node) is None:
        return False
    return is_python_tasklet(node)


def lane_loop_code(widths: tuple[int, ...], rhs: str) -> str:
    """C++ body writing ``rhs`` to every lane of the row-major ``_out`` tile of shape ``widths``.

    Each dim is a ``constexpr``-bounded ``DACE_UNROLL`` loop over ``__l<d>``, which the compiler
    lowers to SIMD.

    :param widths: Tile shape, innermost-last.
    :param rhs: C++ expression stored per lane; may reference ``__l<d>``.
    """
    K = len(widths)
    parts = []
    for i in range(K):
        inner = 1
        for q in range(i + 1, K):
            inner *= widths[q]
        parts.append(f"__l{i}" if inner == 1 else f"(__l{i} * {inner})")
    flat = " + ".join(parts) if parts else "0"
    code_lines = []
    for d in range(K):
        code_lines.append(f"{'    ' * d}constexpr std::size_t __W{d} = {widths[d]};")
        code_lines.append(f"{'    ' * d}DACE_UNROLL")
        code_lines.append(f"{'    ' * d}for (std::size_t __l{d} = 0; __l{d} < __W{d}; ++__l{d}) {{")
    code_lines.append(f"{'    ' * K}_out[{flat}] = {rhs};")
    code_lines.extend(f"{'    ' * d}}}" for d in reversed(range(K)))
    return "\n".join(code_lines)


def materialise_lane_id_index_tile(
    inner_state: "dace.SDFGState",
    expr: str,
    iter_vars: tuple[str, ...],
    widths: tuple[int, ...],
    name_hint: str = "_sym_tile",
) -> "dace.nodes.AccessNode":
    """Mint a per-lane int64 tile = ``expr`` evaluated at ``(iter_var_k -> iter_var_k + __l_k)``
    for each tile dim ``k`` -- i.e. the function is EXPANDED INSIDE per lane, not widened as if
    contiguous.

    K=1, expr ``"ii"``         -> ``_tile[l] = (ii) + l``.
    K=1, expr ``"py_mod(ii,K)"`` -> ``_tile[l] = py_mod((ii + l), K)``  (cyclic per-lane index).
    K=2, expr ``"2 * ii + jj"`` -> ``_tile[l0, l1] = 2*(ii+l0) + (jj+l1)``.

    The lane loop is a ``constexpr``-bounded ``DACE_UNROLL`` (compiler lowers to SIMD), NOT a
    data-dependent CPP gather loop -- the sanctioned tile-op index materialisation. Shared by
    :meth:`ConvertTaskletsToTileOps._materialise_lane_id_tile` (lane-id symbol operands) and
    :meth:`InsertTileLoadStore._stage_index_via_tileops` (modular / non-affine gather indices).

    :returns: the ``AccessNode`` (wired as the fill tasklet's output) holding the
        ``(W_0, .., W_{K-1})`` index tile. Consumers MUST read THIS node so the scheduler orders
        the fill before the gather -- a fresh ``add_access`` of the same array is an unproduced
        orphan the gather may read before it is populated (uninitialised index -> OOB).
    """
    from dace import dtypes, symbolic
    from dace.codegen.common import sym2cpp

    sdfg = inner_state.sdfg
    widths = tuple(int(w) for w in widths)
    # Substitute each tile iter-var ``v -> (v + __l<k>)`` INSIDE the (possibly non-affine)
    # expression, then render to C++ via ``sym2cpp`` so ``**`` becomes multiplication, ``py_mod``
    # stays ``py_mod`` etc. -- a raw string keeps Python ``**`` which is invalid C++.
    parsed = symbolic.pystr_to_symbolic(expr)
    subs = {symbolic.symbol(v): symbolic.symbol(v) + symbolic.symbol(f"__l{k}") for k, v in enumerate(iter_vars)}
    body_expr = sym2cpp(parsed.subs(subs))
    # int64 by the never-narrow index rule -- this tile IS an index tile, and the gather that
    # consumes it must not see a narrowed offset. The C++ cast below reads the descriptor rather
    # than repeating the spelling, so the two can never disagree.
    arr_name, index_desc = sdfg.add_array(
        name_hint,
        shape=widths,
        dtype=dace.int64,
        transient=True,
        storage=dtypes.StorageType.Register,
        find_new_name=True,
    )
    tasklet = inner_state.add_tasklet(
        name=f"{LANE_ID_MATERIALISER_PREFIX}{arr_name}",
        inputs=set(),
        outputs={"_out"},
        code=lane_loop_code(widths, f"({index_desc.dtype.ctype})({body_expr})"),
        language=dtypes.Language.CPP,
    )
    out_an = inner_state.add_access(arr_name)
    out_subset = ", ".join(f"0:{w}" for w in widths)
    inner_state.add_edge(tasklet, "_out", out_an, None, Memlet(f"{arr_name}[{out_subset}]"))
    return out_an


# Operator tables for the CPP emit helpers (``_generate_code``); module-level so
# callers / tests can inspect them.
BINARY_OPERATORS = {"+", "-", "/", "*", "%", "&&", "||", "==", "!=", "<", "<=", ">", ">="}
# ``+`` excluded: unary ``+`` rejected by body (``raise Exception("Unary + …")``);
# keeping it out keeps the ``op in UNARY_OPERATORS`` check honest.
UNARY_OPERATORS = {"!", "-"}

# Dtype-cast op names (``float64`` / ``int32`` / ...). A kept ``dace.float64(x)``
# cast reaches emit as a 1-input "function" op named by its dtype; must lower to
# ``dace::<dtype>(x)`` (cppunparse typecast) -- bare ``float64(x)`` not valid C++.
# Built from the dtype registry so names never hardcoded.
CAST_OP_NAMES = frozenset(s.split("::")[-1] for s in dace.dtypes.TYPECLASS_TO_STRING.values())


def emit_op(op_: str) -> str:
    """C++ spelling of a fallback op: a dtype cast becomes ``dace::<dtype>``.

    :param op_: Op label (operator symbol, function name, or dtype cast name).
    :returns: ``dace::<dtype>`` for a cast op, else ``op_`` unchanged.
    """
    return f"dace::{op_}" if op_ in CAST_OP_NAMES else op_


def binop_cpp(l_op: str, op_: str, r_op: str) -> str:
    """C++ rendering of a binary operator in the CPP fallback lane loop; ``%`` floors, as ``py_mod``."""
    if op_ == "%":
        return f"py_mod({l_op}, {r_op})"
    return f"({l_op} {op_} {r_op})"


def _roundtrip_constant(s: int | float | str | None) -> int | float | str | None:
    """Return the constant verbatim for emission — no ``float()`` round-trip.

    Constant is only ever string-formatted into a C++ template, so the exact
    written form must survive: ``float(s)`` would turn ``"Infinity"`` -> ``"inf"``
    (invalid C++), drop sympy ``"oo"``, re-precision ``"0.1"`` ->
    ``"0.10000000149..."``, rewrite ``"2"`` -> ``"2.0"``.

    :param s: Constant (numeric or string literal), or ``None``.
    :returns: ``s`` unchanged.
    """
    return s


def _is_number(s: int | float | str | None) -> bool:
    """Whether ``s`` is a numeric literal (a constant, not a symbol).

    Recognises ints/floats, numeric strings, IEEE infinity (``inf`` /
    ``Infinity`` via ``float``), sympy ``oo`` / ``-oo``.

    :param s: Candidate token.
    :returns: ``True`` iff ``s`` denotes a numeric constant.
    """
    if s is None:
        return False
    if isinstance(s, (int, float)):
        return True
    txt = str(s).strip()
    try:
        float(txt)
        return True
    except ValueError:
        return txt.lstrip("+-") == "oo"


@dataclass(slots=True)
class EmitCtx:
    """Per-tasklet emission state shared by every per-``TaskletType`` emitter.

    ``mask_connector`` = name of an ``_iter_mask: bool[W]`` input connector; when
    set, the emitter routes to the ``op + "_masked"`` template + passes
    ``mask=<name>``, else uses the unsuffixed template.
    """

    state: dace.SDFGState
    node: dace.nodes.Tasklet
    templates: dict[str, str]
    vector_dtype: typeclass
    vector_width: int
    vector_map_param: str
    is_commutative: bool
    fallbackcode_due_to_types: bool
    mask_connector: str | None = None


def _template_key(ctx: EmitCtx, base_op: str) -> str:
    """Return the templates-dict key for ``base_op``, adjusted for masking.

    :param ctx: Emission context.
    :param base_op: Unmasked operator key.
    :returns: ``base_op + "_masked"`` if a mask connector is set and that key
        exists, else ``base_op`` unchanged.
    """
    if ctx.mask_connector is not None:
        masked = base_op + "_masked"
        if masked in ctx.templates:
            return masked
    return base_op


def _generate_code(
    ctx: EmitCtx,
    rhs1_: str | None,
    rhs2_: str | None,
    const1_: int | float | str | None,
    const2_: int | float | str | None,
    lhs_: str,
    op_: str,
) -> str:
    """Generate the vectorized C++ code string for one tasklet.

    Uses the matching template (array-array, array-scalar, constant variants,
    commutative / non-commutative) or a scalar lane-loop fallback. When
    ``ctx.mask_connector`` set, masked template variants are used + the fallback
    loop is iter-mask-gated.

    :param ctx: Emission context.
    :param rhs1_: First array operand, or ``None``.
    :param rhs2_: Second array operand, or ``None``.
    :param const1_: First constant operand, or ``None``.
    :param const2_: Second constant operand, or ``None``.
    :param lhs_: Output connector name.
    :param op_: Operator string.
    :returns: Generated C++ code string.
    :raises Exception: invalid operand configuration for the fallback.
    """

    # Get out edge and its dtype
    out_edges = ctx.state.out_edges(ctx.node)
    assert len(out_edges) == 1
    out_edge = out_edges[0]

    if out_edge.data.data is None:
        dtype_ = dace.dtypes.TYPECLASS_TO_STRING[ctx.vector_dtype]
    else:
        data_dtype = ctx.state.sdfg.arrays[out_edge.data.data].dtype
        dtype_ = dace.dtypes.TYPECLASS_TO_STRING[data_dtype]

    rhs_left = rhs1_ if rhs1_ is not None else const1_
    rhs_right = rhs2_ if rhs2_ is not None else const2_

    vw = ctx.vector_width
    templates = ctx.templates
    mask_arg = ctx.mask_connector or ""

    # Multiple dtypes involved - fallback code should be used
    if not ctx.fallbackcode_due_to_types:
        # Use template if available
        if op_ in templates:
            # One array + optional constant
            if rhs1_ is None or rhs2_ is None:
                rhs = rhs1_ if rhs1_ is not None else rhs2_
                constant = const1_ if const1_ is not None else const2_
                if constant is None:
                    # Single array or repeated array case
                    key = _template_key(ctx, op_)
                    return templates[key].format(
                        rhs1=rhs, rhs2=rhs, lhs=lhs_, op=op_, vector_width=vw, dtype=dtype_, mask=mask_arg
                    )
                else:
                    # Single array + constant
                    cop_ = None
                    if ctx.is_commutative or op_ == "=":
                        cop_ = op_ + "c"
                    elif constant == const1_:
                        cop_ = "c" + op_
                    else:
                        assert constant == const2_
                        cop_ = op_ + "c"
                    # constant version may not be in templates
                    if cop_ in templates:
                        key = _template_key(ctx, cop_)
                        return templates[key].format(
                            rhs1=rhs,
                            constant=_roundtrip_constant(constant),
                            lhs=lhs_,
                            op=op_,
                            vector_width=vw,
                            dtype=dtype_,
                            mask=mask_arg,
                        )

            else:
                # Two arrays
                key = _template_key(ctx, op_)
                return templates[key].format(
                    rhs1=rhs1_, rhs2=rhs2_, lhs=lhs_, op=op_, vector_width=vw, dtype=dtype_, mask=mask_arg
                )

    # Tasklet bodies must be free of Python ``if ... else ...``: canonicalize
    # passes needing a ternary emit ``ITE(c, t, e)`` (``dace.symbolic`` alias of
    # ``merge``), which ``classify_tasklet`` picks up as ``TERNARY_ARRAY`` +
    # dispatcher lowers via ``vector_select``. A surviving Python ``IfExp`` = a
    # producer-side bug: the comparison-suffix fallback below would silently drop
    # the arms + miscompile, so refuse loudly.
    code_str = (ctx.node.code.as_string or "").strip()
    if " if " in code_str and " else " in code_str:
        raise NotImplementedError(
            f"vectorization: tasklet {ctx.node.label!r} carries a Python ternary "
            f"({code_str!r}); producers must emit ``ITE(c, t, e)`` instead so the "
            f"vectorizer can lower it as a ``TERNARY_ARRAY``."
        )

    # Fallback: unsupported operator (or op with no ``_masked`` template). When
    # ``ctx.mask_connector`` set, the per-lane write MUST be iter-mask-gated: the
    # masked remainder runs this body once over the trailing R<W elements +
    # inactive lanes must NOT write back (an unconditional store clobbers live
    # array data — e.g. the pre-loop ``a[LEN_1D-1]`` scalar write in TSVC s2244).
    comparison_suffix = "? 1.0 : 0.0" if op_ in {">", ">=", "<", "<=", "==", "!="} else ""
    code_lines = [f"_dace_vectorize({ctx.vector_width})"]
    code_lines.append(f"for (int _vi = 0; _vi < {vw}; _vi += 1) {{")
    if ctx.mask_connector:
        code_lines.append(f"if ({ctx.mask_connector}[_vi]) {{")

    # Determine operand order
    lhs_expr = lhs_ + "[_vi]"
    rhs_left = rhs1_ if rhs1_ is not None else const1_
    rhs_right = rhs2_ if rhs2_ is not None else const2_

    if rhs_left is None or rhs_right is None:
        if op_ not in UNARY_OPERATORS and op_ in BINARY_OPERATORS:
            raise Exception(
                f"Invalid operand configuration for fallback vectorization. {rhs_left}, {rhs_right}, {lhs_expr}, {op_}"
            )

    if rhs_left is None or rhs_right is None:
        rhs = rhs_left if rhs_left is not None else rhs_right
        const = const1_ if const1_ is not None else const2_
        if op_ in UNARY_OPERATORS:
            if rhs_left == const:
                code_lines.append(f"{lhs_expr} = {op_}{rhs}{comparison_suffix};")
            else:
                code_lines.append(f"{lhs_expr} = {op_}({rhs}[_vi]){comparison_suffix};")
        elif op_ == "=":
            if rhs_left == const1_:
                code_lines.append(f"{lhs_expr} = {rhs};")
            else:
                code_lines.append(f"{lhs_expr} = {rhs}[_vi];")
        else:
            # Function-form op (incl. a dtype cast, lowered to ``dace::<dtype>``).
            if rhs_left == const:
                code_lines.append(f"{lhs_expr} = {emit_op(op_)}({rhs}){comparison_suffix};")
            else:
                code_lines.append(f"{lhs_expr} = {emit_op(op_)}({rhs}[_vi]){comparison_suffix};")
    else:
        if op_ in BINARY_OPERATORS:
            # Constant operand emitted bare; array operand indexed ``[_vi]``.
            # ``binop_cpp`` renders ``%`` as ``py_mod``, every other operator infix.
            l_operand = rhs_left if rhs_left == const1_ else f"{rhs_left}[_vi]"
            r_operand = rhs_right if rhs_right == const2_ else f"{rhs_right}[_vi]"
            code_lines.append(f"{lhs_expr} = {binop_cpp(l_operand, op_, r_operand)}{comparison_suffix};")
        else:
            if rhs_left == const1_:
                code_lines.append(f"{lhs_expr} = ({op_}({rhs_left}, {rhs_right}[_vi])){comparison_suffix};")
            elif rhs_right == const2_:
                code_lines.append(f"{lhs_expr} = ({op_}({rhs_left}[_vi], {rhs_right})){comparison_suffix};")
            else:
                code_lines.append(f"{lhs_expr} = ({op_}({rhs_left}[_vi], {rhs_right}[_vi])){comparison_suffix};")

    if ctx.mask_connector:
        code_lines.append("}")  # close ``if (mask[_vi]) {``
    code_lines.append("}")
    return "\n".join(code_lines)
