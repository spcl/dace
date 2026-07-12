# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tasklet creation / classification / vectorized-code-emission helpers.

``materialise_lane_id_index_tile`` mints per-lane index tiles for the tile-op
path; the ``EmitCtx`` / ``_generate_code`` helpers pick per-template C++ from an
operator classification, falling back to a scalar lane loop.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import dace
from dace.memlet import Memlet
from dace import typeclass


def materialise_lane_id_index_tile(inner_state, expr: str, iter_vars: Tuple[str, ...], widths: Tuple[int, ...],
                                   name_hint: str = "_sym_tile") -> "dace.nodes.AccessNode":
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
    K = len(widths)
    # Substitute each tile iter-var ``v -> (v + __l<k>)`` INSIDE the (possibly non-affine)
    # expression, then render to C++ via ``sym2cpp`` so ``**`` becomes multiplication, ``py_mod``
    # stays ``py_mod`` etc. -- a raw string keeps Python ``**`` which is invalid C++.
    parsed = symbolic.pystr_to_symbolic(expr)
    subs = {symbolic.symbol(v): symbolic.symbol(v) + symbolic.symbol(f"__l{k}") for k, v in enumerate(iter_vars)}
    body_expr = sym2cpp(parsed.subs(subs))
    arr_name, _ = sdfg.add_array(name_hint,
                                 shape=widths,
                                 dtype=dace.int64,
                                 transient=True,
                                 storage=dtypes.StorageType.Register,
                                 find_new_name=True)
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
    code_lines.append(f"{'    ' * K}_out[{flat}] = (int64_t)({body_expr});")
    for d in reversed(range(K)):
        code_lines.append(f"{'    ' * d}}}")
    tasklet = inner_state.add_tasklet(name=f"lane_id_mat_{arr_name}",
                                      inputs=set(),
                                      outputs={"_out"},
                                      code="\n".join(code_lines),
                                      language=dtypes.Language.CPP)
    out_an = inner_state.add_access(arr_name)
    out_subset = ", ".join(f"0:{w}" for w in widths)
    inner_state.add_edge(tasklet, "_out", out_an, None, Memlet(f"{arr_name}[{out_subset}]"))
    return out_an


# Operator tables for the CPP emit helpers (``_generate_code``); module-level so
# callers / tests can inspect them.
PYTHON_TO_CPP_OPERATORS = {"and": "&&", "or": "||", "not": "!"}
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
    """C++ rendering of a binary operator in the CPP fallback lane loop.

    ``%`` -> ``py_mod(l, r)``: Python/NumPy modulo (result follows the *divisor's*
    sign), NOT C's truncated ``%`` (dividend's) -- matches the scalar reference on
    negative operands + well-formed for floats (C ``%`` ill-formed there).
    ``py_mod`` = GLOBAL runtime helper (outside ``dace::math``), called
    unqualified, same form the tile-op backends / ``np.mod`` ufunc emit. Every
    other operator emitted infix.

    :param l_op: Left operand expression.
    :param op_: Operator symbol.
    :param r_op: Right operand expression.
    :returns: C++ expression string.
    """
    if op_ == "%":
        return f"py_mod({l_op}, {r_op})"
    return f"({l_op} {op_} {r_op})"


def _roundtrip_constant(s: Union[int, float, str, None]):
    """Return the constant verbatim for emission — no ``float()`` round-trip.

    Constant is only ever string-formatted into a C++ template, so the exact
    written form must survive: ``float(s)`` would turn ``"Infinity"`` -> ``"inf"``
    (invalid C++), drop sympy ``"oo"``, re-precision ``"0.1"`` ->
    ``"0.10000000149..."``, rewrite ``"2"`` -> ``"2.0"``.

    :param s: Constant (numeric or string literal), or ``None``.
    :returns: ``s`` unchanged.
    """
    return s


def _is_number(s: Union[int, float, str, None]) -> bool:
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


@dataclass
class EmitCtx:
    """Per-tasklet emission state shared by every per-``TaskletType`` emitter.

    ``mask_connector`` = name of an ``_iter_mask: bool[W]`` input connector; when
    set, the emitter routes to the ``op + "_masked"`` template + passes
    ``mask=<name>``, else uses the unsuffixed template.
    """
    state: dace.SDFGState
    node: dace.nodes.Tasklet
    templates: Dict[str, str]
    vector_dtype: typeclass
    vector_width: int
    vector_map_param: str
    is_commutative: bool
    fallbackcode_due_to_types: bool
    mask_connector: Optional[str] = None


def _emit_ite_with_symbol_arms(ctx: EmitCtx) -> str:
    """Per-lane C++ select for ``ITE(cond, then, else)`` with symbol arms (1
    array input + 2 symbol/literal arms).

    Lowering for the canonicalize ``EarlyExitToFindIndex`` phi tasklet (``__out =
    ITE(__t0, _loop_it_0, LEN_1D)``): cond is an array connector, arms are
    loop-index symbols / literals. Parses the 3 ``ITE`` args from the tasklet's
    Python code, emits per-lane ``out[lane] = cond[lane] ? then(lane) : else``
    with the vectorized map param shifted to ``(<param> + _vi)`` inside arms so a
    lane-index symbol walks W values per call.

    :param ctx: Emission context.
    :returns: Generated C++ for the per-lane select.
    :raises NotImplementedError: tasklet shape isn't ``__out = ITE(cond, t, e)``.
    """
    import ast
    import re

    code_str = (ctx.node.code.as_string or "").strip()
    rhs = code_str.split(" = ", 1)[1] if " = " in code_str else code_str
    try:
        tree = ast.parse(rhs, mode="eval").body
    except SyntaxError as ex:
        raise NotImplementedError(f"_emit_ite_with_symbol_arms: parse failed on {rhs!r}: {ex}")
    if not (isinstance(tree, ast.Call) and isinstance(tree.func, ast.Name) and tree.func.id in ('ITE', 'merge')
            and len(tree.args) == 3):
        raise NotImplementedError(f"_emit_ite_with_symbol_arms: expected ``ITE(c, t, e)``, got {rhs!r}")
    out_conns = list(ctx.node.out_connectors.keys())
    if len(out_conns) != 1:
        raise NotImplementedError(f"_emit_ite_with_symbol_arms: expected 1 output connector, got {out_conns}")
    out_conn = out_conns[0]
    in_conns = list(ctx.node.in_connectors.keys())

    def _shift(expr: str) -> str:
        """Substitute ``conn`` -> ``conn[_vi]`` for in-connectors and shift
        the vectorized map param to ``(<param> + _vi)``."""
        for c in in_conns:
            expr = re.sub(rf"\b{re.escape(c)}\b", f"{c}[_vi]", expr)
        if ctx.vector_map_param and re.search(rf"\b{re.escape(ctx.vector_map_param)}\b", expr):
            expr = re.sub(rf"\b{re.escape(ctx.vector_map_param)}\b", f"({ctx.vector_map_param} + _vi)", expr)
        return expr

    cond = _shift(ast.unparse(tree.args[0]))
    then_arm = _shift(ast.unparse(tree.args[1]))
    else_arm = _shift(ast.unparse(tree.args[2]))

    vw = ctx.vector_width
    lines = [f"_dace_vectorize({vw})", f"for (int _vi = 0; _vi < {vw}; _vi += 1) {{"]
    if ctx.mask_connector:
        lines.append(f"if ({ctx.mask_connector}[_vi]) {{")
    lines.append(f"{out_conn}[_vi] = ({cond}) ? ({then_arm}) : ({else_arm});")
    if ctx.mask_connector:
        lines.append("}")
    lines.append("}")
    return "\n".join(lines)


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


def _generate_code(ctx: EmitCtx, rhs1_, rhs2_, const1_, const2_, lhs_, op_) -> str:
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
                    return templates[key].format(rhs1=rhs,
                                                 rhs2=rhs,
                                                 lhs=lhs_,
                                                 op=op_,
                                                 vector_width=vw,
                                                 dtype=dtype_,
                                                 mask=mask_arg)
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
                        return templates[key].format(rhs1=rhs,
                                                     constant=_roundtrip_constant(constant),
                                                     lhs=lhs_,
                                                     op=op_,
                                                     vector_width=vw,
                                                     dtype=dtype_,
                                                     mask=mask_arg)

            else:
                # Two arrays
                key = _template_key(ctx, op_)
                return templates[key].format(rhs1=rhs1_,
                                             rhs2=rhs2_,
                                             lhs=lhs_,
                                             op=op_,
                                             vector_width=vw,
                                             dtype=dtype_,
                                             mask=mask_arg)

    # Tasklet bodies must be free of Python ``if ... else ...``: canonicalize
    # passes needing a ternary emit ``ITE(c, t, e)`` (``dace.symbolic`` alias of
    # ``merge``), which ``classify_tasklet`` picks up as ``TERNARY_ARRAY`` +
    # dispatcher lowers via ``vector_select``. A surviving Python ``IfExp`` = a
    # producer-side bug: the comparison-suffix fallback below would silently drop
    # the arms + miscompile, so refuse loudly.
    code_str = (ctx.node.code.as_string or "").strip()
    if " if " in code_str and " else " in code_str:
        raise NotImplementedError(f"vectorization: tasklet {ctx.node.label!r} carries a Python ternary "
                                  f"({code_str!r}); producers must emit ``ITE(c, t, e)`` instead so the "
                                  f"vectorizer can lower it as a ``TERNARY_ARRAY``.")

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
                f"Invalid operand configuration for fallback vectorization. {rhs_left}, {rhs_right}, {lhs_expr}, {op_}")

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
            # ``binop_cpp`` renders ``%`` as ``dace::math::py_mod`` (Python
            # semantics), every other operator infix.
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


def _set_template(ctx: EmitCtx, rhs1_, rhs2_, const1_, const2_, lhs_, op_) -> None:
    ctx.node.code = dace.properties.CodeBlock(
        code=_generate_code(ctx, rhs1_, rhs2_, _roundtrip_constant(const1_), _roundtrip_constant(const2_), lhs_, op_),
        language=dace.Language.CPP,
    )


def _binary_expr(l_op: str, op: str, r_op: str) -> str:
    """Binary expression string for the scalar/symbol lane paths.

    A named-function op (``int_floor``, ``int_ceil``, ``min``, ``max``, ...) ->
    call syntax ``op(l, r)``; an operator symbol (``+``, ``<``, ...) -> infix
    ``(l op r)``. Without this a function op was written infix
    (``LEN_1D int_floor 2``) + later failed to sympify (TSVC s276's
    ``int_floor(LEN_1D, 2)`` comparison RHS).

    :param l_op: Left operand.
    :param op: Operator symbol or function name.
    :param r_op: Right operand.
    :return: Expression string.
    """
    if op.isidentifier():
        return f"{op}({l_op}, {r_op})"
    return f"({l_op} {op} {r_op})"


def _connector_reads_invariant_scalar(state: dace.SDFGState, node: dace.nodes.Tasklet, conn: str,
                                      vector_map_param: str) -> bool:
    """Whether input connector ``conn`` reads a lane-invariant value.

    A subset NOT mentioning the vectorized map param is constant across the W
    lanes -- every lane reads the same memory. Two shapes:

    * Unwidened length-1 read (TSVC s176 ``b[i+m-j-1] * c[j]``: outer ``j``
      constant inside the inner ``i`` loop -- subset stays ``[j:j+1]``).
    * Inner subset widened to ``[0:W]`` while the outer-side access was a
      constant index (TSVC s113 ``a[i] = a[0] + b[i]``: widening inflates
      ``a[0:1]`` -> ``a[0:8]`` over the full ``a`` but no ``i`` enters the subset).

    Both route through the ``vector_*_w_scalar`` (broadcast) template; the
    dispatcher dereferences pointer operands via :func:`_scalar_operand_expr`.

    :param state: State containing the tasklet.
    :param node: Tasklet whose input edges are inspected.
    :param conn: Input connector name.
    :param vector_map_param: The vectorized map parameter.
    :return: ``True`` when the connector reads a lane-invariant value.
    """
    # Authoritative per-lane signal: an INNER access subset still mentioning the
    # vectorized map param -> every lane reads a DIFFERENT element = per-lane
    # data, never a broadcast. Legacy widening rewrites many inner views to
    # ``[0:W]`` (dropping the param), so a subset STILL carrying it is unambiguous
    # + must short-circuit the outer-begin heuristic below. Gather-sibling case: a
    # contiguous operand ``c[i:i+W]`` alongside a packed gather whose OUTER NSDFG
    # boundary memlet is the whole array ``c[0:N]`` (begin 0) -- the outer-begin
    # rule would else mis-read it as a broadcast + collapse W lanes to ``c[0]``
    # (TSVC s4113 ``a[ip[i]] = b[ip[i]] + c[i]``).
    for ie in state.in_edges(node):
        if ie.dst_conn == conn and ie.data is not None and ie.data.subset is not None:
            if vector_map_param in {str(s) for s in ie.data.subset.free_symbols}:
                return False
            break
    # Inner subset alone not enough: legacy widening rewrites the inner view's
    # index to ``[0:W]`` for every connector, so neither ``a[0]`` (broadcast) nor
    # ``b[i]`` (sliding window) mentions ``vector_map_param`` inside the body. The
    # OUTER NSDFG-boundary memlet's BEGIN differentiates them:
    #   * ``b[i:i+8]`` -- begin == i -> base pointer slides per iter -> vector read
    #   * ``a[0:i+8]`` -- begin == 0 -> base pointer constant -> every lane reads
    #                     ``a[0]`` -> broadcast
    nsdfg_node = state.sdfg.parent_nsdfg_node
    parent_state = nsdfg_node.sdfg.parent if nsdfg_node is not None else None
    if nsdfg_node is None or parent_state is None:
        # Top-level vectorize: no outer memlet, so classify from the inner subset
        # alone. A length-1 subset independent of the lane param = unambiguous
        # lane-invariant (TSVC s176-shape). A multi-element subset is
        # lane-invariant ONLY when a fixed-position sub-slice of a LARGER source
        # array -- legacy widening inflates an invariant ``a[0]`` -> ``a[0:W]``
        # (TSVC s113) but keeps its lane-invariant begin. A multi-element subset
        # spanning an ENTIRE tile-width transient (e.g. ``B_slice_times_2[0:W]``)
        # = per-lane data, not a broadcast -- treating it as a scalar collapses W
        # lane values to one + miscompiles
        # ``test_knob_only_apply_vectorization_pass_bypass``.
        for ie in state.in_edges(node):
            if ie.dst_conn == conn and ie.data.data is not None:
                sub = ie.data.subset
                if vector_map_param in {str(s) for s in sub.free_symbols}:
                    return False
                try:
                    if int(sub.num_elements()) == 1:
                        return True
                except (TypeError, ValueError):
                    return False
                desc = state.sdfg.arrays.get(ie.data.data)
                if desc is None:
                    return False
                # Broadcast iff the read does NOT span the whole descriptor: a
                # sub-slice of a larger array (widened invariant read) vs a
                # whole tile-width transient (genuine per-lane data).
                try:
                    return dace.symbolic.simplify(sub.num_elements() - desc.total_size) != 0
                except (TypeError, ValueError):
                    return False
        return False
    # Match the outer in-edge whose dst_conn == our connector name -- NSDFG
    # connector names mirror the inner data names.
    for ie in state.in_edges(node):
        if ie.dst_conn != conn or ie.data.data is None:
            continue
        outer_conn = ie.data.data
        for outer_ie in parent_state.in_edges(nsdfg_node):
            if outer_ie.dst_conn != outer_conn or outer_ie.data.data is None:
                continue
            outer_sub = outer_ie.data.subset
            # A packed per-lane buffer -- the ``multiplexed_*`` transient from the
            # halve-index rewrite (``a[i // 2]``; see utils/multiplex.py), or a
            # packed strided-load buffer -- is a W-wide TRANSIENT read in FULL:
            # every lane holds a DISTINCT value = per-lane data, NOT a broadcast,
            # even though its begin is the constant 0 (the outer-begin heuristic
            # below would else collapse the W lanes to element 0 + miscompile e.g.
            # TSVC s4117 ``a[i] = b[i] + c[i // 2] * d[i]``). Mirror the top-level
            # "spans the whole descriptor" rule.
            outer_desc = parent_state.sdfg.arrays.get(outer_ie.data.data)
            if outer_desc is not None and outer_desc.transient:
                try:
                    if (int(outer_sub.num_elements()) > 1
                            and dace.symbolic.simplify(outer_sub.num_elements() - outer_desc.total_size) == 0):
                        return False
                except (TypeError, ValueError):
                    pass
            # ALL dims' begins must be lane-invariant for a clean broadcast.
            for (b, _e, _s) in outer_sub:
                if vector_map_param in {str(s) for s in b.free_symbols}:
                    return False
            return True
        return False
    return False


def _scalar_operand_expr(state: dace.SDFGState, node: dace.nodes.Tasklet, conn: str) -> str:
    """C++ expression that reads the lane-invariant scalar through ``conn``.

    DaCe codegen materialises a tasklet input connector as by-value ``T`` (Scalar,
    or Array shape ``(1,)`` whose memlet covers one element -> read collapses to a
    scalar at the C++ boundary) or as a ``T*`` pointer (Array whose memlet covers
    >1 elements, e.g. the inner-widened ``a[0:W]`` of TSVC s113). Broadcast
    template wants the scalar VALUE: emit ``conn`` when a value, else ``conn[0]``
    to dereference.

    :param state: State containing the tasklet.
    :param node: Tasklet whose input edges are inspected.
    :param conn: Input connector name (classified lane-invariant by
        :func:`_connector_reads_invariant_scalar`).
    :returns: C++ rvalue expression for the scalar.
    """
    for ie in state.in_edges(node):
        if ie.dst_conn != conn or ie.data.data is None:
            continue
        desc = state.sdfg.arrays.get(ie.data.data)
        if isinstance(desc, dace.data.Scalar):
            return conn
        # Array shape (1,) + 1-element memlet: codegen emits the connector as a
        # by-value scalar (TSVC s176: ``double __in2 = c[0];``). Subscripting it
        # would be invalid C++.
        try:
            shape_is_one = (isinstance(desc, dace.data.Array) and len(desc.shape) == 1
                            and bool(dace.symbolic.simplify(desc.shape[0] - 1) == 0))
            ne_is_one = int(ie.data.subset.num_elements()) == 1
        except (TypeError, ValueError):
            shape_is_one = False
            ne_is_one = False
        if shape_is_one and ne_is_one:
            return conn
        return f"{conn}[0]"
