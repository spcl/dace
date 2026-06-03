# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tasklet creation / classification / vectorized-code-emission helpers.

The emission core ``instantiate_tasklet_from_info`` picks per-template C++
code from the ``TaskletType`` classification in ``dace.sdfg.tasklet_utils``,
falling back to a scalar lane loop when no template applies.
"""
import copy
import re
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple, Union

import dace
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg.graph import Edge
from dace import typeclass
import dace.sdfg.tasklet_utils as tutil

from dace.transformation.passes.vectorization.utils.code_rewrite import (
    offset_symbol_in_expression,
    use_laneid_symbol_in_expression,
)
from dace.transformation.passes.vectorization.utils.name_schemes import PackedNameScheme, VecNameScheme


def match_connector_to_data(state: dace.SDFGState, tasklet: dace.nodes.Tasklet) -> dict:
    """Map a tasklet's input connectors to their array descriptors.

    :param state: The state containing the tasklet.
    :param tasklet: The tasklet whose connectors are inspected.
    :returns: Mapping from input connector name to the array descriptor.
    """
    tdict = dict()
    for ie in state.in_edges(tasklet):
        if ie.data is not None:
            tdict[ie.dst_conn] = state.sdfg.arrays[ie.data.data]
    return tdict


def is_assignment_tasklet(node: dace.nodes.Tasklet) -> bool:
    """Check whether a tasklet is a simple one-in one-out assignment.

    Matches ``a = b`` or ``a = b;``.

    :param node: The Tasklet to check.
    :returns: True iff it is a single assignment tasklet.
    """
    if (len(node.in_connectors) == 1 and len(node.out_connectors) == 1):
        in_conn = next(iter(node.in_connectors.keys()))
        out_conn = next(iter(node.out_connectors.keys()))
        body = node.code.as_string.strip().rstrip(";").rstrip()
        return body == f"{out_conn} = {in_conn}"
    return False


_VECTOR_COPY_CALL_RE = re.compile(r"\bvector_copy\s*\(")


def is_vector_assign_tasklet(t: dace.nodes.Tasklet) -> bool:
    """Check whether a tasklet performs a ``vector_copy`` call.

    The match is word-boundary anchored so ``my_vector_copy(`` does not
    falsely match; comments / string literals are not stripped.

    :param t: The tasklet to check.
    :returns: True iff the tasklet's code contains a ``vector_copy(`` call.
    """
    return _VECTOR_COPY_CALL_RE.search(t.code.as_string) is not None


# Operator tables consumed by ``instantiate_tasklet_from_info``. Kept at
# module level so callers / tests can inspect them without poking at
# function-local state.
PYTHON_TO_CPP_OPERATORS = {"and": "&&", "or": "||", "not": "!"}
BINARY_OPERATORS = {"+", "-", "/", "*", "%", "&&", "||", "==", "!=", "<", "<=", ">", ">="}
# ``+`` is excluded — unary ``+`` is rejected by the body with
# ``raise Exception("Unary + …")``; keeping it out of the set lets the
# ``op in UNARY_OPERATORS`` check stay honest.
UNARY_OPERATORS = {"!", "-"}


def _str_to_float_or_str(s: Union[int, float, str, None]):
    if s is None:
        return s
    try:
        return float(s)
    except ValueError:
        return s


def _is_number(s: str) -> bool:
    return s is not None and _str_to_float_or_str(s) != s


@dataclass
class EmitCtx:
    """Per-tasklet emission state shared by every per-``TaskletType`` emitter.

    Bundles the values that ``_generate_code`` and the per-type emitters need.
    ``mask_connector`` is the name of an ``_iter_mask: bool[W]`` input
    connector; when set the emitter routes to the ``op + "_masked"`` template
    and passes ``mask=<name>``, otherwise it uses the unsuffixed template.
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
    ``ctx.mask_connector`` is set, masked template variants are used and the
    fallback loop is iter-mask-gated.

    :param ctx: Emission context.
    :param rhs1_: First array operand, or ``None``.
    :param rhs2_: Second array operand, or ``None``.
    :param const1_: First constant operand, or ``None``.
    :param const2_: Second constant operand, or ``None``.
    :param lhs_: Output connector name.
    :param op_: Operator string.
    :returns: The generated C++ code string.
    :raises Exception: on an invalid operand configuration for the fallback.
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
                    # Maybe this constant version is not implemented in templates
                    if cop_ in templates:
                        key = _template_key(ctx, cop_)
                        return templates[key].format(rhs1=rhs,
                                                     constant=_str_to_float_or_str(constant),
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

    # Fallback: unsupported operator (or op with no ``_masked`` template).
    # When ``ctx.mask_connector`` is set the per-lane write MUST be gated
    # by the iter-mask: the masked remainder runs this body once over
    # the trailing R<W elements and the inactive lanes must NOT write
    # back (an unconditional store clobbers live array data — e.g. the
    # pre-loop ``a[LEN_1D-1]`` scalar write in TSVC s2244).
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
            if rhs_left == const:
                code_lines.append(f"{lhs_expr} = {op_}({rhs}){comparison_suffix};")
            else:
                code_lines.append(f"{lhs_expr} = {op_}({rhs}[_vi]){comparison_suffix};")
    else:
        if op_ in BINARY_OPERATORS:
            if rhs_left == const1_:
                code_lines.append(f"{lhs_expr} = ({rhs_left} {op_} {rhs_right}[_vi]){comparison_suffix};")
            elif rhs_right == const2_:
                code_lines.append(f"{lhs_expr} = ({rhs_left}[_vi] {op_} {rhs_right}){comparison_suffix};")
            else:
                code_lines.append(f"{lhs_expr} = ({rhs_left}[_vi] {op_} {rhs_right}[_vi]){comparison_suffix};")
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
        code=_generate_code(ctx, rhs1_, rhs2_, _str_to_float_or_str(const1_), _str_to_float_or_str(const2_), lhs_, op_),
        language=dace.Language.CPP,
    )


def instantiate_tasklet_from_info(state: dace.SDFGState,
                                  node: dace.nodes.Tasklet,
                                  info: dict,
                                  vector_width: int,
                                  templates: Dict[str, str],
                                  vector_map_param: str,
                                  vector_dtype: typeclass,
                                  mask_connector: Optional[str] = None) -> None:
    """
    Rewrite ``node.code`` into vectorized C++ from ``classify_tasklet`` info.

    Dispatches on the classified ``TaskletType`` (array-array, array-scalar,
    scalar-symbol, ...) and substitutes the matching ``templates`` entry, or
    falls back to a scalar lane loop when no template applies.

    :param state: State containing the tasklet.
    :param node: Tasklet to rewrite.
    :param info: ``classify_tasklet`` dict — ``type`` (``TaskletType``),
        ``lhs``, ``rhs1``/``rhs2`` (optional), ``constant1``/``constant2``
        (optional), ``op``.
    :param vector_width: Lane count.
    :param templates: Op-string to C++ template-string mapping.
    :param vector_map_param: Map param used for lane indexing.
    :param vector_dtype: Fallback dtype when an edge carries no data.
    :param mask_connector: ``_iter_mask`` connector name, or ``None`` for the
        unmasked path.
    """
    # Extract classification info
    ttype: tutil.TaskletType = info.get("type")
    lhs, rhs1, rhs2 = info.get("lhs"), info.get("rhs1"), info.get("rhs2")
    c1, c2, op = info.get("constant1"), info.get("constant2"), info.get("op")
    # Semantic operands for ``TERNARY_ARRAY`` (ITE), populated only for that case.
    cond_arg, then_arm, else_arm = info.get("cond"), info.get("then_arm"), info.get("else_arm")
    vw = vector_width
    is_commutative = op in {"+", "*", "==", "!="}

    # Cast boolean constants to C-compatible names
    op = PYTHON_TO_CPP_OPERATORS.get(op, op)

    ies = state.in_edges(node)
    oes = state.out_edges(node)
    in_dtypes = {state.sdfg.arrays[ie.data.data].dtype for ie in ies if ie.data.data is not None}
    out_dtypes = {state.sdfg.arrays[oe.data.data].dtype for oe in oes if oe.data.data is not None}
    all_dtypes = in_dtypes.union(out_dtypes)

    # NOTE: the C.2-b ``_iter_mask`` (``bool[W]``) connector makes
    # ``all_dtypes`` non-homogeneous so masked tasklets take the
    # fallback lane-loop below rather than the ``vector_*_av_masked``
    # templates. That fallback is now mask-gated (see ``_generate_code``
    # fallback: ``if (mask[_vi]) ...``), which is the correct behaviour;
    # we deliberately do NOT exclude the mask from this check, because
    # the ``vector_*_av_masked`` scalar-variant templates have an
    # inconsistent arg order vs their runtime macros (separate bug),
    # so routing masked-scalar ops through the template path
    # mis-compiles. The gated fallback is correct for all ops.
    fallbackcode_due_to_types = len(all_dtypes) != 1

    ctx = EmitCtx(state=state,
                  node=node,
                  templates=templates,
                  vector_dtype=vector_dtype,
                  vector_width=vw,
                  vector_map_param=vector_map_param,
                  is_commutative=is_commutative,
                  fallbackcode_due_to_types=fallbackcode_due_to_types,
                  mask_connector=mask_connector)

    # Cast python boolean to C++ compatible string
    if c1 == "False":
        c1 = "0"
    if c1 == "True":
        c1 = "1"
    if c2 == "False":
        c2 = "0"
    if c2 == "True":
        c2 = "1"

    # Dispatch based on tasklet type
    if ttype == tutil.TaskletType.ARRAY_ARRAY_ASSIGNMENT:
        # Loop-invariant scalar read into a W-wide vector buffer
        # (s176-style ``c[j]`` where ``j`` is the outer-loop param):
        # the RHS edge's subset is a single element, so the codegen
        # types the ``_in`` connector as ``T``, not ``T*``.  Route to
        # the broadcast template ``=c`` (``vector_copy_w_scalar``) by
        # placing the scalar input name in the constant slot — the
        # dispatcher then picks ``op_ + "c"`` (``"=c"``).  Without this
        # the ``=`` template emits ``vector_copy<T,W>(_out, _in)`` and
        # the compile fails with ``cannot convert 'double' to 'const
        # double*'`` for ``_in``.
        in_scalar_rhs = None
        for ie in ies:
            try:
                if int(ie.data.subset.num_elements()) == 1:
                    in_scalar_rhs = ie.dst_conn
                    break
            except Exception:
                pass
        if in_scalar_rhs is not None:
            _set_template(ctx, None, None, in_scalar_rhs, None, lhs, "=")
        else:
            _set_template(ctx, rhs1, rhs2, c1, c2, lhs, "=")
    elif ttype == tutil.TaskletType.ARRAY_SCALAR_ASSIGNMENT:
        val = None
        if c1 is not None:
            val = c1
            assert c2 is None
            assert rhs1 is None
            assert rhs2 is None
        elif c2 is not None:
            val = c2
            assert rhs1 is None
            assert rhs2 is None
        elif rhs1 is not None:
            val = rhs1
            assert rhs2 is None
        elif rhs2 is not None:
            val = rhs2
        node.code = dace.properties.CodeBlock(code="\n".join([f"{lhs}[{i}] = {val};" for i in range(vw)]) + "\n",
                                              language=dace.Language.CPP)
    elif ttype == tutil.TaskletType.ARRAY_SYMBOL_ASSIGNMENT:
        # It is either a symbol or a constant
        if _is_number(str(c1)):
            _set_template(ctx, None, None, c1, None, lhs, "=")
        else:
            node.code = dace.properties.CodeBlock(code="\n".join([f"{lhs}[{i}] = {c1}_laneid_{i};"
                                                                  for i in range(vw)]) + "\n",
                                                  language=dace.Language.CPP)
    elif ttype in {tutil.TaskletType.ARRAY_SYMBOL, tutil.TaskletType.ARRAY_ARRAY}:
        _set_template(ctx, rhs1, rhs2, c1, c2, lhs, op)
    elif ttype == tutil.TaskletType.TERNARY_ARRAY:
        # ``_o = ITE(_c, _t, _e)`` lowered to ``vector_select<{dtype}, {W}>``.
        # All three operands are arrays, the classifier carries them as
        # semantic ``cond`` / ``then_arm`` / ``else_arm`` names.
        out_edges = state.out_edges(node)
        assert len(out_edges) == 1
        out_data = state.sdfg.arrays[out_edges[0].data.data]
        dtype_ = dace.dtypes.TYPECLASS_TO_STRING[out_data.dtype]
        # In a masked remainder the ITE must be iter-mask-gated: an
        # active lane selects, an INACTIVE lane keeps ``else_arm`` (which
        # branch-normalization always sets to the ITE destination), so
        # the W-wide writeback over R<W lanes is a no-op on the trailing
        # inactive lanes instead of OOB-reading/writing past the array
        # with an unfilled ``cond`` (the TSVC s1161 masked-ITE-65 bug).
        sel_op = "ITE"
        if ctx.mask_connector is not None and "ITE_masked" in templates:
            sel_op = "ITE_masked"
        code = templates[sel_op].format(lhs=lhs,
                                        cond=cond_arg,
                                        then_arm=then_arm,
                                        else_arm=else_arm,
                                        vector_width=vw,
                                        dtype=dtype_,
                                        mask=ctx.mask_connector or "")
        node.code = dace.properties.CodeBlock(code=code, language=dace.Language.CPP)
    elif ttype in {tutil.TaskletType.UNARY_ARRAY}:
        arr_name = rhs1 if rhs1 is not None else rhs2
        occurrences = tutil.count_name_occurrences(node.code.as_string.split(" = ")[1].strip(), arr_name)
        assert occurrences == 1
        if op == "-":
            # Implement (-A) as (0 - A)
            _set_template(ctx, None, arr_name, "0.0", None, lhs, op)
        elif op == "+":
            raise Exception("Unary + operator is not supported")
        else:
            _set_template(ctx, rhs1, rhs2, c1, c2, lhs, op)
    elif ttype in {
            tutil.TaskletType.SCALAR_ARRAY,
    }:
        # The tasklet-info treads scalars as arrays and only symbols as constants
        # For the vector-code scalar is the same as a constant
        _set_template(ctx, None, rhs2, rhs1, None, lhs, op)
    elif ttype in {
            tutil.TaskletType.ARRAY_SCALAR,
    }:
        # The tasklet-info treads scalars as arrays and only symbols as constants
        # For the vector-code scalar is the same as a constant
        _set_template(ctx, rhs1, None, None, rhs2, lhs, op)
    elif ttype == tutil.TaskletType.SCALAR_SYMBOL:
        code_lines = []
        symbols = state.symbols_defined_at(node)
        l_op = rhs1 if rhs1 is not None else c1
        r_op = rhs2 if rhs2 is not None else c2
        c = c1 if c1 is not None else c2
        for i in range(vw):
            expr = f"({l_op} {op} {r_op})"
            if str(c) in symbols:
                expr = offset_symbol_in_expression(expr, vector_map_param, i, arrays=set(state.sdfg.arrays.keys()))
            else:
                if l_op == c:
                    expr = f"({l_op} {op} {r_op})"
                elif r_op == c:
                    expr = f"({l_op} {op} {r_op})"
                else:
                    expr = f"({l_op} {op} {r_op}_laneid_{i})"
            code_lines.append(f"{lhs}[{i}] = {expr}")
        node.code = dace.properties.CodeBlock(code="\n".join(code_lines) + "\n", language=dace.Language.Python)
    elif ttype == tutil.TaskletType.SCALAR_SCALAR:
        out_edges = list(state.out_edges_by_connector(node, lhs))
        assert len(out_edges) == 1
        lhs_data = state.sdfg.arrays[out_edges[0].data.data]
        l_op = rhs1 if rhs1 is not None else c1
        r_op = rhs2 if rhs2 is not None else c2
        expr = f"({l_op} {op} {r_op})"
        if isinstance(lhs_data, dace.data.Array):
            node.code = dace.properties.CodeBlock(code="\n".join([f"{lhs}[{i}] = {expr}" for i in range(vw)]) + "\n",
                                                  language=dace.Language.Python)
        else:
            node.code = dace.properties.CodeBlock(code=f"{lhs} = {expr}", language=dace.Language.Python)
    elif ttype == tutil.TaskletType.SYMBOL_SYMBOL:
        out_edges = list(state.out_edges_by_connector(node, lhs))
        assert len(out_edges) == 1
        lhs_data = state.sdfg.arrays[out_edges[0].data.data]
        l_op = rhs1 if rhs1 is not None else c1
        r_op = rhs2 if rhs2 is not None else c2
        c = c1 if c1 is not None else c2
        expr = f"({l_op} {op} {r_op})"
        if isinstance(lhs_data, dace.data.Array):
            node.code = dace.properties.CodeBlock(code="\n".join([
                f"{lhs}[{i}] = {use_laneid_symbol_in_expression(expr, c, i, vector_map_param=vector_map_param, arrays=set(state.sdfg.arrays.keys()))}"
                for i in range(vw)
            ]) + "\n",
                                                  language=dace.Language.Python)
        else:
            node.code = dace.properties.CodeBlock(code=f"{lhs} = {expr}\n", language=dace.Language.Python)
    elif ttype == tutil.TaskletType.UNARY_SCALAR or ttype == tutil.TaskletType.UNARY_SYMBOL:
        out_edges = list(state.out_edges_by_connector(node, lhs))
        assert len(out_edges) == 1
        lhs_data = state.sdfg.arrays[out_edges[0].data.data]
        l_op = rhs1 if rhs1 is not None else c1
        if op == "!=":
            raise Exception(lhs, rhs1, rhs2, c1, c2)
        expr = f"{op}{l_op}"
        if isinstance(lhs_data, dace.data.Array):
            node.code = dace.properties.CodeBlock(code="\n".join([f"{lhs}[{i}] = {expr}" for i in range(vw)]) + "\n",
                                                  language=dace.Language.Python)
        else:
            node.code = dace.properties.CodeBlock(code=f"{lhs} = {expr}\n", language=dace.Language.Python)
    else:
        raise NotImplementedError(f"Unhandled TaskletType: {ttype}, from: {node.code.as_string} ({node})")


def duplicate_access(state: dace.SDFGState, node: dace.nodes.AccessNode, vector_width: int,
                     vector_map_param: str) -> Tuple[Set[dace.nodes.Node], Set[Edge[Memlet]]]:
    """Duplicate an access node into a packed vector buffer of width ``vector_width``.

    Writes to packed storage using per-lane (laneid-offset) symbols and
    updates the feeding tasklet / memlets accordingly.

    :param state: The SDFG state containing the node.
    :param node: The AccessNode to duplicate.
    :param vector_width: Number of elements to pack.
    :param vector_map_param: Map param used for per-lane symbol offsetting.
    :returns: ``(touched_nodes, touched_edges)`` created during duplication.
    """
    # ``repl_subset_to_use_laneid_offset`` lives in ``utils.subsets`` (S6d-b).
    # Imported lazily to keep this module's top-level import surface narrow.
    from dace.transformation.passes.vectorization.utils.subsets import repl_subset_to_use_laneid_offset

    touched_nodes = set()
    touched_edges = set()

    ies = state.in_edges(node)
    assert len(ies) == 1
    ie = ies[0]
    src = ie.src
    assert isinstance(src, dace.nodes.Tasklet), f"Writes to sink nodes need to go through assignment tasklets, do it"
    inc = next(iter(src.in_connectors))
    outc = next(iter(src.out_connectors))
    if src.code.as_string != f"{outc} = {inc}":
        # If prev tasklet is not assignment then add an intermediate scalar
        scl_name, scl = state.sdfg.add_scalar("tmp",
                                              dtype=state.sdfg.arrays[node.data].dtype,
                                              storage=dace.dtypes.StorageType.Register,
                                              transient=True,
                                              find_new_name=True)
        scl_an = state.add_access(scl_name)
        scl_an.setzero = True
        e = state.add_edge(src, ie.src_conn, scl_an, None, dace.memlet.Memlet(scl_name))
        state.remove_edge(ie)
        assign_tasklet = state.add_tasklet("assign_t", {"_in"}, {"_out"}, "_out = _in")
        e2 = state.add_edge(scl_an, None, assign_tasklet, "_in", dace.memlet.Memlet(scl_name))
        e3 = state.add_edge(assign_tasklet, "_out", ie.dst, ie.dst_conn,
                            dace.memlet.Memlet(data=ie.data.data, subset=copy.deepcopy(ie.data.subset)))
        # Update ndoes/edges
        src = assign_tasklet
        ie = e3
        inc = next(iter(src.in_connectors))
        outc = next(iter(src.out_connectors))

    assert src.code.as_string == f"{outc} = {inc}", f"{src.code.as_string} != {outc} = {inc}"

    src.code = CodeBlock(code="\n".join([f"{outc}[{_i}] = {inc}[{_i}]" for _i in range(vector_width)]))
    touched_nodes.add(src)
    packed_dataname = PackedNameScheme.make(node.data)
    packed_access = state.add_access(packed_dataname)
    packed_access.setzero = True
    touched_nodes.add(packed_access)
    state.remove_edge(ie)
    touched_edges.add(ie)
    if packed_dataname not in state.sdfg.arrays:
        dst_arr = state.sdfg.arrays[node.data]
        state.sdfg.add_array(name=packed_dataname,
                             shape=(vector_width, ),
                             storage=dst_arr.storage,
                             dtype=dst_arr.dtype,
                             location=dst_arr.location,
                             transient=True,
                             lifetime=dst_arr.lifetime,
                             debuginfo=dst_arr.debuginfo,
                             allow_conflicts=dst_arr.allow_conflicts,
                             find_new_name=False,
                             alignment=dst_arr.alignment,
                             may_alias=False)
    e = state.add_edge(ie.src, ie.src_conn, packed_access, None,
                       dace.memlet.Memlet(f"{packed_dataname}[0:{vector_width}]"))
    touched_edges.add(e)

    for i in range(vector_width):
        t = state.add_tasklet(name=f"a_{i}", inputs={"_in"}, outputs={"_out"}, code="_out = _in")
        touched_nodes.add(t)
        t.add_in_connector("_in")
        t.add_out_connector("_out")
        e1 = state.add_edge(packed_access, None, t, "_in",
                            dace.memlet.Memlet(data=packed_dataname, subset=dace.subsets.Range([(str(i), str(i), 1)])))
        touched_edges.add(e1)

        new_subset = repl_subset_to_use_laneid_offset(state.sdfg, ie.data.subset, str(i), vector_map_param)

        e2 = state.add_edge(t, "_out", ie.dst, None, dace.memlet.Memlet(data=node.data, subset=new_subset))
        touched_edges.add(e2)

    return touched_nodes, touched_edges


def _insert_vector_copy_around_edge(state: dace.SDFGState, edge: Edge[Memlet],
                                    vector_storage_type: dace.dtypes.StorageType, vector_width: int, *,
                                    direction: str) -> Tuple[Edge[Memlet], Edge[Memlet], Edge[Memlet]]:
    """Splice a ``vector_copy`` tasklet + transient vector access node onto ``edge``.

    ``"from_src"``: ``src -> tasklet -> vec -> dst``.
    ``"to_dst"``: ``src -> vec -> tasklet -> dst``.

    :param state: The SDFG state containing the edge.
    :param edge: The edge to splice around.
    :param vector_storage_type: Storage type for the vector transient.
    :param vector_width: Lane count.
    :param direction: ``"from_src"`` or ``"to_dst"``.
    :returns: The three new edges in source-to-destination order.
    """
    assert direction in ("from_src", "to_dst"), direction
    src, src_conn = edge.src, edge.src_conn
    dst, dst_conn = edge.dst, edge.dst_conn

    vector_dataname = VecNameScheme.make(edge.data.data)
    if vector_dataname not in state.sdfg.arrays:
        orig_arr = state.sdfg.arrays[edge.data.data]
        _, vector_data = state.sdfg.add_array(name=vector_dataname,
                                              shape=(vector_width, ),
                                              dtype=orig_arr.dtype,
                                              location=orig_arr.location,
                                              transient=True,
                                              find_new_name=False,
                                              storage=vector_storage_type)
    else:
        vector_data = state.sdfg.arrays[vector_dataname]

    tasklet_name = "_assign_vector_from_src" if direction == "from_src" else "_assign_vector_to_dst"
    t = state.add_tasklet(
        name=tasklet_name,
        inputs={"_in"},
        outputs={"_out"},
        code=f"vector_copy<{dace.dtypes.TYPECLASS_TO_STRING[vector_data.dtype]}, {vector_width}>(_out, _in);",
        language=dace.dtypes.Language.CPP)

    an = state.add_access(vector_dataname)
    an.setzero = True

    def _new_vec_memlet():
        return dace.memlet.Memlet.from_array(vector_dataname, vector_data)

    if direction == "from_src":
        e1 = state.add_edge(src, src_conn, t, "_in", copy.deepcopy(edge.data))
        e2 = state.add_edge(t, "_out", an, None, _new_vec_memlet())
        e3 = state.add_edge(an, None, dst, dst_conn, _new_vec_memlet())
    else:
        e1 = state.add_edge(src, src_conn, an, None, _new_vec_memlet())
        e2 = state.add_edge(an, None, t, "_in", _new_vec_memlet())
        e3 = state.add_edge(t, "_out", dst, dst_conn, copy.deepcopy(edge.data))
    state.remove_edge(edge)
    return (e1, e2, e3)


def insert_assignment_tasklet_from_src(state: dace.SDFGState, edge: Edge[Memlet],
                                       vector_storage_type: dace.dtypes.StorageType,
                                       vector_width: int) -> Tuple[Edge[Memlet], Edge[Memlet], Edge[Memlet]]:
    """Splice ``vector_copy`` after the source: ``src -> tasklet -> vec -> dst``.

    :param state: The SDFG state containing the edge.
    :param edge: The edge to splice around.
    :param vector_storage_type: Storage type for the vector transient.
    :param vector_width: Lane count.
    :returns: The three new edges in source-to-destination order.
    """
    return _insert_vector_copy_around_edge(state, edge, vector_storage_type, vector_width, direction="from_src")


def insert_assignment_tasklet_to_dst(state: dace.SDFGState, edge: Edge[Memlet],
                                     vector_storage_type: dace.dtypes.StorageType,
                                     vector_width: int) -> Tuple[Edge[Memlet], Edge[Memlet], Edge[Memlet]]:
    """Splice ``vector_copy`` before the destination: ``src -> vec -> tasklet -> dst``.

    :param state: The SDFG state containing the edge.
    :param edge: The edge to splice around.
    :param vector_storage_type: Storage type for the vector transient.
    :param vector_width: Lane count.
    :returns: The three new edges in source-to-destination order.
    """
    return _insert_vector_copy_around_edge(state, edge, vector_storage_type, vector_width, direction="to_dst")
