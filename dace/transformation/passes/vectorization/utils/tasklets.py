# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tasklet helpers used by the vectorization pipeline.

This module hosts the tasklet-creation / replication / classification
utilities, including the 327-LoC ``instantiate_tasklet_from_info``
emission core that picks per-template C++ code based on the
``TaskletType`` classification from ``dace.sdfg.tasklet_utils``.

Per the locked policy (defensive checks stay, mechanical-only), every
helper is moved verbatim. The planned redesign of the emission core
(tile lib nodes, separate per-ISA expansions) is deferred to a
post-S7 slice; for now the helpers retain the existing template-based
emission.
"""
import copy
from typing import Dict, Set, Tuple, Union

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


def match_connector_to_data(state: dace.SDFGState, tasklet: dace.nodes.Tasklet) -> dict:
    """
    Map tasklet input connectors to their corresponding array descriptors.

    Args:
        state (dace.SDFGState): The state containing the tasklet.
        tasklet (dace.nodes.Tasklet): The tasklet whose connectors are inspected.

    Returns:
        dict[str, dace.data.Data]: A mapping from tasklet input connector names
        to the corresponding array descriptors in the SDFG.
    """
    tdict = dict()
    for ie in state.in_edges(tasklet):
        if ie.data is not None:
            tdict[ie.dst_conn] = state.sdfg.arrays[ie.data.data]
    return tdict


def is_assignment_tasklet(node: dace.nodes.Tasklet) -> bool:
    """
    Checks if a tasklet is a simple assignment (one input to one output).
    Checks `a = b` or `a = b;`
    Args:
        node: The Tasklet to check.

    Returns:
        True if it is a single assignment tasklet, False otherwise.
    """
    if (len(node.in_connectors) == 1 and len(node.out_connectors) == 1):
        in_conn = next(iter(node.in_connectors.keys()))
        out_conn = next(iter(node.out_connectors.keys()))
        return (node.code.as_string == f"{out_conn} = {in_conn}" or node.code.as_string == f"{out_conn} = {in_conn};")
    return False


def is_vector_assign_tasklet(t: dace.nodes.Tasklet) -> bool:
    """
    Check if a tasklet performs a vector copy operation.

    Args:
        t: The tasklet to check

    Returns:
        True if the tasklet's code contains "vector_copy(", False otherwise
    """
    return "vector_copy(" in t.code.as_string


def instantiate_tasklet_from_info(state: dace.SDFGState, node: dace.nodes.Tasklet, info: dict, vector_width: int,
                                  templates: Dict[str, str], vector_map_param: str, vector_dtype: typeclass) -> None:
    """
    Instantiates a tasklet's code block in vectorized form based on classification info.

    This function takes a tasklet and its classification `info` (from `classify_tasklet`) and
    updates `node.code` to a vectorized CodeBlock using the provided templates. Handles
    different tasklet types (array-array, array-scalar, scalar-symbol, etc.) and supports
    vectorization over the specified width.

    Args:
        state: The SDFGState containing the tasklet.
        node: The tasklet node to instantiate.
        info: Classification dictionary containing:
            - "type": TaskletType enum describing operand types.
            - "lhs": Left-hand side variable.
            - "rhs1": First right-hand side variable.
            - "rhs2": Second right-hand side variable (optional).
            - "constant1": First constant operand (optional).
            - "constant2": Second constant operand (optional).
            - "op": Operation string (e.g., "+", "*", "=").
        vector_width: Number of lanes for vectorization.
        templates: Mapping from operation strings to template strings for code generation.
        vector_map_param: Name of the map parameter used for lane indexing in vectorization.
    """
    # Extract classification info
    ttype: tutil.TaskletType = info.get("type")
    lhs, rhs1, rhs2 = info.get("lhs"), info.get("rhs1"), info.get("rhs2")
    c1, c2, op = info.get("constant1"), info.get("constant2"), info.get("op")
    # Semantic operands for ``TERNARY_ARRAY`` (merge), populated only for that case.
    cond_arg, then_arm, else_arm = info.get("cond"), info.get("then_arm"), info.get("else_arm")
    vw = vector_width
    is_commutative = op in {"+", "*", "==", "!="}

    # Cast boolean constants to C-compatible names
    PYTHON_TO_CPP_OPERATORS = {"and": "&&", "or": "||", "not": "!"}
    op = PYTHON_TO_CPP_OPERATORS.get(op, op)

    ies = state.in_edges(node)
    oes = state.out_edges(node)
    in_dtypes = {state.sdfg.arrays[ie.data.data].dtype for ie in ies if ie.data.data is not None}
    out_dtypes = {state.sdfg.arrays[oe.data.data].dtype for oe in oes if oe.data.data is not None}
    all_dtypes = in_dtypes.union(out_dtypes)

    fallbackcode_due_to_types = len(all_dtypes) != 1

    def _str_to_float_or_str(s: Union[int, float, str, None]):
        """Convert string constants to float if possible."""
        if s is None:
            return s
        try:
            return float(s)
        except ValueError:
            return s

    def _is_number(s: str):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _generate_code(rhs1_, rhs2_, const1_, const2_, lhs_, op_):
        """
        Generate the C++ vectorized code string using templates or fallbacks.

        Handles:
        - Array-array, array-scalar, scalar-array
        - Commutative and non-commutative ops
        - Single constant + array/scalar (or array/scalar + constant)
        - Fallback loops if operator not supported (hope compiler will do it)
        """

        # Get out edge and its dtype
        out_edges = state.out_edges(node)
        assert len(out_edges) == 1
        out_edge = out_edges[0]

        if out_edge.data.data is None:
            dtype_ = dace.dtypes.TYPECLASS_TO_STRING[vector_dtype]
        else:
            data_dtype = state.sdfg.arrays[out_edge.data.data].dtype
            dtype_ = dace.dtypes.TYPECLASS_TO_STRING[data_dtype]

        rhs_left = rhs1_ if rhs1_ is not None else const1_
        rhs_right = rhs2_ if rhs2_ is not None else const2_

        # Multiple dtypes involved - fallback code should be used
        if not fallbackcode_due_to_types:
            # Use template if available
            if op_ in templates:
                # One array + optional constant
                if rhs1_ is None or rhs2_ is None:
                    rhs = rhs1_ if rhs1_ is not None else rhs2_
                    constant = const1_ if const1_ is not None else const2_
                    if constant is None:
                        # Single array or repeated array case
                        if is_commutative:
                            return templates[op_].format(rhs1=rhs,
                                                         rhs2=rhs,
                                                         lhs=lhs_,
                                                         op=op_,
                                                         vector_width=vw,
                                                         dtype=dtype_)
                        return templates[op_].format(rhs1=rhs,
                                                     rhs2=rhs,
                                                     lhs=lhs_,
                                                     op=op_,
                                                     vector_width=vw,
                                                     dtype=dtype_)
                    else:
                        # Single array + constant
                        cop_ = None
                        if is_commutative or op_ == "=":
                            cop_ = op_ + "c"
                        elif constant == const1_:
                            cop_ = "c" + op_
                        else:
                            assert constant == const2_
                            cop_ = op_ + "c"
                        # Maybe this constant version is not implemented in templates
                        if cop_ in templates:
                            return templates[cop_].format(rhs1=rhs,
                                                          constant=_str_to_float_or_str(constant),
                                                          lhs=lhs_,
                                                          op=op_,
                                                          vector_width=vw,
                                                          dtype=dtype_)

                else:
                    # Two arrays
                    return templates[op_].format(rhs1=rhs1_,
                                                 rhs2=rhs2_,
                                                 lhs=lhs_,
                                                 op=op_,
                                                 vector_width=vw,
                                                 dtype=dtype_)

        # Fallback: unsupported operator
        comparison_suffix = "? 1.0 : 0.0" if op_ in {">", ">=", "<", "<=", "==", "!="} else ""
        code_lines = [f"_dace_vectorize({vector_width})"]
        code_lines.append(f"for (int _vi = 0; _vi < {vw}; _vi += 1) {{")

        # Determine operand order
        lhs_expr = lhs_ + "[_vi]"
        rhs_left = rhs1_ if rhs1_ is not None else const1_
        rhs_right = rhs2_ if rhs2_ is not None else const2_
        OPERATORS = {"+", "-", "/", "*", "%", "&&", "||", "==", "!=", "<", "<=", ">", ">="}
        UNARY_OPERATORS = {"+", "!", "-"}

        if rhs_left is None or rhs_right is None:
            if op not in UNARY_OPERATORS and op in OPERATORS:
                raise Exception(
                    f"Invalid operand configuration for fallback vectorization. {rhs_left}, {rhs_right}, {lhs_expr}, {op}"
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
                if rhs_left == const:
                    code_lines.append(f"{lhs_expr} = {op_}({rhs}){comparison_suffix};")
                else:
                    code_lines.append(f"{lhs_expr} = {op_}({rhs}[_vi]){comparison_suffix};")
        else:
            if op_ in OPERATORS:
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

        code_lines.append("}")
        return "\n".join(code_lines)

    def _set_template(rhs1_, rhs2_, const1_, const2_, lhs_, op_, ttype):
        """Helper to set tasklet code from template/fallback."""
        node.code = dace.properties.CodeBlock(code=_generate_code(rhs1_, rhs2_, _str_to_float_or_str(const1_),
                                                                  _str_to_float_or_str(const2_), lhs_, op_),
                                              language=dace.Language.CPP)

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
        _set_template(rhs1, rhs2, c1, c2, lhs, "=", ttype)
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
            _set_template(None, None, c1, None, lhs, "=", ttype)
        else:
            node.code = dace.properties.CodeBlock(code="\n".join([f"{lhs}[{i}] = {c1}_laneid_{i};"
                                                                  for i in range(vw)]) + "\n",
                                                  language=dace.Language.CPP)
    elif ttype in {tutil.TaskletType.ARRAY_SYMBOL, tutil.TaskletType.ARRAY_ARRAY}:
        _set_template(rhs1, rhs2, c1, c2, lhs, op, ttype)
    elif ttype == tutil.TaskletType.TERNARY_ARRAY:
        # ``_o = merge(_c, _t, _e)`` lowered to ``vector_select<{dtype}, {W}>``.
        # All three operands are arrays, the classifier carries them as
        # semantic ``cond`` / ``then_arm`` / ``else_arm`` names.
        out_edges = state.out_edges(node)
        assert len(out_edges) == 1
        out_data = state.sdfg.arrays[out_edges[0].data.data]
        dtype_ = dace.dtypes.TYPECLASS_TO_STRING[out_data.dtype]
        code = templates[op].format(lhs=lhs,
                                    cond=cond_arg,
                                    then_arm=then_arm,
                                    else_arm=else_arm,
                                    vector_width=vw,
                                    dtype=dtype_)
        node.code = dace.properties.CodeBlock(code=code, language=dace.Language.CPP)
    elif ttype in {tutil.TaskletType.UNARY_ARRAY}:
        arr_name = rhs1 if rhs1 is not None else rhs2
        occurences = tutil.count_name_occurrences(node.code.as_string.split(" = ")[1].strip(), arr_name)
        assert occurences == 1
        if op == "-":
            # Implement (-A) as (0 - A)
            _set_template(None, arr_name, "0.0", None, lhs, op, tutil.TaskletType.ARRAY_SYMBOL)
        elif op == "+":
            raise Exception("Unary + operator is not supported")
        else:
            _set_template(rhs1, rhs2, c1, c2, lhs, op, ttype)
    elif ttype in {
            tutil.TaskletType.SCALAR_ARRAY,
    }:
        # The tasklet-info treads scalars as arrays and only symbols as constants
        # For the vector-code scalar is the same as a constant
        _set_template(None, rhs2, rhs1, None, lhs, op, ttype)
    elif ttype in {
            tutil.TaskletType.ARRAY_SCALAR,
    }:
        # The tasklet-info treads scalars as arrays and only symbols as constants
        # For the vector-code scalar is the same as a constant
        _set_template(rhs1, None, None, rhs2, lhs, op, ttype)
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
        state.sdfg.save("failing.sdfg")
        raise NotImplementedError(f"Unhandled TaskletType: {ttype}, from: {node.code.as_string} ({node})")


def duplicate_access(state: dace.SDFGState, node: dace.nodes.AccessNode, vector_width: int,
                     vector_map_param: str) -> Tuple[Set[dace.nodes.Node], Set[Edge[Memlet]]]:
    """
    Duplicates an access node into a packed vector of a given width, updating all relevant tasklets and memlets.
    It writes to a packed storage by using the duplicated symbols.

    Args:
        state: The SDFG state containing the node.
        node: The AccessNode to duplicate.
        vector_width: Number of elements to pack.

    Returns:
        A tuple of sets: touched nodes and touched edges created during duplication.
    """
    # ``repl_subset_to_use_laneid_offset`` still lives in ``vectorization_utils.py``
    # (migrates in S6d with the subset/repl helpers). Imported lazily.
    from dace.transformation.passes.vectorization.vectorization_utils import repl_subset_to_use_laneid_offset

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

    assert src.code.as_string == f"{outc} = {inc}", f"{src.code.as_string} != {inc} = {outc}"

    src.code = CodeBlock(code="\n".join([f"{outc}[{_i}] = {inc}[{_i}]" for _i in range(vector_width)]))
    touched_nodes.add(src)
    packed_access = state.add_access(f"{node.data}_packed")
    packed_access.setzero = True
    touched_nodes.add(packed_access)
    state.remove_edge(ie)
    if isinstance(ie, dace.nodes.Node):
        assert False
    touched_edges.add(ie)
    if f"{node.data}_packed" not in state.sdfg.arrays:
        dst_arr = state.sdfg.arrays[node.data]
        state.sdfg.add_array(name=f"{node.data}_packed",
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
                       dace.memlet.Memlet(f"{node.data}_packed[0:{vector_width}]"))
    if isinstance(e, dace.nodes.Node):
        assert False
    touched_edges.add(e)

    for i in range(vector_width):
        t = state.add_tasklet(name=f"a_{i}", inputs={"_in"}, outputs={"_out"}, code="_out = _in")
        touched_nodes.add(t)
        t.add_in_connector("_in")
        t.add_out_connector("_out")
        e1 = state.add_edge(
            packed_access, None, t, "_in",
            dace.memlet.Memlet(data=node.data + "_packed", subset=dace.subsets.Range([(str(i), str(i), 1)])))
        if isinstance(e1, dace.nodes.Node):
            assert False
        touched_edges.add(e1)

        new_subset = repl_subset_to_use_laneid_offset(state.sdfg, ie.data.subset, str(i), vector_map_param)

        e2 = state.add_edge(t, "_out", ie.dst, None, dace.memlet.Memlet(data=node.data, subset=new_subset))
        if isinstance(e2, dace.nodes.Node):
            assert False
        touched_edges.add(e2)

    return touched_nodes, touched_edges


def insert_assignment_tasklet_from_src(state: dace.SDFGState, edge: Edge[Memlet],
                                       vector_storage_type: dace.dtypes.StorageType,
                                       vector_width: int) -> Tuple[Edge[Memlet], Edge[Memlet], Edge[Memlet]]:
    """
    Insert a vector assignment tasklet after the source node of an edge.

    This function transforms:
        src --[memlet]--> dst
    Into:
        src --[memlet]--> copy_tasklet --> access_node[vector] --[memlet2]--> dst

    The tasklet performs a vector_copy operation, and a new transient vector array
    is created with the specified storage type and length.

    Args:
        state: The SDFG state containing the edge
        edge: The edge to transform
        vector_storage_type: Storage type for the new vector array (e.g., Register, FPGA_Local)
        vector_width: Length of the vector array

    Returns:
        A tuple of three new edges: (src->tasklet, tasklet->access, access->dst)

    Side effects:
        - Removes the original edge
        - Creates a new transient vector array if it doesn't exist
        - Adds a tasklet, access node, and three new edges
    """
    src = edge.src
    src_conn = edge.src_conn
    dst = edge.dst
    dst_conn = edge.dst_conn

    # Create or reuse vector array
    vector_dataname = edge.data.data + "_vec"
    if vector_dataname not in state.sdfg.arrays:
        orig_arr = state.sdfg.arrays[edge.data.data]
        arr_name, arr = state.sdfg.add_array(name=vector_dataname,
                                             shape=(vector_width, ),
                                             dtype=orig_arr.dtype,
                                             location=orig_arr.location,
                                             transient=True,
                                             find_new_name=False,
                                             storage=vector_storage_type)
        vector_data = arr
    else:
        vector_data = state.sdfg.arrays[vector_dataname]

    # Create assignment tasklet
    t = state.add_tasklet(
        name="_AssignT3",
        inputs={"_in"},
        outputs={"_out"},
        code=f"vector_copy<{dace.dtypes.TYPECLASS_TO_STRING[vector_data.dtype]}, {vector_width}>(_out, _in);",
        language=dace.dtypes.Language.CPP)

    # Create access node and edges
    an = state.add_access(vector_dataname)
    an.setzero = True
    e1 = state.add_edge(src, src_conn, t, "_in", copy.deepcopy(edge.data))
    e2 = state.add_edge(t, "_out", an, None, dace.memlet.Memlet.from_array(vector_dataname, vector_data))
    e3 = state.add_edge(an, None, dst, dst_conn, dace.memlet.Memlet.from_array(vector_dataname, vector_data))
    state.remove_edge(edge)

    return (e1, e2, e3)


def insert_assignment_tasklet_to_dst(state: dace.SDFGState, edge: Edge[Memlet],
                                     vector_storage_type: dace.dtypes.StorageType,
                                     vector_width: int) -> Tuple[Edge[Memlet], Edge[Memlet], Edge[Memlet]]:
    """
    Insert a vector assignment tasklet before the destination node of an edge.


    This function transforms:
        src --[memlet]--> dst
    Into:
        src --[memlet2]--> access_node[vector] --[memlet2]--> copy_tasklet --[memlet]--> dst


    The tasklet performs a vector_copy operation, and a new transient vector array
    is created with the specified storage type and length.

    Args:
        state: The SDFG state containing the edge
        edge: The edge to transform
        vector_storage_type: Storage type for the new vector array
        vector_width: Length of the vector array

    Returns:
        A tuple of three new edges: (src->access, access->tasklet, tasklet->dst)

    Side effects:
        - Removes the original edge
        - Creates a new transient vector array if it doesn't exist
        - Adds a tasklet, access node, and three new edges
    """
    src = edge.src
    src_conn = edge.src_conn
    dst = edge.dst
    dst_conn = edge.dst_conn

    # Create or reuse vector array
    vector_dataname = edge.data.data + "_vec"
    if vector_dataname not in state.sdfg.arrays:
        orig_arr = state.sdfg.arrays[edge.data.data]
        _, arr = state.sdfg.add_array(name=vector_dataname,
                                      shape=(vector_width, ),
                                      dtype=orig_arr.dtype,
                                      location=orig_arr.location,
                                      transient=True,
                                      find_new_name=False,
                                      storage=vector_storage_type)
        vector_data = arr
    else:
        vector_data = state.sdfg.arrays[vector_dataname]

    # Create assignment tasklet
    t = state.add_tasklet(
        name="_AssignT4",
        inputs={"_in"},
        outputs={"_out"},
        code=f"vector_copy<{dace.dtypes.TYPECLASS_TO_STRING[vector_data.dtype]}, {vector_width}>(_out, _in);",
        language=dace.dtypes.Language.CPP)

    # Create access node and edges
    an = state.add_access(vector_dataname)
    an.setzero = True
    e1 = state.add_edge(src, src_conn, an, None, dace.memlet.Memlet.from_array(vector_dataname, vector_data))
    e2 = state.add_edge(an, None, t, "_in", dace.memlet.Memlet.from_array(vector_dataname, vector_data))
    e3 = state.add_edge(t, "_out", dst, dst_conn, copy.deepcopy(edge.data))
    state.remove_edge(edge)

    return (e1, e2, e3)
