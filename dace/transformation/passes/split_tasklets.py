# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import functools
import re

import networkx as nx

import dace
from typing import Any, Dict, List, Optional, Set, Tuple

from dace import SDFG
from dace.transformation import pass_pipeline as ppl, transformation

import ast
from dace.sdfg.nodes import CodeBlock
from dace.sdfg.type_inference import infer_types


class ASTSplitter:
    """
    Lowers a Python expression AST into a list of single-operation SSA statements.

    Each visited node emits one ``__tN = <op>`` statement and returns the name that
    holds its result, so a nested expression becomes a flat sequence of primitive ops.
    """

    def __init__(self):
        self.n = 0
        self.stmts = []

    def temp(self) -> str:
        """
        Allocate a fresh SSA temporary name.

        :returns: A unique ``__tN`` identifier.
        """
        t = f"__t{self.n}"
        self.n += 1
        return t

    def visit(self, node: ast.AST) -> str:
        """
        Emit SSA statements for one AST node and return the name holding its value.

        :param node: The expression AST node to lower.
        :returns: The variable name (or literal) that holds the node's result.
        """
        if isinstance(node, ast.BinOp):
            l, r = self.visit(node.left), self.visit(node.right)
            t = self.temp()
            # Floor division has no infix C++ form; emit it as the
            # ``int_floor(num, den)`` call the backend recognises (matches
            # ``dace::math::int_floor`` and the ``vector_int_floor`` intrinsic).
            if isinstance(node.op, ast.FloorDiv):
                self.stmts.append(f"{t} = int_floor({l}, {r})")
                return t
            ops = {
                ast.Add: '+',
                ast.Sub: '-',
                ast.Mult: '*',
                ast.Div: '/',
                ast.Pow: '**',
                ast.Mod: '%',
                ast.MatMult: '@',
                ast.BitAnd: '&',
                ast.BitOr: '|',
                ast.BitXor: '^',
                ast.LShift: '<<',
                ast.RShift: '>>',
                ast.Or: 'or',
                ast.And: 'and',
                ast.Eq: '==',
            }
            self.stmts.append(f"{t} = {l} {ops[type(node.op)]} {r}")
            return t

        elif isinstance(node, ast.UnaryOp):
            op = self.visit(node.operand)
            t = self.temp()
            ops = {ast.USub: '-', ast.UAdd: '+', ast.Not: 'not ', ast.Invert: '~'}
            self.stmts.append(f"{t} = {ops[type(node.op)]}{op}")
            return t

        elif isinstance(node, ast.Call):
            func = self.visit(node.func)
            args = [self.visit(arg) for arg in node.args]
            t = self.temp()
            self.stmts.append(f"{t} = {func}({', '.join(args)})")
            return t

        elif isinstance(node, ast.Name):
            return node.id

        elif isinstance(node, ast.Constant):
            return str(node.value)

        elif isinstance(node, ast.Attribute):
            return f"{self.visit(node.value)}.{node.attr}"

        elif isinstance(node, ast.Compare):
            # Handle comparison operators
            left = self.visit(node.left)
            comparisons = []
            current = left

            for op, comparator in zip(node.ops, node.comparators):
                comp = self.visit(comparator)
                t = self.temp()
                ops = {
                    ast.Eq: '==',
                    ast.NotEq: '!=',
                    ast.Lt: '<',
                    ast.LtE: '<=',
                    ast.Gt: '>',
                    ast.GtE: '>=',
                }
                self.stmts.append(f"{t} = {current} {ops[type(op)]} {comp}")
                comparisons.append(t)
                current = comp

            # If multiple comparisons, combine with 'and'
            if len(comparisons) > 1:
                t = self.temp()
                self.stmts.append(f"{t} = {' and '.join(comparisons)}")
                return t
            return comparisons[0] if comparisons else left

        elif isinstance(node, ast.BoolOp):
            # Handle boolean operators (and, or)
            values = [self.visit(v) for v in node.values]
            t = self.temp()
            op = ' and ' if isinstance(node.op, ast.And) else ' or '
            if op == "or":
                assert isinstance(node.op, ast.Or)
            self.stmts.append(f"{t} = {op.join(values)}")
            return t

        elif isinstance(node, ast.IfExp):
            # Lower ``body if test else orelse`` to a 3-input ``ITE(c,
            # t, e)`` call. ``ITE`` is the unified canonical name for
            # the ternary blend; ``dace/ITE.h`` ships the ``ITE``
            # template, and the canonicalize-stage
            # ``LowerITEToFpFactor`` rewrites it to ``c * t + (1 - c) *
            # e`` so codegen never has to lower a Python ternary to
            # ``c ? t : e``.
            cond = self.visit(node.test)
            body = self.visit(node.body)
            orelse = self.visit(node.orelse)
            t = self.temp()
            self.stmts.append(f"{t} = ITE({cond}, {body}, {orelse})")
            return t

        return ast.unparse(node)


def to_ssa(code: str) -> List[str]:
    """
    Convert a single Python assignment (or expression) into single-operation SSA lines.

    :param code: The tasklet source, e.g. ``out = a * b + c``.
    :returns: A list of SSA statements, each performing at most one primitive operation.
    """
    tree = ast.parse(code).body[0]
    ssa = ASTSplitter()
    # Start the temp counter past any ``__t<N>`` already present in the input
    # so a re-split (or an input that happens to use these names) does not
    # collide: a fresh ``__t0`` aliasing an existing operand turned
    # ``__t0 < int_floor(LEN_1D, 2)`` into ``__t0 = int_floor(LEN_1D, 2);
    # __t1 = __t0 < __t0``, silently dropping the comparison's left operand.
    existing = [int(m) for m in re.findall(r"__t(\d+)\b", code)]
    if existing:
        ssa.n = max(existing) + 1
    if isinstance(tree, ast.Assign):
        target = tree.targets[0].id if isinstance(tree.targets[0], ast.Name) else ast.unparse(tree.targets[0])
        # Check if RHS is already a simple variable or constant
        if isinstance(tree.value, (ast.Name, ast.Constant)):
            ssa.stmts.append(f"{target} = {ssa.visit(tree.value)}")
        else:
            rhs = ssa.visit(tree.value)
            # Replace the last temp variable with the target
            if ssa.stmts and ssa.stmts[-1].startswith(rhs + ' ='):
                ssa.stmts[-1] = ssa.stmts[-1].replace(rhs, target, 1)
            else:
                ssa.stmts.append(f"{target} = {rhs}")
    else:
        ssa.visit(tree.value if isinstance(tree, ast.Expr) else tree)
    return ssa.stmts


def _ast_function_names(rhs: str) -> Set[str]:
    """AST-walk ``rhs`` and collect every identifier that appears at a ``Call.func``
    position. This includes any function called regardless of whether it's a built-in
    or a user-defined name (``sqrt``, ``mymath.func``, ``foo``...). Falls back to an
    empty set if parsing fails (the caller adds the legacy allowlist on top).
    """
    try:
        tree = ast.parse(rhs, mode="eval")
    except (SyntaxError, ValueError):
        return set()
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                # Walk down to the leaf (rightmost) attribute name. Also drop the
                # base-name (the module / class on the LHS) -- ``math.sqrt`` becomes
                # ``math`` and ``sqrt`` both excluded as function-path identifiers.
                inner = f
                while isinstance(inner, ast.Attribute):
                    names.add(inner.attr)
                    inner = inner.value
                if isinstance(inner, ast.Name):
                    names.add(inner.id)
    return names


def _get_vars(ssa_line: str) -> Tuple[List[str], List[str]]:
    """
    Extract the left-hand-side and right-hand-side variable names of an SSA line.

    Function names (anything appearing at a ``Call.func`` AST position) are ignored so
    they are not mistaken for per-lane input connectors. Per user direction 2026-06-09:
    "Ensure that we split all function expressions like this regardless of what function
    it was". Previous versions used a hard-coded allowlist (``log`` / ``exp`` / etc.)
    which made ``sqrt``, ``tanh``, user-defined functions, and any other call show up as
    a stale input connector.

    :param ssa_line: A single ``lhs = rhs`` SSA statement.
    :returns: A tuple ``(lhs_vars, rhs_vars)`` of the assigned name and the read names.
    """
    lhs, rhs = ssa_line.split(" = ")
    lhs = lhs.strip()
    rhs = rhs.strip()
    # AST-detected function-position names (works for ANY function call) +
    # the legacy allowlist (covers ``or`` / ``and`` boolean keywords + ``True`` /
    # ``False`` literals + builtin user functions that might be referenced
    # bare without a Call wrapper).
    function_names = (_ast_function_names(rhs).union(dace.symbolic.builtin_userfunctions()).union({
        "log",
        "Log",
        "ln",
        "exp",
        "Exp",
        "or",
        "and",
        "Or",
        "And",
        "OR",
        "AND",
        "math",
        "Math",
        "MATH",
    }).union({"True", "False"}))

    return [lhs], list(dace.symbolic.symbols_in_code(rhs, symbols_to_ignore=function_names))


def _ssa_lhs_is_bool(ssa_line: str) -> bool:
    """True iff the SSA statement's RHS is a comparison / boolean expression, so its
    assigned variable holds a ``bool`` rather than the numeric ``input_type``.

    A split-out condition such as ``__t0 = (_a > 0.0)`` (or ``__t1 = a and b``,
    ``__t2 = not c``) must be typed ``bool``: downstream it feeds a ``TileITE`` /
    mask ``_mask`` connector whose operand contract requires a bool array. Typing
    such an intermediate numerically (the default ``input_type``) trips the
    ``mask_connectors_are_bool`` invariant in ConvertTaskletsToTileOps.
    """
    try:
        rhs = ast.parse(ssa_line.split(" = ", 1)[1].strip(), mode="eval").body
    except (SyntaxError, IndexError):
        return False
    if isinstance(rhs, (ast.Compare, ast.BoolOp)):
        return True
    if isinstance(rhs, ast.UnaryOp) and isinstance(rhs.op, ast.Not):
        return True
    return False


def _infer_ssa_intermediate_types(ssa_statements: List[str], leaf_types: Dict[str, Any], fallback) -> Dict[str, Any]:
    """Infer the dtype of every SSA intermediate, threading resolved types forward.

    The primary inference is DaCe's ``infer_types`` (operand promotion via
    ``result_type_of``, comparisons -> ``bool``, integer-power special cases). When it
    cannot type a statement -- an unknown function-call return such as ``tanh(a)``
    yields ``None`` -- the result is the promotion of that statement's OWN known operand
    types: a single float operand keeps its float, several operands promote by
    ``result_type_of`` (the same rule the primary path applies). Only a statement with
    no known operand type at all falls back to the coarse whole-tasklet ``fallback``.

    Resolved types are threaded forward, so a later statement that reads an earlier
    (fallback-typed) intermediate sees its resolved type rather than ``None``.

    :param ssa_statements: The single-op ``lhs = rhs`` SSA statements, in order.
    :param leaf_types: Known dtypes of the leaf operands (input connectors + symbols).
    :param fallback: Dtype to use for an intermediate with no known operand type.
    :returns: ``{intermediate name -> dtype}`` for every typed SSA assignment.
    """
    known: Dict[str, Any] = dict(leaf_types)
    inferred: Dict[str, Any] = {}
    for stmt in ssa_statements:
        try:
            stmt_types = infer_types([stmt], known)
        except Exception:
            stmt_types = {}
        for lhs, t in stmt_types.items():
            if t is None:
                # infer_types could not type this statement (an unknown function-call
                # return such as ``tanh(a)``). Promote the operand types it DOES know --
                # same float in => same float out; multiple operands => result_type_of --
                # and only then the coarse fallback. (A dtype-cast call ``double(N)`` is
                # typed by ``infer_types`` itself now, via ``_infer_dtype``.)
                _, rhs_vars = _get_vars(stmt)
                operand_types = [known[v] for v in rhs_vars if known.get(v) is not None]
                t = functools.reduce(dace.dtypes.result_type_of, operand_types) if operand_types else fallback
            known[lhs] = t
            inferred[lhs] = t
    return inferred


@transformation.explicit_cf_compatible
class SplitTasklets(ppl.Pass):
    """
    Splits a multi-operation tasklet into one tasklet per primitive operation.

    Each tasklet body is rewritten into single-operation SSA statements, then the tasklet
    is replaced by a chain of one-op tasklets connected through register transients. This
    lets downstream canonicalization and vectorization passes reason about single-op bodies.
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets | ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return []

    tmp_access_identifier = "_split_"

    def token_split_variable_names(self, string_to_check: str) -> Set[str]:
        """
        Split a code string into identifier tokens, dropping whitespace and brackets.

        Used as a fallback when SymPy cannot parse the tasklet (e.g. nested comparisons).

        :param string_to_check: The tasklet code to tokenize.
        :returns: The set of identifier-like tokens in the string.
        """
        # Keep the delimiters so adjacent identifiers stay separated.
        tokens = re.split(r'(\s+|[()\[\]])', string_to_check)
        return {token.strip() for token in tokens if token not in ["[", "]", "(", ")"] and token.isidentifier()}

    def _add_missing_symbols(self, sdfg: SDFG) -> Set[str]:
        """
        Register interstate-edge assignment targets that are not yet declared symbols.

        The dtype is inferred from the arrays/symbols referenced on the right-hand side,
        preferring the widest float and falling back to ``float64`` when nothing matches --
        UNLESS the whole right-hand side is an explicit dace integer typecast
        (``dace.int64(x)`` etc., parsed to the ``symbolic.int64`` sympy ``Function``), in which
        case the cast's kind wins outright: a ``dace.int64(<float expr>)`` truncates to an
        integer in C++ even though every atom in ``<float expr>`` is float64, so the atom-priority
        heuristic below (which always prefers float64 over int64) would silently drop the cast
        and hand back a float-typed symbol.

        :param sdfg: The SDFG to scan (recursively into nested SDFGs).
        :returns: The names of the symbols that were added (empty if the symbol table was already complete).
        """
        added: Set[str] = set()
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    added |= self._add_missing_symbols(node.sdfg)
        for e in sdfg.all_interstate_edges():
            for k, v in e.data.assignments.items():
                if k not in sdfg.symbols:
                    symexpr = dace.symbolic.SymExpr(v)
                    cast_name = getattr(getattr(symexpr, 'func', None), '__name__', None)
                    if cast_name in dace.dtypes.TYPECLASS_STRINGS:
                        cast_dtype = getattr(dace, cast_name)
                        if cast_dtype in dace.dtypes.INTEGER_TYPES:
                            sdfg.add_symbol(k, cast_dtype)
                            added.add(k)
                            continue
                    dtypes = set()
                    # Array accesses ``arr[i]`` are ``Subscript`` nodes; their names
                    # come from ``arrays`` (the old ``atoms(Function).name`` form no
                    # longer reports the array head after the subscript rework).
                    funcs = list(dace.symbolic.arrays(symexpr))
                    for token in funcs:
                        if str(token) in sdfg.arrays:
                            dtypes.add(sdfg.arrays[str(token)].dtype)
                        if str(token) in sdfg.symbols:
                            dtypes.add(sdfg.symbols[str(token)])
                    dtype_priority = [
                        dace.float64,
                        dace.float32,
                        dace.int64,
                        dace.uint64,
                        dace.int32,
                        dace.uint32,
                        dace.int16,
                        dace.uint16,
                        dace.int8,
                        dace.uint8,
                        dace.bool_,
                    ]
                    ktype = None
                    if len(dtypes) > 0:
                        for dt in dtype_priority:
                            if dt in dtypes:
                                ktype = dt
                                break
                    if ktype is None:
                        for token in symexpr.free_symbols:
                            if str(token) in sdfg.arrays:
                                dtypes.add(sdfg.arrays[str(token)].dtype)
                            if str(token) in sdfg.symbols:
                                dtypes.add(sdfg.symbols[str(token)])
                    if len(dtypes) > 0:
                        for dt in dtype_priority:
                            if dt in dtypes:
                                ktype = dt
                                break
                    if ktype is None:
                        ktype = dace.float64
                    sdfg.add_symbol(k, ktype)
                    added.add(k)
        return added

    def _symbol_lifted_data(self, sdfg: SDFG) -> Set[str]:
        """
        Collect the data names read by an interstate-edge assignment right-hand side.

        A scalar promoted to a symbol on an interstate edge (e.g. ``__sym_off = off``,
        an index symbol the frontend lifts out of ``a[off]``) is reconstructed
        symbolically downstream from the *single* tasklet that computes it. Splitting
        that producer across intermediate register transients leaves the downstream
        propagation referencing a transient that is local to another state, so such
        producers must be left intact.

        :param sdfg: The SDFG to scan (recursively into nested SDFGs).
        :returns: The set of data names read by any interstate-edge assignment.
        """
        names: Set[str] = set()
        for e in sdfg.all_interstate_edges():
            for v in e.data.assignments.values():
                try:
                    symexpr = dace.symbolic.pystr_to_symbolic(v)
                except Exception:
                    continue
                names.update(str(s) for s in symexpr.free_symbols)
                names.update(str(s) for s in dace.symbolic.arrays(symexpr))
        return names

    def _plan_multi_output_split(self, tasklet: dace.nodes.Tasklet, state) -> Optional[List[Dict[str, Any]]]:
        """
        Validate and plan splitting a tasklet with more than one output connector into one
        single-output tasklet per output connector.

        The body must be a list of top-level ``target = expr`` assignments, one per output
        connector, in producer-before-consumer order. A statement may read an earlier
        statement's output connector (a RAW that is routed through the produced array by
        :meth:`_apply_multi_output_split`). The split is refused -- ``None`` is returned,
        leaving the tasklet intact -- when a precondition does not hold or when it cannot be
        proven sound (a WAR/WAW that the split would reorder).

        :param tasklet: The candidate multi-output tasklet.
        :param state: The state the tasklet lives in.
        :returns: An ordered list of per-output plan dicts, or ``None`` to refuse the split.
        """
        if tasklet.code.language != dace.dtypes.Language.Python:
            return None
        try:
            body = ast.parse(tasklet.code.as_string).body
        except (SyntaxError, ValueError):
            return None
        if len(body) < 2:
            return None
        # Every top-level statement must be a plain ``name = expr`` assignment: no
        # AugAssign, no ``if``, no subscripted / tuple / starred / multiple targets.
        for stmt in body:
            if not isinstance(stmt, ast.Assign):
                return None
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                return None

        targets = [stmt.targets[0].id for stmt in body]
        out_conns = set(tasklet.out_connectors.keys())
        # One statement per output connector, and no statement targets a shared local temp
        # (a target that is not an output connector cannot be materialised per output).
        if len(targets) != len(set(targets)) or set(targets) != out_conns:
            return None

        in_conns = set(tasklet.in_connectors.keys())
        out_edge_by_conn: Dict[str, Any] = {}
        for oe in state.out_edges(tasklet):
            if oe.src_conn in out_edge_by_conn or oe.data is None or oe.data.data is None:
                return None  # fan-out of one output connector, or a dataless output: refuse
            out_edge_by_conn[oe.src_conn] = oe
        if set(out_edge_by_conn.keys()) != out_conns:
            return None
        in_edge_by_conn: Dict[str, Any] = {}
        for ie in state.in_edges(tasklet):
            if ie.dst_conn is not None:
                in_edge_by_conn[ie.dst_conn] = ie

        # In-place read-modify-write: an input connector reads an array that an output
        # connector also writes (polybench covariance's finalize -- ``cov_ij_in << cov[i,j]``
        # AND ``cov_ij_out >> cov[i,j]``, where ``cov[i,j]`` is produced by an upstream WCR
        # sum-reduction over ``k``). Splitting would route the output through an intermediate
        # access node of that same array while the input still reads the original (WCR-reduced)
        # source; the read / in-place-normalize / mirror ordering is NOT reliably preserved
        # through the downstream nested-SDFG / reduction lowering (the finalize reads
        # ``cov[i,j]`` before it is written and the kernel produces NaN). Leave the tasklet
        # intact -- exactly the pre-existing behaviour that keeps covariance correct.
        read_arrays = {
            ie.data.data
            for ie in in_edge_by_conn.values() if ie.data is not None and ie.data.data is not None
        }
        write_arrays = {oe.data.data for oe in out_edge_by_conn.values()}
        if read_arrays & write_arrays:
            return None

        target_index = {t: k for k, t in enumerate(targets)}

        # Classify every right-hand-side name of each statement.
        ordered: List[Dict[str, Any]] = []
        for k, stmt in enumerate(body):
            rhs_names = {node.id for node in ast.walk(stmt.value) if isinstance(node, ast.Name)}
            input_reads: Set[str] = set()
            cross_reads: Set[str] = set()
            for name in rhs_names:
                if name in target_index and target_index[name] < k:
                    # Reads an output connector produced by an earlier statement (RAW).
                    # A reassigned name shadows any same-named input connector (Python
                    # last-write-wins), so this branch takes priority.
                    cross_reads.add(name)
                elif name in in_conns:
                    if name not in in_edge_by_conn:
                        return None  # an input connector with no in-edge: cannot route
                    input_reads.add(name)  # entry-value read of the input array
                elif name in out_conns:
                    # An output connector referenced at or before its own definition without
                    # also being an input connector: reading an undefined value. Refuse.
                    return None
                # Otherwise a symbol / function / constant: inlined into the body verbatim.
            ordered.append({
                'out_conn': targets[k],
                'code': ast.unparse(stmt),
                'input_reads': input_reads,
                'cross_reads': cross_reads,
            })

        # ``reaches[j][k]`` == statement ``k`` is forced (transitively, through RAW reads)
        # to run strictly after statement ``j``.
        n = len(body)
        reaches = [[False] * n for _ in range(n)]
        for k, plan in enumerate(ordered):
            for name in plan['cross_reads']:
                reaches[target_index[name]][k] = True
        for mid in range(n):
            for a in range(n):
                if reaches[a][mid]:
                    for b in range(n):
                        if reaches[mid][b]:
                            reaches[a][b] = True

        # WAW: two outputs writing the same array at possibly-overlapping subsets must be
        # ordered by a RAW chain; otherwise the surviving value is order dependent. Distinct
        # arrays are assumed non-aliasing (DaCe's standard model); the same array is
        # conservatively treated as possibly overlapping. (Input/output aliasing -- an
        # in-place read-modify-write -- was already refused above.)
        for a in range(n):
            a_arr = out_edge_by_conn[ordered[a]['out_conn']].data.data
            for b in range(a + 1, n):
                b_arr = out_edge_by_conn[ordered[b]['out_conn']].data.data
                if a_arr == b_arr and not (reaches[a][b] or reaches[b][a]):
                    return None
        return ordered

    def _route_read_output_to_destination(self, state, base_name: str, k: int, mid: dace.nodes.AccessNode,
                                          out_edge) -> None:
        """
        Forward a read-later output's intermediate access node ``mid`` (holding the produced
        subset of its array) on to the original output destination.

        The intermediate and the destination name the same array, so the copy reads and
        writes the same subset. How that is expressed depends on the destination:

        - **Access node** (top-level / loop-body state): a direct ``mid -> destination``
          self-copy carrying ``subset`` and a matching ``other_subset``. A same-array
          access-node-to-access-node single-element copy is handled by the downstream
          nested-SDFG passes; without ``other_subset`` the backend would copy from offset 0.
        - **Map exit** (or any non-access-node): the same-array-with-``other_subset`` form is
          rejected on a ``-> MapExit`` edge, so the value is copied out through a scalar
          store tasklet, which also keeps the map scope's write set complete.

        :param state: The state being rewritten.
        :param base_name: The original tasklet name, used to name a store tasklet.
        :param k: The output index, used to name a store tasklet.
        :param mid: The intermediate access node holding the produced value.
        :param out_edge: The original output edge (its ``dst``/``dst_conn``/memlet are reused).
        """
        if isinstance(out_edge.dst, dace.nodes.AccessNode):
            forward = copy.deepcopy(out_edge.data)
            forward.other_subset = copy.deepcopy(forward.subset)
            state.add_edge(mid, None, out_edge.dst, out_edge.dst_conn, forward)
        else:
            store = state.add_tasklet(name=f"{base_name}_out_{k}_store",
                                      inputs={'_store_in'},
                                      outputs={'_store_out'},
                                      code='_store_out = _store_in')
            state.add_edge(mid, None, store, '_store_in', copy.deepcopy(out_edge.data))
            state.add_edge(store, '_store_out', out_edge.dst, out_edge.dst_conn, copy.deepcopy(out_edge.data))

    def _apply_multi_output_split(self, tasklet: dace.nodes.Tasklet, state, ordered: List[Dict[str, Any]]) -> None:
        """
        Rewrite a validated multi-output tasklet into one single-output tasklet per output
        connector (plan produced by :meth:`_plan_multi_output_split`).

        A cross-statement read of an earlier output connector is materialised through that
        output's array via an intermediate access node, so the producer-before-consumer
        order survives as a real data dependence in the same state.

        :param tasklet: The multi-output tasklet to replace.
        :param state: The state the tasklet lives in.
        :param ordered: The ordered per-output plan from :meth:`_plan_multi_output_split`.
        """
        out_edge_by_conn = {oe.src_conn: oe for oe in state.out_edges(tasklet)}
        in_edge_by_conn = {ie.dst_conn: ie for ie in state.in_edges(tasklet) if ie.dst_conn is not None}
        scope_entry = state.entry_node(tasklet)

        # Output connectors that a later statement reads need an intermediate access node
        # (through which the consumer reads the produced array).
        read_outputs: Set[str] = set()
        for plan in ordered:
            read_outputs.update(plan['cross_reads'])

        state.remove_node(tasklet)

        emitted: List[dace.nodes.Tasklet] = []
        mid_access: Dict[str, dace.nodes.AccessNode] = {}
        for k, plan in enumerate(ordered):
            out_conn = plan['out_conn']
            out_edge = out_edge_by_conn[out_conn]
            input_conns = set(plan['input_reads']) | set(plan['cross_reads'])
            t = state.add_tasklet(name=f"{tasklet.name}_out_{k}",
                                  inputs=set(input_conns),
                                  outputs={out_conn},
                                  code=plan['code'])
            emitted.append(t)
            # Inputs read from the original input edges (entry values). A fresh memlet per
            # edge -- DaCe rejects a reused subset object.
            for in_conn in plan['input_reads']:
                ie = in_edge_by_conn[in_conn]
                state.add_edge(ie.src, ie.src_conn, t, in_conn, copy.deepcopy(ie.data))
            # Cross-statement RAW reads route through the earlier output's array node.
            for name in plan['cross_reads']:
                state.add_edge(mid_access[name], None, t, name, copy.deepcopy(out_edge_by_conn[name].data))
            # Output: a value read by a later statement is materialised on its own access
            # node of the produced array (the consumer reads it back from there); otherwise
            # the value goes straight to the original destination.
            if out_conn in read_outputs:
                mid = state.add_access(out_edge.data.data)
                state.add_edge(t, out_conn, mid, None, copy.deepcopy(out_edge.data))
                mid_access[out_conn] = mid
                self._route_read_output_to_destination(state, tasklet.name, k, mid, out_edge)
            else:
                state.add_edge(t, out_conn, out_edge.dst, out_edge.dst_conn, copy.deepcopy(out_edge.data))
            # A data-less sub-tasklet (constant / symbol-only output) inside a scope must be
            # anchored to the scope entry so it does not float outside the map.
            if state.in_degree(t) == 0 and scope_entry is not None:
                state.add_edge(scope_entry, None, t, None, dace.memlet.Memlet(None))

        # Independent outputs (distinct arrays, no cross-statement read, no shared scope /
        # input) leave their sub-tasklets in separate weakly-connected components -- floating
        # disconnected islands. Chain any such consecutive pair with an empty-Memlet ordering
        # edge in body order. A real RAW (routed through the array) or a shared scope /
        # access node already connects the pair, so no extra edge is added there.
        for k in range(len(emitted) - 1):
            if not nx.has_path(state.nx.to_undirected(as_view=True), emitted[k], emitted[k + 1]):
                state.add_edge(emitted[k], None, emitted[k + 1], None, dace.memlet.Memlet(None))

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> Optional[Dict[str, Set[str]]]:
        """
        Split every multi-operation Python tasklet in the SDFG into single-op tasklets.

        :param sdfg: The SDFG to transform in place.
        :param pipeline_results: Results of prior passes in the pipeline (unused).
        :returns: ``{'added_symbols': <symbol names>, 'split_tasklets': <names of the tasklets that were split>}``,
                  or ``None`` if nothing was declared and no tasklet was split.
        """
        added_symbols = self._add_missing_symbols(sdfg)
        split_access_counter = 0

        symbol_lifted_data = self._symbol_lifted_data(sdfg)

        tasklets_to_split = list()  # tasklet, parent_graph, ssa_statements
        multi_output_to_split = list()  # tasklet, parent_graph, ordered per-output plan
        for n, g in sdfg.all_nodes_recursive():
            if isinstance(n, dace.nodes.Tasklet):
                c: CodeBlock = n.code
                # A tasklet with >1 output connector cannot take the single-output SSA path.
                # Split it into one tasklet per output connector instead (materialising each
                # write independently for downstream loop-fission / lift). Respect the same
                # "leave a symbol-lifted producer intact" guard as the single-output path;
                # ``_plan_multi_output_split`` leaves the tasklet intact (returns ``None``)
                # when its preconditions or WAR/WAW soundness do not hold.
                if len(n.out_connectors) > 1:
                    if any(oe.data is not None and oe.data.data in symbol_lifted_data for oe in g.out_edges(n)):
                        continue
                    plan = self._plan_multi_output_split(n, g)
                    if plan is not None:
                        multi_output_to_split.append((n, g, plan))
                    continue

                # Leave a conditional-write tasklet (``if cond: out = e`` -- the frontend's
                # ``A[mask] = value`` form, newast.py:2868) intact. It is not straight-line,
                # so the SSA split below mangles it into a dangling map connector; instead
                # ``ConvertTaskletsToTileOps`` lowers it to a masked ``TileStore`` (the mask
                # ``cond`` gates the store, no old-value read).
                try:
                    _body = ast.parse(c.as_string).body
                    if len(_body) == 1 and isinstance(_body[0], ast.If) and not _body[0].orelse:
                        continue
                except (SyntaxError, ValueError):
                    pass

                # Leave intact a producer whose scalar output is symbol-lifted on an
                # interstate edge (the value is reconstructed symbolically downstream
                # from this one tasklet; splitting it would expose an intermediate
                # transient that is local to another state).
                if any(oe.data is not None and oe.data.data in symbol_lifted_data for oe in g.out_edges(n)):
                    continue

                input_types = set()
                for ie in g.in_edges(n):
                    if ie.data is None:
                        continue
                    if ie.data.data is None:
                        continue
                    input_types.add(g.sdfg.arrays[ie.data.data].dtype)

                # Collect symbolic types
                try:
                    tokens = c.as_string.split(" = ")
                    assert len(tokens) == 2
                    lhs = tokens[0].strip()
                    rhs = tokens[1].strip()
                    code_expr_rhs = dace.symbolic.SymExpr(rhs)
                    code_expr_lhs = dace.symbolic.SymExpr(lhs)
                    for free_sym in code_expr_rhs.free_symbols.union(code_expr_lhs.free_symbols):
                        if str(free_sym) in g.sdfg.symbols:
                            input_types.add(g.sdfg.symbols[str(free_sym)])
                except Exception:
                    # Nested comparisons might make symexpr / sympify crash
                    code_tokens = self.token_split_variable_names(c.as_string)
                    for free_sym in code_tokens:
                        if str(free_sym) in g.sdfg.symbols:
                            input_types.add(g.sdfg.symbols[str(free_sym)])

                # Loop-region iterators (e.g. ``_loop_it_0``) are int64 index
                # variables -- the tile path materialises an int64 lane-id tile for
                # them -- but they are not registered in ``sdfg.symbols`` (only
                # interstate-assignment targets are; see ``_add_missing_symbols``).
                # Without counting them the width inference sees only a narrower
                # visible symbol and mis-types an iterator-fed integer intermediate:
                # TSVC s315 ``(7 * _loop_it_0) % LEN_1D`` with ``LEN_1D`` int32 yields
                # an int32 intermediate that then narrows the int64 iterator operand.
                # Count each unregistered free symbol as int64.
                for free_sym in n.free_symbols:
                    if str(free_sym) not in g.sdfg.symbols:
                        input_types.add(dace.int64)

                # Best-effort *fallback* dtype, used only for an intermediate that the
                # rigorous per-statement inference below cannot type (e.g. an
                # unparseable nested compare). Promote ALL input operand types together
                # via the DaCe promotion lattice (``result_type_of``) rather than a
                # hard-coded float/int bucket -- this covers every dtype uniformly
                # (bool, int8..int64, float16/32/64, complex64/128) with the standard
                # widening rules (same-kind widening; int promotes to float; real
                # promotes to complex).
                if input_types:
                    input_type = functools.reduce(dace.dtypes.result_type_of, input_types)
                else:
                    # Default to float when the tasklet consists purely of constants.
                    input_type = dace.float64

                if c.language == dace.dtypes.Language.Python:
                    ssa_statements = to_ssa(c.as_string)
                    if len(ssa_statements) != 1:
                        # Rigorously infer EACH split intermediate's type from its
                        # operands (DaCe promotion: same-kind widening, fp32->fp64,
                        # int32->int64, complex64->complex128, ...) instead of stamping
                        # the whole chain with one coarse ``input_type``. The leaf types
                        # are the input connectors' array dtypes and the body symbols
                        # (unregistered ones -- e.g. loop iterators -- are int64 index
                        # vars). ``input_type`` stays as the best-effort fallback for any
                        # intermediate the inference cannot type.
                        leaf_types = {}
                        for ie in g.in_edges(n):
                            if ie.data is not None and ie.data.data is not None and ie.dst_conn is not None:
                                leaf_types[ie.dst_conn] = g.sdfg.arrays[ie.data.data].dtype
                        for free_sym in n.free_symbols:
                            nm = str(free_sym)
                            leaf_types[nm] = g.sdfg.symbols[nm] if nm in g.sdfg.symbols else dace.int64
                        inferred = _infer_ssa_intermediate_types(ssa_statements, leaf_types, input_type)
                        tasklets_to_split.append((n, g, ssa_statements, input_type, inferred))

        # Names are collected here, before the rewrites below detach the tasklets from their states.
        split_names = {t.name for t, *_ in tasklets_to_split} | {t.name for t, *_ in multi_output_to_split}

        # Previous tasklet:
        # i1 -> |         |
        # i2 -> | tasklet | -> o1
        # i3 -> |         |
        # Now they need to be split e.g.:
        # i1 -> |         |
        # i2 -> | tasklet | -> tmp1 -> |         |
        #                        i3 -> | tasklet | -> o1

        # For the case a tasklet goes to a taskelt that needs to be split
        # If we have t1 -> t2 but then split t1 to (t1.1, t1.2) -> t2
        # For each tasklet we split we need to track the new input and output maps
        for tasklet, state, ssa_statements, input_type, inferred in tasklets_to_split:
            assert isinstance(state, dace.SDFGState)
            assert isinstance(tasklet, dace.nodes.Tasklet)
            assert tasklet in state.nodes()
            tasklet_input_edges = state.in_edges(tasklet)
            tasklet_output_edges = state.out_edges(tasklet)

            tasklet_in_degree = state.in_degree(tasklet)
            tasklet_in_edges = state.in_edges(tasklet)
            # The scope entry (MapEntry) the tasklet lives under, captured before it
            # is removed. A split sub-tasklet whose SSA statement is symbol-only (e.g.
            # ``__t1 = double(N)`` -- N is a symbol inlined into the body, so the
            # sub-tasklet has NO data inputs) would otherwise have zero in-edges and
            # float OUTSIDE this scope, corrupting scope_dict and making the
            # transient-feeding edges cross into the scope illegally ("sink node should
            # be a data node"). Every such sub-tasklet is anchored to ``scope_entry``
            # with an empty dependency memlet so it stays within the map.
            scope_entry = state.entry_node(tasklet)
            # A body name is a symbol (to be inlined) rather than a data input
            # connector iff it is in scope as a symbol. ``symbols_defined_at`` and
            # ``sdfg.symbols`` miss loop-region iterators (e.g. ``_loop_it_0``),
            # so we also add the original tasklet's own ``free_symbols``: those are
            # exactly the names it reads that are not connectors, i.e. its symbols.
            available_symbols = {str(s)
                                 for s in state.symbols_defined_at(tasklet)
                                 }.union({str(s)
                                          for s in sdfg.symbols.keys()}).union(tasklet.free_symbols)
            state.remove_node(tasklet)
            # Variables assigned by a comparison / boolean SSA statement hold a bool;
            # their split transient must be typed bool (mask / ITE-cond contract), not
            # the numeric ``input_type``.
            bool_vars = {lhs for stmt in ssa_statements if _ssa_lhs_is_bool(stmt) for lhs in _get_vars(stmt)[0]}
            added_tasklets = list()
            for i, ssa_statement in enumerate(ssa_statements):  # Since SSA we are going to add in a line
                lhs_vars, rhs_vars = _get_vars(ssa_statement)
                assert "True" not in rhs_vars
                assert "False" not in rhs_vars

                # Symbols read in the body become inlined values, not input connectors.
                symbol_rhs_vars = {rhs_var for rhs_var in rhs_vars if rhs_var in available_symbols}
                rhs_vars = set(rhs_vars) - symbol_rhs_vars
                assert len(lhs_vars) == 1
                t = state.add_tasklet(
                    name=f"{tasklet.name}_split_{i}",
                    inputs=set(rhs_vars),
                    outputs=set(lhs_vars),
                    code=ssa_statement,
                )
                for rhs_var in rhs_vars:
                    t.add_in_connector(rhs_var)
                for lhs_var in lhs_vars:
                    t.add_out_connector(lhs_var)
                added_tasklets.append(t)

            added_accesses = dict()
            for i, t in enumerate(added_tasklets):
                assert isinstance(t, dace.nodes.Tasklet)
                # First do inputs
                if i == 0:  # First new tasklet
                    # The inputs should be available in the input data
                    # This might not be matched if a symbol is used in the tasklet
                    # for example X = dace.float64(symbol)
                    matched_in_conns = set()
                    for in_conn in t.in_connectors:
                        matching_in_edges = {ie for ie in tasklet_input_edges if ie.dst_conn == in_conn}
                        assert len(
                            matching_in_edges
                        ) <= 1, f"Required 1 matching in edge always, found: {matching_in_edges}, original tasklet code: {tasklet.code.as_string}, current tasklet code: {t.code.as_string}"

                        if len(matching_in_edges) > 0:
                            matching_in_edge = next(iter(matching_in_edges))

                            state.add_edge(matching_in_edge.src, matching_in_edge.src_conn, t, in_conn,
                                           copy.deepcopy(matching_in_edge.data))
                            matched_in_conns.add(in_conn)

                    for in_conn in list(t.in_connectors.keys()):
                        if in_conn not in matched_in_conns:
                            t.remove_in_connector(in_conn)

                    # dace.float64(symbol) has no sources after split,
                    # but if we for example inside a map we need to add a dependency edge
                    if len(matched_in_conns) == 0 and tasklet_in_degree > 0:
                        for ie in tasklet_in_edges:
                            state.add_edge(ie.src, None, t, None, dace.memlet.Memlet(None))
                else:
                    # Input comes from transient accesses (each unique and needs to be added to the SDFG)
                    # or from the unused in edges
                    for in_conn in t.in_connectors.keys():
                        matching_in_edges = {ie for ie in tasklet_input_edges if ie.dst_conn == in_conn}
                        if len(matching_in_edges) == 0:
                            array_name = f"{in_conn}{self.tmp_access_identifier}{split_access_counter}"
                            if array_name not in state.sdfg.arrays:
                                state.sdfg.add_scalar(
                                    name=array_name,
                                    dtype=dace.bool_ if in_conn in bool_vars else inferred.get(in_conn, input_type),
                                    storage=dace.dtypes.StorageType.Register,
                                    transient=True,
                                )
                            # Independently of whether the descriptor already existed: a
                            # second run of this pass over the same SDFG (canonicalize
                            # splits, then the vectorizer splits again) finds the scalar
                            # present but has no access node for it in THIS invocation.
                            if array_name not in added_accesses:
                                added_accesses[array_name] = state.add_access(array_name)
                            state.add_edge(
                                added_accesses[array_name], None, t, in_conn,
                                dace.memlet.Memlet.from_array(dataname=array_name,
                                                              datadesc=state.sdfg.arrays[array_name]))
                        else:
                            assert len(matching_in_edges) == 1
                            matching_in_edge = next(iter(matching_in_edges))
                            state.add_edge(matching_in_edge.src, matching_in_edge.src_conn, t, in_conn,
                                           copy.deepcopy(matching_in_edge.data))

                # Then do the outputs
                if i == len(added_tasklets) - 1:  # last tasklet
                    # The outputs should be available in the output data
                    for out_conn in t.out_connectors:
                        matching_out_edges = {oe for oe in tasklet_output_edges if oe.src_conn == out_conn}
                        assert len(matching_out_edges) == 1
                        matching_out_edge = next(iter(matching_out_edges))
                        state.add_edge(t, out_conn, matching_out_edge.dst, matching_out_edge.dst_conn,
                                       copy.deepcopy(matching_out_edge.data))
                else:
                    # Output should have been added already
                    assert len(t.out_connectors) == 1
                    out_conn = next(iter(t.out_connectors))
                    array_name = f"{out_conn}{self.tmp_access_identifier}{split_access_counter}"
                    if array_name not in state.sdfg.arrays:
                        state.sdfg.add_scalar(
                            name=array_name,
                            dtype=dace.bool_ if out_conn in bool_vars else inferred.get(out_conn, input_type),
                            storage=dace.dtypes.StorageType.Register,
                            transient=True,
                        )
                    if array_name not in added_accesses:
                        added_accesses[array_name] = state.add_access(array_name)
                    state.add_edge(
                        t, out_conn, added_accesses[array_name], None,
                        dace.memlet.Memlet.from_array(dataname=array_name, datadesc=state.sdfg.arrays[array_name]))

            # Anchor every sub-tasklet that ended up with NO incoming edge into the
            # original scope. A symbol-only SSA statement (``__t1 = double(N)``) has no
            # data inputs, so its sub-tasklet has zero in-edges; without an edge from
            # the scope entry it floats outside the map (the i==0 branch only anchors
            # the FIRST sub-tasklet). The empty dependency memlet from ``scope_entry``
            # keeps it inside the map scope (top-level tasklets, scope_entry=None, are
            # legitimately allowed to be sources and need no anchor).
            if scope_entry is not None:
                for t in added_tasklets:
                    if state.in_degree(t) == 0:
                        state.add_edge(scope_entry, None, t, None, dace.memlet.Memlet(None))

            split_access_counter += 1

        # Split every multi-output tasklet into one single-output tasklet per output
        # connector. These tasklets are disjoint from ``tasklets_to_split`` (the
        # single-output SSA path never collected them), so the two rewrites are
        # independent; each plan re-reads the tasklet's edges at apply time.
        for tasklet, state, ordered in multi_output_to_split:
            assert isinstance(state, dace.SDFGState)
            assert isinstance(tasklet, dace.nodes.Tasklet)
            assert tasklet in state.nodes()
            self._apply_multi_output_split(tasklet, state, ordered)

        sdfg.validate()
        if not added_symbols and not split_names:
            return None
        return {'added_symbols': added_symbols, 'split_tasklets': split_names}
