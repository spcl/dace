"""Schedule-based cuTile Python code generation target.

AccessNode-centric design: MapEntry emits only PIDs and map variable
bindings; each CuTile_Tile AccessNode handles its own ``ct.load`` /
``ct.store`` (or ``ct.gather`` / ``ct.scatter`` for non-aligned
tiles).  MapExit is a no-op.
"""

import ast
import warnings
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

import sympy as sp

from dace import data, dtypes, registry, subsets, symbolic
import dace.codegen.dispatcher as dispatcher_mod
from dace.codegen.py import control_flow as py_cflow
from dace.codegen.py.framecode import codeblock_to_python
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.codegen.py.target import PythonTargetCodeGenerator
from dace.sdfg import nodes
from dace.sdfg.state import SymbolResolver
from dace.symbolic import symstr

if TYPE_CHECKING:
    from dace.codegen.py.framecode import DaCePythonCodeGenerator
    from dace.sdfg import SDFG, SDFGState

# ---------------------------------------------------------------------------
# Constants — avoid magic strings throughout the file.
# ---------------------------------------------------------------------------

#: Prefix for outer (input) scope connectors on a MapEntry / MapExit.
_SCOPE_IN_PREFIX: str = "IN_"

#: Prefix for inner (output) scope connectors on a MapEntry / MapExit.
_SCOPE_OUT_PREFIX: str = "OUT_"

#: Kernel-side spellings for math calls that appear verbatim in tasklet
#: bodies (``__out = sqrt(__in1)``). Host code resolves these names via the
#: module-level numpy aliases in ``sympy_function_redefinitions``, but a
#: ``@ct.kernel`` cannot capture a numpy ufunc as a constant, so calls inside
#: a cuTile scope are rewritten to their ``cuda.tile`` equivalents.
_CT_MATH_FUNCS: Dict[str, str] = {
    'sqrt': 'sqrt',
    'rsqrt': 'rsqrt',
    'exp': 'exp',
    'exp2': 'exp2',
    'log': 'log',
    'log2': 'log2',
    'sin': 'sin',
    'cos': 'cos',
    'tan': 'tan',
    'sinh': 'sinh',
    'cosh': 'cosh',
    'tanh': 'tanh',
    'floor': 'floor',
    'ceil': 'ceil',
    'ceiling': 'ceil',
    'abs': 'abs',
    'fabs': 'abs',
    'Abs': 'abs',
    'isnan': 'isnan',
    'atan2': 'atan2',
    'arctan2': 'atan2',
    'pow': 'pow',
    'fmin': 'minimum',
    'minimum': 'minimum',
    'fmax': 'maximum',
    'maximum': 'maximum',
}

#: Dtype names an explicit cast can carry (``float64`` / ``int32`` / ``bool`` / ...),
#: built from the dtype registry. VTI keeps casts rather than stripping them, so a
#: kept ``dace.float64(x)`` / ``np.int32(x)`` reaches the kernel module-qualified;
#: cuda.tile can't resolve ``dace``/``np``, so it is rewritten to ``ct.astype``.
_CT_CAST_DTYPES = {s.split('::')[-1] for s in dtypes.TYPECLASS_TO_STRING.values()}


def _ct_attr(name: str) -> ast.Attribute:
    """Build an ``ast`` node for ``ct.<name>``.

    :param name: Attribute name on the ``cuda.tile`` module alias ``ct``.
    :returns: The ``ast.Attribute`` node (load context).
    """
    return ast.Attribute(value=ast.Name(id='ct', ctx=ast.Load()), attr=name, ctx=ast.Load())


def _literal_value(arm: ast.expr) -> Optional[object]:
    """Extract the numeric value of a literal expression, if it is one.

    Recognizes plain numeric constants and signed constants
    (``ast.UnaryOp(USub/UAdd, Constant)`` — the AST form of ``-1.0``).

    :param arm: The expression to inspect.
    :returns: The (signed) numeric value, or ``None`` if *arm* is not a
        numeric literal.
    """
    if isinstance(arm, ast.UnaryOp) and isinstance(arm.op, (ast.USub, ast.UAdd)):
        inner = _literal_value(arm.operand)
        if inner is None:
            return None
        return -inner if isinstance(arm.op, ast.USub) else inner
    if isinstance(arm, ast.Constant) and isinstance(arm.value, (int, float)) and not isinstance(arm.value, bool):
        return arm.value
    return None


def _tile_tainted_names(tree: ast.AST, seed: Iterable[str]) -> set:
    """Names that (may) hold tile values within a tasklet body.

    Simple flow-insensitive taint pass: seeded with the tile-valued input
    connectors, propagated through ``Assign``/``AugAssign``/``AnnAssign``
    statements whose right-hand side mentions a tainted name.  This is an
    over-approximation (any expression involving a tile taints its target),
    which is safe: rewriting a ternary to ``ct.where`` is also valid for
    scalar conditions.

    :param tree: The parsed tasklet body.
    :param seed: Tile-valued connector names.
    :returns: The set of (possibly) tile-valued names.
    """
    tainted = set(seed)
    changed = True
    while changed:
        changed = False
        for stmt in ast.walk(tree):
            if not isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                continue
            if stmt.value is None:
                continue
            rhs_names = {n.id for n in ast.walk(stmt.value) if isinstance(n, ast.Name)}
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            if isinstance(stmt, ast.AugAssign):
                # x += tile taints x; x (tainted) += y stays tainted anyway.
                rhs_names |= {n.id for n in ast.walk(stmt.target) if isinstance(n, ast.Name)}
            if not (rhs_names & tainted):
                continue
            for tgt in targets:
                for n in ast.walk(tgt):
                    if isinstance(n, ast.Name) and n.id not in tainted:
                        tainted.add(n.id)
                        changed = True
    return tainted


class _CuTileTaskletRewriter(ast.NodeTransformer):
    """AST rewrites needed to run a Python tasklet body inside a ``@ct.kernel``.

    1. Math calls by bare name (host-side numpy aliases) become ``ct.*``
       (:data:`_CT_MATH_FUNCS`); unmapped known-host-alias names cannot run in
       a kernel and trigger a warning.
    2. Conditional expressions whose condition mentions a tile-valued name
       (a tile connector, or a body-local assigned from one — see
       :func:`_tile_tainted_names`) become ``ct.where(cond, then, else)`` —
       tiles cannot be branched on.  A numeric-literal arm opposite a tile
       connector of known dtype is wrapped in ``ct.astype`` (cuda.tile
       rejects where-arms whose dtype cannot implicitly cast to the result),
       EXCEPT for a fractional float literal opposite an integer tile:
       ``ct.astype(0.5, ct.int32)`` would silently truncate to 0, so the arm
       is left uncast (Python promotes; if cuda.tile rejects the int/float
       mix, that loud error is preferable to silent truncation).
    """

    def __init__(self, tile_conn_dtypes: Dict[str, str], tainted_names: Optional[set] = None) -> None:
        #: Tile-valued connector name -> ``ct`` dtype attribute name.
        self._tile_conns = tile_conn_dtypes
        #: All (possibly) tile-valued names, including body locals.
        self._tainted = tainted_names if tainted_names is not None else set(tile_conn_dtypes)

    def visit_Call(self, node: ast.Call) -> ast.Call:
        """Rewrite bare-name math calls to their ``ct.*`` spelling.

        The rewrite is applied regardless of operand tile-ness: probed
        empirically (2026-07, cuda.tile 13.x), the ``ct.*`` math functions
        accept plain Python scalars as operands (``ct.abs(-3.5)``,
        ``ct.pow(2.0, 3.0)``, ``ct.sqrt(4.0)``, ``ct.maximum(1.0, 2.0)`` all
        compile and produce correct values), so scalar-only calls need no
        gating.
        """
        from dace.codegen.py.sympy_function_redefinitions import _NUMPY_EQUIVALENTS
        self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr in _CT_CAST_DTYPES
                and isinstance(node.func.value, ast.Name) and node.func.value.id in ('dace', 'numpy', 'np')
                and len(node.args) == 1):
            # A kept dtype cast reaches the kernel module-qualified (``dace.float64(x)``);
            # cuda.tile can't resolve ``dace``/``np``, so rewrite to ``ct.astype(x, ct.<dtype>)``.
            dt = node.func.attr
            return ast.Call(func=_ct_attr('astype'),
                            args=[node.args[0], _ct_attr('bool_' if dt == 'bool' else dt)],
                            keywords=node.keywords)
        if isinstance(node.func, ast.Name):
            ct_name = _CT_MATH_FUNCS.get(node.func.id)
            if ct_name is not None:
                node.func = _ct_attr(ct_name)
            elif node.func.id in _NUMPY_EQUIVALENTS:
                warnings.warn(f"cuTile codegen: math function {node.func.id!r} has no cuda.tile "
                              f"equivalent; the kernel will fail to compile at runtime.")
        return node

    def _cast_literal_arm(self, arm: ast.expr, other: ast.expr) -> ast.expr:
        """Wrap a numeric-literal where-arm in ``ct.astype`` to the opposite
        tile arm's dtype (when known and lossless)."""
        value = _literal_value(arm)
        if value is None or not (isinstance(other, ast.Name) and other.id in self._tile_conns):
            return arm
        dtype_name = self._tile_conns[other.id]
        if isinstance(value, float) and value != int(value) and ('int' in dtype_name or 'bool' in dtype_name):
            # Fractional literal vs integer tile: ct.astype would silently
            # truncate (0.5 -> 0). Leave the arm uncast — see class docstring.
            return arm
        return ast.Call(func=_ct_attr('astype'), args=[arm, _ct_attr(dtype_name)], keywords=[])

    def visit_IfExp(self, node: ast.IfExp) -> ast.expr:
        """Rewrite ``A if cond else B`` over tiles to ``ct.where(cond, A, B)``."""
        self.generic_visit(node)
        cond_names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        if not (cond_names & self._tainted):
            return node  # Scalar condition: plain Python branching is fine.
        then_arm = self._cast_literal_arm(node.body, node.orelse)
        else_arm = self._cast_literal_arm(node.orelse, node.body)
        return ast.Call(func=_ct_attr('where'), args=[node.test, then_arm, else_arm], keywords=[])


def _rewrite_cutile_tasklet_code(code: str, tile_conn_dtypes: Dict[str, str]) -> str:
    """Rewrite a tasklet body for execution inside a ``@ct.kernel``.

    See :class:`_CuTileTaskletRewriter` for the rewrites applied.

    :param code: The tasklet body (Python source).
    :param tile_conn_dtypes: Tile-valued connector name -> ``ct`` dtype
        attribute name (e.g. ``{"__in2": "float64"}``).
    :returns: The rewritten source (unchanged if parsing fails).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    tainted = _tile_tainted_names(tree, tile_conn_dtypes.keys())
    tree = _CuTileTaskletRewriter(tile_conn_dtypes, tainted).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _is_device_scalar(desc: object) -> bool:
    """Whether *desc* is a numeric ``data.Scalar`` that must reach a cuTile
    kernel through device memory.

    The ``cuda.tile`` launch boundary (1.5.0) cannot pass numeric scalars by
    value faithfully: every by-value Python/numpy float is typed ``float32``
    (silent float64 precision loss), a Python int >= 2**31 raises
    ``OverflowError`` (int32 typing), and numpy int/float scalar types are
    rejected outright. All float/int/uint Scalars are therefore passed as
    1-element device arrays and bound as scalar tiles via
    ``ct.load(name, (0,), shape=()).item()`` in-kernel (bit-exact for f64 and
    int64/uint64). Bool scalars stay by value: the boundary types them
    exactly (``ScalarConstraint(bool_)``).

    :param desc: A data descriptor (or ``None``).
    :returns: ``True`` for float/int/uint ``data.Scalar`` descriptors.
    """
    return isinstance(desc, data.Scalar) and desc.dtype.as_numpy_dtype().kind in "fiu"


def _scalar_tile_load(name: str) -> str:
    """The 0-d tile load binding a device-scalar kernel parameter.

    The trailing ``.item()`` (``cuda.tile``'s ``Tile.item``, equivalent to
    ``reshape(())``) is a device-side scalar extraction -- NOT the host
    ``numpy``/``cupy`` ``.item()`` -- that normalizes the load to a canonical
    scalar tile. It stays on device (no host copy) and yields the scalar form
    contexts like load indices and ``range`` bounds expect.

    :param name: The 1-element device-array parameter name.
    :returns: The ``ct.load(...).item()`` expression producing a scalar tile.
    """
    return f"ct.load({name}, (0,), shape=()).item()"


def _scalar_element_load(name: str, index_exprs: str) -> str:
    """The scalar load of a single addressed element of an in-kernel array.

    Like :func:`_scalar_tile_load`, the trailing device-side ``.item()``
    normalizes the load to a scalar tile (no host copy).

    :param name: The in-kernel array name.
    :param index_exprs: Comma-joined per-dim index expressions.
    :returns: The ``ct.load(...).item()`` expression producing a scalar tile.
    """
    return f"ct.load({name}, ({index_exprs},), shape=()).item()"


def _is_device_staged_scalar(desc: object) -> bool:
    """Whether *desc* is a numeric ``data.Scalar`` materialized as a 1-element
    DEVICE array at kernel entry — either a non-transient kernel parameter
    (launch-site normalization, see :func:`_is_device_scalar`) or a transient
    staged in ``GPU_Global`` storage by the data-copy insertion. Both need the
    0-d ``ct.load`` binding; a transient in Register/Default storage is a plain
    in-kernel Python variable and must be bound by rename instead.

    :param desc: A data descriptor (or ``None``).
    :returns: ``True`` when the in-kernel binding must be a 0-d tile load.
    """
    if not _is_device_scalar(desc):
        return False
    return (not desc.transient) or desc.storage == dtypes.StorageType.GPU_Global


def _element_index_exprs(subset: object, allow_multi_element: bool = False) -> Optional[str]:
    """Comma-joined kernel-safe index expressions when ``subset`` addresses
    exactly one element; ``None`` when it spans more than one element (or size
    cannot be proven 1 symbolically).

    :param subset: A memlet range.
    :param allow_multi_element: Render the per-dim begin expressions even for
        a (possibly) multi-element ``Range`` instead of returning ``None``.
    :returns: ``"i, j"``-style index string, or ``None``.
    """
    if not allow_multi_element:
        try:
            if any(sp.simplify(sp.sympify(r[1] - r[0])) != 0 for r in subset):
                return None
        except Exception:  # noqa: BLE001 -- unprovable extent: treat as non-element.
            return None
    return ", ".join(symstr(r[0]) for r in subset)


def _matching_inner_connector(outer_conn: str) -> str:
    """Convert an outer (input) scope connector name to the matching inner (output) name.

    :param outer_conn: The outer connector name, e.g. ``"IN_A"``.
    :returns: The matching inner connector, e.g. ``"OUT_A"``.
    :raises ValueError: If *outer_conn* does not start with the expected prefix.
    """
    if not outer_conn.startswith(_SCOPE_IN_PREFIX):
        raise ValueError(f"Expected connector starting with {_SCOPE_IN_PREFIX!r}, got {outer_conn!r}")
    return _SCOPE_OUT_PREFIX + outer_conn[len(_SCOPE_IN_PREFIX):]


def _matching_outer_connector(inner_conn: str) -> str:
    """Convert an inner (output) scope connector name to the matching outer (input) name.

    :param inner_conn: The inner connector name, e.g. ``"OUT_A"``.
    :returns: The matching outer connector, e.g. ``"IN_A"``.
    :raises ValueError: If *inner_conn* does not start with the expected prefix.
    """
    if not inner_conn.startswith(_SCOPE_OUT_PREFIX):
        raise ValueError(f"Expected connector starting with {_SCOPE_OUT_PREFIX!r}, got {inner_conn!r}")
    return _SCOPE_IN_PREFIX + inner_conn[len(_SCOPE_OUT_PREFIX):]


def _is_scalar_buffer_name(name: str, desc: data.Data) -> bool:
    """Whether ``name`` is bound as a 0-d numpy scalar buffer at runtime.

    Mirrors ``PythonCodeGen._is_scalar_buffer``: non-transient Scalar
    arguments are marshalled as 0-d numpy arrays; transient Scalars are
    plain Python values.

    :param name: The data name.
    :param desc: The data descriptor.
    :returns: True for non-transient, non-member Scalar descriptors.
    """
    return isinstance(desc, data.Scalar) and not desc.transient and '.' not in name


def _array_runtime_name(sdfg: "SDFG", name: str) -> str:
    """Return the runtime variable name for a data array.

    Handles global/persistent/external arrays that live in ``globals()``.

    :param sdfg: The SDFG containing the array.
    :param name: The array name (possibly dotted).
    :returns: The runtime variable expression.
    """
    root_name, sep, suffix = name.partition(".")
    desc = sdfg.arrays.get(root_name)
    if desc is None:
        return name
    if desc.lifetime in (
            dtypes.AllocationLifetime.Global,
            dtypes.AllocationLifetime.Persistent,
            dtypes.AllocationLifetime.External,
    ):
        base = f"globals()[{root_name!r}]"
        return f"{base}.{suffix}" if sep else base
    return name


def _grid_exprs_from_map_entry(entry: nodes.MapEntry) -> List[str]:
    """Return per-dimension grid size (number of tiles) expressions from a map entry.

    Each grid dimension is the number of tiles ``ceil(extent / step)`` where
    ``extent = end - start + 1`` (the map range end is inclusive). This is
    emitted as a structural integer ceil-division ``int_ceil(extent, step)``
    rather than ``symstr(range.size())``.

    The latter is unsound for the Python/cuTile backend: ``symstr`` rewrites a
    symbolic ceiling using C integer-division semantics (e.g.
    ``ceiling((N-2)/8)`` becomes ``int_ceil(int_floor(N, 8) - 1/4, 1)``). Under
    Python's true division the residual rational ``1/4`` is the float ``0.25``,
    which both yields a non-integer grid dimension (rejected by ``ct.launch``)
    and is off-by-one for non-divisible extents. Building ``int_ceil`` directly
    from ``(start, end, step)`` keeps both operands integral.

    :param entry: The map entry node.
    :returns: List of grid-dimension expression strings (one per map dimension).
    """
    # Delegate to the single shared implementation so the map-entry grid and the
    # tile-op ``cutile`` expansions cannot drift apart (divergent grid strings
    # would silently desynchronize the folded launch-grid PIDs).
    from dace.libraries.tileops._pure_codegen import cutile_grid_size_exprs
    return cutile_grid_size_exprs(entry)


def _fold_grid_to_launch(grid_exprs: List[str]) -> Tuple[List[str], List[str]]:
    """Fold a ``K``-dimensional tile grid onto the cuTile launch grid (rank <= 3).

    The ``cuda.tile`` runtime caps the launch grid at three axes (``Dim3`` in
    ``ct.launch``; ``ct.bid(axis)`` only accepts ``axis in {0, 1, 2}``). When a
    tiled map nest has more than three dimensions (e.g. 4-D ``softmax``, 5-D
    ``conv2d``), the extra dimensions are linearized onto the available axes and
    recovered inside the kernel via integer div/mod.

    The fold layout is the single canonical contract defined in
    :mod:`dace.libraries.tileops._pure_codegen` (``cutile_launch_grid_dims`` /
    ``cutile_bid_expr``), shared with every tile-op ``cutile`` expansion so the
    map-entry block IDs and the tile-op block IDs agree: the two innermost map
    dimensions map to grid axes 1 and 2, and the leading ``K-2`` dimensions are
    folded row-major onto grid axis 0. For ``K <= 3`` the identity mapping is
    used (``__pid{d} = ct.bid(d)``), unchanged from the pre-folding behavior.

    :param grid_exprs: Per-dimension grid-size (tile-count) expression strings,
        in map order (dimension 0 is outermost). Length is ``K``.
    :returns: A tuple ``(launch_dims, pid_stmts)`` where ``launch_dims`` is the
        list of at most three launch-grid dimension expressions passed to
        ``ct.launch``, and ``pid_stmts`` is the list of Python statement strings
        that bind ``__pid{d}`` for every map dimension ``d`` inside the kernel.
    """
    from dace.libraries.tileops._pure_codegen import cutile_bid_expr, cutile_launch_grid_dims
    num_dims = len(grid_exprs)
    launch_dims = cutile_launch_grid_dims(grid_exprs)
    pid_stmts = [f"__pid{d} = {cutile_bid_expr(d, num_dims, grid_exprs)}" for d in range(num_dims)]
    return launch_dims, pid_stmts


def _map_index_exprs(entry: nodes.MapEntry) -> List[str]:
    """Return per-dimension element-coordinate expressions.

    Each expression computes the global element index from the PID
    (block ID) and the map's start/step parameters.

    :param entry: The map entry node.
    :returns: List of index expression strings.
    """
    result: List[str] = []
    for d, (start, _, step) in enumerate(entry.map.range):
        pid = f"__pid{d}"
        start_s = symstr(start)
        step_s = symstr(step)
        if start_s == "0" and step_s == "1":
            result.append(pid)
        elif step_s == "1":
            result.append(f"({start_s} + {pid})")
        else:
            result.append(f"({start_s} + {pid} * {step_s})")
    return result


def _ordered_unique(items: Iterable[str]) -> List[str]:
    """Deduplicate items while preserving insertion order, then sort.

    :param items: Iterable of strings.
    :returns: Sorted list of unique strings.
    """
    return sorted(set(items))


def _collect_free_symbols(entry: nodes.MapEntry, dfg_scope: object, sdfg: "SDFG",
                          symbol_types: Optional[Dict[str, dtypes.typeclass]] = None) -> List[str]:
    """Collect free symbols used in a map scope.

    Only names visible at the launch site can be kernel arguments; names
    defined in other states or sibling scopes are excluded.

    :param entry: The map entry node.
    :param dfg_scope: The scope subgraph view.
    :param sdfg: The SDFG.
    :param symbol_types: Types visible at the entry, or None to resolve them.
    :returns: Sorted list of free symbol names.
    """
    syms = {str(s) for s in entry.map.range.free_symbols}
    for edge in dfg_scope.edges():
        memlet = edge.data
        if memlet is None:
            continue
        syms |= {str(s) for s in memlet.free_symbols}
    # Also collect symbols referenced by NestedSDFG symbol_mapping values,
    # so that symbols like ``cond_val`` that are forwarded into a conditional
    # NestedSDFG become kernel parameters.
    for scope_node in dfg_scope.nodes():
        if isinstance(scope_node, nodes.NestedSDFG):
            for expr in scope_node.symbol_mapping.values():
                if hasattr(expr, 'free_symbols'):
                    syms |= {str(s) for s in expr.free_symbols}
                else:
                    syms.add(str(expr))
        elif isinstance(scope_node, nodes.Tasklet):
            # Symbols referenced only in tasklet code (e.g. a TileBinop
            # Symbol-operand expansion emitting ``_a / N``) appear in no
            # memlet; without this they are undefined inside the kernel.
            syms |= scope_node.free_symbols
    # Names bound *inside* the kernel scope (the map's own parameters and the
    # parameters of any nested maps) are never launch arguments: they are
    # defined by the emitted ``__pid``/index bindings.  Without this, a map
    # param that happens to collide with a host loop variable or an
    # interstate-assignment key would pass the ``runtime_defined`` filter
    # below and become an (undefined at the call site) launch argument.
    bound_names = {str(p) for p in entry.map.params}
    for scope_node in dfg_scope.nodes():
        if isinstance(scope_node, nodes.MapEntry):
            bound_names |= {str(p) for p in scope_node.map.params}
    syms -= bound_names
    if symbol_types is None:
        symbol_types = SymbolResolver().defined_at(dfg_scope.graph, entry)
    syms = {s for s in syms if s in symbol_types and s not in sdfg.constants and s not in sdfg.arrays}
    return sorted(syms)


def _enclosing_cutile_entry(state: "SDFGState", node: nodes.Node) -> Optional[nodes.MapEntry]:
    """Find the nearest enclosing CuTile-scheduled MapEntry for a node.

    :param state: The SDFG state.
    :param node: The node to check.
    :returns: The enclosing CuTile MapEntry, or ``None`` if not inside one.
    """
    scope = state.scope_dict()
    cur = scope.get(node)
    while cur is not None:
        if isinstance(cur, nodes.MapEntry) and cur.map.schedule == dtypes.ScheduleType.CuTile:
            return cur
        cur = scope.get(cur)
    return None


def _is_cutile_node(state: "SDFGState", node: nodes.Node) -> bool:
    """Check whether a node belongs to a CuTile scope (entry, exit, or inside).

    :param state: The SDFG state.
    :param node: The node to check.
    :returns: ``True`` if the node is part of a CuTile scope.
    """
    if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.CuTile:
        return True
    if isinstance(node, nodes.MapExit):
        entry = state.entry_node(node)
        if (entry is not None and isinstance(entry, nodes.MapEntry)
                and entry.map.schedule == dtypes.ScheduleType.CuTile):
            return True
    return _enclosing_cutile_entry(state, node) is not None


# ---------------------------------------------------------------------------
# Code generator class
# ---------------------------------------------------------------------------


@registry.autoregister_params(name="cutile_python")
class CuTilePythonCodeGen(PythonTargetCodeGenerator):
    """Python target for CuTile-scheduled map scopes.

    Uses an AccessNode-centric design where each ``CuTile_Tile``
    AccessNode handles its own ``ct.load`` / ``ct.store`` operations,
    rather than centralizing loads at MapEntry and stores at MapExit.
    """

    title = "CuTilePython"
    target_name = "cutile_python"
    language = "python"

    def __init__(self, frame_codegen: "DaCePythonCodeGenerator", sdfg: "SDFG") -> None:
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        #: Tracks already-generated nested functions by position key to avoid duplicates.
        self._generated_nested_functions: Dict[str, str] = {}
        # Register as the handler for CuTile map scopes.
        self._dispatcher.register_map_dispatcher(dtypes.ScheduleType.CuTile, self)
        # Register as node handler for all nodes inside CuTile scopes.
        self._dispatcher.register_node_dispatcher(self, predicate=self._is_in_cutile_scope)
        # Register copy dispatchers for transfers involving CuTile_Tile storage.
        # The actual loads/stores are emitted by _generate_AccessNode, not by
        # copy_memory, but the dispatcher's target-collection phase needs to
        # find a handler for every memlet-tree edge pair.  Register for all
        # storage types that can appear in SDFG edges touching tile transients.
        _tile_peer_storages = [
            dtypes.StorageType.GPU_Global,
            dtypes.StorageType.Register,
        ]
        _copy_schedules = [dtypes.ScheduleType.CuTile, None]
        for peer_storage in _tile_peer_storages:
            for sched in _copy_schedules:
                self._dispatcher.register_copy_dispatcher(peer_storage, dtypes.StorageType.CuTile_Tile, sched, self)
                self._dispatcher.register_copy_dispatcher(dtypes.StorageType.CuTile_Tile, peer_storage, sched, self)
        # Also register tile-to-tile copies within a CuTile scope.
        for sched in _copy_schedules:
            self._dispatcher.register_copy_dispatcher(dtypes.StorageType.CuTile_Tile, dtypes.StorageType.CuTile_Tile,
                                                      sched, self)
        # Register for cross-storage copies between CPU_Heap and GPU_Global.
        # These arise from apply_gpu_transformations() copy-in/copy-out states
        # that transfer data between host and device outside any map scope.
        self._dispatcher.register_copy_dispatcher(dtypes.StorageType.CPU_Heap, dtypes.StorageType.GPU_Global, None,
                                                  self)
        self._dispatcher.register_copy_dispatcher(dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Heap, None,
                                                  self)
        # Register array dispatcher for CuTile_Tile storage (allocation is a no-op).
        self._dispatcher.register_array_dispatcher(dtypes.StorageType.CuTile_Tile, self)

    def get_generated_codeobjects(self) -> list:
        """Return generated code objects (none for this target).

        :returns: Empty list.
        """
        return []

    def get_includes(self) -> Dict[str, List[str]]:
        """Return import statements needed for cuTile kernels.

        :returns: Mapping from code section to list of import lines.
        """
        return {"frame": ["import cuda.tile as ct", "import cupy"]}

    def preprocess(self, sdfg: "SDFG") -> None:
        """Preprocessing hook (no-op for cuTile).

        :param sdfg: The SDFG to preprocess.
        """
        pass

    @property
    def has_initializer(self) -> bool:
        """Whether this target has an initialization function."""
        return False

    @property
    def has_finalizer(self) -> bool:
        """Whether this target has a finalization function."""
        return False

    # ------------------------------------------------------------------
    # Dispatcher predicates
    # ------------------------------------------------------------------

    def _sdfg_is_cutile_body(self, sdfg: "SDFG") -> bool:
        """Whether *sdfg* is a NestedSDFG body emitted as a cuTile function.

        A NestedSDFG body has no CuTile MapEntry in its own states, so this is
        determined structurally by walking the nested-SDFG parent chain: the
        body is a cuTile function iff the NestedSDFG node that owns it (at any
        level) lives inside a CuTile-scheduled map.

        :param sdfg: The (possibly nested) SDFG to check.
        :returns: ``True`` if *sdfg* is a cuTile NestedSDFG body.
        """
        cur = sdfg
        while cur.parent_nsdfg_node is not None:
            if _is_cutile_node(cur.parent, cur.parent_nsdfg_node):
                return True
            cur = cur.parent_sdfg
        return False

    def _is_in_cutile_scope(self, sdfg: "SDFG", state: "SDFGState", node: nodes.Node) -> bool:
        """Predicate for the node dispatcher: return True for CuTile nodes.

        A node is in a cuTile scope if it lives (transitively) inside a
        CuTile-scheduled map, or if it belongs to a NestedSDFG body emitted as
        a cuTile function (see :meth:`_sdfg_is_cutile_body`).

        :param sdfg: The SDFG.
        :param state: The SDFG state.
        :param node: The node to check.
        :returns: ``True`` if the node belongs to a CuTile scope.
        """
        return _is_cutile_node(state, node) or self._sdfg_is_cutile_body(sdfg)

    def _in_cutile_context(self, sdfg: "SDFG", state: "SDFGState", node: nodes.Node) -> bool:
        """Whether *node* is inside a cuTile scope, counting NestedSDFG bodies.

        :param sdfg: The SDFG containing the node.
        :param state: The state containing the node.
        :param node: The node to check.
        :returns: ``True`` if generating cuTile code is valid for this node.
        """
        return (_enclosing_cutile_entry(state, node) is not None or self._sdfg_is_cutile_body(sdfg))

    # ------------------------------------------------------------------
    # Array dispatcher methods (no-ops for tile arrays)
    # ------------------------------------------------------------------

    def allocate_array(self, sdfg: "SDFG", cfg: object, dfg: object, state_id: int, node: nodes.Node, nodedesc: object,
                       global_stream: PythonCodeIOStream, declaration_stream: PythonCodeIOStream,
                       allocation_stream: PythonCodeIOStream) -> None:
        """Tile arrays inside cuTile kernels are Python locals -- no allocation needed.

        :param sdfg: The SDFG.
        :param cfg: The control flow graph.
        :param dfg: The dataflow graph.
        :param state_id: The state ID.
        :param node: The access node.
        :param nodedesc: The data descriptor.
        :param global_stream: Stream for global code.
        :param declaration_stream: Stream for declarations.
        :param allocation_stream: Stream for allocations.
        """
        pass

    def deallocate_array(self, sdfg: "SDFG", cfg: object, dfg: object, state_id: int, node: nodes.Node,
                         nodedesc: object, function_stream: PythonCodeIOStream,
                         callsite_stream: PythonCodeIOStream) -> None:
        """Tile arrays inside cuTile kernels are Python locals -- no deallocation needed.

        :param sdfg: The SDFG.
        :param cfg: The control flow graph.
        :param dfg: The dataflow graph.
        :param state_id: The state ID.
        :param node: The access node.
        :param nodedesc: The data descriptor.
        :param function_stream: Stream for function code.
        :param callsite_stream: Stream for call-site code.
        """
        pass

    def declare_array(self, sdfg: "SDFG", cfg: object, dfg: object, state_id: int, node: nodes.Node, nodedesc: object,
                      global_stream: PythonCodeIOStream, declaration_stream: PythonCodeIOStream) -> None:
        """Tile arrays are not declared -- they are created by ct.load or tasklet output.

        :param sdfg: The SDFG.
        :param cfg: The control flow graph.
        :param dfg: The dataflow graph.
        :param state_id: The state ID.
        :param node: The access node.
        :param nodedesc: The data descriptor.
        :param global_stream: Stream for global code.
        :param declaration_stream: Stream for declarations.
        """
        pass

    # ------------------------------------------------------------------
    # Copy dispatcher method
    # ------------------------------------------------------------------

    def copy_memory(self, sdfg: "SDFG", cfg: object, dfg: object, state_id: int, src_node: nodes.Node,
                    dst_node: nodes.Node, edge: object, function_stream: PythonCodeIOStream,
                    callsite_stream: PythonCodeIOStream) -> None:
        """Handle copy operations between arrays.

        Supports three categories of copies:

        1. **Cross-storage CPU_Heap/Default <-> GPU_Global** (from
           ``apply_gpu_transformations()`` copy-in/copy-out states): emits
           ``.set()`` (host-to-device) or ``.get(out=...)``
           (device-to-host) transfers.  ``StorageType.Default`` is
           treated as host-side since it resolves to ``CPU_Heap``.
        2. **CuTile_Tile <-> other storage** inside a scope: data flows
           through MapEntry/MapExit and is handled by
           :meth:`_generate_AccessNode`; this path is a fallback for
           direct AccessNode-to-AccessNode edges routed here.
        3. **Same-storage copies**: emits plain numpy-compatible
           assignment.

        :param sdfg: The SDFG.
        :param cfg: The control flow graph.
        :param dfg: The dataflow graph.
        :param state_id: The state ID.
        :param src_node: The source node.
        :param dst_node: The destination node.
        :param edge: The connecting edge.
        :param function_stream: Stream for function code.
        :param callsite_stream: Stream for call-site code.
        :raises NotImplementedError: If the copy is not between two
            AccessNodes.
        """
        memlet = edge.data
        if not isinstance(src_node, nodes.AccessNode) or not isinstance(dst_node, nodes.AccessNode):
            raise NotImplementedError(f"CuTile copy_memory only supports AccessNode-to-AccessNode "
                                      f"copies, got {type(src_node).__name__} -> "
                                      f"{type(dst_node).__name__}")

        src_storage = sdfg.arrays[src_node.data].storage
        dst_storage = sdfg.arrays[dst_node.data].storage

        # --- Cross-storage CPU <-> GPU transfers ---
        # Register counts as host-side: Register transients (e.g. staged
        # scalars) live in host memory in the Python backend.
        _HOST_STORAGES = (dtypes.StorageType.CPU_Heap, dtypes.StorageType.Default, dtypes.StorageType.Register)
        src_on_host = src_storage in _HOST_STORAGES
        dst_on_host = dst_storage in _HOST_STORAGES
        src_on_gpu = src_storage == dtypes.StorageType.GPU_Global
        dst_on_gpu = dst_storage == dtypes.StorageType.GPU_Global
        is_cpu_to_gpu = src_on_host and dst_on_gpu
        is_gpu_to_cpu = src_on_gpu and dst_on_host

        if is_cpu_to_gpu or is_gpu_to_cpu:
            self._emit_cross_storage_copy(sdfg, cfg, state_id, src_node, dst_node, memlet, is_cpu_to_gpu,
                                          callsite_stream)
            return

        # --- Same-storage / CuTile_Tile fallback copies ---
        # Build source expression
        src_expr = src_node.data
        if memlet.src_subset is not None:
            src_subset_str = self._subset_to_python(memlet.src_subset)
            if src_subset_str:
                src_expr = f"{src_node.data}[{src_subset_str}]"

        # Build destination expression
        dst_expr = dst_node.data
        if memlet.dst_subset is not None:
            dst_subset_str = self._subset_to_python(memlet.dst_subset)
            if dst_subset_str:
                dst_expr = f"{dst_node.data}[{dst_subset_str}]"

        # Emit assignment (works for both numpy and cupy)
        callsite_stream.write(f"{dst_expr} = {src_expr}", cfg, state_id)

    def _emit_cross_storage_copy(
        self,
        sdfg: "SDFG",
        cfg: object,
        state_id: int,
        src_node: nodes.AccessNode,
        dst_node: nodes.AccessNode,
        memlet: object,
        cpu_to_gpu: bool,
        callsite_stream: PythonCodeIOStream,
    ) -> None:
        """Emit a cross-storage copy between CPU_Heap and GPU_Global arrays.

        For CPU_Heap -> GPU_Global (copy-in), emits::

            dst.set(src)

        For GPU_Global -> CPU_Heap (copy-out), emits::

            src.get(out=dst)

        When the memlet carries subsets, the subset is applied to both
        source and destination expressions.  For full-array copies
        (typical of ``apply_gpu_transformations()``), the ``[:]`` ensures
        the data is copied into the pre-allocated array.

        Scalar endpoints get dedicated forms, since host-side Scalars are
        Python values or 0-d numpy buffers to which the ``.set()`` /
        ``.get(out=...)`` array APIs do not apply:

        * GPU element -> host Scalar: ``dst[...] = src_elem.item()`` for a
          0-d scalar buffer (non-transient argument), or
          ``dst = src_elem.item()`` for a plain-value transient Scalar.
        * Host Scalar -> GPU element: broadcast assignment
          ``dst[subset or ...] = value`` where ``value`` is ``src.item()``
          for a 0-d scalar buffer or the plain name otherwise (``.set()``
          requires an ndarray source).

        :param sdfg: The SDFG.
        :param cfg: The control flow graph.
        :param state_id: The state ID.
        :param src_node: The source AccessNode.
        :param dst_node: The destination AccessNode.
        :param memlet: The memlet on the connecting edge.
        :param cpu_to_gpu: ``True`` for CPU_Heap -> GPU_Global,
            ``False`` for GPU_Global -> CPU_Heap.
        :param callsite_stream: Stream for call-site code.
        """
        src_desc = sdfg.arrays[src_node.data]
        dst_desc = sdfg.arrays[dst_node.data]

        # Build source expression (with optional subset).
        src_expr = src_node.data
        if not isinstance(src_desc, data.Scalar) and memlet.src_subset is not None:
            src_subset_str = self._subset_to_python(memlet.src_subset)
            if src_subset_str:
                src_expr = f"{src_node.data}[{src_subset_str}]"

        # Host-side Scalars are Python values or 0-d buffers, not arrays:
        # ``.set()``/``.get(out=...)`` do not apply to them.
        if not cpu_to_gpu and isinstance(dst_desc, data.Scalar):
            # GPU element -> host scalar. ``.item()`` transfers and unwraps.
            value_expr = f"{src_expr}.item()"
            if _is_scalar_buffer_name(dst_node.data, dst_desc):
                callsite_stream.write(f"{dst_node.data}[...] = {value_expr}", cfg, state_id)
            else:
                callsite_stream.write(f"{dst_node.data} = {value_expr}", cfg, state_id)
            return
        if cpu_to_gpu and isinstance(src_desc, data.Scalar):
            # Host scalar -> GPU element: broadcast assignment (``.set()``
            # requires an array source).
            if _is_scalar_buffer_name(src_node.data, src_desc):
                value_expr = f"{src_node.data}.item()"
            else:
                value_expr = src_node.data
            dst_subset_str = self._subset_to_python(memlet.dst_subset) if memlet.dst_subset is not None else ""
            dst_expr = f"{dst_node.data}[{dst_subset_str or '...'}]"
            callsite_stream.write(f"{dst_expr} = {value_expr}", cfg, state_id)
            return

        # Build destination LHS (with optional subset, or [:] for full copy).
        if memlet.dst_subset is not None:
            dst_subset_str = self._subset_to_python(memlet.dst_subset)
            if dst_subset_str:
                dst_lhs = f"{dst_node.data}[{dst_subset_str}]"
            else:
                dst_lhs = f"{dst_node.data}"
        else:
            dst_lhs = f"{dst_node.data}"

        if cpu_to_gpu:
            # Scalar sources returned above; only array sources reach here.
            callsite_stream.write(f"{dst_lhs}.set({src_expr})", cfg, state_id)
        else:
            callsite_stream.write(f"{src_expr}.get(out={dst_lhs})", cfg, state_id)

    @staticmethod
    def _subset_to_python(subset: "subsets.Subset") -> str:
        """Convert a subset to a Python indexing string.

        :param subset: The subset to convert.
        :returns: Python-style indexing string (e.g., ``"0:16, 0:8"``),
            or an empty string if *subset* is ``None``.
        """
        if subset is None:
            return ""
        if isinstance(subset, subsets.Range):
            parts: List[str] = []
            for start, end, step in subset:
                start_s = symstr(start) if start != 0 else ""
                end_s = symstr(end + 1)
                step_s = f":{symstr(step)}" if step != 1 else ""
                parts.append(f"{start_s}:{end_s}{step_s}")
            return ", ".join(parts)
        return str(subset)

    # ------------------------------------------------------------------
    # Per-tile alignment check
    # ------------------------------------------------------------------

    @staticmethod
    def _needs_gather_for_tile(entry: nodes.MapEntry, tile_shape: Tuple[int, ...], sdfg: "SDFG") -> bool:
        """Check if gather/scatter is needed for a specific tile.

        Returns ``True`` if the outer map's start is non-zero or the
        outer map's step doesn't match the given tile shape in any
        dimension.

        :param entry: The outer :class:`~dace.sdfg.nodes.MapEntry`.
        :param tile_shape: Shape of the specific tile being loaded/stored.
        :param sdfg: The SDFG (for symbol resolution).
        :returns: ``True`` if gather/scatter is needed for this tile.
        """
        for d, (start, _, step) in enumerate(entry.map.range):
            start_val = sp.sympify(start)
            step_val = sp.sympify(step)
            if start_val != 0:
                return True
            if d < len(tile_shape):
                tile_dim = tile_shape[d]
                if sp.sympify(step_val) != sp.sympify(tile_dim):
                    return True
        return False

    # ------------------------------------------------------------------
    # Single-tile shape resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_single_tile_shape(entry: nodes.MapEntry, tile_name: str, sdfg: "SDFG") -> Tuple[int, ...]:
        """Resolve the shape of a single tile transient to concrete integers.

        Reads the shape from the tile's array descriptor and substitutes
        declared constants. Runtime symbols cannot determine a tile shape.

        :param entry: The enclosing :class:`~dace.sdfg.nodes.MapEntry`.
        :param tile_name: Name of the tile transient in the SDFG.
        :param sdfg: The SDFG.
        :returns: Tuple of resolved integer dimensions.
        :raises RuntimeError: If the tile is not found or shape cannot be resolved.
        """
        desc = sdfg.arrays.get(tile_name)
        if desc is None:
            raise RuntimeError(f"Tile array {tile_name!r} not found in SDFG.")

        constants = {name: value for name, value in sdfg.constants.items() if name not in sdfg.arrays}
        resolved: List[int] = []
        for dimension in desc.shape:
            value = symbolic.simplify(symbolic.replace_symbols(sp.sympify(dimension), constants))
            if value.free_symbols or value.is_integer is not True:
                raise ValueError(f'cuTile array {tile_name!r} needs compile-time integer dimensions, got {dimension}')
            resolved.append(int(value))
        return tuple(resolved)

    # ------------------------------------------------------------------
    # Shape validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_tile_shape(tile_shape: Tuple[int, ...], entry: nodes.MapEntry, tile_name: str, sdfg: "SDFG") -> None:
        """Validate that the resolved tile shape is consistent with the map.

        Checks:

        1. All tile dimensions are positive integers.
        2. The tile does not have more dimensions than the enclosing map.
        3. For the aligned path (``ct.load``), warns if any tile dimension
           does not match the corresponding map step (since such cases
           require ``ct.gather`` instead).

        :param tile_shape: The resolved tile shape.
        :param entry: The enclosing MapEntry.
        :param tile_name: Name of the tile transient.
        :param sdfg: The SDFG.
        :raises RuntimeError: If the tile shape is invalid.
        """
        # 1. All dimensions must be positive.
        for d, dim in enumerate(tile_shape):
            if dim <= 0:
                raise RuntimeError(f"Tile {tile_name!r} has non-positive dimension {dim} "
                                   f"at axis {d}. All tile dimensions must be positive.")

        # 2. Tile dimensionality must not exceed map dimensionality.
        map_ndim = len(entry.map.range)
        if len(tile_shape) > map_ndim:
            raise RuntimeError(f"Tile {tile_name!r} has {len(tile_shape)} dimensions but "
                               f"the enclosing map has only {map_ndim} dimensions.")

        # 3. Advisory: warn when tile dims don't match map steps.
        for d, (_, _, step) in enumerate(entry.map.range):
            if d >= len(tile_shape):
                break
            step_val = int(sp.sympify(step))
            if tile_shape[d] != step_val:
                warnings.warn(
                    f"Tile {tile_name!r} dimension {d} has size "
                    f"{tile_shape[d]} but the enclosing map step is "
                    f"{step_val}. The gather/scatter path will be used.",
                    stacklevel=2)

    # ------------------------------------------------------------------
    # Trace load/store sources and targets through scope boundaries
    # ------------------------------------------------------------------

    @staticmethod
    def _trace_load_source_edge(state: "SDFGState", entry: nodes.MapEntry, in_edge: object) -> Optional[object]:
        """Trace through a MapEntry to the outer edge feeding a scope input.

        Given an edge MapEntry -> (inner node), finds the corresponding outer
        edge AccessNode(global) -> MapEntry (which carries the source memlet,
        including its subset).

        :param state: The SDFG state.
        :param entry: The MapEntry node.
        :param in_edge: The edge from MapEntry to the inner node.
        :returns: The outer edge, or ``None`` if not found.
        """
        src_conn = in_edge.src_conn
        if src_conn is None or not src_conn.startswith(_SCOPE_OUT_PREFIX):
            return None
        outer_conn = _matching_outer_connector(src_conn)
        for outer_edge in state.in_edges_by_connector(entry, outer_conn):
            if isinstance(outer_edge.src, nodes.AccessNode):
                return outer_edge
        return None

    @staticmethod
    def _trace_load_source(state: "SDFGState", entry: nodes.MapEntry, tile_node: nodes.AccessNode,
                           in_edge: object) -> Optional[str]:
        """Trace through a MapEntry to find the global array feeding a tile.

        Given an edge MapEntry -> AccessNode(tile), finds the corresponding
        outer edge AccessNode(global) -> MapEntry and returns the global
        array name.

        :param state: The SDFG state.
        :param entry: The MapEntry node.
        :param tile_node: The tile AccessNode inside the scope.
        :param in_edge: The edge from MapEntry to tile_node.
        :returns: The global array name, or ``None`` if not found.
        """
        outer_edge = CuTilePythonCodeGen._trace_load_source_edge(state, entry, in_edge)
        if outer_edge is None:
            return None
        return outer_edge.data.data if outer_edge.data else outer_edge.src.data

    @staticmethod
    def _trace_store_target(state: "SDFGState", exit_node: nodes.MapExit, tile_node: nodes.AccessNode,
                            out_edge: object) -> Optional[str]:
        """Trace through a MapExit to find the global array receiving a tile.

        Given an edge AccessNode(tile) -> MapExit, finds the corresponding
        outer edge MapExit -> AccessNode(global) and returns the global
        array name.

        :param state: The SDFG state.
        :param exit_node: The MapExit node.
        :param tile_node: The tile AccessNode inside the scope.
        :param out_edge: The edge from tile_node to MapExit.
        :returns: The global array name, or ``None`` if not found.
        """
        dst_conn = out_edge.dst_conn
        if dst_conn is None or not dst_conn.startswith(_SCOPE_IN_PREFIX):
            return None
        outer_conn = _matching_inner_connector(dst_conn)
        for outer_edge in state.out_edges_by_connector(exit_node, outer_conn):
            if isinstance(outer_edge.dst, nodes.AccessNode):
                return outer_edge.data.data if outer_edge.data else outer_edge.dst.data
        return None

    @staticmethod
    def _load_source_begins(state: "SDFGState", entry: nodes.MapEntry, in_edge: object) -> Optional[List[str]]:
        """Per-dim begin expressions of the outer memlet feeding a tile load.

        :param state: The SDFG state.
        :param entry: The MapEntry node.
        :param in_edge: The edge from MapEntry to the tile AccessNode.
        :returns: Per-dim begin expression strings of the outer (global-array)
            memlet, or ``None`` if not resolvable.
        """
        src_conn = in_edge.src_conn
        if src_conn is None or not src_conn.startswith(_SCOPE_OUT_PREFIX):
            return None
        outer_conn = _matching_outer_connector(src_conn)
        for outer_edge in state.in_edges_by_connector(entry, outer_conn):
            if (isinstance(outer_edge.src, nodes.AccessNode) and outer_edge.data is not None
                    and outer_edge.data.subset is not None):
                return [symstr(r[0]) for r in outer_edge.data.subset.ranges]
        return None

    @staticmethod
    def _store_target_begins(state: "SDFGState", exit_node: nodes.MapExit, out_edge: object) -> Optional[List[str]]:
        """Per-dim begin expressions of the outer memlet receiving a tile store.

        :param state: The SDFG state.
        :param exit_node: The MapExit node.
        :param out_edge: The edge from the tile AccessNode to the MapExit.
        :returns: Per-dim begin expression strings of the outer (global-array)
            memlet, or ``None`` if not resolvable.
        """
        dst_conn = out_edge.dst_conn
        if dst_conn is None or not dst_conn.startswith(_SCOPE_IN_PREFIX):
            return None
        outer_conn = _matching_inner_connector(dst_conn)
        for outer_edge in state.out_edges_by_connector(exit_node, outer_conn):
            if (isinstance(outer_edge.dst, nodes.AccessNode) and outer_edge.data is not None
                    and outer_edge.data.subset is not None):
                return [symstr(r[0]) for r in outer_edge.data.subset.ranges]
        return None

    @staticmethod
    def _const_begin_offsets(begins: Optional[List[str]], entry: nodes.MapEntry) -> List[object]:
        """Constant element offset per dim carried by the outer memlet begin.

        The outer memlet begin has the form ``<iter-var> + c`` (e.g. an offset
        slice ``B[1:-1]`` yields begin ``tile_i + 1``). Substituting every map
        iteration variable with ``0`` leaves the constant offset ``c`` that the
        block-id / map-range index reconstruction drops -- it must be added back
        to the ``ct.load`` / ``ct.gather`` / ``ct.scatter`` element index.

        :param begins: Per-dim begin expression strings (or ``None``).
        :param entry: The enclosing MapEntry (for its iteration variables).
        :returns: Per-dim symbolic offsets (``0`` where the begin is unusable or
            anchored at the block-aligned start).
        """
        if begins is None:
            return []
        subs = {str(p): sp.Integer(0) for p in entry.map.params}
        offsets: List[object] = []
        for b in begins:
            try:
                offsets.append(symbolic.simplify(symbolic.replace_symbols(symbolic.pystr_to_symbolic(b), subs)))
            except Exception:  # noqa: BLE001 - non-symbolic begin -> assume anchored
                offsets.append(sp.Integer(0))
        return offsets

    @staticmethod
    def _apply_index_offsets(map_index_exprs: List[str], offsets: List[object]) -> List[str]:
        """Add the constant per-dim offsets to the per-dim map index expressions.

        :param map_index_exprs: Per-dim element-index expression strings.
        :param offsets: Per-dim constant offsets from :meth:`_const_begin_offsets`.
        :returns: Per-dim index expression strings with non-zero offsets folded in.
        """
        adjusted: List[str] = []
        for d, expr in enumerate(map_index_exprs):
            if d < len(offsets) and offsets[d] != 0:
                adjusted.append(f"(({expr}) + ({symstr(offsets[d])}))")
            else:
                adjusted.append(expr)
        return adjusted

    # ------------------------------------------------------------------
    # Gather / scatter emission helpers
    # ------------------------------------------------------------------

    def _emit_gather_load(self, callsite_stream: PythonCodeIOStream, arr: str, tile_var: str,
                          map_index_exprs: List[str], tile_shape: Tuple[int,
                                                                        ...], cfg: object, state_id: int) -> List[str]:
        """Emit ``ct.gather`` with computed index tiles for non-aligned loads.

        Generates per-dimension index tiles via ``ct.arange`` and
        ``ct.broadcast_to``, then calls ``ct.gather`` to load elements
        at arbitrary global positions.  This handles tiles that don't
        align with ``ct.load``'s implicit grid for examples strided maps
        or maps with non-zero start.

        :param callsite_stream: Code output stream.
        :param arr: Global array variable name.
        :param tile_var: Destination tile variable name.
        :param map_index_exprs: Per-dimension map variable expressions
            (e.g. ``["(2 + __pid0 * 32)"]``).
        :param tile_shape: Tile shape tuple (resolved to ints).
        :param cfg: The control flow graph.
        :param state_id: The state ID.
        :returns: List of index variable names (for reuse by scatter store).
        """
        ndim = len(tile_shape)
        idx_vars: List[str] = []

        for d in range(ndim):
            idx_var = f"__dace_ct_gidx_{tile_var}_{d}"
            callsite_stream.write(f"{idx_var} = {map_index_exprs[d]} + ct.arange({tile_shape[d]}, dtype=ct.int32)", cfg,
                                  state_id)
            idx_vars.append(idx_var)

        if ndim > 1:
            broadcast_vars: List[str] = []
            shape_str = ", ".join(str(s) for s in tile_shape)
            for d, idx_var in enumerate(idx_vars):
                reshape_dims = tuple(tile_shape[d] if i == d else 1 for i in range(ndim))
                reshape_str = ", ".join(str(x) for x in reshape_dims)
                bcast_var = f"{idx_var}_nd"
                callsite_stream.write(
                    f"{bcast_var} = ct.broadcast_to("
                    f"ct.reshape({idx_var}, ({reshape_str},)), "
                    f"({shape_str},))", cfg, state_id)
                broadcast_vars.append(bcast_var)
            indices_str = ", ".join(broadcast_vars)
        else:
            indices_str = idx_vars[0]
            broadcast_vars = idx_vars

        callsite_stream.write(f"{tile_var} = ct.gather({arr}, ({indices_str},), "
                              f"padding_value=0)", cfg, state_id)

        return broadcast_vars if ndim > 1 else idx_vars

    def _emit_scatter_store(self, callsite_stream: PythonCodeIOStream, arr: str, tile_expr: str,
                            gather_idx_vars: List[str], cfg: object, state_id: int) -> None:
        """Emit ``ct.scatter`` with precomputed index tiles.

        Reuses the index tile variables generated by
        :meth:`_emit_gather_load` to scatter tile elements back to
        the global array at the correct positions.

        :param callsite_stream: Code output stream.
        :param arr: Global array variable name.
        :param tile_expr: Tile expression to store.
        :param gather_idx_vars: Index variable names from gather load.
        :param cfg: The control flow graph.
        :param state_id: The state ID.
        """
        indices_str = ", ".join(gather_idx_vars)
        callsite_stream.write(f"ct.scatter({arr}, ({indices_str},), {tile_expr})", cfg, state_id)

    # ------------------------------------------------------------------
    # Node dispatch entry point
    # ------------------------------------------------------------------

    def generate_node(self, sdfg: "SDFG", cfg: object, dfg: object, state_id: int, node: nodes.Node,
                      function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        """Dispatch code generation for a single node inside a CuTile scope.

        :param sdfg: The SDFG.
        :param cfg: The control flow graph.
        :param dfg: The dataflow graph.
        :param state_id: The state ID.
        :param node: The node to generate code for.
        :param function_stream: Stream for function-level code.
        :param callsite_stream: Stream for call-site code.
        :raises NotImplementedError: If there is no handler for the node type.
        """
        method = getattr(self, f"_generate_{type(node).__name__}", None)
        if method is None:
            raise NotImplementedError(f"CuTile backend has no handler for {type(node).__name__}.")
        method(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)

    # ------------------------------------------------------------------
    # MapEntry: PIDs + map variable bindings only
    # ------------------------------------------------------------------

    def _generate_MapEntry(self, sdfg: "SDFG", cfg: object, dfg: object, state_id: int, node: nodes.MapEntry,
                           function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        """Emit block IDs and map variable bindings for a cuTile kernel.

        In the AccessNode-centric design, MapEntry only sets up the
        tile-coordinate PIDs and the element-coordinate map variables.
        Tile loads are handled by each AccessNode individually.

        :param sdfg: The SDFG.
        :param cfg: The control flow graph.
        :param dfg: The dataflow graph.
        :param state_id: The state ID.
        :param node: The MapEntry node.
        :param function_stream: Stream for function-level code.
        :param callsite_stream: Stream for call-site code.
        """
        map_index_exprs = _map_index_exprs(node)

        # Bind __pid{d} for every map dimension. When the grid rank exceeds the
        # cuTile launch-grid cap of 3, the extra dimensions are folded onto
        # axis 0 and recovered here via integer div/mod (see
        # ``_fold_grid_to_launch``); the launch site must fold identically.
        grid_exprs = _grid_exprs_from_map_entry(node)
        _, pid_stmts = _fold_grid_to_launch(grid_exprs)
        for stmt in pid_stmts:
            callsite_stream.write(stmt, cfg, state_id)
        for var, expr in zip(node.map.params, map_index_exprs):
            callsite_stream.write(f"{var} = {expr}", cfg, state_id)

    # ------------------------------------------------------------------
    # MapExit: no-op (stores handled at AccessNode)
    # ------------------------------------------------------------------

    def _generate_MapExit(self, sdfg: "SDFG", cfg: object, dfg: object, state_id: int, node: nodes.MapExit,
                          function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        """No-op -- tile stores are handled at each AccessNode.

        :param sdfg: The SDFG.
        :param cfg: The control flow graph.
        :param dfg: The dataflow graph.
        :param state_id: The state ID.
        :param node: The MapExit node.
        :param function_stream: Stream for function-level code.
        :param callsite_stream: Stream for call-site code.
        """
        pass

    # ------------------------------------------------------------------
    # AccessNode: THE CORE of the AccessNode-centric redesign
    # ------------------------------------------------------------------

    @staticmethod
    def _kernel_output_names(state: "SDFGState", entry: nodes.MapEntry) -> set:
        """Data names the kernel writes (out-edges of the map exit).

        :param state: The state containing the map scope.
        :param entry: The CuTile-scheduled MapEntry of the kernel scope.
        :returns: The set of output data names (empty for a scope without a
            matching exit node, e.g. partially constructed test graphs).
        """
        try:
            exit_node = state.exit_node(entry)
        except StopIteration:
            return set()
        return {e.data.data for e in state.out_edges(exit_node) if e.data is not None and e.data.data is not None}

    @staticmethod
    def _kernel_input_names(state: "SDFGState", entry: nodes.MapEntry) -> set:
        """Data names passed into the kernel (in-edges of the map entry) —
        exactly the ``input_arrays`` the launch site normalizes (a numeric
        Scalar param always becomes a 1-element device array; see
        ``generate_scope``).

        :param state: The state containing the map scope.
        :param entry: The CuTile-scheduled MapEntry of the kernel scope.
        :returns: The set of input data names.
        """
        return {
            e.data.data
            for e in state.in_edges(entry)
            if e.data is not None and e.data.data is not None and isinstance(e.src, nodes.AccessNode)
        }

    def _scalar_needs_tile_load(self, state: "SDFGState", entry: Optional[nodes.MapEntry], name: str,
                                desc: object) -> bool:
        """Whether a numeric-Scalar read must be bound as a 0-d tile load.

        Mirrors the launch site exactly when the kernel entry is at hand: every
        numeric (float/int/uint) Scalar in the kernel's INPUT list is a
        1-element device array (regardless of transience/storage —
        covariance's host-created transient prefactor is still a param), while
        kernel-written scalars, bool scalars, and scalars defined inside the
        kernel are plain values. Without an entry (a nested SDFG inside the
        kernel), fall back to the descriptor heuristic
        (:func:`_is_device_staged_scalar`).

        :param state: The state owning the read.
        :param entry: The enclosing CuTile MapEntry, or ``None`` inside a
            nested SDFG.
        :param name: The data name being bound.
        :param desc: Its data descriptor.
        :returns: ``True`` when the binding must be ``ct.load(name, (0,), shape=()).item()``.
        """
        if not _is_device_scalar(desc):
            return False
        if entry is not None:
            return (name in self._kernel_input_names(state, entry)
                    and name not in self._kernel_output_names(state, entry))
        return _is_device_staged_scalar(desc)

    def _emit_scalar_bridge_binding(self, sdfg: "SDFG", state: "SDFGState", node: nodes.AccessNode, cfg: object,
                                    state_id: int, callsite_stream: PythonCodeIOStream) -> None:
        """Bind a Register-storage scalar bridge to its source kernel parameter.

        A loop-invariant scalar (e.g. ``alpha``) that the vectorizer staged via
        :func:`~dace.transformation.passes.vectorization.insert_tile_load_store.stage_constant_access`
        appears inside the cuTile scope as a fresh ``Register`` scalar transient
        (``alpha_const``) fed by an edge from the MapEntry. The scalar itself is
        passed into the kernel as a plain parameter (``alpha``), so the bridge is
        just a rename: emit ``alpha_const = alpha``. Without this the kernel body
        references the undefined ``alpha_const`` and the ``cuda.tile`` compiler
        raises ``Undefined variable alpha_const``.

        When the traced source is an *array* rather than a scalar, the bridge
        stages a single element of it (``stage_constant_access`` with a
        ``src_subset`` like ``aa[0, j]``). cuTile arrays are not subscriptable
        inside a kernel and the *outer* memlet subset is propagated over the
        map range (a non-constant slice the ``cuda.tile`` compiler rejects),
        so the element is read with a scalar tile load using the *inner*
        memlet's subset, which is expressed in kernel-bound map parameters:
        ``aa_const = ct.load(aa, (0, j), shape=()).item()``.

        :param sdfg: The SDFG.
        :param state: The state holding ``node``.
        :param node: The Register-storage scalar bridge AccessNode.
        :param cfg: The control flow graph.
        :param state_id: The state ID.
        :param callsite_stream: Stream for call-site (kernel body) code.
        """
        in_edges = list(state.in_edges(node))
        map_entry_edges = [e for e in in_edges if isinstance(e.src, nodes.MapEntry)]
        if in_edges and not map_entry_edges:
            # A code-node producer (tasklet / nested SDFG) binds the name in its
            # own emission; anything else leaves the bridge undefined in the
            # kernel body -- surface it instead of silently emitting nothing.
            if not any(isinstance(e.src, nodes.CodeNode) for e in in_edges):
                srcs = sorted({type(e.src).__name__ for e in in_edges})
                warnings.warn(f"cuTile codegen: scalar bridge {node.data!r} is fed by {srcs} instead of a "
                              f"MapEntry; no binding emitted (the kernel may reference an undefined name).")
            return
        kernel_entry = _enclosing_cutile_entry(state, node)
        kernel_outputs = (self._kernel_output_names(state, kernel_entry) if kernel_entry is not None else set())
        for in_edge in map_entry_edges:
            outer_edge = self._trace_load_source_edge(state, in_edge.src, in_edge)
            if outer_edge is None:
                warnings.warn(f"cuTile codegen: could not trace the source of scalar bridge {node.data!r} "
                              f"through MapEntry {in_edge.src.map.label!r}; no binding emitted.")
                continue
            src_name = outer_edge.data.data if outer_edge.data else outer_edge.src.data
            src_desc = sdfg.arrays.get(src_name)
            # Mirror the launch-site convention (``_launch_arg_expr``): only
            # *input-only* numeric Scalars in DEVICE memory (kernel params, or
            # GPU_Global-staged transients like gramschmidt's ``R_index``) are
            # 1-element device arrays bound as 0-d tiles; a kernel-written
            # scalar is passed raw (and numeric input+output scalars are
            # rejected in generate_scope). A Register/Default TRANSIENT
            # scalar is bound in-kernel as a plain Python variable (e.g.
            # syrk's ``alpha_times_A_slice``), so a ``ct.load`` on it would be
            # invalid -- it takes the rename path.
            if self._scalar_needs_tile_load(state, kernel_entry, src_name, src_desc):
                # Numeric scalar: the kernel parameter is a 1-element device
                # array (launch-site normalization, full f64/int64 precision);
                # bind it as a 0-d tile.
                source_expr = _scalar_tile_load(src_name)
            elif src_desc is None or isinstance(src_desc, data.Scalar):
                # Non-numeric scalar (bool -- complex is rejected at the
                # launch boundary) or unknown descriptor: the kernel parameter
                # carries the plain value (launch-site ``.item()``), so the
                # bridge is a rename.
                source_expr = src_name
            else:
                # Array source: scalar tile load of the staged element. The
                # inner memlet subset carries the per-element index in map
                # parameters bound inside the kernel; the outer subset is
                # propagated over the map range and unusable in-kernel.
                subset = None
                if in_edge.data is not None and in_edge.data.data == src_name:
                    subset = in_edge.data.subset
                elif outer_edge.data is not None:
                    subset = outer_edge.data.subset
                if subset is None:
                    warnings.warn(f"cuTile codegen: scalar bridge {node.data!r} stages array "
                                  f"{src_name!r} without a usable subset; no binding emitted.")
                    continue
                index_exprs = _element_index_exprs(subset)
                if index_exprs is None:
                    warnings.warn(f"cuTile codegen: scalar bridge {node.data!r} stages a non-element "
                                  f"subset {subset} of {src_name!r}; using the per-dim begins.")
                    index_exprs = _element_index_exprs(subset, allow_multi_element=True)
                source_expr = _scalar_element_load(src_name, index_exprs)
            if source_expr != node.data:
                callsite_stream.write(f"{node.data} = {source_expr}", cfg, state_id)

    def _generate_AccessNode(self, sdfg: "SDFG", cfg: object, dfg: object, state_id: int, node: nodes.AccessNode,
                             function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        """Generate code for an AccessNode inside a cuTile scope.

        Handles four cases:

        1. **Tile loaded from global array** (via MapEntry): emit
           ``ct.load`` / ``ct.gather``.
        2. **Tile written by tasklet**: bind Python variable.
        3. **Tile stored to global array** (via MapExit): emit
           ``ct.store`` / ``ct.scatter``.
        4. **Tile read by tasklet**: no code needed (downstream reads
           the variable).

        :param sdfg: The SDFG.
        :param cfg: The control flow graph.
        :param dfg: The dataflow graph.
        :param state_id: The state ID.
        :param node: The AccessNode.
        :param function_stream: Stream for function-level code.
        :param callsite_stream: Stream for call-site code.
        :raises RuntimeError: For unsupported patterns (e.g. AccessNode -> AccessNode).
        """
        state = cfg.state(state_id)
        desc = sdfg.arrays.get(node.data)

        if desc is None:
            return  # Unknown array, skip

        # Only handle CuTile_Tile storage nodes in the AccessNode-centric path.
        # Non-tile AccessNodes (e.g. global arrays) are outside the scope
        # and connected via MapEntry/MapExit.
        if desc.storage != dtypes.StorageType.CuTile_Tile:
            # A Register-storage scalar bridge (a loop-invariant scalar such as
            # ``alpha`` staged by ``stage_constant_access``) enters the cuTile
            # kernel through the MapEntry, where it is a kernel parameter. Emit
            # the rename ``<bridge> = <scalar_param>`` so the kernel body can
            # read it; without this the tasklet references an undefined
            # ``*_const`` name and the cuda.tile compiler raises
            # ``Undefined variable <name>``.
            if desc.storage == dtypes.StorageType.Register and isinstance(desc, data.Scalar):
                self._emit_scalar_bridge_binding(sdfg, state, node, cfg, state_id, callsite_stream)
            return

        entry = _enclosing_cutile_entry(state, node)
        if entry is None and not self._in_cutile_context(sdfg, state, node):
            raise RuntimeError(f"CuTile_Tile AccessNode {node.data!r} found outside a CuTile scope.")

        # Process incoming edges -- handle loads and variable bindings
        for in_edge in state.in_edges(node):
            src = in_edge.src

            if isinstance(src, nodes.MapEntry):
                # Load from global array through MapEntry
                global_arr = self._trace_load_source(state, src, node, in_edge)
                if global_arr is None:
                    continue

                tile_shape = self._resolve_single_tile_shape(entry, node.data, sdfg)
                self._validate_tile_shape(tile_shape, entry, node.data, sdfg)
                map_index_exprs = _map_index_exprs(entry)
                cutile_index = ", ".join(f"__pid{d}" for d in range(len(entry.map.range)))

                # Fold the outer memlet's constant begin offset (e.g. ``A[1:-1]``
                # -> ``+ 1``) into the element index; a non-zero offset is not
                # block-aligned, so it forces the per-element ``ct.gather`` path.
                begins = self._load_source_begins(state, src, in_edge)
                offsets = self._const_begin_offsets(begins, entry)
                has_offset = any(o != 0 for o in offsets)
                gather_index_exprs = self._apply_index_offsets(map_index_exprs, offsets)

                if has_offset or self._needs_gather_for_tile(entry, tile_shape, sdfg):
                    self._emit_gather_load(callsite_stream, global_arr, node.data, gather_index_exprs, tile_shape, cfg,
                                           state_id)
                else:
                    shape_str = ", ".join(str(s) for s in tile_shape)
                    callsite_stream.write(
                        f"{node.data} = ct.load({global_arr}, "
                        f"index=({cutile_index},), shape=({shape_str},))", cfg, state_id)

            elif isinstance(src, nodes.NestedSDFG):
                # NestedSDFG output -> tile: bind variable. The nested-function
                # call site only assigns the *connector* name (``conn = func(...)``),
                # so the rename to this tile's data name must happen here.
                src_conn = in_edge.src_conn
                if src_conn is not None and src_conn != node.data:
                    callsite_stream.write(f"{node.data} = {src_conn}", cfg, state_id)

            elif isinstance(src, nodes.Tasklet):
                # Tasklet output -> tile: the binding (``tile = conn``) is already
                # emitted by ``_generate_Tasklet``'s output post-bind. Emitting it
                # here too would duplicate the assignment.
                pass

            elif isinstance(src, nodes.AccessNode):
                # Tile-to-tile dataflow is a rename of an immutable value.
                src_desc = sdfg.arrays.get(src.data)
                if (src_desc is not None
                        and src_desc.storage in (dtypes.StorageType.CuTile_Tile, dtypes.StorageType.Register)):
                    # TODO: Add check that shpaes match and we move the full tile
                    if src.data != node.data:
                        callsite_stream.write(f"{node.data} = {src.data}", cfg, state_id)
                else:
                    raise RuntimeError(f"AccessNode-to-AccessNode copy ({src.data!r} -> {node.data!r}) "
                                       f"is not supported in CuTile scope. Use library nodes for copies.")

        # Process outgoing edges -- handle stores
        for out_edge in state.out_edges(node):
            dst = out_edge.dst

            if isinstance(dst, nodes.MapExit):
                # Store tile to global array through MapExit
                global_arr = self._trace_store_target(state, dst, node, out_edge)
                if global_arr is None:
                    continue

                tile_shape = self._resolve_single_tile_shape(entry, node.data, sdfg)
                self._validate_tile_shape(tile_shape, entry, node.data, sdfg)
                map_index_exprs = _map_index_exprs(entry)
                cutile_index = ", ".join(f"__pid{d}" for d in range(len(entry.map.range)))

                # Fold the outer memlet's constant begin offset (e.g. ``B[1:-1]``
                # -> ``+ 1``) into the element index; a non-zero offset is not
                # block-aligned, so it forces the per-element ``ct.scatter`` path.
                begins = self._store_target_begins(state, dst, out_edge)
                offsets = self._const_begin_offsets(begins, entry)
                has_offset = any(o != 0 for o in offsets)
                scatter_index_exprs = self._apply_index_offsets(map_index_exprs, offsets)

                if has_offset or self._needs_gather_for_tile(entry, tile_shape, sdfg):
                    # Build index tiles for scatter
                    idx_vars = self._emit_gather_load(callsite_stream, global_arr, f"__ct_scatter_{node.data}",
                                                      scatter_index_exprs, tile_shape, cfg, state_id)
                    self._emit_scatter_store(callsite_stream, global_arr, node.data, idx_vars, cfg, state_id)
                else:
                    callsite_stream.write(f"ct.store({global_arr}, index=({cutile_index},), "
                                          f"tile={node.data})", cfg, state_id)

            elif isinstance(dst, (nodes.Tasklet, nodes.NestedSDFG)):
                # Tile -> tasklet: no code needed (tasklet reads the variable)
                pass

            elif isinstance(dst, nodes.AccessNode):
                # Tile-to-tile dataflow is a rename emitted at the destination
                # AccessNode's incoming edge; nothing to do here.  Reject only
                # genuinely unsupported cross-storage copies.
                dst_desc = sdfg.arrays.get(dst.data)
                if (dst_desc is None
                        or dst_desc.storage not in (dtypes.StorageType.CuTile_Tile, dtypes.StorageType.Register)):
                    # TODO: Add check that shpaes match and we move the full tile
                    raise RuntimeError(f"AccessNode-to-AccessNode copy ({node.data!r} -> {dst.data!r}) "
                                       f"is not supported in CuTile scope. Use library nodes for copies.")

    # ------------------------------------------------------------------
    # Tasklet
    # ------------------------------------------------------------------

    def _generate_Tasklet(self, sdfg: "SDFG", cfg: object, dfg: object, state_id: int, node: nodes.Tasklet,
                          function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        """Generate code for a Tasklet inside a cuTile scope.

        Binds input connectors from tile variables, emits the tasklet
        body, then binds outputs to downstream tile variables.

        :param sdfg: The SDFG.
        :param cfg: The control flow graph.
        :param dfg: The dataflow graph.
        :param state_id: The state ID.
        :param node: The Tasklet node.
        :param function_stream: Stream for function-level code.
        :param callsite_stream: Stream for call-site code.
        :raises NotImplementedError: If the tasklet uses a non-Python language.
        :raises RuntimeError: If the tasklet is not inside a CuTile scope.
        """
        if node.code.language != dtypes.Language.Python:
            raise NotImplementedError("CuTile backend only supports Python tasklets.")
        state = cfg.state(state_id)
        entry = _enclosing_cutile_entry(state, node)
        if entry is None and not self._in_cutile_context(sdfg, state, node):
            raise RuntimeError("CuTile tasklet handler invoked outside a CuTile scope.")

        if node.instrument != dtypes.InstrumentationType.No_Instrumentation:
            raise RuntimeError("Node-level instrumentation is not supported inside cuTile kernels; "
                               "instrument the enclosing cuTile map (kernel) instead.")

        init_code = codeblock_to_python(node.code_init).strip()
        if init_code:
            self._frame._initcode.write(init_code, sdfg)
        exit_code = codeblock_to_python(node.code_exit).strip()
        if exit_code:
            self._frame._exitcode.write(exit_code, sdfg)

        self._dispatcher.defined_vars.enter_scope(node)
        # Tile-valued input connectors and their ct dtype names, for the
        # kernel-body rewrites (ternary -> ct.where literal-arm casting).
        from dace.libraries.tileops._pure_codegen import ct_dtype_name
        tile_conn_dtypes: Dict[str, str] = {}
        for edge in state.in_edges(node):
            if not edge.dst_conn or not isinstance(edge.src, nodes.AccessNode):
                continue
            src_desc = sdfg.arrays.get(edge.src.data)
            if src_desc is not None and src_desc.storage == dtypes.StorageType.CuTile_Tile:
                tile_conn_dtypes[edge.dst_conn] = ct_dtype_name(src_desc.dtype)
        try:
            # Bind inputs from tile AccessNodes
            for edge in state.in_edges(node):
                if not edge.dst_conn:
                    continue
                rhs: Optional[str] = None
                if isinstance(edge.src, nodes.AccessNode):
                    rhs = edge.src.data
                    src_desc = sdfg.arrays.get(rhs)
                    if (src_desc is not None
                            and src_desc.storage not in (dtypes.StorageType.CuTile_Tile, dtypes.StorageType.Register)):
                        # Non-tile source at kernel scope. Binding the bare name
                        # hands the tasklet the WHOLE ct array, dropping the
                        # memlet subset (syrk's ``alpha * A[i, k]`` became
                        # ``alpha * A`` -- a mixed-rank TileTypeError at
                        # runtime). Bind the addressed element instead.
                        if self._scalar_needs_tile_load(state, entry, rhs, src_desc):
                            # Numeric-scalar kernel param: 1-element device
                            # array (launch-site convention) -> 0-d tile.
                            rhs = _scalar_tile_load(rhs)
                        elif (isinstance(src_desc, data.Array) and not isinstance(src_desc, data.View)
                              and edge.data is not None and edge.data.data == rhs):
                            # Global array read: 0-d tile load of the addressed
                            # element. Multi-element subsets (e.g. gather
                            # sources) keep the raw array binding.
                            idx = _element_index_exprs(edge.data.subset)
                            if idx is not None:
                                rhs = _scalar_element_load(rhs, idx)
                        # Transient scalars are plain in-kernel Python variables
                        # (bound by rename); bool scalar params carry the
                        # plain value -- both keep the bare-name binding.
                elif isinstance(edge.src, (nodes.MapEntry, nodes.ConsumeEntry)):
                    # Trace through the scope entries to the root AccessNode.
                    # The memlet path walks ALL enclosing entries (a one-level
                    # connector hop breaks for doubly-nested scopes, and the
                    # connector name may be a stale transient name that
                    # differs from the array actually flowing through).
                    root = state.memlet_path(edge)[0].src
                    if isinstance(root, nodes.AccessNode):
                        rhs = root.data
                        # An *input-only* numeric scalar in device memory
                        # (kernel param or GPU_Global-staged transient) is a
                        # 1-element device array; bind as 0-d tile. Mirrors the
                        # launch-site condition in ``_launch_arg_expr``
                        # (kernel-written scalars are passed raw; numeric
                        # input+output scalars are rejected in
                        # ``generate_scope``; Register/Default transient
                        # scalars are plain in-kernel variables -- rename
                        # only).
                        root_desc = sdfg.arrays.get(rhs)
                        if self._scalar_needs_tile_load(state, entry, rhs, root_desc):
                            rhs = _scalar_tile_load(rhs)
                        elif (isinstance(root_desc, data.Array) and not isinstance(root_desc, data.View)
                              and root_desc.storage not in (dtypes.StorageType.CuTile_Tile, dtypes.StorageType.Register)
                              and edge.data is not None and edge.data.data == rhs):
                            # Global-array element read through the scope
                            # entry (symm's ``alpha * B[i, j]``): binding the
                            # bare name hands the tasklet the whole ct array.
                            # The INNER memlet subset carries the per-element
                            # index in map params bound in-kernel; emit the
                            # 0-d element load from it. Multi-element subsets
                            # (tile loads, gathers) keep the raw binding.
                            idx = _element_index_exprs(edge.data.subset)
                            if idx is not None:
                                rhs = _scalar_element_load(rhs, idx)
                    elif edge.data is not None and edge.data.data is not None:
                        rhs = edge.data.data
                    elif edge.src_conn is not None:
                        rhs = edge.src_conn  # fallback
                elif edge.src_conn is not None:
                    rhs = edge.src_conn
                if rhs is None:
                    continue
                callsite_stream.write(f"{edge.dst_conn} = {rhs}", cfg, state_id)
                self._dispatcher.defined_vars.add(edge.dst_conn, dispatcher_mod.DefinedType.Scalar, "object")

            # Pre-bind output connectors that trace to actual (global) arrays.
            # This is needed for in-place operations like ct.scatter(_dst, ...)
            # where _dst must be bound to the global array before the body runs.
            # We must NOT pre-bind outputs that go to tile-local AccessNodes
            # (Register or CuTile_Tile storage) because those variables don't
            # exist yet -- they are created by the tasklet body.
            _LOCAL_STORAGES = {
                dtypes.StorageType.Register,
                dtypes.StorageType.CuTile_Tile,
            }
            _prebind_outputs: set = set()
            for edge in state.out_edges(node):
                if not edge.src_conn:
                    continue
                dst_name: Optional[str] = None
                if isinstance(edge.dst, nodes.AccessNode):
                    # Check if this is a local tile variable -- skip pre-bind
                    dst_desc = sdfg.arrays.get(edge.dst.data)
                    if dst_desc is not None and dst_desc.storage in _LOCAL_STORAGES:
                        continue
                    dst_name = edge.dst.data
                elif isinstance(edge.dst, (nodes.MapExit, nodes.ConsumeExit)):
                    # Trace through ALL enclosing scope exits (see input
                    # binding above for why a one-level hop is insufficient).
                    leaf = state.memlet_path(edge)[-1].dst
                    if isinstance(leaf, nodes.AccessNode):
                        dst_name = leaf.data
                if dst_name and dst_name != edge.src_conn:
                    callsite_stream.write(f"{edge.src_conn} = {dst_name}", cfg, state_id)
                    self._dispatcher.defined_vars.add(edge.src_conn, dispatcher_mod.DefinedType.Scalar, "object")
                    _prebind_outputs.add(edge.src_conn)

            # Emit tasklet body (rewritten for in-kernel execution: ct.* math
            # spellings, tile-conditioned ternaries -> ct.where).
            body = codeblock_to_python(node.code).strip() or "pass"
            body = _rewrite_cutile_tasklet_code(body, tile_conn_dtypes)
            callsite_stream.write(f"\n####### Tasklet: {node.label}\n\n", cfg, state_id)
            callsite_stream.write(body)
            callsite_stream.write(f"\n####### End of tasklet: {node.label}\n\n", cfg, state_id)

            # Bind outputs to downstream tile AccessNodes or through MapExit.
            # Skip connectors that were already pre-bound above — the in-place
            # operation (e.g. ct.scatter) already modified the array directly.
            for edge in state.out_edges(node):
                if not edge.src_conn:
                    continue
                if isinstance(edge.dst, nodes.AccessNode):
                    if edge.dst.data == edge.src_conn:
                        continue
                    if edge.src_conn in _prebind_outputs:
                        continue
                    callsite_stream.write(f"{edge.dst.data} = {edge.src_conn}", cfg, state_id)
                    self._dispatcher.defined_vars.add(edge.dst.data, dispatcher_mod.DefinedType.Scalar, "object")
                elif isinstance(edge.dst, (nodes.MapExit, nodes.ConsumeExit)):
                    # Trace through ALL enclosing scope exits to the actual
                    # destination AccessNode (see input binding above for why
                    # a one-level connector hop is insufficient).
                    leaf = state.memlet_path(edge)[-1].dst
                    if isinstance(leaf, nodes.AccessNode):
                        dst_name = leaf.data
                        if dst_name != edge.src_conn and edge.src_conn not in _prebind_outputs:
                            callsite_stream.write(f"{dst_name} = {edge.src_conn}", cfg, state_id)
                            self._dispatcher.defined_vars.add(dst_name, dispatcher_mod.DefinedType.Scalar, "object")
        finally:
            self._dispatcher.defined_vars.exit_scope(node)

    # ------------------------------------------------------------------
    # NestedSDFG — emitted as a module-level function
    # ------------------------------------------------------------------
    #
    # A user helper function compiled by DaCe becomes a NestedSDFG.  It is
    # emitted as a plain module-level Python function that the cuTile kernel
    # (or an enclosing nested function) calls.  The body is generated by the
    # SAME shared node dispatcher used for the kernel body, so every tile op
    # routes back to ``_generate_Tasklet`` / ``_generate_AccessNode`` — there
    # is no bespoke node walking here.  The boundary follows cuTile's value
    # model (https://docs.nvidia.com/cuda/cutile-python/execution.html):
    #
    #   * Tiles (and registers) are immutable, so tile-valued outputs are
    #     *returned* from the function.
    #   * Global arrays are read/write views, so global-array-valued outputs
    #     are passed in as destination parameters and written in place
    #     (``ct.store`` / ``ct.scatter``); they are not returned.
    #
    # The inner (map-less) nodes are recognised as living in a cuTile scope
    # structurally, via :meth:`_sdfg_is_cutile_body`.

    #: Inner-SDFG storage types whose values are immutable Python locals
    #: (returned from the generated function rather than written in place).
    _RETURNED_OUTPUT_STORAGES = frozenset({
        dtypes.StorageType.CuTile_Tile,
        dtypes.StorageType.Register,
    })

    def _is_returned_output(self, inner_sdfg: "SDFG", conn: str) -> bool:
        """Whether an output connector is an immutable value (returned) rather
        than a global-memory destination view (passed in, written in place).

        :param inner_sdfg: The nested SDFG.
        :param conn: The output connector name (== inner array name).
        :returns: ``True`` if the value must be returned from the function.
        """
        desc = inner_sdfg.arrays.get(conn)
        return desc is not None and desc.storage in self._RETURNED_OUTPUT_STORAGES

    def _generate_NestedSDFG(self, sdfg: "SDFG", cfg: object, dfg: object, state_id: int, node: nodes.NestedSDFG,
                             function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        """Emit a module-level function for a NestedSDFG and call it.

        :param sdfg: The containing SDFG.
        :param cfg: The containing control-flow region.
        :param dfg: The dataflow graph (unused; kept for the dispatcher signature).
        :param state_id: The containing state ID.
        :param node: The NestedSDFG node.
        :param function_stream: Stream for module-level code (the function def).
        :param callsite_stream: Stream for the call site.
        """
        state = cfg.state(state_id)
        inner_sdfg = node.sdfg

        # Lower any tile-op library nodes to their cuTile tasklets first.
        inner_sdfg.expand_library_nodes(recursive=True)

        func_name = (f"__dace_nested_{inner_sdfg.name}_{cfg.cfg_id}_"
                     f"{state_id}_{state.node_id(node)}")

        input_conns: List[str] = sorted({e.dst_conn for e in state.in_edges(node) if e.dst_conn is not None})
        output_conns: List[str] = sorted({e.src_conn for e in state.out_edges(node) if e.src_conn is not None})
        returned_outputs = [c for c in output_conns if self._is_returned_output(inner_sdfg, c)]
        dest_outputs = [c for c in output_conns if not self._is_returned_output(inner_sdfg, c)]
        symbol_names = self._nsdfg_runtime_symbols(node)

        # Parameters: inputs, then global-array destinations, then symbols.
        param_conns = list(dict.fromkeys(input_conns + dest_outputs))
        params = param_conns + symbol_names

        if func_name not in self._generated_nested_functions:
            self._emit_nsdfg_function(inner_sdfg, func_name, params, returned_outputs, function_stream)
            self._generated_nested_functions[func_name] = func_name

        # --- Call site ---
        # Mirror the deduplicated ``param_conns`` order so an in-out array
        # (a connector that is both an input and a destination output) is
        # passed exactly once, matching the function's parameter list.
        input_conn_set = set(input_conns)
        call_args: List[str] = [
            self._resolve_nsdfg_input_var(state, node, c) if c in input_conn_set else self._resolve_nsdfg_output_var(
                state, node, c) for c in param_conns
        ]
        for sym_name in symbol_names:
            mapping_expr = node.symbol_mapping.get(sym_name)
            call_args.append(symstr(mapping_expr) if mapping_expr is not None else sym_name)
        args_str = ", ".join(call_args)

        if returned_outputs:
            lhs = ", ".join(returned_outputs)
            callsite_stream.write(f"{lhs} = {func_name}({args_str})", cfg, state_id)
            for c in returned_outputs:
                self._dispatcher.defined_vars.add(c, dispatcher_mod.DefinedType.Scalar, "object")
        else:
            callsite_stream.write(f"{func_name}({args_str})", cfg, state_id)

    def _emit_nsdfg_function(self, inner_sdfg: "SDFG", func_name: str, params: List[str], returned_outputs: List[str],
                             function_stream: PythonCodeIOStream) -> None:
        """Write the module-level function definition for a NestedSDFG.

        The body is generated through the shared node dispatcher — every node
        routes back to the cuTile per-node handlers, since
        :meth:`_sdfg_is_cutile_body` recognises *inner_sdfg* as a cuTile body.

        :param inner_sdfg: The nested SDFG.
        :param func_name: The generated function name.
        :param params: Ordered parameter names (inputs, destinations, symbols).
        :param returned_outputs: Output connectors returned (immutable values).
        :param function_stream: Stream to append the function definition to.
        """
        body_stream = PythonCodeIOStream()

        def dispatch_state(inner_state: "SDFGState") -> str:
            tmp = PythonCodeIOStream()
            self._dispatcher.dispatch_subgraph(inner_sdfg,
                                               inner_state.parent_graph,
                                               inner_state,
                                               inner_state.block_id,
                                               function_stream,
                                               tmp,
                                               skip_entry_node=False)
            return tmp.getvalue()

        py_cflow.control_flow_region_to_code(inner_sdfg, dispatch_state, self._frame, inner_sdfg.symbols, body_stream)

        if returned_outputs:
            body_stream.write(f"return {', '.join(returned_outputs)}")

        function_stream.write("")
        function_stream.write(f"def {func_name}({', '.join(params)}):")
        with function_stream.indented():
            body_code = body_stream.getvalue().rstrip("\n")
            function_stream.write(body_code if body_code.strip() else "pass")
        function_stream.write("")

    # ------------------------------------------------------------------
    # NestedSDFG helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _nsdfg_runtime_symbols(node: nodes.NestedSDFG) -> List[str]:
        """Return sorted list of symbol names that must be passed at runtime.

        Symbols that are free in the inner SDFG and not constants are
        included.

        :param node: The NestedSDFG node.
        :returns: Sorted list of symbol names.
        """
        inner_sdfg = node.sdfg
        free_symbols = set(str(s) for s in inner_sdfg.used_symbols(all_symbols=False, keep_defined_in_mapping=True))
        return [
            sym_name for sym_name in sorted(node.symbol_mapping.keys())
            if sym_name in free_symbols and sym_name not in inner_sdfg.constants
        ]

    @staticmethod
    def _resolve_nsdfg_input_var(state: "SDFGState", node: nodes.NestedSDFG, conn_name: str) -> str:
        """Resolve the variable name for a NestedSDFG input connector at the call site.

        Traces edges to find the actual source variable name.

        :param state: The containing state.
        :param node: The NestedSDFG node.
        :param conn_name: The input connector name.
        :returns: The variable name to pass as argument.
        """
        for edge in state.in_edges(node):
            if edge.dst_conn == conn_name:
                if isinstance(edge.src, nodes.AccessNode):
                    return edge.src.data
                elif edge.src_conn is not None:
                    # Through a MapEntry — trace to the outer edge.
                    if isinstance(edge.src, nodes.MapEntry):
                        outer_conn = _matching_outer_connector(edge.src_conn)
                        for outer_edge in state.in_edges_by_connector(edge.src, outer_conn):
                            if isinstance(outer_edge.src, nodes.AccessNode):
                                return outer_edge.src.data
                    return edge.src_conn
        return conn_name  # fallback: use connector name itself

    @staticmethod
    def _resolve_nsdfg_output_var(state: "SDFGState", node: nodes.NestedSDFG, conn_name: str) -> str:
        """Resolve the variable name for a NestedSDFG output connector at the call site.

        Traces edges to find the actual destination variable name.

        :param state: The containing state.
        :param node: The NestedSDFG node.
        :param conn_name: The output connector name.
        :returns: The variable name to assign the return value to.
        """
        for edge in state.out_edges(node):
            if edge.src_conn == conn_name:
                if isinstance(edge.dst, nodes.AccessNode):
                    return edge.dst.data
                elif edge.dst_conn is not None:
                    # Through a MapExit — trace to the outer edge.
                    if isinstance(edge.dst, nodes.MapExit):
                        outer_conn = _matching_inner_connector(edge.dst_conn)
                        for outer_edge in state.out_edges_by_connector(edge.dst, outer_conn):
                            if isinstance(outer_edge.dst, nodes.AccessNode):
                                return outer_edge.dst.data
                    return edge.dst_conn
        return conn_name  # fallback: use connector name itself

    # ------------------------------------------------------------------
    # Scope generation (kernel wrapper + launch)
    # ------------------------------------------------------------------

    @staticmethod
    def _launch_arg_expr(sdfg: "SDFG", name: str, is_output: bool) -> str:
        """Return the ``ct.launch`` argument expression for a data name.

        Input-only Scalar arguments are unwrapped to native Python scalars:
        at runtime they may be 0-d numpy buffers (scalar SDFG arguments),
        numpy scalars, or 0-d cupy arrays (values read from GPU memory), all
        of which ``ct.launch`` rejects. Arrays and kernel-written scalars are
        passed through unchanged.

        :param sdfg: The SDFG containing the data descriptor.
        :param name: The data name.
        :param is_output: Whether the kernel writes to this data.
        :returns: The Python expression string for the launch argument.
        :raises NotImplementedError: For complex Scalar inputs (no cuTile
            launch convention exists for them).
        """
        expr = _array_runtime_name(sdfg, name)
        desc = sdfg.arrays.get(name)
        if isinstance(desc, data.Scalar) and not is_output:
            if _is_device_scalar(desc):
                # Numeric scalars (float/int/uint) go through device memory
                # as 1-element arrays (``cupy.asarray(x).reshape(1)`` -- a
                # no-copy view for device-resident values) and are bound as
                # 0-d tiles in-kernel: by-value floats are typed float32 by
                # cuda.tile (silent f64 precision loss), by-value Python ints
                # >= 2**31 raise OverflowError, and numpy int scalars are
                # rejected outright. The declared dtype is preserved (int64
                # stays int64, uint64 stays uint64).
                np_name = desc.dtype.as_numpy_dtype().name
                return f"cupy.asarray({expr}, dtype=numpy.{np_name}).reshape(1)"
            if desc.dtype.as_numpy_dtype().kind == 'c':
                raise NotImplementedError(f"cuTile codegen: complex Scalar {name!r} is not supported as a "
                                          f"kernel argument (no by-value or device-memory convention).")
            # Bool scalars (the only remaining kind) are passed by value
            # (typed exactly at the launch boundary); ``.item()`` covers cupy
            # 0-d arrays, numpy scalars, and 0-d numpy buffers; plain Python
            # bools pass through.
            return f"({expr}.item() if hasattr({expr}, 'item') else {expr})"
        return expr

    def generate_scope(self, sdfg: "SDFG", cfg: object, dfg_scope: object, state_id: int,
                       function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        """Generate the cuTile kernel wrapper and launch call for a map scope.

        Emits a ``@ct.kernel``-decorated function containing the scope body,
        then emits a ``ct.launch(...)`` call at the call site.

        :param sdfg: The SDFG.
        :param cfg: The control flow graph.
        :param dfg_scope: The scope subgraph view.
        :param state_id: The state ID.
        :param function_stream: Stream for function-level code.
        :param callsite_stream: Stream for call-site code.
        :raises ValueError: If the scope source is not a MapEntry.
        :raises NotImplementedError: If a Scalar is both a kernel input and a
            kernel output, or a complex Scalar is a kernel argument.
        """
        entry = dfg_scope.source_nodes()[0]
        if not isinstance(entry, nodes.MapEntry):
            raise ValueError("CuTilePythonCodeGen expects a map scope")

        state = cfg.state(state_id)
        exit_node = state.exit_node(entry)
        grid_exprs = _grid_exprs_from_map_entry(entry)

        input_arrays = _ordered_unique(e.data.data for e in state.in_edges(entry)
                                       if e.data and e.data.data and isinstance(e.src, nodes.AccessNode))
        output_arrays = _ordered_unique(e.data.data for e in state.out_edges(exit_node)
                                        if e.data and e.data.data and isinstance(e.dst, nodes.AccessNode))
        symbol_types = self._frame.symbols_defined_at(state, entry)
        free_syms = _collect_free_symbols(entry, dfg_scope, sdfg, symbol_types)
        kernel_params = list(dict.fromkeys(input_arrays + output_arrays + free_syms))

        # A Scalar that is BOTH input and output cannot be launched: numeric
        # scalars cannot satisfy the two conventions at once (the launch site
        # passes kernel-written scalars raw, while in-kernel reads bind input
        # numeric scalars as 0-d tiles from a 1-element device array), and a
        # raw bool/complex in/out scalar is a 0-d host buffer that ct.launch
        # rejects at runtime (probed: ``RuntimeError: NumPy only supports
        # stream=None``) with no writeback path either. The current lowering
        # pipeline never produces this shape; fail loudly if it ever does.
        for name in set(input_arrays) & set(output_arrays):
            if isinstance(sdfg.arrays.get(name), data.Scalar):
                raise NotImplementedError(f"cuTile codegen: Scalar {name!r} is both a kernel input and a "
                                          f"kernel output; the scalar launch conventions do not "
                                          f"support in/out scalars. Stage the scalar through a 1-element array "
                                          f"instead.")

        # Numeric SYMBOLS would otherwise ride the by-value path, where the
        # cuda.tile frontend types every Python/numpy float as float32 (silent
        # f64 precision loss) and every int as int32 (OverflowError >= 2**31
        # -- same issues as numeric Scalars). Stage them as 1-element device
        # arrays at the launch site and rebind them as 0-d tiles at kernel
        # entry (symbols are read-only, so the rebind is safe for every
        # downstream use: binop broadcasting, tile indices, range() bounds,
        # and branch conditions all accept 0-d tiles). Grid-dimension
        # computations at the call site keep the raw host values.
        # Scoped repository types also cover host loop and interstate variables.
        # Numeric symbols use device buffers; bool symbols remain exact by value.
        device_syms: Dict[str, str] = {}
        for symbol in free_syms:
            dtype = symbol_types[symbol]
            if dtype is None:
                raise TypeError(f'Cannot determine the type of cuTile symbol {symbol!r}')
            np_dtype = dtype.as_numpy_dtype()
            if np_dtype.kind in 'fiu':
                device_syms[symbol] = np_dtype.name

        kernel_name = (f"__dace_cutile_{sdfg.name}_{cfg.cfg_id}_"
                       f"{state.block_id}_{state.node_id(entry)}")

        kernel_stream = PythonCodeIOStream()
        kernel_stream.write("@ct.kernel")
        kernel_stream.write(f"def {kernel_name}({', '.join(kernel_params)}):")
        with kernel_stream.indented():
            for sym_name in device_syms:
                kernel_stream.write(f"{sym_name} = {_scalar_tile_load(sym_name)}")
            # Emit MapEntry (pid setup) ourselves; the dispatcher's
            # topological walk treats MapEntry specially (dispatch_scope), so
            # we cannot rely on dispatch_subgraph to invoke our handler for it.
            self.generate_node(sdfg, cfg, dfg_scope, state_id, entry, function_stream, kernel_stream)
            # Walk the rest of the scope. Tasklets, MapExit, AccessNodes, and
            # NestedSDFGs are routed to our predicated handlers.
            self._dispatcher.dispatch_subgraph(
                sdfg,
                cfg,
                dfg_scope,
                state_id,
                function_stream,
                kernel_stream,
                skip_entry_node=True,
            )

        function_stream.write("")
        function_stream.write(kernel_stream.getvalue())
        function_stream.write("")

        # The cuTile launch grid is capped at 3 axes by the runtime. Grids with
        # more than 3 tiled dimensions are folded onto the 3 available axes (the
        # kernel recovers per-dim block IDs via div/mod in ``_generate_MapEntry``,
        # using the same ``_fold_grid_to_launch`` layout).
        launch_dims, _ = _fold_grid_to_launch(grid_exprs)
        grid_tuple = f"({', '.join(launch_dims)})"
        deduped_arrays = list(dict.fromkeys(input_arrays + output_arrays))
        # Input-only Scalar parameters are normalized at the launch site: the
        # host-side value may be a 0-d device array (e.g. a scalar computed by
        # host tasklets from ``gpu_arr[i]``), which the cuda.tile kernel cannot
        # use as an arithmetic operand. Note: passing a 0-d ARRAY as a kernel
        # argument crashes the tile compiler; ``shape=()`` loads of 1-element
        # arrays are fine and are exactly how numeric scalars are bound. See
        # ``_launch_arg_expr`` for the per-dtype normalization.
        launch_args = [self._launch_arg_expr(sdfg, n, is_output=n in output_arrays) for n in deduped_arrays]
        for s in free_syms:
            if s in device_syms:
                # Numeric symbols travel through device memory (see above):
                # symbols retain the dtype visible at the launch site.
                np_name = device_syms[s]
                dtype_arg = f", dtype=numpy.{np_name}"
                launch_args.append(f"cupy.asarray({s}{dtype_arg}).reshape(1)")
            else:
                launch_args.append(s)
        args_tuple = (f"({', '.join(launch_args)},)" if len(launch_args) == 1 else f"({', '.join(launch_args)})")
        instrumented = (entry.map.instrument != dtypes.InstrumentationType.No_Instrumentation)

        # Instrumentation: kernel-scope begin (before launch)
        if instrumented:
            for instr in self._dispatcher.instrumentation.values():
                if instr is not None:
                    instr.on_scope_entry(sdfg, cfg, state, entry, callsite_stream, callsite_stream, function_stream)

        # Zero- or negative-trip maps (e.g. a loop-dependent range ``0:i`` at
        # ``i == 0``, or ``1:N-1`` at ``N == 1``) yield a non-positive grid
        # dimension, which the cuTile runtime rejects ("invalid argument");
        # the launch is a no-op then, so skip it.
        callsite_stream.write(
            f"if min(({', '.join(launch_dims)},)) > 0: "
            f"ct.launch(cupy.cuda.get_current_stream(), {grid_tuple}, "
            f"{kernel_name}, {args_tuple})",
            cfg,
            state_id,
        )

        # Always synchronize so the kernel completes before the host
        # continues (and before any timing measurement ends).
        callsite_stream.write("cupy.cuda.get_current_stream().synchronize()", cfg, state_id)

        # Instrumentation: kernel-scope end (after synchronize). The exit node
        # is passed so the provider resolves the matching entry node (and thus
        # the matching timer-variable id) via ``state.entry_node(exit)``.
        if instrumented:
            for instr in self._dispatcher.instrumentation.values():
                if instr is not None:
                    instr.on_scope_exit(sdfg, cfg, state, exit_node, callsite_stream, callsite_stream, function_stream)
