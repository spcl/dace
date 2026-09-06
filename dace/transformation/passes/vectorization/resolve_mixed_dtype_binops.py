# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Resolve mixed-dtype binary tasklets by inserting explicit type-cast tasklets.

The walker-primary tile pipeline locks a single dtype per lib node (design 6.2): a
``TileBinop``'s two operands and its output must all share one dtype. A frontend binop
whose operands differ in dtype (``int64 + float64``) would otherwise reach
``ConvertTaskletsToTileOps`` and hit its ``NotImplementedError`` mixed-dtype guard.

This pass runs AFTER ``SplitTasklets`` (so every arithmetic tasklet is a single
primitive op ``_o = _a <op> _b``) and BEFORE the tile conversion. For each such tasklet
it promotes the operands to a common dtype and inserts the required casts, following
NumPy promotion (``dace.dtypes.result_type_of``):

* int + float  -> promote the int operand to the float type;
* int + int    -> widen the narrower to the wider int;
* float + float-> widen the narrower to the wider float.

A cast is materialised as a unary tasklet ``_co = dace.<dtype>(_ci)`` writing a register
transient of the promoted dtype -- the tile converter already lowers a dtype-cast call to
a ``TileUnop``. An arithmetic op computes at the promoted dtype and, when the destination
array's dtype differs, a cast tasklet stores the promoted result back into it -- a downcast
when the destination is narrower or a widening store when it is wider (matching the
frontend's implicit assignment cast, so the result stays bit-exact with the unvectorized
reference). Comparison ops keep their ``bool`` output untouched; only their operands are
unified.

The same guard applies to a ternary blend ``_o = ITE(_c, _t, _e)`` / ``_o = _t if _c else
_e`` (lowers to ``TileITE``) and the masked write ``_o = IT(_c, _val)`` (lowers to a masked
``TileStore``): each connector arm is cast to the OUTPUT dtype when it differs, mirroring the
frontend's implicit per-arm assignment cast -- these two forms are NOT ``ast.BinOp`` /
``ast.Compare``, so they need their own detector rather than reusing the binop one.
"""
import ast
from typing import Optional, Tuple

from dace.ordered import OrderedSet

import dace
from dace import dtypes
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.helpers import CodeBlock

#: Comparison ops produce ``bool`` regardless of operand dtype -- unify the operands but
#: never cast the output (mirrors ``convert_tasklets_to_tile_ops._COMPARISON_BINOPS``).
_COMPARISON_AST = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)


def _cast_name(dtype: dtypes.typeclass) -> str:
    """Registered dtype-cast call name (``float64`` / ``int64`` / ...) for ``dtype``.

    Never hardcoded -- taken from the dtype registry, the same source
    ``convert_tasklets_to_tile_ops._CAST_OP_NAMES`` validates against.
    """
    return dtypes.TYPECLASS_TO_STRING[dtype].split("::")[-1]


def ite_operands(tasklet: nodes.Tasklet) -> Optional[Tuple[str, list]]:
    """If ``tasklet`` is a ternary blend -- the Python ``_o = _t if _c else _e`` form or the
    ``_o = ITE(_c, _t, _e)`` call form ``SplitTasklets`` emits for a same-write-set if/else --
    return ``(out_conn, arm_conns)`` where ``arm_conns`` lists the arms that are in-connectors
    (a Symbol/literal arm has no edge, so nothing to cast). Else ``None``.
    """
    if len(tasklet.out_connectors) != 1:
        return None
    try:
        tree = ast.parse(tasklet.code.as_string.strip())
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return None
    assign = tree.body[0]
    if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
        return None
    out_conn = assign.targets[0].id
    rhs = assign.value
    if isinstance(rhs, ast.IfExp):
        arms = [rhs.body, rhs.orelse]
    elif (isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name) and rhs.func.id == "ITE" and len(rhs.args) == 3):
        arms = [rhs.args[1], rhs.args[2]]
    else:
        return None
    in_conns = OrderedSet(tasklet.in_connectors)
    arm_conns = [a.id for a in arms if isinstance(a, ast.Name) and a.id in in_conns]
    return out_conn, arm_conns


def _ite_arm_slots(rhs: ast.expr) -> Optional[list]:
    """The two ``(node, setter)`` pairs addressing an ITE ``rhs``'s then/else arm slots --
    ``IfExp.body`` / ``IfExp.orelse``, or ``Call.args[1]`` / ``Call.args[2]`` for the
    ``ITE(cond, t, e)`` form -- so a caller can read the current arm node and, for a
    literal/Symbol one, replace it in place with a cast. ``None`` when ``rhs`` is not a
    recognised ITE form.
    """
    if isinstance(rhs, ast.IfExp):

        def _set_body(n: ast.expr) -> None:
            rhs.body = n

        def _set_orelse(n: ast.expr) -> None:
            rhs.orelse = n

        return [(rhs.body, _set_body), (rhs.orelse, _set_orelse)]
    if isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name) and rhs.func.id == "ITE" and len(rhs.args) == 3:

        def _set_t(n: ast.expr) -> None:
            rhs.args[1] = n

        def _set_e(n: ast.expr) -> None:
            rhs.args[2] = n

        return [(rhs.args[1], _set_t), (rhs.args[2], _set_e)]
    return None


def masked_write_operand(tasklet: nodes.Tasklet) -> Optional[Tuple[str, str]]:
    """If ``tasklet`` is the masked write ``_o = IT(_cond, _val)`` (the first-class conditional
    write ``NormalizeMaskedWriteTasklets`` emits), return ``(out_conn, val_conn)`` when ``_val``
    is an in-connector; else ``None`` (a constant value arm has no edge to cast).
    """
    if len(tasklet.out_connectors) != 1:
        return None
    try:
        tree = ast.parse(tasklet.code.as_string.strip())
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return None
    assign = tree.body[0]
    if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
        return None
    out_conn = assign.targets[0].id
    rhs = assign.value
    if not (isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name) and rhs.func.id == "IT"
            and len(rhs.args) == 2):
        return None
    val = rhs.args[1]
    if not (isinstance(val, ast.Name) and val.id in tasklet.in_connectors):
        return None
    return out_conn, val.id


def _binop_operands(tasklet: nodes.Tasklet) -> Optional[Tuple[str, str, str, bool]]:
    """If ``tasklet`` is a single ``_o = _a <op> _b`` (arithmetic or comparison) whose two
    operands are input connectors, return ``(out_conn, a_conn, b_conn, is_comparison)``;
    else ``None``.
    """
    if len(tasklet.out_connectors) != 1 or len(tasklet.in_connectors) != 2:
        return None
    try:
        tree = ast.parse(tasklet.code.as_string.strip())
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return None
    assign = tree.body[0]
    if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
        return None
    out_conn = assign.targets[0].id
    rhs = assign.value
    if isinstance(rhs, ast.BinOp):
        left, right, is_cmp = rhs.left, rhs.right, False
    elif isinstance(rhs, ast.Compare) and len(rhs.ops) == 1 and isinstance(rhs.ops[0], _COMPARISON_AST):
        left, right, is_cmp = rhs.left, rhs.comparators[0], True
    else:
        return None
    if not (isinstance(left, ast.Name) and isinstance(right, ast.Name)):
        return None
    a_conn, b_conn = left.id, right.id
    in_conns = OrderedSet(tasklet.in_connectors)
    if a_conn not in in_conns or b_conn not in in_conns or a_conn == b_conn:
        return None
    return out_conn, a_conn, b_conn, is_cmp


class ResolveMixedDtypeBinops(ppl.Pass):
    """Insert cast tasklets so no ``TileBinop`` sees mixed-dtype operands (design 6.2)."""

    CATEGORY: str = "Vectorization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        count = 0
        # Recurse into nested SDFGs: a map body is often a NestedSDFG (frontend ``loop_body`` /
        # ``NestInnermostMapBodyIntoNSDFG``), so the split arithmetic + comparison tasklets that
        # need dtype unifying live inside it -- the sibling ``SplitTasklets`` and
        # ``ConvertTaskletsToTileOps`` both recurse, so a top-level-only sweep here leaves a mixed
        # binop (``int32`` ``mid`` vs ``int64`` map-param compare) to trip the conversion guard.
        for nested in sdfg.all_sdfgs_recursive():
            for state in nested.all_states():
                for tasklet in list(state.nodes()):
                    if not isinstance(tasklet, nodes.Tasklet):
                        continue
                    if tasklet.code.language != dtypes.Language.Python:
                        continue
                    if self._resolve(state, tasklet):
                        count += 1
        return count or None

    def _resolve(self, state: SDFGState, tasklet: nodes.Tasklet) -> bool:
        detected = _binop_operands(tasklet)
        if detected is not None:
            out_conn, a_conn, b_conn, is_cmp = detected
            sdfg = state.sdfg
            in_edges = {e.dst_conn: e for e in state.in_edges(tasklet) if e.data and e.data.data}
            out_edges = [e for e in state.out_edges(tasklet) if e.data and e.data.data]
            if a_conn not in in_edges or b_conn not in in_edges or len(out_edges) != 1:
                return False
            a_edge, b_edge, out_edge = in_edges[a_conn], in_edges[b_conn], out_edges[0]
            a_dt = sdfg.arrays[a_edge.data.data].dtype
            b_dt = sdfg.arrays[b_edge.data.data].dtype
            out_dt = sdfg.arrays[out_edge.data.data].dtype
            promoted = dtypes.result_type_of(a_dt, b_dt)

            need_a = a_dt != promoted
            need_b = b_dt != promoted
            # Comparison output is bool by definition; an arithmetic output needs a cast
            # whenever the destination dtype differs from the promoted compute type -- a
            # downcast (dest narrower) or a widening store (dest wider), both bit-exact with
            # the frontend's implicit assignment cast.
            need_out = (not is_cmp) and (out_dt != promoted)
            if not (need_a or need_b or need_out):
                return False

            if need_a:
                self._insert_operand_cast(state, tasklet, a_edge, a_conn, promoted)
            if need_b:
                self._insert_operand_cast(state, tasklet, b_edge, b_conn, promoted)
            if need_out:
                self._insert_output_cast(state, tasklet, out_edge, out_conn, promoted, out_dt)
            return True
        if self.resolve_ite(state, tasklet):
            return True
        if self.resolve_masked_write(state, tasklet):
            return True
        return self._resolve_assign(state, tasklet)

    def resolve_ite(self, state: SDFGState, tasklet: nodes.Tasklet) -> bool:
        """A same-write-set-if/else blend ``_o = ITE(_c, _t, _e)`` / ``_o = _t if _c else _e``
        lowers to a ``TileITE`` whose ``_t`` / ``_e`` / ``_o`` connectors must share ONE dtype
        (design 6.2, see ``dace.libraries.tileops.nodes.tile_ite``): the pure/ISA expansions
        cast a Symbol or broadcast-Scalar arm to the output dtype, but read a per-lane Tile arm
        RAW, with no cast. An arm whose source differs from the destination dtype must be cast
        here, before ``ConvertTaskletsToTileOps`` wires it straight into the lib node.

        A literal/Symbol arm (no connector, so the loop below never sees it) is NOT touched
        here -- it is left for the pure/ISA expansion to cast inline, exactly as today. A
        tasklet that never reaches that conversion (the untiled remainder of a branched split
        stays a scalar ``ITE(...)`` tasklet forever) needs the SAME casting for its literal arm,
        but doing it at this stage would corrupt a would-be ``TileITE`` Symbol arm instead
        (its ``expr_t`` / ``expr_e`` reaches two different renderers -- the pure expansion's
        ``pyexpr2cpp`` and the ISA expansion's raw C++ embed -- that disagree on whether the
        text is Python or already-C++); see ``CastScalarIteLiteralArms`` below, which runs
        AFTER tile conversion has claimed every arm it will ever claim, so anything it still
        finds is provably staying scalar.
        """
        detected = ite_operands(tasklet)
        if detected is None:
            return False
        out_conn, arm_conns = detected
        sdfg = state.sdfg
        in_edges = {e.dst_conn: e for e in state.in_edges(tasklet) if e.data and e.data.data}
        out_edges = [e for e in state.out_edges(tasklet) if e.data and e.data.data]
        if len(out_edges) != 1:
            return False
        out_dt = sdfg.arrays[out_edges[0].data.data].dtype
        changed = False
        for conn in arm_conns:
            edge = in_edges.get(conn)
            if edge is None:
                continue
            if sdfg.arrays[edge.data.data].dtype != out_dt:
                self._insert_operand_cast(state, tasklet, edge, conn, out_dt)
                changed = True
        return changed

    def resolve_masked_write(self, state: SDFGState, tasklet: nodes.Tasklet) -> bool:
        """A masked write ``_o = IT(_cond, _val)`` lowers to a masked ``TileStore`` that copies
        ``_val`` straight into the destination (``ConvertTaskletsToTileOps._convert_conditional_write``
        strips the wrapper and re-dispatches the bare ``_o = _val`` through the plain-assign
        path, an AN -> AN edge with no cast). Cast ``_val`` to the destination dtype here when
        it differs, so the store never mixes dtypes."""
        detected = masked_write_operand(tasklet)
        if detected is None:
            return False
        out_conn, val_conn = detected
        sdfg = state.sdfg
        in_edges = {e.dst_conn: e for e in state.in_edges(tasklet) if e.data and e.data.data}
        out_edges = [e for e in state.out_edges(tasklet) if e.data and e.data.data]
        if val_conn not in in_edges or len(out_edges) != 1:
            return False
        val_edge = in_edges[val_conn]
        out_dt = sdfg.arrays[out_edges[0].data.data].dtype
        if sdfg.arrays[val_edge.data.data].dtype == out_dt:
            return False
        self._insert_operand_cast(state, tasklet, val_edge, val_conn, out_dt)
        return True

    def _resolve_assign(self, state: SDFGState, tasklet: nodes.Tasklet) -> bool:
        """A bare copy ``_o = _i`` whose destination dtype differs from the source is an
        implicit assignment cast (e.g. SplitTasklets' ``A = A_plus_B`` storing a promoted
        ``double`` result into an ``int64`` array). Rewrite it to an explicit cast
        ``_o = dace.<dst>(_i)`` so the tile converter lowers it as a cast ``TileUnop``
        instead of a dtype-mismatched store."""
        if len(tasklet.out_connectors) != 1 or len(tasklet.in_connectors) != 1:
            return False
        try:
            tree = ast.parse(tasklet.code.as_string.strip())
        except SyntaxError:
            return False
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
            return False
        assign = tree.body[0]
        if (len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name)
                or not isinstance(assign.value, ast.Name)):
            return False
        out_conn, in_conn = assign.targets[0].id, assign.value.id
        if in_conn not in tasklet.in_connectors or out_conn not in tasklet.out_connectors:
            return False
        sdfg = state.sdfg
        in_edges = [e for e in state.in_edges(tasklet) if e.dst_conn == in_conn and e.data and e.data.data]
        out_edges = [e for e in state.out_edges(tasklet) if e.src_conn == out_conn and e.data and e.data.data]
        if len(in_edges) != 1 or len(out_edges) != 1:
            return False
        in_dt = sdfg.arrays[in_edges[0].data.data].dtype
        out_dt = sdfg.arrays[out_edges[0].data.data].dtype
        if in_dt == out_dt:
            return False
        tasklet.code = CodeBlock(f"{out_conn} = dace.{_cast_name(out_dt)}({in_conn})", language=dace.Language.Python)
        return True

    def _new_scalar(self, sdfg: dace.SDFG, dtype: dtypes.typeclass) -> str:
        name, _ = sdfg.add_scalar("__mixcast",
                                  dtype,
                                  storage=dtypes.StorageType.Register,
                                  transient=True,
                                  find_new_name=True)
        return name

    def _insert_operand_cast(self, state: SDFGState, tasklet: nodes.Tasklet, edge, conn: str,
                             promoted: dtypes.typeclass) -> None:
        """Route operand ``edge`` through ``_co = dace.<promoted>(_ci)`` so ``tasklet``'s
        ``conn`` reads a promoted-dtype transient instead of the narrower source."""
        sdfg = state.sdfg
        tmp = self._new_scalar(sdfg, promoted)
        cast = state.add_tasklet(f"{tasklet.label}_cast_{conn}", {"_ci"}, {"_co"},
                                 f"_co = dace.{_cast_name(promoted)}(_ci)")
        tmp_an = state.add_access(tmp)
        state.add_edge(edge.src, edge.src_conn, cast, "_ci", dace.Memlet.from_memlet(edge.data))
        state.add_edge(cast, "_co", tmp_an, None, dace.Memlet(tmp))
        state.add_edge(tmp_an, None, tasklet, conn, dace.Memlet(tmp))
        state.remove_edge(edge)

    def _insert_output_cast(self, state: SDFGState, tasklet: nodes.Tasklet, edge, conn: str, promoted: dtypes.typeclass,
                            out_dt: dtypes.typeclass) -> None:
        """Compute at ``promoted`` into a fresh transient, then ``_co = dace.<out_dt>(_ci)``
        stores the result into the original destination, casting to its dtype (a downcast
        when it is narrower than ``promoted``, a widening store when it is wider)."""
        sdfg = state.sdfg
        tmp = self._new_scalar(sdfg, promoted)
        cast = state.add_tasklet(f"{tasklet.label}_cast_{conn}", {"_ci"}, {"_co"},
                                 f"_co = dace.{_cast_name(out_dt)}(_ci)")
        tmp_an = state.add_access(tmp)
        state.add_edge(tasklet, conn, tmp_an, None, dace.Memlet(tmp))
        state.add_edge(tmp_an, None, cast, "_ci", dace.Memlet(tmp))
        state.add_edge(cast, "_co", edge.dst, edge.dst_conn, dace.Memlet.from_memlet(edge.data))
        state.remove_edge(edge)


class CastScalarIteLiteralArms(ppl.Pass):
    """Cast a literal/Symbol arm of an ``ITE(...)`` tasklet that STAYS scalar (never becomes a
    ``TileITE``) to the output dtype, by rewriting the tasklet's own Python source.

    Must run AFTER ``ConvertTaskletsToTileOps``: at that point every ``ITE(...)`` tasklet inside
    a tiled body has already been converted to a ``TileITE`` lib node, whose pure/ISA expansion
    casts a Symbol arm to the output dtype on its own (see ``ResolveMixedDtypeBinops.resolve_ite``
    and ``dace.libraries.tileops.nodes.tile_ite``). Any ``ITE(...)`` tasklet still standing is
    therefore provably staying scalar -- the untiled remainder of a branched split, most commonly
    -- and reaches codegen's generic Python-tasklet-to-C++ translation, which has no casting logic
    of its own: ``ITE(cond, 0.0, x)`` with ``x`` a ``dace::float16`` is emitted as
    ``ITE(bool, double, dace::float16)`` and fails to deduce ``std::common_type<double,
    dace::float16>`` (``dace/runtime/include/dace/ITE.h``).

    Running this earlier (inside ``ResolveMixedDtypeBinops.resolve_ite``, before tile conversion)
    would be unsafe: a literal arm destined to become a ``TileITE`` Symbol arm has its raw Python
    text read by TWO different renderers that disagree on its language -- the pure expansion's
    ``pyexpr2cpp`` (Python source) and the CUDA/ISA expansion's ``_isa_codegen.make_ite_tasklet``
    (embeds the text as-is, already-C++) -- so casting it here would double-cast one of the two
    and corrupt the other.
    """

    CATEGORY: str = "Vectorization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        count = 0
        for nested in sdfg.all_sdfgs_recursive():
            for state in nested.all_states():
                for tasklet in list(state.nodes()):
                    if not isinstance(tasklet, nodes.Tasklet):
                        continue
                    if tasklet.code.language != dtypes.Language.Python:
                        continue
                    if self._cast_literal_arms(state, tasklet):
                        count += 1
        return count or None

    def _cast_literal_arms(self, state: SDFGState, tasklet: nodes.Tasklet) -> bool:
        detected = ite_operands(tasklet)
        if detected is None:
            return False
        out_conn, _arm_conns = detected
        sdfg = state.sdfg
        out_edges = [e for e in state.out_edges(tasklet) if e.data and e.data.data]
        if len(out_edges) != 1:
            return False
        out_dt = sdfg.arrays[out_edges[0].data.data].dtype
        try:
            tree = ast.parse(tasklet.code.as_string.strip())
        except SyntaxError:
            return False
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
            return False
        assign = tree.body[0]
        slots = _ite_arm_slots(assign.value)
        if slots is None:
            return False
        in_conns = OrderedSet(tasklet.in_connectors)
        changed = False
        for arm, set_arm in slots:
            if isinstance(arm, ast.Name) and arm.id in in_conns:
                continue
            # Bare (undotted) spelling -- ``float16(0.0)``, not ``dace.float16(0.0)`` -- the one
            # form ``cppunparse``'s tasklet-body translator special-cases to ``dace::<type>(...)``
            # (``_typecast_func_to_cpp``) with no preprocessing step required first.
            cast_call = ast.parse(f"{_cast_name(out_dt)}({ast.unparse(arm)})", mode="eval").body
            set_arm(cast_call)
            changed = True
        if changed:
            tasklet.code = CodeBlock(ast.unparse(assign), language=dace.dtypes.Language.Python)
        return changed
