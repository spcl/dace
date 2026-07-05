# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``NormalizeMapReduction`` — normalize a masked map reduction to a SINGLE-STATE
elementwise-select body + an unconditional map-exit WCR, so it tiles.

The DaCe frontend emits a masked scalar reduction ``if cond: acc <op>= x`` as a
map-body NestedSDFG whose control flow is a ``ConditionalBlock`` (condition
assigned on the incoming interstate edge, ``__tmp0 = <cond expr>``) wrapping a
state that writes the accumulator via an in-body WCR (``x ─[wcr:op]→ oc``, ``oc``
a write-only output connector), with a *plain* ``NestedSDFG → MapExit`` edge.

:class:`~dace.transformation.passes.normalize_nested_reduction.NormalizeNestedReduction`
(NNR, used by canonicalize) lifts that to a seeded body-local accumulator +
map-exit WCR -- but it leaves the body **multi-state** (a ``_nnr_seed`` +
``_nnr_copyback`` around the conditional) AND inserts a per-iteration
``_nnr_out`` AccessNode at the map level. Both break tiling: the multi-state
conditional body is not inlinable (``InlineMultistateSDFG`` refuses it), so
``NestInnermostMapBodyIntoNSDFG`` double-wraps it and the walker never widens
the per-lane accesses -- the strided tile map then reads a single scalar ``x``
per tile, summing every W-th element.

This pass (vectorize front block only; canonicalize keeps NNR) rebuilds the body
as ONE state computing the elementwise masked **addend**:

    _cond   = <cond expr>                       (lowers to TileBinop compare(s))
    oc      = ITE(_cond, x, identity(op))       (single ITE tasklet -> TileITE)

with the accumulation moved OUT of the body to an unconditional map-exit WCR
sourced by an AccessNode (the "Test-C" shape codegen accumulates across tiles).
A single-state body with a plain output inlines like the working masked-ITE
*store* -- so the map body collapses to one NestedSDFG, the walker widens it, and
the reduction tiles for every op (``+``/``*``/``min``/``max`` via the op
identity). Multiple accumulators sharing the condition (``if cond: s += x; n +=
1``) are each rewritten with the shared ``_cond``.
"""
import ast
import copy
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy

from dace import SDFG, SDFGState, data, dtypes
from dace.frontend.python.astutils import ASTFindReplace, unparse
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock
from dace.transformation import pass_pipeline as ppl, transformation

#: WCR reduction op (``+``/``*``/``min``/``max``); ``-`` accumulates like ``+``.
_WCR_OP = {"+": "+", "-": "+", "*": "*", "min": "min", "max": "max"}


def _op_from_wcr(wcr: str) -> Optional[str]:
    """Reduction op for a WCR lambda string, or ``None`` if unrecognized."""
    try:
        tree = ast.parse(wcr.strip(), mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(tree, ast.Lambda):
        return None
    body = tree.body
    if isinstance(body, ast.BinOp):
        return _WCR_OP.get({ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}.get(type(body.op)))
    if isinstance(body, ast.Call) and isinstance(body.func, ast.Name) and body.func.id in ("min", "max"):
        return _WCR_OP.get(body.func.id)
    return None


def _identity_literal(op: str, dtype: dtypes.typeclass) -> str:
    """C-Python literal for ``op``'s identity at ``dtype`` (as a tasklet RHS string)."""
    is_int = numpy.issubdtype(dtype.type, numpy.integer)
    if op == "+":
        return "0" if is_int else "0.0"
    if op == "*":
        return "1" if is_int else "1.0"
    if op == "min":
        return str(int(numpy.iinfo(dtype.type).max)) if is_int else "float('inf')"
    if op == "max":
        return str(int(numpy.iinfo(dtype.type).min)) if is_int else "float('-inf')"
    raise ValueError(f"unknown op {op!r}")


def _referenced_arrays(expr: str, arrays) -> List[str]:
    """The array names in ``arrays`` referenced by the Python expression ``expr``."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return []
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return [a for a in arrays if a in names]


class _MaskedReduction:
    """The pieces of a recognized masked reduction inside one body NSDFG."""

    def __init__(self, cond_expr: str, cond_sym: Optional[str], writes: List[Tuple[str, str, str]]):
        #: The condition expression string (e.g. ``(__tmp_21_11_r > 0)``).
        self.cond_expr = cond_expr
        #: The interstate symbol the condition was assigned to (``__tmp0``), pruned
        #: after the rewrite; ``None`` if the condition was a direct expression.
        self.cond_sym = cond_sym
        #: One ``(kind, value, out_conn, op)`` per accumulator: ``kind`` is
        #: ``"array"`` (``value`` is a body-input array read in the then-branch) or
        #: ``"const"`` (``value`` is a literal expression, e.g. ``count += 1``);
        #: ``out_conn`` is the write-only output; ``op`` is the reduction op.
        self.writes = writes


@transformation.explicit_cf_compatible
class NormalizeMapReduction(ppl.Pass):
    """Rewrite a masked map reduction to a single-state select body + map-exit WCR."""

    CATEGORY: str = "Vectorization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> Set:
        return set()

    def _recognize(self, body: SDFG) -> Optional[_MaskedReduction]:
        """Recognize the frontend masked-reduction shape in body NSDFG ``body``:
        exactly one ConditionalBlock whose single non-else branch contains
        write-only-output WCR writes, condition assigned on the incoming edge.
        Returns the extracted pieces or ``None``.
        """
        cblocks = [b for b in body.all_control_flow_blocks() if isinstance(b, ConditionalBlock)]
        if len(cblocks) != 1:
            return None
        cb = cblocks[0]
        # The condition is a symbol assigned on the interstate edge into ``cb``.
        branches = [(c, r) for c, r in cb.branches]
        guarded = [(c, r) for c, r in branches if c is not None]
        if len(guarded) != 1 or any(c is None for c, _ in branches) and len(branches) > 2:
            return None
        cond_code, region = guarded[0]
        cond_sym = cond_code.as_string.strip()
        cond_expr = None
        cond_sym_defined = None
        for e in body.all_interstate_edges():
            if cond_sym in e.data.assignments:
                cond_expr = e.data.assignments[cond_sym]
                cond_sym_defined = cond_sym
                break
        if cond_expr is None:
            # The condition is already a direct expression (no symbol indirection).
            cond_expr = cond_sym
        # Collect the write-only-output WCR writes in the guarded region.
        writes: List[Tuple[str, str, str]] = []
        for st in region.all_states():
            for e in st.edges():
                if (e.data is None or e.data.wcr is None or not isinstance(e.dst, nodes.AccessNode)
                        or e.data.subset is None or e.data.subset.num_elements() != 1):
                    continue
                oc = e.dst.data
                if oc in body.arrays and oc not in body.arrays.get(oc).free_symbols and st.out_degree(e.dst) == 0:
                    op = _op_from_wcr(e.data.wcr)
                    if op is None:
                        return None
                    # The addend feeding the WCR write's source tasklet (array or const).
                    addend = self._addend_value(st, e.src)
                    if addend is None:
                        return None
                    kind, val = addend
                    writes.append((kind, val, oc, op))
        if not writes:
            return None
        return _MaskedReduction(cond_expr, cond_sym_defined, writes)

    def _addend_value(self, state: SDFGState, wcr_src) -> Optional[Tuple[str, str]]:
        """The addend written by the WCR: ``("array", name)`` or ``("const", expr)``.

        The frontend then-branch is either ``val_array → assign(__out=__inp) ─[wcr]→
        oc`` (an array addend, e.g. ``tmp += data[j]``) or a 0-input constant tasklet
        ``__out = <literal> ─[wcr]→ oc`` (a const addend, e.g. ``count += 1``).
        Returns ``None`` on any other source shape.
        """
        if isinstance(wcr_src, nodes.AccessNode):
            return ("array", wcr_src.data)
        if not isinstance(wcr_src, nodes.Tasklet):
            return None
        ins = [e for e in state.in_edges(wcr_src) if e.data is not None and e.data.data is not None]
        if len(ins) == 1:
            return ("array", ins[0].data.data)
        if len(ins) == 0:
            # Constant addend: the RHS of the ``__out = <rhs>`` assignment tasklet.
            try:
                tree = ast.parse(wcr_src.code.as_string.strip())
            except SyntaxError:
                return None
            for stmt in tree.body:
                if isinstance(stmt, ast.Assign):
                    return ("const", unparse(stmt.value))
        return None

    def _rebuild_body(self, body: SDFG, red: _MaskedReduction) -> None:
        """Replace ``body``'s control flow with ONE state computing the masked addends."""
        # New transients: the condition value (bool) + one identity source per distinct op.
        cond_name, _ = body.add_scalar("_nmr_cond", dtypes.bool_, transient=True, find_new_name=True)
        new = SDFGState("_nmr_body", body)
        # Read every array referenced by the condition + the addend values.
        cond_inputs = sorted(set(_referenced_arrays(red.cond_expr, body.arrays)))
        reads: Dict[str, nodes.AccessNode] = {a: new.add_read(a) for a in cond_inputs}
        # A tasklet connector may not share a name with an array; alias each condition
        # input to a distinct connector and substitute it into the expression (an
        # AST-aware name rewrite, not a regex, so subscripts/attributes are safe).
        alias = {a: f"_ci{k}" for k, a in enumerate(cond_inputs)}
        cond_body_expr = unparse(ASTFindReplace(dict(alias)).visit(ast.parse(red.cond_expr, mode="eval")))
        ct = new.add_tasklet("_nmr_cond", set(alias.values()), {"_c"}, f"_c = {cond_body_expr}")
        for a, cn in alias.items():
            new.add_edge(reads[a], None, ct, cn, Memlet(data=a, subset="0"))
        cond_an = new.add_access(cond_name)
        new.add_edge(ct, "_c", cond_an, None, Memlet(data=cond_name, subset="0"))
        # One ITE per accumulator: ``oc = ITE(_c, addend, identity(op))``.
        for i, (kind, val, oc, op) in enumerate(red.writes):
            odtype = body.arrays[oc].dtype
            ident_lit = _identity_literal(op, odtype)
            ident_name, _ = body.add_scalar(f"_nmr_ident{i}", odtype, transient=True, find_new_name=True)
            it = new.add_tasklet(f"_nmr_ident{i}", set(), {"_e"}, f"_e = {ident_lit}")
            ident_an = new.add_access(ident_name)
            new.add_edge(it, "_e", ident_an, None, Memlet(data=ident_name, subset="0"))
            ite = new.add_tasklet(f"_nmr_ite{i}", {"_c", "_t", "_e"}, {"_o"}, "_o = ITE(_c, _t, _e)")
            new.add_edge(cond_an, None, ite, "_c", Memlet(data=cond_name, subset="0"))
            if kind == "array":
                if val not in reads:
                    reads[val] = new.add_read(val)
                new.add_edge(reads[val], None, ite, "_t", Memlet(data=val, subset="0"))
            else:  # const addend (e.g. ``count += 1``): materialise the literal
                then_name, _ = body.add_scalar(f"_nmr_then{i}", odtype, transient=True, find_new_name=True)
                tt = new.add_tasklet(f"_nmr_then{i}", set(), {"_tv"}, f"_tv = {val}")
                then_an = new.add_access(then_name)
                new.add_edge(tt, "_tv", then_an, None, Memlet(data=then_name, subset="0"))
                new.add_edge(then_an, None, ite, "_t", Memlet(data=then_name, subset="0"))
            new.add_edge(ident_an, None, ite, "_e", Memlet(data=ident_name, subset="0"))
            new.add_edge(ite, "_o", new.add_write(oc), None, Memlet(data=oc, subset="0"))
        # Swap the CFG: drop every existing block, install ``new`` as the sole start block.
        for blk in list(body.nodes()):
            body.remove_node(blk)
        body.add_node(new, is_start_block=True)
        # The condition symbol was defined only on the (now-removed) interstate edge;
        # drop it so it is not a dangling free symbol on the body NSDFG.
        if red.cond_sym is not None and red.cond_sym in body.symbols:
            body.remove_symbol(red.cond_sym)
        body.reset_cfg_list()

    def _lift_to_map_exit(self, state: SDFGState, nsdfg: nodes.NestedSDFG, red: _MaskedReduction) -> None:
        """Put each accumulator's reduction on the ``NestedSDFG → AccessNode ─[wcr]→
        MapExit → acc`` chain (Test-C: AccessNode-sourced so codegen accumulates)."""
        wcr_lambdas = {
            "+": "lambda x, y: (x + y)",
            "*": "lambda x, y: (x * y)",
            "min": "lambda x, y: min(x, y)",
            "max": "lambda x, y: max(x, y)"
        }
        for _kind, _val, oc, op in red.writes:
            out_edge = next((oe for oe in state.out_edges(nsdfg) if oe.src_conn == oc), None)
            if out_edge is None or not isinstance(out_edge.dst, nodes.MapExit):
                continue
            wcr = wcr_lambdas[op]
            desc = copy.deepcopy(state.sdfg.arrays[out_edge.data.data])
            desc.transient = True
            priv = state.sdfg.add_datadesc(f"_nmr_out_{oc}", desc, find_new_name=True)
            priv_node = state.add_access(priv)
            state.add_edge(nsdfg, oc, priv_node, None,
                           Memlet(data=priv, subset="0" if isinstance(desc, data.Scalar) else None))
            for e in state.memlet_path(out_edge):
                e.data.wcr = wcr
            state.add_edge(priv_node, None, out_edge.dst, out_edge.dst_conn, copy.deepcopy(out_edge.data))
            state.remove_edge(out_edge)

    def _apply(self, sdfg: SDFG) -> int:
        total = 0
        for sd in sdfg.all_sdfgs_recursive():
            for state in sd.all_states():
                for nsdfg in [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG)]:
                    if not isinstance(state.entry_node(nsdfg), nodes.MapEntry):
                        continue
                    red = self._recognize(nsdfg.sdfg)
                    if red is None:
                        continue
                    self._rebuild_body(nsdfg.sdfg, red)
                    self._lift_to_map_exit(state, nsdfg, red)
                    total += 1
        return total

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        """Normalize every masked map reduction to the single-state select form.

        :param sdfg: The SDFG to normalize (in place).
        :param pipeline_results: Prior pass results (unused).
        :returns: ``None`` if nothing changed; else a single-entry summary dict.
        """
        n = self._apply(sdfg)
        if n == 0:
            return None
        sdfg.reset_cfg_list()
        sdfg.validate()
        return {"normalized_map_reductions": {str(n)}}
