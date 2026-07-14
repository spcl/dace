# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fuse a multiply feeding an add into a single fused multiply-add.

``t = a * b ; d = t + c`` (the ``t`` a single-use transient) becomes ``d = fma(a, b, c)``,
using the :class:`dace.symbolic.fma` function so the downstream tile-op lowering
(:class:`~dace.transformation.passes.vectorization.convert_tasklets_to_tile_ops.ConvertTaskletsToTileOps`)
emits a single :class:`~dace.libraries.tileops.nodes.tile_fma.TileFMA` -- a native FMA on every
ISA that has one (``__hfma2`` / ``_mm*_fmadd`` / ``vfmaq`` / ``svmla`` / ``std::fma``), and
multiply-then-add where it does not. Runs BEFORE the tasklets become ``TileBinop`` nodes.

A fused multiply-add rounds once where the separate ``*`` then ``+`` rounds twice, so the result
differs from a plain ``a*b + c`` (and from a NumPy reference) by up to one ULP. The pass is
therefore OFF by default and enabled only through ``VectorizeConfig.fuse_multiply_add`` -- a caller
opts into the fused numerics for the throughput win.
"""
import ast
from typing import List, Optional, Tuple

import dace
from dace import properties
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl


def _binop_tasklet(tasklet: nodes.Tasklet, op: str) -> Optional[Tuple[str, List[str]]]:
    """If ``tasklet`` is a two-input ``__out = __a <op> __b`` body, return ``(out_conn, [a, b])``.

    Matches the parenthesised (``__out = (__a + __b)``) and bare forms the frontend / tasklet
    splitter emit. Refuses anything else (a single input, a call, a compound expression).
    """
    if len(tasklet.out_connectors) != 1 or len(tasklet.in_connectors) != 2:
        return None
    if tasklet.language is not dace.dtypes.Language.Python:
        return None
    if len(tasklet.code.code) != 1 or not isinstance(tasklet.code.code[0], ast.Assign):
        return None
    node = tasklet.code.code[0]
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        return None
    out_conn = node.targets[0].id
    if out_conn != next(iter(tasklet.out_connectors)):
        return None
    rhs = node.value
    pyop = ast.Mult if op == '*' else ast.Add
    if not (isinstance(rhs, ast.BinOp) and isinstance(rhs.op, pyop)):
        return None
    if not (isinstance(rhs.left, ast.Name) and isinstance(rhs.right, ast.Name)):
        return None
    a, b = rhs.left.id, rhs.right.id
    in_conns = set(tasklet.in_connectors)
    if a not in in_conns or b not in in_conns or a == b:
        return None
    return out_conn, [a, b]


@properties.make_properties
class FuseMultiplyAdd(ppl.Pass):
    """Fuse ``t = a*b ; d = t + c`` (single-use ``t``) into ``d = fma(a, b, c)``.

    Off unless explicitly enabled (the fused single-rounding changes results by up to one ULP);
    the vectorizer runs it -- gated on ``VectorizeConfig.fuse_multiply_add`` -- before the tasklets
    are lowered to tile ops."""

    CATEGORY: str = 'Vectorization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Nodes | ppl.Modifies.Edges))

    def _data_used_elsewhere(self, sdfg: dace.SDFG, state: SDFGState, name: str) -> bool:
        """True if ``name`` is referenced by any access node other than in ``state`` (a
        cross-state / cross-scope reuse that would make removing the intermediate unsound)."""
        for s in sdfg.states():
            for n in s.nodes():
                if isinstance(n, nodes.AccessNode) and n.data == name and s is not state:
                    return True
        return False

    def _fuse_in_state(self, sdfg: dace.SDFG, state: SDFGState) -> int:
        fused = 0
        for mul in [n for n in state.nodes() if isinstance(n, nodes.Tasklet)]:
            m = _binop_tasklet(mul, '*')
            if m is None:
                continue
            mul_out_conn, mul_ins = m
            # The product must flow into a single-use transient scalar access node.
            out_edges = [e for e in state.out_edges(mul) if e.src_conn == mul_out_conn]
            if len(out_edges) != 1:
                continue
            prod = out_edges[0].dst
            if not isinstance(prod, nodes.AccessNode):
                continue
            desc = sdfg.arrays.get(prod.data)
            if desc is None or not desc.transient:
                continue
            if state.in_degree(prod) != 1 or state.out_degree(prod) != 1:
                continue
            if self._data_used_elsewhere(sdfg, state, prod.data):
                continue
            add_edge = state.out_edges(prod)[0]
            add = add_edge.dst
            if not isinstance(add, nodes.Tasklet):
                continue
            a = _binop_tasklet(add, '+')
            if a is None:
                continue
            add_out_conn, add_ins = a
            prod_conn = add_edge.dst_conn  # the add input fed by the product
            if prod_conn not in add_ins:
                continue
            addend_conn = add_ins[0] if add_ins[1] == prod_conn else add_ins[1]
            if addend_conn == prod_conn:
                continue
            self._rewrite(sdfg, state, mul, mul_ins, prod, add, add_out_conn, addend_conn)
            fused += 1
        return fused

    def _rewrite(self, sdfg, state, mul, mul_ins, prod, add, add_out_conn, addend_conn) -> None:
        """Replace the ``mul -> prod -> add`` chain with one ``fma`` tasklet."""
        # Source edges to preserve: the two multiplicands (into ``mul``) and the addend (into ``add``).
        a_edge = next(e for e in state.in_edges(mul) if e.dst_conn == mul_ins[0])
        b_edge = next(e for e in state.in_edges(mul) if e.dst_conn == mul_ins[1])
        c_edge = next(e for e in state.in_edges(add) if e.dst_conn == addend_conn)
        out_edge = next(e for e in state.out_edges(add) if e.src_conn == add_out_conn)

        fma = state.add_tasklet(name='fma',
                                inputs={'__in1', '__in2', '__in3'},
                                outputs={'__out'},
                                code='__out = fma(__in1, __in2, __in3)')
        state.add_edge(a_edge.src, a_edge.src_conn, fma, '__in1', dace.Memlet.from_memlet(a_edge.data))
        state.add_edge(b_edge.src, b_edge.src_conn, fma, '__in2', dace.Memlet.from_memlet(b_edge.data))
        state.add_edge(c_edge.src, c_edge.src_conn, fma, '__in3', dace.Memlet.from_memlet(c_edge.data))
        state.add_edge(fma, '__out', out_edge.dst, out_edge.dst_conn, dace.Memlet.from_memlet(out_edge.data))

        # Drop the fused nodes + the now-orphaned intermediate transient.
        state.remove_node(mul)
        state.remove_node(add)
        state.remove_node(prod)
        if not any(isinstance(n, nodes.AccessNode) and n.data == prod.data for s in sdfg.states() for n in s.nodes()):
            try:
                sdfg.remove_data(prod.data, validate=False)
            except (KeyError, ValueError):
                pass

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        total = 0
        for sd in sdfg.all_sdfgs_recursive():
            for state in sd.states():
                total += self._fuse_in_state(sd, state)
        return total or None
