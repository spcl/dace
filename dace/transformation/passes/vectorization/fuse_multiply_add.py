# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fuse a multiply feeding an add into a single fused multiply-add.

``t = a * b ; d = t + c`` (single-use transient ``t``) becomes ``d = fma(a, b, c)``, so
tile-op lowering emits a native FMA per ISA instead of multiply-then-add. Runs before
tasklets become ``TileBinop`` nodes.

FMA rounds once vs. two roundings for plain ``a*b + c``, differing by up to 1 ULP from
NumPy. OFF by default; enabled via ``VectorizeConfig.fuse_multiply_add``.
"""
import ast

from typing import Any

import dace
from dace import properties
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation.passes.vectorization.utils.tasklets import is_vectorizable_tasklet
from dace.transformation import pass_pipeline as ppl
from dace.ordered import OrderedSet


def _binop_tasklet(tasklet: nodes.Tasklet, op: str) -> tuple[str, list[str]] | None:
    # Matches a two-input "__out = __a <op> __b" body; returns (out_conn, [a, b]).
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

    Off by default (1-ULP result change); gated on ``VectorizeConfig.fuse_multiply_add``,
    runs before tasklets lower to tile ops."""

    CATEGORY: str = 'Vectorization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Nodes | ppl.Modifies.Edges))

    def _data_used_elsewhere(self, sdfg: dace.SDFG, state: SDFGState, name: str) -> bool:
        # True if a cross-state access would make removing the intermediate unsound.
        for s in sdfg.states():
            for n in s.nodes():
                if isinstance(n, nodes.AccessNode) and n.data == name and s is not state:
                    return True
        return False

    def _fuse_in_state(self, sdfg: dace.SDFG, state: SDFGState) -> int:
        fused = 0
        for mul in [n for n in state.nodes() if isinstance(n, nodes.Tasklet) and is_vectorizable_tasklet(state, n)]:
            m = _binop_tasklet(mul, '*')
            if m is None:
                continue
            mul_out_conn, mul_ins = m
            # Product must flow into a single-use transient scalar access node.
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

    def _rewrite(self, sdfg: dace.SDFG, state: SDFGState, mul: nodes.Tasklet, mul_ins: list[str],
                 prod: nodes.AccessNode, add: nodes.Tasklet, add_out_conn: str, addend_conn: str) -> None:
        # Replace the mul -> prod -> add chain with one fma tasklet.
        a_edge = next(e for e in state.in_edges(mul) if e.dst_conn == mul_ins[0])
        b_edge = next(e for e in state.in_edges(mul) if e.dst_conn == mul_ins[1])
        c_edge = next(e for e in state.in_edges(add) if e.dst_conn == addend_conn)
        out_edge = next(e for e in state.out_edges(add) if e.src_conn == add_out_conn)

        fma = state.add_tasklet(name='fma',
                                inputs=OrderedSet(('__in1', '__in2', '__in3')),
                                outputs={'__out'},
                                code='__out = fma(__in1, __in2, __in3)')
        state.add_edge(a_edge.src, a_edge.src_conn, fma, '__in1', dace.Memlet.from_memlet(a_edge.data))
        state.add_edge(b_edge.src, b_edge.src_conn, fma, '__in2', dace.Memlet.from_memlet(b_edge.data))
        state.add_edge(c_edge.src, c_edge.src_conn, fma, '__in3', dace.Memlet.from_memlet(c_edge.data))
        state.add_edge(fma, '__out', out_edge.dst, out_edge.dst_conn, dace.Memlet.from_memlet(out_edge.data))

        # Drop fused nodes + the now-orphaned intermediate transient.
        state.remove_node(mul)
        state.remove_node(add)
        state.remove_node(prod)
        if not any(isinstance(n, nodes.AccessNode) and n.data == prod.data for s in sdfg.states() for n in s.nodes()):
            try:
                sdfg.remove_data(prod.data, validate=False)
            except (KeyError, ValueError):
                pass

    def apply_pass(self, sdfg: dace.SDFG, _: dict[str, Any]) -> int | None:
        total = 0
        for sd in sdfg.all_sdfgs_recursive():
            for state in sd.states():
                total += self._fuse_in_state(sd, state)
        return total or None
