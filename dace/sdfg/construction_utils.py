# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Graph construction utilities."""

import ast

import dace


def generate_assignment_as_tasklet_in_state(state: dace.SDFGState, lhs: str, rhs: str) -> dace.nodes.Tasklet:
    """Write ``lhs = rhs`` (an interstate-edge assignment) as a Python tasklet into ``state``.

    Every read of a container in ``rhs`` -- ``A[i, j]`` or a bare scalar ``s`` -- becomes an input connector fed by a
    memlet of that access; ``lhs`` names the scalar written through the one output connector. The expression is kept
    as written (Python AST, never a sympy round-trip): sympy distributes numeric factors (``(x - 0.8) / 0.2`` into
    ``5.0 * x - 4.0``), which changes the rounding of the value the interstate edge used to compute.
    """
    arrays = state.sdfg.arrays
    out_name = lhs.strip()
    if out_name not in arrays:
        raise ValueError(f"Assignment target {out_name!r} is not a container of {state.sdfg.label}")
    reads: dict[str, str] = {}  # memlet expression -> input connector

    def connector(data: str, expr: str) -> ast.Name:
        conn = reads.setdefault(expr, f"_in_{data}_{len(reads)}")
        return ast.Name(id=conn, ctx=ast.Load())

    class _ReadsToConnectors(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            if isinstance(node.value, ast.Name) and node.value.id in arrays:
                return connector(node.value.id, ast.unparse(node))
            return self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id in arrays:
                return connector(node.id, f"{node.id}[0]")
            return node

    body = _ReadsToConnectors().visit(ast.parse(rhs.strip(), mode="eval")).body
    out_conn = f"_out_{out_name}"
    tasklet = state.add_tasklet(
        name=f"assign_{out_name}",
        inputs=set(reads.values()),
        outputs={out_conn},
        code=f"{out_conn} = {ast.unparse(body)}",
    )
    accesses: dict[str, dace.nodes.AccessNode] = {}
    for expr, conn in reads.items():
        memlet = dace.Memlet(expr)
        if memlet.data not in accesses:
            accesses[memlet.data] = state.add_access(memlet.data)
        state.add_edge(accesses[memlet.data], None, tasklet, conn, memlet)
    state.add_edge(tasklet, out_conn, state.add_access(out_name), None, dace.Memlet(f"{out_name}[0]"))
    return tasklet
