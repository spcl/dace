# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Symbolic execution of a control-flow region over z3 bitvectors.

:mod:`~dace.transformation.passes.analysis.smt_body` runs ONE tasklet body. A loop body after canonicalization is
a region: tasklets in states, scalars passed between them through access nodes, symbols assigned on interstate
edges, branches as :class:`~dace.sdfg.state.ConditionalBlock` and short constant-trip loops. This module runs such
a region in program order, with C semantics on the containers' own types, and returns the value of every scalar
and symbol it assigns.

Every value is a ``width``-bit term holding the two's-complement extension of the container's type, and every
write converts to the type written (:func:`~dace.transformation.passes.analysis.smt_body.fit`), as the generated
code does. Three kinds of input stand in the term:

- names in the caller's environment, such as a carried state;
- a read of an array element, one variable per distinct subset (``data[_loop_it_0]``), which is how a step's
  per-iteration data enters;
- any other symbol or scalar the region reads before writing, one variable per name: loop-invariant parameters
  such as a CRC polynomial.

Anything else, from a map to a write into an array, makes the region unsupported and the encoder returns
``None``. Refusing is always a correct answer for an oracle.
"""

import ast
from typing import Any

import numpy

from dace import data, dtypes
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.sdfg import SDFG, InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation.passes.analysis import loop_analysis, smt_body

#: Longest constant-trip loop the encoder unrolls.
UNROLL_LIMIT = 64


def value_type(sdfg: SDFG, name: str) -> tuple[int, bool] | None:
    """``(bits, signed)`` of the integer container or symbol ``name``, else ``None``."""
    dtype = sdfg.arrays[name].dtype if name in sdfg.arrays else sdfg.symbols.get(name)
    if dtype is None or not numpy.issubdtype(dtype.type, numpy.integer):
        return None
    return dtype.bytes * 8, bool(numpy.issubdtype(dtype.type, numpy.signedinteger))


def is_scalar(desc: data.Data) -> bool:
    return isinstance(desc, data.Scalar) or (isinstance(desc, data.Array) and tuple(desc.shape) == (1,))


class RegionEncoder:
    """Runs regions of ``sdfg`` symbolically; see the module docstring.

    :ivar inputs: every input the encoder created for an array read or a parameter, by its key, as a term of the
                  input's type.
    :ivar raw: the free variable under each input.
    """

    def __init__(self, sdfg: SDFG, iterate: str, width: int = smt_body.DEFAULT_WIDTH):
        self.sdfg = sdfg
        self.iterate = iterate
        self.width = width
        self.inputs: dict[str, Any] = {}
        self.raw: dict[str, Any] = {}

    def variable(self, key: str, name: str) -> Any | None:
        """The input variable for ``key``, of the type of container or symbol ``name``."""
        kind = value_type(self.sdfg, name)
        if kind is None:
            return None
        if key not in self.inputs:
            self.raw[key] = smt_body.bitvec(key, self.width)
            self.inputs[key] = smt_body.fit(self.raw[key], kind[0], kind[1], self.width)
        return self.inputs[key]

    def data_variables(self) -> list[Any]:
        """The free variables of the array element reads, which vary per iteration; the rest are parameters."""
        return [variable for key, variable in self.raw.items() if "[" in key]

    def is_value(self, name: str) -> bool:
        """Whether ``name`` is an integer symbol or scalar, as opposed to an array or a function."""
        desc = self.sdfg.arrays.get(name)
        return value_type(self.sdfg, name) is not None and (desc is None or is_scalar(desc))

    def is_input_index(self, index: str, env: dict[str, Any]) -> bool:
        """Whether an element read at ``index`` is an input: its index names only the loop variable and values
        the region does not compute. A read indexed by a computed value (``table[state]``) is a function of that
        value, not a free variable, and the encoder does not model it."""
        used = {node.id for node in ast.walk(ast.parse(index)) if isinstance(node, ast.Name)}
        return not (used & (set(env) - {self.iterate}))

    def bind_names(self, tree: ast.AST, env: dict[str, Any]) -> dict[str, Any] | None:
        """``env`` extended with a variable per subscripted read and per name ``tree`` reads that ``env`` lacks."""
        env = dict(env)
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                if not self.is_input_index(ast.unparse(node.slice), env):
                    return None
                key = ast.unparse(node)
                term = self.variable(key, node.value.id)
                if term is None:
                    return None
                env[key] = term
            elif isinstance(node, ast.Name) and node.id not in env and self.is_value(node.id):
                env[node.id] = self.variable(node.id, node.id)
        return env

    def expression(self, text: str, env: dict[str, Any]) -> Any | None:
        """The term of the Python expression ``text``; subscripted reads become input variables."""
        tree = ast.parse(text, mode="eval")
        bound = self.bind_names(tree, env)
        if bound is None:
            return None
        flat = SubscriptsToNames().visit(tree)
        return smt_body.encode_expr(flat.body, bound, self.width, signed=True)

    def assign(self, env: dict[str, Any], name: str, term: Any) -> dict[str, Any] | None:
        kind = value_type(self.sdfg, name)
        if kind is None:
            return None
        return {**env, name: smt_body.fit(term, kind[0], kind[1], self.width)}

    def edge(self, env: dict[str, Any], edge: InterstateEdge) -> dict[str, Any] | None:
        """``env`` after ``edge``'s assignments, run in order as the generated code runs them."""
        for name, rhs in edge.assignments.items():
            term = self.expression(rhs, env)
            env = None if term is None else self.assign(env, name, term)
            if env is None:
                return None
        return env

    def read(self, state: SDFGState, edge, env: dict[str, Any]) -> Any | None:
        """The value an edge into a tasklet carries."""
        name = edge.data.data
        desc = self.sdfg.arrays.get(name)
        if desc is None or edge.data.is_empty():
            return None
        if is_scalar(desc):
            return env[name] if name in env else self.variable(name, name)
        if not self.is_input_index(str(edge.data.subset), env):
            return None
        return self.variable(f"{name}[{edge.data.subset}]", name)

    def state(self, state: SDFGState, env: dict[str, Any]) -> dict[str, Any] | None:
        """``env`` after the tasklets and scalar copies of ``state``, in topological order."""
        for node in sdutil.dfs_topological_sort(state):
            if isinstance(node, nodes.AccessNode):
                if not is_scalar(node.desc(self.sdfg)) and state.in_degree(node) > 0:
                    return None
                for edge in state.in_edges(node):
                    if isinstance(edge.src, nodes.AccessNode):
                        value = self.read(state, edge, env)
                        env = None if value is None else self.assign(env, node.data, value)
                        if env is None:
                            return None
                continue
            if not isinstance(node, nodes.Tasklet) or node.code.language is not dtypes.Language.Python:
                return None
            env = self.tasklet(state, node, env)
            if env is None:
                return None
        return env

    def tasklet(self, state: SDFGState, tasklet: nodes.Tasklet, env: dict[str, Any]) -> dict[str, Any] | None:
        inner: dict[str, Any] = {}
        for edge in state.in_edges(tasklet):
            value = self.read(state, edge, env)
            if value is None:
                return None
            inner[edge.dst_conn] = value
        code = tasklet.code.as_string
        bound = self.bind_names(ast.parse(code), inner)
        after = None if bound is None else smt_body.encode_body(code, bound, self.width, signed=True)
        if after is None:
            return None
        for edge in state.out_edges(tasklet):
            if edge.src_conn not in after:
                return None
            env = self.assign(env, edge.data.data, after[edge.src_conn])
            if env is None:
                return None
        return env

    def block(self, block, env: dict[str, Any]) -> dict[str, Any] | None:
        if isinstance(block, SDFGState):
            return self.state(block, env)
        if isinstance(block, ConditionalBlock):
            return self.conditional(block, env)
        if isinstance(block, LoopRegion):
            return self.unrolled(block, env)
        return None

    def conditional(self, block: ConditionalBlock, env: dict[str, Any]) -> dict[str, Any] | None:
        """``env`` after the branch the conditions pick, as one term per name."""
        branches = []
        for condition, branch in block.branches:
            test = None if condition is None else self.expression(condition.as_string, env)
            after = self.region(branch, env)
            if after is None or (condition is not None and test is None):
                return None
            branches.append((test, after))
        if not branches or branches[-1][0] is not None:
            branches.append((None, env))
        merged = branches[-1][1]
        for test, after in reversed(branches[:-1]):
            merged = smt_body.merge_branches(smt_body.truth(test), after, merged)
        return merged

    def unrolled(self, loop: LoopRegion, env: dict[str, Any]) -> dict[str, Any] | None:
        """``env`` after a loop with constant bounds and stride, run iteration by iteration."""
        start, end = loop_analysis.get_init_assignment(loop), loop_analysis.get_loop_end(loop)
        stride = loop_analysis.get_loop_stride(loop)
        if start is None or end is None or stride is None:
            return None
        try:
            first, last, step = int(start), int(end), int(stride)
        except TypeError:
            return None
        trips = range(first, last + 1, step) if step > 0 else range(first, last - 1, step)
        if step == 0 or len(trips) > UNROLL_LIMIT:
            return None
        for value in trips:
            # a loop variable is a scope symbol: the SDFG declares no type for it
            env = self.region(loop, {**env, loop.loop_variable: smt_body.constant(value, self.width)})
            if env is None:
                return None
        return env

    def region(self, region: ControlFlowRegion, env: dict[str, Any]) -> dict[str, Any] | None:
        """``env`` after ``region``, whose blocks must run one after the other on unconditional edges."""
        block = region.start_block
        while True:
            env = self.block(block, env)
            if env is None:
                return None
            edges = region.out_edges(block)
            if not edges:
                return env
            if len(edges) != 1 or not edges[0].data.is_unconditional():
                return None
            env = self.edge(env, edges[0].data)
            if env is None:
                return None
            block = edges[0].dst


class SubscriptsToNames(ast.NodeTransformer):
    """Replaces every subscripted read by a name equal to its source text, the key the encoder binds it under."""

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        return ast.copy_location(ast.Name(id=ast.unparse(node), ctx=ast.Load()), node)
