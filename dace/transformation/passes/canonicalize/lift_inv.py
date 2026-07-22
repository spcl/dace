# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a ``solve(A, I)`` matrix inversion to a single ``Inv`` library node.

The dedicated way to invert a matrix is ``numpy.linalg.inv``, which the Python
frontend already lowers straight to a
:class:`~dace.libraries.linalg.nodes.inv.Inv` node -- so an inverse written that
way needs no lifting. This pass targets the OTHER idiomatic spelling of a matrix
inverse: solving the system against an identity right-hand side::

    out = numpy.linalg.solve(A, numpy.eye(N))     # == A^-1

which the frontend lowers to a :class:`~dace.libraries.linalg.nodes.solve.Solve`
node (``_ain`` <- A, ``_bin`` <- the identity, ``_bout`` -> out) whose RHS operand
is a freshly-built identity matrix (``numpy.eye`` / ``numpy.identity``: a square
``[0:N, 0:N]`` mapped tasklet computing ``val = 1 if i == j else 0``). Because
``solve(A, I)`` is exactly ``A^-1``, this pass recognises the pair -- the ``Solve``
node plus its identity-construction map -- and replaces it with one ``Inv`` node
(``getrf`` + ``getri``, the direct inverse that never materialises an identity
RHS). ``Inv`` and ``Solve(A, I)`` both compute ``A^-1`` from the same LU
factorisation, so the rewrite is value-preserving to floating-point tolerance.

What is detected (narrow and conservative -- a no-op on any deviation):

* a ``Solve`` node with the frontend's three connectors (``_ain``, ``_bin``,
  ``_bout``), ``overwrite`` off, whose operands ``A`` / ``out`` are distinct
  square 2-D arrays of matching size and base dtype;
* whose ``_bin`` is a TRANSIENT square array produced, in the same state, ONLY by
  an identity map: two loop parameters over ``[0:n, 0:n]``, no map inputs, a
  single input-less tasklet ``out = 1 if p == q else 0`` (the true identity, so
  a shifted diagonal ``numpy.eye(N, k != 0)`` is refused), writing the whole
  array; and consumed ONLY by this ``Solve`` (so removing it is safe).

What is NOT detected: a hand-written Gauss-Jordan / LU elimination loop nest (the
inverse-as-elimination form); a non-identity RHS (that is a genuine linear
solve, left as ``Solve``); a shifted / partial / rectangular identity; an
identity array reused elsewhere.

Placement: this runs in the early semantic-lift phase (alongside
``LoopToSymm``), on the raw frontend shape, BEFORE ``MapToForLoop`` lowers the
identity map to a loop and before the ITE-lowering passes rewrite its
``1 if ... else 0`` body. Like the other semantic lifts it is gated on the
``semantic_lifting`` knob, so the vectorizer path leaves ``solve(A, I)`` intact.
"""
import ast
from typing import List, Optional, Tuple

import dace
from dace import symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.subsets import Range
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible


def full_range(desc) -> Range:
    """A fresh full-array :class:`Range` for ``desc`` (one per edge, never shared)."""
    return Range([(0, s - 1, 1) for s in desc.shape])


def is_square_matrix(desc, dtype=None) -> bool:
    """``desc`` is a 2-D array whose two axes have equal length (and, when
    ``dtype`` is given, whose base dtype matches ``dtype``)."""
    if not isinstance(desc, dace.data.Array) or len(desc.shape) != 2:
        return False
    if symbolic.simplify(desc.shape[0] - desc.shape[1]) != 0:
        return False
    if dtype is not None and desc.dtype.base_type != dtype.base_type:
        return False
    return True


def const_is(node: ast.AST, want: float) -> bool:
    """``node`` is a numeric AST constant equal to ``want``."""
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and float(node.value) == want


def is_identity_tasklet(tasklet: nodes.Tasklet, params: List[str]) -> bool:
    """Whether ``tasklet`` is the input-less identity body ``out = 1 if p == q
    else 0`` over exactly the two map parameters ``params`` (diagonal offset 0)."""
    if tasklet.in_connectors or len(tasklet.out_connectors) != 1:
        return False
    if tasklet.code.language != dace.dtypes.Language.Python:
        return False
    out_conn = next(iter(tasklet.out_connectors))
    try:
        tree = ast.parse(tasklet.code.as_string.strip())
    except SyntaxError:
        return False
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return False
    assign = tree.body[0]
    if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
        return False
    if assign.targets[0].id != out_conn:
        return False
    ifexp = assign.value
    if not isinstance(ifexp, ast.IfExp) or not const_is(ifexp.body, 1.0) or not const_is(ifexp.orelse, 0.0):
        return False
    test = ifexp.test
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1):
        return False
    try:
        left = symbolic.pystr_to_symbolic(ast.unparse(test.left), simplify=False)
        right = symbolic.pystr_to_symbolic(ast.unparse(test.comparators[0]), simplify=False)
    except Exception:
        return False
    p0 = symbolic.pystr_to_symbolic(params[0], simplify=False)
    p1 = symbolic.pystr_to_symbolic(params[1], simplify=False)
    diff = left - right
    # ``p == q`` (offset 0) is the true identity; ``p == q - k`` (k != 0) is a
    # shifted diagonal (numpy.eye's k argument) and must be refused.
    return symbolic.simplify(diff - (p0 - p1)) == 0 or symbolic.simplify(diff - (p1 - p0)) == 0


@explicit_cf_compatible
class LiftInv(ppl.Pass):
    """Lift ``solve(A, identity)`` to a single ``Inv`` library node."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        # Solve is imported inside the method: the linalg library nodes import
        # dace.transformation.transformation (ExpandTransformation), so a
        # top-level import here would form a cycle -- the same reason the sibling
        # linalg lift passes (loop_to_symmetrize / loop_to_transpose / loop_to_einsum)
        # import their library nodes locally.
        from dace.libraries.linalg.nodes.solve import Solve

        candidates: List[Tuple[dace.SDFG, SDFGState, nodes.LibraryNode]] = []
        for sd in sdfg.all_sdfgs_recursive():
            for state in sd.all_states():
                for node in state.nodes():
                    if isinstance(node, Solve):
                        candidates.append((sd, state, node))

        count = 0
        for sd, state, solve in candidates:
            if self._try_lift(sd, state, solve):
                count += 1
        return count or None

    def _try_lift(self, sdfg: dace.SDFG, state: SDFGState, solve: nodes.LibraryNode) -> bool:
        if solve.overwrite:
            return False  # in-place solve -- distinct-array inverse shape only

        # The Solve must have exactly the frontend's connectors: _ain, _bin, _bout.
        in_edges = list(state.in_edges(solve))
        out_edges = list(state.out_edges(solve))
        if len(in_edges) != 2 or len(out_edges) != 1:
            return False
        ain_edge = next((e for e in in_edges if e.dst_conn == "_ain"), None)
        bin_edge = next((e for e in in_edges if e.dst_conn == "_bin"), None)
        bout_edge = out_edges[0]
        if ain_edge is None or bin_edge is None or bout_edge.src_conn != "_bout":
            return False

        ain_node, b_node, out_node = ain_edge.src, bin_edge.src, bout_edge.dst
        if not (isinstance(ain_node, nodes.AccessNode) and isinstance(b_node, nodes.AccessNode)
                and isinstance(out_node, nodes.AccessNode)):
            return False

        a_name, b_name, out_name = ain_node.data, b_node.data, out_node.data
        # A, out, and the identity must be three DISTINCT arrays.
        if len({a_name, b_name, out_name}) != 3:
            return False
        a_desc = sdfg.arrays.get(a_name)
        b_desc = sdfg.arrays.get(b_name)
        out_desc = sdfg.arrays.get(out_name)
        if a_desc is None or b_desc is None or out_desc is None:
            return False
        # A and out square, matching size, matching base dtype (Inv's contract).
        if not is_square_matrix(a_desc) or not is_square_matrix(out_desc, dtype=a_desc.dtype):
            return False
        if symbolic.simplify(a_desc.shape[0] - out_desc.shape[0]) != 0:
            return False
        n = a_desc.shape[0]
        # The RHS operand must be a transient identity of the same size/dtype.
        if not b_desc.transient or not is_square_matrix(b_desc, dtype=a_desc.dtype):
            return False
        if symbolic.simplify(b_desc.shape[0] - n) != 0:
            return False

        eye = self._identity_producer(state, b_node, n)
        if eye is None:
            return False
        map_entry, map_exit, tasklet = eye

        # Safe to delete the identity: it is consumed only by this Solve (its sole
        # non-empty out-edge) and appears nowhere else in the SDFG.
        b_out = [e for e in state.out_edges(b_node) if e.data is not None and not e.data.is_empty()]
        if len(b_out) != 1 or b_out[0] is not bin_edge:
            return False
        refs = sum(1 for st in sdfg.all_states() for nd in st.nodes()
                   if isinstance(nd, nodes.AccessNode) and nd.data == b_name)
        if refs != 1:
            return False

        self._replace(sdfg, state, solve, ain_node, out_node, b_node, map_entry, map_exit, tasklet, a_desc, out_desc,
                      b_name)
        return True

    def _identity_producer(self, state: SDFGState, b_node: nodes.AccessNode, n):
        """If ``b_node`` is written, in ``state``, by exactly one identity map --
        two parameters over ``[0:n, 0:n]``, no map inputs, a single input-less
        ``out = 1 if p == q else 0`` tasklet, writing the whole array -- return
        ``(map_entry, map_exit, tasklet)``; else ``None``."""
        b_in = [e for e in state.in_edges(b_node) if e.data is not None and not e.data.is_empty()]
        if len(b_in) != 1:
            return None
        map_exit = b_in[0].src
        if not isinstance(map_exit, nodes.MapExit):
            return None
        # The producer writes the whole identity array.
        write = b_in[0].data
        if write.data != b_node.data or write.subset is None:
            return None
        for (lo, hi, st), sz in zip(write.subset.ndrange(), b_node.desc(state.sdfg).shape):
            if symbolic.simplify(lo) != 0 or symbolic.simplify(st - 1) != 0 or symbolic.simplify(hi - (sz - 1)) != 0:
                return None

        map_entry = state.entry_node(map_exit)
        if map_entry is None:
            return None
        params = map_entry.map.params
        if len(params) != 2:
            return None
        for (lo, hi, st) in map_entry.map.range.ndrange():
            if symbolic.simplify(lo) != 0 or symbolic.simplify(st - 1) != 0 or symbolic.simplify(hi - (n - 1)) != 0:
                return None
        # An identity reads nothing: the map has no data inputs.
        if any(e.data is not None and not e.data.is_empty() for e in state.in_edges(map_entry)):
            return None
        # The scope holds exactly the identity tasklet (no scratch, no arithmetic).
        body = list(state.all_nodes_between(map_entry, map_exit))
        if len(body) != 1 or not isinstance(body[0], nodes.Tasklet):
            return None
        tasklet = body[0]
        if not is_identity_tasklet(tasklet, params):
            return None
        # The map writes only the identity array.
        if any(e.dst is not b_node for e in state.out_edges(map_exit) if e.data is not None and not e.data.is_empty()):
            return None
        return map_entry, map_exit, tasklet

    def _replace(self, sdfg: dace.SDFG, state: SDFGState, solve: nodes.LibraryNode, ain_node: nodes.AccessNode,
                 out_node: nodes.AccessNode, b_node: nodes.AccessNode, map_entry: nodes.MapEntry,
                 map_exit: nodes.MapExit, tasklet: nodes.Tasklet, a_desc, out_desc, b_name: str) -> None:
        """Replace the ``Solve`` + identity map with a single ``Inv`` node wired
        ``A -> _ain`` / ``_aout -> out`` (mirroring the frontend's ``Inv``
        wiring), then remove the identity subgraph and its transient."""
        from dace.libraries.linalg.nodes.inv import Inv

        inv = Inv(solve.name + "_inv", overwrite_a=False, use_getri=True)
        state.add_node(inv)
        state.add_edge(ain_node, None, inv, "_ain", Memlet(data=ain_node.data, subset=full_range(a_desc)))
        state.add_edge(inv, "_aout", out_node, None, Memlet(data=out_node.data, subset=full_range(out_desc)))

        # remove_node drops incident edges, so the Solve's _ain/_bin/_bout edges
        # and the identity map's internal edges go with their nodes.
        state.remove_node(solve)
        state.remove_node(tasklet)
        state.remove_node(map_entry)
        state.remove_node(map_exit)
        state.remove_node(b_node)
        sdfg.remove_data(b_name, validate=False)


__all__ = ["LiftInv"]
