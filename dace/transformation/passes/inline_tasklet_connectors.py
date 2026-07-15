# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Pass that rewrites tasklets to access their arrays directly instead of through
copy-in / copy-out connector temporaries. Used by the experimental (readable)
code generator.

For every eligible connector of a tasklet, the connector reference in the body
is replaced by a direct access to the underlying array (e.g. ``__in1`` becomes
``A[__i0, __i1]``) and the array name is added to the tasklet's
``ignored_symbols`` (so it is not treated as an undefined free symbol). The
connectors and their memlet edges are kept intact, so data-flow, scheduling, and
allocation are unchanged.

The per-dimension access index is the *lower bound* of the connector's memlet
subset in the outer array's coordinate system -- the same quantity
``expand_nested_sdfg_inputs`` uses as its offset ("offset = lower bound of the
narrowed subset"). For a single-element access every dimension is a point
(``lo == hi``), so ``ranges[d][0]`` per dimension is the exact multi-dimensional
index. The code generator then linearizes it with the descriptor's strides and
``offset`` exactly as the classic path does (``cpp_offset_expr``), so the flat
index is identical to the legacy connector copy -- only the presentation (a
named ``<array>_idx(...)`` function) differs.

No marker is stored on the tasklet: the readable code generator derives which
connectors were inlined from the rewritten body -- a connector whose name no
longer appears in the body needs no copy-in / copy-out.

Connectors that cannot be rewritten safely are left untouched (they keep the
classic connector-based lowering), so the pass is always correctness-preserving:

* Write-conflict (WCR) outputs are left alone -- they must go through the
  atomic ``write_and_resolve`` path.
* Non-single-element accesses (vector / pointer connectors, e.g. a whole row
  passed to a library call) are left alone.
* Only Python tasklet bodies are rewritten here. C++ / library tasklet bodies are
  handled at code-generation time (``ExperimentalCPUCodeGen.rewrite_cpp_tasklet_body``),
  which tokenizes with the pygments C++ lexer so string / char literals and comments
  are never touched; this pass leaves them in classic connector form.
* Reference-set, stream push/pop, and non-Array/Scalar data are left alone.
"""
import ast
import warnings
from typing import Dict, List, Optional, Set, Tuple

from dace import data as dt
from dace import dtypes
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.sdfg import SDFG
from dace.transformation import pass_pipeline as ppl
from dace.transformation.pass_pipeline import Modifies


class InlineTaskletConnectors(ppl.Pass):
    """ Rewrites eligible tasklet connectors into direct array accesses. """

    CATEGORY = 'Optimization Preparation'

    def modifies(self) -> Modifies:
        return Modifies.Tasklets

    def should_reapply(self, modified: Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _) -> Optional[Set[str]]:
        inlined_tasklets: Set[str] = set()
        for node, parent in sdfg.all_nodes_recursive():
            if not isinstance(node, nodes.Tasklet):
                continue
            state = parent
            osdfg = state.sdfg
            # Per-tasklet resilience: a tasklet we cannot rewrite is simply left
            # in classic connector form (still correct), never crashing codegen.
            try:
                if self._inline_tasklet(osdfg, state, node):
                    inlined_tasklets.add(node.label)
            except Exception as ex:  # noqa: BLE001
                warnings.warn(f'InlineTaskletConnectors: left tasklet {node.label!r} in classic form: '
                              f'{type(ex).__name__}: {ex}')
        return inlined_tasklets or None

    def _connector_access(self, osdfg: SDFG, state, node: nodes.Tasklet, edge,
                          is_output: bool) -> Optional[Tuple[str, str, List[str]]]:
        """
        Decides whether ``edge``'s connector can be inlined, and if so returns
        ``(connector_name, data_name, index_expressions)`` where the index
        expressions are the per-dimension access indices into the array.
        Returns None if the connector must keep the classic lowering.
        """
        conn = edge.src_conn if is_output else edge.dst_conn
        if not conn:
            return None
        memlet = edge.data
        if memlet.data is None or memlet.data not in osdfg.arrays:
            return None
        # Only inline accesses to a real AccessNode (array/scalar in memory), never
        # tasklet<->tasklet (code->code) register connectors.
        path = state.memlet_path(edge)
        far = path[0].src if not is_output else path[-1].dst
        if not isinstance(far, nodes.AccessNode):
            return None
        desc = osdfg.arrays[memlet.data]
        # Only plain arrays/scalars; never streams, references, structures, or
        # container-arrays (whose element addressing is not the plain flat index).
        # An ArrayView IS a plain-flat pointer into its source (``source + offset``,
        # its own strides), so it is inlined like any Array -- ``V[V_idx(...)]``.
        if not isinstance(desc, (dt.Array, dt.Scalar)):
            return None
        if isinstance(desc, (dt.Reference, dt.ContainerArray)):
            return None
        # WCR outputs must go through the atomic resolve path.
        if is_output and memlet.wcr is not None:
            return None
        # Dynamic (data-dependent) accesses keep the classic lowering.
        if memlet.dynamic:
            return None
        subset = memlet.subset
        if subset is None or subset.num_elements() != 1:
            # Only single-element (scalar-like) accesses are inlined for now.
            return None
        # The per-dimension access index is the start of each range.
        indices = [str(rb) for (rb, _re, _rs) in subset.ranges]
        return (conn, memlet.data, indices)

    def _inline_tasklet(self, osdfg: SDFG, state, node: nodes.Tasklet) -> bool:
        in_acc: Dict[str, Tuple[str, List[str]]] = {}
        out_acc: Dict[str, Tuple[str, List[str]]] = {}
        for edge in state.in_edges(node):
            info = self._connector_access(osdfg, state, node, edge, is_output=False)
            if info is not None:
                conn, data, indices = info
                in_acc[conn] = (data, indices)
        for edge in state.out_edges(node):
            info = self._connector_access(osdfg, state, node, edge, is_output=True)
            if info is not None:
                conn, data, indices = info
                out_acc[conn] = (data, indices)

        # Connector names are unique within the in-set and within the out-set, but
        # an inout connector shares a name across both. Inline such a name only if
        # BOTH sides are inlinable and refer to the same array element (so a single
        # ``A[..]`` correctly stands for both the read and the write). Otherwise
        # (WCR output, different array, one side not inlinable) keep the connector
        # for BOTH sides -- a single identifier in the body cannot mean two things.
        inout = set(node.in_connectors) & set(node.out_connectors)
        accesses: Dict[str, Tuple[str, List[str]]] = {}
        for name in set(in_acc) | set(out_acc):
            if name in inout:
                if name in in_acc and name in out_acc and in_acc[name] == out_acc[name]:
                    accesses[name] = in_acc[name]
                # else: leave the inout connector in classic form
            else:
                accesses[name] = in_acc.get(name, out_acc.get(name))

        if not accesses:
            return False

        # Only Python tasklet bodies are rewritten. C++ (and other) tasklet
        # bodies are emitted verbatim by the code generator -- there is no
        # subscript-flattening step for them -- so an inlined ``A[i, j]`` would
        # be emitted literally (a comma-operator bug) and non-unit strides would
        # be ignored. Such tasklets keep their classic connector copy-in/out.
        if node.language != dtypes.Language.Python:
            return False
        new_code, inlined = self._rewrite_python(node, accesses)

        if not inlined:
            return False

        node.code = CodeBlock(new_code, node.language)
        node.ignored_symbols = set(node.ignored_symbols) | {accesses[c][0] for c in inlined}
        return True

    def _rewrite_python(self, node: nodes.Tasklet, accesses: Dict[str, Tuple[str, List[str]]]) -> Tuple[str, Set[str]]:
        try:
            tree = ast.parse(node.code.as_string)
        except SyntaxError:
            return node.code.as_string, set()
        inliner = _ConnectorInliner(accesses)
        new_tree = inliner.visit(tree)
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree), inliner.inlined


def tasklet_emits_brace_free(sdfg: SDFG, state, tasklet: nodes.Tasklet) -> bool:
    """True iff ``InlineTaskletConnectors`` will inline EVERY connector of ``tasklet``,
    so the readable code generator emits it as a single brace-free statement with no
    copy-in/out local.

    ``MarkConstInit`` uses this to decide whether a fused ``const T x = <expr>;`` binding
    lands at the enclosing scope (visible to the reads) or is trapped inside the tasklet's
    ``{ }`` block (a use-before-declaration miscompile). The predicate is SOUND: it returns
    True only when every connector is individually inlinable AND there is no inout connector
    (ITC may keep an inout connector classic even when each side is individually inlinable),
    so a True answer guarantees the brace-free emission.
    """
    if tasklet.language != dtypes.Language.Python:
        return False
    if set(tasklet.in_connectors) & set(tasklet.out_connectors):  # inout -> may stay classic
        return False
    checker = InlineTaskletConnectors()
    for edge in state.in_edges(tasklet):
        if not edge.data.is_empty() and checker._connector_access(sdfg, state, tasklet, edge, is_output=False) is None:
            return False
    for edge in state.out_edges(tasklet):
        if not edge.data.is_empty() and checker._connector_access(sdfg, state, tasklet, edge, is_output=True) is None:
            return False
    return True


class _ConnectorInliner(ast.NodeTransformer):
    """ Replaces connector names with direct ``data[indices]`` subscripts. """

    def __init__(self, accesses: Dict[str, Tuple[str, List[str]]]):
        self.accesses = accesses
        self.inlined: Set[str] = set()

    def _make_access(self, data: str, indices: List[str]) -> ast.AST:
        elts = [ast.parse(ix, mode='eval').body for ix in indices]
        if len(elts) == 1:
            sl = elts[0]
        else:
            sl = ast.Tuple(elts=elts, ctx=ast.Load())
        return ast.Subscript(value=ast.Name(id=data, ctx=ast.Load()), slice=sl, ctx=ast.Load())

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        # A subscripted inlined connector (e.g. pointer-to-scalar ``conn[0]``):
        # replace the whole thing with the direct access.
        base = node.value
        if isinstance(base, ast.Name) and base.id in self.accesses:
            data, indices = self.accesses[base.id]
            self.inlined.add(base.id)
            return ast.copy_location(self._make_access(data, indices), node)
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.accesses:
            data, indices = self.accesses[node.id]
            self.inlined.add(node.id)
            return ast.copy_location(self._make_access(data, indices), node)
        return node
