# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Rewrites Python tasklet bodies to access their arrays directly instead of through copy-in / copy-out
connector temporaries, for the experimental (readable) code generator. An eligible connector
(``__in1``) becomes a direct subscript on the array (``A[__i0, __i1]``); connectors and memlet edges
stay intact (data-flow / scheduling / allocation unchanged), and the generator derives which
connectors were inlined from the rewritten body. The per-dimension index is the memlet subset's lower
bound, linearized with the descriptor's strides / offset exactly as the classic path, so the flat
index matches the legacy copy -- only the presentation (a named ``<array>_idx(...)``) differs.

Correctness-preserving: these keep the classic connector lowering -- WCR outputs, non-single-element
(vector / pointer) accesses, reference-set / stream / non-Array-or-Scalar data. Only Python bodies are
rewritten here; C++ / library bodies are handled at code-gen time (``rewrite_cpp_tasklet_body``).
"""

import ast
import keyword
import warnings

from dace import data as dt
from dace import dtypes, subsets
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.scope import is_in_scope
from dace.sdfg.sdfg import SDFG
from dace.transformation import pass_pipeline as ppl
from dace.transformation.pass_pipeline import Modifies


class InlineTaskletConnectors(ppl.Pass):
    """Rewrites eligible tasklet connectors into direct array accesses."""

    def modifies(self) -> Modifies:
        return Modifies.Tasklets

    def should_reapply(self, modified: Modifies) -> bool:
        return False

    def plan(self, sdfg: SDFG) -> tuple[list[tuple[nodes.Tasklet, dict[str, tuple[str, list[str]]]]], set[str]]:
        """The inlining plan and the containers it may be applied to.

        Decided PER CONTAINER, not per tasklet. A reader rewritten to name the array directly
        relies on the writer's inlining to declare it -- inlining the reader while the writer stays
        classic emits a name nothing declares. So a container is inlined only where every tasklet
        touching it can be, and one tasklet left classic keeps its containers classic everywhere.

        :param sdfg: the SDFG to plan over.
        :returns: ``(plans, safe)`` -- the per-tasklet connector accesses, and the container names
                  every toucher of which can be inlined.
        """
        plans: list[tuple[nodes.Tasklet, dict[str, tuple[str, list[str]]]]] = []
        touched: dict[str, int] = {}
        inlinable: dict[str, int] = {}
        for node, parent in sdfg.all_nodes_recursive():
            if not isinstance(node, nodes.Tasklet):
                continue
            state = parent
            for is_output, edges in ((False, state.in_edges(node)), (True, state.out_edges(node))):
                for edge in edges:
                    if edge.data is None or not edge.data.data:
                        continue
                    if _binds_base_pointer(node, edge, is_output):
                        continue
                    touched[edge.data.data] = touched.get(edge.data.data, 0) + 1
            try:
                accesses = self._plan_tasklet(state.sdfg, state, node)
            except Exception as ex:  # noqa: BLE001
                warnings.warn(
                    f"InlineTaskletConnectors: left tasklet {node.label!r} in classic form: {type(ex).__name__}: {ex}"
                )
                accesses = {}
            for data, _indices in accesses.values():
                inlinable[data] = inlinable.get(data, 0) + 1
            if accesses:
                plans.append((node, accesses))
        return plans, {data for data, count in touched.items() if inlinable.get(data, 0) == count}

    def apply_pass(self, sdfg: SDFG, _) -> set[str] | None:
        plans, safe = self.plan(sdfg)

        inlined_tasklets: set[str] = set()
        for node, accesses in plans:
            accesses = {conn: acc for conn, acc in accesses.items() if acc[0] in safe}
            if not accesses:
                continue
            try:
                if self._apply_plan(node, accesses):
                    inlined_tasklets.add(node.label)
            except Exception as ex:  # noqa: BLE001
                warnings.warn(
                    f"InlineTaskletConnectors: left tasklet {node.label!r} in classic form: {type(ex).__name__}: {ex}"
                )
        return inlined_tasklets or None

    def _connector_access(
        self, osdfg: SDFG, state, node: nodes.Tasklet, edge, is_output: bool
    ) -> tuple[str, str, list[str]] | None:
        """
        Decides whether ``edge``'s connector can be inlined, and if so returns
        ``(connector_name, data_name, index_expressions)`` where the index
        expressions are the per-dimension access indices into the array.
        Returns None if the connector must keep the classic lowering.
        """
        conn = edge.src_conn if is_output else edge.dst_conn
        if not conn:
            return None
        # A connector DECLARED as a pointer is used as a pointer by the body, so it cannot be
        # rewritten into a value access. The case that matters is a callback: the body is a call
        # into a foreign C function whose signature we do not own, and an out-parameter is a ``T*``
        # the callee writes through. The classic lowering binds it as ``T* __out_x = &x;``;
        # inlining it emits the bare ``x``, which the C++ compiler rejects with
        # "cannot convert 'double' to 'double*' in argument passing". Array connectors are
        # unaffected -- their whole-array subsets already fail the single-element test below.
        conntype = node.out_connectors[conn] if is_output else node.in_connectors[conn]
        if isinstance(conntype, (dtypes.pointer, dtypes.vector)):
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
        # Only plain arrays/scalars; never streams, references, structures, or container-arrays
        # (whose element addressing is not the plain flat index). An ArrayView is a plain-flat pointer
        # into its source, so it is inlined like any Array. dt.Reference stays excluded explicitly:
        # ArrayReference subclasses Array, so it would otherwise pass the first test.
        if not isinstance(desc, (dt.Array, dt.Scalar)) or isinstance(desc, (dt.Reference, dt.ContainerArray)):
            return None
        # WCR outputs must go through the atomic resolve path.
        if is_output and memlet.wcr is not None:
            return None
        # A scalar that another pass has promoted to an SDFG constant (e.g.
        # PromoteConstantTransients) is emitted inline as that constant; rewriting a read of it to
        # ``<name>[<idx>]`` would subscript a 0-stride scalar the classic lowering cannot express.
        # Leave it classic -- the connector copy-in reads the constant directly.
        if memlet.data in osdfg.constants:
            return None
        # Dynamic (data-dependent) accesses keep the classic lowering.
        if memlet.dynamic:
            return None
        # The rewritten body is unparsed and reparsed, so a container whose name is not a usable
        # Python identifier (``in``) would come back a SyntaxError. Keep those classic.
        if not memlet.data.isidentifier() or keyword.iskeyword(memlet.data):
            return None
        subset = memlet.subset
        if subset is None or subset.num_elements() != 1:
            # Only single-element (scalar-like) accesses are inlined for now.
            return None
        # The per-dimension access index is the start of each range.
        indices = [str(rb) for (rb, _re, _rs) in subset.ranges]
        return (conn, memlet.data, indices)

    def _plan_tasklet(self, osdfg: SDFG, state, node: nodes.Tasklet) -> dict[str, tuple[str, list[str]]]:
        """The connectors of ``node`` that could be inlined. Pure -- decides, never rewrites."""
        in_acc: dict[str, tuple[str, list[str]]] = {}
        out_acc: dict[str, tuple[str, list[str]]] = {}
        in_subset: dict[str, subsets.Subset] = {}
        out_subset: dict[str, subsets.Subset] = {}
        for edge in state.in_edges(node):
            info = self._connector_access(osdfg, state, node, edge, is_output=False)
            if info is not None:
                conn, data, indices = info
                in_acc[conn] = (data, indices)
                in_subset[conn] = edge.data.subset
        for edge in state.out_edges(node):
            info = self._connector_access(osdfg, state, node, edge, is_output=True)
            if info is not None:
                conn, data, indices = info
                out_acc[conn] = (data, indices)
                out_subset[conn] = edge.data.subset

        # Connector names are unique within the in-set and within the out-set, but
        # an inout connector shares a name across both. Inline such a name only if
        # BOTH sides are inlinable and refer to the same array element (so a single
        # ``A[..]`` correctly stands for both the read and the write). Otherwise
        # (WCR output, different array, one side not inlinable) keep the connector
        # for BOTH sides -- a single identifier in the body cannot mean two things.
        inout = set(node.in_connectors) & set(node.out_connectors)
        accesses: dict[str, tuple[str, list[str]]] = {}
        for name in set(in_acc) | set(out_acc):
            if name in inout:
                if name in in_acc and name in out_acc and in_acc[name] == out_acc[name]:
                    accesses[name] = in_acc[name]
                # else: leave the inout connector in classic form
            else:
                accesses[name] = in_acc.get(name, out_acc.get(name))

        # Only Python bodies are rewritten. A C++/other body is emitted verbatim (no subscript
        # flattening), so an inlined ``A[i, j]`` would become a comma-operator bug -- keep it classic.
        if node.language != dtypes.Language.Python:
            return {}
        # The SVE generator unparses its tasklets itself and types every name through the connectors.
        if is_in_scope(osdfg, state, node, [dtypes.ScheduleType.SVE_Map]):
            return {}

        candidates = [name for name in in_acc if name not in inout and name in accesses]
        for name in reads_after_aliased_writes(node, candidates, in_acc, out_acc, in_subset, out_subset):
            del accesses[name]
        return accesses

    def _apply_plan(self, node: nodes.Tasklet, accesses: dict[str, tuple[str, list[str]]]) -> bool:
        """Rewrite ``node``'s body for the planned connectors."""
        new_code, inlined = self._rewrite_python(node, accesses)
        if not inlined:
            return False
        node.code = CodeBlock(new_code, node.language)
        node.ignored_symbols = set(node.ignored_symbols) | {accesses[c][0] for c in inlined}
        return True

    def _rewrite_python(self, node: nodes.Tasklet, accesses: dict[str, tuple[str, list[str]]]) -> tuple[str, set[str]]:
        # ``as_string`` unparses the tasklet's already-parsed AST, so it is always valid Python;
        # any unexpected failure is still caught by apply_pass and the tasklet left classic.
        tree = ast.parse(node.code.as_string)
        # A name a nested scope rebinds (lambda / def parameter, comprehension target) denotes that
        # binding inside the scope, not the connector. The generator tells inlined connectors apart
        # from live ones by whether the name still occurs in the body, so such a name can be neither
        # rewritten (wrong value) nor left alone (the write-back of an uninitialized copy). Keep the
        # whole connector classic instead.
        rebound = _rebound_names(tree)
        accesses = {conn: acc for conn, acc in accesses.items() if conn not in rebound}
        if not accesses:
            return node.code.as_string, set()
        inliner = _ConnectorInliner(accesses)
        new_tree = inliner.visit(tree)
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree), inliner.inlined


def _binds_base_pointer(node: nodes.Tasklet, edge, is_output: bool) -> bool:
    """True when ``edge``'s connector is bound to the container's base pointer.

    Such a connector is never inlined, but it does not make its container unsafe either. The
    per-container rule exists for a container whose only declaration is the fused binding an
    inlined write emits; a pointer connector's classic binding (``double* _cpy_in = A;``) already
    spells the container's name, so the container is declared no matter what the other tasklets do.

    Counting one as a toucher declined inlining for every other tasklet on the same container. On
    GPU that is every device array -- the host/device memcpy tasklets take them by pointer -- so
    kernel bodies came out in classic connector form.
    """
    conn = edge.src_conn if is_output else edge.dst_conn
    if not conn:
        return False
    conntype = node.out_connectors.get(conn) if is_output else node.in_connectors.get(conn)
    return isinstance(conntype, dtypes.pointer)


def _rebound_names(tree: ast.AST) -> set[str]:
    """Names bound by a nested scope inside a tasklet body: lambda / function parameters, the
    function's own name, and comprehension targets."""
    names: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
            args = n.args
            names.update(a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs))
            names.update(a.arg for a in (args.vararg, args.kwarg) if a is not None)
            if not isinstance(n, ast.Lambda):
                names.add(n.name)
        elif isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for gen in n.generators:
                names.update(t.id for t in ast.walk(gen.target) if isinstance(t, ast.Name))
    return names


def aliased_writers(
    name: str,
    in_acc: dict[str, tuple[str, list[str]]],
    out_acc: dict[str, tuple[str, list[str]]],
    in_subset: dict[str, subsets.Subset],
    out_subset: dict[str, subsets.Subset],
) -> list[str]:
    """The outputs other than input ``name`` that write an element of its container it may read."""
    return [
        other
        for other in out_acc
        if other != name
        and out_acc[other][0] == in_acc[name][0]
        and subsets.intersects(in_subset[name], out_subset[other]) is not False
    ]


def reads_after_aliased_writes(
    node: nodes.Tasklet,
    candidates: list[str],
    in_acc: dict[str, tuple[str, list[str]]],
    out_acc: dict[str, tuple[str, list[str]]],
    in_subset: dict[str, subsets.Subset],
    out_subset: dict[str, subsets.Subset],
) -> list[str]:
    """The ``candidates`` inputs a statement of ``node`` may read after an output writing an aliased element."""
    aliased = {name: aliased_writers(name, in_acc, out_acc, in_subset, out_subset) for name in candidates}
    aliased = {name: writers for name, writers in aliased.items() if writers}
    if not aliased:
        return []
    body = ast.parse(node.code.as_string).body
    return [name for name, writers in aliased.items() if reads_after_write(body, name, writers)]


def reads_after_write(body: list[ast.stmt], read: str, writers: list[str]) -> bool:
    """True when a statement of ``body`` may read ``read`` after one of ``writers`` was stored."""
    written = False
    for stmt in body:
        walked = list(ast.walk(stmt))
        reads = any(isinstance(n, ast.Name) and n.id == read for n in walked)
        stores = any(stored_name(n) in writers for n in walked)
        if reads and (written or (stores and not evaluates_before_storing(stmt, walked))):
            return True
        written = written or stores
    return False


def stored_name(n: ast.AST) -> str | None:
    if not isinstance(n, (ast.Name, ast.Subscript)) or not isinstance(n.ctx, ast.Store):
        return None
    base = n if isinstance(n, ast.Name) else n.value
    return base.id if isinstance(base, ast.Name) else None


def evaluates_before_storing(stmt: ast.stmt, walked: list[ast.AST]) -> bool:
    """A plain assignment evaluates its whole value first; a walrus inside it stores early."""
    return isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign)) and not any(
        isinstance(n, ast.NamedExpr) for n in walked
    )


class _ConnectorInliner(ast.NodeTransformer):
    """Replaces connector names with direct ``data[indices]`` subscripts."""

    def __init__(self, accesses: dict[str, tuple[str, list[str]]]):
        self.accesses = accesses
        self.inlined: set[str] = set()

    def _make_access(self, data: str, indices: list[str]) -> ast.AST:
        elts = [ast.parse(ix, mode="eval").body for ix in indices]
        if len(elts) == 1:
            sl = elts[0]
        else:
            sl = ast.Tuple(elts=elts, ctx=ast.Load())
        return ast.Subscript(value=ast.Name(id=data, ctx=ast.Load()), slice=sl, ctx=ast.Load())

    def _replace(self, name: str, node: ast.AST) -> ast.AST:
        """Replaces an inlined connector ``name`` with its direct ``data[indices]`` access."""
        data, indices = self.accesses[name]
        self.inlined.add(name)
        return ast.copy_location(self._make_access(data, indices), node)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        # A subscripted inlined connector (e.g. pointer-to-scalar ``conn[0]``):
        # replace the whole thing with the direct access.
        base = node.value
        if isinstance(base, ast.Name) and base.id in self.accesses:
            return self._replace(base.id, node)
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.accesses:
            return self._replace(node.id, node)
        return node
