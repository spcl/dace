# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Passes that move data between length-1 ``Array`` and ``Scalar`` form.

``ConvertLengthOneArraysToScalars`` rewrites every length-1 ``Array`` (shape ``(1,)``) to a true
``Scalar`` and drops the now-redundant ``[0]`` accessors from interstate-edge assignments,
conditional-block guards, loop-region conditions and memlet subsets.
``ConvertScalarsToLengthOneArrays`` is the inverse (``Scalar`` -> length-1 ``Array``).

Transient descriptors are rewritten IN PLACE (same name): a transient is SDFG-internal, so turning it
into the other form changes nothing a caller sees.

A NON-transient descriptor is part of the SDFG signature -- rewriting it in place would change the
caller's contract (the caller passes a 1-element numpy buffer for an ``Array`` but a by-value scalar
for a ``Scalar``, and a by-value scalar cannot receive a written-back result). ``preserve_abi`` is how
such a descriptor is converted anyway WITHOUT touching that contract: it is never rewritten, only
STAGED through copy states, so the SDFG signature is byte-identical after the pass while the body
computes on the other form. Set ``preserve_abi`` to opt in; left clear, non-transients are skipped
entirely and only transients convert.

Staging one non-transient means:

* the signature descriptor ``alpha`` (a length-1 ``Array``) is KEPT,
* a fresh transient ``Scalar`` ``scal_alpha`` is introduced and every body reference to ``alpha`` is
  repointed to it,
* a copy-IN ``scal_alpha = alpha[0]`` is prepended in a new start state (only if ``alpha`` is read),
* a copy-OUT ``alpha[0] = scal_alpha`` is appended in a new sink state (only if ``alpha`` is written).

The whole SDFG body then computes on the plain scalar while the signature is unchanged. The inverse
pass stages a non-transient ``Scalar`` into a length-1 ``Array`` the same way. The fresh descriptor is
always allocated with ``find_new_name`` so repeated forward/inverse application never collides on a
name it created earlier.

The HLFIR Fortran frontend uses ``ConvertLengthOneArraysToScalars`` as a post-generation cleanup:
``Scalar`` data on the SDFG signature binds to a plain Python ``int`` / ``float`` whereas a length-1
``Array`` needs a 1-element numpy buffer.
"""
import ast
import itertools
from typing import Callable, Dict, List, Optional, Set, Tuple

import dace
from dace import Memlet, dtypes, properties, subsets
from dace.properties import CodeBlock
from dace.sdfg import SDFG, SDFGState, InterstateEdge, nodes, utils as sdutil
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation

#: Rewrites one source-text slot; see :func:`rewrite_code_slots`.
CodeSlotRewriter = Callable[[str], str]


def rewrite_code_slots(sdfg: SDFG, rewrite: CodeSlotRewriter) -> None:
    """Apply ``rewrite`` to every slot of ``sdfg`` that names a descriptor as source TEXT rather than
    through a memlet: interstate-edge assignments and conditions, ``ConditionalBlock`` branch guards,
    ``LoopRegion`` init / update / condition statements, and ``NestedSDFG.symbol_mapping`` values.

    Shared by every pass that swaps a descriptor between ``Scalar`` and length-1 ``Array`` form: the
    slots to visit are a property of the SDFG state machine, not of the direction of the swap, so only
    the per-slot string transform differs between callers.

    Scoped to ``sdfg`` alone -- it does NOT descend into nested SDFGs, because which inner descriptor
    a rewritten outer name corresponds to is pass-specific (connector matching vs. a transient-only
    rule), and a blind descent would rewrite an unrelated inner descriptor that merely shares the name.

    :param sdfg: SDFG whose code slots are rewritten in place.
    :param rewrite: maps a slot's source text to its replacement.
    """
    for edge in sdfg.all_interstate_edges():
        ise = edge.data
        if ise is None:
            continue
        for key, value in list(ise.assignments.items()):
            if isinstance(value, str):
                ise.assignments[key] = rewrite(value)
        if isinstance(ise.condition, CodeBlock):
            ise.condition = CodeBlock(rewrite(ise.condition.as_string), ise.condition.language)

    for block in sdfg.all_control_flow_blocks():
        if isinstance(block, ConditionalBlock):
            for branch in block.branches:
                # Mutate the CodeBlock rather than reassigning the entry: ``add_branch`` appends a
                # list but ``remove_branch`` rebuilds them as tuples, so assignment raises there.
                if isinstance(branch[0], CodeBlock):
                    branch[0].as_string = rewrite(branch[0].as_string)
        elif isinstance(block, LoopRegion):
            # init/update are optional -- a ``while`` LoopRegion has only the condition.
            if isinstance(block.init_statement, CodeBlock):
                block.init_statement = CodeBlock(rewrite(block.init_statement.as_string), block.init_statement.language)
            if isinstance(block.update_statement, CodeBlock):
                block.update_statement = CodeBlock(rewrite(block.update_statement.as_string),
                                                   block.update_statement.language)
            if isinstance(block.loop_condition, CodeBlock):
                block.loop_condition = CodeBlock(rewrite(block.loop_condition.as_string), block.loop_condition.language)

    for state in sdfg.states():
        for node in state.nodes():
            if not isinstance(node, nodes.NestedSDFG):
                continue
            for key, value in list(node.symbol_mapping.items()):
                text = value if isinstance(value, str) else str(value)
                rewritten = rewrite(text)
                if rewritten != text:
                    node.symbol_mapping[key] = rewritten


class ScalarRefRewriter(ast.NodeTransformer):
    """Collapse ``old[0]`` to ``new`` and rename a bare ``old`` to ``new``.

    :param rename: Mapping from each rewritten descriptor's old name to its new name.
    """

    def __init__(self, rename: Dict[str, str]):
        self.rename = rename

    def visit_Subscript(self, node: ast.Subscript):
        index = node.slice.value if isinstance(node.slice, ast.Index) else node.slice
        if (isinstance(node.value, ast.Name) and node.value.id in self.rename and isinstance(index, ast.Constant)
                and index.value == 0):
            return ast.copy_location(ast.Name(id=self.rename[node.value.id], ctx=node.ctx), node)
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id in self.rename:
            return ast.copy_location(ast.Name(id=self.rename[node.id], ctx=node.ctx), node)
        return node


class ElementRefRewriter(ast.NodeTransformer):
    """Inverse of :class:`ScalarRefRewriter`: point a bare ``old`` at ``new[0]``.

    :param rename: Mapping from each rewritten descriptor's old name to its new name.
    """

    def __init__(self, rename: Dict[str, str]):
        self.rename = rename

    def visit_Subscript(self, node: ast.Subscript):
        # An already-subscripted reference is only renamed; its index is rewritten on its own.
        if isinstance(node.value, ast.Name) and node.value.id in self.rename:
            node.slice = self.visit(node.slice)
            node.value = ast.Name(id=self.rename[node.value.id], ctx=node.value.ctx)
            return node
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id not in self.rename:
            return node
        element = ast.Subscript(value=ast.Name(id=self.rename[node.id], ctx=ast.Load()),
                                slice=ast.Constant(value=0),
                                ctx=node.ctx)
        return ast.fix_missing_locations(ast.copy_location(element, node))


def _rewrite_with(expr: str, rewriter: ast.NodeTransformer) -> str:
    """Apply ``rewriter`` to ``expr`` parsed as Python.

    :param expr: Source expression to rewrite; text that does not parse is returned unchanged.
    :param rewriter: Transformer to apply to the parsed tree.
    :returns: The rewritten source.
    """
    try:
        tree = ast.parse(expr)
    except SyntaxError:
        return expr
    # ``ast.unparse``, not ``astutils.unparse``: the latter parenthesizes every binary operation,
    # which would rewrite slots this pass did not touch.
    return ast.unparse(rewriter.visit(tree))


def rewrite_refs(expr: str, rename: Dict[str, str]) -> str:
    """Collapse each rewritten descriptor's redundant ``old[0]`` accessor and rename it to ``new``.

    An attribute (``obj.old``) is never a descriptor reference, and a name inside a string literal is
    not code; the AST distinguishes both. When ``old == new`` this only strips the ``[0]``.

    :param expr: Source expression to rewrite.
    :param rename: Mapping from each rewritten descriptor's old name to its new name.
    :returns: ``expr`` with each ``old[0]`` / ``old`` rewritten to ``new``.
    """
    return _rewrite_with(expr, ScalarRefRewriter(rename))


def rewrite_refs_to_element(expr: str, rename: Dict[str, str]) -> str:
    """Inverse of :func:`rewrite_refs`: point a bare reference at element 0 of a now-length-1 array.

    :param expr: Source expression to rewrite.
    :param rename: Mapping from each rewritten descriptor's old name to its new name.
    :returns: ``expr`` with each bare ``old`` rewritten to ``new[0]``.
    """
    return _rewrite_with(expr, ElementRefRewriter(rename))


def repoint_memlet_to_element(edge: 'dace.sdfg.graph.MultiConnectorEdge', rename: Dict[str, str]) -> None:
    """Re-point one edge's memlet at the rewritten descriptors, collapsing each rewritten side's subset
    to the single element ``0``.

    A memlet names only ONE side of a copy edge (``data`` / ``subset``); the opposite side's index
    lives in ``other_subset``. The two sides are rewritten INDEPENDENTLY here -- the named side being
    a rewritten descriptor says nothing about the other side, whose index must survive untouched when
    it is not itself rewritten. Mutating in place (rather than building a replacement ``Memlet`` from
    a handful of fields) is what keeps ``other_subset`` -- and volume, ``wcr_nonatomic``, ``allow_oob``
    and the cached src/dst orientation -- from being silently dropped.

    :param edge: edge whose memlet is rewritten in place.
    :param rename: mapping from each rewritten descriptor's old name to its new name.
    """
    mem = edge.data
    # An empty memlet is a pure ORDERING edge -- it names no descriptor and must survive untouched.
    if mem is None or mem.is_empty() or mem.data is None:
        return
    if mem.data in rename:
        mem.data = rename[mem.data]
        mem.subset = subsets.Range.from_string('0')
    # The other side collapses only when IT is a rewritten descriptor, else validation rejects the rank.
    if mem.other_subset is not None and any(
            isinstance(n, nodes.AccessNode) and n.data in rename.values() and n.data != mem.data
            for n in (edge.src, edge.dst)):
        mem.other_subset = subsets.Range.from_string('0')


def descriptor_is_read(sdfg: SDFG, name: str) -> bool:
    """True if ``name`` is read anywhere in ``sdfg`` (some AccessNode of it has an out-edge)."""
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode) and node.data == name and state.out_degree(node) > 0:
                return True
    return False


def descriptor_is_written(sdfg: SDFG, name: str) -> bool:
    """True if ``name`` is written anywhere in ``sdfg`` (some AccessNode of it has an in-edge)."""
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode) and node.data == name and state.in_degree(node) > 0:
                return True
    return False


def descriptor_written_by_gpu_map(sdfg: SDFG, name: str) -> bool:
    """True if a GPU-scheduled map writes ``name``.

    :meth:`ConvertLengthOneArraysToScalars._blocked_by_unscalarizable_neighbors` already refuses a
    descriptor a GPU map reads or holds inside its body, so a kernel output is the only device
    adjacency a staged descriptor can still have -- and the only reason its scalar must stay in
    device memory.
    """
    for state in sdfg.all_states():
        for node in state.nodes():
            if not isinstance(node, nodes.AccessNode) or node.data != name:
                continue
            for edge in state.in_edges(node):
                if isinstance(edge.src, nodes.MapExit) and edge.src.map.schedule in dtypes.GPU_SCHEDULES:
                    return True
    return False


def _copyin_state(sdfg: SDFG) -> SDFGState:
    """A new start state to hold copy-IN edges (prepended before the current start)."""
    return sdfg.add_state_before(sdfg.start_state, 'stage_copyin', is_start_block=True)


def _copyout_state(sdfg: SDFG) -> SDFGState:
    """A new sink state to hold copy-OUT edges (all current top-level sinks lead into it)."""
    sinks = [n for n in sdfg.nodes() if sdfg.out_degree(n) == 0]
    out = sdfg.add_state('stage_copyout')
    for s in sinks:
        sdfg.add_edge(s, out, InterstateEdge())
    return out


@properties.make_properties
@transformation.explicit_cf_compatible
class ConvertLengthOneArraysToScalars(ppl.Pass):
    """Rewrite every length-1 ``Array`` (shape ``(1,)``) to a true ``Scalar`` of the same dtype, and
    drop the ``[0]`` accessors that referenced it. Transient arrays are rewritten in place; a
    non-transient array is STAGED into a fresh transient scalar (with copy-in/out) only when
    ``preserve_abi`` is set -- see the module docstring.

    Length-1 arrays of an ``opaque`` dtype (an external handle such as ``MPI_Request`` / ``MPI_Comm``)
    are left untouched: a consumer that takes the handle through a pointer connector needs the source
    to stay an ``Array`` so it lowers to a pointer. A ``View`` (or a length-1 array that backs one) is
    left untouched too: a ``Scalar`` cannot carry the ``views`` alias edge, and a scalar source is
    emitted ``const`` so a write through the view fails to compile.

    :param recursive: Recurse into nested SDFGs (only their transient length-1 arrays are rewritten --
        a non-transient nested-SDFG arg is part of its parent's signature).
    :param preserve_abi: Also convert non-transient (signature) length-1 arrays, keeping the ABI
        intact: the array stays on the signature and is STAGED into a fresh transient scalar with a
        copy-in (first state) and copy-out (last state), never rewritten. Default ``False`` -- only
        transient arrays are scalarized.
    :param filter: Optional whitelist NARROWING which top-level descriptors are eligible. ``None``
        (default) -- no restriction. A set -- rewrite ONLY named descriptors that are ALSO eligible
        under the other gates; an empty set rewrites nothing. Gates the top-level rewrite only, not the
        nested-SDFG transient recursion.
    :param single_element: Also rewrite a higher-rank single-element array (every dim == 1, e.g. a
        ``(1, 1)`` map-fusion scratch buffer), not just a rank-1 length-1 array.
    :param skip_gpu_outputs: If ``True``, length-1 arrays that are outputs of a GPU-scheduled map are
        left as arrays. This is useful right before CPU codegen, where scalarizing a GPU kernel output
        would force an expensive round-trip through the host scalar ABI.
    """

    recursive = properties.Property(dtype=bool, default=True, desc="Recurse into nested SDFGs (transient-only there).")
    preserve_abi = properties.Property(
        dtype=bool,
        default=False,
        desc="Convert non-transient length-1 arrays too, keeping the ABI intact by staging them into "
        "fresh transient scalars (copy-in/out) instead of rewriting the signature array. Default only "
        "scalarizes transients.")
    filter = properties.SetProperty(
        element_type=str,
        default=None,
        allow_none=True,
        desc="Optional whitelist restricting which top-level descriptors are eligible. ``None`` -- no "
        "restriction. A set -- only named descriptors that are ALSO eligible under the other gates are "
        "rewritten; an empty set rewrites nothing. Does not gate the nested-SDFG recursion.")
    single_element = properties.Property(
        dtype=bool,
        default=False,
        desc="Also rewrite a higher-rank single-element array (every dim == 1, e.g. a (1, 1) map-fusion "
        "scratch buffer), not just a rank-1 length-1 array.")
    skip_gpu_outputs = properties.Property(
        dtype=bool, default=False, desc="Leave length-1 arrays that are outputs of GPU-scheduled maps as arrays.")

    def __init__(self,
                 recursive: bool = True,
                 preserve_abi: bool = False,
                 filter: 'Optional[Set[str]]' = None,
                 single_element: bool = False,
                 skip_gpu_outputs: bool = False):
        super().__init__()
        self.recursive = recursive
        self.preserve_abi = preserve_abi
        self.filter = None if filter is None else frozenset(filter)
        self.single_element = single_element
        self.skip_gpu_outputs = skip_gpu_outputs

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets | ppl.Modifies.Symbols | ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _blocked_sources(self, sdfg: SDFG) -> Set[str]:
        """Descriptor names that must stay ``Array`` regardless of shape: a ``View`` cannot carry the
        ``views`` alias edge, and a length-1 array that BACKS a view must stay an aliasable source."""
        blocked: Set[str] = set()
        for state in sdfg.states():
            for node in state.nodes():
                if not isinstance(node, nodes.AccessNode):
                    continue
                if isinstance(sdfg.arrays.get(node.data), dace.data.View):
                    ve = sdutil.get_view_edge(state, node)
                    if ve is None:
                        continue
                    other = ve.src if ve.dst is node else ve.dst
                    if isinstance(other, nodes.AccessNode):
                        blocked.add(other.data)
        return blocked

    def _blocked_by_unscalarizable_neighbors(self, sdfg: SDFG) -> Set[str]:
        """Descriptor names that must stay ``Array`` because a neighbor generates code we cannot
        rewrite from ``buf[0]`` to ``buf``: an unexpanded library node, a non-Python (C++) tasklet,
        or a GPU-scheduled map that uses the buffer as a multi-thread collective.

        For GPU maps we only block the *input* side (edges into a ``MapEntry``) and the *partial* side
        (edges from an inside node into a ``MapExit``). A ``MapExit``-to-outside edge is a normal map
        output; it can be scalarized and is widened back by ``PromoteGPUScalarsToArrays`` when needed."""
        blocked: Set[str] = set()
        for state in sdfg.states():
            for node in state.nodes():
                if not isinstance(node, nodes.AccessNode):
                    continue
                arr = sdfg.arrays.get(node.data)
                if not isinstance(arr, dace.data.Array):
                    continue
                for edge in itertools.chain(state.in_edges(node), state.out_edges(node)):
                    nb = edge.src if edge.dst is node else edge.dst
                    if isinstance(nb, nodes.LibraryNode):
                        blocked.add(node.data)
                        break
                    if isinstance(nb, nodes.Tasklet) and nb.language != dtypes.Language.Python:
                        blocked.add(node.data)
                        break
                    if isinstance(nb, nodes.MapEntry) and nb.map.schedule in dtypes.GPU_SCHEDULES:
                        # Block arrays that feed the map entry (node -> MapEntry) or that live inside the
                        # map body and receive an input from the entry (MapEntry -> node).
                        blocked.add(node.data)
                        break
                    if isinstance(nb, nodes.MapExit) and nb.map.schedule in dtypes.GPU_SCHEDULES:
                        # Partial WCR outputs from inside the map body (node -> MapExit) are always
                        # blocked because the code generator cannot emit them as a scalar collective.
                        if edge.src is node:
                            blocked.add(node.data)
                            break
                        # Normal MapExit -> outside outputs are allowed by default, but the caller may
                        # opt out: scalarizing a non-transient GPU kernel output forces a host scalar ABI
                        # round-trip. Transient outputs are scalarized and widened back by the GPU-scalar
                        # promotion pass when they live in device memory.
                        if self.skip_gpu_outputs and edge.dst is node and not arr.transient:
                            blocked.add(node.data)
                            break
                        if edge.src is node:
                            blocked.add(node.data)
                            break
        return blocked

    def _is_eligible(self, sdfg: SDFG, arr_name: str, arr: 'dace.data.Data', blocked: Set[str],
                     apply_filter: bool) -> bool:
        """Whether a descriptor is a length-1 (or, with ``single_element``, all-ones) array we may
        rewrite: not a View / view source / opaque, and passing the filter."""
        # ``ArrayReference`` derives from ``Array``, so it reaches here like any other array. Rewriting
        # one to a transient Scalar destroys the pointer alias and sends writes to a local instead --
        # the same reason ``View`` is excluded, and a silent miscompile rather than a validation error.
        if not isinstance(arr, dace.data.Array) or isinstance(arr, (dace.data.View, dace.data.Reference)):
            return False
        if arr_name in blocked:
            return False
        is_len1 = arr.shape == (1, ) or arr.shape == [1]
        is_single = self.single_element and len(arr.shape) >= 1 and all(d == 1 for d in arr.shape)
        if not (is_len1 or is_single):
            return False
        if isinstance(arr.dtype, dace.dtypes.opaque):
            return False
        if apply_filter and self.filter is not None and arr_name not in self.filter:
            return False
        return True

    def _rewrite(self, sdfg: SDFG, apply_filter: bool, stage_nontransients: bool) -> Set[str]:
        """Scalarize length-1 arrays in ``sdfg`` (modified in place).

        Transient arrays are scalarized in place (same name). If ``stage_nontransients``, each
        non-transient length-1 array is staged into a fresh transient scalar with copy-in/out.

        :param sdfg: SDFG to rewrite.
        :param apply_filter: Whether the top-level ``filter`` set gates the rewrite here (``False`` in
            the nested-SDFG recursion: the filter names outer-level descriptors).
        :param stage_nontransients: Whether to stage non-transient arrays (top level only).
        :returns: Names of the descriptors that are now scalar-referenced in the body.
        """
        blocked = self._blocked_sources(sdfg) | self._blocked_by_unscalarizable_neighbors(sdfg)
        # rename[old] = the name the body should reference after the rewrite (== old for a transient
        # scalarized in place; a fresh scalar name for a staged non-transient). staged carries the
        # kept signature array plus its read/write direction so copy-in/out can be wired afterwards.
        rename: Dict[str, str] = {}
        staged: List[Tuple[str, str, bool, bool]] = []  # (array_name, scalar_name, is_read, is_written)

        for arr_name, arr in list(sdfg.arrays.items()):
            if not self._is_eligible(sdfg, arr_name, arr, blocked, apply_filter):
                continue
            if arr.transient:
                sdfg.remove_data(arr_name, validate=False)
                sdfg.add_scalar(arr_name,
                                dtype=arr.dtype,
                                storage=arr.storage,
                                transient=True,
                                lifetime=arr.lifetime,
                                debuginfo=arr.debuginfo,
                                find_new_name=False)
                rename[arr_name] = arr_name
            elif stage_nontransients:
                is_read = descriptor_is_read(sdfg, arr_name)
                is_written = descriptor_is_written(sdfg, arr_name)
                # The staged scalar is what the BODY accesses, and the copy edges below are the
                # transfer; inheriting device storage makes every host reference to it invalid.
                storage = arr.storage if descriptor_written_by_gpu_map(sdfg, arr_name) else dtypes.StorageType.Default
                # Fresh name every time (find_new_name): a re-run over an already-staged array never
                # collides with the scalar an earlier run created.
                scal_name, _ = sdfg.add_scalar(f'scal_{arr_name}',
                                               dtype=arr.dtype,
                                               storage=storage,
                                               transient=True,
                                               lifetime=arr.lifetime,
                                               debuginfo=arr.debuginfo,
                                               find_new_name=True)
                rename[arr_name] = scal_name
                staged.append((arr_name, scal_name, is_read, is_written))

        # Rewrite every body reference of a rewritten descriptor to its target name, collapsing the
        # length-1 subset to the scalar element.
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode) and node.data in rename:
                    node.data = rename[node.data]
            for edge in state.edges():
                repoint_memlet_to_element(edge, rename)

        rewrite_code_slots(sdfg, lambda text: rewrite_refs(text, rename))

        # Wire copy-in / copy-out for the staged non-transients. One shared start/sink state holds all.
        if staged:
            copyin = _copyin_state(sdfg) if any(r for _, _, r, _ in staged) else None
            copyout = _copyout_state(sdfg) if any(w for _, _, _, w in staged) else None
            for arr_name, scal_name, is_read, is_written in staged:
                if is_read:
                    a = copyin.add_read(arr_name)
                    s = copyin.add_write(scal_name)
                    copyin.add_nedge(a, s, Memlet(data=arr_name, subset='0'))
                if is_written:
                    s = copyout.add_read(scal_name)
                    a = copyout.add_write(arr_name)
                    copyout.add_nedge(s, a, Memlet(data=arr_name, subset='0'))

        # Offset / dimension symbols carried purely for the rewritten arrays are now dead; drop them so
        # the signature shrinks. ``used_symbols(all_symbols=True)`` covers every reference site.
        referenced: Set[str] = {str(s) for s in sdfg.used_symbols(all_symbols=True)}
        for nm in list(sdfg.symbols):
            if nm in referenced:
                continue
            prefixes = [f'offset_{a}_d' for a in rename] + [f'{a}_d' for a in rename]
            if any(nm.startswith(p) for p in prefixes):
                sdfg.symbols.pop(nm, None)

        if self.recursive:
            for state in sdfg.all_states():
                for node in state.nodes():
                    if isinstance(node, nodes.NestedSDFG):
                        # Nested-SDFG recursion is transient-only (a non-transient inner arg belongs to
                        # the parent's signature) and the filter names outer descriptors, so neither
                        # staging nor filtering applies inside.
                        self._rewrite(node.sdfg, apply_filter=False, stage_nontransients=False)

        return set(rename)

    def apply_pass(self, sdfg: SDFG, _: dict) -> Optional[Set[str]]:
        rewritten = self._rewrite(sdfg, apply_filter=True, stage_nontransients=self.preserve_abi)
        return rewritten or None


@properties.make_properties
@transformation.explicit_cf_compatible
class ConvertScalarsToLengthOneArrays(ppl.Pass):
    """Inverse of ``ConvertLengthOneArraysToScalars``: rewrite every ``Scalar`` to a length-1 ``Array``
    (shape ``(1,)``). Transient scalars are rewritten in place; a non-transient scalar is STAGED into a
    fresh transient length-1 array (with copy-in/out) only when ``preserve_abi`` is set. Useful when a
    consumer requires a 1-element buffer rather than a by-value scalar.

    ``Scalar`` data of an ``opaque`` dtype (an external handle such as ``MPI_Request`` / ``MPI_Comm``)
    is left untouched, the symmetric counterpart of the ``opaque`` exemption in the forward pass.

    :param recursive: Recurse into nested SDFGs (transient-only there).
    :param preserve_abi: Also convert non-transient (signature) scalars, keeping the ABI intact: the
        scalar stays on the signature and is STAGED into a fresh transient length-1 array with
        copy-in/out, never rewritten. Default ``False`` -- only transient scalars are arrayized.
    :param filter: Optional whitelist NARROWING which top-level descriptors are eligible (mirrors the
        forward pass). ``None`` (default) -- no restriction; an empty set rewrites nothing.
    """

    recursive = properties.Property(dtype=bool, default=True, desc="Recurse into nested SDFGs (transient-only there).")
    preserve_abi = properties.Property(
        dtype=bool,
        default=False,
        desc="Convert non-transient scalars too, keeping the ABI intact by staging them into fresh "
        "transient length-1 arrays (copy-in/out) instead of rewriting the signature scalar. Default "
        "only arrayizes transients.")
    filter = properties.SetProperty(
        element_type=str,
        default=None,
        allow_none=True,
        desc="Optional whitelist restricting which top-level descriptors are eligible. ``None`` -- no "
        "restriction; an empty set rewrites nothing. Does not gate the nested-SDFG recursion.")

    def __init__(self, recursive: bool = True, preserve_abi: bool = False, filter: 'Optional[Set[str]]' = None):
        super().__init__()
        self.recursive = recursive
        self.preserve_abi = preserve_abi
        self.filter = None if filter is None else frozenset(filter)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets | ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _rewrite(self, sdfg: SDFG, apply_filter: bool, stage_nontransients: bool) -> Set[str]:
        """Arrayize scalars in ``sdfg`` (modified in place); mirror of the forward ``_rewrite``.

        :param sdfg: SDFG to rewrite.
        :param apply_filter: Whether the top-level ``filter`` set gates the rewrite here.
        :param stage_nontransients: Whether to stage non-transient scalars (top level only).
        :returns: Names of the descriptors that are now array-referenced in the body.
        """
        rename: Dict[str, str] = {}
        staged: List[Tuple[str, str, bool, bool]] = []  # (scalar_name, array_name, is_read, is_written)

        for name, desc in list(sdfg.arrays.items()):
            if not isinstance(desc, dace.data.Scalar) or isinstance(desc.dtype, dace.dtypes.opaque):
                continue
            if apply_filter and self.filter is not None and name not in self.filter:
                continue
            if desc.transient:
                sdfg.remove_data(name, validate=False)
                sdfg.add_array(name,
                               shape=(1, ),
                               dtype=desc.dtype,
                               storage=desc.storage,
                               transient=True,
                               lifetime=desc.lifetime,
                               debuginfo=desc.debuginfo,
                               find_new_name=False)
                rename[name] = name
            elif stage_nontransients:
                is_read = descriptor_is_read(sdfg, name)
                is_written = descriptor_is_written(sdfg, name)
                # ``find_new_name`` makes add_array return ``(name, desc)``; binding the tuple as the
                # name leaves every rename target a tuple and the first Memlet built from it raises
                # ``Invalid type "tuple" for property data``. The forward pass unpacks the same way.
                arr_name, _ = sdfg.add_array(f'arr_{name}',
                                             shape=(1, ),
                                             dtype=desc.dtype,
                                             storage=desc.storage,
                                             transient=True,
                                             lifetime=desc.lifetime,
                                             debuginfo=desc.debuginfo,
                                             find_new_name=True)
                rename[name] = arr_name
                staged.append((name, arr_name, is_read, is_written))

        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode) and node.data in rename:
                    node.data = rename[node.data]
            for edge in state.edges():
                repoint_memlet_to_element(edge, rename)

        # A bare textual reference now names an Array, so it must index element 0 to stay a value.
        rewrite_code_slots(sdfg, lambda text: rewrite_refs_to_element(text, rename))

        if staged:
            copyin = _copyin_state(sdfg) if any(r for _, _, r, _ in staged) else None
            copyout = _copyout_state(sdfg) if any(w for _, _, _, w in staged) else None
            for scal_name, arr_name, is_read, is_written in staged:
                if is_read:
                    s = copyin.add_read(scal_name)
                    a = copyin.add_write(arr_name)
                    copyin.add_nedge(s, a, Memlet(data=arr_name, subset='0'))
                if is_written:
                    a = copyout.add_read(arr_name)
                    s = copyout.add_write(scal_name)
                    copyout.add_nedge(a, s, Memlet(data=arr_name, subset='0'))

        if self.recursive:
            for state in sdfg.all_states():
                for node in state.nodes():
                    if isinstance(node, nodes.NestedSDFG):
                        self._rewrite(node.sdfg, apply_filter=False, stage_nontransients=False)
        return set(rename)

    def apply_pass(self, sdfg: SDFG, _: dict) -> Optional[Set[str]]:
        rewritten = self._rewrite(sdfg, apply_filter=True, stage_nontransients=self.preserve_abi)
        return rewritten or None
