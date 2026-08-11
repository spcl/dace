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
import re
from typing import Callable, Dict, List, Optional, Set, Tuple

import dace
from dace import Memlet, properties, subsets
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


def _rewrite_refs(expr: str, rename: Dict[str, str]) -> str:
    """Rewrite references to rewritten descriptors in a source ``expr``.

    For each ``old -> new`` in ``rename``, collapse ``old[0]`` to ``new`` (the redundant length-1
    accessor) and rename a bare ``old`` to ``new``. Only a token not preceded by a word character or
    ``.`` is matched, so a literal ``[0]`` on a different descriptor whose name merely ends in ``old``
    (``bar[0]`` vs rewritten ``ar``) keeps its subscript. When ``old == new`` (an in-place rewrite)
    this only strips the ``[0]``.

    :param expr: source expression to rewrite.
    :param rename: mapping from each rewritten descriptor's old name to its new name.
    :returns: ``expr`` with each ``old[0]`` / ``old`` rewritten to ``new``.
    """
    for old, new in rename.items():
        expr = re.sub(rf'(?<![\w.]){re.escape(old)}\[0\]', new, expr)
        if old != new:
            expr = re.sub(rf'(?<![\w.]){re.escape(old)}\b', new, expr)
    return expr


def rewrite_refs_to_element(expr: str, rename: Dict[str, str]) -> str:
    """Inverse of :func:`_rewrite_refs`: point a bare reference at element 0 of a now-length-1 array.

    For each ``old -> new`` in ``rename``, a bare ``old`` that is not already subscripted becomes
    ``new[0]``. A token preceded by a word character or ``.`` is skipped, mirroring the forward
    direction, and an already-subscripted ``old[...]`` is left alone so the rewrite is idempotent.

    :param expr: source expression to rewrite.
    :param rename: mapping from each rewritten descriptor's old name to its new name.
    :returns: ``expr`` with each bare ``old`` rewritten to ``new[0]``.
    """
    for old, new in rename.items():
        expr = re.sub(rf'(?<![\w.]){re.escape(old)}(?!\s*\[)\b', f'{new}[0]', expr)
    return expr


def _descriptor_is_read(sdfg: SDFG, name: str) -> bool:
    """True if ``name`` is read anywhere in ``sdfg`` (some AccessNode of it has an out-edge)."""
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode) and node.data == name and state.out_degree(node) > 0:
                return True
    return False


def _descriptor_is_written(sdfg: SDFG, name: str) -> bool:
    """True if ``name`` is written anywhere in ``sdfg`` (some AccessNode of it has an in-edge)."""
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode) and node.data == name and state.in_degree(node) > 0:
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

    def __init__(self,
                 recursive: bool = True,
                 preserve_abi: bool = False,
                 filter: 'Optional[Set[str]]' = None,
                 single_element: bool = False):
        super().__init__()
        self.recursive = recursive
        self.preserve_abi = preserve_abi
        self.filter = None if filter is None else frozenset(filter)
        self.single_element = single_element

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
        blocked = self._blocked_sources(sdfg)
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
                is_read = _descriptor_is_read(sdfg, arr_name)
                is_written = _descriptor_is_written(sdfg, arr_name)
                # Fresh name every time (find_new_name): a re-run over an already-staged array never
                # collides with the scalar an earlier run created.
                scal_name, _ = sdfg.add_scalar(f'scal_{arr_name}',
                                               dtype=arr.dtype,
                                               storage=arr.storage,
                                               transient=True,
                                               lifetime=arr.lifetime,
                                               debuginfo=arr.debuginfo,
                                               find_new_name=True)
                rename[arr_name] = scal_name
                staged.append((arr_name, scal_name, is_read, is_written))

        # Rewrite every body reference of a rewritten descriptor to its target name, collapsing the
        # length-1 subset to the scalar element. AccessNodes carry the data name; memlets carry it
        # plus a subset; interstate/condition/loop code carry it textually.
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode) and node.data in rename:
                    node.data = rename[node.data]
            for edge in state.edges():
                mem = edge.data
                if mem is None or mem.data is None:
                    continue
                if mem.data in rename:
                    edge.data = Memlet(data=rename[mem.data], subset='0', wcr=mem.wcr, dynamic=mem.dynamic)
                    continue
                # A copy edge names only ONE side; the opposite is ``other_subset``. When THAT side is
                # a rewritten descriptor, its subset collapses too, else validation rejects the rank.
                if mem.other_subset is not None and any(
                        isinstance(n, nodes.AccessNode) and n.data in rename.values() and n.data != mem.data
                        for n in (edge.src, edge.dst)):
                    mem.other_subset = subsets.Range.from_string('0')

        rewrite_code_slots(sdfg, lambda text: _rewrite_refs(text, rename))

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
                is_read = _descriptor_is_read(sdfg, name)
                is_written = _descriptor_is_written(sdfg, name)
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

        # Re-point every body reference of a rewritten scalar at element 0 of its length-1 array.
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode) and node.data in rename:
                    node.data = rename[node.data]
            for edge in state.edges():
                mem = edge.data
                if mem is None or mem.data is None or mem.data not in rename:
                    continue
                edge.data = Memlet(data=rename[mem.data], subset='0', wcr=mem.wcr, dynamic=mem.dynamic)

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
