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
for a ``Scalar``, and a by-value scalar cannot receive a written-back result). ``preserve_abi``
(default ``True``) forbids that: under it no top-level non-transient descriptor is ever rewritten, so
the signature is byte-identical after the pass. A non-transient is instead STAGED, only when
``stage_nontransients_arrays_into_scalars`` is set:

* the signature descriptor ``alpha`` (a length-1 ``Array``) is KEPT,
* a fresh transient ``Scalar`` ``scal_alpha`` is introduced and every body reference to ``alpha`` is
  repointed to it,
* a copy-IN ``scal_alpha = alpha[0]`` is prepended in a new start state (only if ``alpha`` is read),
* a copy-OUT ``alpha[0] = scal_alpha`` is appended in a new sink state (only if ``alpha`` is written).

The whole SDFG body then computes on the plain scalar while the signature is unchanged. The inverse
pass stages a non-transient ``Scalar`` into a length-1 ``Array`` the same way. The fresh descriptor is
always allocated with ``find_new_name`` so repeated forward/inverse application never collides on a
name it created earlier.

Clearing ``preserve_abi`` is the opt-in for the opposite intent: a top-level non-transient is then
rewritten IN PLACE and the SDFG's call signature changes with it. That is what the HLFIR Fortran
frontend wants from ``ConvertLengthOneArraysToScalars`` as a post-generation cleanup -- ``Scalar``
data on the signature binds to a plain Python ``int`` / ``float`` whereas a length-1 ``Array`` needs
a 1-element numpy buffer -- so the caller is rewritten to match. Because the descriptor a
:class:`~dace.sdfg.nodes.NestedSDFG` sees through a connector is a separate descriptor of the nested
SDFG, an in-place rewrite is propagated to it; otherwise the two sides of the connector would
disagree on the rank. Nested non-transients are never converted on their own -- they are a nested
SDFG's own signature, owned by the parent's memlets.
"""
import re
from typing import Dict, List, Optional, Set, Tuple

import dace
from dace import Memlet, properties, subsets
from dace.properties import CodeBlock
from dace.sdfg import SDFG, SDFGState, InterstateEdge, nodes, utils as sdutil
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation


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


def _connector_image(sdfg: SDFG, names: Set[str]) -> Dict[nodes.NestedSDFG, Set[str]]:
    """Per :class:`~dace.sdfg.nodes.NestedSDFG` node, the names of ITS descriptors bound through a
    connector to one of ``names`` in the parent.

    A nested SDFG's connector descriptor is a separate descriptor object, so an in-place rank change
    on the parent's side does not reach it. Rewriting only one side leaves the connector's two ends
    disagreeing on the rank, which validation rejects."""
    image: Dict[nodes.NestedSDFG, Set[str]] = {}
    if not names:
        return image
    for state in sdfg.all_states():
        for node in state.nodes():
            if not isinstance(node, nodes.NestedSDFG):
                continue
            bound = [e.dst_conn for e in state.in_edges(node) if e.data is not None and e.data.data in names]
            bound += [e.src_conn for e in state.out_edges(node) if e.data is not None and e.data.data in names]
            for conn in bound:
                if conn:
                    image.setdefault(node, set()).add(conn)
    return image


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
    ``stage_nontransients_arrays_into_scalars`` is set -- see the module docstring.

    Length-1 arrays of an ``opaque`` dtype (an external handle such as ``MPI_Request`` / ``MPI_Comm``)
    are left untouched: a consumer that takes the handle through a pointer connector needs the source
    to stay an ``Array`` so it lowers to a pointer. A ``View`` (or a length-1 array that backs one) is
    left untouched too: a ``Scalar`` cannot carry the ``views`` alias edge, and a scalar source is
    emitted ``const`` so a write through the view fails to compile.

    :param recursive: Recurse into nested SDFGs (only their transient length-1 arrays are rewritten --
        a non-transient nested-SDFG arg is part of its parent's signature).
    :param stage_nontransients_arrays_into_scalars: Also stage non-transient length-1 arrays into
        fresh transient scalars with a copy-in (first state) and copy-out (last state), leaving the
        signature array in place. Default ``False`` -- only transient arrays are scalarized.
    :param preserve_abi: Never rewrite a top-level non-transient descriptor, so the SDFG's call
        signature is byte-identical after the pass; a non-transient is converted only by STAGING
        (above). Default ``True``. Clearing it rewrites the non-transient in place -- the signature
        changes to a by-value scalar and every caller must be rewritten to match.
    :param filter: Optional whitelist NARROWING which top-level descriptors are eligible. ``None``
        (default) -- no restriction. A set -- rewrite ONLY named descriptors that are ALSO eligible
        under the other gates; an empty set rewrites nothing. Gates the top-level rewrite only, not the
        nested-SDFG transient recursion.
    :param single_element: Also rewrite a higher-rank single-element array (every dim == 1, e.g. a
        ``(1, 1)`` map-fusion scratch buffer), not just a rank-1 length-1 array.
    """

    recursive = properties.Property(dtype=bool, default=True, desc="Recurse into nested SDFGs (transient-only there).")
    stage_nontransients_arrays_into_scalars = properties.Property(
        dtype=bool,
        default=False,
        desc="Stage non-transient length-1 arrays into fresh transient scalars (copy-in/out), keeping "
        "the signature array. Default only scalarizes transients.")
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
    preserve_abi = properties.Property(
        dtype=bool,
        default=True,
        desc="Never rewrite a top-level non-transient descriptor, keeping the SDFG signature "
        "byte-identical; such a descriptor is converted only by staging. Clearing this rewrites it in "
        "place and changes the call signature.")

    def __init__(self,
                 recursive: bool = True,
                 stage_nontransients_arrays_into_scalars: bool = False,
                 filter: 'Optional[Set[str]]' = None,
                 single_element: bool = False,
                 preserve_abi: bool = True):
        super().__init__()
        self.recursive = recursive
        self.stage_nontransients_arrays_into_scalars = stage_nontransients_arrays_into_scalars
        self.filter = None if filter is None else frozenset(filter)
        self.single_element = single_element
        self.preserve_abi = preserve_abi

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
        if not isinstance(arr, dace.data.Array) or isinstance(arr, dace.data.View):
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

    def _rewrite(self,
                 sdfg: SDFG,
                 apply_filter: bool,
                 stage_nontransients: bool,
                 force_names: 'Optional[Set[str]]' = None) -> Set[str]:
        """Scalarize length-1 arrays in ``sdfg`` (modified in place).

        Transient arrays are scalarized in place (same name). If ``stage_nontransients``, each
        non-transient length-1 array is converted too -- by staging into a fresh transient scalar with
        copy-in/out under ``preserve_abi``, or in place (changing the signature) without it.

        :param sdfg: SDFG to rewrite.
        :param apply_filter: Whether the top-level ``filter`` set gates the rewrite here (``False`` in
            the nested-SDFG recursion: the filter names outer-level descriptors).
        :param stage_nontransients: Whether to convert non-transient arrays (top level only).
        :param force_names: Non-transient descriptors to rewrite in place regardless of
            ``stage_nontransients``. Set only by the recursion, to carry a parent's in-place rewrite
            through to the nested SDFG's side of the connector.
        :returns: Names of the descriptors that are now scalar-referenced in the body.
        """
        force_names = force_names or frozenset()
        blocked = self._blocked_sources(sdfg)
        # rename[old] = the name the body should reference after the rewrite (== old for a transient
        # scalarized in place; a fresh scalar name for a staged non-transient). staged carries the
        # kept signature array plus its read/write direction so copy-in/out can be wired afterwards.
        rename: Dict[str, str] = {}
        staged: List[Tuple[str, str, bool, bool]] = []  # (array_name, scalar_name, is_read, is_written)
        # Non-transients rewritten in place here; their nested-SDFG connector image must follow.
        signature_rewritten: Set[str] = set()

        for arr_name, arr in list(sdfg.arrays.items()):
            if not self._is_eligible(sdfg, arr_name, arr, blocked, apply_filter):
                continue
            in_place = arr.transient or arr_name in force_names or (stage_nontransients and not self.preserve_abi)
            if in_place:
                if not arr.transient:
                    signature_rewritten.add(arr_name)
                sdfg.remove_data(arr_name, validate=False)
                sdfg.add_scalar(arr_name,
                                dtype=arr.dtype,
                                storage=arr.storage,
                                transient=arr.transient,
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

        for edge in sdfg.all_interstate_edges():
            edge.data.assignments = {k: _rewrite_refs(v, rename) for k, v in edge.data.assignments.items()}
        for node in sdfg.all_control_flow_blocks():
            if isinstance(node, ConditionalBlock):
                for cond, _body in node.branches:
                    if isinstance(cond, CodeBlock):
                        cond.as_string = _rewrite_refs(cond.as_string, rename)
        for node in sdfg.all_control_flow_regions():
            if isinstance(node, LoopRegion):
                cond = node.loop_condition
                src = _rewrite_refs(cond.as_string if isinstance(cond, CodeBlock) else str(cond), rename)
                if isinstance(cond, CodeBlock):
                    cond.as_string = src
                else:
                    node.loop_condition = CodeBlock(src, dace.dtypes.Language.Python)

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
            image = _connector_image(sdfg, signature_rewritten)
            for state in sdfg.all_states():
                for node in state.nodes():
                    if isinstance(node, nodes.NestedSDFG):
                        # Nested-SDFG recursion is transient-only (a non-transient inner arg belongs to
                        # the parent's signature) and the filter names outer descriptors, so neither
                        # staging nor filtering applies inside. The one exception is the connector image
                        # of a descriptor just rewritten in place here: both sides must change together.
                        self._rewrite(node.sdfg,
                                      apply_filter=False,
                                      stage_nontransients=False,
                                      force_names=image.get(node))

        return set(rename)

    def apply_pass(self, sdfg: SDFG, _: dict) -> Optional[Set[str]]:
        rewritten = self._rewrite(sdfg,
                                  apply_filter=True,
                                  stage_nontransients=self.stage_nontransients_arrays_into_scalars)
        return rewritten or None


@properties.make_properties
@transformation.explicit_cf_compatible
class ConvertScalarsToLengthOneArrays(ppl.Pass):
    """Inverse of ``ConvertLengthOneArraysToScalars``: rewrite every ``Scalar`` to a length-1 ``Array``
    (shape ``(1,)``). Transient scalars are rewritten in place; a non-transient scalar is STAGED into a
    fresh transient length-1 array (with copy-in/out) only when
    ``stage_nontransients_arrays_into_scalars`` is set. Useful when a consumer requires a 1-element
    buffer rather than a by-value scalar.

    ``Scalar`` data of an ``opaque`` dtype (an external handle such as ``MPI_Request`` / ``MPI_Comm``)
    is left untouched, the symmetric counterpart of the ``opaque`` exemption in the forward pass.

    :param recursive: Recurse into nested SDFGs (transient-only there).
    :param stage_nontransients_arrays_into_scalars: Also stage non-transient scalars into fresh
        transient length-1 arrays with copy-in/out, leaving the signature scalar in place. Default
        ``False`` -- only transient scalars are arrayized.
    :param filter: Optional whitelist NARROWING which top-level descriptors are eligible (mirrors the
        forward pass). ``None`` (default) -- no restriction; an empty set rewrites nothing.
    :param preserve_abi: Never rewrite a top-level non-transient descriptor, so the SDFG's call
        signature is byte-identical after the pass; a non-transient is converted only by STAGING.
        Default ``True``. Clearing it rewrites the non-transient in place -- the signature changes to
        a 1-element buffer and every caller must be rewritten to match.
    """

    recursive = properties.Property(dtype=bool, default=True, desc="Recurse into nested SDFGs (transient-only there).")
    stage_nontransients_arrays_into_scalars = properties.Property(
        dtype=bool,
        default=False,
        desc="Stage non-transient scalars into fresh transient length-1 arrays (copy-in/out), keeping "
        "the signature scalar. Default only arrayizes transients.")
    filter = properties.SetProperty(
        element_type=str,
        default=None,
        allow_none=True,
        desc="Optional whitelist restricting which top-level descriptors are eligible. ``None`` -- no "
        "restriction; an empty set rewrites nothing. Does not gate the nested-SDFG recursion.")
    preserve_abi = properties.Property(
        dtype=bool,
        default=True,
        desc="Never rewrite a top-level non-transient descriptor, keeping the SDFG signature "
        "byte-identical; such a descriptor is converted only by staging. Clearing this rewrites it in "
        "place and changes the call signature.")

    def __init__(self,
                 recursive: bool = True,
                 stage_nontransients_arrays_into_scalars: bool = False,
                 filter: 'Optional[Set[str]]' = None,
                 preserve_abi: bool = True):
        super().__init__()
        self.recursive = recursive
        self.stage_nontransients_arrays_into_scalars = stage_nontransients_arrays_into_scalars
        self.filter = None if filter is None else frozenset(filter)
        self.preserve_abi = preserve_abi

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets | ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _rewrite(self,
                 sdfg: SDFG,
                 apply_filter: bool,
                 stage_nontransients: bool,
                 force_names: 'Optional[Set[str]]' = None) -> Set[str]:
        """Arrayize scalars in ``sdfg`` (modified in place); mirror of the forward ``_rewrite``.

        :param sdfg: SDFG to rewrite.
        :param apply_filter: Whether the top-level ``filter`` set gates the rewrite here.
        :param stage_nontransients: Whether to convert non-transient scalars (top level only).
        :param force_names: Non-transient descriptors to rewrite in place regardless of
            ``stage_nontransients``, carrying a parent's in-place rewrite through to the nested SDFG's
            side of the connector.
        :returns: Names of the descriptors that are now array-referenced in the body.
        """
        force_names = force_names or frozenset()
        rename: Dict[str, str] = {}
        staged: List[Tuple[str, str, bool, bool]] = []  # (scalar_name, array_name, is_read, is_written)
        signature_rewritten: Set[str] = set()

        for name, desc in list(sdfg.arrays.items()):
            if not isinstance(desc, dace.data.Scalar) or isinstance(desc.dtype, dace.dtypes.opaque):
                continue
            if apply_filter and self.filter is not None and name not in self.filter:
                continue
            in_place = desc.transient or name in force_names or (stage_nontransients and not self.preserve_abi)
            if in_place:
                if not desc.transient:
                    signature_rewritten.add(name)
                sdfg.remove_data(name, validate=False)
                sdfg.add_array(name,
                               shape=(1, ),
                               dtype=desc.dtype,
                               storage=desc.storage,
                               transient=desc.transient,
                               lifetime=desc.lifetime,
                               debuginfo=desc.debuginfo,
                               find_new_name=False)
                rename[name] = name
            elif stage_nontransients:
                is_read = _descriptor_is_read(sdfg, name)
                is_written = _descriptor_is_written(sdfg, name)
                arr_name = sdfg.add_array(f'arr_{name}',
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
            image = _connector_image(sdfg, signature_rewritten)
            for state in sdfg.all_states():
                for node in state.nodes():
                    if isinstance(node, nodes.NestedSDFG):
                        self._rewrite(node.sdfg,
                                      apply_filter=False,
                                      stage_nontransients=False,
                                      force_names=image.get(node))
        return set(rename)

    def apply_pass(self, sdfg: SDFG, _: dict) -> Optional[Set[str]]:
        rewritten = self._rewrite(sdfg,
                                  apply_filter=True,
                                  stage_nontransients=self.stage_nontransients_arrays_into_scalars)
        return rewritten or None
