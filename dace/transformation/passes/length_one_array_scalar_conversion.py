# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Passes that move data between length-1 ``Array`` and ``Scalar`` form.

``ConvertLengthOneArraysToScalars`` rewrites every length-1 ``Array``
(shape ``(1,)``) to a true ``Scalar`` and drops the now-redundant
``[0]`` accessors from interstate-edge assignments, conditional-block
guards, loop-region conditions and memlet subsets.
``ConvertScalarsToLengthOneArrays`` is the inverse (``Scalar`` ->
length-1 ``Array``).

The HLFIR Fortran frontend uses ``ConvertLengthOneArraysToScalars`` as
a post-generation cleanup: ``Scalar`` data on the SDFG signature binds
to a plain Python ``int`` / ``float`` whereas a length-1 ``Array``
needs a 1-element numpy buffer, so this moves bridge outputs/locals
from the latter to the former wherever it is safe.
"""
import re
from typing import Optional, Set

import dace
from dace import Memlet, properties
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation


def _strip_elem_zero(expr: str, names: Set[str]) -> str:
    """Drop the redundant ``[0]`` accessor from references to scalarized ``names`` in ``expr``.

    Only a ``name[0]`` not preceded by a word character or ``.`` is rewritten,
    so a literal ``[0]`` index on a different, non-scalarized array whose name
    ends in one of ``names`` (e.g. ``bar[0]`` against scalarized ``ar``) keeps
    its subscript.

    :param expr: source expression to rewrite.
    :param names: names of the arrays that were scalarized.
    :returns: ``expr`` with ``name[0]`` collapsed to ``name`` for each scalarized name.
    """
    for nm in names:
        expr = re.sub(rf'(?<![\w.]){re.escape(nm)}\[0\]', nm, expr)
    return expr


@properties.make_properties
@transformation.explicit_cf_compatible
class ConvertLengthOneArraysToScalars(ppl.Pass):
    """Rewrite every length-1 ``Array`` (shape ``(1,)``) to a true
    ``Scalar`` of the same dtype, and drop the ``[0]`` accessors that
    referenced it from interstate-edge assignments, conditional-block
    branch guards, loop-region conditions and memlet subsets.

    Length-1 arrays of an ``opaque`` dtype (an external handle such as
    ``MPI_Request`` / ``MPI_Comm``) are left untouched: a consumer that
    takes the handle through a pointer connector (e.g. a ``dace``
    ``Wait`` / ``Isend`` library node) needs the source to stay an
    ``Array`` so it lowers to ``DefinedType.Pointer`` and decays to a
    pointer.  A scalarized opaque source is copied by value instead,
    which miscompiles (``MPI_Wait`` wants ``MPI_Request*``, not a
    ``MPI_Request`` copy).

    :param recursive: Recurse into nested SDFGs (only their TRANSIENT
        length-1 arrays are rewritten -- a non-transient nested-SDFG
        arg is part of its parent's signature and rewriting it would
        change the caller's contract).
    :param transient_only: Restrict the top-level rewrite to transient
        arrays (default ``False`` -- both signature and local rewrites).
    :param keep_program_outputs: When ``transient_only`` is ``False``, keep a
        non-transient length-1 array that the SDFG WRITES (a program output --
        the caller passes a 1-element numpy buffer to receive the return) as an
        ``Array``, and scalarize every other length-1 array (transients and
        read-only non-transient inputs). Has no effect when ``transient_only``
        is ``True`` (no non-transient is rewritten then anyway).
    :param filter: When ``None`` (default), scalarize every eligible top-level
        array as governed by ``transient_only``. When a set is provided,
        scalarize *only* named arrays -- and being in the filter overrides
        ``transient_only``. Has no effect on the nested-SDFG transient-only
        recursion.
    """

    recursive = properties.Property(dtype=bool, default=True, desc="Recurse into nested SDFGs (transient-only there).")
    transient_only = properties.Property(dtype=bool,
                                         default=False,
                                         desc="Restrict the top-level rewrite to transient arrays.")
    keep_program_outputs = properties.Property(
        dtype=bool,
        default=False,
        desc="Keep written non-transient length-1 arrays (program outputs) as Arrays; scalarize the rest.")
    filter = properties.SetProperty(
        element_type=str,
        default=None,
        allow_none=True,
        desc="If ``None``, no filtering -- every eligible array is scalarized. If a set is "
        "provided, only top-level arrays whose name appears in it are scalarized, and being "
        "in the filter overrides the ``transient_only`` check.")
    single_element = properties.Property(
        dtype=bool,
        default=False,
        desc="Also scalarize higher-rank single-element arrays (every dim == 1, e.g. a (1, 1) "
        "map-fusion scratch buffer), not just rank-1 length-1 arrays.")

    def __init__(self,
                 recursive: bool = True,
                 transient_only: bool = False,
                 keep_program_outputs: bool = False,
                 filter: 'Optional[Set[str]]' = None,
                 single_element: bool = False):
        super().__init__()
        self.recursive = recursive
        self.transient_only = transient_only
        self.keep_program_outputs = keep_program_outputs
        self.filter = None if filter is None else frozenset(filter)
        self.single_element = single_element

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets | ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _rewrite(self, sdfg: dace.SDFG, transient_only: bool, apply_filter: bool) -> Set[str]:
        """Scalarize length-1 arrays in ``sdfg`` (modified in place).

        :param sdfg: SDFG to rewrite.
        :param transient_only: Restrict the rewrite to transient arrays.
        :param apply_filter: Whether the top-level ``filter`` set gates the rewrite here.
                             ``False`` for the nested-SDFG recursion: the filter refers to
                             outer-level names, not inner descriptors.
        :returns: Names of the arrays that were scalarized.
        """
        scalarized: Set[str] = set()
        # Program outputs (non-transient arrays the SDFG writes) must stay Arrays: the caller passes a
        # 1-element numpy buffer to receive the return, which a scalar-by-value output cannot fill.
        program_outputs: Set[str] = set()
        if not transient_only and self.keep_program_outputs:
            for state in sdfg.states():
                for node in state.data_nodes():
                    if state.in_degree(node) > 0 and not sdfg.arrays[node.data].transient:
                        program_outputs.add(node.data)
        # Arrays written through a nested SDFG connector must stay Arrays: a nested SDFG argument is
        # a pointer/reference, but a scalar is passed by value -- scalarizing a written array would
        # silently drop the write. Collect the out-edge data of every NestedSDFG node.
        nsdfg_written: Set[str] = set()
        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    for edge in state.out_edges(node):
                        if edge.data is not None and edge.data.data is not None:
                            nsdfg_written.add(edge.data.data)
        for arr_name, arr in list(sdfg.arrays.items()):
            if not isinstance(arr, dace.data.Array):
                continue
            is_len1 = arr.shape == (1, ) or arr.shape == [1]
            # ``single_element`` additionally scalarizes a higher-rank all-ones array (e.g. a (1, 1)
            # map-fusion scratch buffer), not just a rank-1 length-1 array.
            is_single = self.single_element and len(arr.shape) >= 1 and all(d == 1 for d in arr.shape)
            if not (is_len1 or is_single):
                continue
            if arr_name in program_outputs or arr_name in nsdfg_written:
                continue
            if isinstance(arr.dtype, dace.dtypes.opaque):
                continue
            # Firewall: a length-1 dim marked with the ``ONE`` broadcast sentinel
            # (design 3.8.2) must stay an Array -- scalarising would erase the
            # broadcast intent that downstream gather / scatter lib-node lowerings
            # need. The identity check survives sympy round-trips.
            from dace.symbolic import ONE
            if any(ONE in dim.free_symbols for dim in arr.shape if hasattr(dim, "free_symbols")):
                continue
            in_filter = apply_filter and self.filter is not None and arr_name in self.filter
            if apply_filter and self.filter is not None and not in_filter:
                continue
            if transient_only and not arr.transient and not in_filter:
                continue
            sdfg.remove_data(arr_name, validate=False)
            sdfg.add_scalar(name=arr_name,
                            dtype=arr.dtype,
                            storage=arr.storage,
                            transient=arr.transient,
                            lifetime=arr.lifetime,
                            debuginfo=arr.debuginfo,
                            find_new_name=False)
            scalarized.add(arr_name)

        # Strip ``[0]`` from interstate-edge assignment RHSs.
        for edge in sdfg.all_interstate_edges():
            edge.data.assignments = {k: _strip_elem_zero(v, scalarized) for k, v in edge.data.assignments.items()}

        # Strip ``[0]`` from conditional-block branch guards.
        for node in sdfg.all_control_flow_blocks():
            if isinstance(node, ConditionalBlock):
                for cond, _body in node.branches:
                    if isinstance(cond, CodeBlock):
                        cond.as_string = _strip_elem_zero(cond.as_string, scalarized)

        # Strip ``[0]`` from loop-region condition expressions.
        for node in sdfg.all_control_flow_regions():
            if isinstance(node, LoopRegion):
                cond = node.loop_condition
                src = _strip_elem_zero(cond.as_string if isinstance(cond, CodeBlock) else str(cond), scalarized)
                if isinstance(cond, CodeBlock):
                    cond.as_string = src
                else:
                    node.loop_condition = CodeBlock(src, dace.dtypes.Language.Python)

        # Any subset (not just ``[0]``) of a scalarized array collapses to scalar
        # ``0``: the bridge sometimes synthesises ``arr[(je) - offset_arr_d0]`` even
        # for size-1 arrays, and every subset resolves to the single element.
        for state in sdfg.all_states():
            for edge in state.edges():
                mem = edge.data
                if mem is None or mem.data is None:
                    continue
                if mem.data not in scalarized:
                    continue
                edge.data = Memlet(data=mem.data, subset='0', wcr=mem.wcr, dynamic=mem.dynamic)

        # Offset / dimension symbols carried purely for the rewritten arrays are
        # now dead; drop them so the signature shrinks and codegen doesn't pass
        # unused parameters. ``used_symbols(all_symbols=True)`` covers every site a
        # symbol can be referenced (shapes, bounds, tasklets, memlets, interstate
        # edges), so only genuinely-dead offset/dim symbols are removed.
        referenced: Set[str] = {str(s) for s in sdfg.used_symbols(all_symbols=True)}
        for nm in list(sdfg.symbols):
            if nm in referenced:
                continue
            prefixes = [f'offset_{a}_d' for a in scalarized] + [f'{a}_d' for a in scalarized]
            if any(nm.startswith(p) for p in prefixes):
                sdfg.symbols.pop(nm, None)

        if self.recursive:
            for state in sdfg.all_states():
                for node in state.nodes():
                    if isinstance(node, dace.nodes.NestedSDFG):
                        # ``apply_filter=False`` -- the filter is only meaningful at the root
                        # SDFG level. Inner descriptors follow the transient-only rule.
                        self._rewrite(node.sdfg, transient_only=True, apply_filter=False)

        return scalarized

    def apply_pass(self, sdfg: dace.SDFG, _: dict) -> Optional[Set[str]]:
        rewritten = self._rewrite(sdfg, self.transient_only, apply_filter=True)
        return rewritten or None


@properties.make_properties
@transformation.explicit_cf_compatible
class ConvertScalarsToLengthOneArrays(ppl.Pass):
    """Inverse of ``ConvertLengthOneArraysToScalars``: rewrite every
    ``Scalar`` to a length-1 ``Array`` (shape ``(1,)``).  Useful when a
    consumer requires a 1-element buffer rather than a by-value scalar.

    ``Scalar`` data of an ``opaque`` dtype (an external handle such as
    ``MPI_Request`` / ``MPI_Comm``) is left untouched: opaque handles
    must keep the exact form their producer chose so the
    array-vs-scalar / pointer-vs-value contract with the consuming
    library node is preserved (the symmetric counterpart of the
    ``opaque`` exemption in ``ConvertLengthOneArraysToScalars``).

    :param recursive: Recurse into nested SDFGs (transient-only there).
    :param transient_only: Restrict the top-level rewrite to transient
        scalars.
    """

    recursive = properties.Property(dtype=bool, default=True, desc="Recurse into nested SDFGs (transient-only there).")
    transient_only = properties.Property(dtype=bool,
                                         default=False,
                                         desc="Restrict the top-level rewrite to transient scalars.")

    def __init__(self, recursive: bool = True, transient_only: bool = False):
        super().__init__()
        self.recursive = recursive
        self.transient_only = transient_only

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _rewrite(self, sdfg: dace.SDFG, transient_only: bool) -> Set[str]:
        arrayized: Set[str] = set()
        for name, desc in [(k, v) for k, v in sdfg.arrays.items()]:
            if isinstance(desc.dtype, dace.dtypes.opaque):
                continue
            if isinstance(desc, dace.data.Scalar) and ((not transient_only) or desc.transient):
                sdfg.remove_data(name, validate=False)
                sdfg.add_array(name=name,
                               shape=(1, ),
                               dtype=desc.dtype,
                               storage=desc.storage,
                               transient=desc.transient,
                               lifetime=desc.lifetime,
                               debuginfo=desc.debuginfo,
                               find_new_name=False)
                arrayized.add(name)
        # Re-point scalar memlets at element 0 of the new length-1 array.
        for state in sdfg.all_states():
            for edge in state.edges():
                mem = edge.data
                if mem is None or mem.data is None or mem.data not in arrayized:
                    continue
                edge.data = Memlet(data=mem.data, subset='0', wcr=mem.wcr)
        if self.recursive:
            for state in sdfg.all_states():
                for node in state.nodes():
                    if isinstance(node, dace.nodes.NestedSDFG):
                        self._rewrite(node.sdfg, transient_only=True)
        return arrayized

    def apply_pass(self, sdfg: dace.SDFG, _: dict) -> Optional[Set[str]]:
        rewritten = self._rewrite(sdfg, self.transient_only)
        return rewritten or None
