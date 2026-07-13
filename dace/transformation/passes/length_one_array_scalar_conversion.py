# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Passes that move data between length-1 ``Array`` and ``Scalar`` form.

``ConvertLengthOneArraysToScalars`` rewrites every length-1 ``Array``
(shape ``(1,)``) to a true ``Scalar`` and drops the now-redundant
``[0]`` accessors from interstate-edge assignments, conditional-block
guards, loop-region conditions and memlet subsets.

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

    :param expr: Expression source to rewrite.
    :param names: Names of the scalarized (now single-value) descriptors.
    :returns: ``expr`` with the ``[0]`` accessors of ``names`` removed.
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

    :param recursive: Recurse into nested SDFGs (only their TRANSIENT
        length-1 arrays are rewritten -- a non-transient nested-SDFG
        arg is part of its parent's signature and rewriting it would
        change the caller's contract).
    :param transient_only: Restrict the top-level rewrite to transient
        arrays (default ``False`` -- both signature and local rewrites).
    :param filter: When ``None`` (default), the pass scalarizes every
        eligible top-level array as governed by ``transient_only``. When
        a set is provided, the pass scalarizes *only* arrays whose name
        appears in it -- and **being in the filter overrides
        ``transient_only``**, i.e. a named array is always scalarized
        regardless of its ``transient`` flag. The filter has no effect
        on the nested-SDFG transient-only recursion; inner descriptors
        keep following the recursion rule.
    """

    recursive = properties.Property(dtype=bool, default=True, desc="Recurse into nested SDFGs (transient-only there).")
    transient_only = properties.Property(dtype=bool,
                                         default=False,
                                         desc="Restrict the top-level rewrite to transient arrays.")
    filter = properties.SetProperty(
        element_type=str,
        default=None,
        allow_none=True,
        desc="If ``None``, no filtering -- every eligible array is scalarized. If a set is "
        "provided, only top-level arrays whose name appears in it are scalarized, and being "
        "in the filter overrides the ``transient_only`` check.")

    def __init__(self, recursive: bool = True, transient_only: bool = False, filter: 'Optional[Set[str]]' = None):
        super().__init__()
        self.recursive = recursive
        self.transient_only = transient_only
        self.filter = None if filter is None else frozenset(filter)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets | ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _rewrite(self, sdfg: dace.SDFG, transient_only: bool, apply_filter: bool) -> Set[str]:
        """Scalarize length-1 arrays in ``sdfg``.

        :param sdfg: Target SDFG (modified in place).
        :param transient_only: Restrict to transient arrays.
        :param apply_filter: Whether the top-level ``filter`` set should gate the rewrite at
                             this level. ``False`` for the nested-SDFG recursion: the filter
                             refers to outer-level names, not inner descriptors.
        :returns: Names of the arrays that were scalarized.
        """
        scalarized: Set[str] = set()
        for arr_name, arr in list(sdfg.arrays.items()):
            if not (isinstance(arr, dace.data.Array) and (arr.shape == (1, ) or arr.shape == [1])):
                continue
            # ``filter`` semantics: ``None`` -> no filtering (default behaviour);
            # ``set`` -> only names in the set are scalarized, and being in the filter
            # overrides the ``transient_only`` check.
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

        # Strip ``[<expr>]`` -- any subset, not just ``[0]`` -- from
        # memlet subsets that reference the scalarized arrays.  A
        # length-1 array has a single element, so any subset resolves
        # to that one value; the bridge sometimes synthesises
        # ``arr[(je) - offset_arr_d0]`` even for size-1 arrays, so
        # collapse those to a scalar memlet.
        for state in sdfg.all_states():
            for edge in state.edges():
                mem = edge.data
                if mem is None or mem.data is None:
                    continue
                if mem.data not in scalarized:
                    continue
                edge.data = Memlet(data=mem.data, subset='0', wcr=mem.wcr, dynamic=mem.dynamic)

        # The offset / dimension symbols that were carried purely for
        # the rewritten arrays are now dead.  Drop them so the signature
        # shrinks and codegen doesn't pass unused parameters.  Keep
        # symbols still referenced by another array's shape / bounds.
        # A symbol is still needed if it is used anywhere the SDFG references symbols (array
        # shapes / bounds, tasklets, memlets, interstate edges); ``used_symbols(all_symbols=True)``
        # captures all of those, so only the offset/dimension symbols that scalarization left
        # genuinely dead are dropped.
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
