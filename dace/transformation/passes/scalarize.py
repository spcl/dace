# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Passes that move data between length-1 ``Array`` and ``Scalar`` form.

``ScalarizeLengthOneArrays`` rewrites every length-1 ``Array`` (shape
``(1,)``) to a true ``Scalar`` and drops the now-redundant ``[0]``
accessors from interstate-edge assignments, conditional-block guards,
loop-region conditions and memlet subsets.  ``ArrayizeScalars`` is the
inverse (``Scalar`` -> length-1 ``Array``).

The HLFIR Fortran frontend uses ``ScalarizeLengthOneArrays`` as a
post-generation cleanup: ``Scalar`` data on the SDFG signature binds to
a plain Python ``int`` / ``float`` whereas a length-1 ``Array`` needs a
1-element numpy buffer, so this moves bridge outputs/locals from the
latter to the former wherever it is safe.

The core rewrite is also exposed as the module-level function
``replace_length_one_arrays_with_scalars`` (the Pass is a thin wrapper)
so callers that need the rewritten-name set directly, and the existing
unit test, keep a stable entry point.
"""
from typing import Optional, Set

import dace
from dace import Memlet, properties
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation


def replace_length_one_arrays_with_scalars(sdfg: dace.SDFG,
                                           recursive: bool = True,
                                           transient_only: bool = False) -> Set[str]:
    """Rewrite every length-1 ``Array`` (shape ``(1,)``) on ``sdfg`` to
    a true ``Scalar`` of the same dtype, and drop the ``[0]`` accessors
    from interstate-edge assignments, conditional-block branch guards,
    and loop-region condition expressions.

    :param sdfg: Top-level SDFG to rewrite in place.
    :param recursive: Recurse into nested SDFGs (only their TRANSIENT
        length-1 arrays get rewritten -- a non-transient nested-SDFG
        arg is part of its parent's signature and rewriting it would
        change the caller's contract).
    :param transient_only: Restrict the top-level pass to transient
        arrays (default ``False`` -- both signature and local rewrites).
    :returns: The set of array names that were rewritten to scalars.
    """
    scalarized: Set[str] = set()
    for arr_name, arr in [(k, v) for k, v in sdfg.arrays.items()]:
        if isinstance(arr, dace.data.Array) and (arr.shape == (1, ) or arr.shape == [1]):
            if (not transient_only) or arr.transient:
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
        new_assigns = {}
        for k, v in edge.data.assignments.items():
            nv = v
            for nm in scalarized:
                if f'{nm}[0]' in nv:
                    nv = nv.replace(f'{nm}[0]', nm)
            new_assigns[k] = nv
        edge.data.assignments = new_assigns

    # Strip ``[0]`` from conditional-block branch guards.
    for node in sdfg.all_control_flow_blocks():
        if isinstance(node, ConditionalBlock):
            for cond, _body in node.branches:
                if cond is None:
                    continue
                src = cond.as_string if isinstance(cond, CodeBlock) else str(cond)
                for nm in scalarized:
                    if f'{nm}[0]' in src:
                        src = src.replace(f'{nm}[0]', nm)
                if isinstance(cond, CodeBlock):
                    cond.as_string = src

    # Strip ``[0]`` from loop-region condition expressions.
    for node in sdfg.all_control_flow_regions():
        if isinstance(node, LoopRegion):
            cond = node.loop_condition
            src = cond.as_string if isinstance(cond, CodeBlock) else str(cond)
            for nm in scalarized:
                if f'{nm}[0]' in src:
                    src = src.replace(f'{nm}[0]', nm)
            if isinstance(cond, CodeBlock):
                cond.as_string = src
            else:
                node.loop_condition = CodeBlock(src, dace.dtypes.Language.Python)

    # Strip ``[<expr>]`` -- any subset, not just ``[0]`` -- from memlet
    # subsets that reference the scalarized arrays.  A length-1 array
    # has a single element, so any subset resolves to that one value;
    # the bridge sometimes synthesises ``arr[(je) - offset_arr_d0]``
    # even for size-1 arrays, so collapse those to a scalar memlet.
    for state in sdfg.all_states():
        for edge in state.edges():
            mem = edge.data
            if mem is None or mem.data is None:
                continue
            if mem.data not in scalarized:
                continue
            edge.data = Memlet(data=mem.data, subset='0', wcr=mem.wcr)

    # The offset / dimension symbols that were carried purely for the
    # rewritten arrays are now dead.  Drop them so the signature shrinks
    # and codegen doesn't pass unused parameters.  Keep symbols still
    # referenced by another array's shape / lower bounds.
    referenced: Set[str] = set()
    for desc in sdfg.arrays.values():
        for s in getattr(desc, 'shape', ()):
            referenced.update(str(x) for x in dace.symbolic.symlist(s).values())
        for s in getattr(desc, 'offset', ()):
            referenced.update(str(x) for x in dace.symbolic.symlist(s).values())
    for nm in list(sdfg.symbols):
        if nm in referenced:
            continue
        prefixes = [f'offset_{a}_d' for a in scalarized] + [f'{a}_d' for a in scalarized]
        if any(nm.startswith(p) for p in prefixes):
            sdfg.symbols.pop(nm, None)

    if recursive:
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    replace_length_one_arrays_with_scalars(node.sdfg, recursive=True, transient_only=True)

    return scalarized


@properties.make_properties
@transformation.explicit_cf_compatible
class ScalarizeLengthOneArrays(ppl.Pass):
    """Rewrite every length-1 ``Array`` to a ``Scalar`` (see
    ``replace_length_one_arrays_with_scalars``).

    :param recursive: Recurse into nested SDFGs (transient-only there).
    :param transient_only: Restrict the top-level rewrite to transient
        arrays.
    """

    recursive = properties.Property(dtype=bool,
                                    default=True,
                                    desc="Recurse into nested SDFGs (transient-only there).")
    transient_only = properties.Property(dtype=bool,
                                         default=False,
                                         desc="Restrict the top-level rewrite to transient arrays.")

    def __init__(self, recursive: bool = True, transient_only: bool = False):
        super().__init__()
        self.recursive = recursive
        self.transient_only = transient_only

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets | ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _: dict) -> Optional[Set[str]]:
        rewritten = replace_length_one_arrays_with_scalars(sdfg,
                                                           recursive=self.recursive,
                                                           transient_only=self.transient_only)
        return rewritten or None


@properties.make_properties
@transformation.explicit_cf_compatible
class ArrayizeScalars(ppl.Pass):
    """Inverse of ``ScalarizeLengthOneArrays``: rewrite every
    ``Scalar`` to a length-1 ``Array`` (shape ``(1,)``).  Useful when a
    consumer requires a 1-element buffer rather than a by-value scalar.

    :param recursive: Recurse into nested SDFGs (transient-only there).
    :param transient_only: Restrict the top-level rewrite to transient
        scalars.
    """

    recursive = properties.Property(dtype=bool,
                                    default=True,
                                    desc="Recurse into nested SDFGs (transient-only there).")
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
