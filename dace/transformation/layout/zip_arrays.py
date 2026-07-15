# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ZipArrays -- the Zip layout primitive: fuse F fully-unzipped (SoA) arrays into one ``Z``.

Start from fully unzipped: F separate arrays ``A_0..A_{F-1}`` of identical logical shape ``S``.
Zip combines them:

  * **Homogeneous fields** (all the same dtype): ``Z`` is a plain array ``[*S, F]`` (field-minor =
    interleaved AoS). An access ``A_f[idx]`` becomes ``Z[idx, f]`` -- a plain scalar access, so the
    tasklets are untouched. This is the DEFAULT.
  * **Heterogeneous fields** ("emit structs"): ``Z`` is a ``dace.data.Structure`` whose members are
    the field arrays. An access ``A_f[idx]`` becomes a DOTTED memlet ``Z.A_f[idx]`` -- just the
    memlet/access-node data name changes (``A_f`` -> ``Z.A_f``); tasklets are untouched. Struct
    ARRAYS (``ContainerArray`` of struct = jagged / array-of-pointers) are NOT supported.

Run after ``prepare_for_layout``.
"""
import copy
from dataclasses import dataclass
from typing import Any, Dict, List

import dace
from dace.sdfg import nodes as nd
from dace.transformation import pass_pipeline as ppl


@dataclass
class ZipArrays(ppl.Pass):
    """Fuse groups of same-shape arrays into field-minor AoS arrays.

    :param zip_map: ``{new_name: [field_array_0, field_array_1, ...]}``. The fields are fused in the
                    given order along a new dimension of extent ``len(fields)``.
    :param field_axis: insertion index of the field dimension in the fused array's shape. ``None``
                       (default) appends it as the trailing dimension (plain AoS, field-minor). Pass
                       ``ndim-1`` on already-blocked ``[*S, V]`` fields to place the field dim BEFORE
                       the lane dim, producing AoSoA ``[*S, F, V]``.
    """

    def __init__(self, zip_map: Dict[str, List[str]], field_axis: int = None):
        self._zip_map = zip_map
        self._field_axis = field_axis

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _check_nested(self, sdfg: dace.SDFG, fields: List[str]):
        for state in sdfg.all_states():
            for n in state.nodes():
                if isinstance(n, nd.NestedSDFG):
                    conns = set(n.in_connectors) | set(n.out_connectors)
                    if any(f in conns for f in fields):
                        raise NotImplementedError("ZipArrays: fields passed to nested SDFGs not supported yet")

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        for new_name, fields in self._zip_map.items():
            descs = [sdfg.arrays[f] for f in fields]
            shape0 = tuple(descs[0].shape)
            for f, d in zip(fields, descs):
                if tuple(d.shape) != shape0:
                    raise ValueError(f"ZipArrays: field '{f}' shape {tuple(d.shape)} != {shape0}")
            self._check_nested(sdfg, fields)

            dtypes = {d.dtype for d in descs}
            if len(dtypes) == 1:
                self._zip_homogeneous(sdfg, new_name, fields, descs, shape0)
            else:
                self._zip_struct(sdfg, new_name, fields, descs)

            for f in fields:
                sdfg.remove_data(f, validate=False)
        return 0

    def _zip_homogeneous(self, sdfg, new_name, fields, descs, shape0):
        """Same-dtype fields -> one plain array with the field dim at ``field_axis`` (default
        trailing = field-minor AoS ``[*S, F]``; ``ndim-1`` on blocked fields = AoSoA ``[*S, F, V]``)."""
        field_index = {f: k for k, f in enumerate(fields)}
        axis = len(shape0) if self._field_axis is None else self._field_axis
        new_shape = list(shape0)
        new_shape.insert(axis, len(fields))
        transient = all(d.transient for d in descs)
        sdfg.add_array(name=new_name,
                       shape=new_shape,
                       dtype=descs[0].dtype,
                       storage=descs[0].storage,
                       transient=transient,
                       lifetime=descs[0].lifetime,
                       find_new_name=False)
        for state in sdfg.all_states():
            for edge in state.edges():
                if edge.data is not None and edge.data.data in field_index:
                    k = field_index[edge.data.data]
                    new_ranges = list(edge.data.subset.ranges)
                    new_ranges.insert(axis, (k, k, 1))
                    edge.data = dace.memlet.Memlet(data=new_name,
                                                   subset=dace.subsets.Range(new_ranges),
                                                   other_subset=None)
            # Point field access nodes at the fused array (connector names are left as-is;
            # they need not match the data name).
            for node in state.nodes():
                if isinstance(node, nd.AccessNode) and node.data in field_index:
                    node.data = new_name

    def _zip_struct(self, sdfg, new_name, fields, descs):
        """Different-dtype fields -> a ``Structure`` bundle, accessed via dotted memlets ``Z.A_f``.

        Only the memlet/access-node DATA name changes (``A_f`` -> ``Z.A_f``); tasklets and subsets
        are untouched. No struct arrays / ContainerArray (no jagged array).
        """
        members = {f: copy.deepcopy(descs[i]) for i, f in enumerate(fields)}
        struct = dace.data.Structure(members, name=new_name)
        sdfg.add_datadesc(new_name, struct)
        rename = {f: f"{new_name}.{f}" for f in fields}
        for state in sdfg.all_states():
            for edge in state.edges():
                if edge.data is not None and edge.data.data in rename:
                    edge.data.data = rename[edge.data.data]
            for node in state.nodes():
                if isinstance(node, nd.AccessNode) and node.data in rename:
                    node.data = rename[node.data]


def aosoa_layout(sdfg: dace.SDFG, new_name: str, fields: List[str], vector_width: int) -> None:
    """Apply the AoSoA layout (``Zip ∘ Block``) to a group of same-shape, same-dtype ``fields``.

    Blocks each field's LAST (particle) axis by ``vector_width`` -- ``[.., N]`` -> ``[.., N/V, V]`` --
    then zips the fields with the field dimension placed BEFORE the lane, producing one array
    ``new_name`` of shape ``[.., N/V, F, V]`` (Array-of-Struct-of-Arrays). Particle ``i``, field ``f``
    lands at ``new_name[.., i//V, f, i%V]``.

    This is the composition ``SplitDimensions`` (Block) + ``ZipArrays`` (Zip). Run after
    ``prepare_for_layout``; a divisible ``N`` (``N % V == 0``) keeps the blocks perfect.
    """
    from dace.transformation.layout.split_dimensions import SplitDimensions

    ndim = len(sdfg.arrays[fields[0]].shape)
    masks = [False] * (ndim - 1) + [True]
    factors = [1] * (ndim - 1) + [vector_width]
    SplitDimensions(split_map={f: (list(masks), list(factors)) for f in fields}).apply_pass(sdfg, {})
    # Blocked fields now have ndim+1 dims ([.., N/V, V]); put the field dim just before the lane.
    ZipArrays(zip_map={new_name: fields}, field_axis=ndim).apply_pass(sdfg, {})
