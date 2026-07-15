# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""PadDimensions -- the Pad layout primitive: GROW a dimension's extent, keep packed strides.

Pad increases the physical extent of chosen dimensions by adding (unused) trailing elements,
keeping the array in the packed normal form (C- or Fortran-contiguous). The logical shape grows;
every existing memlet subset indexes the live region (``< old_extent < new_extent``) and stays
valid -- the pad cells are simply never accessed. So only the DESCRIPTOR changes (recursed into
nested SDFGs); no memlet / tasklet / interstate rewrite is needed. The caller must allocate the
padded size.

Main uses: make a subsequent Block factor divide exactly (``N -> ceil(N/f)*f``); break L1
conflict-set periodicity; GPU shared-memory bank padding.
"""
from dataclasses import dataclass
from typing import Any, Dict, List

import dace
from dace.transformation import pass_pipeline as ppl


@dataclass
class PadDimensions(ppl.Pass):
    """Grow chosen array dimensions by trailing pad, preserving the packed layout.

    :param pad_map: ``{array_name: [p_0, ..., p_{d-1}]}`` -- per-dimension pad amount (0 = none),
                    one entry per array dimension.
    """

    def __init__(self, pad_map: Dict[str, List[int]]):
        self._pad_map = pad_map

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _packed_strides(self, shape: List, fortran: bool) -> List:
        n = len(shape)
        strides = [1] * n
        if fortran:
            for i in range(1, n):
                strides[i] = strides[i - 1] * shape[i - 1]
        else:
            for i in range(n - 2, -1, -1):
                strides[i] = strides[i + 1] * shape[i + 1]
        return strides

    def _grow(self, sdfg: dace.SDFG, arr_name: str, pads: List[int]):
        desc = sdfg.arrays[arr_name]
        if len(pads) != len(desc.shape):
            raise ValueError(f"PadDimensions: pad {pads} length != rank {len(desc.shape)} of '{arr_name}'")
        # Preserve the array's packed base (Fortran only if it is packed-Fortran but not packed-C;
        # 1-D arrays are both, and default to C).
        fortran = desc.is_packed_fortran_strides() and not desc.is_packed_c_strides()
        new_shape = [s + p for s, p in zip(desc.shape, pads)]
        strides = self._packed_strides(new_shape, fortran)
        total = 1
        for s in new_shape:
            total = total * s
        desc.set_shape(new_shape, strides=strides, total_size=total)

    def _grow_recursive(self, sdfg: dace.SDFG, arr_name: str, pads: List[int]):
        self._grow(sdfg, arr_name, pads)
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    in_map = {ie.data.data: ie.dst_conn for ie in state.in_edges(node)}
                    if arr_name in in_map:
                        self._grow_recursive(node.sdfg, in_map[arr_name], pads)
                    out_map = {oe.data.data: oe.src_conn for oe in state.out_edges(node)}
                    if arr_name in out_map:
                        self._grow_recursive(node.sdfg, out_map[arr_name], pads)

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        for arr_name, pads in self._pad_map.items():
            self._grow_recursive(sdfg, arr_name, pads)
        return 0
