# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pad layout primitive: grow a dimension's extent with trailing unused elements, keeping packed strides. Only the descriptor changes (recursed into nested SDFGs); existing memlets still index the live region."""
from dataclasses import dataclass
from typing import Any, Dict, List

import dace
from dace.transformation import pass_pipeline as ppl


@dataclass
class PadDimensions(ppl.Pass):
    """Grow chosen array dimensions by trailing pad, preserving the packed layout."""

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
        # Packed-Fortran only if not also packed-C; 1-D arrays default to C.
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
                if not isinstance(node, dace.nodes.NestedSDFG):
                    continue
                # Read-write array shares the same inner name on in+out edge; grow it once.
                inner_names = set()
                for ie in state.in_edges(node):
                    if ie.data is not None and ie.data.data == arr_name:
                        inner_names.add(ie.dst_conn)
                for oe in state.out_edges(node):
                    if oe.data is not None and oe.data.data == arr_name:
                        inner_names.add(oe.src_conn)
                for inner in inner_names:
                    self._grow_recursive(node.sdfg, inner, pads)

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        for arr_name, pads in self._pad_map.items():
            self._grow_recursive(sdfg, arr_name, pads)
        return 0
