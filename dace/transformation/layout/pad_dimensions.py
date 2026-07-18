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


@dataclass
class PadZeroFill(ppl.Pass):
    """Companion to :class:`PadDimensions`: zero the pad cells once at program entry so a sum-of-products
    contraction (TensorDot/Gemm) over a padded contracted dimension stays legal (the extra products are
    ``0``) and identically-padded copies carry zeros. A non-sum reduction (max/min/product) over a padded
    axis is refused -- ``0`` is not its identity, so zero-fill would silently miscompile.

    Takes the same ``pad_map`` as :class:`PadDimensions` (``{array: [p_0, .., p_{d-1}]}``); run it right after.
    """

    def __init__(self, pad_map: Dict[str, List[int]]):
        self._pad_map = pad_map

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.AccessNodes | ppl.Modifies.Edges | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        self._refuse_nonsum_reductions(sdfg)
        init = None
        for arr, pads in self._pad_map.items():
            desc = sdfg.arrays[arr]
            if len(pads) != len(desc.shape):
                raise ValueError(f"PadZeroFill: pad {pads} length != rank {len(desc.shape)} of '{arr}'")
            new_shape = list(desc.shape)
            for dim, p in enumerate(pads):
                if p == 0:
                    continue
                if init is None:
                    init = sdfg.add_state_before(sdfg.start_state, "pad_zero_fill")
                self._zero_slab(init, arr, new_shape, dim, new_shape[dim] - p)
        return 0

    def _zero_slab(self, state, arr: str, new_shape: List, dim: int, old_extent) -> None:
        """Zero ``A[.., old_extent:new, ..]`` -- the pad slab of one dimension (all other dims full). The
        union of the per-dim slabs is exactly the complement of the live box, so overlaps re-zero harmlessly."""
        params = [f"__p{k}" for k in range(len(new_shape))]
        map_ranges = {params[k]: (f"{old_extent}:{s}" if k == dim else f"0:{s}") for k, s in enumerate(new_shape)}
        write_ranges = [(dace.symbolic.symbol(p), dace.symbolic.symbol(p), 1) for p in params]
        state.add_mapped_tasklet(
            name=f"zero_pad_{arr}_d{dim}",
            map_ranges=map_ranges,
            inputs={},
            code="__out = 0",
            outputs={"__out": dace.Memlet(data=arr, subset=dace.subsets.Range(write_ranges))},
            external_edges=True,
        )

    def _refuse_nonsum_reductions(self, sdfg: dace.SDFG) -> None:
        from dace.libraries.standard.nodes.reduce import Reduce
        from dace.frontend.operations import detect_reduction_type
        from dace.dtypes import ReductionType

        padded = {arr: {d for d, p in enumerate(pads) if p > 0} for arr, pads in self._pad_map.items()}
        for state in sdfg.all_states():
            for node in state.nodes():
                if not isinstance(node, Reduce):
                    continue
                for e in state.in_edges(node):
                    arr = e.data.data
                    if arr not in padded or not padded[arr]:
                        continue
                    ndim = len(sdfg.arrays[arr].shape)
                    reduced = set(range(ndim)) if node.axes is None else set(node.axes)
                    if padded[arr] & reduced and detect_reduction_type(node.wcr) != ReductionType.Sum:
                        raise NotImplementedError(
                            f"PadZeroFill: '{arr}' is padded on a dimension reduced by a non-sum reduction "
                            f"({detect_reduction_type(node.wcr).name}); zero-fill is legal only for a sum "
                            f"reduction. Pad a free dimension instead, or do not pad this array.")
