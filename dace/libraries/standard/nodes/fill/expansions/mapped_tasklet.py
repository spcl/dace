# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Element-wise mapped fill: the device-neutral parallel form, and the fallback for everything a
single call cannot express (non-contiguous subsets, non-byte-splat GPU values, device scope)."""
from typing import TYPE_CHECKING

import dace
from dace import library
from dace.libraries.standard.nodes.fill.common import make_fill_skeleton, python_literal
from dace.transformation.transformation import ExpandTransformation

if TYPE_CHECKING:
    from dace.libraries.standard.nodes.fill.node import FillLibraryNode


@library.expansion
class ExpandPure(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: "FillLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
        sdfg, state, out_name, out, map_lengths = make_fill_skeleton(node, parent_state)

        # Must not collide with the wrapper SDFG's parameter array (named after outer connector).
        inner_out = "_out"
        map_params = [f"__i{i}" for i in range(len(map_lengths))]
        map_rng = {i: f"0:{s}" for i, s in zip(map_params, map_lengths)}
        outputs = {inner_out: dace.memlet.Memlet(f"{out_name}[{','.join(map_params)}]")}
        schedule = (dace.dtypes.ScheduleType.GPU_Device
                    if out.storage == dace.dtypes.StorageType.GPU_Global else dace.dtypes.ScheduleType.Default)
        state.add_mapped_tasklet(f"{node.label}_tasklet",
                                 map_rng,
                                 dict(),
                                 f"{inner_out} = {python_literal(node.value, out.dtype)}",
                                 outputs,
                                 schedule=schedule,
                                 external_edges=True)

        return sdfg
