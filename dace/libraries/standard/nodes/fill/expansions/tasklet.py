# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Single-element same-side scalar assignment."""
from typing import TYPE_CHECKING

import dace
from dace import library, nodes
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.libraries.standard.nodes.fill.common import OUTPUT_CONNECTOR_NAME, python_literal
from dace.sdfg.scope import is_devicelevel_gpu
from dace.transformation.transformation import ExpandTransformation

if TYPE_CHECKING:
    from dace.libraries.standard.nodes.fill.node import FillLibraryNode


@library.expansion
class ExpandTasklet(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: "FillLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        out_name, out, out_subset = node.validate(parent_sdfg, parent_state)
        out_volume = out_subset.num_elements_exact()
        if out_volume != 1:
            raise ValueError(f"Tasklet expansion requires single-element subsets "
                             f"(got output volume {out_volume}). Use 'pure' for multi-element fills.")

        # Host scope can't write device memory directly; route that case to 'CUDA' instead.
        if (not is_devicelevel_gpu(parent_state.sdfg, parent_state, node) and out.storage in GPU_RESIDENT_STORAGES):
            raise ValueError(f"Tasklet expansion cannot fill GPU-resident storage ({out.storage}) for "
                             f"'{out_name}' from host scope; use the 'CUDA' Fill expansion instead.")

        return nodes.Tasklet(node.name,
                             inputs={},
                             outputs={OUTPUT_CONNECTOR_NAME: out.dtype},
                             code=f"{OUTPUT_CONNECTOR_NAME} = {python_literal(node.value, out.dtype)}",
                             language=dace.Language.Python)
