# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Single-element same-side scalar assignment.
"""
from typing import TYPE_CHECKING

import dace
from dace import library, nodes
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.sdfg.scope import is_devicelevel_gpu
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.standard.nodes.copy.common import (_is_cross_cpu_gpu, copy_assignment_code, INPUT_CONNECTOR_NAME,
                                                       OUTPUT_CONNECTOR_NAME)

if TYPE_CHECKING:
    pass


@library.register_expansion(CopyLibraryNode, 'Tasklet')
class ExpandTasklet(ExpandTransformation):
    """Single-element same-side scalar copy: ``_cpy_out = _cpy_in`` as a Python tasklet"""
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        inp_name, inp, in_subset, out_name, out, out_subset = node.validate(parent_sdfg,
                                                                            parent_state,
                                                                            allow_cross_storage=True)
        in_volume = in_subset.num_elements_exact()
        out_volume = out_subset.num_elements_exact()
        if in_volume != 1 or out_volume != 1:
            raise ValueError(f"Tasklet expansion requires single-element subsets "
                             f"(got input volume {in_volume}, output volume {out_volume}). "
                             f"Use MappedTasklet for multi-element copies.")
        # Single-element Shared involvement is a valid thread-level assignment, routed here.
        # In-kernel the boundary does not exist for a single element: the host side arrives as a
        # by-value kernel argument (or is pinned, hence device-addressable), so assignment is right.
        if not is_devicelevel_gpu(parent_sdfg, parent_state, node) and _is_cross_cpu_gpu(
                inp.storage, out.storage, node, parent_state):
            raise ValueError(f"Tasklet expansion: storage types must match (no CPU/GPU boundary); "
                             f"got {inp.storage} -> {out.storage}. Use a MemcpyCUDA1D variant instead.")

        return nodes.Tasklet(node.name,
                             inputs={INPUT_CONNECTOR_NAME: inp.dtype},
                             outputs={OUTPUT_CONNECTOR_NAME: out.dtype},
                             code=copy_assignment_code(inp, out, INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME),
                             language=dace.Language.Python)
