# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``FillLibraryNode``: write one constant over a contiguous output subset."""
from typing import Tuple

import dace
from dace import library, nodes, properties
from dace.libraries.standard.helper import CURRENT_STREAM_NAME
from dace.libraries.standard.nodes.fill.common import OUTPUT_CONNECTOR_NAME
from dace.libraries.standard.nodes.fill.expansions import (ExpandAuto, ExpandCPU, ExpandCUDA, ExpandPure, ExpandTasklet)


@library.node
class FillLibraryNode(nodes.LibraryNode):
    """Library node writing a constant over a contiguous output subset.

    Does NOT accept dynamic (Scalar) input connectors: subset expressions must use symbols
    already in scope, so the auto selector reasons purely from the static memlet subset.
    """

    implementations = {
        "Auto": ExpandAuto,
        "pure": ExpandPure,
        "CUDA": ExpandCUDA,
        "CPU": ExpandCPU,
        "tasklet": ExpandTasklet
    }
    default_implementation = 'Auto'

    OUTPUT_CONNECTOR_NAME = OUTPUT_CONNECTOR_NAME

    # dtype=None takes any Python constant; numpy scalars are normalized by Property.__set__.
    value = properties.Property(dtype=None, default=0, desc='The constant written over the subset.')

    def __init__(self, name: str, *args, value=0, **kwargs):
        # Dotted structure-member data names reach here through the callers that build the label;
        # the label names the wrapper SDFG, i.e. a C++ function. See CopyLibraryNode.__init__.
        super().__init__(name.replace('.', '_'), *args, outputs={OUTPUT_CONNECTOR_NAME}, **kwargs)
        self.value = value

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> Tuple[str, dace.data.Data, dace.subsets.Range]:
        """Validate wiring and resolve the output edge.

        :param sdfg: The SDFG owning the data descriptors.
        :param state: The state containing this node.
        :returns: ``(out_name, out, out_subset)``.
        :raises ValueError: If the node lacks exactly one output edge, or has a non-empty
            non-reserved input connector wired.
        """
        data_oes = [oe for oe in state.out_edges(self) if oe.src_conn == OUTPUT_CONNECTOR_NAME]
        if len(data_oes) != 1:
            raise ValueError(f"{type(self).__name__} expects exactly one "
                             f"``{OUTPUT_CONNECTOR_NAME}`` output edge.")

        reserved = {CURRENT_STREAM_NAME}
        extra = [ie.dst_conn for ie in state.in_edges(self) if ie.dst_conn not in reserved and not ie.data.is_empty()]
        if extra:
            raise ValueError(f"{type(self).__name__} does not accept dynamic input connectors; got {extra}. "
                             f"Subset expressions must use symbols already in scope.")

        oe = data_oes[0]
        out = sdfg.arrays[oe.data.data]
        out_subset = oe.data.subset
        out_name = oe.src_conn

        return out_name, out, out_subset
