# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``FillLibraryNode``: write one constant over a contiguous output subset, or broadcast a
single-element value supplied through an input connector."""
from typing import Any, Optional, Tuple

import dace
from dace import library, nodes, properties
from dace.libraries.standard.helper import CURRENT_STREAM_NAME


@library.node
class FillLibraryNode(nodes.LibraryNode):
    """Library node writing a constant over a contiguous output subset.

    A static ``value`` property is used when the optional ``_fill_val`` input connector is
    unwired. When wired, the input must be a single-element subset from an access node and its
    dtype must match the output array's dtype.
    """

    implementations = {}
    default_implementation = 'Auto'

    OUTPUT_CONNECTOR_NAME = "_fill_out"
    VALUE_CONNECTOR_NAME = "_fill_val"

    # dtype=None takes any Python constant; numpy scalars are normalized by Property.__set__.
    value = properties.Property(dtype=None, default=0, desc='The constant written over the subset.')

    def __init__(self, name: str, *args, value=0, **kwargs):
        # Dotted structure-member data names reach here through the callers that build the label;
        # the label names the wrapper SDFG, i.e. a C++ function. See CopyLibraryNode.__init__.
        super().__init__(name.replace('.', '_'), *args,
                         inputs={self.VALUE_CONNECTOR_NAME}, outputs={self.OUTPUT_CONNECTOR_NAME}, **kwargs)
        self.value = value

    def value_edge(self, state: dace.SDFGState) -> Optional[Any]:
        """Return the edge carrying the dynamic fill value, or ``None`` if unwired."""
        in_edges = [
            e for e in state.in_edges(self)
            if e.dst_conn == self.VALUE_CONNECTOR_NAME and not e.data.is_empty()
        ]
        return in_edges[0] if len(in_edges) == 1 else None

    def value_descriptor(
            self, state: dace.SDFGState) -> Optional[Tuple[str, dace.data.Data, dace.subsets.Range]]:
        """Return ``(data_name, data, subset)`` for the dynamic fill value, or ``None``.

        :raises ValueError: If the value edge is not a single-element access-node subset, or if
            its dtype does not match the output dtype.
        """
        out_edge = next((e for e in state.out_edges(self) if e.src_conn == self.OUTPUT_CONNECTOR_NAME), None)
        if out_edge is None:
            return None
        out = state.sdfg.arrays[out_edge.data.data]

        edge = self.value_edge(state)
        if edge is None:
            return None

        src = state.memlet_path(edge)[0].src
        if not isinstance(src, nodes.AccessNode):
            raise ValueError(f"{type(self).__name__} dynamic fill value must come from an access node, "
                             f"got {type(src).__name__}.")

        name = src.data
        desc = state.sdfg.arrays[name]
        subset = edge.data.subset
        if subset.num_elements_exact() != 1:
            raise ValueError(f"{type(self).__name__} dynamic fill value must be a single element, "
                             f"got subset {subset} on '{name}'.")
        if desc.dtype != out.dtype:
            raise ValueError(f"{type(self).__name__} dynamic fill value dtype ({desc.dtype}) must match "
                             f"output dtype ({out.dtype}).")

        return name, desc, subset

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> Tuple[str, dace.data.Data, dace.subsets.Range]:
        """Validate wiring and resolve the output edge.

        :param sdfg: The SDFG owning the data descriptors.
        :param state: The state containing this node.
        :returns: ``(out_name, out, out_subset)``.
        :raises ValueError: If the node lacks exactly one output edge, has a non-empty
            non-reserved input connector wired, or has an invalid dynamic value input.
        """
        data_oes = [oe for oe in state.out_edges(self) if oe.src_conn == self.OUTPUT_CONNECTOR_NAME]
        if len(data_oes) != 1:
            raise ValueError(f"{type(self).__name__} expects exactly one "
                             f"``{self.OUTPUT_CONNECTOR_NAME}`` output edge.")

        reserved = {CURRENT_STREAM_NAME, self.VALUE_CONNECTOR_NAME}
        extra = [ie.dst_conn for ie in state.in_edges(self) if ie.dst_conn not in reserved and not ie.data.is_empty()]
        if extra:
            raise ValueError(f"{type(self).__name__} does not accept arbitrary dynamic input connectors; got {extra}. "
                             f"Subset expressions must use symbols already in scope.")

        # Also raises if the dynamic value connector is wired but invalid.
        self.value_descriptor(state)

        # If the optional value connector is not actually wired, drop it so SDFG validation does not
        # treat it as a dangling connector. It is recreated on deserialization via __init__.
        if self.value_edge(state) is None and self.VALUE_CONNECTOR_NAME in self.in_connectors:
            self.remove_in_connector(self.VALUE_CONNECTOR_NAME)

        oe = data_oes[0]
        out = sdfg.arrays[oe.data.data]
        out_subset = oe.data.subset
        out_name = oe.src_conn

        return out_name, out, out_subset
