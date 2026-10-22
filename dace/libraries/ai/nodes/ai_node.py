# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A library node whose implementation is described in natural language."""

from collections.abc import Iterable
from typing import Any

import dace.library
from dace import properties
from dace.sdfg import nodes

ConnectorSpec = Iterable[str] | dict[str, Any] | None


@dace.library.node
class AINode(nodes.LibraryNode):
    """
    A library node specified by a natural-language description, implemented by the reserved
    ``'ai'`` implementation::

        node = AINode('fma', 'Compute _out = _a * _b + _c elementwise.',
                      inputs={'_a', '_b', '_c'}, outputs={'_out'})
    """

    implementations = {}  # 'ai' is a reserved name, resolved by LibraryNode.expand
    default_implementation = "ai"

    description = properties.Property(
        dtype=str,
        default="",
        desc="What the node computes, in natural language. This is the specification given to the "
        'model, including any required technique (e.g. "use AVX2 intrinsics").',
    )

    def __init__(
        self,
        name: str,
        description: str = "",
        *args,
        inputs: ConnectorSpec = None,
        outputs: ConnectorSpec = None,
        **kwargs,
    ) -> None:
        """
        Creates an AI-implemented library node.

        :param name: Name of the node.
        :param description: What the node must compute, in natural language.
        :param inputs: Input connector names, or a mapping from name to type.
        :param outputs: Output connector names, or a mapping from name to type.
        """
        super().__init__(name, *args, inputs=inputs or set(), outputs=outputs or set(), **kwargs)
        self.description = description

    def validate(self, sdfg, state) -> None:
        """
        Checks that the node can be expanded.

        :param sdfg: The SDFG containing the node.
        :param state: The state containing the node.
        :raises ValueError: If the node has no description.
        """
        if not self.description.strip():
            raise ValueError(f'AI node "{self.name}" has no description of what it must compute.')
