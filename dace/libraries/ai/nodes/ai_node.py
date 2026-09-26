# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" A library node whose implementation is described in natural language. """

from typing import Any, Dict, Iterable, Union

import dace.library
from dace import properties
from dace.sdfg import nodes

ConnectorSpec = Union[Iterable[str], Dict[str, Any], None]


@dace.library.node
class AINode(nodes.LibraryNode):
    """
    A library node that carries a natural-language description instead of an implementation.

    Every DaCe library node can be expanded with the reserved ``'ai'`` implementation, which asks a
    language model to write the tasklet. This node exists for the case where there is no library
    node to begin with -- a microkernel DaCe does not expose, a vendor intrinsic sequence, a call
    into an external library -- so the description *is* the specification::

        node = AINode('fma', 'Compute out = a * b + c elementwise over the whole tile.',
                      inputs={'_a', '_b', '_c'}, outputs={'_out'})
        state.add_node(node)

    The description is a regular property, so it is saved with the SDFG and shown in the viewer.
    Once expanded, the generated tasklet is part of the SDFG like any other node: it serializes,
    and re-running the SDFG does not query the model again.
    """

    implementations = {}  # 'ai' is a reserved name, resolved by LibraryNode.expand
    default_implementation = 'ai'

    description = properties.Property(dtype=str,
                                      default='',
                                      desc='Natural-language description of what this node must compute. This is '
                                      'the specification handed to the model, so state the intended semantics, '
                                      'the expected numerical behavior, and any implementation technique that is '
                                      'required (e.g. "use AVX2 intrinsics").')

    def __init__(self,
                 name: str,
                 description: str = '',
                 *args,
                 inputs: ConnectorSpec = None,
                 outputs: ConnectorSpec = None,
                 **kwargs) -> None:
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
        :raises ValueError: If the node has no description to generate code from.
        """
        if not self.description.strip():
            raise ValueError(f'AI node "{self.name}" has no description, so there is nothing to generate code from. '
                             'Set its `description` property to what the node must compute.')
