# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

from dace.sdfg import nodes


class MPINode(nodes.LibraryNode):
    """
    Abstract class representing an MPI library node.
    """
    @property
    def has_side_effects(self) -> bool:
        return True
