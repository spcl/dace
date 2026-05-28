"""
***ETH Zurich SPCL - Bachelors Thesis: BSc thesis: Deep learning transformations with semantic-preservation in DaCe by Severin Huber***

This module contains several classes that implement various transformations and optimizations for SDFGs (Stateful DataFlow Graphs) using the DaCe (Data-Centric Parallel Programming) framework.

This is part of the bachelor thesis of Severin Huber at SPCL. The topic is semantic-preserving transformations.

This optimization transformations are implemented not for general sdfgs, but for those generated 
from Andrei Ivanovs microkernel IR with a converter implemented as a part of this thesis. 
General usage of these transformations is not guaranteed.

This class implements the changeMemoryLayout transformation.

"""

import dace

class changeMemoryLayout():
    """
    This class implements the changeMemoryLayout transformation. It changes the memory layout of the array.
    """

    def __init__(self):
        self.__name = 'changeMemoryLayout'
        self.checked = False

    @property
    def name(self):
        return self.__name

    def find(self, sdfg):
        """
        This method searches for arrays in the SDFG on which the changeMemeoryLayout Transformation can be applied.

        Appicable to all data arrays which are not used as input or output to the kernel.

        Args:
            sdfg (dace.SDFG): The SDFG to search for arrays.
        
        Returns:
            list: A list of data arrays on which the transformation can be applied.
        """
        changeable_arrays = set()
        for state in sdfg.nodes():
            source, sink = state.source_nodes(), state.sink_nodes()
            for node in state.nodes():
                if isinstance(node, dace.nodes.AccessNode):
                    if not node in source and not node in sink:
                        changeable_arrays.add(node)
        self.checked = True
        return (list(changeable_arrays))
    
    def apply(self, sdfg, node, dim_order):
        """
        This method applies the changeMemoryLayout transformation to the given SDFG. It changes the memory layout of the array by chaning the strides of the array.

        Args:
            sdfg (dace.SDFG): The SDFG to apply the transformation to.
            node (dace.nodes.AccessNode): The array to apply the transformation to.
            dim_order (list): The new order of the dimensions. E.g. [2, 0, 1] for a 3D array.
        """
        if not self.checked and not node in self.find(sdfg):
            return
        self.checked = False

        data = sdfg.arrays[node.data]
        shape = data.shape

        # check if order of dimension is correct
        assert len(shape) == len(dim_order)
        assert max(dim_order) == len(shape) - 1
        assert min(dim_order) == 0

        m = dict(zip(shape, dim_order))
        new_strides = list()
        for dim in shape:
            stride = 1
            for dim2 in shape:
                if m[dim2] > m[dim]:
                    stride *= dim2
            new_strides.append(stride)

        # set new stride
        data.strides = tuple(new_strides)
        return