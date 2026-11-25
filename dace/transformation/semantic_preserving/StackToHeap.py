"""
***ETH Zurich SPCL - Bachelors Thesis: BSc thesis: Deep learning transformations with semantic-preservation in DaCe by Severin Huber***

This module contains several classes that implement various transformations and optimizations for SDFGs (Stateful DataFlow Graphs) using the DaCe (Data-Centric Parallel Programming) framework.

This is part of the bachelor thesis of Severin Huber at SPCL. The topic is semantic-preserving transformations.

This optimization transformations are implemented not for general sdfgs, but for those generated 
from Andrei Ivanovs microkernel IR with a converter implemented as a part of this thesis. 
General usage of these transformations is not guaranteed.

This class implements the StackToHeap transformation.

"""

import dace

class StackToHeap():
    """
    This class implements the StackToHeap transformation. It changes the storage type of the array to the heap.
    """
    def __init__(self):
        self.__name = 'StackToHeap'
        self.checked = False

    @property
    def name(self):
        return self.__name

    def find(self, sdfg):
        """
        This method searches for arrays in the SDFG that are not yet stored on the heap (CPU_HEAP).

        Args:
            sdfg (dace.SDFG): The SDFG to search for arrays.

        Returns:
            list: A list of data arrays that are not stored on the heap.
        """
        result = []
        for name, data in sdfg.arrays.items():
            if not data.storage == dace.StorageType.CPU_Heap:
                result.append(name)
        self.checked = True
        return result
    
    def apply(self, sdfg, name):
        """
        This method applies the StackToHeap transformation to the given SDFG. It changes the storage type of the array to the heap.
        
        Args:
            sdfg (dace.SDFG): The SDFG to apply the transformation to.
            name (String): The array to apply the transformation to.
        """
        if not self.checked and not name in self.find(sdfg):
            return 
        self.checked = False
        data = sdfg.arrays[name]
        data.storage = dace.StorageType.CPU_Heap