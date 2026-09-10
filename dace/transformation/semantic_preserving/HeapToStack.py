"""
***ETH Zurich SPCL - Bachelors Thesis: BSc thesis: Deep learning transformations with semantic-preservation in DaCe by Severin Huber***

This module contains several classes that implement various transformations and optimizations for SDFGs (Stateful DataFlow Graphs) using the DaCe (Data-Centric Parallel Programming) framework.

This is part of the bachelor thesis of Severin Huber at SPCL. The topic is semantic-preserving transformations.

This optimization transformations are implemented not for general sdfgs, but for those generated 
from Andrei Ivanovs microkernel IR with a converter implemented as a part of this thesis. 
General usage of these transformations is not guaranteed.

This class implements the HeapToStack transformation.

"""

import dace

class HeapToStack():
    """
    This class implements the HeapToStack transformation. It changes the storage type of the array to the stack.
    
    Inputs:
        - hard_limit (int): The hard limit of the stack size. If the stack size exceeds this limit, the transformation is not applied. Default is None (no hard limit).
        - stack_size_factor (int): The factor to calculate the stack size. Default is 1. Which means the stack size is 1 * 1MB (minus a small amount). This is just a heuristic.


    Attributes:
        check_limit (bool): If True, the transformation checks if the stack size is exceeded.
        stack_size (int): The size of the stack. Default is 1MB. This can be changed by the user. This is just a heuristic.
        hard_limit (int): The hard limit of the stack size. If the stack size exceeds this limit, the transformation is not applied. Default is None (no hard limit).
    """

    def __init__(self, check_limit = True, hard_limit=None, stack_size_factor=1.0):
        self.__name = 'HeapToStack'
        self.checked = False
        # check if stack size is exceeded
        self.check_limit = check_limit
        # Default assumption: stack size is 1MB = 1 * 1'048'576 bytes
        self.stack_size = stack_size_factor * 1048576 * 0.95 #1MB * 0.95 
        self.hard_limit = hard_limit #65536 #(64KB)

    @property
    def name(self):
        return self.__name

    def get_stack_size(self, sdfg, limit='soft'):
        import os
        if os.name == 'nt': #windows
            self.stack_size = None
        else: #unix like systems
            import resource
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_STACK)
            if limit == 'soft':
                self.stack_size = soft_limit
            else:
                self.stack_size = hard_limit

    def current_stack_size(self, sdfg):
        current_size = 0
        all_nodes = set()
        # we want only the arrays that are represented in the SDFG
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.AccessNode):
                    all_nodes.add(node.label)
    
        for name, data in sdfg.arrays.items():
            if not name in all_nodes:
                continue
            if data.storage == dace.StorageType.Register:
                current_size += data.total_size * data.dtype.bytes
        return current_size

    def check_current_stack_usage(self):
        """Check the current stack size usage in bytes."""
        import sys
        frame = sys._getframe()
        stack_usage = 0
        while frame is not None:
            # Each frame has its own local variables and other information
            stack_usage += sum(sys.getsizeof(v) for v in frame.f_locals.values())
            frame = frame.f_back  # Move to the previous frame
        return stack_usage
    
    def stack_size_exceeded(self, sdfg, data, safety_factor=1.0):
        size = data.total_size * data.dtype.bytes
        available_stack = self.stack_size - self.current_stack_size(sdfg)
        if safety_factor*size > available_stack:
            return True
        return False

    def find(self, sdfg):
        """
        This method searches for arrays in the SDFG that are not yet stored on the stack (Register).
        
        This transformation is only applied if the stack size is not exceeded.
        
        We try not to exceed the stack size.
        Args:
            sdfg (dace.SDFG): The SDFG to search for arrays.
        
        Returns:
            list: A list of name of arrays that are not stored on the stack ordered by array size * type size (ascending order).
        """
        result = []
        for name, data in sdfg.arrays.items():
            if data.storage == dace.StorageType.Register:
                continue
            if self.hard_limit:
                if self.hard_limit < data.total_size * data.dtype.bytes:
                    continue
            if self.check_limit and self.stack_size_exceeded(sdfg, data):
                continue
            result.append(name)
        self.checked = True
        return sorted(result, key=lambda x: (sdfg.arrays[x].total_size*sdfg.arrays[x].dtype.bytes, sdfg.arrays[x].total_size))
    
    def apply(self, sdfg, name):
        """
        This method applies the HeapToStack transformation to the given SDFG. It changes the storage type of the array to the stack.

        Args:
            sdfg (dace.SDFG): The SDFG to apply the transformation to.
            name (String): The array to apply the transformation to.
        """
        if not self.checked and not name in self.find(sdfg):
            return
        self.checked = False
        data = sdfg.arrays[name]
        data.storage = dace.StorageType.Register