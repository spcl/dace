"""
***ETH Zurich SPCL - Bachelors Thesis: BSc thesis: Deep learning transformations with semantic-preservation in DaCe by Severin Huber***

This module contains several classes that implement various transformations and optimizations for SDFGs (Stateful DataFlow Graphs) using the DaCe (Data-Centric Parallel Programming) framework.

This is part of the bachelor thesis of Severin Huber at SPCL. The topic is semantic-preserving transformations.

This optimization transformations are implemented not for general sdfgs, but for those generated 
from Andrei Ivanovs microkernel IR with a converter implemented as a part of this thesis. 
General usage of these transformations is not guaranteed.

This class implements the SwapOperations transformation.

"""

import dace

class SwapOperations():
    """ 
    This class implements the SwapOperations transformation. It searches for two tasklets in the SDFG and swaps their execution order. The transformation is only applied if the tasklets are swappable.
    """
    
    def __init__(self):
        self.__name = 'SwapOperations'
        self.checked = False
    
    @property
    def name(self):
        return self.__name
    
    def find(self, sdfg):
        """
        This method searches for two tasklets in the SDFG and returns a list of tuples containing the two tasklets.
        
        Two tasklets are swappable if the following conditions are met:
            - The two tasklets are in the same state
            - The two tasklets lie in the same scope
            - The output of the first tasklet is not used in the input of the second tasklet
            - There is no path from the first tasklet to the second tasklet
        
        Args:
            sdfg (dace.SDFG): The SDFG to search for swappable tasklets.
        
        Returns:
            list: A list of tuples, where each tuple contains two swappable tasklets.
        """
        result = []
        tasklets = self.find_tasklets(sdfg)
        for tasklet1, state1 in tasklets:
            for tasklet2, state2 in tasklets:
                if tasklet1 == tasklet2:
                    continue
                if not state1 == state2:
                    continue
                if not state1.entry_node(tasklet1) == state2.entry_node(tasklet2):
                    continue
                """
                output_1 = {edge.data.data for edge in state1.out_edges(tasklet1)}
                input_2 = {edge.data.data for edge in state2.in_edges(tasklet2)}
                if input_2 & output_1:
                    continue
                """
                if not self.check(state1, tasklet1, tasklet2) or not self.check(state2, tasklet2, tasklet1):
                    continue
                result.append((tasklet1, tasklet2))
        self.checked = True
        return result
    
    def check(self, state, tasklet1, tasklet2):
        """
        This method checks if there is a path from first tasklet to second tasklet.
        Returns True if there is no path, False otherwise.
        """
        def traverse(node, t2, visited=None):
            if visited is None:
                visited = set()
            visited.add(node)
            #if isinstance(node, dace.nodes.MapExit):
            #   return False
            for edge in state.out_edges(node):
                if edge.dst == t2:
                    return True
                if not edge.dst in visited:
                    if traverse(edge.dst, t2, visited):
                        return True
            return False
        
        return not traverse(tasklet1, tasklet2)
    
    def find_tasklets(self, sdfg):
        result = []
        for state in sdfg.nodes():
                for node in state.nodes():
                    if isinstance(node, dace.nodes.Tasklet):
                        result.append((node, state))
        return result
    
    def apply(self, sdfg, op1, op2):
        """
        This method applies the SwapOperations transformation to the given SDFG. It swaps the order of execution of the two given tasklets by adding an empty edge from the output of the second tasklet to the first tasklet.
        op1 -> op2
        """
        if not self.checked and not (op1, op2) in self.find(sdfg):
            return
        self.checked = False
        state = find_state(sdfg, op1)
        out_node = [edge.dst for edge in state.out_edges(op1)][0]
        state.add_nedge(out_node, op2, dace.memlet.Memlet())

def find_state(sdfg, node):
    for sdfg_state in sdfg.nodes():
        if node in sdfg_state.nodes():
            return sdfg_state
