
from dace.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, LoopRegion, ControlFlowBlock

class OffloadingIRNode:
    # INVARIANT: IR-trees are always DAGs
    STATE = -1
    OPEN = 0
    CLOSE = 1
    OPEN_LOOP = 2
    OPEN_COND = 3
    EDGE = 4 # interstate edge

    def __init__(self, type:int, block:ControlFlowBlock, cpu_set:set, gpu_set:set, next:list, close):
        assert block is None or isinstance(block, ControlFlowBlock), f"{block}, {block.__class__.__name__}"
        self.type = type
        self.block : ControlFlowBlock = block
        self.cpu_set : set[str] = cpu_set
        self.gpu_set : set[str] = gpu_set
        self.next : list[OffloadingIRNode] = next

        self.close = close # corresponding open and close nodes refer to each other
        self.open = None
        self.debug_name = "debug"
        
        # there should be a reference to the corresponding close node IFF the current node is an open node
        assert (self.close is not None) == self.is_open_node(), f"node {self.debug_name} of type {self.type} has close {self.close}"
        

    def __repr__(self):
        return self._get_str(set(), -4)
    def __str__(self): 
        return self.__repr__()
    def _get_str(self, visited_set, len_before):
        s = f"{self.debug_name}:"
        spaces = 40 - (len_before + len(s))
        s += spaces * " " + f"cpu = {sorted([name for name in self.cpu_set])}, gpu = {sorted([name for name in self.gpu_set])}\n"

        if self in visited_set:
            return s
        visited_set.add(self)

        next_list = sorted(self.next, key=lambda x:x.debug_name)
        for next in next_list:
            s += f"{self.debug_name} => {next._get_str(visited_set, len(self.debug_name))}"
        return s
    
    # utility functions
    def is_empty(self):
        return not self.cpu_set and not self.gpu_set
    
    def is_open_node(self):
        return self.type in [OffloadingIRNode.OPEN, OffloadingIRNode.OPEN_LOOP, OffloadingIRNode.OPEN_COND]

    def is_close_node(self):
        return self.type in [OffloadingIRNode.CLOSE]

    def append_node(self, node):
        self.next.append(node)

    def get_all_tails(self):
        assert self.is_open_node()

        def recursion(node, result:list):
            for next in node.next:
                if next == self.close: # definition of a tail: a node that points at this section's end (close-node)
                    result.append(node)
                    return
                recursion(next, result)

        result = []
        recursion(self, result)
        return result
    
    # static makers
    def new_open_node(block:ControlFlowBlock):
        close = OffloadingIRNode(OffloadingIRNode.CLOSE, None, set(), set(), [], None)
        close.debug_name = f"_close_{block.label}"

        type : int
        if isinstance(block, LoopRegion):
            type = OffloadingIRNode.OPEN_LOOP
        elif isinstance(block, ConditionalBlock):
            type = OffloadingIRNode.OPEN_COND
        else:
            type = OffloadingIRNode.OPEN

        open = OffloadingIRNode(type, block, set(), set(), [], close)
        open.debug_name = f"_{OffloadingIRNode.get_type_as_str(type)}_{block.label}"
        close.open = open

        return open
    
    def new_state_node(block:ControlFlowBlock, cpu_set:set, gpu_set:set):
        state = OffloadingIRNode(OffloadingIRNode.STATE, block, cpu_set, gpu_set, [], None)
        state.debug_name = f"_state_{block.label}"
        return state
    
    def new_edge_node(edge:InterstateEdge, cpu_set:set):
        edge_node = OffloadingIRNode(OffloadingIRNode.EDGE, edge, cpu_set, set(), [], None)
        edge_node.debug_name = f"_edge_{edge.label}"
        return edge_node
    
    def get_type_as_str(type:int):
        match type:
            case OffloadingIRNode.STATE: return "state"
            case OffloadingIRNode.OPEN: return "open"
            case OffloadingIRNode.OPEN: return "close"
            case OffloadingIRNode.OPEN_LOOP: return "loop"
            case OffloadingIRNode.OPEN_COND: return "cond"
        raise ValueError(f"Invalid IR type to convert to string: {type}")
