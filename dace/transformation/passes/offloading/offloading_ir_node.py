# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from ordered_set import OrderedSet

from dace.sdfg.state import ConditionalBlock, LoopRegion, ControlFlowBlock

#: Array names longer than this are left out of the IR dump.
PRINT_NAMES = 500


class OffloadingIRNode:
    # INVARIANT: IR-trees are always DAGs
    STATE = -1
    OPEN = 0
    CLOSE = 1
    OPEN_LOOP = 2
    OPEN_COND = 3
    EDGE = 4  # interstate edge

    def __init__(self, type: int, block: ControlFlowBlock | None, cpu_set: OrderedSet[str], gpu_set: OrderedSet[str],
                 next: list['OffloadingIRNode'], close: 'OffloadingIRNode | None'):
        assert block is None or isinstance(block, ControlFlowBlock), f"{block}, {block.__class__.__name__}"
        self.type = type
        self.block: ControlFlowBlock | None = block
        self.cpu_set: OrderedSet[str] = cpu_set
        self.gpu_set: OrderedSet[str] = gpu_set
        self.next: list[OffloadingIRNode] = next
        self.close = close  # corresponding open and close nodes refer to each other
        self.open: OffloadingIRNode | None = None
        self.debug_name = "debug"

        # there should be a reference to the corresponding close node IFF the current node is an open node
        assert (
            self.close
            is not None) == self.is_open_node(), f"node {self.debug_name} of type {self.type} has close {self.close}"

    def __repr__(self) -> str:
        return self.render(OrderedSet(), -4)

    def __str__(self) -> str:
        return self.__repr__()

    def render(self, visited_set: OrderedSet['OffloadingIRNode'], len_before: int) -> str:
        s = f"{self.debug_name}:"
        spaces = 40 - (len_before + len(s))
        cpu = sorted(name for name in self.cpu_set if len(name) <= PRINT_NAMES)
        gpu = sorted(name for name in self.gpu_set if len(name) <= PRINT_NAMES)
        s += spaces * " " + f"cpu = {cpu}, gpu = {gpu}\n"

        if self in visited_set:
            return s
        visited_set.add(self)

        for next in sorted(self.next, key=lambda x: x.debug_name):
            s += f"{self.debug_name} => {next.render(visited_set, len(self.debug_name))}"
        return s

    def is_empty(self) -> bool:
        return not self.cpu_set and not self.gpu_set

    def is_open_node(self) -> bool:
        return self.type in (OffloadingIRNode.OPEN, OffloadingIRNode.OPEN_LOOP, OffloadingIRNode.OPEN_COND)

    def append_node(self, node: 'OffloadingIRNode') -> None:
        self.next.append(node)

    def get_all_tails(self) -> list['OffloadingIRNode']:
        """The nodes of this section that point at its close node, each listed once.

        Iterative, one visit per node: the IR has a node per block, so recursion overflows the stack
        on large programs, and a walk per route doubles with every conditional in a row.
        """
        assert self.is_open_node()
        result: list[OffloadingIRNode] = []
        seen: OrderedSet[OffloadingIRNode] = OrderedSet()
        stack = [self]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            children = []
            for next in node.next:
                if next is self.close:  # a tail contributes itself and none of its other children
                    result.append(node)
                    children.clear()
                    break
                children.append(next)
            stack.extend(reversed(children))
        return result

    def has_one_route(self) -> bool:
        """Whether exactly one route leads from this open node to its close node.

        A conditional inside the section makes two routes even when its arms meet again before one
        tail. Routes are counted per node, saturating at two.
        """
        assert self.is_open_node()
        routes: dict[OffloadingIRNode, int] = {}
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if node in routes:
                continue
            if any(next is self.close for next in node.next):
                routes[node] = 1
            elif expanded:  # the IR is a DAG, so every child is counted before its parent pops again
                routes[node] = min(2, sum(routes[next] for next in node.next))
            else:
                stack.append((node, True))
                stack.extend((next, False) for next in node.next if next not in routes)
        return routes[self] == 1

    @staticmethod
    def new_open_node(block: ControlFlowBlock) -> 'OffloadingIRNode':
        close = OffloadingIRNode(OffloadingIRNode.CLOSE, None, OrderedSet(), OrderedSet(), [], None)
        close.debug_name = f"_close_{block.label}"

        if isinstance(block, LoopRegion):
            type = OffloadingIRNode.OPEN_LOOP
        elif isinstance(block, ConditionalBlock):
            type = OffloadingIRNode.OPEN_COND
        else:
            type = OffloadingIRNode.OPEN

        open = OffloadingIRNode(type, block, OrderedSet(), OrderedSet(), [], close)
        open.debug_name = f"_{OffloadingIRNode.get_type_as_str(type)}_{block.label}"
        close.open = open
        return open

    @staticmethod
    def new_state_node(block: ControlFlowBlock, cpu_set: OrderedSet[str],
                       gpu_set: OrderedSet[str]) -> 'OffloadingIRNode':
        state = OffloadingIRNode(OffloadingIRNode.STATE, block, cpu_set, gpu_set, [], None)
        state.debug_name = f"_state_{block.label}"
        return state

    @staticmethod
    def new_edge_node(block: ControlFlowBlock, cpu_set: OrderedSet[str]) -> 'OffloadingIRNode':
        """The interstate edges reaching ``block``, which read ``cpu_set`` on the host."""
        edge_node = OffloadingIRNode(OffloadingIRNode.EDGE, block, cpu_set, OrderedSet(), [], None)
        edge_node.debug_name = f"_edge_{block.label}"
        return edge_node

    @staticmethod
    def get_type_as_str(type: int) -> str:
        names = {
            OffloadingIRNode.STATE: "state",
            OffloadingIRNode.OPEN: "open",
            OffloadingIRNode.CLOSE: "close",
            OffloadingIRNode.OPEN_LOOP: "loop",
            OffloadingIRNode.OPEN_COND: "cond",
            OffloadingIRNode.EDGE: "edge",
        }
        if type not in names:
            raise ValueError(f"Invalid IR type to convert to string: {type}")
        return names[type]
