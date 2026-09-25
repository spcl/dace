# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from dace.ordered import OrderedSet

from dace.sdfg import InterstateEdge
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
        self.close = close

        self.open: OffloadingIRNode | None = None
        self.debug_name = "debug"

        # there should be a reference to the corresponding close node IFF the current node is an open node
        assert (
            self.close
            is not None) == self.is_open_node(), f"node {self.debug_name} of type {self.type} has close {self.close}"

    def __repr__(self) -> str:
        return self._get_str(OrderedSet(), -4)

    def __str__(self) -> str:
        return self.__repr__()

    def _get_str(self, visited_set: OrderedSet['OffloadingIRNode'], len_before: int) -> str:
        s = f"{self.debug_name}:"
        spaces = 40 - (len_before + len(s))
        cpu = sorted(name for name in self.cpu_set if len(name) <= PRINT_NAMES)
        gpu = sorted(name for name in self.gpu_set if len(name) <= PRINT_NAMES)
        s += spaces * " " + f"cpu = {cpu}, gpu = {gpu}\n"

        if self in visited_set:
            return s
        visited_set.add(self)

        next_list = sorted(self.next, key=lambda x: x.debug_name)
        for next in next_list:
            s += f"{self.debug_name} => {next._get_str(visited_set, len(self.debug_name))}"
        return s

    # utility functions
    def is_empty(self) -> bool:
        return not self.cpu_set and not self.gpu_set

    def is_open_node(self) -> bool:
        return self.type in [OffloadingIRNode.OPEN, OffloadingIRNode.OPEN_LOOP, OffloadingIRNode.OPEN_COND]

    def is_close_node(self) -> bool:
        return self.type in [OffloadingIRNode.CLOSE]

    def append_node(self, node: 'OffloadingIRNode') -> None:
        self.next.append(node)

    def get_all_tails(self) -> list['OffloadingIRNode']:
        assert self.is_open_node()

        # ITERATIVE, and it has to be: the IR is one node per state and per interstate edge, so the
        # chain is as long as the program has blocks and a recursive walk overran Python's stack on
        # the first application-sized graph it met (CloudSC, ~2k blocks -- "RecursionError: maximum
        # recursion depth exceeded" out of ``apply_gpu_transformations``, which is where the whole
        # GPU canonicalization of that kernel stopped). Children are pushed REVERSED so they pop in
        # ``node.next`` order, and a node that reaches the close node contributes itself and none of
        # its remaining children.
        #
        # Each node is walked ONCE. Every conditional in the section is a diamond whose arms meet
        # again at its close node, so walking the section once per ROUTE doubles the work per
        # conditional in a row: ls3df_scf's SCF loop holds 48 of them, 2^48 routes, and the canon
        # GPU offload never finished. How many routes there are is :meth:`has_one_route`'s question.
        result: list[OffloadingIRNode] = []
        seen: set = set()
        stack = [self]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            children = []
            for next in node.next:
                if next == self.close:  # a tail: a node that points at this section's end (close-node)
                    result.append(node)
                    children.clear()
                    break
                children.append(next)
            stack.extend(reversed(children))
        return result

    def has_one_route(self) -> bool:
        """Whether exactly one route leads from this open node to its close node.

        A route ends at the first node that points at the close node, so this is whether
        :meth:`get_all_tails` would list one tail if it walked the section once per route instead
        of once per node: a conditional anywhere inside it makes two routes even when both arms
        meet again before the section's single tail. Counted per node, saturating at two, so the
        cost is one visit per node however many routes there are.
        """
        assert self.is_open_node()
        routes: dict = {}
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

    # static makers
    @staticmethod
    def new_open_node(block: ControlFlowBlock) -> 'OffloadingIRNode':
        close = OffloadingIRNode(OffloadingIRNode.CLOSE, None, OrderedSet(), OrderedSet(), [], None)
        close.debug_name = f"_close_{block.label}"

        type: int
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
    def new_edge_node(edge: InterstateEdge, cpu_set: OrderedSet[str]) -> 'OffloadingIRNode':
        edge_node = OffloadingIRNode(OffloadingIRNode.EDGE, edge, cpu_set, OrderedSet(), [], None)
        edge_node.debug_name = f"_edge_{edge.label}"
        return edge_node

    @staticmethod
    def get_type_as_str(type: int) -> str:
        match type:
            case OffloadingIRNode.STATE:
                return "state"
            case OffloadingIRNode.OPEN:
                return "open"
            case OffloadingIRNode.CLOSE:
                return "close"
            case OffloadingIRNode.OPEN_LOOP:
                return "loop"
            case OffloadingIRNode.OPEN_COND:
                return "cond"
        raise ValueError(f"Invalid IR type to convert to string: {type}")
