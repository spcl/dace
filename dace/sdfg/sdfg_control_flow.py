# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Optional, Set

from dace import data as dt
from dace.memlet import Memlet
from dace.sdfg import InterstateEdge, SDFGState
from dace.sdfg.nodes import Node
from dace.sdfg.graph import OrderedDiGraph, OrderedMultiDiConnectorGraph
from dace.properties import Property, make_properties, CodeProperty, CodeBlock


class ControlFlowBlock(object):

    _parent_cfg: Optional['ControlFlowGraph'] = None
    _label: str

    def __init__(self, label: str='', parent: Optional['ControlFlowGraph']=None):
        super(ControlFlowBlock, self).__init__()
        self._label = label
        self._parent_cfg = parent

    def __str__(self):
        return self._label

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, label: str):
        self._label = label

    @property
    def name(self) -> str:
        return self._label

    @property
    def parent_cfg(self):
        """ Returns the parent graph of this block. """
        return self._parent_cfg

    @parent_cfg.setter
    def parent_cfg(self, value):
        self._parent_cfg = value


class BasicBlock(OrderedMultiDiConnectorGraph[Node, Memlet], ControlFlowBlock):

    def __init__(self, label: str='', parent: Optional['ControlFlowGraph']=None):
        OrderedMultiDiConnectorGraph[Node, Memlet].__init__(self)
        ControlFlowBlock.__init__(self, label, parent)


class ControlFlowGraph(OrderedDiGraph[ControlFlowBlock, InterstateEdge]):

    def __init__(self):
        super(ControlFlowGraph, self).__init__()

        self._labels: Set[str] = set()
        self._start_block: Optional[int] = None
        self._cached_start_block: Optional[ControlFlowBlock] = None

    def add_node(self, node, is_start_block=False):
        if not isinstance(node, ControlFlowBlock):
            raise TypeError('Expected ControlFlowBlock, got ' + str(type(node)))
        super(ControlFlowGraph, self).add_node(node)
        self._cached_start_block = None
        if is_start_block is True:
            self.start_block = len(self.nodes()) - 1
            self._cached_start_block = node

    def add_state(self, label=None, is_start_block=False) -> SDFGState:
        if self._labels is None or len(self._labels) != self.number_of_nodes():
            self._labels = set(s.label for s in self.nodes())
        label = label or 'state'
        existing_labels = self._labels
        label = dt.find_new_name(label, existing_labels)
        state = SDFGState(label, self)
        self._labels.add(label)
        self.add_node(state, is_start_block=is_start_block)
        return state

    @property
    def start_block(self):
        """ Returns the starting block of this ControlFlowGraph. """
        if self._cached_start_block is not None:
            return self._cached_start_block

        source_nodes = self.source_nodes()
        if len(source_nodes) == 1:
            self._cached_start_block = source_nodes[0]
            return source_nodes[0]
        # If the starting block is ambiguous allow manual override.
        if self._start_block is not None:
            self._cached_start_block = self.node(self._start_block)
            return self._cached_start_block
        raise ValueError('Ambiguous or undefined starting block for ControlFlowGraph, '
                         'please use "is_start_block=True" when adding the '
                         'starting block with "add_state" or "add_node"')

    @start_block.setter
    def start_block(self, block_id):
        """ Manually sets the starting block of this ControlFlowGraph.

            :param block_id: The node ID (use `node_id(block)`) of the block to set.
        """
        if block_id < 0 or block_id >= self.number_of_nodes():
            raise ValueError("Invalid state ID")
        self._start_block = block_id
        self._cached_start_block = self.node(block_id)


class ScopeBlock(ControlFlowGraph, ControlFlowBlock):

    def __init__(self, label: str='', parent: Optional[ControlFlowGraph]=None):
        ControlFlowGraph.__init__(self)
        ControlFlowBlock.__init__(self, label, parent)


@make_properties
class LoopScopeBlock(ScopeBlock):

    update_statement = CodeProperty(optional=True, allow_none=True, default=None)
    init_statement = CodeProperty(optional=True, allow_none=True, default=None)
    scope_condition = CodeProperty()
    inverted = Property(dtype=bool, default=False)

    def __init__(self,
                 loop_var: str,
                 initialize_expr: str,
                 condition_expr: str,
                 update_expr: str,
                 label: str = '',
                 parent: Optional[ControlFlowGraph] = None,
                 inverted: bool = False):
        super(LoopScopeBlock, self).__init__(label, parent)

        if initialize_expr is not None:
            self.init_statement = CodeBlock('%s = %s' % (loop_var, initialize_expr))
        else:
            self.init_statement = None

        if condition_expr:
            self.scope_condition = CodeBlock(condition_expr)
        else:
            self.scope_condition = CodeBlock('True')

        if update_expr is not None:
            self.update_statement = CodeBlock('%s = %s' % (loop_var, update_expr))
        else:
            self.update_statement = None

        self.inverted = inverted


@make_properties
class BranchScopeBlock(ScopeBlock):

    def __init__(self):
        super(BranchScopeBlock, self).__init__()
