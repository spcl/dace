# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Dict, List, Optional, Set, Generator

import dace
from dace import symbolic, data as dt
from dace.memlet import Memlet
from dace.properties import CodeBlock, CodeProperty, Property, make_properties
from dace.sdfg import nodes as nd
from dace.sdfg.graph import OrderedDiGraph, OrderedMultiDiConnectorGraph


@make_properties
class ControlFlowBlock(object):

    is_collapsed = Property(dtype=bool, desc='Show this block as collapsed', default=False)

    _parent_cfg: Optional['ControlFlowGraph'] = None
    _label: str

    def __init__(self, label: str='', parent: Optional['ControlFlowGraph']=None):
        super(ControlFlowBlock, self).__init__()
        self._label = label
        self._parent_cfg = parent
        self._default_lineinfo = None
        self.is_collapsed = False

    def set_default_lineinfo(self, lineinfo: dace.dtypes.DebugInfo):
        """
        Sets the default source line information to be lineinfo, or None to
        revert to default mode.
        """
        self._default_lineinfo = lineinfo

    def data_nodes(self) -> List[nd.AccessNode]:
        return []

    def replace_dict(self,
                     repl: Dict[str, str],
                     symrepl: Optional[Dict[symbolic.SymbolicType, symbolic.SymbolicType]] = None):
        """ Finds and replaces all occurrences of a set of symbols or arrays in this state.

            :param repl: Mapping from names to replacements.
            :param symrepl: Optional symbolic version of ``repl``.
        """
        from dace.sdfg.replace import replace_dict
        replace_dict(self, repl, symrepl)

    def to_json(self, parent=None):
        tmp = {}
        tmp['id'] = parent.node_id(self) if parent is not None else None
        tmp['label'] = self._label
        tmp['collapsed'] = self.is_collapsed
        return tmp

    def __str__(self):
        return self._label

    def __repr__(self) -> str:
        return f'ControlFlowBlock ({self.label})'

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


@make_properties
class BasicBlock(OrderedMultiDiConnectorGraph[nd.Node, Memlet], ControlFlowBlock):

    def __init__(self, label: str='', parent: Optional['ControlFlowGraph']=None):
        OrderedMultiDiConnectorGraph.__init__(self)
        ControlFlowBlock.__init__(self, label, parent)

    def __repr__(self) -> str:
        return f'BasicBlock ({self.label})'


@make_properties
class ControlFlowGraph(OrderedDiGraph[ControlFlowBlock, 'dace.sdfg.InterstateEdge']):

    def __init__(self):
        super(ControlFlowGraph, self).__init__()

        self._labels: Set[str] = set()
        self._start_block: Optional[int] = None
        self._cached_start_block: Optional[ControlFlowBlock] = None

    def add_edge(self, src: 'ControlFlowBlock', dst: 'ControlFlowBlock', data: 'dace.sdfg.InterstateEdge'):
        """ Adds a new edge to the graph. Must be an InterstateEdge or a subclass thereof.

            :param u: Source node.
            :param v: Destination node.
            :param edge: The edge to add.
        """
        if not isinstance(src, ControlFlowBlock):
            raise TypeError('Expected ControlFlowBlock, got ' + str(type(src)))
        if not isinstance(dst, ControlFlowBlock):
            raise TypeError('Expected ControlFlowBlock, got ' + str(type(dst)))
        if not isinstance(data, dace.sdfg.InterstateEdge):
            raise TypeError('Expected InterstateEdge, got ' + str(type(data)))
        if dst is self._cached_start_block:
            self._cached_start_block = None
        return super(ControlFlowGraph, self).add_edge(src, dst, data)

    def add_node(self, node, is_start_block=False):
        if not isinstance(node, ControlFlowBlock):
            raise TypeError('Expected ControlFlowBlock, got ' + str(type(node)))
        super().add_node(node)
        node.parent_cfg = self
        self._cached_start_block = None
        if is_start_block is True:
            self.start_block = len(self.nodes()) - 1
            self._cached_start_block = node

    def add_state(self, label=None, is_start_block=False, parent_sdfg=None) -> 'dace.SDFGState':
        if self._labels is None or len(self._labels) != self.number_of_nodes():
            self._labels = set(s.label for s in self.nodes())
        label = label or 'state'
        existing_labels = self._labels
        label = dt.find_new_name(label, existing_labels)
        state = dace.SDFGState(label, parent_sdfg)
        self._labels.add(label)
        self.add_node(state, is_start_block=is_start_block)
        return state

    def all_cfgs_recursive(self, recurse_into_sdfgs=True) -> Generator['ControlFlowGraph', None, None]:
        """ Iterate over this and all nested CFGs. """
        yield self
        for block in self.nodes():
            if isinstance(block, BasicBlock) and recurse_into_sdfgs:
                for node in block.nodes():
                    if isinstance(node, nd.NestedSDFG):
                        yield from node.sdfg.all_cfgs_recursive()
            elif isinstance(block, ControlFlowGraph):
                yield from block.all_cfgs_recursive()

    def all_sdfgs_recursive(self) -> Generator['dace.SDFG', None, None]:
        """ Iterate over this and all nested SDFGs. """
        for cfg in self.all_cfgs_recursive(recurse_into_sdfgs=True):
            if isinstance(cfg, dace.SDFG):
                yield cfg

    def all_states_recursive(self) -> Generator['dace.SDFGState', None, None]:
        """ Iterate over all states in this control flow graph. """
        for block in self.nodes():
            if isinstance(block, dace.SDFGState):
                yield block
            elif isinstance(block, ControlFlowGraph):
                yield from block.all_states_recursive()

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
            raise ValueError('Invalid state ID')
        self._start_block = block_id
        self._cached_start_block = self.node(block_id)


@make_properties
class ScopeBlock(ControlFlowGraph, ControlFlowBlock):

    def __init__(self, label: str='', parent: Optional[ControlFlowGraph]=None):
        ControlFlowGraph.__init__(self)
        ControlFlowBlock.__init__(self, label, parent)

    def data_nodes(self) -> List[nd.AccessNode]:
        """ Returns all data_nodes (arrays) present in this state. """
        data_nodes = []
        for n in self.nodes():
            data_nodes.append(n.data_nodes())

        return [n for n in self.nodes() if isinstance(n, nd.AccessNode)]

    def replace_dict(self,
                     repl: Dict[str, str],
                     symrepl: Optional[Dict[symbolic.SymbolicType, symbolic.SymbolicType]] = None):
        """ Finds and replaces all occurrences of a set of symbols or arrays in this state.

            :param repl: Mapping from names to replacements.
            :param symrepl: Optional symbolic version of ``repl``.
        """
        for n in self.nodes():
            n.replace_dict(repl, symrepl)

    def to_json(self, parent=None):
        graph_json = ControlFlowGraph.to_json(self)
        block_json = ControlFlowBlock.to_json(self, parent)
        graph_json.update(block_json)
        return graph_json

    def all_nodes_recursive(self):
        for node in self.nodes():
            yield node, self
            if isinstance(node, (ScopeBlock, dace.sdfg.StateGraphView)):
                yield from node.all_nodes_recursive()

    def __str__(self):
        return ControlFlowBlock.__str__(self)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} ({self.label})'


@make_properties
class LoopScopeBlock(ScopeBlock):

    update_statement = CodeProperty(optional=True, allow_none=True, default=None)
    init_statement = CodeProperty(optional=True, allow_none=True, default=None)
    scope_condition = CodeProperty(allow_none=True, default=None)
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

    def to_json(self, parent=None):
        return super().to_json(parent)


@make_properties
class BranchScopeBlock(ScopeBlock):

    def __init__(self):
        super(BranchScopeBlock, self).__init__()
