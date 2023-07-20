# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from dataclasses import dataclass, field
from dace import nodes, data, subsets
from dace.codegen import control_flow as cf
from dace.dtypes import TYPECLASS_TO_STRING
from dace.properties import CodeBlock
from dace.sdfg import SDFG, InterstateEdge
from dace.sdfg.state import SDFGState
from dace.symbolic import symbol
from dace.memlet import Memlet
from functools import reduce
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

INDENTATION = '  '


class UnsupportedScopeException(Exception):
    pass


@dataclass
class ScheduleTreeNode:
    # sdfg: SDFG
    # sdfg: Optional[SDFG] = field(default=None, init=False)
    parent: Optional['ScheduleTreeScope'] = field(default=None, init=False)

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + 'UNSUPPORTED'

    def preorder_traversal(self) -> Iterator['ScheduleTreeNode']:
        """
        Traverse tree nodes in a pre-order manner.
        """
        yield self


@dataclass
class ScheduleTreeScope(ScheduleTreeNode):
    sdfg: SDFG
    top_level: bool
    children: List['ScheduleTreeNode']
    containers: Optional[Dict[str, data.Data]] = field(default_factory=dict, init=False)
    symbols: Optional[Dict[str, symbol]] = field(default_factory=dict, init=False)

    def __init__(self,
                 sdfg: Optional[SDFG] = None,
                 top_level: Optional[bool] = False,
                 children: Optional[List['ScheduleTreeNode']] = None):
        self.sdfg = sdfg
        self.top_level = top_level
        self.children = children or []
        if self.children:
            for child in children:
                child.parent = self

    def as_string(self, indent: int = 0):
        return '\n'.join([child.as_string(indent + 1) for child in self.children])

    def preorder_traversal(self) -> Iterator['ScheduleTreeNode']:
        """
        Traverse tree nodes in a pre-order manner.
        """
        yield from super().preorder_traversal()
        for child in self.children:
            yield from child.preorder_traversal()

    # TODO: Get input/output memlets?


@dataclass
class ControlFlowScope(ScheduleTreeScope):
    pass


@dataclass
class DataflowScope(ScheduleTreeScope):
    node: nodes.EntryNode


@dataclass
class GBlock(ControlFlowScope):
    """
    General control flow block. Contains a list of states
    that can run in arbitrary order based on edges (gotos).
    Normally contains irreducible control flow.
    """

    def as_string(self, indent: int = 0):
        result = indent * INDENTATION + 'gblock:\n'
        return result + super().as_string(indent)


@dataclass
class StateLabel(ScheduleTreeNode):
    state: SDFGState

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + f'label {self.state.name}:'


@dataclass
class GotoNode(ScheduleTreeNode):
    target: Optional[str] = None  #: If None, equivalent to "goto exit" or "return"

    def as_string(self, indent: int = 0):
        name = self.target or 'exit'
        return indent * INDENTATION + f'goto {name}'


@dataclass
class AssignNode(ScheduleTreeNode):
    """
    Represents a symbol assignment that is not part of a structured control flow block.
    """
    name: str
    value: CodeBlock
    edge: InterstateEdge

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + f'assign {self.name} = {self.value.as_string}'


@dataclass
class ForScope(ControlFlowScope):
    """
    For loop scope.
    """
    header: cf.ForScope

    def as_string(self, indent: int = 0):
        node = self.header

        result = (indent * INDENTATION + f'for {node.itervar} = {node.init}; {node.condition.as_string}; '
                  f'{node.itervar} = {node.update}:\n')
        return result + super().as_string(indent)


@dataclass
class WhileScope(ControlFlowScope):
    """
    While loop scope.
    """
    header: cf.WhileScope

    def as_string(self, indent: int = 0):
        result = indent * INDENTATION + f'while {self.header.test.as_string}:\n'
        return result + super().as_string(indent)


@dataclass
class DoWhileScope(ControlFlowScope):
    """
    Do/While loop scope.
    """
    header: cf.DoWhileScope

    def as_string(self, indent: int = 0):
        header = indent * INDENTATION + 'do:\n'
        footer = indent * INDENTATION + f'while {self.header.test.as_string}\n'
        return header + super().as_string(indent) + footer


@dataclass
class IfScope(ControlFlowScope):
    """
    If branch scope.
    """
    condition: CodeBlock

    def as_string(self, indent: int = 0):
        result = indent * INDENTATION + f'if {self.condition.as_string}:\n'
        return result + super().as_string(indent)


@dataclass
class StateIfScope(IfScope):
    """
    A special class of an if scope in general blocks for if statements that are part of a state transition.
    """

    def as_string(self, indent: int = 0):
        result = indent * INDENTATION + f'stateif {self.condition.as_string}:\n'
        return result + super().as_string(indent)


@dataclass
class BreakNode(ScheduleTreeNode):
    """
    Represents a break statement.
    """

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + 'break'


@dataclass
class ContinueNode(ScheduleTreeNode):
    """
    Represents a continue statement.
    """

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + 'continue'


@dataclass
class ElifScope(ControlFlowScope):
    """
    Else-if branch scope.
    """
    condition: CodeBlock

    def as_string(self, indent: int = 0):
        result = indent * INDENTATION + f'elif {self.condition.as_string}:\n'
        return result + super().as_string(indent)


@dataclass
class ElseScope(ControlFlowScope):
    """
    Else branch scope.
    """

    def as_string(self, indent: int = 0):
        result = indent * INDENTATION + 'else:\n'
        return result + super().as_string(indent)


@dataclass
class MapScope(DataflowScope):
    """
    Map scope.
    """

    def as_string(self, indent: int = 0):
        rangestr = ', '.join(subsets.Range.dim_to_string(d) for d in self.node.map.range)
        result = indent * INDENTATION + f'map {", ".join(self.node.map.params)} in [{rangestr}]:\n'
        return result + super().as_string(indent)


@dataclass
class ConsumeScope(DataflowScope):
    """
    Consume scope.
    """

    def as_string(self, indent: int = 0):
        node: nodes.ConsumeEntry = self.node
        cond = 'stream not empty' if node.consume.condition is None else node.consume.condition.as_string
        result = indent * INDENTATION + f'consume (PE {node.consume.pe_index} out of {node.consume.num_pes}) while {cond}:\n'
        return result + super().as_string(indent)


@dataclass
class PipelineScope(DataflowScope):
    """
    Pipeline scope.
    """

    def as_string(self, indent: int = 0):
        rangestr = ', '.join(subsets.Range.dim_to_string(d) for d in self.node.map.range)
        result = indent * INDENTATION + f'pipeline {", ".join(self.node.map.params)} in [{rangestr}]:\n'
        return result + super().as_string(indent)


@dataclass
class TaskletNode(ScheduleTreeNode):
    node: nodes.Tasklet
    in_memlets: Dict[str, Memlet]
    out_memlets: Dict[str, Memlet]

    def as_string(self, indent: int = 0):
        in_memlets = ', '.join(f'{v}' for v in self.in_memlets.values())
        out_memlets = ', '.join(f'{v}' for v in self.out_memlets.values())
        return indent * INDENTATION + f'{out_memlets} = tasklet({in_memlets})'


@dataclass
class LibraryCall(ScheduleTreeNode):
    node: nodes.LibraryNode
    in_memlets: Union[Dict[str, Memlet], Set[Memlet]]
    out_memlets: Union[Dict[str, Memlet], Set[Memlet]]

    def as_string(self, indent: int = 0):
        if isinstance(self.in_memlets, set):
            in_memlets = ', '.join(f'{v}' for v in self.in_memlets)
        else:
            in_memlets = ', '.join(f'{v}' for v in self.in_memlets.values())
        if isinstance(self.out_memlets, set):
            out_memlets = ', '.join(f'{v}' for v in self.out_memlets)
        else:
            out_memlets = ', '.join(f'{v}' for v in self.out_memlets.values())
        libname = type(self.node).__name__
        # Get the properties of the library node without its superclasses
        own_properties = ', '.join(f'{k}={getattr(self.node, k)}' for k, v in self.node.__properties__.items()
                                   if v.owner not in {nodes.Node, nodes.CodeNode, nodes.LibraryNode})
        return indent * INDENTATION + f'{out_memlets} = library {libname}[{own_properties}]({in_memlets})'


@dataclass
class CopyNode(ScheduleTreeNode):
    target: str
    memlet: Memlet

    def as_string(self, indent: int = 0):
        if self.memlet.other_subset is not None and any(s != 0 for s in self.memlet.other_subset.min_element()):
            offset = f'[{self.memlet.other_subset}]'
        else:
            offset = ''
        if self.memlet.wcr is not None:
            wcr = f' with {self.memlet.wcr}'
        else:
            wcr = ''

        return indent * INDENTATION + f'{self.target}{offset} = copy {self.memlet.data}[{self.memlet.subset}]{wcr}'


@dataclass
class DynScopeCopyNode(ScheduleTreeNode):
    """
    A special case of a copy node that is used in dynamic scope inputs (e.g., dynamic map ranges).
    """
    target: str
    memlet: Memlet

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + f'{self.target} = dscopy {self.memlet.data}[{self.memlet.subset}]'


@dataclass
class ViewNode(ScheduleTreeNode):
    target: str  #: View name
    source: str  #: Viewed container name
    memlet: Memlet
    src_desc: data.Data
    view_desc: data.Data

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + f'{self.target} = view {self.memlet} as {self.view_desc.shape}'


@dataclass
class NView(ViewNode):
    """
    Nested SDFG view node. Subclass of a view that specializes in nested SDFG boundaries.
    """

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + f'{self.target} = nview {self.memlet} as {self.view_desc.shape}'


@dataclass
class RefSetNode(ScheduleTreeNode):
    """
    Reference set node. Sets a reference to a data container.
    """
    target: str
    memlet: Memlet
    src_desc: data.Data
    ref_desc: data.Data

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + f'{self.target} = refset to {self.memlet}'


# Classes based on Python's AST NodeVisitor/NodeTransformer for schedule tree nodes
class ScheduleNodeVisitor:

    def visit(self, node: ScheduleTreeNode):
        """Visit a node."""
        if isinstance(node, list):
            return [self.visit(snode) for snode in node]

        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ScheduleTreeNode):
        if isinstance(node, ScheduleTreeScope):
            for child in node.children:
                self.visit(child)


class ScheduleNodeTransformer(ScheduleNodeVisitor):

    def visit(self, node: ScheduleTreeNode):
        if isinstance(node, list):
            result = []
            for snode in node:
                new_node = self.visit(snode)
                if new_node is not None:
                    result.append(new_node)
            return result

        return super().visit(node)

    def generic_visit(self, node: ScheduleTreeNode):
        new_values = []
        if isinstance(node, ScheduleTreeScope):
            for value in node.children:
                if isinstance(value, ScheduleTreeNode):
                    value = self.visit(value)
                    if value is None:
                        continue
                    elif not isinstance(value, ScheduleTreeNode):
                        new_values.extend(value)
                        continue
                new_values.append(value)
            for val in new_values:
                val.parent = node
            node.children[:] = new_values
        return node
