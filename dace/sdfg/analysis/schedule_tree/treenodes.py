# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import ast
from dataclasses import dataclass, field
from dace import nodes, data, subsets
from dace.codegen import control_flow as cf
from dace.properties import CodeBlock
from dace.sdfg.memlet_utils import MemletSet
from dace.sdfg.sdfg import InterstateEdge, SDFG, memlets_in_ast
from dace.sdfg.state import SDFGState
from dace.symbolic import symbol
from dace.memlet import Memlet
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

INDENTATION = '  '


class UnsupportedScopeException(Exception):
    pass


@dataclass
class ScheduleTreeNode:
    parent: Optional['ScheduleTreeScope'] = field(default=None, init=False)

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + 'UNSUPPORTED'

    def preorder_traversal(self) -> Iterator['ScheduleTreeNode']:
        """
        Traverse tree nodes in a pre-order manner.
        """
        yield self

    def get_root(self) -> 'ScheduleTreeRoot':
        if self.parent is None:
            raise ValueError('Non-root schedule tree node has no parent')
        return self.parent.get_root()

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        """
        Returns a set of inputs for this node. For scopes, returns the union of its contents.

        :param root: An optional argument specifying the schedule tree's root. If not given,
                     the value is computed from the current tree node.
        :return: A set of memlets representing the inputs of this node.
        """
        raise NotImplementedError

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        """
        Returns a set of outputs for this node. For scopes, returns the union of its contents.

        :param root: An optional argument specifying the schedule tree's root. If not given,
                     the value is computed from the current tree node.
        :return: A set of memlets representing the inputs of this node.
        """
        raise NotImplementedError


@dataclass
class ScheduleTreeScope(ScheduleTreeNode):
    children: List['ScheduleTreeNode']

    def __init__(self, children: Optional[List['ScheduleTreeNode']] = None):
        self.children = children or []
        if self.children:
            for child in children:
                child.parent = self

    def as_string(self, indent: int = 0):
        if not self.children:
            return (indent + 1) * INDENTATION + 'pass'
        return '\n'.join([child.as_string(indent + 1) for child in self.children])

    def preorder_traversal(self) -> Iterator['ScheduleTreeNode']:
        """
        Traverse tree nodes in a pre-order manner.
        """
        yield from super().preorder_traversal()
        for child in self.children:
            yield from child.preorder_traversal()

    # TODO: Missing propagation and locals
    # TODO: Add symbol ranges as an argument
    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        return MemletSet().union(*(c.input_memlets(root) for c in self.children))

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        return MemletSet().union(*(c.output_memlets(root) for c in self.children))


@dataclass
class ScheduleTreeRoot(ScheduleTreeScope):
    """
    A root of an SDFG schedule tree. This is a schedule tree scope with additional information on
    the available descriptors, symbol types, and constants of the tree, aka the descriptor repository.
    """
    name: str
    containers: Dict[str, data.Data] = field(default_factory=dict)
    symbols: Dict[str, symbol] = field(default_factory=dict)
    constants: Dict[str, Tuple[data.Data, Any]] = field(default_factory=dict)
    callback_mapping: Dict[str, str] = field(default_factory=dict)
    arg_names: List[str] = field(default_factory=list)

    def as_sdfg(self) -> SDFG:
        from dace.sdfg.analysis.schedule_tree import tree_to_sdfg as t2s  # Avoid import loop
        return t2s.from_schedule_tree(self)

    def get_root(self) -> 'ScheduleTreeRoot':
        return self


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

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        return set()

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        return set()


@dataclass
class GotoNode(ScheduleTreeNode):
    target: Optional[str] = None  #: If None, equivalent to "goto exit" or "return"

    def as_string(self, indent: int = 0):
        name = self.target or 'exit'
        return indent * INDENTATION + f'goto {name}'

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        return set()

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        return set()


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

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        root = root if root is not None else self.get_root()
        return set(self.edge.get_read_memlets(root.containers))

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        return set()


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

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        root = root if root is not None else self.get_root()
        result = set()
        result.update(memlets_in_ast(ast.parse(self.header.init), root.containers))
        result.update(memlets_in_ast(self.header.condition.code[0], root.containers))
        result.update(memlets_in_ast(ast.parse(self.header.update), root.containers))
        result.update(super().input_memlets(root))
        return result


@dataclass
class WhileScope(ControlFlowScope):
    """
    While loop scope.
    """
    header: cf.WhileScope

    def as_string(self, indent: int = 0):
        result = indent * INDENTATION + f'while {self.header.test.as_string}:\n'
        return result + super().as_string(indent)

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        root = root if root is not None else self.get_root()
        result = set()
        result.update(memlets_in_ast(self.header.test.code[0], root.containers))
        result.update(super().input_memlets(root))
        return result


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

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        root = root if root is not None else self.get_root()
        result = set()
        result.update(memlets_in_ast(self.header.test.code[0], root.containers))
        result.update(super().input_memlets(root))
        return result


@dataclass
class IfScope(ControlFlowScope):
    """
    If branch scope.
    """
    condition: CodeBlock

    def as_string(self, indent: int = 0):
        result = indent * INDENTATION + f'if {self.condition.as_string}:\n'
        return result + super().as_string(indent)

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        root = root if root is not None else self.get_root()
        result = set()
        result.update(memlets_in_ast(self.condition.code[0], root.containers))
        result.update(super().input_memlets(root))
        return result


@dataclass
class StateIfScope(IfScope):
    """
    A special class of an if scope in general blocks for if statements that are part of a state transition.
    """

    def as_string(self, indent: int = 0):
        result = indent * INDENTATION + f'stateif {self.condition.as_string}:\n'
        return result + super(IfScope, self).as_string(indent)


@dataclass
class BreakNode(ScheduleTreeNode):
    """
    Represents a break statement.
    """

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + 'break'

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        return set()

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        return set()


@dataclass
class ContinueNode(ScheduleTreeNode):
    """
    Represents a continue statement.
    """

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + 'continue'

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        return set()

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        return set()


@dataclass
class ElifScope(ControlFlowScope):
    """
    Else-if branch scope.
    """
    condition: CodeBlock

    def as_string(self, indent: int = 0):
        result = indent * INDENTATION + f'elif {self.condition.as_string}:\n'
        return result + super().as_string(indent)

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        root = root if root is not None else self.get_root()
        result = set()
        result.update(memlets_in_ast(self.condition.code[0], root.containers))
        result.update(super().input_memlets(root))
        return result


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
        if not out_memlets:
            return indent * INDENTATION + f'tasklet({in_memlets})'
        return indent * INDENTATION + f'{out_memlets} = tasklet({in_memlets})'

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        return set(self.in_memlets.values())

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        return set(self.out_memlets.values())


@dataclass
class LibraryCall(ScheduleTreeNode):
    node: nodes.LibraryNode
    in_memlets: Union[Dict[str, Memlet], MemletSet]
    out_memlets: Union[Dict[str, Memlet], MemletSet]

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

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        if isinstance(self.in_memlets, set):
            return set(self.in_memlets)
        return set(self.in_memlets.values())

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        if isinstance(self.out_memlets, set):
            return set(self.out_memlets)
        return set(self.out_memlets.values())


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

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        return {self.memlet}

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        root = root if root is not None else self.get_root()
        if self.memlet.other_subset is not None:
            return {Memlet(data=self.target, subset=self.memlet.other_subset, wcr=self.memlet.wcr)}

        return {Memlet.from_array(self.target, root.containers[self.target], self.memlet.wcr)}


@dataclass
class DynScopeCopyNode(ScheduleTreeNode):
    """
    A special case of a copy node that is used in dynamic scope inputs (e.g., dynamic map ranges).
    """
    target: str
    memlet: Memlet

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + f'{self.target} = dscopy {self.memlet.data}[{self.memlet.subset}]'

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        return {self.memlet}

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        return set()


@dataclass
class ViewNode(ScheduleTreeNode):
    target: str  #: View name
    source: str  #: Viewed container name
    memlet: Memlet
    src_desc: data.Data
    view_desc: data.Data

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + f'{self.target} = view {self.memlet} as {self.view_desc.shape}'

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        return {self.memlet}

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        return {Memlet.from_array(self.target, self.view_desc)}


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
    src_desc: Union[data.Data, nodes.CodeNode]
    ref_desc: data.Data

    def as_string(self, indent: int = 0):
        if isinstance(self.src_desc, nodes.CodeNode):
            return indent * INDENTATION + f'{self.target} = refset from {type(self.src_desc).__name__.lower()}'
        return indent * INDENTATION + f'{self.target} = refset to {self.memlet}'

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        return {self.memlet}

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        return {Memlet.from_array(self.target, self.ref_desc)}


@dataclass
class StateBoundaryNode(ScheduleTreeNode):
    """
    A node that represents a state boundary (e.g., when a write-after-write is encountered). This node
    is used only during conversion from a schedule tree to an SDFG.
    """
    due_to_control_flow: bool = False

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + 'state boundary'

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        return set()

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None) -> MemletSet:
        return set()


# Classes based on Python's AST NodeVisitor/NodeTransformer for schedule tree nodes
class ScheduleNodeVisitor:

    def visit(self, node: ScheduleTreeNode):
        """Visit a node."""
        if isinstance(node, list):
            return [self.visit(snode) for snode in node]
        if isinstance(node, ScheduleTreeScope) and hasattr(self, 'visit_scope'):
            return self.visit_scope(node)

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
