# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dataclasses import dataclass, field
import copy

from dace import dtypes, nodes, data, subsets
from dace.properties import CodeBlock
from dace.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, LoopRegion, SDFGState
from dace.symbolic import symbol
from dace.memlet import Memlet
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Literal, Optional, Sequence, Set, Tuple, Union

if TYPE_CHECKING:
    from dace import SDFG

INDENTATION = '  '


class UnsupportedScopeException(Exception):
    pass


def _format_frontend_range(start: str, stop: str, step: str) -> str:
    if step == '1':
        return f'{start}:{stop}'
    return f'{start}:{stop}:{step}'


@dataclass(frozen=True)
class FrontendLoop:
    """
    Lightweight loop metadata used by frontends that construct schedule trees
    without first materializing an SDFG control-flow region.
    """
    loop_condition: CodeBlock
    init_statement: Optional[CodeBlock] = None
    update_statement: Optional[CodeBlock] = None
    loop_variable: Optional[str] = None
    inverted: bool = False
    update_before_condition: bool = False


@dataclass(frozen=True)
class FrontendMap:
    """
    Lightweight map metadata used by frontends that construct schedule trees
    without first materializing SDFG map nodes.
    """
    params: Sequence[str]
    ranges: Sequence[Tuple[str, str, str]]
    schedule: Optional[str] = None


@dataclass(frozen=True)
class FrontendConsume:
    """
    Lightweight consume-scope metadata used by frontend-produced schedule
    trees.
    """
    pe_index: str
    num_pes: str
    condition: Optional[CodeBlock] = None


@dataclass(frozen=True)
class FrontendTasklet:
    """
    Lightweight tasklet metadata used by frontend-produced schedule trees.
    """
    name: str
    code: CodeBlock = field(default_factory=lambda: CodeBlock(''))


@dataclass(frozen=True)
class FrontendLibrary:
    """
    Lightweight library call metadata used by frontend-produced schedule trees.
    """
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrontendFunctionCall:
    """
    Lightweight function-call metadata used by frontend-produced schedule trees
    to represent a call to another ``@dace.program``.
    """
    callee_name: str
    arguments: Dict[str, str] = field(default_factory=dict)  # callee_param -> caller_expression


@dataclass
class ScheduleTreeNode:
    parent: Optional['ScheduleTreeScope'] = field(default=None, init=False, repr=False)

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + 'UNSUPPORTED'

    def preorder_traversal(self) -> Iterator['ScheduleTreeNode']:
        """
        Traverse tree nodes in a pre-order manner.
        """
        yield self


@dataclass
class ScheduleTreeScope(ScheduleTreeNode):
    children: List['ScheduleTreeNode']
    containers: Optional[Dict[str, data.Data]] = field(default_factory=dict, init=False)
    symbols: Optional[Dict[str, symbol]] = field(default_factory=dict, init=False)

    def __init__(self, children: Optional[List['ScheduleTreeNode']] = None):
        self.children = children or []
        if self.children:
            for child in children:
                child.parent = self
        self.containers = {}
        self.symbols = {}

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

    # TODO: Helper function that gets input/output memlets of the scope


@dataclass
class ScheduleTreeRoot(ScheduleTreeScope):
    """
    A root of an SDFG schedule tree. This is a schedule tree scope with additional information on
    the available descriptors, symbol types, and constants of the tree, aka the descriptor repository.
    """
    name: str
    containers: Dict[str, data.Data]
    symbols: Dict[str, dtypes.typeclass]
    constants: Dict[str, Tuple[data.Data, Any]]
    callback_mapping: Dict[str, str]
    arg_names: List[str]

    def __init__(
        self,
        *,
        name: str,
        children: List[ScheduleTreeNode],
        containers: Optional[Dict[str, data.Data]] = None,
        symbols: Optional[Dict[str, dtypes.typeclass]] = None,
        constants: Optional[Dict[str, Tuple[data.Data, Any]]] = None,
        callback_mapping: Optional[Dict[str, str]] = None,
        arg_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__(children=children)

        self.name = name
        self.containers = containers if containers is not None else dict()
        self.symbols = symbols if symbols is not None else dict()
        self.constants = constants if constants is not None else dict()
        self.callback_mapping = callback_mapping if callback_mapping is not None else dict()
        self.arg_names = arg_names if arg_names is not None else list()


@dataclass
class ControlFlowScope(ScheduleTreeScope):
    pass


@dataclass
class FunctionCallScope(ControlFlowScope):
    """
    Represents a call to another ``@dace.program`` whose schedule tree body
    is inlined as children of this scope.
    """
    call: FrontendFunctionCall = field(default_factory=lambda: FrontendFunctionCall(''))

    def as_string(self, indent: int = 0):
        args = ', '.join(f'{k}={v}' for k, v in self.call.arguments.items())
        result = indent * INDENTATION + f'call {self.call.callee_name}({args}):\n'
        return result + super().as_string(indent)


@dataclass
class SDFGCallNode(ScheduleTreeNode):
    """
    Represents a call to an SDFG-valued callee that remains explicit in the
    schedule tree instead of being inlined structurally.
    """
    sdfg: 'SDFG'
    call: FrontendFunctionCall = field(default_factory=lambda: FrontendFunctionCall(''))
    return_targets: List[str] = field(default_factory=list)

    def as_string(self, indent: int = 0):
        args = ', '.join(f'{k}={v}' for k, v in self.call.arguments.items())
        call = f'sdfg_call {self.call.callee_name}({args})'
        if not self.return_targets:
            return indent * INDENTATION + call
        targets = ', '.join(self.return_targets)
        return indent * INDENTATION + f'{targets} = {call}'


@dataclass
class DataflowScope(ScheduleTreeScope):
    node: Union[nodes.EntryNode, FrontendMap, FrontendConsume]
    state: Optional[SDFGState] = None


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
    state: Union[SDFGState, str]

    def as_string(self, indent: int = 0):
        if isinstance(self.state, str):
            name = self.state
        else:
            name = self.state.name
        return indent * INDENTATION + f'label {name}:'


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
    edge: Optional[InterstateEdge] = None

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + f'assign {self.name} = {self.value.as_string}'


@dataclass
class ReassignExternalNode(ScheduleTreeNode):
    """
    Explicit reassignment of an external Python binding captured via
    ``global`` or ``nonlocal``.
    """
    name: str
    value: CodeBlock
    scope: Literal['global', 'nonlocal']

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + f'reassign_external {self.scope} {self.name} = {self.value.as_string}'


@dataclass
class StatementNode(ScheduleTreeNode):
    """
    Opaque statement node used by source frontends when a statement has not yet
    been lowered into a more structured dataflow node.
    """
    code: CodeBlock

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + f'stmt {self.code.as_string}'


@dataclass
class PythonCallbackNode(ScheduleTreeNode):
    """
    Python code that cannot be represented in the dataflow model and must be
    executed via native Python callback at runtime. Distinct from StatementNode
    in that it explicitly marks code as never lowerable.
    """
    code: CodeBlock
    reason: str
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + f'python_callback "{self.reason}" {{ {self.code.as_string} }}'


@dataclass
class RaiseNode(ScheduleTreeNode):
    """
    Explicit raise statement emitted by source frontends when the exception
    shape is known well enough to remain compilable.
    """
    exception_type: Optional[CodeBlock] = None
    args: List[CodeBlock] = field(default_factory=list)
    kwargs: Dict[str, CodeBlock] = field(default_factory=dict)

    def as_string(self, indent: int = 0):
        if self.exception_type is None:
            return indent * INDENTATION + 'raise'

        call_args = [argument.as_string for argument in self.args]
        call_args.extend(f'{name}={value.as_string}' for name, value in self.kwargs.items())
        rendered = self.exception_type.as_string
        if call_args:
            rendered = f'{rendered}({", ".join(call_args)})'
        return indent * INDENTATION + f'raise {rendered}'


@dataclass
class ReturnNode(ScheduleTreeNode):
    """
    Explicit return node used by source frontends before lowering returns to a
    backend-specific representation.
    """
    values: List[CodeBlock] = field(default_factory=list)

    def as_string(self, indent: int = 0):
        if not self.values:
            return indent * INDENTATION + 'return'
        joined = ', '.join(value.as_string for value in self.values)
        return indent * INDENTATION + f'return {joined}'


@dataclass
class LoopScope(ControlFlowScope):
    """
    General loop scope (representing a loop region).
    """
    loop: Union[LoopRegion, FrontendLoop]

    def _check_loop_variant(
        self
    ) -> Union[Literal['for'], Literal['while'], Literal['do-while'], Literal['do-for-uncond-increment'],
               Literal['do-for']]:
        if self.loop.update_statement and self.loop.init_statement and self.loop.loop_variable:
            if self.loop.inverted:
                if self.loop.update_before_condition:
                    return 'do-for-uncond-increment'
                else:
                    return 'do-for'
            else:
                return 'for'
        else:
            if self.loop.inverted:
                return 'do-while'
            else:
                return 'while'

    def as_string(self, indent: int = 0):
        loop = self.loop
        loop_variant = self._check_loop_variant()
        if loop_variant == 'do-for-uncond-increment':
            pre_header = indent * INDENTATION + f'{loop.init_statement.as_string}\n'
            header = indent * INDENTATION + 'do:\n'
            pre_footer = (indent + 1) * INDENTATION + f'{loop.update_statement.as_string}\n'
            footer = indent * INDENTATION + f'while {loop.loop_condition.as_string}'
            return pre_header + header + super().as_string(indent) + '\n' + pre_footer + footer
        elif loop_variant == 'do-for':
            pre_header = indent * INDENTATION + f'{loop.init_statement.as_string}\n'
            header = indent * INDENTATION + 'while True:\n'
            pre_footer = (indent + 1) * INDENTATION + f'if (not {loop.loop_condition.as_string}):\n'
            pre_footer += (indent + 2) * INDENTATION + 'break\n'
            footer = (indent + 1) * INDENTATION + f'{loop.update_statement.as_string}\n'
            return pre_header + header + super().as_string(indent) + '\n' + pre_footer + footer
        elif loop_variant == 'for':
            result = (indent * INDENTATION + f'for {loop.init_statement.as_string}; ' +
                      f'{loop.loop_condition.as_string}; ' + f'{loop.update_statement.as_string}:\n')
            return result + super().as_string(indent)
        elif loop_variant == 'while':
            result = indent * INDENTATION + f'while {loop.loop_condition.as_string}:\n'
            return result + super().as_string(indent)
        else:  # 'do-while'
            header = indent * INDENTATION + 'do:\n'
            footer = indent * INDENTATION + f'while {loop.loop_condition.as_string}'
            return header + super().as_string(indent) + '\n' + footer


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
        return result + super(IfScope, self).as_string(indent)


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
        if isinstance(self.node, FrontendMap):
            rangestr = ', '.join(_format_frontend_range(start, stop, step) for start, stop, step in self.node.ranges)
            params = ', '.join(self.node.params)
        else:
            rangestr = ', '.join(subsets.Range.dim_to_string(d) for d in self.node.map.range)
            params = ', '.join(self.node.map.params)
        result = indent * INDENTATION + f'map {params} in [{rangestr}]:\n'
        return result + super().as_string(indent)


@dataclass
class ConsumeScope(DataflowScope):
    """
    Consume scope.
    """

    def as_string(self, indent: int = 0):
        if isinstance(self.node, FrontendConsume):
            cond = 'stream not empty' if self.node.condition is None else self.node.condition.as_string
            pe_index = self.node.pe_index
            num_pes = self.node.num_pes
        else:
            node: nodes.ConsumeEntry = self.node
            cond = 'stream not empty' if node.consume.condition is None else node.consume.condition.as_string
            pe_index = node.consume.pe_index
            num_pes = node.consume.num_pes
        result = indent * INDENTATION + f'consume (PE {pe_index} out of {num_pes}) while {cond}:\n'
        return result + super().as_string(indent)


@dataclass
class TaskletNode(ScheduleTreeNode):
    node: Union[nodes.Tasklet, FrontendTasklet]
    in_memlets: Dict[str, Memlet]
    out_memlets: Dict[str, Memlet]

    def as_string(self, indent: int = 0):
        in_memlets = ', '.join(f'{v}' for v in self.in_memlets.values())
        out_memlets = ', '.join(f'{v}' for v in self.out_memlets.values())
        if not out_memlets:
            return indent * INDENTATION + f'tasklet({in_memlets})'
        return indent * INDENTATION + f'{out_memlets} = tasklet({in_memlets})'


@dataclass
class LibraryCall(ScheduleTreeNode):
    node: Union[nodes.LibraryNode, FrontendLibrary]
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
        if isinstance(self.node, FrontendLibrary):
            libname = self.node.name
            own_properties = ', '.join(f'{k}={v}' for k, v in self.node.properties.items())
        else:
            libname = type(self.node).__name__
            own_properties = ', '.join(f'{k}={getattr(self.node, k)}' for k, v in self.node.__properties__.items()
                                       if v.owner not in {nodes.Node, nodes.CodeNode, nodes.LibraryNode})
        call = f'library {libname}[{own_properties}]({in_memlets})'
        if not out_memlets:
            return indent * INDENTATION + call
        return indent * INDENTATION + f'{out_memlets} = {call}'


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
    memlet: Optional[Memlet]
    src_desc: Union[data.Data, nodes.CodeNode]
    ref_desc: data.Data
    source_expr: Optional[str] = None

    def as_string(self, indent: int = 0):
        if isinstance(self.src_desc, nodes.CodeNode):
            return indent * INDENTATION + f'{self.target} = refset from {type(self.src_desc).__name__.lower()}'
        if self.source_expr is not None:
            return indent * INDENTATION + f'{self.target} = refset to {self.source_expr}'
        return indent * INDENTATION + f'{self.target} = refset to {self.memlet}'


def clone_descriptor_with_shape(descriptor: data.Data, shape: Sequence[Any]) -> data.Data:
    """
    Clone a data descriptor and update its shape if supported.
    """
    result = copy.deepcopy(descriptor)
    if hasattr(result, 'set_shape'):
        result.set_shape(list(shape))
    elif hasattr(result, 'shape'):
        result.shape = list(shape)
    return result


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


def validate_has_no_other_node_types(stree: ScheduleTreeScope) -> None:
    """
    Validates that the schedule tree contains only nodes of type ScheduleTreeNode or its subclasses.
    Raises an exception if any other node type is found.
    """
    for child in stree.children:
        if not isinstance(child, ScheduleTreeNode):
            raise RuntimeError(f'Unsupported node type: {type(child).__name__}')
        if isinstance(child, ScheduleTreeScope):
            validate_has_no_other_node_types(child)
