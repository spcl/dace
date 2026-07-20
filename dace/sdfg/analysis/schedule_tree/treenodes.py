# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from collections.abc import Mapping
from dataclasses import dataclass, field
import copy
import sympy
from dace import nodes, data, subsets, dtypes, symbolic
from dace.properties import CodeBlock
from dace.sdfg import InterstateEdge
from dace.sdfg.memlet_utils import MemletSet
from dace.sdfg.propagation import propagate_subset
from dace.sdfg.sdfg import InterstateEdge, SDFG, memlets_in_ast
from dace.sdfg.state import LoopRegion, SDFGState
from dace.memlet import Memlet
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Literal, Optional, Sequence, Set, Tuple, Union

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
    """Base class for nodes in the schedule tree."""
    parent: Optional['ScheduleTreeScope'] = field(default=None, init=False, repr=False)

    def as_string(self, indent: int = 0) -> str:
        return indent * INDENTATION + 'UNSUPPORTED'

    def preorder_traversal(self) -> Iterator['ScheduleTreeNode']:
        """
        Traverse tree nodes in a pre-order manner.
        """
        yield self

    def get_root(self) -> 'ScheduleTreeRoot':
        if self.parent is None:
            raise ValueError('Non-root schedule tree node has no parent.')
        return self.parent.get_root()

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs: dict[str, Any]) -> MemletSet:
        """
        Returns a set of inputs for this node. For scopes, returns the union of its contents.

        :param root: An optional argument specifying the schedule tree's root. If not given,
                     the value is computed from the current tree node.
        :return: A set of memlets representing the inputs of this node.
        """
        raise NotImplementedError

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs: dict[str, Any]) -> MemletSet:
        """
        Returns a set of outputs for this node. For scopes, returns the union of its contents.

        :param root: An optional argument specifying the schedule tree's root. If not given,
                     the value is computed from the current tree node.
        :return: A set of memlets representing the inputs of this node.
        """
        raise NotImplementedError


@dataclass
class ScheduleTreeScope(ScheduleTreeNode):
    """
    A `ScheduleTreeScope` is the base class for grouping `ScheduleTreeNode`s hierarchically.

    Each scope holds a list of children and a reference to the parent.
    """
    children: list[ScheduleTreeNode]

    def __init__(self, *, children: list[ScheduleTreeNode], parent: Optional['ScheduleTreeScope'] = None) -> None:
        for child in children:
            child.parent = self

        self.children = children
        self.parent = parent

    def add_children(self, children: Iterable[ScheduleTreeNode]) -> None:
        for child in children:
            child.parent = self
            self.children.append(child)

    def add_child(self, child: ScheduleTreeNode) -> None:
        self.add_children([child])

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

    def _gather_memlets_in_scope(self, inputs: bool, root: Optional['ScheduleTreeRoot'], keep_locals: bool,
                                 propagate: dict[str,
                                                 subsets.Range], disallow_propagation: set[str], **kwargs) -> MemletSet:
        gather = (lambda n, root: n.input_memlets(root, **kwargs)) if inputs else (
            lambda n, root: n.output_memlets(root, **kwargs))

        # Fast path, no propagation necessary
        if keep_locals:
            return MemletSet().union(*(gather(c, root) for c in self.children))

        root = root if root is not None else self.get_root()

        if propagate:
            to_propagate = list(propagate.items())
            propagate_keys = [a[0] for a in to_propagate]
            propagate_values = subsets.Range([a[1] for a in to_propagate])

        current_locals = set()
        current_locals |= disallow_propagation
        result = MemletSet()

        # Loop over children in order, if any new symbol is defined within this scope (e.g., symbol assignment,
        # dynamic map range), consider it as a new local
        for c in self.children:
            # Add new locals
            if isinstance(c, AssignNode):
                current_locals.add(c.name)
            elif isinstance(c, DynScopeCopyNode):
                current_locals.add(c.target)

            internal_memlets: MemletSet = gather(c, root)
            if propagate:
                for memlet in internal_memlets:
                    result.add(
                        propagate_subset([memlet],
                                         root.containers[memlet.data],
                                         propagate_keys,
                                         propagate_values,
                                         undefined_variables=current_locals,
                                         use_dst=not inputs))

        return result

    def input_memlets(self,
                      root: Optional['ScheduleTreeRoot'] = None,
                      keep_locals: bool = False,
                      propagate: dict[str, subsets.Range] | None = None,
                      disallow_propagation: set[str] | None = None,
                      **kwargs) -> MemletSet:
        """
        Returns a union of the set of inputs for this scope. Propagates the memlets used in the scope if ``keep_locals``
        is set to False.

        :param root: An optional argument specifying the schedule tree's root. If not given,
                     the value is computed from the current tree node.
        :param keep_locals: If True, keeps the local symbols defined within the scope as part of the resulting memlets.
                            Otherwise, performs memlet propagation (see ``propagate`` and ``disallow_propagation``) or
                            assumes the entire container is used.
        :param propagate: An optional dictionary mapping symbols to their corresponding ranges outside of this scope.
                          For example, the range of values a for-loop may take.
                          If ``keep_locals`` is False, this dictionary will be used to create projection memlets over
                          the ranges. See :ref:`memprop` in the documentation for more information.
        :param disallow_propagation: If ``keep_locals`` is False, this optional set of strings will be considered
                                     as additional locals.
        :return: A set of memlets representing the inputs of this scope.
        """
        return self._gather_memlets_in_scope(True, root, keep_locals, propagate or {}, disallow_propagation or set(),
                                             **kwargs)

    def output_memlets(self,
                       root: Optional['ScheduleTreeRoot'] = None,
                       keep_locals: bool = False,
                       propagate: dict[str, subsets.Range] | None = None,
                       disallow_propagation: set[str] | None = None,
                       **kwargs) -> MemletSet:
        """
        Returns a union of the set of outputs for this scope. Propagates the memlets used in the scope if
        ``keep_locals`` is set to False.

        :param root: An optional argument specifying the schedule tree's root. If not given,
                     the value is computed from the current tree node.
        :param keep_locals: If True, keeps the local symbols defined within the scope as part of the resulting memlets.
                            Otherwise, performs memlet propagation (see ``propagate`` and ``disallow_propagation``) or
                            assumes the entire container is used.
        :param propagate: An optional dictionary mapping symbols to their corresponding ranges outside of this scope.
                          For example, the range of values a for-loop may take.
                          If ``keep_locals`` is False, this dictionary will be used to create projection memlets over
                          the ranges. See :ref:`memprop` in the documentation for more information.
        :param disallow_propagation: If ``keep_locals`` is False, this optional set of strings will be considered
                                     as additional locals.
        :return: A set of memlets representing the inputs of this scope.
        """
        return self._gather_memlets_in_scope(False, root, keep_locals, propagate or {}, disallow_propagation or set(),
                                             **kwargs)


@dataclass
class ScheduleTreeRoot(ScheduleTreeScope):
    """
    The root of a schedule tree. This is a ``ScheduleTreeScope`` with additional information on
    the available descriptors, symbol types, and constants of the tree, aka the descriptor repository.

    Each schedule tree has only one ``ScheduleTreeRoot``. The ``ScheduleTreeRoot`` is the only ``ScheduleTreeScope``
    without a parent (because it is the root node of the tree).
    """
    name: str
    containers: dict[str, data.Data]
    symbols: Mapping[str, dtypes.typeclass | symbolic.symbol]
    constants: dict[str, tuple[data.Data, Any]]
    callback_mapping: dict[str, str]
    arg_names: list[str]

    def __init__(
        self,
        *,
        name: str,
        children: list[ScheduleTreeNode],
        containers: dict[str, data.Data] | None = None,
        symbols: Mapping[str, dtypes.typeclass | symbolic.symbol] | None = None,
        constants: dict[str, tuple[data.Data, Any]] | None = None,
        callback_mapping: dict[str, str] | None = None,
        arg_names: list[str] | None = None,
    ) -> None:
        super().__init__(children=children, parent=None)

        self.name = name
        self.containers = containers if containers is not None else dict()
        self.symbols = symbols if symbols is not None else dict()
        self.constants = constants if constants is not None else dict()
        self.callback_mapping = callback_mapping if callback_mapping is not None else dict()
        self.arg_names = arg_names if arg_names is not None else list()

    def as_sdfg(self,
                validate: bool = True,
                simplify: bool = True,
                validate_all: bool = False,
                skip: set[str] | None = None,
                verbose: bool = False) -> SDFG:
        """
        Convert this schedule tree representation (back) into an SDFG.

        :param validate: If true, validate generated SDFG.
        :param simplify: If true, simplify generated SDFG. The conversion might insert things like extra
                         empty states that can be cleaned up automatically. The value of `validate` is
                         passed on to `simplify()`.
        :param validate_all: When simplifying, validate all intermediate SDFGs. Unused if simplify is False.
        :param skip: Set of names of simplify passes to skip. Unused if simplify is False.
        :param verbose: Turn on verbose logging of simplify. Unused if simplify is False.

        :return: SDFG version of this schedule tree.
        """
        from dace.sdfg.analysis.schedule_tree import tree_to_sdfg as t2s  # Avoid import loop
        sdfg = t2s.from_schedule_tree(self)

        if validate:
            sdfg.validate()

        if simplify:
            skip = set() if skip is None else skip
            sdfg.simplify(validate=validate, validate_all=validate_all, verbose=verbose, skip=skip)

        return sdfg

    def get_root(self) -> 'ScheduleTreeRoot':
        return self


@dataclass
class ControlFlowScope(ScheduleTreeScope):

    def __init__(self, *, children: list[ScheduleTreeNode], parent: ScheduleTreeScope | None = None) -> None:
        super().__init__(children=children, parent=parent)


@dataclass
class FunctionCallScope(ControlFlowScope):
    """
    Represents a call to another ``@dace.program`` whose schedule tree body
    is inlined as children of this scope.

    A :class:`ReturnNode` inside this scope exits the scope (the inlined
    callee), not the surrounding program.
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

    def input_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        root = root if root is not None else self.get_root()
        return MemletSet(
            Memlet.from_array(name, root.containers[name]) for name in self.call.arguments.values()
            if name in root.containers)

    def output_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        # Without callee dataflow analysis, arguments are conservatively
        # treated as potentially written; return targets always are.
        root = root if root is not None else self.get_root()
        written = list(self.call.arguments.values()) + list(self.return_targets)
        return MemletSet(Memlet.from_array(name, root.containers[name]) for name in written if name in root.containers)


@dataclass
class DataflowScope(ScheduleTreeScope):
    node: Union[nodes.EntryNode, FrontendMap, FrontendConsume]
    state: Optional[SDFGState] = None

    def __init__(self,
                 *,
                 node: nodes.EntryNode,
                 children: list[ScheduleTreeNode],
                 parent: ScheduleTreeScope | None = None,
                 state: SDFGState | None = None) -> None:
        super().__init__(children=children, parent=parent)

        self.node = node
        self.state = state


@dataclass
class GBlock(ControlFlowScope):
    """
    General control flow block. Contains a list of states
    that can run in arbitrary order based on edges (gotos).
    Normally contains irreducible control flow.
    """

    def __init__(self, *, children: list[ScheduleTreeNode], parent: ScheduleTreeScope | None = None) -> None:
        super().__init__(children=children, parent=parent)

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

    def input_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        return MemletSet()

    def output_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        return MemletSet()


@dataclass
class GotoNode(ScheduleTreeNode):
    target: str | None = None  #: If None, equivalent to "goto exit" or "return"

    def as_string(self, indent: int = 0):
        name = self.target or 'exit'
        return indent * INDENTATION + f'goto {name}'

    def input_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        return MemletSet()

    def output_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        return MemletSet()


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

    def input_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        root = root if root is not None else self.get_root()
        return MemletSet(self.edge.get_read_memlets(root.containers))

    def output_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        return MemletSet()


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

    The node is a side-effect fence: transformations must not reorder memory
    accesses or other callbacks across it.
    """
    code: CodeBlock
    reason: str
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    outlined_function_name: Optional[str] = None
    outlined_function_code: Optional[CodeBlock] = None
    outlined_call_code: Optional[CodeBlock] = None

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + f'python_callback "{self.reason}" {{\n' + '\n'.join(
            (indent + 1) * INDENTATION + line
            for line in self.code.as_string.splitlines()) + f'\n{indent * INDENTATION}}}'

    def input_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        root = root if root is not None else self.get_root()
        return MemletSet(
            Memlet.from_array(name, root.containers[name]) for name in self.input_names if name in root.containers)

    def output_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        root = root if root is not None else self.get_root()
        return MemletSet(
            Memlet.from_array(name, root.containers[name]) for name in self.output_names if name in root.containers)


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
    values: List[str] = field(default_factory=list)
    """
    If non-empty, represents the return value(s) of this return statement as a list of data descriptor names.
    """

    def as_string(self, indent: int = 0):
        if not self.values:
            return indent * INDENTATION + 'return'
        joined = ', '.join(self.values)
        return indent * INDENTATION + f'return {joined}'

    def input_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        root = root if root is not None else self.get_root()
        return MemletSet(
            Memlet.from_array(name, root.containers[name]) for name in self.values if name in root.containers)

    def output_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        return MemletSet()


@dataclass
class LoopScope(ControlFlowScope):
    """
    General loop scope (representing a loop region).
    """
    loop: Union[LoopRegion, FrontendLoop]

    def __init__(self,
                 *,
                 loop: LoopRegion,
                 children: list[ScheduleTreeNode],
                 parent: ScheduleTreeScope | None = None) -> None:
        super().__init__(children=children, parent=parent)

        self.loop = loop

    def as_string(self, indent: int = 0):
        loop = self.loop
        variant = loop_variant(loop)
        if variant == 'do-for-uncond-increment':
            pre_header = indent * INDENTATION + f'{loop.init_statement.as_string}\n'
            header = indent * INDENTATION + 'do:\n'
            pre_footer = (indent + 1) * INDENTATION + f'{loop.update_statement.as_string}\n'
            footer = indent * INDENTATION + f'while {loop.loop_condition.as_string}'
            return pre_header + header + super().as_string(indent) + '\n' + pre_footer + footer

        if variant == 'do-for':
            pre_header = indent * INDENTATION + f'{loop.init_statement.as_string}\n'
            header = indent * INDENTATION + 'while True:\n'
            pre_footer = (indent + 1) * INDENTATION + f'if (not {loop.loop_condition.as_string}):\n'
            pre_footer += (indent + 2) * INDENTATION + 'break\n'
            footer = (indent + 1) * INDENTATION + f'{loop.update_statement.as_string}\n'
            return pre_header + header + super().as_string(indent) + '\n' + pre_footer + footer

        if variant in ["for", "while", "do-while"]:
            return super().as_string(indent)

        raise NotImplementedError(f"Unknown LoopRegion variant '{variant}.")


@dataclass
class ForScope(LoopScope):
    """Specialized LoopScope for for-loops."""

    def __init__(self,
                 *,
                 loop: LoopRegion,
                 children: list[ScheduleTreeNode],
                 parent: ScheduleTreeScope | None = None) -> None:
        super().__init__(loop=loop, children=children, parent=parent)

    def as_string(self, indent: int = 0) -> str:
        init_statement = self.loop.init_statement.as_string
        condition = self.loop.loop_condition.as_string
        update_statement = self.loop.update_statement.as_string
        result = indent * INDENTATION + f"for {init_statement}; {condition}; {update_statement}:\n"
        return result + super().as_string(indent)

    def input_memlets(self,
                      root: ScheduleTreeRoot | None = None,
                      keep_locals: bool = False,
                      propagate: dict[str, subsets.Range] | None = None,
                      disallow_propagation: set[str] | None = None,
                      **kwargs) -> MemletSet:
        root = root if root is not None else self.get_root()

        result = MemletSet()
        result.update(self.loop.get_meta_read_memlets(arrays=root.containers))

        # If loop range is well-formed, use it in propagation
        range = _loop_range(self.loop)
        if range is not None:
            propagate = {self.loop.loop_variable: range}
        else:
            propagate = None

        result.update(super().input_memlets(root, propagate=propagate, **kwargs))
        return result

    def output_memlets(self,
                       root: ScheduleTreeRoot | None = None,
                       keep_locals: bool = False,
                       propagate: dict[str, subsets.Range] | None = None,
                       disallow_propagation: set[str] | None = None,
                       **kwargs) -> MemletSet:

        # If loop range is well-formed, use it in propagation
        range = _loop_range(self.loop)
        if range is not None:
            propagate = {self.loop.loop_variable: range}
        else:
            propagate = None

        return super().output_memlets(root, propagate=propagate, **kwargs)


@dataclass
class WhileScope(LoopScope):
    """Specialized LoopScope for while-loops."""

    def __init__(self,
                 *,
                 loop: LoopRegion,
                 children: list[ScheduleTreeNode],
                 parent: ScheduleTreeScope | None = None) -> None:
        super().__init__(loop=loop, children=children, parent=parent)

    def as_string(self, indent: int = 0) -> str:
        condition = self.loop.loop_condition.as_string
        result = indent * INDENTATION + f'while {condition}:\n'
        return result + super().as_string(indent)

    def input_memlets(self,
                      root: ScheduleTreeRoot | None = None,
                      keep_locals: bool = False,
                      propagate: dict[str, subsets.Range] | None = None,
                      disallow_propagation: set[str] | None = None,
                      **kwargs) -> MemletSet:
        root = root if root is not None else self.get_root()

        result = MemletSet()
        result.update(self.loop.get_meta_read_memlets(arrays=root.containers))
        result.update(super().input_memlets(root, **kwargs))
        return result


@dataclass
class DoWhileScope(LoopScope):
    """Specialized LoopScope for do-while-loops"""

    def __init__(self,
                 *,
                 loop: LoopRegion,
                 children: list[ScheduleTreeNode],
                 parent: ScheduleTreeScope | None = None) -> None:
        super().__init__(loop=loop, children=children, parent=parent)

    def as_string(self, indent: int = 0) -> str:
        header = indent * INDENTATION + 'do:\n'
        footer = indent * INDENTATION + f'while {self.loop.loop_condition.as_string}'
        return header + super().as_string(indent) + '\n' + footer

    def input_memlets(self,
                      root: ScheduleTreeRoot | None = None,
                      keep_locals: bool = False,
                      propagate: dict[str, subsets.Range] | None = None,
                      disallow_propagation: set[str] | None = None,
                      **kwargs) -> MemletSet:
        root = root if root is not None else self.get_root()

        result = MemletSet()
        result.update(self.loop.get_meta_read_memlets(arrays=root.containers))
        result.update(super().input_memlets(root, **kwargs))
        return result


@dataclass
class IfScope(ControlFlowScope):
    """
    If branch scope.
    """
    condition: CodeBlock

    def __init__(self,
                 *,
                 condition: CodeBlock,
                 children: list[ScheduleTreeNode],
                 parent: ScheduleTreeScope | None = None) -> None:
        super().__init__(children=children, parent=parent)

        self.condition = condition

    def as_string(self, indent: int = 0):
        result = indent * INDENTATION + f'if {self.condition.as_string}:\n'
        return result + super().as_string(indent)

    def input_memlets(self,
                      root: ScheduleTreeRoot | None = None,
                      keep_locals: bool = False,
                      propagate: dict[str, subsets.Range] | None = None,
                      disallow_propagation: set[str] | None = None,
                      **kwargs) -> MemletSet:
        root = root if root is not None else self.get_root()
        result = MemletSet()
        result.update(memlets_in_ast(self.condition.code[0], root.containers))
        result.update(super().input_memlets(root, **kwargs))
        return result


@dataclass
class StateIfScope(IfScope):
    """
    A special class of an if scope in general blocks for if statements that are part of a state transition.
    """

    def __init__(self,
                 *,
                 condition: CodeBlock,
                 children: list[ScheduleTreeNode],
                 parent: ScheduleTreeScope | None = None) -> None:
        super().__init__(condition=condition, children=children, parent=parent)

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

    def input_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        return MemletSet()

    def output_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        return MemletSet()


@dataclass
class ContinueNode(ScheduleTreeNode):
    """
    Represents a continue statement.
    """

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + 'continue'

    def input_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        return MemletSet()

    def output_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        return MemletSet()


@dataclass
class ElifScope(ControlFlowScope):
    """
    Else-if branch scope.
    """
    condition: CodeBlock

    def __init__(self,
                 *,
                 condition: CodeBlock,
                 children: list[ScheduleTreeNode],
                 parent: ScheduleTreeScope | None = None) -> None:
        super().__init__(children=children, parent=parent)

        self.condition = condition

    def as_string(self, indent: int = 0):
        result = indent * INDENTATION + f'elif {self.condition.as_string}:\n'
        return result + super().as_string(indent)

    def input_memlets(self,
                      root: ScheduleTreeRoot | None = None,
                      keep_locals: bool = False,
                      propagate: dict[str, subsets.Range] | None = None,
                      disallow_propagation: set[str] | None = None,
                      **kwargs) -> MemletSet:
        root = root if root is not None else self.get_root()
        result = MemletSet()
        result.update(memlets_in_ast(self.condition.code[0], root.containers))
        result.update(super().input_memlets(root, **kwargs))
        return result


@dataclass
class ElseScope(ControlFlowScope):
    """
    Else branch scope.
    """

    def __init__(self, *, children: list[ScheduleTreeNode], parent: ScheduleTreeScope | None = None) -> None:
        super().__init__(children=children, parent=parent)

    def as_string(self, indent: int = 0):
        result = indent * INDENTATION + 'else:\n'
        return result + super().as_string(indent)


@dataclass
class MapScope(DataflowScope):
    """
    Map scope.
    """
    node: nodes.MapEntry

    def __init__(self,
                 *,
                 node: nodes.MapEntry,
                 children: list[ScheduleTreeNode],
                 parent: ScheduleTreeScope | None = None,
                 state: SDFGState | None = None) -> None:
        super().__init__(node=node, state=state, children=children, parent=parent)

    def as_string(self, indent: int = 0):
        if isinstance(self.node, FrontendMap):
            rangestr = ', '.join(_format_frontend_range(start, stop, step) for start, stop, step in self.node.ranges)
            params = ', '.join(self.node.params)
        else:
            rangestr = ', '.join(subsets.Range.dim_to_string(d) for d in self.node.map.range)
            params = ', '.join(self.node.map.params)
        result = indent * INDENTATION + f'map {params} in [{rangestr}]:\n'
        return result + super().as_string(indent)

    def input_memlets(self,
                      root: ScheduleTreeRoot | None = None,
                      keep_locals: bool = False,
                      propagate: dict[str, subsets.Range] | None = None,
                      disallow_propagation: set[str] | None = None,
                      **kwargs) -> MemletSet:
        return super().input_memlets(root,
                                     propagate={
                                         k: v
                                         for k, v in zip(self.node.map.params, self.node.map.range)
                                     },
                                     **kwargs)

    def output_memlets(self,
                       root: ScheduleTreeRoot | None = None,
                       keep_locals: bool = False,
                       propagate: dict[str, subsets.Range] | None = None,
                       disallow_propagation: set[str] | None = None,
                       **kwargs) -> MemletSet:
        return super().output_memlets(root,
                                      propagate={
                                          k: v
                                          for k, v in zip(self.node.map.params, self.node.map.range)
                                      },
                                      **kwargs)


@dataclass
class ConsumeScope(DataflowScope):
    """
    Consume scope.
    """
    node: nodes.ConsumeEntry

    def __init__(self,
                 *,
                 node: nodes.ConsumeEntry,
                 children: list[ScheduleTreeNode],
                 parent: ScheduleTreeScope | None = None,
                 state: SDFGState | None = None) -> None:
        super().__init__(node=node, state=state, children=children, parent=parent)

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

    def input_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        return MemletSet(self.in_memlets.values())

    def output_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        return MemletSet(self.out_memlets.values())


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

    def input_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        if isinstance(self.in_memlets, set):
            return MemletSet(self.in_memlets)
        return MemletSet(self.in_memlets.values())

    def output_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        if isinstance(self.out_memlets, set):
            return MemletSet(self.out_memlets)
        return MemletSet(self.out_memlets.values())


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

    def input_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        return MemletSet({self.memlet})

    def output_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        root = root if root is not None else self.get_root()
        if self.memlet.other_subset is not None:
            return MemletSet({Memlet(data=self.target, subset=self.memlet.other_subset, wcr=self.memlet.wcr)})

        return MemletSet({Memlet.from_array(self.target, root.containers[self.target], self.memlet.wcr)})


@dataclass
class DynScopeCopyNode(ScheduleTreeNode):
    """
    A special case of a copy node that is used in dynamic scope inputs (e.g., dynamic map ranges).
    """
    target: str
    memlet: Memlet

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + f'{self.target} = dscopy {self.memlet.data}[{self.memlet.subset}]'

    def input_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        return MemletSet({self.memlet})

    def output_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        return MemletSet()


@dataclass
class ViewNode(ScheduleTreeNode):
    target: str  #: View name
    source: str  #: Viewed container name
    memlet: Memlet
    src_desc: data.Data
    view_desc: data.Data

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + f'{self.target} = view {self.memlet} as {self.view_desc.shape}'

    def input_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        return MemletSet({self.memlet})

    def output_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        return MemletSet({Memlet.from_array(self.target, self.view_desc)})


@dataclass
class NView(ViewNode):
    """
    Nested SDFG view node. Subclass of a view that specializes in nested SDFG boundaries.
    """

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + f'{self.target} = nview {self.memlet} as {self.view_desc.shape}'


@dataclass
class NViewEnd(ScheduleTreeNode):
    """
    Artificial node to denote the scope end of the associated Nested SDFG view node.
    """

    target: str
    """Target name of the associated NView container."""

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + f"end nview {self.target}"

    def input_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        return MemletSet()

    def output_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        return MemletSet()


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

    def input_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        return MemletSet({self.memlet})

    def output_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        return MemletSet({Memlet.from_array(self.target, self.ref_desc)})


@dataclass
class ReplacementCallNode(ScheduleTreeNode):
    """
    A deferred call to a function-replacement from the frontend replacement
    registry (:class:`dace.frontend.common.op_repository.Replacements`).

    The frontend records the registry-qualified name and the resolved
    arguments; lowering to an SDFG invokes the registered replacement on the
    target state and copies its result into ``target``. This lets schedule-tree
    frontends reuse the classic replacement implementations without
    reimplementing each call as tree emission.

    :param qualname: Replacement registry key (e.g. ``numpy.sum``). For a
                     bound-method call (``receiver`` set), this is the bare
                     method name (e.g. ``copy``), looked up through
                     ``Replacements.get_method`` against the receiver's
                     descriptor type rather than through ``Replacements.get``.
    :param target: Repository container the call result is written to.
    :param arguments: Positional arguments; entries listed in
                      ``data_arguments`` are repository container names, all
                      other entries are compile-time Python values. For a
                      bound-method call, the receiver is already included as
                      the first entry (matching the classic frontend's
                      convention of prepending ``self`` to the replacement's
                      argument list).
    :param keyword_arguments: Keyword arguments, same convention.
    :param data_arguments: The container-name argument values (includes the
                           receiver, when set).
    :param receiver: For a bound-method call (``obj.method(...)``), the
                     repository container ``obj`` resolves to. ``None`` for a
                     free-function replacement.
    """
    qualname: str = ''
    target: str = ''
    arguments: List[Any] = field(default_factory=list)
    keyword_arguments: Dict[str, Any] = field(default_factory=dict)
    data_arguments: Set[str] = field(default_factory=set)
    receiver: Optional[str] = None

    def as_string(self, indent: int = 0):
        rendered = [str(argument) for argument in self.arguments]
        rendered += [f'{name}={value}' for name, value in self.keyword_arguments.items()]
        return indent * INDENTATION + f'{self.target} = replacement {self.qualname}({", ".join(rendered)})'

    def input_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        root = root if root is not None else self.get_root()
        return MemletSet(
            Memlet.from_array(name, root.containers[name]) for name in self.data_arguments if name in root.containers)

    def output_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        root = root if root is not None else self.get_root()
        if self.target not in root.containers:
            return MemletSet()
        return MemletSet({Memlet.from_array(self.target, root.containers[self.target])})


@dataclass
class StateBoundaryNode(ScheduleTreeNode):
    """
    A node that represents a state boundary (e.g., when a write-after-write is encountered). This node
    is used only during conversion from a schedule tree to an SDFG.
    """
    due_to_control_flow: bool = False

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + 'state boundary'

    def input_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        return MemletSet()

    def output_memlets(self, root: ScheduleTreeRoot | None = None, **kwargs) -> MemletSet:
        return MemletSet()


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

    def visit(self, node: ScheduleTreeNode | list[ScheduleTreeNode], **kwargs: Any):
        """Visit a node."""
        if isinstance(node, list):
            return [self.visit(snode, **kwargs) for snode in node]

        if isinstance(node, ScheduleTreeScope) and hasattr(self, 'visit_scope'):
            return self.visit_scope(node, **kwargs)  # type: ignore

        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, **kwargs)

    def generic_visit(self, node: ScheduleTreeNode, **kwargs: Any):
        if isinstance(node, ScheduleTreeScope):
            for child in node.children:
                self.visit(child, **kwargs)


class ScheduleNodeTransformer(ScheduleNodeVisitor):

    def visit(self, node: ScheduleTreeNode | list[ScheduleTreeNode], **kwargs: Any):
        if isinstance(node, list):
            result = []
            for snode in node:
                new_node = self.visit(snode, **kwargs)
                if new_node is not None:
                    result.append(new_node)
            return result

        return super().visit(node, **kwargs)

    def generic_visit(self, node: ScheduleTreeNode, **kwargs: Any):
        new_values = []
        if isinstance(node, ScheduleTreeScope):
            for value in node.children:
                if isinstance(value, ScheduleTreeNode):
                    value = self.visit(value, **kwargs)
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


def validate_children_and_parents_align(stree: ScheduleTreeScope, *, root: bool = False) -> None:
    """
    Validates the child/parent information of schedule tree scopes are consistent.

    Walks through all children of a scope and raises if the children's parent isn't
    the scope. If `root` is true, we additionally check that the top-most scope is
    of type `ScheduleTreeRoot`.

    :param stree: Schedule tree scope to be analyzed
    :param root: If true, we raise if the top-most scope isn't of type `ScheduleTreeRoot`.
    """
    if root and not isinstance(stree, ScheduleTreeRoot):
        raise RuntimeError("Expected schedule tree root.")

    for child in stree.children:
        if child.parent is not stree:
            raise RuntimeError(f"Inconsistent parent/child relationship. child: {child}, parent: {stree}")
        if isinstance(child, ScheduleTreeScope):
            validate_children_and_parents_align(child)


def loop_variant(
    loop: LoopRegion
) -> Literal['for'] | Literal['while'] | Literal['do-while'] | Literal['do-for-uncond-increment'] | Literal['do-for']:
    if loop.update_statement and loop.init_statement and loop.loop_variable:
        if loop.inverted:
            if loop.update_before_condition:
                return 'do-for-uncond-increment'
            return 'do-for'
        return 'for'

    if loop.inverted:
        return 'do-while'
    return 'while'


def _loop_range(loop: LoopRegion) -> tuple[symbolic.SymbolicType, symbolic.SymbolicType, symbolic.SymbolicType] | None:
    """
    Derive loop range for well-formed `for`-loops.

    :param: loop The loop to be analyzed.
    :return: If well formed, `(start, end, step)` where `end` is inclusive, otherwise `None`.
    """

    if loop_variant(loop) != "for" or loop.loop_variable is None:
        # Loop range is only defined in for-loops
        # and we need to know the loop variable.
        return None

    # Avoid cyclic import
    from dace.transformation.passes.analysis import loop_analysis

    # If loop information cannot be determined, we cannot derive loop range
    start = loop_analysis.get_init_assignment(loop)
    step = loop_analysis.get_loop_stride(loop)
    end = _match_loop_condition(loop)
    if start is None or step is None or end is None:
        return None

    return (start, end, step)


def _match_loop_condition(loop: LoopRegion) -> symbolic.SymbolicType | None:
    """
    Try to find the end of a for-loop by symbolically matching the loop condition.

    :return: loop end (inclusive) or `None` if matching failed.
    """

    condition = symbolic.pystr_to_symbolic(loop.loop_condition.as_string)
    loop_symbol = symbolic.pystr_to_symbolic(loop.loop_variable)
    a = sympy.Wild('a')

    match = condition.match(loop_symbol < a)
    if match is not None:
        return match[a] - 1

    match = condition.match(loop_symbol <= a)
    if match is not None:
        return match[a]

    match = condition.match(loop_symbol >= a)
    if match is not None:
        return match[a]

    match = condition.match(loop_symbol > a)
    if match is not None:
        return match[a] + 1

    # Matching failed - we can't derive end of loop
    return None
