# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dataclasses import dataclass, field
import sympy

from dace import nodes, data, subsets, dtypes, symbolic
from dace.properties import CodeBlock
from dace.sdfg import InterstateEdge
from dace.sdfg.memlet_utils import MemletSet
from dace.sdfg.propagation import propagate_subset
from dace.sdfg.sdfg import InterstateEdge, SDFG, memlets_in_ast
from dace.sdfg.state import LoopRegion, SDFGState
from dace.memlet import Memlet
from types import TracebackType
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Literal, Optional, Set, Tuple, Union

if TYPE_CHECKING:
    from dace import SDFG

INDENTATION = '  '


class UnsupportedScopeException(Exception):
    pass


@dataclass
class Context:
    root: 'ScheduleTreeRoot'
    current_scope: Optional['ScheduleTreeScope']

    access_cache: Dict[Tuple[SDFGState, str], Dict[str, nodes.AccessNode]]
    """Per scope (hashed by id(scope_node) access_cache."""


class ContextPushPop:
    """Append the given node to the scope, then push/pop the scope."""

    def __init__(self, ctx: Context, state: SDFGState, node: 'ScheduleTreeScope') -> None:
        if ctx.current_scope is None and not isinstance(node, ScheduleTreeRoot):
            raise ValueError("ctx.current_scope is only allowed to be 'None' when node it tree root.")

        self._ctx = ctx
        self._parent_scope = ctx.current_scope
        self._node = node
        self._state = state

        cache_key = (state, id(node))
        assert cache_key not in self._ctx.access_cache
        self._ctx.access_cache[cache_key] = {}

    def __enter__(self) -> None:
        assert not self._ctx.access_cache[(self._state, id(
            self._node))], "Expecting an empty access_cache when entering the context."
        # self._node.parent = self._parent_scope
        # if self._parent_scope is not None: # Exception for ScheduleTreeRoot
        #     self._parent_scope.children.append(self._node)
        self._ctx.current_scope = self._node

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        cache_key = (self._state, id(self._node))
        assert cache_key in self._ctx.access_cache
        # self._ctx.access_cache[cache_key].clear()

        self._ctx.current_scope = self._parent_scope


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

    def get_root(self) -> 'ScheduleTreeRoot':
        if self.parent is None:
            raise ValueError('Non-root schedule tree node has no parent')
        return self.parent.get_root()

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        """
        Returns a set of inputs for this node. For scopes, returns the union of its contents.

        :param root: An optional argument specifying the schedule tree's root. If not given,
                     the value is computed from the current tree node.
        :return: A set of memlets representing the inputs of this node.
        """
        raise NotImplementedError

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        """
        Returns a set of outputs for this node. For scopes, returns the union of its contents.

        :param root: An optional argument specifying the schedule tree's root. If not given,
                     the value is computed from the current tree node.
        :return: A set of memlets representing the inputs of this node.
        """
        raise NotImplementedError


@dataclass
class ScheduleTreeScope(ScheduleTreeNode):
    children: List[ScheduleTreeNode]

    def __init__(self, *, children: List[ScheduleTreeNode], parent: Optional['ScheduleTreeScope'] = None) -> None:
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
                                 propagate: Dict[str,
                                                 subsets.Range], disallow_propagation: Set[str], **kwargs) -> MemletSet:
        gather = (lambda n, root: n.input_memlets(root, **kwargs)) if inputs else (
            lambda n, root: n.output_memlets(root, **kwargs))

        # Fast path, no propagation necessary
        if keep_locals:
            return MemletSet().union(*(gather(c) for c in self.children))

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
                      propagate: Optional[Dict[str, subsets.Range]] = None,
                      disallow_propagation: Optional[Set[str]] = None,
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
                       propagate: Optional[Dict[str, subsets.Range]] = None,
                       disallow_propagation: Optional[Set[str]] = None,
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
                skip: Set[str] = set(),
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
            from dace.transformation.passes.simplify import SimplifyPass
            SimplifyPass(validate=validate, validate_all=validate_all, skip=skip, verbose=verbose).apply_pass(sdfg, {})

        return sdfg

    def get_root(self) -> 'ScheduleTreeRoot':
        return self

    def scope(self, state: SDFGState, ctx: Context) -> ContextPushPop:
        return ContextPushPop(ctx, state, self)


@dataclass
class ControlFlowScope(ScheduleTreeScope):

    def __init__(self, *, children: List[ScheduleTreeNode], parent: Optional[ScheduleTreeScope] = None) -> None:
        super().__init__(children=children, parent=parent)


@dataclass
class DataflowScope(ScheduleTreeScope):
    node: nodes.EntryNode
    state: Optional[SDFGState] = None

    def __init__(self,
                 *,
                 node: nodes.EntryNode,
                 children: List[ScheduleTreeNode],
                 parent: Optional[ScheduleTreeScope] = None,
                 state: Optional[SDFGState] = None) -> None:
        super().__init__(children=children, parent=parent)

        self.node = node
        self.state = state

    def scope(self, state: SDFGState, ctx: Context) -> ContextPushPop:
        return ContextPushPop(ctx, state, self)


@dataclass
class GBlock(ControlFlowScope):
    """
    General control flow block. Contains a list of states
    that can run in arbitrary order based on edges (gotos).
    Normally contains irreducible control flow.
    """

    def __init__(self, *, children: List[ScheduleTreeNode], parent: Optional[ScheduleTreeScope] = None) -> None:
        super().__init__(children=children, parent=parent)

    def as_string(self, indent: int = 0):
        result = indent * INDENTATION + 'gblock:\n'
        return result + super().as_string(indent)


@dataclass
class StateLabel(ScheduleTreeNode):
    state: SDFGState

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + f'label {self.state.name}:'

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        return MemletSet()

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        return MemletSet()


@dataclass
class GotoNode(ScheduleTreeNode):
    target: Optional[str] = None  #: If None, equivalent to "goto exit" or "return"

    def as_string(self, indent: int = 0):
        name = self.target or 'exit'
        return indent * INDENTATION + f'goto {name}'

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        return MemletSet()

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        return MemletSet()


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

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        root = root if root is not None else self.get_root()
        return MemletSet(self.edge.get_read_memlets(root.containers))

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        return MemletSet()


@dataclass
class LoopScope(ControlFlowScope):
    """
    General loop scope (representing a loop region).
    """
    loop: LoopRegion

    def __init__(self,
                 *,
                 loop: LoopRegion,
                 children: List[ScheduleTreeNode],
                 parent: Optional[ScheduleTreeScope] = None) -> None:
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

        return NotImplementedError  # TODO: nice error message


@dataclass
class ForScope(LoopScope):
    """Specialized LoopScope for for-loops."""

    def __init__(self,
                 *,
                 loop: LoopRegion,
                 children: List[ScheduleTreeNode],
                 parent: Optional[ScheduleTreeScope] = None) -> None:
        super().__init__(loop=loop, children=children, parent=parent)

    def as_string(self, indent: int = 0) -> str:
        init_statement = self.loop.init_statement.as_string
        condition = self.loop.loop_condition.as_string
        update_statement = self.loop.update_statement.as_string
        result = indent * INDENTATION + f"for {init_statement}; {condition}; {update_statement}:\n"
        return result + super().as_string(indent)

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        result = MemletSet()
        result.update(self.loop.get_meta_read_memlets())

        # If loop range is well-formed, use it in propagation
        range = _loop_range(self.loop)
        if range is not None:
            propagate = {self.loop.loop_variable: range}
        else:
            propagate = None

        result.update(super().input_memlets(root, propagate=propagate, **kwargs))
        return result

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
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
                 children: List[ScheduleTreeNode],
                 parent: Optional[ScheduleTreeScope] = None) -> None:
        super().__init__(loop=loop, children=children, parent=parent)

    def as_string(self, indent: int = 0) -> str:
        condition = self.loop.loop_condition.as_string
        result = indent * INDENTATION + f'while {condition}:\n'
        return result + super().as_string(indent)

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        root = root if root is not None else self.get_root()
        result = MemletSet()
        result.update(self.loop.get_meta_read_memlets())
        result.update(super().input_memlets(root, **kwargs))
        return result


@dataclass
class DoWhileScope(LoopScope):
    """Specialized LoopScope for do-while-loops"""

    def __init__(self,
                 *,
                 loop: LoopRegion,
                 children: List[ScheduleTreeNode],
                 parent: Optional[ScheduleTreeScope] = None) -> None:
        super().__init__(loop=loop, children=children, parent=parent)

    def as_string(self, indent: int = 0) -> str:
        header = indent * INDENTATION + 'do:\n'
        footer = indent * INDENTATION + f'while {self.loop.loop_condition.as_string}'
        return header + super().as_string(indent) + '\n' + footer

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        root = root if root is not None else self.get_root()
        result = MemletSet()
        result.update(self.loop.get_meta_read_memlets())
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
                 children: List[ScheduleTreeNode],
                 parent: Optional[ScheduleTreeScope] = None) -> None:
        super().__init__(children=children, parent=parent)

        self.condition = condition

    def as_string(self, indent: int = 0):
        result = indent * INDENTATION + f'if {self.condition.as_string}:\n'
        return result + super().as_string(indent)

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
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
                 children: List[ScheduleTreeNode],
                 parent: Optional[ScheduleTreeScope] = None) -> None:
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

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        return MemletSet()

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        return MemletSet()


@dataclass
class ContinueNode(ScheduleTreeNode):
    """
    Represents a continue statement.
    """

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + 'continue'

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        return MemletSet()

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
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
                 children: List[ScheduleTreeNode],
                 parent: Optional[ScheduleTreeScope] = None) -> None:
        super().__init__(children=children, parent=parent)

        self.condition = condition

    def as_string(self, indent: int = 0):
        result = indent * INDENTATION + f'elif {self.condition.as_string}:\n'
        return result + super().as_string(indent)

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
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

    def __init__(self, *, children: List[ScheduleTreeNode], parent: Optional[ScheduleTreeScope] = None) -> None:
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
                 children: List[ScheduleTreeNode],
                 parent: Optional[ScheduleTreeScope] = None,
                 state: Optional[SDFGState] = None) -> None:
        super().__init__(node=node, state=state, children=children, parent=parent)

    def as_string(self, indent: int = 0):
        rangestr = ', '.join(subsets.Range.dim_to_string(d) for d in self.node.map.range)
        result = indent * INDENTATION + f'map {", ".join(self.node.map.params)} in [{rangestr}]:\n'
        return result + super().as_string(indent)

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        return super().input_memlets(root,
                                     propagate={
                                         k: v
                                         for k, v in zip(self.node.map.params, self.node.map.range)
                                     },
                                     **kwargs)

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
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
                 children: List[ScheduleTreeNode],
                 parent: Optional[ScheduleTreeScope] = None,
                 state: Optional[SDFGState] = None) -> None:
        super().__init__(node=node, state=state, children=children, parent=parent)

    def as_string(self, indent: int = 0):
        node: nodes.ConsumeEntry = self.node
        cond = 'stream not empty' if node.consume.condition is None else node.consume.condition.as_string
        result = indent * INDENTATION + f'consume (PE {node.consume.pe_index} out of {node.consume.num_pes}) while {cond}:\n'
        return result + super().as_string(indent)


# TODO: to be removed. looks like `Pipeline` nodes aren't a thing anymore
# @dataclass
# class PipelineScope(MapScope):
#     """
#     Pipeline scope.
#     """
#     node: nodes.PipelineEntry
#
#     def as_string(self, indent: int = 0):
#         rangestr = ', '.join(subsets.Range.dim_to_string(d) for d in self.node.map.range)
#         result = indent * INDENTATION + f'pipeline {", ".join(self.node.map.params)} in [{rangestr}]:\n'
#         return result + super().as_string(indent)


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

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        return MemletSet(self.in_memlets.values())

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        return MemletSet(self.out_memlets.values())


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

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        if isinstance(self.in_memlets, set):
            return MemletSet(self.in_memlets)
        return MemletSet(self.in_memlets.values())

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
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

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        return MemletSet({self.memlet})

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
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

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        return MemletSet({self.memlet})

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
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

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        return MemletSet({self.memlet})

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
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

    target: str  #: target name of the associated NView container

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + f"end nview {self.target}"

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        return MemletSet()

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        return MemletSet()


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

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        return MemletSet({self.memlet})

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        return MemletSet({Memlet.from_array(self.target, self.ref_desc)})


@dataclass
class StateBoundaryNode(ScheduleTreeNode):
    """
    A node that represents a state boundary (e.g., when a write-after-write is encountered). This node
    is used only during conversion from a schedule tree to an SDFG.
    """
    due_to_control_flow: bool = False

    def as_string(self, indent: int = 0):
        return indent * INDENTATION + 'state boundary'

    def input_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        return MemletSet()

    def output_memlets(self, root: Optional['ScheduleTreeRoot'] = None, **kwargs) -> MemletSet:
        return MemletSet()


# Classes based on Python's AST NodeVisitor/NodeTransformer for schedule tree nodes
class ScheduleNodeVisitor:

    def visit(self, node: ScheduleTreeNode, **kwargs: Any):
        """Visit a node."""
        if isinstance(node, list):
            return [self.visit(snode, **kwargs) for snode in node]
        if isinstance(node, ScheduleTreeScope) and hasattr(self, 'visit_scope'):
            return self.visit_scope(node, **kwargs)

        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, **kwargs)

    def generic_visit(self, node: ScheduleTreeNode, **kwargs: Any):
        if isinstance(node, ScheduleTreeScope):
            for child in node.children:
                self.visit(child, **kwargs)


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
        if id(child.parent) != id(stree):
            raise RuntimeError(f"Inconsistent parent/child relationship. child: {child}, parent: {stree}")
        if isinstance(child, ScheduleTreeScope):
            validate_children_and_parents_align(child)


def loop_variant(
    loop: LoopRegion
) -> Union[Literal['for'], Literal['while'], Literal['do-while'], Literal['do-for-uncond-increment'],
           Literal['do-for']]:
    if loop.update_statement and loop.init_statement and loop.loop_variable:
        if loop.inverted:
            if loop.update_before_condition:
                return 'do-for-uncond-increment'
            return 'do-for'
        return 'for'

    if loop.inverted:
        return 'do-while'
    return 'while'


def _loop_range(
        loop: LoopRegion) -> Optional[Tuple[symbolic.SymbolicType, symbolic.SymbolicType, symbolic.SymbolicType]]:
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

    return (start, end, step)  # `end` is inclusive


def _match_loop_condition(loop: LoopRegion) -> Optional[symbolic.SymbolicType]:
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
