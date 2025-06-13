# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Various classes to facilitate the code generation of structured control
flow elements (e.g., ``for``, ``if``, ``while``) from state machines in SDFGs.

SDFGs are state machines of dataflow graphs, where each node is a state and each
edge may contain a state transition condition and assignments. As such, when
generating code from an SDFG, the straightforward way would be to generate code
for each state and conditional ``goto`` statements for the state transitions.
However, this inhibits compiler optimizations on the generated code, which rely
on loops and branches.

This file contains analyses that extract structured control flow constructs from
the state machine and emit code with the correct C keywords. It does so by
iteratively converting the SDFG into a control flow tree when certain control
flow patterns are detected (using the ``structured_control_flow_tree``
function). The resulting tree classes (which all extend ``ControlFlow``) contain
the original states, and upon code generation are traversed recursively into
the tree rather than in arbitrary order.

Each individual state is first wrapped with the ``SingleState`` control flow
"block", and then upon analysis can be grouped into larger control flow blocks,
such as ``ForScope`` or ``IfElseChain``. If no structured control flow pattern
is detected (or this analysis is disabled in configuration), the group of states
is wrapped in a ``GeneralBlock``, which generates the aforementioned conditional
``goto`` code.

For example, the following SDFG::


          x < 5
         /------>[s2]--------\\
    [s1] \\                   ->[s5]
          ------>[s3]->[s4]--/
          x >= 5


would create the control flow tree below::


    GeneralBlock({
        IfScope(condition=x<5, body={
            GeneralBlock({
                SingleState(s2)
            })
        }, orelse={
            GeneralBlock({
                SingleState(s3),
                SingleState(s4),
            })
        }),
        SingleState(s5)
    })


"""

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
import networkx as nx
import sympy as sp
from dace import dtypes
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.sdfg.state import (BreakBlock, ConditionalBlock, ContinueBlock, ControlFlowBlock, ControlFlowRegion,
                             LoopRegion, ReturnBlock, SDFGState)
from dace.sdfg.sdfg import SDFG, InterstateEdge
from dace.sdfg.graph import Edge
from dace.properties import CodeBlock
from dace.codegen import cppunparse
from dace.codegen.common import unparse_interstate_edge, sym2cpp

if TYPE_CHECKING:
    from dace.codegen.targets.framecode import DaCeCodeGenerator


@dataclass
class ControlFlow:
    """
    Abstract class representing a control flow block.
    """

    # A callback to the code generator that receives an SDFGState and returns a string with its generated code.
    dispatch_state: Callable[[SDFGState], str]

    # The parent control flow block of this one, used to avoid generating extraneous ``goto``s
    parent: Optional['ControlFlow']

    # Set to true if this is the last block in the parent control flow block, in order to avoid generating an
    # extraneous "goto exit" statement.
    last_block: bool

    @property
    def first_block(self) -> ControlFlowBlock:
        """
        Returns the first or initializing block in this control flow block.
        Used to determine which will be the next block in a control flow block to avoid generating extraneous
        ``goto`` calls.
        """
        return None

    @property
    def children(self) -> List['ControlFlow']:
        """
        Returns a list of control flow blocks that exist within this block.
        """
        return []

    def as_cpp(self, codegen: 'DaCeCodeGenerator', symbols: Dict[str, dtypes.typeclass]) -> str:
        """
        Returns C++ code for this control flow block.

        :param codegen: A code generator object, used for allocation information and defined variables in scope.
        :param symbols: A dictionary of symbol names and their types.
        :return: C++ string with the generated code of the control flow block.
        """
        raise NotImplementedError

    def generate_transition(self,
                            sdfg: SDFG,
                            cfg: ControlFlowRegion,
                            edge: Edge[InterstateEdge],
                            successor: Optional[ControlFlowBlock] = None,
                            assignments_only: bool = False,
                            framecode: 'DaCeCodeGenerator' = None) -> str:
        """
        Helper function that generates a state transition (conditional goto) from a control flow block and an SDFG edge.

        :param sdfg: The parent SDFG.
        :param edge: The state transition edge to generate.
        :param successor: If not None, the state that will be generated right after the current state (used to avoid
                          extraneous gotos).
        :param assignments_only: If True, generates only the assignments of the inter-state edge.
        :param framecode: Code generator object (used for allocation information).
        :return: A c++ string representing the state transition code.
        """
        expr = ''
        condition_string = unparse_interstate_edge(edge.data.condition.code[0], sdfg, codegen=framecode)

        if not edge.data.is_unconditional() and not assignments_only:
            expr += f'if ({condition_string}) {{\n'

        if len(edge.data.assignments) > 0:
            expr += ';\n'.join([
                "{} = {}".format(variable, unparse_interstate_edge(value, sdfg, codegen=framecode))
                for variable, value in edge.data.assignments.items()
            ] + [''])

        generate_goto = False
        if not edge.data.is_unconditional():
            generate_goto = True
        elif not assignments_only:
            if successor is None:
                generate_goto = True
            elif isinstance(edge.dst, SDFGState) and edge.dst is not successor:
                generate_goto = True
            elif isinstance(edge.dst, ControlFlowRegion) and edge.dst.start_block is not successor:
                generate_goto = True
        if generate_goto and not assignments_only:
            expr += 'goto __state_{}_{};\n'.format(cfg.cfg_id, edge.dst.label)

        if not edge.data.is_unconditional() and not assignments_only:
            expr += '}\n'
        return expr


@dataclass
class BasicCFBlock(ControlFlow):
    """ A CFG basic block, representing a single dataflow state """

    # The state in this element.
    state: SDFGState

    def as_cpp(self, codegen, symbols) -> str:
        cfg = self.state.parent_graph

        expr = '__state_{}_{}:;\n'.format(cfg.cfg_id, self.state.label)
        if self.state.number_of_nodes() > 0:
            expr += '{\n'
            expr += self.dispatch_state(self.state)
            expr += '\n}\n'
        else:
            # Dispatch empty state in any case in order to register that the state was dispatched.
            expr += self.dispatch_state(self.state)

        # If any state has no children, it should jump to the end of the SDFG
        if not self.last_block and cfg.out_degree(self.state) == 0:
            expr += 'goto __state_exit_{};\n'.format(cfg.cfg_id)
        return expr

    @property
    def first_block(self) -> SDFGState:
        return self.state


@dataclass
class BreakCFBlock(ControlFlow):
    """ A CFG block that generates a 'break' statement. """

    block: BreakBlock

    def as_cpp(self, codegen, symbols) -> str:
        cfg = self.block.parent_graph
        expr = '__state_{}_{}:;\n'.format(cfg.cfg_id, self.block.label)
        expr += 'break;\n'
        return expr

    @property
    def first_block(self) -> BreakBlock:
        return self.block


@dataclass
class ContinueCFBlock(ControlFlow):
    """ A CFG block that generates a 'continue' statement. """

    block: ContinueBlock

    def as_cpp(self, codegen, symbols) -> str:
        cfg = self.block.parent_graph
        expr = '__state_{}_{}:;\n'.format(cfg.cfg_id, self.block.label)
        expr += 'continue;\n'
        return expr

    @property
    def first_block(self) -> ContinueBlock:
        return self.block


@dataclass
class ReturnCFBlock(ControlFlow):
    """ A CFG block that generates a 'return' statement. """

    block: ReturnBlock

    def as_cpp(self, codegen, symbols) -> str:
        cfg = self.block.parent_graph
        expr = '__state_{}_{}:;\n'.format(cfg.cfg_id, self.block.label)
        expr += 'return;\n'
        return expr

    @property
    def first_block(self) -> ReturnBlock:
        return self.block


@dataclass
class RegionBlock(ControlFlow):

    # The control flow region that this block corresponds to (may be the SDFG in the absence of hierarchical regions).
    region: Optional[ControlFlowRegion]


@dataclass
class GeneralBlock(RegionBlock):
    """
    General (or unrecognized) control flow block with gotos between blocks.
    """

    # List of children control flow blocks
    elements: List[ControlFlow]

    # List or set of edges to not generate conditional gotos for. This is used to avoid generating extra assignments or
    # gotos before entering a for loop, for example.
    gotos_to_ignore: Sequence[Edge[InterstateEdge]]

    # List or set of edges to generate `continue;` statements in lieu of goto. This is used for loop blocks.
    # NOTE: Can be removed after a full conversion to only using hierarchical control flow and ditching CF detection.
    gotos_to_continue: Sequence[Edge[InterstateEdge]]

    # List or set of edges to generate `break;` statements in lieu of goto. This is used for loop blocks.
    # NOTE: Can be removed after a full conversion to only using hierarchical control flow and ditching CF detection.
    gotos_to_break: Sequence[Edge[InterstateEdge]]

    # List or set of edges to not generate inter-state assignments for.
    assignments_to_ignore: Sequence[Edge[InterstateEdge]]

    # True if control flow is sequential between elements, or False if contains irreducible control flow
    sequential: bool

    def as_cpp(self, codegen, symbols) -> str:
        expr = ''
        for i, elem in enumerate(self.elements):
            expr += elem.as_cpp(codegen, symbols)
            # In a general block, emit transitions and assignments after each individual block or region.
            if isinstance(elem, BasicCFBlock) or (isinstance(elem, RegionBlock) and elem.region):
                if isinstance(elem, BasicCFBlock):
                    g_elem = elem.state
                else:
                    g_elem = elem.region
                cfg = g_elem.parent_graph
                sdfg = cfg if isinstance(cfg, SDFG) else cfg.sdfg
                out_edges = cfg.out_edges(g_elem)
                for j, e in enumerate(out_edges):
                    if e not in self.gotos_to_ignore:
                        # Skip gotos to immediate successors
                        successor = None
                        # If this is the last generated edge
                        if j == (len(out_edges) - 1):
                            if (i + 1) < len(self.elements):
                                # If last edge leads to next state in block
                                successor = self.elements[i + 1].first_block
                            elif i == len(self.elements) - 1:
                                # If last edge leads to first state in next block
                                next_block = find_next_block(self)
                                if next_block is not None:
                                    successor = next_block.first_block

                        expr += elem.generate_transition(sdfg, cfg, e, successor)
                    else:
                        if e not in self.assignments_to_ignore:
                            # Need to generate assignments but not gotos
                            expr += elem.generate_transition(sdfg, cfg, e, assignments_only=True)
                        if e in self.gotos_to_break:
                            expr += 'break;\n'
                        elif e in self.gotos_to_continue:
                            expr += 'continue;\n'
                # Add exit goto as necessary
                if elem.last_block:
                    continue
                # Two negating conditions
                if (len(out_edges) == 2
                        and out_edges[0].data.condition_sympy() == sp.Not(out_edges[1].data.condition_sympy())):
                    continue
                # One unconditional edge
                if (len(out_edges) == 1 and out_edges[0].data.is_unconditional()):
                    continue
                if self.region:
                    expr += f'goto __state_exit_{self.region.cfg_id};\n'
                else:
                    expr += f'goto __state_exit_{sdfg.cfg_id};\n'

        if self.region and not isinstance(self.region, SDFG):
            expr += f'__state_exit_{self.region.cfg_id}:;\n'

        return expr

    @property
    def first_block(self) -> Optional[ControlFlowBlock]:
        if not self.elements:
            return None
        return self.elements[0].first_block

    @property
    def children(self) -> List[ControlFlow]:
        return self.elements


def _clean_loop_body(body: str) -> str:
    """ Cleans loop body from extraneous statements. """
    # Remove extraneous "continue" statement for code clarity
    if body.endswith('continue;\n'):
        body = body[:-len('continue;\n')]
    return body


@dataclass
class GeneralLoopScope(RegionBlock):
    """ General loop block based on a loop control flow region. """

    body: ControlFlow

    def as_cpp(self, codegen, symbols) -> str:
        sdfg = self.loop.sdfg

        cond = unparse_interstate_edge(self.loop.loop_condition.code[0], sdfg, codegen=codegen, symbols=symbols)
        cond = cond.strip(';')

        expr = ''

        if self.loop.update_statement and self.loop.init_statement and self.loop.loop_variable:
            lsyms = {}
            lsyms.update(symbols)
            if codegen.dispatcher.defined_vars.has(self.loop.loop_variable) and not self.loop.loop_variable in lsyms:
                lsyms[self.loop.loop_variable] = codegen.dispatcher.defined_vars.get(self.loop.loop_variable)[1]
            init = unparse_interstate_edge(self.loop.init_statement.code[0], sdfg, codegen=codegen, symbols=lsyms)
            init = init.strip(';')

            update = unparse_interstate_edge(self.loop.update_statement.code[0], sdfg, codegen=codegen, symbols=lsyms)
            update = update.strip(';')

            if self.loop.inverted:
                if self.loop.update_before_condition:
                    expr += f'{init};\n'
                    expr += 'do {\n'
                    expr += _clean_loop_body(self.body.as_cpp(codegen, symbols))
                    expr += f'{update};\n'
                    expr += f'}} while({cond});\n'
                else:
                    expr += f'{init};\n'
                    expr += 'while (1) {\n'
                    expr += _clean_loop_body(self.body.as_cpp(codegen, symbols))
                    expr += f'if (!({cond}))\n'
                    expr += 'break;\n'
                    expr += f'{update};\n'
                    expr += '}\n'
            else:
                expr += f'for ({init}; {cond}; {update}) {{\n'
                expr += _clean_loop_body(self.body.as_cpp(codegen, symbols))
                expr += '\n}\n'
        else:
            if self.loop.inverted:
                expr += 'do {\n'
                expr += _clean_loop_body(self.body.as_cpp(codegen, symbols))
                expr += f'\n}} while({cond});\n'
            else:
                expr += f'while ({cond}) {{\n'
                expr += _clean_loop_body(self.body.as_cpp(codegen, symbols))
                expr += '\n}\n'

        expr += f'__state_exit_{self.loop.cfg_id}:;\n'

        return expr

    @property
    def loop(self) -> LoopRegion:
        return self.region

    @property
    def first_block(self) -> ControlFlowBlock:
        return self.loop.start_block

    @property
    def children(self) -> List[ControlFlow]:
        return [self.body]


@dataclass
class GeneralConditionalScope(RegionBlock):
    """ General conditional block based on a conditional control flow region. """

    branch_bodies: List[Tuple[Optional[CodeBlock], ControlFlow]]

    def as_cpp(self, codegen, symbols) -> str:
        sdfg = self.conditional.sdfg
        expr = ''
        for i in range(len(self.branch_bodies)):
            branch = self.branch_bodies[i]
            if branch[0] is not None:
                cond = unparse_interstate_edge(branch[0].code, sdfg, codegen=codegen, symbols=symbols)
                cond = cond.strip(';')
                if i == 0:
                    expr += f'if ({cond}) {{\n'
                else:
                    expr += f'}} else if ({cond}) {{\n'
            else:
                if i < len(self.branch_bodies) - 1 or i == 0:
                    raise RuntimeError('Missing branch condition for non-final conditional branch')
                expr += '} else {\n'
            expr += branch[1].as_cpp(codegen, symbols)
            if i == len(self.branch_bodies) - 1:
                expr += '}\n'
        return expr

    @property
    def conditional(self) -> ConditionalBlock:
        return self.region

    @property
    def first_block(self) -> ControlFlowBlock:
        return self.conditional

    @property
    def children(self) -> List[ControlFlow]:
        return [b for _, b in self.branch_bodies]


def _child_of(node: SDFGState, parent: SDFGState, ptree: Dict[SDFGState, SDFGState]) -> bool:
    curnode = node
    while curnode is not None:
        if curnode is parent:
            return True
        curnode = ptree[curnode]
    return False


def find_next_block(block: ControlFlow) -> Optional[ControlFlow]:
    """
    Returns the immediate successor control flow block.
    """
    # Find block in parent
    parent = block.parent
    if parent is None:
        return None
    ind = next(i for i, b in enumerate(parent.children) if b is block)
    if ind == len(parent.children) - 1:
        # If last block, recursively continue upwards
        return find_next_block(parent)
    return parent.children[ind + 1]


def _reset_block_parents(block: ControlFlow):
    """
    Fixes block parents after processing.
    """
    for child in block.children:
        child.parent = block
        _reset_block_parents(child)


def _structured_control_flow_traversal(cfg: ControlFlowRegion,
                                       dispatch_state: Callable[[SDFGState], str],
                                       parent_block: GeneralBlock,
                                       start: Optional[ControlFlowBlock] = None,
                                       stop: Optional[ControlFlowBlock] = None,
                                       generate_children_of: Optional[ControlFlowBlock] = None,
                                       ptree: Optional[Dict[ControlFlowBlock, ControlFlowBlock]] = None,
                                       visited: Optional[Set[ControlFlowBlock]] = None):
    if ptree is None:
        ptree = cfg_analysis.block_parent_tree(cfg, with_loops=False)

    start = start if start is not None else cfg.start_block

    def make_empty_block(region):
        return GeneralBlock(dispatch_state,
                            parent_block,
                            last_block=False,
                            region=region,
                            elements=[],
                            gotos_to_ignore=[],
                            gotos_to_break=[],
                            gotos_to_continue=[],
                            assignments_to_ignore=[],
                            sequential=True)

    # Traverse states in custom order
    visited = set() if visited is None else visited
    stack = [start]
    while stack:
        node = stack.pop()
        if (generate_children_of is not None and not _child_of(node, generate_children_of, ptree)):
            continue
        if node in visited or node is stop:
            continue
        visited.add(node)

        cfg_block: ControlFlow
        if isinstance(node, SDFGState):
            cfg_block = BasicCFBlock(dispatch_state, parent_block, False, node)
        elif isinstance(node, BreakBlock):
            cfg_block = BreakCFBlock(dispatch_state, parent_block, True, node)
        elif isinstance(node, ContinueBlock):
            cfg_block = ContinueCFBlock(dispatch_state, parent_block, True, node)
        elif isinstance(node, ReturnBlock):
            cfg_block = ReturnCFBlock(dispatch_state, parent_block, True, node)
        elif isinstance(node, ConditionalBlock):
            cfg_block = GeneralConditionalScope(dispatch_state, parent_block, False, node, [])
            for cond, branch in node.branches:
                if branch is not None:
                    body = make_empty_block(branch)
                    body.parent = cfg_block
                    _structured_control_flow_traversal(branch, dispatch_state, body)
                    cfg_block.branch_bodies.append((cond, body))
        elif isinstance(node, ControlFlowRegion):
            if isinstance(node, LoopRegion):
                body = make_empty_block(node)
                cfg_block = GeneralLoopScope(dispatch_state, parent_block, False, node, body)
                body.parent = cfg_block
                _structured_control_flow_traversal(node, dispatch_state, body)
            else:
                cfg_block = make_empty_block(node)
                cfg_block.region = node
                _structured_control_flow_traversal(node, dispatch_state, cfg_block)

        oe = cfg.out_edges(node)
        if len(oe) == 0:  # End state
            # If there are no remaining nodes, this is the last state and it can
            # be marked as such
            if len(stack) == 0:
                cfg_block.last_block = True
            parent_block.elements.append(cfg_block)
            continue
        elif len(oe) == 1:  # No traversal change
            stack.append(oe[0].dst)
            parent_block.elements.append(cfg_block)
            continue
        else:
            # Unstructured control flow.
            parent_block.sequential = False
            parent_block.elements.append(cfg_block)
            stack.extend([e.dst for e in oe])

    return visited - {stop}


def structured_control_flow_tree(cfg: ControlFlowRegion, dispatch_state: Callable[[SDFGState], str]) -> ControlFlow:
    """
    Returns a structured control-flow tree (i.e., with constructs such as branches and loops) from a CFG based on the
    control flow regions it contains.

    :param cfg: The graph to iterate over.
    :return: Control-flow block representing the entire graph.
    """
    root_block = GeneralBlock(dispatch_state=dispatch_state,
                              parent=None,
                              last_block=False,
                              region=None,
                              elements=[],
                              gotos_to_ignore=[],
                              gotos_to_continue=[],
                              gotos_to_break=[],
                              assignments_to_ignore=[],
                              sequential=True)
    _structured_control_flow_traversal(cfg, dispatch_state, root_block)
    _reset_block_parents(root_block)
    return root_block
