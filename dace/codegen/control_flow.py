# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" 
Various classes to facilitate the code generation of structured control 
flow elements (e.g., ``for``, ``if``, ``while``) from state machines in SDFGs. 

SDFGs are state machines of dataflow graphs, where each node is a state and each
edge may contain a state transition condition and assignments. As such, when 
generating code from an SDFG, the straightforward way would be to generate code
for each state and conditional ``goto``s for the state transitions.
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
     /------>[s2]--------\
[s1] \                    ->[s5]
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

from dataclasses import dataclass
from typing import (Callable, Dict, Iterator, List, Optional, Sequence, Set,
                    Tuple, Union)
import sympy as sp
from dace import dtypes
from dace.sdfg.state import SDFGState
from dace.sdfg.sdfg import SDFG, InterstateEdge
from dace.sdfg.graph import Edge
from dace.properties import CodeBlock
from dace.codegen import cppunparse
from dace.codegen.targets import cpp

###############################################################################


@dataclass
class ControlFlow:
    """
    Abstract class representing a control flow block.
    """

    # A callback to the code generator that receives an SDFGState and returns
    # a string with its generated code.
    dispatch_state: Callable[[SDFGState], str]

    @property
    def first_state(self) -> SDFGState:
        """ 
        Returns the first or initializing state in this control flow block. 
        Used to determine which will be the next state in a control flow block
        to avoid generating extraneous ``goto``s.
        """
        return None

    @property
    def children(self) -> List['ControlFlow']:
        """ 
        Returns a list of control flow blocks that exist within this block.
        """
        return []

    def as_cpp(self, defined_vars: 'DefinedVars',
               symbols: Dict[str, dtypes.typeclass]) -> str:
        """ 
        Returns C++ code for this control flow block.
        :param defined_vars: A ``DefinedVars`` object with the variables defined
                             in the scope.
        :param symbols: A dictionary of symbol names and their types.
        :return: C++ string with the generated code of the control flow block.
        """
        raise NotImplementedError


@dataclass
class SingleState(ControlFlow):
    """ A control flow element containing a single state. """

    # The state in this element.
    state: SDFGState

    # Set to true if this is the last state in the parent control flow block,
    # in order to avoid generating an extraneous "goto exit" statement.
    last_state: bool = False

    def as_cpp(self, defined_vars, symbols) -> str:
        sdfg = self.state.parent

        expr = '__state_{}_{}:;\n'.format(sdfg.sdfg_id, self.state.label)
        if self.state.number_of_nodes() > 0:
            expr += '{\n'
            expr += self.dispatch_state(self.state)
            expr += '\n}\n'
        else:
            # Dispatch empty state in any case in order to register that the
            # state was dispatched
            self.dispatch_state(self.state)

        # If any state has no children, it should jump to the end of the SDFG
        if not self.last_state and sdfg.out_degree(self.state) == 0:
            expr += 'goto __state_exit_{};\n'.format(sdfg.sdfg_id)
        return expr

    def generate_transition(self,
                            sdfg: SDFG,
                            edge: Edge[InterstateEdge],
                            successor: SDFGState = None) -> str:
        """ 
        Helper function that generates a state transition (conditional goto) 
        from a state and an SDFG edge.
        :param sdfg: The parent SDFG.
        :param edge: The state transition edge to generate.
        :param successor: If not None, the state that will be generated right
                          after the current state (used to avoid extraneous 
                          gotos).
        :return: A c++ string representing the state transition code.
        """
        expr = ''
        condition_string = cpp.unparse_interstate_edge(
            edge.data.condition.code[0], sdfg)

        if not edge.data.is_unconditional():
            expr += f'if ({condition_string}) {{\n'

        if len(edge.data.assignments) > 0:
            expr += ';\n'.join([
                "{} = {}".format(variable,
                                 cpp.unparse_interstate_edge(value, sdfg))
                for variable, value in edge.data.assignments.items()
            ] + [''])

        if successor is None or edge.dst is not successor:
            expr += 'goto __state_{}_{};\n'.format(sdfg.sdfg_id, edge.dst.label)

        if not edge.data.is_unconditional():
            expr += '}\n'
        return expr

    @property
    def first_state(self) -> SDFGState:
        return self.state


@dataclass
class GeneralBlock(ControlFlow):
    """ 
    General (or unrecognized) control flow block with gotos between states. 
    """

    # List of children control flow blocks
    elements: List[ControlFlow]

    # List or set of edges to not generate conditional gotos for. This is used
    # to avoid generating extra assignments or gotos before entering a for loop,
    # for example.
    edges_to_ignore: Sequence[Edge[InterstateEdge]]

    def as_cpp(self, defined_vars, symbols) -> str:
        expr = ''
        for i, elem in enumerate(self.elements):
            expr += elem.as_cpp(defined_vars, symbols)
            # In a general block, emit transitions and assignments after each
            # individual state
            if isinstance(elem, SingleState):
                sdfg = elem.state.parent
                out_edges = sdfg.out_edges(elem.state)
                for j, e in enumerate(out_edges):
                    if e not in self.edges_to_ignore:
                        # If this is the last generated edge and it leads
                        # to the next state, skip emitting goto
                        successor = None
                        if (j == (len(out_edges) - 1)
                                and (i + 1) < len(self.elements)):
                            successor = self.elements[i + 1].first_state

                        expr += elem.generate_transition(sdfg, e, successor)
                # Add exit goto as necessary
                if elem.last_state:
                    continue
                # Two negating conditions
                if (len(out_edges) == 2
                        and out_edges[0].data.condition_sympy() == sp.Not(
                            out_edges[1].data.condition_sympy())):
                    continue
                # One unconditional edge
                if (len(out_edges) == 1
                        and out_edges[0].data.is_unconditional()):
                    continue
                expr += f'goto __state_exit_{sdfg.sdfg_id};\n'

        return expr

    @property
    def first_state(self) -> SDFGState:
        if not self.elements:
            return None
        return self.elements[0].first_state

    @property
    def children(self) -> List[ControlFlow]:
        return self.elements


@dataclass
class IfScope(ControlFlow):
    """ A control flow scope of an if (else) block. """

    sdfg: SDFG  #: Parent SDFG
    branch_state: SDFGState  #: State that branches out to if/else scopes
    condition: CodeBlock  #: If-condition
    body: GeneralBlock  #: Body of if condition
    orelse: Optional[GeneralBlock] = None  #: Optional body of else condition

    def as_cpp(self, defined_vars, symbols) -> str:
        condition_string = cpp.unparse_interstate_edge(self.condition.code[0],
                                                       self.sdfg)
        expr = f'if ({condition_string}) {{\n'
        expr += self.body.as_cpp(defined_vars, symbols)
        expr += '\n}'
        if self.orelse:
            expr += ' else {\n'
            expr += self.orelse.as_cpp(defined_vars, symbols)
            expr += '\n}'
        expr += '\n'
        return expr

    @property
    def first_state(self) -> SDFGState:
        return self.branch_state

    @property
    def children(self) -> List[ControlFlow]:
        return [self.body] + ([self.orelse] if self.orelse else [])


@dataclass
class IfElseChain(ControlFlow):
    """ A control flow scope of "if, else if, ..., else" chain of blocks. """
    sdfg: SDFG  #: Parent SDFG
    branch_state: SDFGState  #: State that branches out to all blocks
    body: List[Tuple[CodeBlock, GeneralBlock]]  #: List of (condition, block)

    def as_cpp(self, defined_vars, symbols) -> str:
        expr = ''
        for i, (condition, body) in enumerate(self.body):
            # First block in the chain is just "if", rest are "else if"
            prefix = '' if i == 0 else ' else '

            condition_string = cpp.unparse_interstate_edge(
                condition.code[0], self.sdfg)
            expr += f'{prefix}if ({condition_string}) {{\n'
            expr += body.as_cpp(defined_vars, symbols)
            expr += '\n}'

        # If we generate an if/else if blocks, we cannot guarantee that all
        # cases have been covered. In SDFG semantics, this means that the SDFG
        # execution should end, so we emit an "else goto exit" here.
        if len(self.body) > 0:
            expr += ' else {\n'
        expr += 'goto __state_exit_{};\n'.format(self.sdfg.sdfg_id)
        if len(self.body) > 0:
            expr += '\n}'
        return expr

    @property
    def first_state(self) -> SDFGState:
        return self.branch_state

    @property
    def children(self) -> List[ControlFlow]:
        return [block for _, block in self.body]


@dataclass
class ForScope(ControlFlow):
    """ For loop block (without break or continue statements). """
    itervar: str  #: Name of iteration variable
    guard: SDFGState  #: Loop guard state
    init: str  #: C++ code for initializing iteration variable
    condition: CodeBlock  #: For-loop condition
    update: str  #: C++ code for updating iteration variable
    body: GeneralBlock  #: Loop body as a control flow block
    init_edges: List[InterstateEdge]  #: All initialization edges

    def as_cpp(self, defined_vars, symbols) -> str:
        # Initialize to either "int i = 0" or "i = 0" depending on whether
        # the type has been defined
        init = ''
        if self.init is not None:
            if defined_vars.has(self.itervar):
                init = self.itervar
            else:
                init = f'{symbols[self.itervar]} {self.itervar}'
            init += ' = ' + self.init

        sdfg = self.guard.parent

        preinit = ''
        if self.init_edges:
            for edge in self.init_edges:
                for k, v in edge.data.assignments.items():
                    if k != self.itervar:
                        cppinit = cpp.unparse_interstate_edge(v, sdfg)
                        preinit += f'{k} = {cppinit};\n'

        if self.condition is not None:
            cond = cpp.unparse_interstate_edge(self.condition.code[0], sdfg)
        else:
            cond = ''

        update = ''
        if self.update is not None:
            update = f'{self.itervar} = {self.update}'

        expr = f'{preinit}\nfor ({init}; {cond}; {update}) {{\n'
        expr += self.body.as_cpp(defined_vars, symbols)
        expr += '\n}\n'
        return expr

    @property
    def first_state(self) -> SDFGState:
        return self.guard

    @property
    def children(self) -> List[ControlFlow]:
        return [self.body]


@dataclass
class WhileScope(ControlFlow):
    """ While loop block (without break or continue statements). """
    guard: SDFGState  #: Loop guard state
    test: CodeBlock  #: While-loop condition
    body: GeneralBlock  #: Loop body as control flow block

    def as_cpp(self, defined_vars, symbols) -> str:
        if self.test is not None:
            sdfg = self.guard.parent
            test = cpp.unparse_interstate_edge(self.test.code[0], sdfg)
        else:
            test = 'true'

        expr = f'while ({test}) {{\n'
        expr += self.body.as_cpp(defined_vars, symbols)
        expr += '\n}\n'
        return expr

    @property
    def first_state(self) -> SDFGState:
        return self.guard

    @property
    def children(self) -> List[ControlFlow]:
        return [self.body]


@dataclass
class DoWhileScope(ControlFlow):
    """ Do-while loop block (without break or continue statements). """
    sdfg: SDFG  #: Parent SDFG
    test: CodeBlock  #: Do-while loop condition
    body: GeneralBlock  #: Loop body as control flow block

    def as_cpp(self, defined_vars, symbols) -> str:
        if self.test is not None:
            test = cpp.unparse_interstate_edge(self.test.code[0], self.sdfg)
        else:
            test = 'true'

        expr = 'do {\n'
        expr += self.body.as_cpp(defined_vars, symbols)
        expr += f'\n}} while ({test});\n'
        return expr

    @property
    def first_state(self) -> SDFGState:
        return self.body[0].first_state

    @property
    def children(self) -> List[ControlFlow]:
        return [self.body]


@dataclass
class SwitchCaseScope(ControlFlow):
    """ Simple switch-case scope without fall-through cases. """
    sdfg: SDFG  #: Parent SDFG
    branch_state: SDFGState  #: Branching state
    switchvar: str  #: C++ code for switch expression
    cases: Dict[str, GeneralBlock]  #: Mapping of cases to control flow blocks

    def as_cpp(self, defined_vars, symbols) -> str:
        expr = f'switch ({self.switchvar}) {{\n'
        for case, body in self.cases.items():
            expr += f'case {case}: {{\n'
            expr += body.as_cpp(defined_vars, symbols)
            expr += 'break;\n}\n'
        expr += f'default: goto __state_exit_{self.sdfg.sdfg_id};'
        expr += '\n}\n'
        return expr

    @property
    def first_state(self) -> SDFGState:
        return self.branch_state

    @property
    def children(self) -> List[ControlFlow]:
        return list(self.cases.values())


def _loop_from_structure(
    sdfg: SDFG, guard: SDFGState, enter_edge: Edge[InterstateEdge],
    leave_edge: Edge[InterstateEdge], back_edges: List[Edge[InterstateEdge]],
    dispatch_state: Callable[[SDFGState], str]
) -> Union[ForScope, WhileScope]:
    """ 
    Helper method that constructs the correct structured loop construct from a
    set of states. Can construct for or while loops.
    """

    body = GeneralBlock(dispatch_state, [], [])

    guard_inedges = sdfg.in_edges(guard)
    increment_edges = [e for e in guard_inedges if e in back_edges]
    init_edges = [e for e in guard_inedges if e not in back_edges]

    # If no back edge found (or more than one, indicating a "continue"
    # statement), disregard
    if len(increment_edges) > 1 or len(increment_edges) == 0:
        return None
    increment_edge = increment_edges[0]

    # Mark increment edge to be ignored in body
    body.edges_to_ignore.append(increment_edge)

    # Outgoing edges must be a negation of each other
    if enter_edge.data.condition_sympy() != (sp.Not(
            leave_edge.data.condition_sympy())):
        return None

    # Body of guard state must be empty
    if not guard.is_empty():
        return None

    if not increment_edge.data.is_unconditional():
        return None
    if len(enter_edge.data.assignments) > 0:
        return None

    condition = enter_edge.data.condition

    # Detect whether this loop is a for loop:
    # All incoming edges to the guard must set the same variable
    itvars = None
    for iedge in guard_inedges:
        if itvars is None:
            itvars = set(iedge.data.assignments.keys())
        else:
            itvars &= iedge.data.assignments.keys()
    if itvars and len(itvars) == 1:
        itvar = next(iter(itvars))
        init = init_edges[0].data.assignments[itvar]

        # Check that all init edges are the same and that increment edge only
        # increments
        if (all(e.data.assignments[itvar] == init for e in init_edges)
                and len(increment_edge.data.assignments) == 1):
            update = increment_edge.data.assignments[itvar]
            return ForScope(dispatch_state, itvar, guard, init, condition,
                            update, body, init_edges)

    # Otherwise, it is a while loop
    return WhileScope(dispatch_state, guard, condition, body)


def _cases_from_branches(
    edges: List[Edge[InterstateEdge]],
    cblocks: Dict[Edge[InterstateEdge], GeneralBlock],
) -> Tuple[str, Dict[str, GeneralBlock]]:
    """ 
    If the input list of edges correspond to a switch/case scope (with all
    conditions being "x == y" for a unique symbolic x and integers y),
    returns the switch/case scope parameters.
    :param edges: List of inter-state edges.
    :return: Tuple of (case variable C++ expression, mapping from case to 
             control flow block). If not a valid switch/case scope, 
             returns None.
    """
    cond = edges[0].data.condition_sympy()
    a = sp.Wild('a')
    b = sp.Wild('b', properties=[lambda k: k.is_Integer])
    m = cond.match(sp.Eq(a, b))
    if m:
        # Obtain original code for variable
        astvar = edges[0].data.condition.code[0].value.left
    else:
        # Try integer == symbol
        m = cond.match(sp.Eq(b, a))
        if m:
            astvar = edges[0].data.condition.code[0].value.right
        else:
            return None

    # Get C++ expression from AST
    switchvar = cppunparse.pyexpr2cpp(astvar)

    # Check that all edges match criteria
    result = {}
    for e in edges:
        ematch = e.data.condition_sympy().match(sp.Eq(m[a], b))
        if not ematch:
            ematch = e.data.condition_sympy().match(sp.Eq(b, m[a]))
            if not ematch:
                return None
        # Create mapping to codeblocks
        result[cpp.sym2cpp(ematch[b])] = cblocks[e]

    return switchvar, result


def _ignore_recursive(edges: List[Edge[InterstateEdge]], block: ControlFlow):
    """ 
    Ignore a list of edges recursively in a control flow block and its children.
    """
    if isinstance(block, GeneralBlock):
        block.edges_to_ignore.extend(edges)
    for subblock in block.children:
        _ignore_recursive(edges, subblock)


def _child_of(node: SDFGState, parent: SDFGState,
              ptree: Dict[SDFGState, SDFGState]) -> bool:
    curnode = node
    while curnode is not None:
        if curnode is parent:
            return True
        curnode = ptree[curnode]
    return False


def _structured_control_flow_traversal(
        sdfg: SDFG,
        start: SDFGState,
        ptree: Dict[SDFGState, SDFGState],
        branch_merges: Dict[SDFGState, SDFGState],
        back_edges: List[Edge[InterstateEdge]],
        dispatch_state: Callable[[SDFGState], str],
        parent_block: GeneralBlock,
        stop: SDFGState = None,
        generate_children_of: SDFGState = None) -> Set[SDFGState]:
    """ 
    Helper function for ``structured_control_flow_tree``. 
    :param sdfg: SDFG.
    :param start: Starting state for traversal.
    :param ptree: State parent tree (computed from ``state_parent_tree``).
    :param branch_merges: Dictionary mapping from branch state to its merge
                          state.
    :param dispatch_state: A function that dispatches code generation for a 
                           single state.
    :param parent_block: The block to append children to.
    :param stop: Stopping state to not traverse through (merge state of a 
                 branch or guard state of a loop).
    :return: Generator that yields states in state-order from ``start`` to 
             ``stop``.
    """
    # Traverse states in custom order
    visited = set()
    if stop is not None:
        visited.add(stop)
    stack = [start]
    while stack:
        node = stack.pop()
        if (generate_children_of is not None
                and not _child_of(node, generate_children_of, ptree)):
            continue
        if node in visited:
            continue
        visited.add(node)
        stateblock = SingleState(dispatch_state, node)

        oe = sdfg.out_edges(node)
        if len(oe) == 0:  # End state
            # If there are no remaining nodes, this is the last state and it can
            # be marked as such
            if len(stack) == 0:
                stateblock.last_state = True
            parent_block.elements.append(stateblock)
            continue
        elif len(oe) == 1:  # No traversal change
            stack.append(oe[0].dst)
            parent_block.elements.append(stateblock)
            continue

        # Potential branch or loop
        if node in branch_merges:
            mergestate = branch_merges[node]

            # Add branching node and ignore outgoing edges
            parent_block.elements.append(stateblock)
            parent_block.edges_to_ignore.extend(oe)
            stateblock.last_state = True

            # Parse all outgoing edges recursively first
            cblocks: Dict[Edge[InterstateEdge], GeneralBlock] = {}
            for branch in oe:
                cblocks[branch] = GeneralBlock(dispatch_state, [], [])
                visited |= _structured_control_flow_traversal(
                    sdfg,
                    branch.dst,
                    ptree,
                    branch_merges,
                    back_edges,
                    dispatch_state,
                    cblocks[branch],
                    stop=mergestate,
                    generate_children_of=node)

            # Classify branch type:
            branch_block = None
            # If there are 2 out edges, one negation of the other:
            #   * if/else in case both branches are not merge state
            #   * if without else in case one branch is merge state
            if (len(oe) == 2 and oe[0].data.condition_sympy() == sp.Not(
                    oe[1].data.condition_sympy())):
                # If without else
                if oe[0].dst is mergestate:
                    branch_block = IfScope(dispatch_state, sdfg, node,
                                           oe[1].data.condition, cblocks[oe[1]])
                elif oe[1].dst is mergestate:
                    branch_block = IfScope(dispatch_state, sdfg, node,
                                           oe[0].data.condition, cblocks[oe[0]])
                else:
                    branch_block = IfScope(dispatch_state, sdfg, node,
                                           oe[0].data.condition, cblocks[oe[0]],
                                           cblocks[oe[1]])
            else:
                # If there are 2 or more edges (one is not the negation of the
                # other):
                switch = _cases_from_branches(oe, cblocks)
                if switch:
                    # If all edges are of form "x == y" for a single x and
                    # integer y, it is a switch/case
                    branch_block = SwitchCaseScope(dispatch_state, sdfg, node,
                                                   switch[0], switch[1])
                else:
                    # Otherwise, create if/else if/.../else goto exit chain
                    branch_block = IfElseChain(dispatch_state, sdfg, node,
                                               [(e.data.condition, cblocks[e])
                                                for e in oe])
            # End of branch classification
            parent_block.elements.append(branch_block)
            if mergestate != stop:
                stack.append(mergestate)

        elif len(oe) == 2:  # Potential loop
            # TODO(later): Recognize do/while loops
            # If loop, traverse body, then exit
            body_start = None
            loop_exit = None
            scope = None
            if ptree[oe[0].dst] == node and ptree[oe[1].dst] != node:
                scope = _loop_from_structure(sdfg, node, oe[0], oe[1],
                                             back_edges, dispatch_state)
                body_start = oe[0].dst
                loop_exit = oe[1].dst
            elif ptree[oe[1].dst] == node and ptree[oe[0].dst] != node:
                scope = _loop_from_structure(sdfg, node, oe[1], oe[0],
                                             back_edges, dispatch_state)
                body_start = oe[1].dst
                loop_exit = oe[0].dst

            if scope:
                visited |= _structured_control_flow_traversal(
                    sdfg,
                    body_start,
                    ptree,
                    branch_merges,
                    back_edges,
                    dispatch_state,
                    scope.body,
                    stop=node,
                    generate_children_of=node)

                # Add branching node and ignore outgoing edges
                parent_block.elements.append(stateblock)
                parent_block.edges_to_ignore.extend(oe)

                parent_block.elements.append(scope)

                # If for loop, ignore certain edges
                if isinstance(scope, ForScope):
                    # Mark init edge(s) to ignore in parent_block and all children
                    _ignore_recursive(
                        [e for e in sdfg.in_edges(node) if e not in back_edges],
                        parent_block)
                    # Mark back edge for ignoring in all children of loop body
                    _ignore_recursive(
                        [e for e in sdfg.in_edges(node) if e in back_edges],
                        scope.body)

                stack.append(loop_exit)
                continue

            # No proper loop detected: Unstructured control flow
            parent_block.elements.append(stateblock)
            stack.extend([e.dst for e in oe])
        else:  # No merge state: Unstructured control flow
            parent_block.elements.append(stateblock)
            stack.extend([e.dst for e in oe])

    return visited - {stop}


def structured_control_flow_tree(
        sdfg: SDFG, dispatch_state: Callable[[SDFGState], str]) -> ControlFlow:
    """
    Returns a structured control-flow tree (i.e., with constructs such as 
    branches and loops) from an SDFG, which can be used to generate its code
    in a compiler- and human-friendly way.
    :param sdfg: The SDFG to iterate over.
    :return: Control-flow block representing the entire SDFG.
    """
    # Avoid import loops
    from dace.sdfg.analysis import cfg

    # Get parent states and back-edges
    ptree = cfg.state_parent_tree(sdfg)
    back_edges = cfg.back_edges(sdfg)

    # Annotate branches
    branch_merges: Dict[SDFGState, SDFGState] = {}
    adf = cfg.acyclic_dominance_frontier(sdfg)
    for state in sdfg.nodes():
        oedges = sdfg.out_edges(state)
        # Skip if not branch
        if len(oedges) <= 1:
            continue
        # Skip if natural loop
        if len(oedges) == 2 and (
            (ptree[oedges[0].dst] == state and ptree[oedges[1].dst] != state) or
            (ptree[oedges[1].dst] == state and ptree[oedges[0].dst] != state)):
            continue

        common_frontier = set()
        for oedge in oedges:
            frontier = adf[oedge.dst]
            if not frontier:
                frontier = {oedge.dst}
            common_frontier |= frontier
        if len(common_frontier) == 1:
            branch_merges[state] = next(iter(common_frontier))

    root_block = GeneralBlock(dispatch_state, [], [])
    _structured_control_flow_traversal(sdfg, sdfg.start_state, ptree,
                                       branch_merges, back_edges,
                                       dispatch_state, root_block)
    return root_block
