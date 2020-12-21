# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Various classes to facilitate the code generation of structured control 
    flow elements (e.g., `for`, `if`, `while`) from state machines in SDFGs. """

from dataclasses import dataclass, field
from typing import (Callable, Dict, Iterator, List, Optional, Sequence, Set,
                    Tuple, Union)
import sympy as sp
from dace.sdfg.state import SDFGState
from dace.sdfg.sdfg import SDFG, InterstateEdge
from dace.sdfg.graph import Edge
from dace.properties import CodeBlock, Property, make_properties
from dace.codegen.targets import cpp

###############################################################################


@dataclass
class ControlFlow:
    dispatch_state: Callable[[SDFGState], str]


@dataclass
class SingleState(ControlFlow):
    """ A control flow element containing a single state. """
    state: SDFGState
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

    def generate_transition(self, sdfg: SDFG,
                            edge: Edge[InterstateEdge]) -> str:
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

        expr += 'goto __state_{}_{};\n'.format(sdfg.sdfg_id, edge.dst.label)

        if not edge.data.is_unconditional():
            expr += '}\n'
        return expr


@dataclass
class GeneralBlock(ControlFlow):
    """ General (or unrecognized) control flow block with gotos. """
    elements: List[ControlFlow]
    edges_to_ignore: Sequence[Edge[InterstateEdge]]

    def as_cpp(self, defined_vars, symbols) -> str:
        expr = ''
        for elem in self.elements:
            expr += elem.as_cpp(defined_vars, symbols)
            # In a general block, emit transitions and assignments after each
            # individual state
            if isinstance(elem, SingleState):
                sdfg = elem.state.parent
                out_edges = sdfg.out_edges(elem.state)
                for e in out_edges:
                    if e not in self.edges_to_ignore:
                        expr += elem.generate_transition(sdfg, e)
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


@dataclass
class IfScope(ControlFlow):
    """ A control flow scope of an if (else) block. """
    sdfg: SDFG
    condition: CodeBlock
    body: GeneralBlock
    orelse: Optional[GeneralBlock] = None

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
        return expr


@dataclass
class IfElseChain(ControlFlow):
    """ A control flow scope of "if, else if, ..., else" chain of blocks. """
    sdfg: SDFG
    body: List[Tuple[CodeBlock, GeneralBlock]]

    def as_cpp(self, defined_vars, symbols) -> str:
        expr = ''
        for i, (condition, body) in self.body:
            # First block in the chain is just "if", rest are "else if"
            prefix = '' if i == 0 else ' else '

            expr += f'{prefix}if ({condition}) {{\n'
            expr += body.as_cpp(defined_vars, symbols)
            expr += '\n}'

        if len(self.body) > 0:
            expr += ' else {\n'
        expr += 'goto __state_exit_{};\n'.format(self.sdfg.sdfg_id)
        if len(self.body) > 0:
            expr += '\n}'
        return expr


@dataclass
class ForScope(ControlFlow):
    """ For loop block (without break or continue statements). """
    itervar: str
    init: CodeBlock
    condition: CodeBlock
    update: CodeBlock
    body: GeneralBlock

    def as_cpp(self, defined_vars, symbols) -> str:
        # Initialize to either "int i = 0" or "i = 0" depending on whether
        # the type has been defined
        init = ''
        if self.init is not None:
            if defined_vars.has(self.itervar):
                init = self.itervar
            else:
                init = f'{symbols[self.itervar]} {self.itervar}'
            init += ' = ' + self.init.as_string

        cond = self.condition.as_string if self.condition is not None else ''

        update = ''
        if self.update is not None:
            update = f'{self.itervar} = {self.update.as_string}'

        expr = f'for ({init}; {cond}; {update}) {{\n'
        expr += self.body.as_cpp(defined_vars, symbols)
        expr += '\n}'
        return expr


@dataclass
class WhileScope(ControlFlow):
    """ While loop block (without break or continue statements). """
    test: CodeBlock
    body: GeneralBlock

    def as_cpp(self, defined_vars, symbols) -> str:
        test = self.test.as_string if self.test is not None else 'true'
        expr = f'while ({test}) {{\n'
        expr += self.body.as_cpp(defined_vars, symbols)
        expr += '\n}'
        return expr


@dataclass
class DoWhileScope(ControlFlow):
    """ Do-while loop block (without break or continue statements). """
    test: CodeBlock
    body: GeneralBlock

    def as_cpp(self, defined_vars, symbols) -> str:
        test = self.test.as_string if self.test is not None else 'true'
        expr = 'do {\n'
        expr += self.body.as_cpp(defined_vars, symbols)
        expr += f'\n}} while ({test});'
        return expr


@dataclass
class SwitchCaseScope(ControlFlow):
    """ Simple switch-case scope wihout fallbacks. """
    sdfg: SDFG
    switchvar: str
    cases: Dict[str, GeneralBlock]

    def as_cpp(self, defined_vars, symbols) -> str:
        expr = f'switch ({self.switchvar}) {{\n'
        for case, body in self.cases.items():
            expr += f'case {case}: {{\n'
            expr += body.as_cpp(defined_vars, symbols)
            expr += 'break;\n}'
        expr += f'default:\n goto __state_exit_{self.sdfg.sdfg_id};'
        expr += '\n}'
        return expr



def _structured_control_flow_traversal(
        sdfg: SDFG,
        start: SDFGState,
        ptree: Dict[SDFGState, SDFGState],
        branch_merges: Dict[SDFGState, SDFGState],
        dispatch_state: Callable[[SDFGState], str],
        parent_block: GeneralBlock,
        stop: SDFGState = None) -> Set[SDFGState]:
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

            # If there is no condition/assignment, there is no need to generate
            # state transition code (since the next popped element is the
            # succeeding state)
            if (oe[0].data.is_unconditional() and not oe[0].data.assignments
                    and oe[0].dst not in visited):
                parent_block.edges_to_ignore.append(oe[0])

            continue

        # Potential branch or loop
        if node in branch_merges:
            mergestate = branch_merges[node]

            # Add branching node and ignore outgoing edges
            parent_block.elements.append(stateblock)
            parent_block.edges_to_ignore.extend(oe)

            # Parse all outgoing edges recursively first
            cblocks: Dict[Edge[InterstateEdge], GeneralBlock] = {}
            for branch in oe:
                cblocks[branch] = GeneralBlock(dispatch_state, [], [])
                visited |= _structured_control_flow_traversal(sdfg,
                                                              branch.dst,
                                                              ptree,
                                                              branch_merges,
                                                              dispatch_state,
                                                              cblocks[branch],
                                                              stop=mergestate)

            # Classify branch type:
            branch_block = None
            # If there are 2 out edges, one negation of the other:
            #   * if/else in case both branches are not merge state
            #   * if without else in case one branch is merge state
            if (len(oe) == 2 and oe[0].data.condition_sympy() == sp.Not(
                    oe[1].data.condition_sympy())):
                # If without else
                if oe[0].dst is mergestate:
                    branch_block = IfScope(dispatch_state, sdfg,
                                           oe[1].data.condition, cblocks[oe[1]])
                elif oe[1].dst is mergestate:
                    branch_block = IfScope(dispatch_state, sdfg,
                                           oe[0].data.condition, cblocks[oe[0]])
                else:
                    branch_block = IfScope(dispatch_state, sdfg,
                                           oe[0].data.condition, cblocks[oe[0]],
                                           cblocks[oe[1]])
            # TODO: If there are 2 or more edges (one is not the negation of the other):
            #   * if all edges are of form "x == y" for a single x and integer
            #     y, it is a switch/case
            #   * otherwise, create if/else if/else if.../else goto exit chain
            else:
                raise NotImplementedError
            parent_block.elements.append(branch_block)
            if mergestate != stop:
                stack.append(mergestate)

        elif len(oe) == 2:  # Potential loop
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

    # Get parent states
    ptree = cfg.state_parent_tree(sdfg)

    # Construct inverse tree (node to children nodes) to traverse
    pchildren = {
        elem: [k for k, v in ptree.items() if v is elem]
        for elem in ptree.keys() | {None}
    }

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
                                       branch_merges, dispatch_state,
                                       root_block)
    return root_block
