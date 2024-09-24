# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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
from dace.sdfg.state import (BreakBlock, ContinueBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion,
                             ReturnBlock, SDFGState)
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
        return 'break;\n'

    @property
    def first_block(self) -> BreakBlock:
        return self.block


@dataclass
class ContinueCFBlock(ControlFlow):
    """ A CFG block that generates a 'continue' statement. """

    block: ContinueBlock

    def as_cpp(self, codegen, symbols) -> str:
        return 'continue;\n'

    @property
    def first_block(self) -> ContinueBlock:
        return self.block


@dataclass
class ReturnCFBlock(ControlFlow):
    """ A CFG block that generates a 'return' statement. """

    block: ReturnBlock

    def as_cpp(self, codegen, symbols) -> str:
        return 'return;\n'

    @property
    def first_block(self) -> ReturnBlock:
        return self.block


@dataclass
class GeneralBlock(ControlFlow):
    """ 
    General (or unrecognized) control flow block with gotos between blocks. 
    """

    # The control flow region that this block corresponds to (may be the SDFG in the absence of hierarchical regions).
    region: Optional[ControlFlowRegion]

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
            if isinstance(elem, BasicCFBlock) or (isinstance(elem, GeneralBlock) and elem.region):
                cfg = elem.state.parent_graph if isinstance(elem, BasicCFBlock) else elem.region.parent_graph
                sdfg = cfg if isinstance(cfg, SDFG) else cfg.sdfg
                out_edges = cfg.out_edges(elem.state) if isinstance(elem, BasicCFBlock) else cfg.out_edges(elem.region)
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
                expr += f'goto __state_exit_{sdfg.cfg_id};\n'

        return expr

    @property
    def first_block(self) -> Optional[ControlFlowBlock]:
        if not self.elements:
            return None
        return self.elements[0].first_block

    @property
    def children(self) -> List[ControlFlow]:
        return self.elements


@dataclass
class IfScope(ControlFlow):
    """ A control flow scope of an if (else) block. """

    branch_block: ControlFlowBlock  #: Block that branches out to if/else scopes
    condition: CodeBlock  #: If-condition
    body: GeneralBlock  #: Body of if condition
    orelse: Optional[GeneralBlock] = None  #: Optional body of else condition

    def as_cpp(self, codegen, symbols) -> str:
        condition_string = unparse_interstate_edge(self.condition.code[0], self.branch_block.sdfg, codegen=codegen)
        expr = f'if ({condition_string}) {{\n'
        expr += self.body.as_cpp(codegen, symbols)
        expr += '\n}'
        if self.orelse:
            expr += ' else {\n'
            expr += self.orelse.as_cpp(codegen, symbols)
            expr += '\n}'
        expr += '\n'
        return expr

    @property
    def first_block(self) -> ControlFlowBlock:
        return self.branch_block

    @property
    def children(self) -> List[ControlFlow]:
        return [self.body] + ([self.orelse] if self.orelse else [])


@dataclass
class IfElseChain(ControlFlow):
    """ A control flow scope of "if, else if, ..., else" chain of blocks. """

    branch_block: ControlFlowBlock  #: Block that branches out to all blocks
    body: List[Tuple[CodeBlock, GeneralBlock]]  #: List of (condition, block)

    def as_cpp(self, codegen, symbols) -> str:
        expr = ''
        for i, (condition, body) in enumerate(self.body):
            # First block in the chain is just "if", rest are "else if"
            prefix = '' if i == 0 else ' else '

            condition_string = unparse_interstate_edge(condition.code[0], self.branch_block.sdfg, codegen=codegen)
            expr += f'{prefix}if ({condition_string}) {{\n'
            expr += body.as_cpp(codegen, symbols)
            expr += '\n}'

        # If we generate an if/else if blocks, we cannot guarantee that all
        # cases have been covered. In SDFG semantics, this means that the SDFG
        # execution should end, so we emit an "else goto exit" here.
        if len(self.body) > 0:
            expr += ' else {\n'
        expr += 'goto __state_exit_{};\n'.format(self.branch_block.sdfg.cfg_id)
        if len(self.body) > 0:
            expr += '\n}'
        return expr

    @property
    def first_block(self) -> ControlFlowBlock:
        return self.branch_block

    @property
    def children(self) -> List[ControlFlow]:
        return [block for _, block in self.body]


def _clean_loop_body(body: str) -> str:
    """ Cleans loop body from extraneous statements. """
    # Remove extraneous "continue" statement for code clarity
    if body.endswith('continue;\n'):
        body = body[:-len('continue;\n')]
    return body


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

    def as_cpp(self, codegen, symbols) -> str:

        sdfg = self.guard.parent

        # Initialize to either "int i = 0" or "i = 0" depending on whether
        # the type has been defined
        defined_vars = codegen.dispatcher.defined_vars
        init = ''
        if self.init is not None:
            if defined_vars.has(self.itervar):
                init = self.itervar
            else:
                init = f'{symbols[self.itervar]} {self.itervar}'
            init += ' = ' + unparse_interstate_edge(
                self.init_edges[0].data.assignments[self.itervar], sdfg, codegen=codegen)

        preinit = ''
        if self.init_edges:
            for edge in self.init_edges:
                for k, v in edge.data.assignments.items():
                    if k != self.itervar:
                        cppinit = unparse_interstate_edge(v, sdfg, codegen=codegen)
                        preinit += f'{k} = {cppinit};\n'

        if self.condition is not None:
            cond = unparse_interstate_edge(self.condition.code[0], sdfg, codegen=codegen)
        else:
            cond = ''

        update = ''
        if self.update is not None:
            cppupdate = unparse_interstate_edge(self.update, sdfg, codegen=codegen)
            update = f'{self.itervar} = {cppupdate}'

        expr = f'{preinit}\nfor ({init}; {cond}; {update}) {{\n'
        expr += _clean_loop_body(self.body.as_cpp(codegen, symbols))
        expr += '\n}\n'
        return expr

    @property
    def first_block(self) -> SDFGState:
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

    def as_cpp(self, codegen, symbols) -> str:
        if self.test is not None:
            sdfg = self.guard.parent
            test = unparse_interstate_edge(self.test.code[0], sdfg, codegen=codegen)
        else:
            test = 'true'

        expr = f'while ({test}) {{\n'
        expr += _clean_loop_body(self.body.as_cpp(codegen, symbols))
        expr += '\n}\n'
        return expr

    @property
    def first_block(self) -> SDFGState:
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

    def as_cpp(self, codegen, symbols) -> str:
        if self.test is not None:
            test = unparse_interstate_edge(self.test.code[0], self.sdfg, codegen=codegen)
        else:
            test = 'true'

        expr = 'do {\n'
        expr += _clean_loop_body(self.body.as_cpp(codegen, symbols))
        expr += f'\n}} while ({test});\n'
        return expr

    @property
    def first_block(self) -> SDFGState:
        return self.body[0].first_state

    @property
    def children(self) -> List[ControlFlow]:
        return [self.body]


@dataclass
class GeneralLoopScope(ControlFlow):
    """ General loop block based on a loop control flow region. """

    loop: LoopRegion
    body: ControlFlow

    def as_cpp(self, codegen, symbols) -> str:
        sdfg = self.loop.sdfg

        cond = unparse_interstate_edge(self.loop.loop_condition.code[0], sdfg, codegen=codegen, symbols=symbols)
        cond = cond.strip(';')

        expr = ''

        if self.loop.update_statement and self.loop.init_statement and self.loop.loop_variable:
            # Initialize to either "int i = 0" or "i = 0" depending on whether the type has been defined.
            defined_vars = codegen.dispatcher.defined_vars
            if not defined_vars.has(self.loop.loop_variable):
                try:
                    init = f'{symbols[self.loop.loop_variable]} '
                except KeyError:
                    init = 'auto '
                    symbols[self.loop.loop_variable] = None
            init += unparse_interstate_edge(self.loop.init_statement.code[0], sdfg, codegen=codegen, symbols=symbols)
            init = init.strip(';')

            update = unparse_interstate_edge(self.loop.update_statement.code[0], sdfg, codegen=codegen, symbols=symbols)
            update = update.strip(';')

            if self.loop.inverted:
                expr += f'{init};\n'
                expr += 'do {\n'
                expr += _clean_loop_body(self.body.as_cpp(codegen, symbols))
                expr += f'{update};\n'
                expr += f'\n}} while({cond});\n'
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

        return expr

    @property
    def first_block(self) -> ControlFlowBlock:
        return self.loop.start_block

    @property
    def children(self) -> List[ControlFlow]:
        return [self.body]


@dataclass
class SwitchCaseScope(ControlFlow):
    """ Simple switch-case scope without fall-through cases. """

    branch_block: ControlFlowBlock  #: Branching block
    switchvar: str  #: C++ code for switch expression
    cases: Dict[str, GeneralBlock]  #: Mapping of cases to control flow blocks

    def as_cpp(self, codegen, symbols) -> str:
        expr = f'switch ({self.switchvar}) {{\n'
        for case, body in self.cases.items():
            expr += f'case {case}: {{\n'
            expr += body.as_cpp(codegen, symbols)
            expr += 'break;\n}\n'
        expr += f'default: goto __state_exit_{self.branch_block.sdfg.cfg_id};'
        expr += '\n}\n'
        return expr

    @property
    def first_block(self) -> ControlFlowBlock:
        return self.branch_block

    @property
    def children(self) -> List[ControlFlow]:
        return list(self.cases.values())


def _loop_from_structure(sdfg: SDFG, guard: SDFGState, enter_edge: Edge[InterstateEdge],
                         leave_edge: Edge[InterstateEdge], back_edges: List[Edge[InterstateEdge]],
                         dispatch_state: Callable[[SDFGState],
                                                  str], parent_block: GeneralBlock) -> Union[ForScope, WhileScope]:
    """ 
    Helper method that constructs the correct structured loop construct from a
    set of states. Can construct for or while loops.
    """

    body = GeneralBlock(dispatch_state=dispatch_state,
                        parent=parent_block,
                        last_block=False,
                        region=None,
                        elements=[],
                        gotos_to_ignore=[],
                        gotos_to_continue=[],
                        gotos_to_break=[],
                        assignments_to_ignore=[],
                        sequential=True)

    guard_inedges = sdfg.in_edges(guard)
    increment_edges = [e for e in guard_inedges if e in back_edges]
    init_edges = [e for e in guard_inedges if e not in back_edges]

    # If no back edge found (or more than one, indicating a "continue"
    # statement), disregard
    if len(increment_edges) > 1 or len(increment_edges) == 0:
        return None
    increment_edge = increment_edges[0]

    # Increment edge goto to be ignored in body
    body.gotos_to_ignore.append(increment_edge)
    body.gotos_to_continue.append(increment_edge)

    # Outgoing edges must be a negation of each other
    if enter_edge.data.condition_sympy() != (sp.Not(leave_edge.data.condition_sympy())):
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
        if (all(e.data.assignments[itvar] == init for e in init_edges) and len(increment_edge.data.assignments) == 1):
            update = increment_edge.data.assignments[itvar]

            # Also ignore assignments in increment edge (handled in for stmt)
            body.assignments_to_ignore.append(increment_edge)

            return ForScope(dispatch_state, parent_block, False, itvar, guard, init, condition, update, body,
                            init_edges)

    # Otherwise, it is a while loop
    return WhileScope(dispatch_state, parent_block, False, guard, condition, body)


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
    if not isinstance(cond, sp.Basic):
        return None
    a = sp.Wild('a')
    b = sp.Wild('b', properties=[lambda k: k.is_Integer])
    m = cond.match(sp.Eq(a, b))
    if m:
        # Obtain original code for variable
        call_or_compare = edges[0].data.condition.code[0].value
        if isinstance(call_or_compare, ast.Call):
            astvar = call_or_compare.args[0]
        else:  # Binary comparison
            astvar = call_or_compare.left
    else:
        # Try integer == symbol
        m = cond.match(sp.Eq(b, a))
        if m:
            call_or_compare = edges[0].data.condition.code[0].value
            if isinstance(call_or_compare, ast.Call):
                astvar = call_or_compare.args[1]
            else:  # Binary comparison
                astvar = call_or_compare.right
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
        result[sym2cpp(ematch[b])] = cblocks[e]

    return switchvar, result


def _ignore_recursive(edges: List[Edge[InterstateEdge]], block: ControlFlow):
    """ 
    Ignore a list of edges recursively in a control flow block and its children.
    """
    if isinstance(block, GeneralBlock):
        block.gotos_to_ignore.extend(edges)
        block.assignments_to_ignore.extend(edges)
    for subblock in block.children:
        _ignore_recursive(edges, subblock)


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
    if ind == len(parent.children) - 1 or isinstance(parent, (IfScope, IfElseChain, SwitchCaseScope)):
        # If last block, or other children are not reachable from current node (branches),
        # recursively continue upwards
        return find_next_block(parent)
    return parent.children[ind + 1]


def _reset_block_parents(block: ControlFlow):
    """
    Fixes block parents after processing.
    """
    for child in block.children:
        child.parent = block
        _reset_block_parents(child)


def _structured_control_flow_traversal(sdfg: SDFG,
                                       start: SDFGState,
                                       ptree: Dict[SDFGState, SDFGState],
                                       branch_merges: Dict[SDFGState, SDFGState],
                                       back_edges: List[Edge[InterstateEdge]],
                                       dispatch_state: Callable[[SDFGState], str],
                                       parent_block: GeneralBlock,
                                       stop: SDFGState = None,
                                       generate_children_of: SDFGState = None,
                                       visited: Set[SDFGState] = None):
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

    def make_empty_block():
        return GeneralBlock(dispatch_state=dispatch_state,
                            last_block=False,
                            parent=parent_block,
                            region=None,
                            elements=[],
                            gotos_to_ignore=[],
                            gotos_to_continue=[],
                            gotos_to_break=[],
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
        stateblock = BasicCFBlock(dispatch_state=dispatch_state, parent=parent_block, last_block=False, state=node)

        oe = sdfg.out_edges(node)
        if len(oe) == 0:  # End state
            # If there are no remaining nodes, this is the last state and it can
            # be marked as such
            if len(stack) == 0:
                stateblock.last_block = True
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
            parent_block.gotos_to_ignore.extend(oe)
            parent_block.assignments_to_ignore.extend(oe)
            stateblock.last_block = True

            # Parse all outgoing edges recursively first
            cblocks: Dict[Edge[InterstateEdge], GeneralBlock] = {}
            for branch in oe:
                if branch.dst is mergestate:
                    # If we hit the merge state (if without else), defer to end of branch traversal
                    continue
                cblocks[branch] = make_empty_block()
                _structured_control_flow_traversal(sdfg,
                                                   branch.dst,
                                                   ptree,
                                                   branch_merges,
                                                   back_edges,
                                                   dispatch_state,
                                                   cblocks[branch],
                                                   stop=mergestate,
                                                   generate_children_of=node,
                                                   visited=visited)

            # Classify branch type:
            branch_block = None
            # If there are 2 out edges, one negation of the other:
            #   * if/else in case both branches are not merge state
            #   * if without else in case one branch is merge state
            if (len(oe) == 2 and oe[0].data.condition_sympy() == sp.Not(oe[1].data.condition_sympy())):
                # If without else
                if oe[0].dst is mergestate:
                    branch_block = IfScope(dispatch_state, parent_block, False, node, oe[1].data.condition,
                                           cblocks[oe[1]])
                elif oe[1].dst is mergestate:
                    branch_block = IfScope(dispatch_state, parent_block, False, node, oe[0].data.condition,
                                           cblocks[oe[0]])
                else:
                    branch_block = IfScope(dispatch_state, parent_block, False, node, oe[0].data.condition,
                                           cblocks[oe[0]], cblocks[oe[1]])
            else:
                # If there are 2 or more edges (one is not the negation of the
                # other):
                switch = _cases_from_branches(oe, cblocks)
                if switch:
                    # If all edges are of form "x == y" for a single x and
                    # integer y, it is a switch/case
                    branch_block = SwitchCaseScope(dispatch_state, parent_block, False, node, switch[0], switch[1])
                else:
                    # Otherwise, create if/else if/.../else goto exit chain
                    branch_block = IfElseChain(dispatch_state, parent_block, False, node,
                                               [(e.data.condition, cblocks[e] if e in cblocks else make_empty_block())
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
                scope = _loop_from_structure(sdfg, node, oe[0], oe[1], back_edges, dispatch_state, parent_block)
                body_start = oe[0].dst
                loop_exit = oe[1].dst
            elif ptree[oe[1].dst] == node and ptree[oe[0].dst] != node:
                scope = _loop_from_structure(sdfg, node, oe[1], oe[0], back_edges, dispatch_state, parent_block)
                body_start = oe[1].dst
                loop_exit = oe[0].dst

            if scope:
                visited |= _structured_control_flow_traversal(sdfg,
                                                              body_start,
                                                              ptree,
                                                              branch_merges,
                                                              back_edges,
                                                              dispatch_state,
                                                              scope.body,
                                                              stop=node,
                                                              generate_children_of=node,
                                                              visited=visited)

                # Add branching node and ignore outgoing edges
                parent_block.elements.append(stateblock)
                parent_block.gotos_to_ignore.extend(oe)
                parent_block.assignments_to_ignore.extend(oe)

                parent_block.elements.append(scope)

                # If for loop, ignore certain edges
                if isinstance(scope, ForScope):
                    # Mark init edge(s) to ignore in parent_block and all children
                    _ignore_recursive([e for e in sdfg.in_edges(node) if e not in back_edges], parent_block)
                    # Mark back edge for ignoring in all children of loop body
                    _ignore_recursive([e for e in sdfg.in_edges(node) if e in back_edges], scope.body)

                stack.append(loop_exit)
                continue

            # No proper loop detected: Unstructured control flow
            parent_block.sequential = False
            parent_block.elements.append(stateblock)
            stack.extend([e.dst for e in oe])
        else:  # No merge state: Unstructured control flow
            parent_block.sequential = False
            parent_block.elements.append(stateblock)
            stack.extend([e.dst for e in oe])

    return visited - {stop}


def _structured_control_flow_traversal_with_regions(cfg: ControlFlowRegion,
                                                    dispatch_state: Callable[[SDFGState], str],
                                                    parent_block: GeneralBlock,
                                                    start: Optional[ControlFlowBlock] = None,
                                                    stop: Optional[ControlFlowBlock] = None,
                                                    generate_children_of: Optional[ControlFlowBlock] = None,
                                                    branch_merges: Optional[Dict[ControlFlowBlock,
                                                                                 ControlFlowBlock]] = None,
                                                    ptree: Optional[Dict[ControlFlowBlock, ControlFlowBlock]] = None,
                                                    visited: Optional[Set[ControlFlowBlock]] = None):
    if branch_merges is None:
        branch_merges = cfg_analysis.branch_merges(cfg)


    if ptree is None:
        ptree = cfg_analysis.block_parent_tree(cfg, with_loops=False)

    start = start if start is not None else cfg.start_block

    def make_empty_block():
        return GeneralBlock(dispatch_state, parent_block,
                            last_block=False, region=None, elements=[], gotos_to_ignore=[],
                            gotos_to_break=[], gotos_to_continue=[], assignments_to_ignore=[], sequential=True)

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
        elif isinstance(node, ControlFlowRegion):
            if isinstance(node, LoopRegion):
                body = make_empty_block()
                cfg_block = GeneralLoopScope(dispatch_state, parent_block, False, node, body)
                body.parent = cfg_block
                _structured_control_flow_traversal_with_regions(node, dispatch_state, body)
            else:
                cfg_block = make_empty_block()
                cfg_block.region = node
                _structured_control_flow_traversal_with_regions(node, dispatch_state, cfg_block)

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

        # Potential branch or loop
        if node in branch_merges:
            mergeblock = branch_merges[node]

            # Add branching node and ignore outgoing edges
            parent_block.elements.append(cfg_block)
            parent_block.gotos_to_ignore.extend(oe)  # TODO: why?
            parent_block.assignments_to_ignore.extend(oe)  # TODO: why?
            cfg_block.last_block = True

            # Parse all outgoing edges recursively first
            cblocks: Dict[Edge[InterstateEdge], GeneralBlock] = {}
            for branch in oe:
                if branch.dst is mergeblock:
                    # If we hit the merge state (if without else), defer to end of branch traversal
                    continue
                cblocks[branch] = make_empty_block()
                _structured_control_flow_traversal_with_regions(cfg=cfg,
                                                                dispatch_state=dispatch_state,
                                                                parent_block=cblocks[branch],
                                                                start=branch.dst,
                                                                stop=mergeblock,
                                                                generate_children_of=node,
                                                                branch_merges=branch_merges,
                                                                ptree=ptree,
                                                                visited=visited)

            # Classify branch type:
            branch_block = None
            # If there are 2 out edges, one negation of the other:
            #   * if/else in case both branches are not merge state
            #   * if without else in case one branch is merge state
            if (len(oe) == 2 and oe[0].data.condition_sympy() == sp.Not(oe[1].data.condition_sympy())):
                if oe[0].dst is mergeblock:
                    # If without else
                    branch_block = IfScope(dispatch_state, parent_block, False, node, oe[1].data.condition,
                                           cblocks[oe[1]])
                elif oe[1].dst is mergeblock:
                    branch_block = IfScope(dispatch_state, parent_block, False, node, oe[0].data.condition,
                                           cblocks[oe[0]])
                else:
                    branch_block = IfScope(dispatch_state, parent_block, False, node, oe[0].data.condition,
                                           cblocks[oe[0]], cblocks[oe[1]])
            else:
                # If there are 2 or more edges (one is not the negation of the
                # other):
                switch = _cases_from_branches(oe, cblocks)
                if switch:
                    # If all edges are of form "x == y" for a single x and
                    # integer y, it is a switch/case
                    branch_block = SwitchCaseScope(dispatch_state, parent_block, False, node, switch[0], switch[1])
                else:
                    # Otherwise, create if/else if/.../else goto exit chain
                    branch_block = IfElseChain(dispatch_state, parent_block, False, node,
                                               [(e.data.condition, cblocks[e] if e in cblocks else make_empty_block())
                                                for e in oe])
            # End of branch classification
            parent_block.elements.append(branch_block)
            if mergeblock != stop:
                stack.append(mergeblock)

        else:  # No merge state: Unstructured control flow
            parent_block.sequential = False
            parent_block.elements.append(cfg_block)
            stack.extend([e.dst for e in oe])

    return visited - {stop}


def structured_control_flow_tree_with_regions(sdfg: SDFG, dispatch_state: Callable[[SDFGState], str]) -> ControlFlow:
    """
    Returns a structured control-flow tree (i.e., with constructs such as branches and loops) from an SDFG based on the
    control flow regions it contains.
    
    :param sdfg: The SDFG to iterate over.
    :return: Control-flow block representing the entire SDFG.
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
    _structured_control_flow_traversal_with_regions(sdfg, dispatch_state, root_block)
    _reset_block_parents(root_block)
    return root_block


def structured_control_flow_tree(sdfg: SDFG, dispatch_state: Callable[[SDFGState], str]) -> ControlFlow:
    """
    Returns a structured control-flow tree (i.e., with constructs such as 
    branches and loops) from an SDFG, which can be used to generate its code
    in a compiler- and human-friendly way.
    
    :param sdfg: The SDFG to iterate over.
    :return: Control-flow block representing the entire SDFG.
    """
    if sdfg.root_sdfg.using_experimental_blocks:
        return structured_control_flow_tree_with_regions(sdfg, dispatch_state)

    # Avoid import loops
    from dace.sdfg.analysis import cfg

    # Get parent states and back-edges
    idom = nx.immediate_dominators(sdfg.nx, sdfg.start_block)
    alldoms = cfg.all_dominators(sdfg, idom)
    ptree = cfg.block_parent_tree(sdfg, idom=idom)
    back_edges = cfg.back_edges(sdfg, idom, alldoms)

    # Annotate branches
    branch_merges: Dict[SDFGState, SDFGState] = cfg.branch_merges(sdfg, idom, alldoms)
   
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
    _structured_control_flow_traversal(sdfg, sdfg.start_state, ptree, branch_merges, back_edges, dispatch_state,
                                       root_block)
    _reset_block_parents(root_block)
    return root_block
