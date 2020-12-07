# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Various classes to facilitate the code generation of structured control 
    flow elements (e.g., `for`, `if`, `while`) from state machines in SDFGs. """

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Sequence
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
    last_state: bool

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
                for e in sdfg.out_edges(elem.state):
                    if e not in self.edges_to_ignore:
                        expr += elem.generate_transition(sdfg, e)

        return expr


@dataclass
class IfScope(ControlFlow):
    """ A control flow scope of an if (else) block. """
    condition: CodeBlock
    body: List[ControlFlow]
    orelse: List[ControlFlow] = None

    def as_cpp(self, defined_vars, symbols) -> str:
        expr = f'if ({self.condition}) {{\n'
        expr += '\n'.join(
            elem.as_cpp(defined_vars, symbols) for elem in self.body)
        expr += '\n}'
        if self.orelse:
            expr += ' else {\n'
            expr += '\n'.join(
                elem.as_cpp(defined_vars, symbols) for elem in self.orelse)
            expr += '\n}'
        return expr


@dataclass
class ForScope(ControlFlow):
    """ For loop block (without break or continue statements). """
    itervar: str
    init: CodeBlock
    condition: CodeBlock
    update: CodeBlock
    body: List[ControlFlow]

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
        expr += '\n'.join(
            elem.as_cpp(defined_vars, symbols) for elem in self.body)
        expr += '\n}'
        return expr


@dataclass
class WhileScope(ControlFlow):
    """ While loop block (without break or continue statements). """
    test: CodeBlock
    body: List[ControlFlow]

    def as_cpp(self, defined_vars, symbols) -> str:
        test = self.test.as_string if self.test is not None else 'true'
        expr = f'while ({test}) {{\n'
        expr += '\n'.join(
            elem.as_cpp(defined_vars, symbols) for elem in self.body)
        expr += '\n}'
        return expr


@dataclass
class DoWhileScope(ControlFlow):
    """ Do-while loop block (without break or continue statements). """
    test: CodeBlock
    body: List[ControlFlow]

    def as_cpp(self, defined_vars, symbols) -> str:
        test = self.test.as_string if self.test is not None else 'true'
        expr = 'do {\n'
        expr += '\n'.join(
            elem.as_cpp(defined_vars, symbols) for elem in self.body)
        expr += f'\n}} while ({test});'
        return expr


@dataclass
class SwitchCaseScope(ControlFlow):
    """ Simple switch-case scope wihout fallbacks. """
    switchvar: str
    cases: Dict[str, List[ControlFlow]]

    def as_cpp(self, defined_vars, symbols) -> str:
        expr = f'switch ({self.switchvar}) {{\n'
        for case, body in self.cases.items():
            expr += f'case {case}: {{\n'
            expr += '\n'.join(
                elem.as_cpp(defined_vars, symbols) for elem in body)
            expr += 'break;\n}'
        expr += '\n}'
        return expr

