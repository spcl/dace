import ast
import re

import dace
from dace import symbolic
from dace.graph.graph import Edge
from dace.frontend.python import astutils
from dace.properties import Property, CodeProperty, make_properties, CodeBlock


def _assignments_from_string(astr):
    """ Returns a dictionary of assignments from a semicolon-delimited
        string of expressions. """

    result = {}
    for aitem in astr.split(';'):
        aitem = aitem.strip()
        m = re.search(r'([^=\s]+)\s*=\s*([^=]+)', aitem)
        result[m.group(1)] = m.group(2)

    return result


def _assignments_to_string(assdict):
    """ Returns a semicolon-delimited string from a dictionary of assignment
        expressions. """
    return '; '.join(['%s=%s' % (k, v) for k, v in assdict.items()])


@make_properties
class InterstateEdge(object):
    """ An SDFG state machine edge. These edges can contain a condition
        (which may include data accesses for data-dependent decisions) and
        zero or more assignments of values to inter-state variables (e.g.,
        loop iterates).
    """

    assignments = Property(
        dtype=dict,
        desc="Assignments to perform upon transition (e.g., 'x=x+1; y = 0')",
        from_string=_assignments_from_string,
        to_string=_assignments_to_string)
    condition = CodeProperty(desc="Transition condition",
                             default=CodeBlock("1"))

    def __init__(self, condition: CodeBlock = None, assignments=None):

        if condition is None:
            condition = CodeBlock("1")

        if assignments is None:
            assignments = {}

        if isinstance(condition, str):
            self.condition = CodeBlock(condition)
        elif isinstance(condition, ast.AST):
            self.condition = CodeBlock([condition])
        elif isinstance(condition, list):
            self.condition = CodeBlock(condition)
        else:
            self.condition = condition
        self.assignments = assignments

    def is_unconditional(self):
        """ Returns True if the state transition is unconditional. """
        return (self.condition is None or InterstateEdge.condition.to_string(
            self.condition).strip() == "1" or self.condition.as_string == "")

    def condition_sympy(self):
        return symbolic.pystr_to_symbolic(self.condition.as_string)

    def condition_symbols(self):
        return dace.symbolic.symbols_in_ast(self.condition.code[0])

    def to_json(self, parent=None):
        ret = {
            'type': type(self).__name__,
            'attributes': dace.serialize.all_properties_to_json(self),
            'label': self.label
        }

        return ret

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj['type'] != "InterstateEdge":
            raise TypeError("Invalid data type")

        # Create dummy object
        ret = InterstateEdge()
        dace.serialize.set_properties_from_json(ret, json_obj, context=context)

        return ret

    @property
    def label(self):
        assignments = ','.join(
            ['%s=%s' % (k, v) for k, v in self.assignments.items()])

        # Edge with assigment only (no condition)
        if self.condition.as_string == '1':
            # Edge without conditions or assignments
            if len(self.assignments) == 0:
                return ''
            return assignments

        # Edge with condition only (no assignment)
        if len(self.assignments) == 0:
            return self.condition.as_string

        # Edges with assigments and conditions
        return self.condition.as_string + '; ' + assignments

