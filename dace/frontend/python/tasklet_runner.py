# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
A Python runner for DaCe tasklets. Used to execute  ``with dace.tasklet`` 
statements.
"""

import ast
import copy
import inspect
import sys
from typing import Any, Dict, List, Optional, Tuple

from dace import data, symbolic
from dace.config import Config
from dace.frontend.python import ndloop, wrappers
from dace.frontend.python import astutils
from dace.frontend.python.astutils import unparse, rname
from dace.frontend.python.parser import DaceProgram


def get_tasklet_ast(stack_depth=2, frame=None) -> ast.With:
    """
    Returns the AST of the contents of the statement that called this function.
    :param stack_depth: If ``frame`` is None, how many levels up to go in the 
                        stack, default: 2 
                        (with statement->``__enter__``->``get_tasklet_ast``).
    :param frame: An optional argument specifying the frame, in order to avoid
                  inspecting it twice.
    :return: The AST of the body of the statement, or a FileNotFoundError if
             the AST cannot be recovered.
    """
    if frame is None:
        frame = inspect.stack()[stack_depth][0]
    caller = inspect.getframeinfo(frame)
    try:
        with open(caller.filename, 'r') as fp:
            pysrc = fp.read()
    except (OSError, FileNotFoundError):
        try:
            pysrc = inspect.getsource(frame)
        except (OSError, FileNotFoundError):
            raise FileNotFoundError('Cannot recover tasklet source code. This is likely because '
                                    'you are calling "with dace.tasklet:" from a Python terminal. '
                                    'Try to use Python code without tasklets instead, run from '
                                    'IPython, or a file.')

    module: ast.Module = ast.parse(pysrc)
    for node in ast.walk(module):
        if (getattr(node, 'lineno', -1) == caller.lineno and isinstance(node, ast.With)):
            return node

    raise FileNotFoundError('Cannot recover "with" statement from calling ' 'function.')


def _copy_location(newnode, node):
    return ast.fix_missing_locations(ast.copy_location(newnode, node))


class TaskletRewriter(astutils.ExtNodeTransformer):
    """ 
    Rewrites memlet expressions in tasklets such that the code can run in
    pure Python. This means placing all the reads before computations, writes
    after computations, handling write-conflict resolution, and dynamic memlets.
    """
    def __init__(self) -> None:
        super().__init__()
        # Used for incoming memlets
        self.pre_statements: List[ast.AST] = []
        # Used for outgoing memlets
        self.post_statements: List[ast.AST] = []
        # Used for dynamic memlets, which should be replaced within the tasklet
        self.name_replacements: Dict[str, ast.AST] = {}
        self.assign_replacements: Dict[str, ast.AST] = {}
        self.wcr_replacements: Dict[str, Tuple[ast.AST, ast.Lambda]] = {}
        self.replace: bool = False

    def clear_statements(self):
        self.pre_statements = []
        self.post_statements = []
        self.name_replacements = {}
        self.assign_replacements = {}
        self.wcr_replacements = {}
        self.replace = False

    def rewrite_tasklet(self, node: ast.With) -> ast.Module:
        self.clear_statements()

        # Visit a first time to collect and clean memlets
        newnode: ast.With = self.visit(node)

        # Visit a second time to replace dynamic memlets
        if (len(self.name_replacements) + len(self.assign_replacements) + len(self.wcr_replacements)) > 0:
            self.replace = True
            newnode: ast.With = self.visit(newnode)

        # Replace "with" statement with "if True:" and add memlet statements
        iftrue = ast.parse('if True: pass')
        iftrue.body[0] = _copy_location(iftrue.body[0], newnode)
        iftrue.body[0].body = (self.pre_statements + newnode.body + self.post_statements)

        return iftrue

    def _analyze_call(self, node: ast.Call) -> Tuple[bool, Optional[ast.Lambda]]:
        """ 
        Analyze memlet expression if a function call (e.g.
        "A(1, lambda a,b: a+b)[:]").
        :param node: The AST node representing the call.
        :return: A 2-tuple of (is memlet dynamic, write conflict resolution)
        """
        if len(node.args) < 1 or len(node.args) > 3:
            raise SyntaxError('Memlet expression must have one or two arguments:'
                              ' (memory movement volume, write conflict resolution function)')
        try:
            volume = ast.literal_eval(node.args[0])
        except ValueError:
            volume = None
        if len(node.args) >= 2:
            wcr = node.args[1]
        else:
            wcr = None

        return (volume == -1), wcr

    def _clean_memlet(
        self,
        node: ast.AST,
    ) -> Tuple[ast.AST, bool, Optional[ast.Lambda]]:
        result = node
        dynamic = False
        wcr = None
        # Clean up memlet syntax
        if isinstance(node, ast.Call):
            # If A(...), strip call and get dynamic and wcr properties
            dynamic, wcr = self._analyze_call(node)
            result = node.func
        elif (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Call)):
            # If A(...)[...], strip call from subscript value
            dynamic, wcr = self._analyze_call(node.value)
            result: ast.Subscript = node
            result.value = node.value.func

        # If memlet is just an array name, add "[:]"
        if isinstance(result, ast.Name):
            result = ast.parse(f'{result.id}[:]').body[0].value

        return result, dynamic, wcr

    # Visit and transform memlet expressions
    def visit_TopLevelExpr(self, node):
        if isinstance(node.value, ast.BinOp):
            if isinstance(node.value.op, ast.LShift):
                # Obtain memlet metadata and clean AST node from memlet syntax
                cleaned_right, dynamic, _ = self._clean_memlet(node.value.right)

                # Replace "a << A[i]" with "a = A[i]" at the beginning
                if not dynamic:
                    storenode = copy.deepcopy(node.value.left)
                    storenode.ctx = ast.Store()
                    self.pre_statements.append(
                        _copy_location(ast.Assign(targets=[storenode], value=cleaned_right), node))
                else:
                    # In-place replacement
                    self.name_replacements[rname(node.value.left)] = cleaned_right
                return None  # Remove from final tasklet code
            elif isinstance(node.value.op, ast.RShift):
                # Obtain memlet metadata and clean AST node from memlet syntax
                cleaned_right, dynamic, wcr = self._clean_memlet(node.value.right)

                # Replace "a >> A[i]" with "A[i] = a" at the end
                if not dynamic:
                    rhs = node.value.left
                    if wcr is not None:
                        # If WCR is involved, change expression to include
                        # lambda: "A[i] = (lambda a,b: a+b)(A[i], a)"
                        rhs = _copy_location(ast.Call(func=wcr, args=[cleaned_right, rhs], keywords=[]), rhs)

                    lhs = copy.deepcopy(cleaned_right)
                    lhs.ctx = ast.Store()
                    self.post_statements.append(_copy_location(ast.Assign(targets=[lhs], value=rhs), node))
                else:
                    if wcr is not None:
                        # Replace Assignments with lambda every time
                        self.wcr_replacements[rname(node.value.left)] = (cleaned_right, wcr)
                    else:
                        # In-place replacement
                        self.assign_replacements[rname(node.value.left)] = cleaned_right

                return None  # Remove from final tasklet code

        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        # Replace dynamic reads
        if self.replace and isinstance(node.ctx, ast.Load) and node.id in self.name_replacements:
            return _copy_location(copy.deepcopy(self.name_replacements[node.id]), node)
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        if not self.replace:
            return self.generic_visit(node)
        # Replace dynamic writes
        result = []
        rhs = self.visit(node.value)
        for target in node.targets:
            if (isinstance(target, ast.Name) and target.id in self.assign_replacements):
                # Replace assignment
                newtarget = copy.deepcopy(self.assign_replacements[target.id])
                newtarget.ctx = ast.Store()
                result.append(_copy_location(ast.Assign(targets=[newtarget], value=rhs), node))
            elif (isinstance(target, ast.Name) and target.id in self.wcr_replacements):
                # Replace WCR assignment
                newtarget, wcr = copy.deepcopy(self.wcr_replacements[target.id])
                new_old_rhs = copy.deepcopy(newtarget)
                newtarget.ctx = ast.Store()
                rhs = _copy_location(ast.Call(func=wcr, args=[new_old_rhs, rhs], keywords=[]), rhs)
                result.append(_copy_location(ast.Assign(targets=[newtarget], value=rhs), node))
            else:
                # Keep assignment as-is
                target = self.visit(target)
                result.append(_copy_location(ast.Assign(targets=[target], value=rhs), node))
        return result


def run_tasklet(tasklet_ast: ast.With, filename: str, gvars: Dict[str, Any], lvars: Dict[str, Any]):
    """
    Transforms and runs a tasklet given by its AST, filename, global, and local
    variables.
    :param tasklet_ast: AST of the "with dace.tasklet" statement.
    :param filename: A string representing the originating filename or code
                     snippet (IPython, Jupyter) of the tasklet, used for
                     debuggers to attach to.
    :param gvars: Globals defined at the calling point of the tasklet.
    :param lvars: Locals defined at the calling point of the tasklet.
    """
    # Transform the decorated AST into working Python code (annotated with
    # the correct line locations so that debugging works)
    runnable_ast = TaskletRewriter().rewrite_tasklet(tasklet_ast)
    mod = ast.fix_missing_locations(runnable_ast)

    # Compile the transformed AST
    codeobj = compile(mod, filename, 'exec')

    # Run tasklet
    exec(codeobj, gvars, lvars)
