""" A Python simulator for DaCe programs. Currently reads and runs Python 
    functions rather than any SDFG. """

from __future__ import print_function
import ast
import copy
from functools import wraps
import inspect
import numpy
import sys
import numpy

from dace import data, symbolic, types
from dace.config import Config
from dace.frontend.python import astparser, astnodes, astutils, ndloop, ndarray
from dace.frontend.python.astutils import unparse
from dace.frontend.python.parser import DaceProgram


def simulate(dace_program: DaceProgram, *args):
    """ Simulate a DaCe program using Python. 
        @param dace_program: A program function annotated with `@dace.program`.
        @param *args: Program arguments to pass.
    """
    pdp, modules = dace_program.generate_pdp()

    # Transform the decorated AST into working python code (annotated so
    # that debugging works)
    simulated_ast = SimulatorTransformer(pdp).visit(pdp.ast)
    mod = ast.Module(body=simulated_ast, lineno=1)
    mod = ast.fix_missing_locations(mod)

    # Compile the transformed AST
    codeobj = compile(mod, pdp.filename, 'exec')

    fname = dace_program.name

    if Config.get_bool('debugprint'):
        print("Simulating DaCe program with name", fname)

    param_symbols = {}

    if len(pdp.params) != len(args):
        raise SyntaxError('Argument number mismatch in \'' + fname +
                          '\', expecting ' + str(len(args)))

    ##################################################################
    # Disallow external variables
    # EXCEPTIONS:
    #   * The dace module ('import dace')
    #   * The math module ('import math')
    #   * Constants (types int, float, dace.int*, dace.float*)
    #   * DaCe symbols that have been defined in @dace.program args
    ##################################################################

    f_globals = {}

    # WORKAROUND: Works around a bug in CPython 2.x where True and
    # False are undefined
    f_globals['True'] = True
    f_globals['False'] = False
    ######################

    # Allow certain namespaces/modules and constants
    f_globals.update(pdp.globals)

    # Resolve symbols
    symbols = {}
    symbols.update(symbolic.getsymbols(
        args))  # from parameter values (externally defined as "dace.symbol")
    symbols.update(param_symbols)  # from parameter values (constant inputs)

    resolve = {}
    for gname, gval in f_globals.items():
        if isinstance(gval, symbolic.symbol):
            if gval.name in symbols:
                resolve[gname] = gval.get()  # Raise exception if undefined
            else:
                resolve[gname] = None  # Mark unrelated symbols for removal

    f_globals.update(resolve)

    # Remove unrelated symbols from globals
    for rk, rv in resolve.items():
        if rv is None:
            del f_globals[rk]

    # Resolve symbols in arguments as well
    newargs = tuple(symbolic.eval(a) for a in args)
    ##################################################################

    # Store parameter objects
    pdp.arrayobjs = {
        k: v
        for k, v in zip(pdp.params, newargs) if isinstance(v, ndarray.ndarray)
    }

    # Simulate f
    ################################
    # Obtain function object
    gen_module = {}
    gen_module.update(f_globals)
    exec(codeobj, gen_module)
    cfunc = gen_module[fname]

    # Run function
    result = cfunc(*newargs)
    ################################

    return result


class RangeStorage:
    """ Range storage object that is injected to the `_` variable in order to 
        determine DaCe primitive extents at runtime. """

    def __init__(self):
        self.range = []

    def __getitem__(
            self,
            key):  # Set object's range every time it is called with a range
        self.range = key
        return self


def converttype(argument, cvt_type, argname):
    """ Helper function to convert a scalar argument to its type. """
    if isinstance(argument, ndarray.ndarray):
        return argument

    # Convert type
    converted = cvt_type.type(argument)

    # Try to cast back to the original type. If the value has changed
    # (e.g., out of bounds, lost precision), raise exception
    origtype = type(argument)
    if origtype(converted) != argument:
        raise TypeError('Type conversion of argument \'' + argname +
                        '\' resulted in loss of precision, please ' +
                        'cast explicitly before calling program')

    return converted


def _copy_location(newnode, node):
    return ast.fix_missing_locations(ast.copy_location(newnode, node))


class SimulatorTransformer(ast.NodeTransformer):
    """ A Python AST transformer that converts a DaCe program into runnable
        Python code for the simulator. """

    def __init__(self, pdp):
        self.pdp = pdp
        self.curprim = None
        self.module_name = None
        self.storeOnAssignment = {}  # Mapping from local names to memlets
        self.accumOnAssignment = {}  # Mapping from local names to memlets
        self.curchild = -1

    # Visiting a DaCe primitive
    def visit_FunctionDef(self, node):
        after_nodes = []

        if self.curprim is None:
            self.curprim = self.pdp
            self.curchild = -1
            if isinstance(node.decorator_list[0], ast.Call):
                self.module_name = node.decorator_list[0].func.value.id
            else:
                self.module_name = node.decorator_list[0].value.id
            # Strip decorator
            del node.decorator_list[0]

            oldchild = self.curchild
            oldprim = self.curprim

        else:
            if len(node.decorator_list) == 0:
                return self.generic_visit(node)
            dec = node.decorator_list[0]
            if isinstance(dec, ast.Call):
                decname = astparser.rname(dec.func.attr)
            else:
                decname = astparser.rname(dec.attr)

            if decname in [
                    'map', 'async_map', 'reduce', 'async_reduce', 'consume',
                    'async_consume', 'tasklet', 'async_tasklet', 'iterate',
                    'loop', 'conditional'
            ]:
                self.curchild += 1

                oldchild = self.curchild
                oldprim = self.curprim
                self.curprim = self.curprim.children[self.curchild]
                self.curchild = -1

                if isinstance(self.curprim, astnodes._MapNode):
                    newnode = \
                        _copy_location(ast.For(target=ast.Tuple(ctx=ast.Store(),
                                                    elts=[ast.Name(id=name, ctx=ast.Store()) for name in self.curprim.params]),
                                                    iter=ast.parse('%s.ndrange(%s)' % (self.module_name, self.curprim.range.pystr())).body[0].value,
                                                    body=node.body, orelse=[]),
                                            node)
                    node = newnode
                elif isinstance(self.curprim, astnodes._ConsumeNode):
                    stream = self.curprim.stream
                    if isinstance(self.curprim.stream, ast.AST):
                        stream = unparse(self.curprim.stream)
                    if '[' not in stream:
                        stream += '[0]'

                    newnode = \
                        _copy_location(ast.While(
                            test=ast.parse('len(%s) > 0' % stream).body[0].value,
                                           body=node.body, orelse=[]),
                                       node)
                    node = newnode
                    node.body.insert(
                        0,
                        _copy_location(
                            ast.parse('%s = %s.popleft()' % (str(
                                self.curprim.params[0]), stream)).body[0],
                            node))

                elif isinstance(self.curprim, astnodes._TaskletNode):
                    # Strip decorator
                    del node.decorator_list[0]

                    newnode = \
                        _copy_location(ast.parse('if True: pass').body[0], node)
                    newnode.body = node.body
                    newnode = ast.fix_missing_locations(newnode)
                    node = newnode
                elif isinstance(self.curprim, astnodes._ReduceNode):
                    in_memlet = self.curprim.inputs['input']
                    out_memlet = self.curprim.outputs['output']
                    # Create reduction call
                    params = [unparse(p) for p in node.decorator_list[0].args]
                    params.extend([
                        unparse(kp) for kp in node.decorator_list[0].keywords
                    ])
                    reduction = ast.parse(
                        '%s.simulator.simulate_reduce(%s, %s)' %
                        (self.module_name, node.name,
                         ', '.join(params))).body[0]
                    reduction = _copy_location(reduction, node)
                    reduction = ast.increment_lineno(reduction,
                                                     len(node.body) + 1)
                    reduction = ast.fix_missing_locations(reduction)

                    # Strip decorator
                    del node.decorator_list[0]

                    after_nodes.append(reduction)
                elif isinstance(self.curprim, astnodes._IterateNode):
                    newnode = \
                        _copy_location(ast.For(target=ast.Tuple(ctx=ast.Store(),
                                                    elts=[ast.Name(id=name, ctx=ast.Store()) for name in self.curprim.params]),
                                                    iter=ast.parse('%s.ndrange(%s)' % (self.module_name, self.curprim.range.pystr())).body[0].value,
                                                    body=node.body, orelse=[]),
                                            node)
                    newnode = ast.fix_missing_locations(newnode)
                    node = newnode
                elif isinstance(self.curprim, astnodes._LoopNode):
                    newnode = \
                        _copy_location(ast.While(test=node.decorator_list[0].args[0],
                                                    body=node.body, orelse=[]),
                                            node)
                    newnode = ast.fix_missing_locations(newnode)
                    node = newnode
                else:
                    raise RuntimeError('Unimplemented primitive %s' % decname)
            else:
                return self.generic_visit(node)

        newbody = []
        end_stmts = []
        substitute_stmts = []
        # Incrementally build new body from original body
        for stmt in node.body:
            if isinstance(stmt, ast.Expr):
                res, append, prepend = self.VisitTopLevelExpr(stmt)
                if res is not None:
                    newbody.append(res)
                if append is not None:
                    end_stmts.extend(append)
                if prepend is not None:
                    substitute_stmts.extend(prepend)
            else:
                subnodes = self.visit(stmt)
                if subnodes is not None:
                    if isinstance(subnodes, list):
                        newbody.extend(subnodes)
                    else:
                        newbody.append(subnodes)
        node.body = newbody + end_stmts

        self.curchild = oldchild
        self.curprim = oldprim

        substitute_stmts.append(node)
        if len(after_nodes) > 0:
            return substitute_stmts + after_nodes
        return substitute_stmts

    def VisitTopLevelExpr(self, node):
        # DaCe memlet expression
        if isinstance(node.value, ast.BinOp):
            rhs = node.value.right
            lhs = node.value.left
            arrays = self.curprim.arrays()

            if isinstance(node.value.op, ast.LShift):
                # Dynamic access. Emit nothing and load memory on encounter
                if isinstance(rhs, ast.Call) and ast.literal_eval(
                        rhs.args[0]) == -1:
                    array_name = rhs.func.id
                    stripped_subscript = '%s[:]' % (array_name)
                    self.storeOnAssignment[node.value.left.id] = \
                        ast.parse(stripped_subscript).body[0].value
                    return None, None, None

                if isinstance(rhs, ast.Subscript) and isinstance(
                        rhs.value, ast.Call):

                    # Dynamic access. Emit nothing and load memory on encounter
                    if ast.literal_eval(rhs.value.args[0]) == -1:
                        array_name = rhs.value.func.id
                        stripped_subscript = '%s[%s]' % (array_name,
                                                         unparse(rhs.slice))
                        self.storeOnAssignment[node.value.left.id] = \
                            ast.parse(stripped_subscript).body[0].value
                        return None, None, None

                    rhs = ast.Subscript(
                        value=rhs.value.func, ctx=ast.Load(), slice=rhs.slice)

                result = _copy_location(
                    ast.Assign(targets=[node.value.left], value=rhs), node)
                result.targets[0].ctx = ast.Store()
                return result, None, None
            # END of "a << b"
            elif isinstance(node.value.op, ast.RShift):
                # If the memlet refers to a sub-array (view), also add an expression to initialize it
                init_expr = None
                result = None
                prefix = []

                if isinstance(rhs, ast.Subscript):
                    # Index subscript expression ("tmp >> b(1, sum)[i,j,k,l]")
                    if isinstance(rhs.value, ast.Call):
                        # Only match expressions with possible write-conflict resolution, such as "A(...)[...]"
                        array_name = rhs.value.func.id
                        stripped_subscript = '%s[%s]' % (array_name,
                                                         unparse(rhs.slice))

                        # WCR initialization with identity value
                        if len(rhs.value.args) >= 3:
                            prefix.append(
                                _copy_location(
                                    ast.parse(
                                        '%s = %s' %
                                        (stripped_subscript,
                                         unparse(rhs.value.args[2]))).body[0],
                                    node))

                        # Dynamic access. Emit nothing and store memory on assignment
                        if ast.literal_eval(rhs.value.args[0]) == -1:
                            if len(rhs.value.args) >= 2:
                                self.accumOnAssignment[node.value.left.id] = \
                                    (stripped_subscript, rhs.value.args[1])
                            else:
                                self.storeOnAssignment[node.value.left.id] = \
                                    ast.parse(stripped_subscript).body[0].value
                            return init_expr, None, prefix

                        # Make sure WCR function exists
                        if len(rhs.value.args) >= 2:
                            result = ast.parse(
                                '%s = (%s)(%s, %s)' %
                                (stripped_subscript, unparse(
                                    rhs.value.args[1]), stripped_subscript,
                                 node.value.left.id)).body[0]
                            result = _copy_location(result, node)
                        else:
                            result = ast.parse(
                                '%s = %s' % (stripped_subscript,
                                             node.value.left.id)).body[0]
                            result = _copy_location(result, node)
                    else:
                        array_name = rhs.value.id

                    if not isinstance(rhs.slice, ast.Index):
                        init_expr = _copy_location(
                            ast.Assign(
                                targets=[
                                    ast.Name(
                                        id=node.value.left.id, ctx=ast.Store())
                                ],
                                value=ast.Subscript(
                                    value=ast.Name(
                                        id=array_name, ctx=ast.Load()),
                                    slice=rhs.slice,
                                    ctx=ast.Load())), node)
                elif not isinstance(rhs, ast.Subscript):
                    if isinstance(rhs, ast.Call):
                        array_name = rhs.func
                    else:
                        array_name = rhs

                    lhs_name = lhs.id

                    # In case of "tmp >> array", write "array[:]"
                    if node.value.left.id in self.curprim.transients:
                        init_expr = None
                    # If reading from a single stream ("b << stream")
                    elif (array_name.id in arrays
                          and isinstance(arrays[array_name.id], data.Stream)):
                        if arrays[array_name.id].shape == [1]:
                            init_expr = _copy_location(
                                ast.parse('{v} = {q}[0]'.format(
                                    v=lhs_name, q=array_name.id)).body[0],
                                node)
                        return init_expr, None, []
                    else:
                        init_expr = _copy_location(
                            ast.Assign(
                                targets=[
                                    ast.Name(id=lhs_name, ctx=ast.Store())
                                ],
                                value=ast.Subscript(
                                    value=ast.Name(
                                        id=array_name.id, ctx=ast.Load()),
                                    slice=ast.Slice(
                                        lower=None, upper=None, step=None),
                                    ctx=ast.Load())), node)

                    # If we are setting a stream's sink
                    if lhs_name in arrays and isinstance(
                            arrays[lhs_name], data.Stream):
                        result = ast.parse(
                            '{arr}[0:len({q}[0])] = list({q}[0])'.format(
                                arr=rhs.id, q=lhs.id)).body[0]
                        result = _copy_location(result, node)

                    # If WCR function exists
                    elif isinstance(rhs, ast.Call) and len(rhs.args) >= 2:
                        # WCR initialization with identity value
                        if len(rhs.args) >= 3:
                            prefix.append(
                                _copy_location(
                                    ast.parse('%s[:] = %s' %
                                              (array_name.id,
                                               unparse(rhs.args[2]))).body[0],
                                    node))

                        # Dynamic access. Emit nothing and store memory on assignment
                        if ast.literal_eval(rhs.args[0]) == -1:
                            self.accumOnAssignment[lhs.id] = (array_name.id,
                                                              rhs.args[1])
                            return init_expr, None, prefix

                        result = ast.parse(
                            '%s[:] = (%s)(%s[:], %s)' %
                            (array_name.id, unparse(rhs.args[1]),
                             array_name.id, node.value.left.id)).body[0]
                        result = _copy_location(result, node)

                    else:
                        result = _copy_location(
                            ast.Assign(
                                targets=[
                                    ast.Subscript(
                                        value=ast.Name(
                                            id=array_name.id, ctx=ast.Load()),
                                        slice=ast.Slice(
                                            lower=None, upper=None, step=None),
                                        ctx=ast.Store())
                                ],
                                value=node.value.left), node)

                if result is None:
                    result = _copy_location(
                        ast.Assign(
                            targets=[node.value.right], value=node.value.left),
                        node)
                result.targets[0].ctx = ast.Store()
                return init_expr, [result], prefix
            # END of "a >> b"

        return self.generic_visit(node), [], None

    def visit_Name(self, node):
        if node.id in self.storeOnAssignment:
            subscript = self.storeOnAssignment[node.id]
            newnode = copy.deepcopy(subscript)
            newnode.ctx = node.ctx
            return _copy_location(newnode, node)

        return self.generic_visit(node)

    def visit_Assign(self, node):
        if astutils.rname(node.targets[0]) in self.accumOnAssignment:
            var_name = astutils.rname(node.targets[0])
            array_name, accum = self.accumOnAssignment[var_name]
            if isinstance(node.targets[0], ast.Subscript):
                array_name += '[' + unparse(node.targets[0].slice) + ']'
            if '[' not in array_name:
                array_name += '[:]'

            newnode = ast.parse('{out} = {accum}({out}, {val})'.format(
                out=array_name, accum=unparse(accum),
                val=unparse(node.value))).body[0]
            newnode = _copy_location(newnode, node)
            return newnode

        return self.generic_visit(node)

    def visit_Call(self, node):
        if '.push' in astutils.rname(node.func):
            node.func.attr = 'append'
        return self.generic_visit(node)

    # Control flow: for-loop is the same as dace.iterate in the right context
    def visit_For(self, node):
        if not isinstance(self.curprim, astnodes._DataFlowNode):
            self.curchild += 1

            oldchild = self.curchild
            oldprim = self.curprim
            self.curprim = self.curprim.children[self.curchild]
            self.curchild = -1

            newbody = []
            end_stmts = []
            substitute_stmts = []
            # Incrementally build new body from original body
            for stmt in node.body:
                if isinstance(stmt, ast.Expr):
                    res, append, prepend = self.VisitTopLevelExpr(stmt)
                    if res is not None:
                        newbody.append(res)
                    if append is not None:
                        end_stmts.extend(append)
                    if prepend is not None:
                        substitute_stmts.extend(prepend)
                else:
                    subnodes = self.visit(stmt)
                    if subnodes is not None:
                        if isinstance(subnodes, list):
                            newbody.extend(subnodes)
                        else:
                            newbody.append(subnodes)
            node.body = newbody + end_stmts
            substitute_stmts.append(node)

            self.curchild = oldchild
            self.curprim = oldprim
            return substitute_stmts
        return self.generic_visit(node)

    # Control flow: while-loop is the same as dace.loop in the right context
    def visit_While(self, node):
        return self.visit_For(node)

    # Control flow: if-condition is the same as dace.conditional in the right context
    def visit_If(self, node):
        if not isinstance(self.curprim, astnodes._DataFlowNode):
            self.curchild += 1

            oldchild = self.curchild
            oldprim = self.curprim
            self.curprim = self.curprim.children[self.curchild]
            self.curchild = -1

            newbody = []
            end_stmts = []
            substitute_stmts = []
            # Incrementally build new body from original body
            for stmt in node.body:
                if isinstance(stmt, ast.Expr):
                    res, append, prepend = self.VisitTopLevelExpr(stmt)
                    if res is not None:
                        newbody.append(res)
                    if append is not None:
                        end_stmts.extend(append)
                    if prepend is not None:
                        substitute_stmts.extend(prepend)
                else:
                    subnodes = self.visit(stmt)
                    if subnodes is not None:
                        if isinstance(subnodes, list):
                            newbody.extend(subnodes)
                        else:
                            newbody.append(subnodes)
            node.body = newbody + end_stmts

            self.curchild = oldchild
            self.curprim = oldprim

            # Process 'else'/'elif' statements
            if len(node.orelse) > 0:
                self.curchild += 1

                oldchild = self.curchild
                oldprim = self.curprim
                self.curprim = self.curprim.children[self.curchild]
                self.curchild = -1

                newbody = []
                end_stmts = []
                # Incrementally build new body from original body
                for stmt in node.orelse:
                    if isinstance(stmt, ast.Expr):
                        res, append, prepend = self.VisitTopLevelExpr(stmt)
                        if res is not None:
                            newbody.append(res)
                        if append is not None:
                            end_stmts.extend(append)
                        if prepend is not None:
                            substitute_stmts.extend(prepend)
                    else:
                        subnodes = self.visit(stmt)
                        if subnodes is not None:
                            if isinstance(subnodes, list):
                                newbody.extend(subnodes)
                            else:
                                newbody.append(subnodes)
                node.orelse = newbody + end_stmts

                self.curchild = oldchild
                self.curprim = oldprim

            substitute_stmts.append(node)
            return substitute_stmts

        return self.generic_visit(node)


def simulate_reduce(op, in_array, out_array, axis=None, identity=None):
    inshape = numpy.shape(in_array)
    outshape = numpy.shape(out_array)

    # Argument validation
    if axis is None and (len(outshape) != 1 or outshape[0] != 1):
        raise RuntimeError("Cannot reduce to non-scalar value")
    if axis is not None and (axis < 0 or axis >= len(in_array.shape)):
        raise RuntimeError("Cannot reduce in nonexistent axis " + str(axis))

        unreduced = outshape[:axis] + (inshape[axis], ) + outshape[axis:]
        if unreduced != inshape:
            raise RuntimeError("Incompatible shapes in reduction: " +
                               str(inshape) + " -> " + str(outshape))
    # End of argument validation

    # Reduce everything
    if axis is None:
        storevalue = True

        # If we have an initial value to insert
        if identity is not None:
            out_array[0] = identity
            storevalue = False

        for i in numpy.nditer(in_array):
            if storevalue:  # If no identity value given, store first value as output
                out_array[0] = i
                storevalue = False
            else:
                out_array[0] = op(out_array[0], i)

    else:  # Reduce a single axis
        storevalue = True

        # If we have an initial value to insert
        if identity is not None:
            # Store identity scalar in output array
            out_array[:] = identity
            storevalue = False

        # Determine reduction slice (A[:,:,...,:,i,:,...,:])
        red_slice = [slice(None, None, None) for i in inshape]
        for i in ndloop.xxrange(inshape[axis]):
            red_slice[axis] = slice(i, i + 1, None)

            inslice = in_array[red_slice]

            if storevalue:
                # Store initial value
                for arrout, arrin in zip(
                        numpy.nditer(out_array, op_flags=['readwrite']),
                        numpy.nditer(inslice)):
                    arrout[...] = arrin
                storevalue = False
            else:
                # Reduce entire (N-1)-dimensional tensor for the given slice
                for arrout, arrin in zip(
                        numpy.nditer(out_array, op_flags=['readwrite']),
                        numpy.nditer(inslice)):
                    arrout[...] = op(arrout, arrin)
