# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import copy

from .ast_node import AST_Node


class AST_EndFunc(AST_Node):
    def __init__(self, context):
        AST_Node.__init__(self, context)

    def get_children(self):
        return []

    def replace_child(self, old, new):
        raise ValueError("AST_EndFunc has no children")

    def generate_code(self, sdfg, state):
        pass

    def __repr__(self):
        return "AST_EndFunc()"


class AST_Function(AST_Node):
    def __init__(self, context, name, args, retvals):
        AST_Node.__init__(self, context)
        self.name = name
        self.args = args
        self.retvals = retvals
        self.statements = None

    def __repr__(self):
        return "AST_Function(" + self.name.get_name() + ", args=[" + ", ".join(
            [str(x) for x in self.args]) + "], retvals=[" + ", ".join([str(x) for x in self.retvals]) + "])"

    def set_statements(self, stmtlist):
        self.statements = AST_Statements(None, stmtlist)
        self.statements.provide_parents(self)

    def get_children(self):
        ret = []
        ret.append(self.name)
        ret += self.args
        ret += self.retvals
        return ret

    def replace_child(self, old, new):
        if self.name == old:
            self.name = new
        elif old in self.args:
            newargs = [new if x == old else x for x in self.args]
            self.args = newargs
        elif old in self.retvals:
            newret = [new if x == old else x for x in self.retvals]
            self.retvals = newret

    def generate_code(self, sdfg, state):
        # This does not do anything, since we inline functions at the call site,
        # so the code generation happens there.
        pass

    __str__ = __repr__


class AST_Argument(AST_Node):
    def __init__(self, context, name, default=None):
        AST_Node.__init__(self, context)
        self.name = name
        self.default = default

    def get_children(self):
        ret = [self.name]
        if self.default is not None:
            ret += [self.default]
        return ret

    def __repr__(self):
        return "AST_Argument(" + self.name.get_name() + ", default=" + str(self.default) + ")"

    __str__ = __repr__


class AST_BuiltInFunCall(AST_Node):
    def __init__(self, context, funname, args):
        AST_Node.__init__(self, context)
        self.funname = funname
        self.args = args

    def __repr__(self):
        return "AST_BuiltInFunCall(" + str(self.funname) + ", " + str(self.args) + ")"

    def get_children(self):
        retval = self.args[:]
        retval.append(self.funname)
        return retval

    def replace_child(self, old, new):
        if old == self.funname:
            self.funname = new
            return
        if old in self.args:
            newargs = [new if x == old else x for x in self.args]
            self.args = newargs

    def get_basetype(self):
        # For now assume it is always double
        return dace.dtypes.float64

    def get_dims(self):
        from .ast_matrix import AST_Matrix
        dims = None
        if self.funname.get_name() in ["zeros", "ones", "rand", "eye"]:
            # The dimensions for these functions are the arguments, but we
            # need to convert them to values, if we cannot they are symbolic
            for arg in self.args:
                if not arg.is_constant():

                    return self.args
            if isinstance(self.args[0], AST_Matrix):
                dims = self.args[0].get_values_row_major()
            else:
                dims = [self.args[0].get_value(), self.args[1].get_value()]
        elif self.funname.get_name() in ["sqrt"]:
            return self.args[0].get_dims()
        elif self.funname.get_name() in ["length"]:
            dims = [1]
        if dims is None:
            raise NotImplementedError("Cannot infer dimensions for " + str(self))
        return dims

    def generate_code(self, sdfg, state):

        # TODO: rand has options for setting seed/state and controlling
        # accuracy. We only deal with the simple use-case for now.

        if self.funname.get_name() in ["sqrt"]:
            dims = self.get_dims()
            name = self.get_name_in_sdfg(sdfg)
            basetype = dace.dtypes.float64
            sdfg.add_transient(name, dims, basetype, debuginfo=self.context)
            print("The result of expr " + str(self) + " will be stored in " + str(name))

            self.args[0].generate_code(sdfg, state)

            resnode = self.get_datanode(sdfg, state)
            if len(dims) == 1:
                s = sdfg.nodes()[state]
                A = self.args[0].get_datanode(sdfg, state)
                tasklet = sdfg.nodes()[state].add_tasklet('sqrt', {'in'}, {'out'}, "out=sqrt(in);", dace.Language.CPP)
                s.add_edge(A, None, tasklet, "in", dace.memlet.Memlet.from_array(A.data, A.desc(sdfg)))
                s.add_edge(tasklet, "out", resnode, None,
                           dace.memlet.Memlet.from_array(resnode.data, resnode.desc(sdfg)))
            elif len(dims) == 2:
                M = str(dims[0])
                N = str(dims[1])

                men, mex = sdfg.nodes()[state].add_map(self.funname.get_name() + 'map', dict(i="0:" + N, j="0:" + M))
                tasklet = None
                s = sdfg.nodes()[state]
                A = self.args[0].get_datanode(sdfg, state)
                s.add_edge(A, None, men, None, dace.memlet.Memlet.from_array(A.data, A.desc(sdfg)))
                tasklet = sdfg.nodes()[state].add_tasklet('sqrt', {'in'}, {'out'}, "out=sqrt(in);", dace.Language.CPP)
                s.add_edge(men, None, tasklet, "in", dace.memlet.Memlet.simple(A, 'i,j'))
                s.add_edge(tasklet, "out", mex, None, dace.memlet.Memlet.simple(resnode, 'i,j'))
                s.add_edge(mex, None, resnode, None, dace.memlet.Memlet.simple(resnode, '0:' + N + ',0:' + M))
            else:
                raise ValueError("sqrt of tensors with more than 2 dims not supported")

        if self.funname.get_name() in ["zeros", "rand"]:
            dims = self.get_dims()
            name = self.get_name_in_sdfg(sdfg)
            basetype = dace.dtypes.float64
            sdfg.add_transient(name, dims, basetype, debuginfo=self.context)
            print("The result of expr " + str(self) + " will be stored in " + str(name))

            # Add a map over all dimensions with a tasklet that will initialize
            # the array to random values (0,1).

            if len(dims) > 2:
                raise NotImplementedError("Code generation only implemented for 2 arguments")

            resnode = self.get_datanode(sdfg, state)
            M = str(dims[0])
            N = str(dims[1])

            s = sdfg.nodes()[state]
            men, mex = s.add_map(self.funname.get_name() + 'map', dict(i="0:" + N, j="0:" + M))
            tasklet = None
            if self.funname.get_name() == "zeros":
                tasklet = sdfg.nodes()[state].add_tasklet('zero', {}, {'out'}, "out=0")
                s.add_edge(men, None, tasklet, None, dace.memlet.Memlet())
            elif self.funname.get_name() == "rand":
                tasklet = sdfg.nodes()[state].add_tasklet('rand', {}, {'out'}, "out=drand48()")
                s.add_edge(men, None, tasklet, None, dace.memlet.Memlet())
            elif self.funname.get_name() == "sqrt":
                A = self.args[0].get_datanode(sdfg, state)
                tasklet = sdfg.nodes()[state].add_tasklet('sqrt', {'in'}, {'out'}, "out=sqrt(in)")
                s.add_edge(men, None, tasklet, "in", dace.memlet.Memlet.simple(A, 'i,j'))
            else:
                raise NotImplementedError("Code generation for " + str(self.funname.get_name()) +
                                          " is not implemented.")
            s = sdfg.nodes()[state]
            s.add_edge(tasklet, "out", mex, None, dace.memlet.Memlet.simple(resnode, 'i,j'))
            s.add_edge(mex, None, resnode, None, dace.memlet.Memlet.simple(resnode, '0:' + N + ',0:' + M))

    def specialize(self):
        from .ast_matrix import AST_Matrix, AST_Matrix_Row
        from .ast_values import AST_Constant, AST_Ident

        # First try to specialize the arguments (for constant propagation)
        for c in self.get_children():
            n = c.specialize()
            while n is not None:
                n.replace_parent(c.get_parent())
                self.replace_child(old=c, new=n)
                c = n
                n = n.specialize()
        for c in self.get_children():
            if isinstance(c, AST_Ident):
                if isinstance(c.get_propagated_value(), AST_Constant):
                    n = copy.deepcopy(c.get_propagated_value())
                    self.replace_child(old=c, new=n)

        # If this is a call to zeros, ones, or eye, and the arguments are
        # constants, we can generate a constant expression. `length` is a
        # special case, since for now we require that all dimensions are
        # compile time constants.

        if self.funname.get_name() == "length":
            vardef = self.search_vardef_in_scope(self.args[0].get_name())
            if vardef is None:
                raise ValueError("No definition found for " + self.args[0].get_name())
            dims = vardef.get_dims()
            length = max(dims)
            return AST_Constant(None, length)

        if not self.funname.get_name() in ["zeros", "ones", "eye"]:
            return None

        for arg in self.args:
            if not arg.is_constant():
                return None

        # The args to those functions can be supplied as a 1x2 matrix or
        # two seperate values, the semantics are the same.
        dims = []
        if isinstance(self.args, AST_Matrix):
            dims = self.args.get_values_row_major()
        else:
            dims = [x.get_value() for x in self.args]

        rows = []
        for r in range(0, dims[0]):
            rowelems = []
            for c in range(0, dims[1]):
                zero = AST_Constant(self.context, 0)
                one = AST_Constant(self.context, 1)
                if self.funname.get_name() == "zeros":
                    rowelems.append(zero)
                if self.funname.get_name() == "ones":
                    rowelems.append(one)
                if self.funname.get_name() == "eye":
                    if r == c:
                        rowelems.append(one)
                    else:
                        rowelems.append(zero)
            rows.append(AST_Matrix_Row(self.context, rowelems))
        res = AST_Matrix(self.context, rows)
        res.provide_parents(self.get_parent())
        res.next = self.next
        res.prev = self.prev
        return res

    __str__ = __repr__


class AST_FunCall(AST_Node):
    # NOTE: When parsing, array references, i.e., A(1,2) is the same as
    #       function calls, so after parsing this node will be used for both,
    #       and we resolve this later.
    def __init__(self, context, funname, args):
        AST_Node.__init__(self, context)
        self.funname = funname
        self.args = args

    def get_children(self):
        retval = self.args[:]
        retval.append(self.funname)
        return retval

    def replace_child(self, old, new):
        if old == self.funname:
            self.funname = new
            return
        if old in self.args:
            newargs = [new if x == old else x for x in self.args]
            self.args = newargs

    def __repr__(self):
        return "AST_FunCall(" + str(self.funname) + ", " + str(self.args) + ")"

    def specialize(self):
        # This function will be called after we have the complete AST.
        # Thus we know if this is a real function call or an array access.
        # If it is a function call, differentiate between built-in functions
        # and user-defined ones.
        from .ast_arrayaccess import AST_ArrayAccess

        if self.funname.get_name() in ["zeros", "eye", "rand", "ones", "length", "sqrt"]:
            new = AST_BuiltInFunCall(self.context, self.funname, self.args)
            new.next = self.next
            new.prev = self.prev
            new.parent = self.parent
            for c in new.get_children():
                c.provide_parents(new)
            return new
        else:
            # find the definition of self.funname, if it is anything else
            # than an AST_Function this is an array subaccess
            vardef = self.search_vardef_in_scope(self.funname.get_name())
            if vardef is None:
                raise ValueError("No definition found for " + self.funname.get_name() + " searching from " + str(self))
            if isinstance(vardef, AST_Function):
                return None
            else:
                new = AST_ArrayAccess(self.context, self.funname, self.args)
                new.next = self.next
                new.prev = self.prev
                new.parent = self.parent
                for c in new.get_children():
                    c.provide_parents(new)
                return new
        return None

    __str__ = __repr__
