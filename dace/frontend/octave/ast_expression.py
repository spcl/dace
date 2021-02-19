# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace

from .ast_node import AST_Node

_OP_TO_STRING = {
    '+': 'plus',
    '-': 'minus',
    '*': 'emult',
    '/': 'ediv',
    '%': 'mod'
}


class AST_UnaryExpression(AST_Node):
    def __init__(self, context, arg, op, order):
        AST_Node.__init__(self, context)
        self.arg = arg
        self.op = op
        self.order = order  # can be "pre" or "post" (++A vs A++)
        self.children = [self.arg]

    def __repr__(self):
        return "AST_UnaryExpression(" + str(self.arg) + ", " + str(self.op) + \
                ", " + str(self.order) + ")"

    def get_children(self):
        return [self.arg]

    def replace_child(self, old, new):
        if self.arg == old:
            self.arg = new
        else:
            raise ValueError(str(old) + " is not a child of " + str(self))

    def specialize(self):
        from .ast_values import AST_Constant
        # -A is syntactic sugar for -1*A
        if (self.op == "-") and isinstance(self.arg, AST_Constant):
            new = AST_Constant(self.context, -self.arg.get_value())
            new.next = self.next
            new.prev = self.prev
            new.parent = self.parent
            return new
        elif (self.op == "-"):
            new = AST_BinExpression(self.context, self.arg,
                                    AST_Constant(None, -1), "*")
            new.next = self.next
            new.prev = self.prev
            new.parent = self.parent
            return new

    __str__ = __repr__


class AST_BinExpression(AST_Node):
    def __init__(self, context, lhs, rhs, op):
        AST_Node.__init__(self, context)
        self.lhs = lhs
        self.rhs = rhs
        self.op = op
        self.children = [self.lhs, self.rhs]

    def provide_parents(self, parent):
        self.parent = parent
        self.lhs.provide_parents(self)
        self.rhs.provide_parents(self)

    def get_children(self):
        return [self.lhs, self.rhs]

    def replace_child(self, old, new):
        if self.lhs == old:
            self.lhs = new
        if self.rhs == old:
            self.rhs = new

    def __repr__(self):
        return "AST_BinExpression(" + str(self.lhs) + ", " + str(
            self.op) + ", " + str(self.rhs) + ")"

    def get_dims(self):
        left_dims = self.lhs.get_dims()
        right_dims = self.rhs.get_dims()
        if len(left_dims) > 2 or len(right_dims) > 2:
            raise ValueError("Only 2D matrices can be multiplied")
        outdims = None
        if self.op == "*":
            # if lhs is a scalar, outdims = rhs
            if left_dims == [1]:
                outdims = right_dims
                # elif rhs is a scalar, outdims = lhs
            elif right_dims == [1]:
                outdims = left_dims
                # elif lhs is a matrix, check if dims match, compute new outdims
            elif left_dims[1] != right_dims[0]:
                print(str(left_dims) + "type: " + str(type(left_dims[1])))
                print(str(right_dims) + "type: " + str(type(right_dims[0])))
                raise ValueError("Dims do not match!")
            else:
                outdims = [left_dims[0], right_dims[1]]
        elif self.op == "+" or self.op == "-" or self.op == "/":
            # if lhs is a scalar, outdims = rhs
            if left_dims == [1]:
                outdims = right_dims
                # elif rhs is a scalar, outdims = lhs
            elif right_dims == [1]:
                outdims = left_dims
                # elif lhs is a matrix, check if dims match, compute new outdims
            elif left_dims != right_dims:
                raise ValueError("Dimensions do not match")
            else:
                outdims = left_dims
        else:
            raise NotImplementedError("Unhandled binary operator: " +
                                      str(self.op))
        if outdims == [1, 1]:
            outdims = [1]
        return outdims

    def get_basetype(self):
        # The basetype of a binary expression should be the more accurate
        # type of lhs and rhs
        return dace.dtypes.float64

    def matrix2d_scalar(self, sdfg, state, op):
        lhs_dims = self.lhs.get_dims()
        rhs_dims = self.rhs.get_dims()
        M = str(lhs_dims[-2])
        N = str(lhs_dims[-1])
        A = self.lhs.get_datanode(sdfg, state)
        B = self.rhs.get_datanode(sdfg, state)
        C = self.get_datanode(sdfg, state)

        s = sdfg.nodes()[state]
        map_entry, map_exit = s.add_map('M' + _OP_TO_STRING[op] + 'M',
                                        dict(i='0:' + M, j='0:' + N))
        map_entry.add_in_connector('IN_1')
        map_entry.add_in_connector('IN_2')
        map_entry.add_out_connector('OUT_1')
        map_entry.add_out_connector('OUT_2')
        s.add_edge(A, None, map_entry, 'IN_1',
                   dace.memlet.Memlet.simple(A, '0:' + N + ',0:' + M))
        s.add_edge(B, None, map_entry, 'IN_2',
                   dace.memlet.Memlet.simple(B, '0'))
        tasklet = s.add_tasklet(_OP_TO_STRING[op], {'a', 'b'}, {'c'},
                                'c = a' + op + 'b')
        s.add_edge(map_entry, "OUT_1", tasklet, "a",
                   dace.memlet.Memlet.simple(A, 'i,j'))
        s.add_edge(map_entry, "OUT_2", tasklet, "b",
                   dace.memlet.Memlet.simple(B, '0'))
        s.add_edge(tasklet, "c", map_exit, None,
                   dace.memlet.Memlet.simple(C, 'i,j'))
        s.add_edge(map_exit, None, C, None,
                   dace.memlet.Memlet.simple(C, '0:' + N + ', 0:' + M))

    def matrix2d_matrix2d_mult(self, sdfg, state):
        lhs_dims = self.lhs.get_dims()
        rhs_dims = self.rhs.get_dims()
        A = self.lhs.get_datanode(sdfg, state)
        B = self.rhs.get_datanode(sdfg, state)
        C = self.get_datanode(sdfg, state)

        M = str(lhs_dims[-1])
        N = str(lhs_dims[-1])
        K = str(rhs_dims[-1])

        s = sdfg.nodes()[state]
        map_entry, map_exit = s.add_map(
            'MMM', dict(i='0:' + M, j='0:' + N, k='0:' + K))
        map_entry.add_in_connector('IN_1')
        map_entry.add_in_connector('IN_2')
        map_entry.add_out_connector('OUT_1')
        map_entry.add_out_connector('OUT_2')
        s.add_edge(A, None, map_entry, 'IN_1',
                   dace.memlet.Memlet.simple(A, '0:' + M + ',0:' + K))
        s.add_edge(B, None, map_entry, 'IN_2',
                   dace.memlet.Memlet.simple(B, '0:' + K + ', 0:' + N))
        tasklet = s.add_tasklet('mult', {'a', 'b'}, {'c'}, 'c = a*b')
        s.add_edge(map_entry, "OUT_1", tasklet, "a",
                   dace.memlet.Memlet.simple(A, 'i,k'))
        s.add_edge(map_entry, "OUT_2", tasklet, "b",
                   dace.memlet.Memlet.simple(B, 'k,j'))
        tmpname = self.get_new_tmpvar(sdfg)
        sdfg.add_transient(tmpname, [M, N, K], self.get_basetype())
        tmp = s.add_access(tmpname)
        s.add_edge(tasklet, "c", map_exit, None,
                   dace.memlet.Memlet.simple(tmp, 'i,j,k'))
        rednode = s.add_reduce('lambda a,b: a+b', (2, ), 0)
        s.add_edge(
            map_exit, None, tmp, None,
            dace.memlet.Memlet.simple(tmp, '0:' + M + ',0:' + N + ',0:' + K))
        s.add_edge(
            tmp, None, rednode, None,
            dace.memlet.Memlet.simple(tmp, '0:' + M + ',0:' + N + ',0:' + K))
        s.add_edge(rednode, None, C, None,
                   dace.memlet.Memlet.simple(C, '0:' + M + ',0:' + N))

    def vec_mult_vect(self, sdfg, state, op):
        lhs_dims = self.lhs.get_dims()
        rhs_dims = self.rhs.get_dims()
        A = self.lhs.get_datanode(sdfg, state)
        B = self.rhs.get_datanode(sdfg, state)
        C = self.get_datanode(sdfg, state)

        N = str(lhs_dims[-1])

        s = sdfg.nodes()[state]
        map_entry, map_exit = s.add_map('VVM', dict(i='0:' + N))
        map_entry.add_in_connector('IN_1')
        map_entry.add_in_connector('IN_2')
        map_entry.add_out_connector('OUT_1')
        map_entry.add_out_connector('OUT_2')
        s.add_edge(A, None, map_entry, 'IN_1',
                   dace.memlet.Memlet.simple(A, '0:' + N))
        s.add_edge(B, None, map_entry, 'IN_2',
                   dace.memlet.Memlet.simple(B, '0:' + N))
        tasklet = s.add_tasklet('mult', {'a', 'b'}, {'c'}, 'c = a*b')
        s.add_edge(map_entry, "OUT_1", tasklet, "a",
                   dace.memlet.Memlet.simple(A, '0,i'))
        s.add_edge(map_entry, "OUT_2", tasklet, "b",
                   dace.memlet.Memlet.simple(B, 'i,0'))
        tmpname = self.get_new_tmpvar(sdfg)
        sdfg.add_transient(tmpname, [N], self.get_basetype())
        tmp = s.add_access(tmpname)
        s.add_edge(tasklet, "c", map_exit, None,
                   dace.memlet.Memlet.simple(tmp, 'i'))
        rednode = s.add_reduce('lambda a,b: a+b', (0, ), 0)
        s.add_edge(map_exit, None, tmp, None,
                   dace.memlet.Memlet.simple(tmp, '0:' + N))
        s.add_edge(tmp, None, rednode, None,
                   dace.memlet.Memlet.simple(tmp, '0:' + N))
        s.add_edge(rednode, None, C, None, dace.memlet.Memlet.simple(C, '0'))

    def matrix2d_matrix2d_plus_or_minus(self, sdfg, state, op):
        lhs_dims = self.lhs.get_dims()
        rhs_dims = self.rhs.get_dims()
        M = str(lhs_dims[-2])
        N = str(lhs_dims[-1])
        A = self.lhs.get_datanode(sdfg, state)
        B = self.rhs.get_datanode(sdfg, state)
        C = self.get_datanode(sdfg, state)

        s = sdfg.nodes()[state]
        map_entry, map_exit = s.add_map('M' + _OP_TO_STRING[op] + 'M',
                                        dict(i='0:' + M, j='0:' + N))
        map_entry.add_in_connector('IN_1')
        map_entry.add_in_connector('IN_2')
        map_entry.add_out_connector('OUT_1')
        map_entry.add_out_connector('OUT_2')
        s.add_edge(A, None, map_entry, 'IN_1',
                   dace.memlet.Memlet.simple(A, '0:' + N + ',0:' + M))
        s.add_edge(B, None, map_entry, 'IN_2',
                   dace.memlet.Memlet.simple(B, '0:' + N + ', 0:' + M))
        tasklet = s.add_tasklet(_OP_TO_STRING[op], {'a', 'b'}, {'c'},
                                'c = a' + op + 'b')
        s.add_edge(map_entry, "OUT_1", tasklet, "a",
                   dace.memlet.Memlet.simple(A, 'i,j'))
        s.add_edge(map_entry, "OUT_2", tasklet, "b",
                   dace.memlet.Memlet.simple(B, 'i,j'))
        s.add_edge(tasklet, "c", map_exit, None,
                   dace.memlet.Memlet.simple(C, 'i,j'))
        s.add_edge(map_exit, None, C, None,
                   dace.memlet.Memlet.simple(C, '0:' + N + ', 0:' + M))

    def scalar_scalar(self, sdfg, state, op):
        A = self.lhs.get_datanode(sdfg, state)
        B = self.rhs.get_datanode(sdfg, state)
        C = self.get_datanode(sdfg, state)

        s = sdfg.nodes()[state]
        tasklet = s.add_tasklet(_OP_TO_STRING[op], {'a', 'b'}, {'c'},
                                'c = a' + op + 'b')
        s.add_edge(A, None, tasklet, 'a', dace.memlet.Memlet.simple(A, '0'))
        s.add_edge(B, None, tasklet, 'b', dace.memlet.Memlet.simple(B, '0'))
        s.add_edge(tasklet, "c", C, None, dace.memlet.Memlet.simple(C, '0'))

    def generate_code(self, sdfg, state):
        # Generate code for the lhs and rhs
        self.lhs.generate_code(sdfg, state)
        self.rhs.generate_code(sdfg, state)

        # Add a new variable to hold the result of this expression
        dims = self.get_dims()
        basetype = self.get_basetype()
        name = self.get_name_in_sdfg(sdfg)
        sdfg.add_transient(name, dims, basetype, debuginfo=self.context)
        print("The result of " + str(self) + " will be stored in " + str(name))

        lhs_dims = self.lhs.get_dims()
        rhs_dims = self.rhs.get_dims()

        if rhs_dims == [1, 1] or rhs_dims == [1]:
            if lhs_dims == [1, 1] or lhs_dims == [1]:
                self.scalar_scalar(sdfg, state, self.op)
            else:
                self.matrix2d_scalar(sdfg, state, self.op)
            return
        if lhs_dims[0] == 1 and rhs_dims[1] == 1 and self.op == "*":
            self.vec_mult_vect(sdfg, state, self.op)
        elif lhs_dims == [1, 1] or lhs_dims == [1]:
            raise NotImplementedError(
                "Binary expression with scalar on lhs not implemented: " +
                str(self) + ", lhs dims: " + str(lhs_dims) + ", rhs dims: " +
                str(rhs_dims))
        else:
            if self.op == "*":
                self.matrix2d_matrix2d_mult(sdfg, state)
            elif self.op == "-" or self.op == "+":
                self.matrix2d_matrix2d_plus_or_minus(sdfg, state, self.op)
            else:
                raise NotImplementedError("Binary expression with two " +
                                          "matrices and op=" + str(self.op) +
                                          " not implemented")

    __str__ = __repr__
