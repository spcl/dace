# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from .ast_node import AST_Node
from .ast_values import AST_Constant

import dace


class AST_Matrix_Row(AST_Node):
    def __init__(self, context, elements):
        AST_Node.__init__(self, context)
        self.elements = elements
        if not isinstance(self.elements, list):
            raise ValueError(
                "AST_Matrix_Row() expects a list of elements, got " +
                str(type(self.elements)))

    def provide_parents(self, parent):
        self.parent = parent
        for e in self.elements:
            e.provide_parents(self)

    def __repr__(self):
        return "AST_MatrixRow(" + ", ".join([str(i)
                                             for i in self.elements]) + ")"

    def get_dims(self):
        return len(self.elements)

    def get_children(self):
        return self.elements[:]

    def replace_child(self, old, new):
        newelems = [new if x == old else x for x in self.elements]
        self.elements = newelems

    def is_constant(self):
        for r in self.elements:
            if not isinstance(r, AST_Constant):
                return False
            return True

    def __getitem__(self, item):
        if item >= len(self):
            raise IndexError("AST_Matrix_Row index out of range")
        return self.elements[item]

    def __len__(self):
        return len(self.elements)

    __str__ = __repr__


class AST_Matrix(AST_Node):
    def __init__(self, context, rows):
        AST_Node.__init__(self, context)
        self.rows = rows
        self.children = self.rows
        if not isinstance(self.rows, list):
            raise ValueError("AST_Matrix() expects a list of rows, got " +
                             str(type(self.rows)))
        for r in self.rows:
            if not isinstance(r, AST_Matrix_Row):
                raise ValueError("AST_Matrix() expects a list of rows, got " +
                                 str(r) + " of type " + str(type(r)))

    def __repr__(self):
        return "AST_Matrix(" + ", ".join([str(i) for i in self.rows]) + ")"

    def provide_parents(self, parent):
        self.parent = parent
        for e in self.rows:
            e.provide_parents(self)

    def get_dims(self):
        dims = -1
        for r in self.rows:
            if (dims > 0) and (r.get_dims() != dims):
                raise ValueError(
                    "Matrices with unequal row lengths are currently not "
                    "supported.")
            else:
                dims = r.get_dims()
        return [len(self.rows), dims]

    def get_basetype(self):
        # This should be double, unless we have a complex inside, for now just
        # return double.
        return dace.dtypes.float64

    def is_constant(self):
        for r in self.rows:
            if not r.is_constant():
                return False
        return True

    def get_values_row_major(self):
        values = []
        for r in self.rows:
            for c in r:
                if isinstance(c, AST_Constant):
                    values.append(c.get_value())
                else:
                    values.append(0)
        return values

    def generate_code(self, sdfg, state):
        if self.is_constant():
            name = self.get_name_in_sdfg(sdfg)
            dims = self.get_dims()
            basetype = self.get_basetype()
            sdfg.add_transient(name, dims, basetype)
            trans = sdfg.nodes()[state].add_access(name)
            # Add map over dims, and a taklet which puts the values into the
            # transient.
            arrlen = 1
            for d in dims:
                arrlen *= d
            vals = self.get_values_row_major()
            code = "constexpr double VALUES[" + str(arrlen) + "] = {"
            code += ", ".join(str(i) for i in vals) + "};\n"
            code += "out[i] = VALUES[i];"

            tasklet = sdfg.nodes()[state].add_tasklet('init', {}, {'out'}, code,
                                                      dace.Language.CPP)
            me, mx = sdfg.nodes()[state].add_map('init',
                                                 dict(i='0:' + str(arrlen)))
            sdfg.nodes()[state].add_edge(me, None, tasklet, None,
                                         dace.memlet.Memlet())
            sdfg.nodes()[state].add_edge(
                tasklet, "out", mx, None,
                dace.memlet.Memlet.from_array(trans.data, trans.desc(sdfg)))
            sdfg.nodes()[state].add_edge(
                mx, None, trans, None,
                dace.memlet.Memlet.from_array(trans.data, trans.desc(sdfg)))

            print("The const expr " + str(self) + " will be stored in " +
                  str(name) + ", values are: " +
                  str(self.get_values_row_major()))
        else:
            raise ValueError(
                "Non-constant matrices are currently not supported")

    def get_children(self):
        return self.rows[:]

    def replace_child(self, old, new):
        newrows = [new if x == old else x for x in self.rows]
        self.rows = newrows

    __str__ = __repr__


class AST_Transpose(AST_Node):
    def __init__(self, context, arg, op):
        AST_Node.__init__(self, context)
        self.arg = arg
        self.op = op

    def __repr__(self):
        return "AST_Transpose(" + str(self.arg) + ", " + str(self.op) + ")"

    def get_children(self):
        return [self.arg]

    def get_dims(self):
        dims = self.arg.get_dims()
        return dims[::-1]

    def get_basetype(self):
        return self.arg.get_basetype()

    def generate_code(self, sdfg, state):
        dims = self.get_dims()
        name = self.get_name_in_sdfg(sdfg)
        basetype = self.get_basetype()
        if basetype.is_complex():
            raise NotImplementedError(
                "Transpose of complex matrices not implemented (we might need "
                "to conjugate)")
        if len(dims) != 2:
            raise NotImplementedError(
                "Transpose only implemented for 2D matrices")
        sdfg.add_transient(name, dims, basetype, debuginfo=self.context)

        resnode = self.get_datanode(sdfg, state)
        self.arg.generate_code(sdfg, state)
        A = self.arg.get_datanode(sdfg, state)

        N = str(dims[0])
        M = str(dims[1])
        s = sdfg.nodes()[state]
        map_entry, map_exit = s.add_map('transpose', dict(i='0:' + N,
                                                          j='0:' + M))
        map_entry.add_in_connector('IN_1')
        map_entry.add_out_connector('OUT_1')
        s.add_edge(A, None, map_entry, 'IN_1',
                   dace.memlet.Memlet.simple(A, '0:' + N + ',0:' + M))
        tasklet = s.add_tasklet('identity', {'a'}, {'out'}, 'out = a')
        s.add_edge(map_entry, "OUT_1", tasklet, "a",
                   dace.memlet.Memlet.simple(A, 'i,j'))
        s.add_edge(tasklet, "out", map_exit, None,
                   dace.memlet.Memlet.simple(resnode, 'j,i'))
        s.add_edge(map_exit, None, resnode, None,
                   dace.memlet.Memlet.simple(resnode, '0:' + M + ', 0:' + N))
        print("The result of expr " + str(self) + " will be stored in " +
              str(name))

    def replace_child(self, old, new):
        if old == self.arg:
            self.arg = new
            return
        raise ValueError("The child " + str(old) + " is not a child of " +
                         str(self))

    __str__ = __repr__
