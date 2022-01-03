# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from .ast_node import AST_Node
from .ast_values import AST_Ident

import dace


class AST_Assign(AST_Node):
    def __init__(self, context, lhs, rhs, op):
        # for a normal assignment op is "=", but there is also
        # in place modification, i.e., "+="
        AST_Node.__init__(self, context)
        self.lhs = lhs
        self.rhs = rhs
        self.op = op
        self.children = [self.lhs, self.rhs]

    def get_children(self):
        retval = [self.lhs, self.rhs]
        return retval

    def replace_child(self, old, new):
        if old == self.lhs:
            self.lhs = new
        if old == self.rhs:
            self.rhs = new

    def defined_variables(self):
        # check if this adds something to the scope, if yes add it.
        # assume A is undefined before this node, then:
        # A = expr defines A, A(5) = expr defines A, but
        # A += expr or A(5) += expr is illegal.
        if self.op == "=":
            if isinstance(self.lhs, AST_Ident):
                return [self.lhs.get_name()]
            else:
                return []

    def provide_parents(self, parent):
        self.parent = parent
        self.lhs.provide_parents(self)
        self.rhs.provide_parents(self)

    def __repr__(self):
        return "AST_Assign(" + str(self.lhs) + ", " + str(self.op) + ", " + str(self.rhs) + ")"

    def print_nodes(self, state):
        for n in state.nodes():
            print(str(n))
        print("---")

    def generate_code(self, sdfg, state):
        from .ast_arrayaccess import AST_ArrayAccess
        from .ast_values import AST_Constant
        from .ast_loop import AST_ForLoop

        self.rhs.generate_code(sdfg, state)
        s = sdfg.nodes()[state]
        if self.op == "=":
            # We assign to an entire array
            if isinstance(self.lhs, AST_Ident):
                dims = self.rhs.get_dims()
                basetype = self.rhs.get_basetype()
                name = self.lhs.get_name()

                if name not in sdfg.arrays:
                    sdfg.add_array(name, dims, basetype, debuginfo=self.context)
                rhs_datanode = self.rhs.get_datanode(sdfg, state)
                lhs_datanode = self.lhs.get_datanode(sdfg, state)

                s.add_edge(rhs_datanode, None, lhs_datanode, None,
                           dace.memlet.Memlet.from_array(lhs_datanode.data, lhs_datanode.desc(sdfg)))

            # We assign only to a part of an (existing) array, in order to not
            # create cycles we need to add a new data-node, the add_array()
            # interface will make sure it is connected to the same memory than
            # the existing array node.
            elif isinstance(self.lhs, AST_ArrayAccess):
                # get the definition of the array we are assigning to
                lhs_data = self.lhs.arrayname.get_datanode(sdfg, state)
                vardef = self.search_vardef_in_scope(self.lhs.arrayname.get_name())
                if vardef is None:
                    raise ValueError("No definition found for " + self.lhs.arrayname.get_name() + " searching from " +
                                     str(self))
                dims = vardef.get_dims()
                basetype = vardef.get_basetype()
                if self.lhs.arrayname.get_name() not in sdfg.arrays:
                    sdfg.add_array(self.lhs.arrayname.get_name(), dims, basetype, debuginfo=self.context)
                dn = sdfg.nodes()[state].add_access(self.lhs.arrayname.get_name())

                # check if the write is "out of bounds": this _is_ allowed in
                # matlab, but not in SDFGs, since it would require to
                # dynamically reallocate the array

                # create a memlet which connects the rhs of the assignment to dn
                rhs_datanode = self.rhs.get_datanode(sdfg, state)

                if self.lhs.is_data_dependent_access() == False:
                    msubset = self.lhs.make_range_from_accdims()
                    writem = dace.memlet.Memlet.simple(self.lhs.arrayname.get_name(), msubset, debuginfo=self.context)

                    sdfg.nodes()[state].add_edge(rhs_datanode, None, dn, None, writem)
                else:
                    s = sdfg.nodes()[state]
                    acc_data_nodes = set()
                    acc_dims = []
                    for a in self.lhs.accdims:
                        if isinstance(a, AST_Constant):
                            acc_dims.append(a.get_value())
                        elif isinstance(a, AST_Ident):
                            vardef = self.search_vardef_in_scope(a.get_name())
                            if vardef is None:
                                raise ValueError('No definition found for ' + str(acc.get_name()))
                            elif isinstance(vardef, AST_ForLoop):
                                acc_data_nodes.add(vardef.var)
                                acc_dims.append(vardef.var.get_name())
                        else:
                            raise ValueError(str(type(a)) + " in data dependent write not allowed.")
                    mapdict = {}
                    for a in acc_dims:
                        mapdict[a] = str(a)
                    men, mex = s.add_map('datedepwrite', mapdict)
                    men.add_in_connector('IN_1')  # the data to write goes here
                    men.add_out_connector('OUT_1')  # and comes out here
                    for d in acc_data_nodes:
                        dname = d.get_name_in_sdfg(sdfg)
                        men.add_in_connector(dname)
                        datanode = d.get_datanode(sdfg, state)
                        s.add_edge(datanode, None, men, dname,
                                   dace.memlet.Memlet.from_array(datanode.data, datanode.desc(sdfg)))
                    s.add_edge(rhs_datanode, None, men, 'IN_1',
                               dace.memlet.Memlet.from_array(rhs_datanode.data, rhs_datanode.desc(sdfg)))
                    s.add_edge(
                        men, 'OUT_1', dn, None,
                        dace.memlet.Memlet.simple(self.lhs.arrayname.get_name(), ','.join([str(d) for d in acc_dims])))
                    s.add_edge(dn, None, mex, None, dace.memlet.Memlet())

            else:
                raise NotImplementedError("Assignment with lhs of type " + str(type(self.lhs)) +
                                          " has not been implemented yet.")
        else:
            raise NotImplementedError("Assignment operator " + self.op + " has not been implemented yet.")

    __str__ = __repr__
