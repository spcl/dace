# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace

from .ast_node import AST_Node


class AST_ArrayAccess(AST_Node):
    def __init__(self, context, arrayname, accdims):
        AST_Node.__init__(self, context)
        self.arrayname = arrayname
        self.accdims = accdims

    def __repr__(self):
        return "AST_ArrayAccess(" + str(self.arrayname) + ", " + str(
            self.accdims) + ")"

    def get_children(self):
        ret = [self.arrayname]
        ret += self.accdims
        return ret

    def replace_child(self, old, new):
        if old == self.arrayname:
            self.arrayname = new
            return
        if old in self.accdims:
            newaccdims = [new if x == old else x for x in self.accdims]
            self.accdims = newaccdims

    def get_basetype(self):
        # The basetype of an array access is the same as the basetype as the
        # array that is acccessed.
        vardef = self.search_vardef_in_scope(self.arrayname.get_name())
        return (vardef.get_basetype())

    def get_dims(self):
        from .ast_matrix import AST_Matrix
        from .ast_loop import AST_ForLoop
        from .ast_values import AST_Constant, AST_Ident
        from .ast_range import AST_RangeExpression
        # array indexing has many forms/cases in matlab, here we implement
        # the semantics we are sure about
        dims = []
        if isinstance(self.accdims, list):
            for acc in self.accdims:
                if isinstance(acc, AST_Constant):
                    dims.append(1)
                elif isinstance(acc, AST_Matrix):
                    dims.append(len(acc.get_values_row_major()))
                elif isinstance(acc, AST_RangeExpression):
                    if isinstance(acc.lhs, AST_Constant) and isinstance(
                            acc.rhs, AST_Constant):
                        l = acc.lhs.get_value()
                        r = acc.rhs.get_value()
                        dims.append(r - l + 1)
                    elif (acc.lhs is None) and (acc.rhs is None):
                        # Get the dims of the array itself
                        vardef = self.search_vardef_in_scope(
                            self.arrayname.get_name())
                        if vardef is None:
                            raise ValueError("No definition found for Array " +
                                             self.arrayname.get_name())
                        d = vardef.get_dims()
                        dims.append(d[len(dims)])
                    else:
                        raise NotImplementedError(
                            "range with non-constant bounds not supported")
                elif isinstance(acc, AST_Ident):
                    vardef = self.search_vardef_in_scope(acc.get_name())
                    if vardef is None:
                        raise ValueError("No definition found for " +
                                         acc.get_name() +
                                         " which is used in Array Access: " +
                                         str(self))
                    if isinstance(vardef, AST_ForLoop) and acc.get_name(
                    ) == vardef.var.get_name():
                        d = vardef.initializer.get_dims()[:-1]
                        if d != [1]:
                            raise NotImplementedError(
                                "Complicated slicing not implemented yet.")
                        else:
                            dims.append(d[0])
                else:
                    raise NotImplementedError(
                        "unimplemented method of array access (" + str(acc) +
                        ")")
        else:
            raise NotImplementedError("unimplemented method of array access")

        # simplify [1,1] to [1]
        if dims == [1, 1]:
            dims = [1]
        return dims

    def make_range_from_accdims(self):
        from .ast_range import AST_RangeExpression
        from .ast_values import AST_Constant

        rangelist = []
        for acc in self.accdims:
            if isinstance(acc, AST_Constant):
                rangelist.append((acc.get_value() - 1, acc.get_value() - 1, 1))
            elif isinstance(acc, AST_RangeExpression):
                if isinstance(acc.lhs, AST_Constant) and isinstance(
                        acc.rhs, AST_Constant):
                    l = acc.lhs.get_value()
                    r = acc.rhs.get_value()
                    rangelist.append((l, r, 1))
                else:
                    raise NotImplementedError(
                        "range with non-constant bounds not supported: " +
                        str(self))
            else:
                raise NotImplementedError(
                    "Non-constant array indexing not implemented: " + str(self))
        ret = dace.subsets.Range(rangelist)
        return ret

    def is_data_dependent_access(self):
        from .ast_values import AST_Constant
        res = False
        for a in self.accdims:
            if not isinstance(a, AST_Constant):
                return True

    def generate_code(self, sdfg, state):
        from .ast_values import AST_Ident
        from .ast_loop import AST_ForLoop
        from .ast_range import AST_RangeExpression
        # add a new variable to hold the result of this expression
        dims = self.get_dims()
        basetype = self.get_basetype()
        name = self.get_name_in_sdfg(sdfg)
        if name not in sdfg.arrays:
            sdfg.add_transient(name, dims, basetype, debuginfo=self.context)
        # add a memlet from the original array to the transient
        resnode = self.get_datanode(sdfg, state)
        arrnode = self.arrayname.get_datanode(sdfg, state)
        arrdesc = arrnode.desc(sdfg)

        if self.is_data_dependent_access() == False:
            msubset = self.make_range_from_accdims()
            memlet = dace.memlet.Memlet.simple(arrnode,
                                               msubset,
                                               debuginfo=self.context)
            sdfg.nodes()[state].add_edge(arrnode, None, resnode, None, memlet)
        else:
            # add a map around the access and feed the access dims that are
            # runtime dependent into a connector which is _not_ named IN
            access_data_nodes = set()
            access_dims = []
            for idx, acc in enumerate(self.accdims):
                if isinstance(acc, AST_Ident):
                    vardef = self.search_vardef_in_scope(acc.get_name())
                    if vardef is None:
                        raise ValueError('No definition found for ' +
                                         str(acc.get_name()))
                    elif isinstance(vardef, AST_ForLoop):
                        access_data_nodes.add(vardef.var)
                        access_dims.append(vardef.var.get_name())
                elif isinstance(acc, AST_RangeExpression):
                    # if the bounds are identifiers, we need them on the map
                    # otherwise we do not need to do anything here
                    if isinstance(acc.lhs, AST_Ident):
                        access_data_nodes.add(acc.lhs)
                    if isinstance(acc.rhs, AST_Ident):
                        access_data_nodes.add(acc.rhs)
                    if (acc.lhs is None) and (acc.rhs is None):
                        d = arrdesc.shape
                        access_dims.append('0:' + str(d[idx]))
                else:
                    acc.generate_code(sdfg, state)
                    access_data_nodes.add(acc)
                    access_dims.append(acc.get_name_in_sdfg(sdfg))
            # now construct the dictionary for the map range
            s = sdfg.nodes()[state]
            mdict = {}
            for aa in access_data_nodes:
                a = aa.get_name_in_sdfg(sdfg)
                mdict[a] = a
            if len(mdict) == 0:
                mdict = {'__DAPUNUSED_i': '0:1'}
            men, mex = s.add_map('datadepacc', mdict)
            men.add_in_connector('IN_1')
            men.add_out_connector('OUT_1')
            s.add_edge(arrnode, None, men, 'IN_1',
                       dace.memlet.Memlet.from_array(arrnode.data, arrdesc))
            for a in access_data_nodes:
                aname = a.get_name_in_sdfg(sdfg)
                men.add_in_connector(aname)
                datanode = a.get_datanode(sdfg, state)
                s.add_edge(
                    datanode, None, men, aname,
                    dace.memlet.Memlet.from_array(datanode.data,
                                                  datanode.desc(sdfg)))
            tasklet = s.add_tasklet('ident', {'in'}, {'out'}, 'in=out;',
                                    dace.Language.CPP)
            s.add_edge(
                men, 'OUT_1', tasklet, 'in',
                dace.memlet.Memlet.simple(arrnode, ','.join(access_dims)))
            s.add_edge(
                tasklet, 'out', mex, None,
                dace.memlet.Memlet.from_array(resnode.data, resnode.desc(sdfg)))
            s.add_edge(
                mex, None, resnode, None,
                dace.memlet.Memlet.from_array(resnode.data, resnode.desc(sdfg)))

        print("The result of " + str(self) + " will be stored in " + str(name))

    __str__ = __repr__
