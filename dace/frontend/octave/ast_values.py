# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace

from .ast_node import AST_Node


class AST_Ident(AST_Node):
    def __init__(self, context, value):
        AST_Node.__init__(self, context)
        if isinstance(value, str):
            self.value = value
        else:
            raise ValueError("Expected str, got " + str(type(value)))

    def __repr__(self):
        return "AST_Ident(" + str(self.value) + ")"

    def get_name(self):
        return self.value

    def is_constant(self):
        return False

    def get_name_in_sdfg(self, sdfg):
        return self.value

    def get_children(self):
        return []

    def replace_child(self, old, new):
        raise ValueError("This node does not have children!")

    def generate_code(self, sdfg, state):
        # An identifier never generates code
        pass

    def get_dims(self):
        from .ast_loop import AST_ForLoop
        """ Check in the scope if this is defined and return the dims of the
            corresponding SDFG access node it currently maps to. """
        vardef = self.search_vardef_in_scope(self.value)
        if vardef is None:
            raise ValueError("Request for dims of identifier " + self.value +
                             " which is not defined in the current scope")
        elif isinstance(vardef, AST_ForLoop):
            dims = vardef.initializer.get_dims()[:1]
            return dims
        else:
            return vardef.get_dims()

    def specialize(self):
        pass

    def get_propagated_value(self):
        vardef = self.search_vardef_in_scope(self.get_name())
        if isinstance(vardef, AST_Constant):
            return vardef
        return None

    def get_basetype(self):
        """ Check in the scope if this is defined and return the basetype of the
            corresponding SDFG access node this currently maps to. """
        bt = self.search_vardef_in_scope(self.value).get_basetype()
        if bt is None:
            raise ValueError("Request for basetype of identifier " + self.value +
                             " which is not defined in the current scope")
        else:
            return bt

    __str__ = __repr__


class AST_Constant(AST_Node):
    def __init__(self, context, value):
        AST_Node.__init__(self, context)
        self.value = value

    def __repr__(self):
        return "AST_Constant(" + str(self.value) + ")"

    def get_value(self):
        return self.value

    def get_dims(self):
        return [1]

    def get_basetype(self):
        return dace.dtypes.float64

    def generate_code(self, sdfg, state):
        dims = self.get_dims()
        name = self.get_name_in_sdfg(sdfg)
        basetype = dace.dtypes.float64
        if name not in sdfg.arrays:
            sdfg.add_transient(name, dims, basetype, debuginfo=self.context)
        trans = sdfg.nodes()[state].add_access(name)
        code = "out = " + str(self.get_value()) + ";"
        tasklet = sdfg.nodes()[state].add_tasklet('init', {}, {'out'}, code, dace.Language.CPP)
        sdfg.nodes()[state].add_edge(tasklet, 'out', trans, None,
                                     dace.memlet.Memlet.from_array(trans.data, trans.desc(sdfg)))
        print("The result of expr " + str(self) + " will be stored in " + str(name))

    def get_children(self):
        return []

    def is_constant(self):
        return True

    def replace_child(self, old, new):
        raise ValueError("This node does not have children!")

    __str__ = __repr__
