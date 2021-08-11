# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import re
import dace
from collections import OrderedDict


class AST_Node():
    def __init__(self, context):
        self.context = context
        self.name = None  # Name of the variable holding the result in the SDFG
        self.parent = None
        self.next = None
        self.prev = None
        self.initializers = {}

    def get_parent(self):
        return self.parent

    def replace_parent(self, newparent):
        self.parent = newparent

    def get_children(self):
        raise NotImplementedError(
            str(type(self)) + " does not implement get_children()")

    def replace_child(self, old, new):
        raise NotImplementedError(
            str(type(self)) + " does not implement replace_child()")

    def specialize(self):
        """ Some nodes can be simplified after parsing the complete AST and
            before actually generating code, i.e., AST_FunCall nodes could be
            function calls or array accesses, and we don't really know unless
            we know the context of the call.

            This function traverses the AST
            and tries to specialize nodes after completing the AST. It should
            be called on the top-level AST_Statements node, and a node that
            wants to be specialized should return its new instance. If no
            specialzation should take place, it should return None.
        """
        for c in self.get_children():
            n = c.specialize()
            while n is not None:
                n.replace_parent(c.get_parent())
                self.replace_child(old=c, new=n)
                c = n
                n = n.specialize()

    def find_data_node_in_sdfg_state(self, sdfg, state, nodename=None):
        if nodename is None:
            nodename = self.get_name_in_sdfg(sdfg)
        sdfg_state = sdfg.nodes()[state]
        for node in sdfg_state.nodes():
            if isinstance(node, dace.sdfg.nodes.AccessNode):
                if node.label == nodename:
                    return node

        raise ValueError("No AccessNode with name " + nodename + " found.")

    def get_initializers(self, sdfg):
        initializers = self.initializers
        for c in self.get_children():
            initializers.update(c.get_initializers(sdfg))
        return initializers

    def provide_parents(self, parent):
        self.parent = parent
        for c in self.get_children():
            c.provide_parents(self)

    def search_vardef_in_scope(self, name):
        from .ast_assign import AST_Assign
        from .ast_values import AST_Ident
        from .ast_loop import AST_ForLoop
        current_node = self

        # check if we found the definition:
        # * current_node is an AST_Assign with name as lhs or
        # * a loop with name as the iterator
        if isinstance(current_node, AST_Assign) and \
           isinstance(current_node.lhs, AST_Ident) and \
           (current_node.lhs.get_name() == name):
            return current_node.rhs
        elif isinstance(current_node, AST_ForLoop) and \
            current_node.var.get_name() == name:
            return current_node

        # if current node is inside list of stmts, traverse this list using
        # prev, but first find the enclosing AST_Statements
        while current_node.get_parent() is not None:
            old_current_node = current_node
            if isinstance(current_node.get_parent(), AST_Statements):
                while current_node.prev is not None:
                    res = current_node.prev.search_vardef_in_scope(name)
                    if res is not None:
                        return res
                    current_node = current_node.prev
            current_node = current_node.get_parent()
            res = current_node.search_vardef_in_scope(name)
            if res is not None:
                return res

        return None

    def defined_variables(self):
        # Override this to return the string names of variables defined by an
        # AST_Node
        return []

    def get_datanode(self, sdfg, state):
        try:
            result = self.find_data_node_in_sdfg_state(
                sdfg=sdfg,
                state=state,
                nodename=self.get_name_in_sdfg(sdfg=sdfg))
        except ValueError:
            result = sdfg.nodes()[state].add_access(
                self.get_name_in_sdfg(sdfg=sdfg))
        return result

    def get_new_tmpvar(self, sdfg):
        TEMPVARS_PREFIX = "__tmp_"
        maxvar = 0
        for state in range(0, len(sdfg.nodes())):
            sdfg_state = sdfg.nodes()[state]
            for node in sdfg_state.nodes():
                if isinstance(node, dace.sdfg.nodes.AccessNode):
                    m = re.match(TEMPVARS_PREFIX + r"(\d+)", node.label)
                    if m is not None:
                        if maxvar < int(m.group(1)):
                            maxvar = int(m.group(1))
        newvar = maxvar + 1
        new_name = TEMPVARS_PREFIX + str(newvar)
        return new_name

    def get_name_in_sdfg(self, sdfg):
        """ If this node has no name assigned yet, create a new one of the form
            `__tmp_X` where `X` is an integer, such that this node does not yet
            exist in the given SDFG.
            @note: We assume that we create exactly one SDFG from each AST,
                   otherwise we need to store the hash of the SDFG the name was
                   created for (would be easy but seems useless at this point).
        """
        if self.name is not None:
            return self.name
        self.name = self.get_new_tmpvar(sdfg)
        return self.name

    def generate_code(self, *args):
        raise NotImplementedError(
            "Class " + type(self).__name__ +
            " does not implement the generate_code method.")

    def shortdesc(self):
        ret = str(self)
        ret = re.sub(r"\n", " ; ", ret)
        return "\"" + ret[0:70] + "\""

    def print_as_tree(self):
        ret = ""
        ret += self.shortdesc() + ";\n"
        for c in self.get_children():
            ret += self.shortdesc() + "->" + c.shortdesc(
            ) + "[label=\"child\", color=\"red\"] ;\n"
            ret += c.print_as_tree()

        if self.get_parent() is None:
            ret += self.shortdesc(
            ) + " -> \"None\" [label=\"parent\", color=\"blue\"];\n"
        else:
            ret += self.shortdesc() + " -> " + self.get_parent().shortdesc(
            ) + "[label=\"parent\", color=\"blue\"];\n"

        if isinstance(self, AST_Statements):
            ret += "{ rank=same; "
            for c in self.get_children():
                ret += c.shortdesc() + "; "
            ret += "}\n"
            for c in self.get_children():
                if c.next is not None:
                    ret += c.shortdesc() + " -> " + c.next.shortdesc(
                    ) + "[label=\"next\", color=\"green\"]"
                if c.prev is not None:
                    ret += c.shortdesc() + " -> " + c.prev.shortdesc(
                    ) + "[label=\"prev\", color=\"yellow\"]"

        return ret


class AST_Statements(AST_Node):
    def __init__(self, context, stmts):
        AST_Node.__init__(self, context)
        self.statements = stmts

        # we expect stmts to be a list of AST_Node objects
        for s in stmts:
            if not isinstance(s, AST_Node):
                raise ValueError(
                    "Expected a list of AST_Nodes, but one of the members is: "
                    + str(s) + " type " + str(type(s)))

    def __repr__(self):
        res = ["Statements:"]
        for s in self.statements:
            res.append("    " + str(s))
        return "\n".join(res)

    def get_children(self):
        return self.statements[:]

    def replace_child(self, old, new):
        newstmts = [new if x == old else x for x in self.statements]
        self.provide_parents(self.get_parent())

    def append_statement(self, stmt):
        if isinstance(stmt, list):
            self.statements += stmt
        else:
            self.statements.append(stmt)

    def provide_parents(self, parent=None):
        # Overwrite the AST_Node provide_parents() function
        # because we also set next and prev for statements, which
        # should be null for most / all AST_Nodes
        self.parent = parent

        # fix prev
        prev = None
        for s in self.statements:
            s.prev = prev
            prev = s

        # fix next
        next = None
        for s in reversed(self.statements):
            s.next = next
            next = s

        for s in self.statements:
            s.provide_parents(parent=self)

    def specialize(self):
        # If we have an AST_Function() node, pull all statements between that
        # and the next AST_EndFunction() into the function. Do that until there
        # are no more changes.
        rerun = True
        while rerun:
            rerun = False
            stmts = None
            func = None
            for c in self.get_children():
                from .ast_function import AST_Function, AST_EndFunc
                if isinstance(c, AST_Function):
                    func = c
                    stmts = []
                elif isinstance(c, AST_EndFunc):
                    func.set_statements(stmts)
                    self.statements = [
                        x for x in self.statements if x not in stmts + [c]
                    ]
                    rerun = True
                elif func is not None:
                    stmts.append(c)

        # Remove NullStatements, they are only useful during parsing
        from .ast_nullstmt import AST_NullStmt
        self.statements = [
            x for x in self.statements if not isinstance(x, AST_NullStmt)
        ]
        self.provide_parents(self.parent)

        # Lastly, specialize all children
        for c in self.get_children():
            n = c.specialize()
            while n is not None:
                n.replace_parent(c.get_parent())
                self.replace_child(old=c, new=n)
                c = n
                n = n.specialize()

        self.provide_parents(self.parent)

        return None

    def generate_code(self, sdfg=None, state=None):
        if sdfg is None:
            sdfg = dace.SDFG("dacelab", OrderedDict(), {})
            prevstate = None
            for s in self.statements:
                state = len(sdfg.nodes())
                newstate = dace.SDFGState("s" + str(state),
                                          sdfg,
                                          debuginfo=s.context)
                sdfg.add_node(newstate)
                last_state = s.generate_code(sdfg, state)
                if prevstate is not None:
                    edge = dace.sdfg.InterstateEdge()
                    sdfg.add_edge(prevstate, newstate, edge)
                if last_state is None:
                    prevstate = newstate
                else:
                    prevstate = sdfg.nodes()[last_state]

            return sdfg
        else:
            raise ValueError(
                "Appending statements to an SDFG is not supported.")

    __str__ = __repr__
