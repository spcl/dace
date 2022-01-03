# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace

from .ast_node import AST_Node


class AST_ForLoop(AST_Node):
    def __init__(self, context, var, initializer, stmts):
        AST_Node.__init__(self, context)
        self.var = var
        self.initializer = initializer
        self.stmts = stmts

    def __repr__(self):
        return "AST_ForLoop(" + str(self.var) + " = " + str(self.initializer) + ", stmts: {\n" + str(
            self.stmts) + "\n})"

    def get_children(self):
        return [self.var, self.initializer, self.stmts]

    def replace_child(self, old, new):
        if old == self.var:
            self.var = new
            return
        if old == self.initializer:
            self.initializer = new
            return
        if old == self.stmts:
            self.stmts = new
            return
        raise ValueError("The child " + str(old) + " is not a child of " + str(self))

    def generate_code(self, sdfg, state):
        from .ast_range import AST_RangeExpression
        # This ignores matlab semantics and only works for loops of the form
        # for var = start:end where start and end are expressions which
        # evaluate to scalars.
        if isinstance(self.initializer, AST_RangeExpression):
            # Generate the initializer:
            # lhs and rhs of the iteration range as two transients, and a
            # transient for i (we also have a symbol for i which states will
            # use)
            initializer_state_num = state
            s = sdfg.nodes()[state]
            self.initializer.lhs.generate_code(sdfg, state)
            lhs_node = self.initializer.lhs.get_datanode(sdfg, state)
            self.initializer.rhs.generate_code(sdfg, state)
            rhs_node = self.initializer.rhs.get_datanode(sdfg, state)
            sdfg.add_transient(self.var.get_name_in_sdfg(sdfg), [1], self.initializer.lhs.get_basetype())
            var_node = s.add_access(self.var.get_name_in_sdfg(sdfg))
            s.add_edge(lhs_node, None, var_node, None,
                       dace.memlet.Memlet.from_array(var_node.data, var_node.desc(sdfg)))
            loop_guard_var = '_loopiter_' + str(state)
            loop_end_var = '_loopend_' + str(state)

            # Generate guard state, write loop iter symbol into loop iter
            # datanode
            guard_state_num = initializer_state_num + 1
            s_guard = sdfg.add_state('s' + str(guard_state_num))
            task = s_guard.add_tasklet('reinitloopiter', {}, {'out'}, "out=" + loop_guard_var)

            if self.var.get_name_in_sdfg(sdfg) not in sdfg.arrays:
                sdfg.add_transient(self.var.get_name_in_sdfg(sdfg), [1], self.initializer.lhs.get_basetype())
            trans = s_guard.add_access(self.var.get_name_in_sdfg(sdfg))
            # Workaround until "condition for putting a variable as top-level
            # doesn't take inter-state edges into account" is solved.
            # When fixed, the line below can be removed.
            self.initializer.rhs.generate_code(sdfg, guard_state_num)

            s_guard.add_edge(task, 'out', trans, None, dace.memlet.Memlet.from_array(trans.data, trans.desc(sdfg)))
            lg_init = dace.sdfg.InterstateEdge(
                assignments={
                    loop_guard_var: self.var.get_name_in_sdfg(sdfg) + '(0)',
                    loop_end_var: self.initializer.rhs.get_name_in_sdfg(sdfg) + '(0)'
                })
            sdfg.add_edge(sdfg.nodes()[state], s_guard, lg_init)

            # Add state for each statement within the for loop
            prev = s_guard
            for s in self.stmts.statements:
                state = len(sdfg.nodes())
                newstate = dace.SDFGState("s" + str(state), sdfg, debuginfo=s.context)
                sdfg.add_node(newstate)
                last_state = s.generate_code(sdfg, state)
                if last_state is None: last_state = state
                if prev != s_guard:
                    edge = dace.sdfg.InterstateEdge()
                    sdfg.add_edge(prev, newstate, edge)
                else:
                    edge = dace.sdfg.InterstateEdge(condition=dace.properties.CodeProperty.from_string(
                        loop_guard_var + " <= " + loop_end_var, language=dace.dtypes.Language.Python))
                    sdfg.add_edge(prev, newstate, edge)
                prev = sdfg.nodes()[last_state]

            # Create inter-state back-edge
            edge = dace.sdfg.InterstateEdge(assignments={loop_guard_var: loop_guard_var + '+1'})
            sdfg.add_edge(prev, s_guard, edge)

            # Create the loop exit state
            state = len(sdfg.nodes())
            s_lexit = dace.SDFGState("s" + str(state), sdfg, debuginfo=s.context)
            lend_val = str(self.initializer.get_dims()[-1])
            for_exit = dace.sdfg.InterstateEdge(condition=dace.properties.CodeProperty.from_string(
                loop_guard_var + " > " + loop_end_var, language=dace.dtypes.Language.Python))
            sdfg.add_edge(s_guard, s_lexit, for_exit)

            return state

        else:
            raise NotImplementedError("Loops over anything but ranges are not implemented.")

    def generate_code_proper(self, sdfg, state):
        # This follows matlab semantics, i.e., a loop iterates over the columns
        # of a matrix. This does not work well for sdfgs for all but the
        # simplest case (a matrix which is a compile time constant, ie. 1:10).
        # To support programs like Cholesky, we try to transform the matlab for
        # loop into a C-style loop, this is implemented in generate_code().

        # Generate the initializer:
        # Each iteration of the for loop will use one column
        initializer_state_num = state
        self.initializer.generate_code(sdfg, state)
        loop_guard_var = '_lg_' + str(state)
        # Generate an (empty) guard state
        guard_state_num = initializer_state_num + 1
        s_guard = sdfg.add_state('s' + str(guard_state_num))
        lg_init = dace.sdfg.InterstateEdge(assignments={loop_guard_var: '0'})
        sdfg.add_edge(sdfg.nodes()[state], s_guard, lg_init)

        # Read a column of the initializer
        get_initializer_state_num = guard_state_num + 1
        s_getinit = sdfg.add_state('s' + str(get_initializer_state_num))
        initializer_name = self.initializer.get_name_in_sdfg(sdfg)
        loopvar_name = self.var.get_name_in_sdfg(sdfg)
        dims = self.initializer.get_dims()[:1]
        sdfg.add_transient(loopvar_name, dims, self.initializer.get_basetype())
        part = s_getinit.add_access(loopvar_name)
        sdfg.add_transient(initializer_name, self.initializer.get_dims(), self.initializer.get_basetype())
        full = s_getinit.add_read(initializer_name)
        s_getinit.add_edge(full, None, part, None, dace.memlet.Memlet.simple(initializer_name, 'i,0'))

        # Add edge from guard to getinit
        lend_val = str(self.initializer.get_dims()[-1])
        for_entry = dace.sdfg.InterstateEdge(condition=dace.properties.CodeProperty.from_string(
            loop_guard_var + " < " + lend_val, language=dace.dtypes.Language.Python))
        sdfg.add_edge(s_guard, s_getinit, for_entry)

        # Add state for each statement within the for loop
        prev = s_getinit
        for s in self.stmts.statements:
            state = len(sdfg.nodes())
            newstate = dace.SDFGState("s" + str(state), sdfg, debuginfo=s.context)
            sdfg.add_node(newstate)
            last_state = s.generate_code(sdfg, state)
            if last_state is None: last_state = state
            edge = dace.sdfg.InterstateEdge()
            sdfg.add_edge(prev, newstate, edge)
            prev = sdfg.nodes()[last_state]

        # Create inter-state back-edge
        edge = dace.sdfg.InterstateEdge(assignments={loop_guard_var: loop_guard_var + '+1'})
        sdfg.add_edge(prev, s_guard, edge)

        # Create the loop exit state
        state = len(sdfg.nodes())
        s_lexit = dace.SDFGState("s" + str(state), sdfg, debuginfo=s.context)
        lend_val = str(self.initializer.get_dims()[-1])
        for_exit = dace.sdfg.InterstateEdge(condition=dace.properties.CodeProperty.from_string(
            loop_guard_var + " >= " + lend_val, language=dace.dtypes.Language.Python))
        sdfg.add_edge(s_guard, s_lexit, for_exit)

        return state

    __str__ = __repr__
