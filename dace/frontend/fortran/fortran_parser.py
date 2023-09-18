# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

from venv import create
import warnings

from dace.data import Scalar

import dace.frontend.fortran.ast_components as ast_components
import dace.frontend.fortran.ast_transforms as ast_transforms
import dace.frontend.fortran.ast_utils as ast_utils
import dace.frontend.fortran.ast_internal_classes as ast_internal_classes
from typing import List, Tuple, Set
from dace import dtypes
from dace import Language as lang
from dace import data as dat
from dace import SDFG, InterstateEdge, Memlet, pointer, nodes
from dace import symbolic as sym
from copy import deepcopy as dpcp

from dace.properties import CodeBlock
from fparser.two.parser import ParserFactory as pf
from fparser.common.readfortran import FortranStringReader as fsr
from fparser.common.readfortran import FortranFileReader as ffr
from fparser.two.symbol_table import SymbolTable


class AST_translator:
    """  
    This class is responsible for translating the internal AST into a SDFG.
    """
    def __init__(self, ast: ast_components.InternalFortranAst, source: str):
        """
        :ast: The internal fortran AST to be used for translation
        :source: The source file name from which the AST was generated
        """
        self.tables = ast.tables
        self.top_level = None
        self.globalsdfg = None
        self.functions_and_subroutines = ast.functions_and_subroutines
        self.name_mapping = ast_utils.NameMap()
        self.contexts = {}
        self.views = 0
        self.libstates = []
        self.file_name = source
        self.unallocated_arrays = []
        self.all_array_names = []
        self.last_sdfg_states = {}
        self.last_loop_continues = {}
        self.last_loop_breaks = {}
        self.last_returns = {}
        self.module_vars = []
        self.libraries = {}
        self.last_call_expression = {}
        self.ast_elements = {
            ast_internal_classes.If_Stmt_Node: self.ifstmt2sdfg,
            ast_internal_classes.For_Stmt_Node: self.forstmt2sdfg,
            ast_internal_classes.Map_Stmt_Node: self.forstmt2sdfg,
            ast_internal_classes.Execution_Part_Node: self.basicblock2sdfg,
            ast_internal_classes.Subroutine_Subprogram_Node: self.subroutine2sdfg,
            ast_internal_classes.BinOp_Node: self.binop2sdfg,
            ast_internal_classes.Decl_Stmt_Node: self.declstmt2sdfg,
            ast_internal_classes.Var_Decl_Node: self.vardecl2sdfg,
            ast_internal_classes.Symbol_Decl_Node: self.symbol2sdfg,
            ast_internal_classes.Symbol_Array_Decl_Node: self.symbolarray2sdfg,
            ast_internal_classes.Call_Expr_Node: self.call2sdfg,
            ast_internal_classes.Program_Node: self.ast2sdfg,
            ast_internal_classes.Write_Stmt_Node: self.write2sdfg,
            ast_internal_classes.Allocate_Stmt_Node: self.allocate2sdfg,
        }

    def get_dace_type(self, type):
        """  
        This function matches the fortran type to the corresponding dace type
        by referencing the ast_utils.fortrantypes2dacetypes dictionary.
        """
        if isinstance(type, str):
            return ast_utils.fortrantypes2dacetypes[type]

    def get_name_mapping_in_context(self, sdfg: SDFG):
        """
        This function returns a copy of the name mapping union
         for the given sdfg and the top-level sdfg.
        """
        a = self.name_mapping[self.globalsdfg].copy()
        if sdfg is not self.globalsdfg:
            a.update(self.name_mapping[sdfg])
        return a

    def get_arrays_in_context(self, sdfg: SDFG):
        """
        This function returns a copy of the union of arrays 
        for the given sdfg and the top-level sdfg.
        """
        a = self.globalsdfg.arrays.copy()
        if sdfg is not self.globalsdfg:
            a.update(sdfg.arrays)
        return a

    def get_memlet_range(self, sdfg: SDFG, variables: List[ast_internal_classes.FNode], var_name: str,
                         var_name_tasklet: str) -> str:
        """
        This function returns the memlet range for the given variable.
        :param sdfg: The sdfg in which the variable is used
        :param variables: The list of variables in the current context
        :param var_name: The name of the variable for which the memlet range should be returned
        :param var_name_tasklet: The name of the variable in the tasklet
        :return: The memlet range for the given variable
        """
        var = self.get_arrays_in_context(sdfg).get(var_name)

        if len(var.shape) == 0:
            return ""

        if (len(var.shape) == 1 and var.shape[0] == 1):
            return "0"

        for o_v in variables:
            if o_v.name == var_name_tasklet:
                return ast_utils.generate_memlet(o_v, sdfg, self)

    def translate(self, node: ast_internal_classes.FNode, sdfg: SDFG):
        """
        This function is responsible for translating the AST into a SDFG.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        :note: This function is recursive and will call itself for all child nodes
        :note: This function will call the appropriate function for the node type
        :note: The dictionary ast_elements, part of the class itself contains all functions that are called for the different node types
        """
        if node.__class__ in self.ast_elements:
            self.ast_elements[node.__class__](node, sdfg)
        elif isinstance(node, list):
            for i in node:
                self.translate(i, sdfg)
        else:
            warnings.warn(f"WARNING: {node.__class__.__name__}")

    def ast2sdfg(self, node: ast_internal_classes.Program_Node, sdfg: SDFG):
        """
        This function is responsible for translating the Fortran AST into a SDFG.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        :note: This function is recursive and will call itself for all child nodes
        :note: This function will call the appropriate function for the node type
        :note: The dictionary ast_elements, part of the class itself contains all functions that are called for the different node types
        """
        self.globalsdfg = sdfg
        for i in node.modules:
            for j in i.specification_part.typedecls:
                self.translate(j, sdfg)
                for k in j.vardecl:
                    self.module_vars.append((k.name, i.name))
            for j in i.specification_part.symbols:
                self.translate(j, sdfg)
                for k in j.vardecl:
                    self.module_vars.append((k.name, i.name))
            for j in i.specification_part.specifications:
                self.translate(j, sdfg)
                for k in j.vardecl:
                    self.module_vars.append((k.name, i.name))

        for i in node.main_program.specification_part.typedecls:
            self.translate(i, sdfg)
        for i in node.main_program.specification_part.symbols:
            self.translate(i, sdfg)
        for i in node.main_program.specification_part.specifications:
            self.translate(i, sdfg)
        self.translate(node.main_program.execution_part.execution, sdfg)

    def basicblock2sdfg(self, node: ast_internal_classes.Execution_Part_Node, sdfg: SDFG):
        """
        This function is responsible for translating Fortran basic blocks into a SDFG.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        """

        for i in node.execution:
            self.translate(i, sdfg)

    def allocate2sdfg(self, node: ast_internal_classes.Allocate_Stmt_Node, sdfg: SDFG):
        """
        This function is responsible for translating Fortran allocate statements into a SDFG.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        :note: We pair the allocate with a list of unallocated arrays.
        """
        for i in node.allocation_list:
            for j in self.unallocated_arrays:
                if j[0] == i.name.name and sdfg == j[2]:
                    datatype = j[1]
                    transient = j[3]
                    self.unallocated_arrays.remove(j)
                    offset_value = -1
                    sizes = []
                    offset = []
                    for j in i.shape.shape_list:
                        tw = ast_utils.TaskletWriter([], [], sdfg, self.name_mapping)
                        text = tw.write_code(j)
                        sizes.append(sym.pystr_to_symbolic(text))
                        offset.append(offset_value)
                    strides = [dat._prod(sizes[:i]) for i in range(len(sizes))]
                    self.name_mapping[sdfg][i.name.name] = sdfg._find_new_name(i.name.name)

                    self.all_array_names.append(self.name_mapping[sdfg][i.name.name])
                    if self.contexts.get(sdfg.name) is None:
                        self.contexts[sdfg.name] = ast_utils.Context(name=sdfg.name)
                    if i.name.name not in self.contexts[sdfg.name].containers:
                        self.contexts[sdfg.name].containers.append(i.name.name)
                    sdfg.add_array(self.name_mapping[sdfg][i.name.name],
                                   shape=sizes,
                                   dtype=datatype,
                                   offset=offset,
                                   strides=strides,
                                   transient=transient)


    def write2sdfg(self, node: ast_internal_classes.Write_Stmt_Node, sdfg: SDFG):
        #TODO implement
        raise NotImplementedError("Fortran write statements are not implemented yet")

    def ifstmt2sdfg(self, node: ast_internal_classes.If_Stmt_Node, sdfg: SDFG):
        """
        This function is responsible for translating Fortran if statements into a SDFG.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        """

        name = f"If_l_{str(node.line_number[0])}_c_{str(node.line_number[1])}"
        begin_state = ast_utils.add_simple_state_to_sdfg(self, sdfg, f"Begin{name}")
        guard_substate = sdfg.add_state(f"Guard{name}")
        sdfg.add_edge(begin_state, guard_substate, InterstateEdge())

        condition = ast_utils.ProcessedWriter(sdfg, self.name_mapping).write_code(node.cond)

        body_ifstart_state = sdfg.add_state(f"BodyIfStart{name}")
        self.last_sdfg_states[sdfg] = body_ifstart_state
        self.translate(node.body, sdfg)
        final_substate = sdfg.add_state(f"MergeState{name}")

        sdfg.add_edge(guard_substate, body_ifstart_state, InterstateEdge(condition))

        if self.last_sdfg_states[sdfg] not in [
                self.last_loop_breaks.get(sdfg),
                self.last_loop_continues.get(sdfg),
                self.last_returns.get(sdfg)
        ]:
            body_ifend_state = ast_utils.add_simple_state_to_sdfg(self, sdfg, f"BodyIfEnd{name}")
            sdfg.add_edge(body_ifend_state, final_substate, InterstateEdge())

        if len(node.body_else.execution) > 0:
            name_else = f"Else_l_{str(node.line_number[0])}_c_{str(node.line_number[1])}"
            body_elsestart_state = sdfg.add_state("BodyElseStart" + name_else)
            self.last_sdfg_states[sdfg] = body_elsestart_state
            self.translate(node.body_else, sdfg)
            body_elseend_state = ast_utils.add_simple_state_to_sdfg(self, sdfg, f"BodyElseEnd{name_else}")
            sdfg.add_edge(guard_substate, body_elsestart_state, InterstateEdge("not (" + condition + ")"))
            sdfg.add_edge(body_elseend_state, final_substate, InterstateEdge())
        else:
            sdfg.add_edge(guard_substate, final_substate, InterstateEdge("not (" + condition + ")"))
        self.last_sdfg_states[sdfg] = final_substate

    def forstmt2sdfg(self, node: ast_internal_classes.For_Stmt_Node, sdfg: SDFG):
        """
        This function is responsible for translating Fortran for statements into a SDFG.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        """

        declloop = False
        name = "FOR_l_" + str(node.line_number[0]) + "_c_" + str(node.line_number[1])
        begin_state = ast_utils.add_simple_state_to_sdfg(self, sdfg, "Begin" + name)
        guard_substate = sdfg.add_state("Guard" + name)
        final_substate = sdfg.add_state("Merge" + name)
        self.last_sdfg_states[sdfg] = final_substate
        decl_node = node.init
        entry = {}
        if isinstance(decl_node, ast_internal_classes.BinOp_Node):
            if sdfg.symbols.get(decl_node.lval.name) is not None:
                iter_name = decl_node.lval.name
            elif self.name_mapping[sdfg].get(decl_node.lval.name) is not None:
                iter_name = self.name_mapping[sdfg][decl_node.lval.name]
            else:
                raise ValueError("Unknown variable " + decl_node.lval.name)
            entry[iter_name] = ast_utils.ProcessedWriter(sdfg, self.name_mapping).write_code(decl_node.rval)

        sdfg.add_edge(begin_state, guard_substate, InterstateEdge(assignments=entry))

        condition = ast_utils.ProcessedWriter(sdfg, self.name_mapping).write_code(node.cond)

        increment = "i+0+1"
        if isinstance(node.iter, ast_internal_classes.BinOp_Node):
            increment = ast_utils.ProcessedWriter(sdfg, self.name_mapping).write_code(node.iter.rval)
        entry = {iter_name: increment}

        begin_loop_state = sdfg.add_state("BeginLoop" + name)
        end_loop_state = sdfg.add_state("EndLoop" + name)
        self.last_sdfg_states[sdfg] = begin_loop_state
        self.last_loop_continues[sdfg] = end_loop_state
        self.translate(node.body, sdfg)

        sdfg.add_edge(self.last_sdfg_states[sdfg], end_loop_state, InterstateEdge())
        sdfg.add_edge(guard_substate, begin_loop_state, InterstateEdge(condition))
        sdfg.add_edge(end_loop_state, guard_substate, InterstateEdge(assignments=entry))
        sdfg.add_edge(guard_substate, final_substate, InterstateEdge(f"not ({condition})"))
        self.last_sdfg_states[sdfg] = final_substate

    def symbol2sdfg(self, node: ast_internal_classes.Symbol_Decl_Node, sdfg: SDFG):
        """
        This function is responsible for translating Fortran symbol declarations into a SDFG.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        """

        if self.contexts.get(sdfg.name) is None:
            self.contexts[sdfg.name] = ast_utils.Context(name=sdfg.name)
        if self.contexts[sdfg.name].constants.get(node.name) is None:
            if isinstance(node.init, ast_internal_classes.Int_Literal_Node) or isinstance(
                    node.init, ast_internal_classes.Real_Literal_Node):
                self.contexts[sdfg.name].constants[node.name] = node.init.value
            if isinstance(node.init, ast_internal_classes.Name_Node):
                self.contexts[sdfg.name].constants[node.name] = self.contexts[sdfg.name].constants[node.init.name]
        datatype = self.get_dace_type(node.type)
        if node.name not in sdfg.symbols:
            sdfg.add_symbol(node.name, datatype)
            if self.last_sdfg_states.get(sdfg) is None:
                bstate = sdfg.add_state("SDFGbegin", is_start_state=True)
                self.last_sdfg_states[sdfg] = bstate
            if node.init is not None:
                substate = sdfg.add_state(f"Dummystate_{node.name}")
                increment = ast_utils.TaskletWriter([], [], sdfg, self.name_mapping).write_code(node.init)

                entry = {node.name: increment}
                sdfg.add_edge(self.last_sdfg_states[sdfg], substate, InterstateEdge(assignments=entry))
                self.last_sdfg_states[sdfg] = substate

    def symbolarray2sdfg(self, node: ast_internal_classes.Symbol_Array_Decl_Node, sdfg: SDFG):

        return NotImplementedError(
            "Symbol_Decl_Node not implemented. This should be done via a transformation that itemizes the constant array."
        )

    def subroutine2sdfg(self, node: ast_internal_classes.Subroutine_Subprogram_Node, sdfg: SDFG):
        """
        This function is responsible for translating Fortran subroutine declarations into a SDFG.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        """

        if node.execution_part is None:
            return

        # First get the list of read and written variables
        inputnodefinder = ast_transforms.FindInputs()
        inputnodefinder.visit(node)
        input_vars = inputnodefinder.nodes
        outputnodefinder = ast_transforms.FindOutputs()
        outputnodefinder.visit(node)
        output_vars = outputnodefinder.nodes
        write_names = list(dict.fromkeys([i.name for i in output_vars]))
        read_names = list(dict.fromkeys([i.name for i in input_vars]))

        # Collect the parameters and the function signature to comnpare and link
        parameters = node.args.copy()

        new_sdfg = SDFG(node.name.name)
        substate = ast_utils.add_simple_state_to_sdfg(self, sdfg, "state" + node.name.name)
        variables_in_call = []
        if self.last_call_expression.get(sdfg) is not None:
            variables_in_call = self.last_call_expression[sdfg]

        # Sanity check to make sure the parameter numbers match
        if not ((len(variables_in_call) == len(parameters)) or
                (len(variables_in_call) == len(parameters) + 1
                 and not isinstance(node.result_type, ast_internal_classes.Void))):
            for i in variables_in_call:
                print("VAR CALL: ", i.name)
            for j in parameters:
                print("LOCAL TO UPDATE: ", j.name)
            raise ValueError("number of parameters does not match the function signature")

        # creating new arrays for nested sdfg
        ins_in_new_sdfg = []
        outs_in_new_sdfg = []

        views = []
        ind_count = 0

        var2 = []
        literals = []
        literal_values = []
        par2 = []

        symbol_arguments = []

        # First we need to check if the parameters are literals or variables
        for arg_i, variable in enumerate(variables_in_call):
            if isinstance(variable, ast_internal_classes.Name_Node):
                varname = variable.name
            elif isinstance(variable, ast_internal_classes.Array_Subscript_Node):
                varname = variable.name.name
            if isinstance(variable, ast_internal_classes.Literal) or varname == "LITERAL":
                literals.append(parameters[arg_i])
                literal_values.append(variable)
                continue
            elif varname in sdfg.symbols:
                symbol_arguments.append((parameters[arg_i], variable))
                continue

            par2.append(parameters[arg_i])
            var2.append(variable)

        #This handles the case where the function is called with literals
        variables_in_call = var2
        parameters = par2
        assigns = []
        for lit, litval in zip(literals, literal_values):
            local_name = lit
            assigns.append(
                ast_internal_classes.BinOp_Node(lval=ast_internal_classes.Name_Node(name=local_name.name),
                                                rval=litval,
                                                op="=",
                                                line_number=node.line_number))

        # This handles the case where the function is called with symbols
        for parameter, symbol in symbol_arguments:
            if parameter.name != symbol.name:
                assigns.append(
                    ast_internal_classes.BinOp_Node(lval=ast_internal_classes.Name_Node(name=parameter.name),
                                                    rval=ast_internal_classes.Name_Node(name=symbol.name),
                                                    op="=",
                                                    line_number=node.line_number))

        # This handles the case where the function is called with variables starting with the case that the variable is local to the calling SDFG
        for variable_in_call in variables_in_call:
            all_arrays = self.get_arrays_in_context(sdfg)

            sdfg_name = self.name_mapping.get(sdfg).get(ast_utils.get_name(variable_in_call))
            globalsdfg_name = self.name_mapping.get(self.globalsdfg).get(ast_utils.get_name(variable_in_call))
            matched = False
            for array_name, array in all_arrays.items():
                if array_name in [sdfg_name]:
                    matched = True
                    local_name = parameters[variables_in_call.index(variable_in_call)]
                    self.name_mapping[new_sdfg][local_name.name] = new_sdfg._find_new_name(local_name.name)
                    self.all_array_names.append(self.name_mapping[new_sdfg][local_name.name])
                    if local_name.name in read_names:
                        ins_in_new_sdfg.append(self.name_mapping[new_sdfg][local_name.name])
                    if local_name.name in write_names:
                        outs_in_new_sdfg.append(self.name_mapping[new_sdfg][local_name.name])

                    indices = 0
                    index_list = []
                    shape = []
                    tmp_node = variable_in_call
                    strides = list(array.strides)
                    offsets = list(array.offset)
                    mysize = 1

                    if isinstance(variable_in_call, ast_internal_classes.Array_Subscript_Node):
                        changed_indices = 0
                        for i in variable_in_call.indices:
                            if isinstance(i, ast_internal_classes.ParDecl_Node):
                                if i.type == "ALL":
                                    shape.append(array.shape[indices])
                                    mysize = mysize * array.shape[indices]
                                    index_list.append(None)
                                else:
                                    raise NotImplementedError("Index in ParDecl should be ALL")
                            else:
                                text = ast_utils.ProcessedWriter(sdfg, self.name_mapping).write_code(i)
                                index_list.append(sym.pystr_to_symbolic(text))
                                strides.pop(indices - changed_indices)
                                offsets.pop(indices - changed_indices)
                                changed_indices += 1
                            indices = indices + 1

                    if isinstance(variable_in_call, ast_internal_classes.Name_Node):
                        shape = list(array.shape)
                    # Functionally, this identifies the case where the array is in fact a scalar
                    if shape == () or shape == (1, ) or shape == [] or shape == [1]:
                        new_sdfg.add_scalar(self.name_mapping[new_sdfg][local_name.name], array.dtype, array.storage)
                    else:
                        # This is the case where the array is not a scalar and we need to create a view
                        if not isinstance(variable_in_call, ast_internal_classes.Name_Node):
                            offsets_zero = []
                            for index in offsets:
                                offsets_zero.append(0)
                            viewname, view = sdfg.add_view(array_name + "_view_" + str(self.views),
                                                           shape,
                                                           array.dtype,
                                                           storage=array.storage,
                                                           strides=strides,
                                                           offset=offsets_zero)
                            from dace import subsets

                            all_indices = [None] * (len(array.shape) - len(index_list)) + index_list
                            subset = subsets.Range([(i, i, 1) if i is not None else (1, s, 1)
                                                    for i, s in zip(all_indices, array.shape)])
                            smallsubset = subsets.Range([(0, s - 1, 1) for s in shape])

                            memlet = Memlet(f'{array_name}[{subset}]->{smallsubset}')
                            memlet2 = Memlet(f'{viewname}[{smallsubset}]->{subset}')
                            wv = None
                            rv = None
                            if local_name.name in read_names:
                                r = substate.add_read(array_name)
                                wv = substate.add_write(viewname)
                                substate.add_edge(r, None, wv, 'views', dpcp(memlet))
                            if local_name.name in write_names:
                                rv = substate.add_read(viewname)
                                w = substate.add_write(array_name)
                                substate.add_edge(rv, 'views2', w, None, dpcp(memlet2))

                            self.views = self.views + 1
                            views.append([array_name, wv, rv, variables_in_call.index(variable_in_call)])

                        new_sdfg.add_array(self.name_mapping[new_sdfg][local_name.name],
                                           shape,
                                           array.dtype,
                                           array.storage,
                                           strides=strides,
                                           offset=offsets)
            if not matched:
                # This handles the case where the function is called with global variables
                for array_name, array in all_arrays.items():
                    if array_name in [globalsdfg_name]:
                        local_name = parameters[variables_in_call.index(variable_in_call)]
                        self.name_mapping[new_sdfg][local_name.name] = new_sdfg._find_new_name(local_name.name)
                        self.all_array_names.append(self.name_mapping[new_sdfg][local_name.name])
                        if local_name.name in read_names:
                            ins_in_new_sdfg.append(self.name_mapping[new_sdfg][local_name.name])
                        if local_name.name in write_names:
                            outs_in_new_sdfg.append(self.name_mapping[new_sdfg][local_name.name])

                        indices = 0
                        if isinstance(variable_in_call, ast_internal_classes.Array_Subscript_Node):
                            indices = len(variable_in_call.indices)

                        shape = array.shape[indices:]

                        if shape == () or shape == (1, ):
                            new_sdfg.add_scalar(self.name_mapping[new_sdfg][local_name.name], array.dtype,
                                                array.storage)
                        else:
                            new_sdfg.add_array(self.name_mapping[new_sdfg][local_name.name],
                                               shape,
                                               array.dtype,
                                               array.storage,
                                               strides=array.strides,
                                               offset=array.offset)

        # Preparing symbol dictionary for nested sdfg
        sym_dict = {}
        for i in sdfg.symbols:
            sym_dict[i] = i

        not_found_write_names = []
        not_found_read_names = []
        for i in write_names:
            if self.name_mapping[new_sdfg].get(i) is None:
                not_found_write_names.append(i)
        for i in read_names:
            if self.name_mapping[new_sdfg].get(i) is None:
                not_found_read_names.append(i)

        # This handles the library states that are needed to inject dataflow to prevent library calls from being reordered
        # Currently not sufficient for all cases
        for i in self.libstates:
            self.name_mapping[new_sdfg][i] = new_sdfg._find_new_name(i)
            self.all_array_names.append(self.name_mapping[new_sdfg][i])
            if i in read_names:
                ins_in_new_sdfg.append(self.name_mapping[new_sdfg][i])
            if i in write_names:
                outs_in_new_sdfg.append(self.name_mapping[new_sdfg][i])
            new_sdfg.add_scalar(self.name_mapping[new_sdfg][i], dtypes.int32, transient=False)
        addedmemlets = []
        globalmemlets = []
        # This handles the case where the function is called with read variables found in a module
        for i in not_found_read_names:
            if i in [a[0] for a in self.module_vars]:
                if self.name_mapping[sdfg].get(i) is not None:
                    self.name_mapping[new_sdfg][i] = new_sdfg._find_new_name(i)
                    addedmemlets.append(i)
                    self.all_array_names.append(self.name_mapping[new_sdfg][i])
                    if i in read_names:
                        ins_in_new_sdfg.append(self.name_mapping[new_sdfg][i])
                    if i in write_names:
                        outs_in_new_sdfg.append(self.name_mapping[new_sdfg][i])
                    array_in_global = sdfg.arrays[self.name_mapping[sdfg][i]]
                    if isinstance(array_in_global, Scalar):
                        new_sdfg.add_scalar(self.name_mapping[new_sdfg][i], array_in_global.dtype, transient=False)
                    elif array_in_global.type == "Array":
                        new_sdfg.add_array(self.name_mapping[new_sdfg][i],
                                           array_in_global.shape,
                                           array_in_global.dtype,
                                           array_in_global.storage,
                                           transient=False,
                                           strides=array_in_global.strides,
                                           offset=array_in_global.offset)
                elif self.name_mapping[self.globalsdfg].get(i) is not None:
                    self.name_mapping[new_sdfg][i] = new_sdfg._find_new_name(i)
                    globalmemlets.append(i)
                    self.all_array_names.append(self.name_mapping[new_sdfg][i])
                    if i in read_names:
                        ins_in_new_sdfg.append(self.name_mapping[new_sdfg][i])
                    if i in write_names:
                        outs_in_new_sdfg.append(self.name_mapping[new_sdfg][i])

                    array_in_global = self.globalsdfg.arrays[self.name_mapping[self.globalsdfg][i]]
                    if isinstance(array_in_global, Scalar):
                        new_sdfg.add_scalar(self.name_mapping[new_sdfg][i], array_in_global.dtype, transient=False)
                    elif array_in_global.type == "Array":
                        new_sdfg.add_array(self.name_mapping[new_sdfg][i],
                                           array_in_global.shape,
                                           array_in_global.dtype,
                                           array_in_global.storage,
                                           transient=False,
                                           strides=array_in_global.strides,
                                           offset=array_in_global.offset)
        # This handles the case where the function is called with wrriten but not read variables found in a module
        for i in not_found_write_names:
            if i in not_found_read_names:
                continue
            if i in [a[0] for a in self.module_vars]:
                if self.name_mapping[sdfg].get(i) is not None:
                    self.name_mapping[new_sdfg][i] = new_sdfg._find_new_name(i)
                    addedmemlets.append(i)
                    self.all_array_names.append(self.name_mapping[new_sdfg][i])
                    if i in read_names:
                        ins_in_new_sdfg.append(self.name_mapping[new_sdfg][i])
                    if i in write_names:
                        outs_in_new_sdfg.append(self.name_mapping[new_sdfg][i])

                    array = sdfg.arrays[self.name_mapping[sdfg][i]]
                    if isinstance(array_in_global, Scalar):
                        new_sdfg.add_scalar(self.name_mapping[new_sdfg][i], array_in_global.dtype, transient=False)
                    elif array_in_global.type == "Array":
                        new_sdfg.add_array(self.name_mapping[new_sdfg][i],
                                           array_in_global.shape,
                                           array_in_global.dtype,
                                           array_in_global.storage,
                                           transient=False,
                                           strides=array_in_global.strides,
                                           offset=array_in_global.offset)
                elif self.name_mapping[self.globalsdfg].get(i) is not None:
                    self.name_mapping[new_sdfg][i] = new_sdfg._find_new_name(i)
                    globalmemlets.append(i)
                    self.all_array_names.append(self.name_mapping[new_sdfg][i])
                    if i in read_names:
                        ins_in_new_sdfg.append(self.name_mapping[new_sdfg][i])
                    if i in write_names:
                        outs_in_new_sdfg.append(self.name_mapping[new_sdfg][i])

                    array = self.globalsdfg.arrays[self.name_mapping[self.globalsdfg][i]]
                    if isinstance(array_in_global, Scalar):
                        new_sdfg.add_scalar(self.name_mapping[new_sdfg][i], array_in_global.dtype, transient=False)
                    elif array_in_global.type == "Array":
                        new_sdfg.add_array(self.name_mapping[new_sdfg][i],
                                           array_in_global.shape,
                                           array_in_global.dtype,
                                           array_in_global.storage,
                                           transient=False,
                                           strides=array_in_global.strides,
                                           offset=array_in_global.offset)

        internal_sdfg = substate.add_nested_sdfg(new_sdfg,
                                                 sdfg,
                                                 ins_in_new_sdfg,
                                                 outs_in_new_sdfg,
                                                 symbol_mapping=sym_dict)

        # Now adding memlets
        for i in self.libstates:
            memlet = "0"
            if i in write_names:
                ast_utils.add_memlet_write(substate, self.name_mapping[sdfg][i], internal_sdfg,
                                           self.name_mapping[new_sdfg][i], memlet)
            if i in read_names:
                ast_utils.add_memlet_read(substate, self.name_mapping[sdfg][i], internal_sdfg,
                                          self.name_mapping[new_sdfg][i], memlet)

        for i in variables_in_call:

            local_name = parameters[variables_in_call.index(i)]
            if self.name_mapping.get(sdfg).get(ast_utils.get_name(i)) is not None:
                var = sdfg.arrays.get(self.name_mapping[sdfg][ast_utils.get_name(i)])
                mapped_name = self.name_mapping[sdfg][ast_utils.get_name(i)]
            # TODO: FIx symbols in function calls
            elif ast_utils.get_name(i) in sdfg.symbols:
                var = ast_utils.get_name(i)
                mapped_name = ast_utils.get_name(i)
            elif self.name_mapping.get(self.globalsdfg).get(ast_utils.get_name(i)) is not None:
                var = self.globalsdfg.arrays.get(self.name_mapping[self.globalsdfg][ast_utils.get_name(i)])
                mapped_name = self.name_mapping[self.globalsdfg][ast_utils.get_name(i)]
            else:
                raise NameError("Variable name not found: " + ast_utils.get_name(i))

            if not hasattr(var, "shape") or len(var.shape) == 0:
                memlet = ""
            elif (len(var.shape) == 1 and var.shape[0] == 1):
                memlet = "0"
            else:
                memlet = ast_utils.generate_memlet(i, sdfg, self)

            found = False
            for elem in views:
                if mapped_name == elem[0] and elem[3] == variables_in_call.index(i):
                    found = True

                    if local_name.name in write_names:
                        memlet = subsets.Range([(0, s - 1, 1) for s in sdfg.arrays[elem[2].label].shape])
                        substate.add_memlet_path(internal_sdfg,
                                                 elem[2],
                                                 src_conn=self.name_mapping[new_sdfg][local_name.name],
                                                 memlet=Memlet(expr=elem[2].label, subset=memlet))
                    if local_name.name in read_names:
                        memlet = subsets.Range([(0, s - 1, 1) for s in sdfg.arrays[elem[1].label].shape])
                        substate.add_memlet_path(elem[1],
                                                 internal_sdfg,
                                                 dst_conn=self.name_mapping[new_sdfg][local_name.name],
                                                 memlet=Memlet(expr=elem[1].label, subset=memlet))

            if not found:
                if local_name.name in write_names:
                    ast_utils.add_memlet_write(substate, mapped_name, internal_sdfg,
                                               self.name_mapping[new_sdfg][local_name.name], memlet)
                if local_name.name in read_names:
                    ast_utils.add_memlet_read(substate, mapped_name, internal_sdfg,
                                              self.name_mapping[new_sdfg][local_name.name], memlet)

        for i in addedmemlets:

            memlet = ast_utils.generate_memlet(ast_internal_classes.Name_Node(name=i), sdfg, self)
            if local_name.name in write_names:
                ast_utils.add_memlet_write(substate, self.name_mapping[sdfg][i], internal_sdfg,
                                           self.name_mapping[new_sdfg][i], memlet)
            if local_name.name in read_names:
                ast_utils.add_memlet_read(substate, self.name_mapping[sdfg][i], internal_sdfg,
                                          self.name_mapping[new_sdfg][i], memlet)
        for i in globalmemlets:

            memlet = ast_utils.generate_memlet(ast_internal_classes.Name_Node(name=i), sdfg, self)
            if local_name.name in write_names:
                ast_utils.add_memlet_write(substate, self.name_mapping[self.globalsdfg][i], internal_sdfg,
                                           self.name_mapping[new_sdfg][i], memlet)
            if local_name.name in read_names:
                ast_utils.add_memlet_read(substate, self.name_mapping[self.globalsdfg][i], internal_sdfg,
                                          self.name_mapping[new_sdfg][i], memlet)

        #Finally, now that the nested sdfg is built and the memlets are added, we can parse the internal of the subroutine and add it to the SDFG.

        if node.execution_part is not None:
            for j in node.specification_part.uses:
                for k in j.list:
                    if self.contexts.get(new_sdfg.name) is None:
                        self.contexts[new_sdfg.name] = ast_utils.Context(name=new_sdfg.name)
                    if self.contexts[new_sdfg.name].constants.get(
                            ast_utils.get_name(k)) is None and self.contexts[self.globalsdfg.name].constants.get(
                                ast_utils.get_name(k)) is not None:
                        self.contexts[new_sdfg.name].constants[ast_utils.get_name(k)] = self.contexts[
                            self.globalsdfg.name].constants[ast_utils.get_name(k)]

                    pass
            for j in node.specification_part.specifications:
                self.declstmt2sdfg(j, new_sdfg)
            for i in assigns:
                self.translate(i, new_sdfg)
            self.translate(node.execution_part, new_sdfg)

    def binop2sdfg(self, node: ast_internal_classes.BinOp_Node, sdfg: SDFG):
        """
        This parses binary operations to tasklets in a new state or creates
        a function call with a nested SDFG if the operation is a function
        call rather than a simple assignment.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        """

        calls = ast_transforms.FindFunctionCalls()
        calls.visit(node)
        if len(calls.nodes) == 1:
            augmented_call = calls.nodes[0]
            if augmented_call.name.name not in ["sqrt", "exp", "pow", "max", "min", "abs", "tanh", "__dace_epsilon"]:
                augmented_call.args.append(node.lval)
                augmented_call.hasret = True
                self.call2sdfg(augmented_call, sdfg)
                return

        outputnodefinder = ast_transforms.FindOutputs()
        outputnodefinder.visit(node)
        output_vars = outputnodefinder.nodes
        output_names = []
        output_names_tasklet = []

        for i in output_vars:
            mapped_name = self.get_name_mapping_in_context(sdfg).get(i.name)
            arrays = self.get_arrays_in_context(sdfg)

            if mapped_name in arrays and mapped_name not in output_names:
                output_names.append(mapped_name)
                output_names_tasklet.append(i.name)

        inputnodefinder = ast_transforms.FindInputs()
        inputnodefinder.visit(node)
        input_vars = inputnodefinder.nodes
        input_names = []
        input_names_tasklet = []

        for i in input_vars:
            mapped_name = self.get_name_mapping_in_context(sdfg).get(i.name)
            arrays = self.get_arrays_in_context(sdfg)
            if i.name in sdfg.symbols:
                continue
            if mapped_name in arrays:  # and mapped_name not in input_names:
                count = input_names.count(mapped_name)
                input_names.append(mapped_name)
                input_names_tasklet.append(i.name + "_" + str(count) + "_in")

        substate = ast_utils.add_simple_state_to_sdfg(
            self, sdfg, "_state_l" + str(node.line_number[0]) + "_c" + str(node.line_number[1]))

        output_names_changed = [o_t + "_out" for o_t in output_names]

        tasklet = ast_utils.add_tasklet(substate, "_l" + str(node.line_number[0]) + "_c" + str(node.line_number[1]),
                                        input_names_tasklet, output_names_changed, "text", node.line_number,
                                        self.file_name)

        for i, j in zip(input_names, input_names_tasklet):
            memlet_range = self.get_memlet_range(sdfg, input_vars, i, j)
            ast_utils.add_memlet_read(substate, i, tasklet, j, memlet_range)

        for i, j, k in zip(output_names, output_names_tasklet, output_names_changed):

            memlet_range = self.get_memlet_range(sdfg, output_vars, i, j)
            ast_utils.add_memlet_write(substate, i, tasklet, k, memlet_range)
        tw = ast_utils.TaskletWriter(output_names, output_names_changed, sdfg, self.name_mapping, input_names,
                                     input_names_tasklet)

        text = tw.write_code(node)
        tasklet.code = CodeBlock(text, lang.Python)

    def call2sdfg(self, node: ast_internal_classes.Call_Expr_Node, sdfg: SDFG):
        """
        This parses function calls to a nested SDFG 
        or creates a tasklet with an external library call.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        """

        self.last_call_expression[sdfg] = node.args
        match_found = False
        rettype = "INTEGER"
        hasret = False
        if node.name in self.functions_and_subroutines:
            for i in self.top_level.function_definitions:
                if i.name == node.name:
                    self.function2sdfg(i, sdfg)
                    return
            for i in self.top_level.subroutine_definitions:
                if i.name == node.name:
                    self.subroutine2sdfg(i, sdfg)
                    return
            for j in self.top_level.modules:
                for i in j.function_definitions:
                    if i.name == node.name:
                        self.function2sdfg(i, sdfg)
                        return
                for i in j.subroutine_definitions:
                    if i.name == node.name:
                        self.subroutine2sdfg(i, sdfg)
                        return
        else:
            # This part handles the case that it's an external library call
            libstate = self.libraries.get(node.name.name)
            if not isinstance(rettype, ast_internal_classes.Void) and hasattr(node, "hasret"):
                if node.hasret:
                    hasret = True
                    retval = node.args.pop(len(node.args) - 1)
            if node.name == "free":
                return
            input_names_tasklet = {}
            output_names_tasklet = []
            input_names = []
            output_names = []
            special_list_in = {}
            special_list_out = []
            if libstate is not None:
                special_list_in[self.name_mapping[sdfg][libstate] + "_task"] = dtypes.pointer(
                    sdfg.arrays.get(self.name_mapping[sdfg][libstate]).dtype)
                special_list_out.append(self.name_mapping[sdfg][libstate] + "_task_out")
            used_vars = [
                node for node in ast_transforms.mywalk(node) if isinstance(node, ast_internal_classes.Name_Node)
            ]

            for i in used_vars:
                for j in sdfg.arrays:
                    if self.name_mapping.get(sdfg).get(i.name) == j and j not in input_names:
                        elem = sdfg.arrays.get(j)
                        scalar = False
                        if len(elem.shape) == 0:
                            scalar = True
                        elif (len(elem.shape) == 1 and elem.shape[0] == 1):
                            scalar = True
                        if not scalar and not node.name.name in ["fprintf", "printf"]:
                            output_names.append(j)
                            output_names_tasklet.append(i.name)

                        input_names_tasklet[i.name] = dtypes.pointer(elem.dtype)
                        input_names.append(j)

            output_names_changed = []
            for o, o_t in zip(output_names, output_names_tasklet):
                output_names_changed.append(o_t + "_out")

            tw = ast_utils.TaskletWriter(output_names_tasklet.copy(), output_names_changed.copy(), sdfg,
                                         self.name_mapping)
            if not isinstance(rettype, ast_internal_classes.Void) and hasret:
                special_list_in[retval.name] = pointer(self.get_dace_type(rettype))
                special_list_out.append(retval.name + "_out")
                text = tw.write_code(
                    ast_internal_classes.BinOp_Node(lval=retval, op="=", rval=node, line_number=node.line_number))

            else:
                text = tw.write_code(node)
            substate = ast_utils.add_simple_state_to_sdfg(self, sdfg, "_state" + str(node.line_number[0]))

            tasklet = ast_utils.add_tasklet(substate, str(node.line_number[0]), {
                **input_names_tasklet,
                **special_list_in
            }, output_names_changed + special_list_out, "text", node.line_number, self.file_name)
            if libstate is not None:
                ast_utils.add_memlet_read(substate, self.name_mapping[sdfg][libstate], tasklet,
                                          self.name_mapping[sdfg][libstate] + "_task", "0")

                ast_utils.add_memlet_write(substate, self.name_mapping[sdfg][libstate], tasklet,
                                           self.name_mapping[sdfg][libstate] + "_task_out", "0")
            if not isinstance(rettype, ast_internal_classes.Void) and hasret:
                ast_utils.add_memlet_read(substate, self.name_mapping[sdfg][retval.name], tasklet, retval.name, "0")

                ast_utils.add_memlet_write(substate, self.name_mapping[sdfg][retval.name], tasklet,
                                           retval.name + "_out", "0")

            for i, j in zip(input_names, input_names_tasklet):
                memlet_range = self.get_memlet_range(sdfg, used_vars, i, j)
                ast_utils.add_memlet_read(substate, i, tasklet, j, memlet_range)

            for i, j, k in zip(output_names, output_names_tasklet, output_names_changed):

                memlet_range = self.get_memlet_range(sdfg, used_vars, i, j)
                ast_utils.add_memlet_write(substate, i, tasklet, k, memlet_range)

            setattr(tasklet, "code", CodeBlock(text, lang.Python))

    def declstmt2sdfg(self, node: ast_internal_classes.Decl_Stmt_Node, sdfg: SDFG):
        """
        This function translates a variable declaration statement to an access node on the sdfg
        :param node: The node to translate
        :param sdfg: The sdfg to attach the access node to
        :note This function is the top level of the declaration, most implementation is in vardecl2sdfg
        """
        for i in node.vardecl:
            self.translate(i, sdfg)

    def vardecl2sdfg(self, node: ast_internal_classes.Var_Decl_Node, sdfg: SDFG):
        """
        This function translates a variable declaration to an access node on the sdfg
        :param node: The node to translate
        :param sdfg: The sdfg to attach the access node to

        """
        #if the sdfg is the toplevel-sdfg, the variable is a global variable
        transient = True
        # find the type
        datatype = self.get_dace_type(node.type)
        if hasattr(node, "alloc"):
            if node.alloc:
                self.unallocated_arrays.append([node.name, datatype, sdfg, transient])
                return
        # get the dimensions
        if node.sizes is not None:
            sizes = []
            offset = []
            offset_value = -1
            for i in node.sizes:
                tw = ast_utils.TaskletWriter([], [], sdfg, self.name_mapping)
                text = tw.write_code(i)
                sizes.append(sym.pystr_to_symbolic(text))
                offset.append(offset_value)

        else:
            sizes = None
        # create and check name - if variable is already defined (function argument and defined in declaration part) simply stop
        if self.name_mapping[sdfg].get(node.name) is not None:
            return

        if node.name in sdfg.symbols:
            return

        self.name_mapping[sdfg][node.name] = sdfg._find_new_name(node.name)

        if sizes is None:
            sdfg.add_scalar(self.name_mapping[sdfg][node.name], dtype=datatype, transient=transient)
        else:
            strides = [dat._prod(sizes[:i]) for i in range(len(sizes))]
            sdfg.add_array(self.name_mapping[sdfg][node.name],
                           shape=sizes,
                           dtype=datatype,
                           offset=offset,
                           strides=strides,
                           transient=transient)

        self.all_array_names.append(self.name_mapping[sdfg][node.name])
        if self.contexts.get(sdfg.name) is None:
            self.contexts[sdfg.name] = ast_utils.Context(name=sdfg.name)
        if node.name not in self.contexts[sdfg.name].containers:
            self.contexts[sdfg.name].containers.append(node.name)

def create_ast_from_string(
    source_string: str,
    sdfg_name: str,
    transform: bool = False,
    normalize_offsets: bool = False
):
    """
    Creates an AST from a Fortran file in a string
    :param source_string: The fortran file as a string
    :param sdfg_name: The name to be given to the resulting SDFG
    :return: The resulting AST

    """
    parser = pf().create(std="f2008")
    reader = fsr(source_string)
    ast = parser(reader)
    tables = SymbolTable
    own_ast = ast_components.InternalFortranAst(ast, tables)
    program = own_ast.create_ast(ast)

    functions_and_subroutines_builder = ast_transforms.FindFunctionAndSubroutines()
    functions_and_subroutines_builder.visit(program)
    functions_and_subroutines = functions_and_subroutines_builder.nodes

    if transform:
        program = ast_transforms.functionStatementEliminator(program)
        program = ast_transforms.CallToArray(functions_and_subroutines_builder.nodes).visit(program)
        program = ast_transforms.CallExtractor().visit(program)
        program = ast_transforms.SignToIf().visit(program)
        program = ast_transforms.ArrayToLoop(program).visit(program)
        program = ast_transforms.SumToLoop(program).visit(program)
        program = ast_transforms.ForDeclarer().visit(program)
        program = ast_transforms.IndexExtractor(program, normalize_offsets).visit(program)

    return (program, own_ast)

def create_sdfg_from_string(
    source_string: str,
    sdfg_name: str,
    normalize_offsets: bool = False
):
    """
    Creates an SDFG from a fortran file in a string
    :param source_string: The fortran file as a string
    :param sdfg_name: The name to be given to the resulting SDFG
    :return: The resulting SDFG
    
    """
    parser = pf().create(std="f2008")
    reader = fsr(source_string)
    ast = parser(reader)
    tables = SymbolTable
    own_ast = ast_components.InternalFortranAst(ast, tables)
    program = own_ast.create_ast(ast)
    functions_and_subroutines_builder = ast_transforms.FindFunctionAndSubroutines()
    functions_and_subroutines_builder.visit(program)
    own_ast.functions_and_subroutines = functions_and_subroutines_builder.nodes
    program = ast_transforms.functionStatementEliminator(program)
    program = ast_transforms.CallToArray(functions_and_subroutines_builder.nodes).visit(program)
    program = ast_transforms.CallExtractor().visit(program)
    program = ast_transforms.SignToIf().visit(program)
    program = ast_transforms.ArrayToLoop(program).visit(program)
    program = ast_transforms.SumToLoop(program).visit(program)
    program = ast_transforms.ForDeclarer().visit(program)
    program = ast_transforms.IndexExtractor(program, normalize_offsets).visit(program)
    ast2sdfg = AST_translator(own_ast, __file__)
    sdfg = SDFG(sdfg_name)
    ast2sdfg.top_level = program
    ast2sdfg.globalsdfg = sdfg
    ast2sdfg.translate(program, sdfg)

    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.NestedSDFG):
            if 'test_function' in node.sdfg.name:
                sdfg = node.sdfg
                break
    sdfg.parent = None
    sdfg.parent_sdfg = None
    sdfg.parent_nsdfg_node = None
    sdfg.reset_sdfg_list()
    return sdfg


def create_sdfg_from_fortran_file(source_string: str):
    """
    Creates an SDFG from a fortran file
    :param source_string: The fortran file name
    :return: The resulting SDFG

    """
    parser = pf().create(std="f2008")
    reader = ffr(source_string)
    ast = parser(reader)
    tables = SymbolTable
    own_ast = ast_components.InternalFortranAst(ast, tables)
    program = own_ast.create_ast(ast)
    functions_and_subroutines_builder = ast_transforms.FindFunctionAndSubroutines()
    functions_and_subroutines_builder.visit(program)
    own_ast.functions_and_subroutines = functions_and_subroutines_builder.nodes
    program = ast_transforms.functionStatementEliminator(program)
    program = ast_transforms.CallToArray(functions_and_subroutines_builder.nodes).visit(program)
    program = ast_transforms.CallExtractor().visit(program)
    program = ast_transforms.SignToIf().visit(program)
    program = ast_transforms.ArrayToLoop(program).visit(program)
    program = ast_transforms.SumToLoop(program).visit(program)
    program = ast_transforms.ForDeclarer().visit(program)
    program = ast_transforms.IndexExtractor(program).visit(program)
    ast2sdfg = AST_translator(own_ast, __file__)
    sdfg = SDFG(source_string)
    ast2sdfg.top_level = program
    ast2sdfg.globalsdfg = sdfg
    ast2sdfg.translate(program, sdfg)

    return sdfg
