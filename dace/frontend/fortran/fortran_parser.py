# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import copy
import os
import warnings
from copy import deepcopy as dpcp
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import List, Optional, Set, Dict, Tuple, Union

import networkx as nx
from fparser.common.readfortran import FortranStringReader
from fparser.common.readfortran import FortranStringReader as fsr
from fparser.two.Fortran2003 import Program, Name, Module_Stmt
from fparser.two.parser import ParserFactory as pf, ParserFactory
from fparser.two.symbol_table import SymbolTable
from fparser.two.utils import Base, walk

import dace.frontend.fortran.ast_components as ast_components
import dace.frontend.fortran.ast_internal_classes as ast_internal_classes
import dace.frontend.fortran.ast_transforms as ast_transforms
from dace import Language as lang
from dace import SDFG, InterstateEdge, Memlet, pointer, SDFGState
from dace import data as dat
from dace import dtypes
from dace import symbolic as sym
from dace.data import Scalar, Structure
from dace.frontend.fortran import ast_utils
from dace.frontend.fortran.ast_desugaring import ENTRY_POINT_OBJECT_CLASSES, NAMED_STMTS_OF_INTEREST_CLASSES, SPEC, \
    find_name_of_stmt, find_name_of_node, append_children, correct_for_function_calls, sort_modules, \
    deconstruct_enums, deconstruct_interface_calls, deconstruct_procedure_calls, prune_unused_objects, \
    deconstruct_associations, consolidate_uses, prune_branches, const_eval_nodes, lower_identifier_names, \
    inject_const_evals, remove_access_statements, ident_spec, ConstTypeInjection, ConstInjection, \
    make_practically_constant_arguments_constants, make_practically_constant_global_vars_constants, \
    exploit_locally_constant_variables, assign_globally_unique_subprogram_names, \
    create_global_initializers, convert_data_statements_into_assignments, deconstruct_statement_functions, \
    assign_globally_unique_variable_names
from dace.frontend.fortran.ast_internal_classes import FNode, Main_Program_Node
from dace.frontend.fortran.ast_utils import children_of_type
from dace.frontend.fortran.intrinsics import IntrinsicSDFGTransformation, NeedsTypeInferenceException
from dace.properties import CodeBlock
from dace.sdfg import nodes as nd
from dace.sdfg.state import BreakBlock, ConditionalBlock, ContinueBlock, ControlFlowRegion, LoopRegion

global_struct_instance_counter = 0


def find_access_in_destinations(substate, substate_destinations, name):
    wv = None
    already_there = False
    for i in substate_destinations:
        if i.data == name:
            wv = i
            already_there = True
            break
    if not already_there:
        wv = substate.add_write(name)
    return wv, already_there


def find_access_in_sources(substate, substate_sources, name):
    re = None
    already_there = False
    for i in substate_sources:
        if i.data == name:
            re = i
            already_there = True
            break
    if not already_there:
        re = substate.add_read(name)
    return re, already_there


def add_views_recursive(sdfg, name, datatype_to_add, struct_views, name_mapping, registered_types, chain,
                        actual_offsets_per_sdfg, names_of_object_in_parent_sdfg, actual_offsets_per_parent_sdfg):
    if not isinstance(datatype_to_add, dat.Structure):
        # print("Not adding: ", str(datatype_to_add))
        if isinstance(datatype_to_add, dat.ContainerArray):
            datatype_to_add = datatype_to_add.stype
    for i in datatype_to_add.members:
        current_dtype = datatype_to_add.members[i].dtype
        for other_type in registered_types:
            if current_dtype.dtype == registered_types[other_type].dtype:
                other_type_obj = registered_types[other_type]
                add_views_recursive(sdfg, name, datatype_to_add.members[i], struct_views, name_mapping,
                                    registered_types, chain + [i], actual_offsets_per_sdfg,
                                    names_of_object_in_parent_sdfg, actual_offsets_per_parent_sdfg)
                # for j in other_type_obj.members:
                #    sdfg.add_view(name_mapping[name] + "_" + i +"_"+ j,other_type_obj.members[j].shape,other_type_obj.members[j].dtype)
                #    name_mapping[name + "_" + i +"_"+ j] = name_mapping[name] + "_" + i +"_"+ j
                #    struct_views[name_mapping[name] + "_" + i+"_"+ j]=[name_mapping[name],i,j]
        if len(chain) > 0:
            join_chain = "_" + "_".join(chain)
        else:
            join_chain = ""
        current_member = datatype_to_add.members[i]

        if str(datatype_to_add.members[i].dtype.base_type) in registered_types:

            view_to_member = dat.View.view(datatype_to_add.members[i])
            if sdfg.arrays.get(name_mapping[name] + join_chain + "_" + i) is None:
                sdfg.arrays[name_mapping[name] + join_chain + "_" + i] = view_to_member
        else:
            if sdfg.arrays.get(name_mapping[name] + join_chain + "_" + i) is None:
                sdfg.add_view(name_mapping[name] + join_chain + "_" + i, datatype_to_add.members[i].shape,
                              datatype_to_add.members[i].dtype, strides=datatype_to_add.members[i].strides)
        if names_of_object_in_parent_sdfg.get(name_mapping[name]) is not None:
            if actual_offsets_per_parent_sdfg.get(
                    names_of_object_in_parent_sdfg[name_mapping[name]] + join_chain + "_" + i) is not None:
                actual_offsets_per_sdfg[name_mapping[name] + join_chain + "_" + i] = actual_offsets_per_parent_sdfg[
                    names_of_object_in_parent_sdfg[name_mapping[name]] + join_chain + "_" + i]
            else:
                # print("No offsets in sdfg: ",sdfg.name ," for: ",names_of_object_in_parent_sdfg[name_mapping[name]]+ join_chain + "_" + i)
                actual_offsets_per_sdfg[name_mapping[name] + join_chain + "_" + i] = [1] * len(
                    datatype_to_add.members[i].shape)
        name_mapping[name_mapping[name] + join_chain + "_" + i] = name_mapping[name] + join_chain + "_" + i
        struct_views[name_mapping[name] + join_chain + "_" + i] = [name_mapping[name]] + chain + [i]


def add_deferred_shape_assigns_for_structs(structures: ast_transforms.Structures,
                                           decl: ast_internal_classes.Var_Decl_Node, sdfg: SDFG,
                                           assign_state: SDFGState, name: str, name_: str, placeholders,
                                           placeholders_offsets, object, names_to_replace, actual_offsets_per_sdfg):
    if not structures.is_struct(decl.type):
        # print("Not adding defferred shape assigns for: ", decl.type,decl.name)
        return

    if isinstance(object, dat.ContainerArray):
        struct_type = object.stype
    else:
        struct_type = object
    global global_struct_instance_counter
    local_counter = global_struct_instance_counter
    global_struct_instance_counter += 1
    overall_ast_struct_type = structures.get_definition(decl.type)
    counter = 0
    listofmember = list(struct_type.members)
    # print("Struct: "+decl.name +" Struct members: "+ str(len(listofmember))+ " Definition members: "+str(len(list(overall_ast_struct_type.vars.items()))))

    for ast_struct_type in overall_ast_struct_type.vars.items():
        ast_struct_type = ast_struct_type[1]
        var = struct_type.members[ast_struct_type.name]

        if isinstance(var, dat.ContainerArray):
            var_type = var.stype
        else:
            var_type = var

        # print(ast_struct_type.name,var_type.__class__)
        if isinstance(struct_type.members[ast_struct_type.name], dat.Structure):

            add_deferred_shape_assigns_for_structs(structures, ast_struct_type, sdfg, assign_state,
                                                   f"{name}->{ast_struct_type.name}", f"{ast_struct_type.name}_{name_}",
                                                   placeholders, placeholders_offsets,
                                                   object.members[ast_struct_type.name], names_to_replace,
                                                   actual_offsets_per_sdfg)
        elif isinstance(var_type, dat.Structure):
            add_deferred_shape_assigns_for_structs(structures, ast_struct_type, sdfg, assign_state,
                                                   f"{name}->{ast_struct_type.name}", f"{ast_struct_type.name}_{name_}",
                                                   placeholders, placeholders_offsets, var_type, names_to_replace,
                                                   actual_offsets_per_sdfg)
        # print(ast_struct_type)
        # print(ast_struct_type.__class__)

        if ast_struct_type.sizes is None or len(ast_struct_type.sizes) == 0:
            continue
        offsets_to_replace = []
        sanity_count = 0

        for offset in ast_struct_type.offsets:
            if isinstance(offset, ast_internal_classes.Name_Node):
                if hasattr(offset, "name"):
                    if sdfg.symbols.get(offset.name) is None:
                        sdfg.add_symbol(offset.name, dtypes.int32)
                sanity_count += 1
                if offset.name.startswith('__f2dace_SOA'):
                    newoffset = offset.name + "_" + name_ + "_" + str(local_counter)
                    sdfg.append_global_code(f"{dtypes.int32.ctype} {newoffset};\n")
                    # prog hack
                    if name.endswith("prog"):
                        sdfg.append_init_code(f"{newoffset} = {name}[0]->{offset.name};\n")
                    else:
                        sdfg.append_init_code(f"{newoffset} = {name}->{offset.name};\n")

                    sdfg.add_symbol(newoffset, dtypes.int32)
                    offsets_to_replace.append(newoffset)
                    names_to_replace[offset.name] = newoffset
                else:
                    # print("not replacing",offset.name)
                    offsets_to_replace.append(offset.name)
            else:
                sanity_count += 1
                # print("not replacing not namenode",offset)
                offsets_to_replace.append(offset)
        if sanity_count == len(ast_struct_type.offsets):
            # print("adding offsets for: "+name.replace("->","_")+"_"+ast_struct_type.name)
            actual_offsets_per_sdfg[name.replace("->", "_") + "_" + ast_struct_type.name] = offsets_to_replace

        # for assumed shape, all vars starts with the same prefix
        for size in ast_struct_type.sizes:
            if isinstance(size, ast_internal_classes.Name_Node):  # and  size.name.startswith('__f2dace_A'):

                if hasattr(size, "name"):
                    if sdfg.symbols.get(size.name) is None:
                        # new_name=sdfg._find_new_name(size.name)
                        sdfg.add_symbol(size.name, dtypes.int32)

                    if size.name.startswith('__f2dace_SA'):
                        # newsize=ast_internal_classes.Name_Node(name=size.name+"_"+str(local_counter),parent=size.parent,type=size.type)
                        newsize = size.name + "_" + name_ + "_" + str(local_counter)
                        names_to_replace[size.name] = newsize
                        # var_type.sizes[var_type.sizes.index(size)]=newsize
                        sdfg.append_global_code(f"{dtypes.int32.ctype} {newsize};\n")
                        if name.endswith("prog"):
                            sdfg.append_init_code(f"{newsize} = {name}[0]->{size.name};\n")
                        else:
                            sdfg.append_init_code(f"{newsize} = {name}->{size.name};\n")
                        sdfg.add_symbol(newsize, dtypes.int32)
                        if isinstance(object, dat.Structure):
                            shape2 = dpcp(object.members[ast_struct_type.name].shape)
                        else:
                            shape2 = dpcp(object.stype.members[ast_struct_type.name].shape)
                        shapelist = list(shape2)
                        shapelist[ast_struct_type.sizes.index(size)] = sym.pystr_to_symbolic(newsize)
                        shape_replace = tuple(shapelist)
                        viewname = f"{name}->{ast_struct_type.name}"

                        viewname = viewname.replace("->", "_")
                        # view=sdfg.arrays[viewname]
                        strides = [dat._prod(shapelist[:i]) for i in range(len(shapelist))]
                        if isinstance(object.members[ast_struct_type.name], dat.ContainerArray):
                            tmpobject = dat.ContainerArray(object.members[ast_struct_type.name].stype, shape_replace,
                                                           strides=strides)


                        elif isinstance(object.members[ast_struct_type.name], dat.Array):
                            tmpobject = dat.Array(object.members[ast_struct_type.name].dtype, shape_replace,
                                                  strides=strides)

                        else:
                            raise ValueError("Unknown type" + str(tmpobject.__class__))
                        object.members.pop(ast_struct_type.name)
                        object.members[ast_struct_type.name] = tmpobject
                        tmpview = dat.View.view(object.members[ast_struct_type.name])
                        if sdfg.arrays.get(viewname) is not None:
                            del sdfg.arrays[viewname]
                        sdfg.arrays[viewname] = tmpview
                        # if placeholders.get(size.name) is not None:
                        #    placeholders[newsize]=placeholders[size.name]


class AST_translator:
    """
    This class is responsible for translating the internal AST into a SDFG.
    """

    def __init__(self, source: str, multiple_sdfgs: bool = False, startpoint=None, sdfg_path=None,
                 toplevel_subroutine: Optional[str] = None, subroutine_used_names: Optional[Set[str]] = None,
                 normalize_offsets=False, do_not_make_internal_variables_argument: bool = False):
        """
        :ast: The internal fortran AST to be used for translation
        :source: The source file name from which the AST was generated
        :do_not_make_internal_variables_argument: Avoid turning internal variables of the entry point into arguments.
            This essentially avoids the hack with `transient_mode = False`, since we can rely on `startpoint` for
            arbitrary entry point anyway.
        """
        # TODO: Refactor the callers who rely on the hack with `transient_mode = False`, then remove the
        #  `do_not_make_internal_variables_argument` argument entirely, since we don't need it at that point.
        self.sdfg_path = sdfg_path
        self.count_of_struct_symbols_lifted = 0
        self.registered_types = {}
        self.transient_mode = True
        self.startpoint = startpoint
        self.top_level = None
        self.globalsdfg = None
        self.multiple_sdfgs = multiple_sdfgs
        self.name_mapping = ast_utils.NameMap()
        self.actual_offsets_per_sdfg = {}
        self.names_of_object_in_parent_sdfg = {}
        self.contexts = {}
        self.views = 0
        self.libstates = []
        self.file_name = source
        self.unallocated_arrays = []
        self.all_array_names = []
        self.last_sdfg_states = {}
        self.module_vars = []
        self.sdfgs_count = 0
        self.libraries = {}
        self.local_not_transient_because_assign = {}
        self.struct_views = {}
        self.last_call_expression = {}
        self.struct_view_count = 0
        self.structures = None
        self.placeholders = None
        self.placeholders_offsets = None
        self.replace_names = {}
        self.toplevel_subroutine = toplevel_subroutine
        self.subroutine_used_names = subroutine_used_names
        self.normalize_offsets = normalize_offsets
        self.temporary_sym_dict = {}
        self.temporary_link_to_parent = {}
        self.do_not_make_internal_variables_argument = do_not_make_internal_variables_argument
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
            ast_internal_classes.Break_Node: self.break2sdfg,
            ast_internal_classes.Continue_Node: self.continue2sdfg,
            ast_internal_classes.Derived_Type_Def_Node: self.derivedtypedef2sdfg,
            ast_internal_classes.Pointer_Assignment_Stmt_Node: self.pointerassignment2sdfg,
            ast_internal_classes.While_Stmt_Node: self.whilestmt2sdfg,
        }

    def get_dace_type(self, type):
        """
        This function matches the fortran type to the corresponding dace type
        by referencing the ast_utils.fortrantypes2dacetypes dictionary.
        """
        if isinstance(type, str):
            if type in ast_utils.fortrantypes2dacetypes:
                return ast_utils.fortrantypes2dacetypes[type]
            elif type in self.registered_types:
                return self.registered_types[type]
            else:
                # TODO: This is bandaid.
                if type == "VOID":
                    return ast_utils.fortrantypes2dacetypes["DOUBLE"]
                    raise ValueError("Unknown type " + type)
                else:
                    raise ValueError("Unknown type " + type)

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
                return ast_utils.generate_memlet(o_v, sdfg, self, self.normalize_offsets)


    def _add_tasklet(self, substate: SDFGState, name: str, vars_in: Set[str], vars_out: Set[str], code: str,
                     debuginfo: list, source: str):
        tasklet = substate.add_tasklet(name="T" + name, inputs=vars_in, outputs=vars_out, code=code,
                                       debuginfo=dtypes.DebugInfo(start_line=debuginfo[0], start_column=debuginfo[1],
                                                                  filename=source), language=dtypes.Language.Python)
        return tasklet


    def _add_simple_state_to_cfg(self, cfg: ControlFlowRegion, state_name: str):
        if cfg in self.last_sdfg_states and self.last_sdfg_states[cfg] is not None:
            substate = cfg.add_state(state_name)
        else:
            substate = cfg.add_state(state_name, is_start_block=True)
        self._finish_add_state_to_cfg(cfg, substate)
        return substate


    def _finish_add_state_to_cfg(self, cfg: ControlFlowRegion, substate: SDFGState):
        if cfg in self.last_sdfg_states and self.last_sdfg_states[cfg] is not None:
            cfg.add_edge(self.last_sdfg_states[cfg], substate, InterstateEdge())
        self.last_sdfg_states[cfg] = substate


    def translate(self, node: ast_internal_classes.FNode, sdfg: SDFG, cfg: ControlFlowRegion):
        """
        This function is responsible for translating the AST into a SDFG.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        :param cfg: The control flow region into which the node should be translated
        :note: This function is recursive and will call itself for all child nodes
        :note: This function will call the appropriate function for the node type
        :note: The dictionary ast_elements, part of the class itself contains all functions that are called for the different node types
        """
        if node.__class__ in self.ast_elements:
            self.ast_elements[node.__class__](node, sdfg, cfg)
        elif isinstance(node, list):
            for i in node:
                self.translate(i, sdfg, cfg)
        else:
            warnings.warn(f"WARNING: {node.__class__.__name__}")

    def ast2sdfg(self, node: ast_internal_classes.Program_Node, sdfg: SDFG, cfg: ControlFlowRegion):
        """
        This function is responsible for translating the Fortran AST into a SDFG.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        :param cfg: The control flow region into which the node should be translated
        :note: This function is recursive and will call itself for all child nodes
        :note: This function will call the appropriate function for the node type
        :note: The dictionary ast_elements, part of the class itself contains all functions that are called for the different node types
        """
        self.globalsdfg = sdfg
        for i in node.modules:
            structs_lister = ast_transforms.StructLister()
            if i.specification_part is not None:
                structs_lister.visit(i.specification_part)
            struct_dep_graph = nx.DiGraph()
            for ii, name in zip(structs_lister.structs, structs_lister.names):
                if name not in struct_dep_graph.nodes:
                    struct_dep_graph.add_node(name)
                struct_deps_finder = ast_transforms.StructDependencyLister(structs_lister.names)
                struct_deps_finder.visit(ii)
                struct_deps = struct_deps_finder.structs_used
                # print(struct_deps)
                for j, pointing, point_name in zip(struct_deps, struct_deps_finder.is_pointer,
                                                   struct_deps_finder.pointer_names):
                    if j not in struct_dep_graph.nodes:
                        struct_dep_graph.add_node(j)
                    struct_dep_graph.add_edge(name, j, pointing=pointing, point_name=point_name)
            parse_order = list(reversed(list(nx.topological_sort(struct_dep_graph))))
            for jj in parse_order:
                for j in i.specification_part.typedecls:
                    if j.name.name == jj:
                        self.translate(j, sdfg, cfg)
                        if j.__class__.__name__ != "Derived_Type_Def_Node":
                            for k in j.vardecl:
                                self.module_vars.append((k.name, i.name))
            if i.specification_part is not None:

                # this works with CloudSC
                # unsure about ICON
                self.transient_mode = self.do_not_make_internal_variables_argument

                for j in i.specification_part.symbols:
                    self.translate(j, sdfg, cfg)
                    if isinstance(j, ast_internal_classes.Symbol_Array_Decl_Node):
                        self.module_vars.append((j.name, i.name))
                    elif isinstance(j, ast_internal_classes.Symbol_Decl_Node):
                        self.module_vars.append((j.name, i.name))
                    else:
                        raise ValueError("Unknown symbol type")
                for j in i.specification_part.specifications:
                    self.translate(j, sdfg, cfg)
                    for k in j.vardecl:
                        self.module_vars.append((k.name, i.name))
        # this works with CloudSC
        # unsure about ICON
        self.transient_mode = True
        ast_utils.add_simple_state_to_sdfg(self, sdfg, "GlobalDefEnd")
        if self.startpoint is None:
            self.startpoint = node.main_program
        assert self.startpoint is not None, "No main program or start point found"

        if self.startpoint.specification_part is not None:
            # this works with CloudSC
            # unsure about ICON
            self.transient_mode = self.do_not_make_internal_variables_argument

            for i in self.startpoint.specification_part.typedecls:
                self.translate(i, sdfg, cfg)
            for i in self.startpoint.specification_part.symbols:
                self.translate(i, sdfg, cfg)

            for i in self.startpoint.specification_part.specifications:
                self.translate(i, sdfg, cfg)
            for i in self.startpoint.specification_part.specifications:
                ast_utils.add_simple_state_to_sdfg(self, sdfg, "start_struct_size")
                assign_state = ast_utils.add_simple_state_to_sdfg(self, sdfg, "assign_struct_sizes")
                for decl in i.vardecl:
                    if decl.name in sdfg.symbols:
                        continue
                    add_deferred_shape_assigns_for_structs(self.structures, decl, sdfg, assign_state, decl.name,
                                                           decl.name, self.placeholders,
                                                           self.placeholders_offsets,
                                                           sdfg.arrays[self.name_mapping[sdfg][decl.name]],
                                                           self.replace_names,
                                                           self.actual_offsets_per_sdfg[sdfg])

        if not isinstance(self.startpoint, Main_Program_Node):
            # this works with CloudSC
            # unsure about ICON
            arg_names = [ast_utils.get_name(i) for i in self.startpoint.args]
            for arr_name, arr in sdfg.arrays.items():

                if arr.transient and arr_name in arg_names:
                    print(f"Changing the transient status to false of {arr_name} because it's a function argument")
                    arr.transient = False

        # for i in sdfg.arrays:
        #     if i in sdfg.symbols:
        #         sdfg.arrays.pop(i)

        self.transient_mode = True
        self.translate(self.startpoint.execution_part.execution, sdfg, cfg)
        sdfg.validate()

    def pointerassignment2sdfg(self, node: ast_internal_classes.Pointer_Assignment_Stmt_Node, sdfg: SDFG,
                               cfg: ControlFlowRegion):
        """
        This function is responsible for translating Fortran pointer assignments into a SDFG.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        :param cfg: The control flow region into which the node should be translated
        """
        if self.name_mapping[sdfg][node.name_pointer.name] in sdfg.arrays:
            shapenames = [sdfg.arrays[self.name_mapping[sdfg][node.name_pointer.name]].shape[i] for i in
                          range(len(sdfg.arrays[self.name_mapping[sdfg][node.name_pointer.name]].shape))]
            offsetnames = self.actual_offsets_per_sdfg[sdfg][node.name_pointer.name]
            [sdfg.arrays[self.name_mapping[sdfg][node.name_pointer.name]].offset[i] for i in
             range(len(sdfg.arrays[self.name_mapping[sdfg][node.name_pointer.name]].offset))]
            # for i in shapenames:
            #    if str(i) in sdfg.symbols:
            #        sdfg.symbols.pop(str(i))
            # if sdfg.parent_nsdfg_node is not None:
            # if str(i) in sdfg.parent_nsdfg_node.symbol_mapping:
            # sdfg.parent_nsdfg_node.symbol_mapping.pop(str(i))

            # for i in offsetnames:
            # if str(i) in sdfg.symbols:
            #    sdfg.symbols.pop(str(i))
            # if sdfg.parent_nsdfg_node is not None:
            # if str(i) in sdfg.parent_nsdfg_node.symbol_mapping:
            # sdfg.parent_nsdfg_node.symbol_mapping.pop(str(i))
            sdfg.arrays.pop(self.name_mapping[sdfg][node.name_pointer.name])
        if isinstance(node.name_target, ast_internal_classes.Data_Ref_Node):
            if node.name_target.parent_ref.name not in self.name_mapping[sdfg]:
                raise ValueError("Unknown variable " + node.name_target.name)
            if isinstance(node.name_target.part_ref, ast_internal_classes.Data_Ref_Node):
                self.name_mapping[sdfg][node.name_pointer.name] = self.name_mapping[sdfg][
                    node.name_target.parent_ref.name + "_" + node.name_target.part_ref.parent_ref.name + "_" + node.name_target.part_ref.part_ref.name]
                # self.replace_names[node.name_pointer.name]=self.name_mapping[sdfg][node.name_target.parent_ref.name+"_"+node.name_target.part_ref.parent_ref.name+"_"+node.name_target.part_ref.part_ref.name]
                target = sdfg.arrays[self.name_mapping[sdfg][
                    node.name_target.parent_ref.name + "_" + node.name_target.part_ref.parent_ref.name + "_" + node.name_target.part_ref.part_ref.name]]
                # for i in self.actual_offsets_per_sdfg[sdfg]:
                #    print(i)
                actual_offsets = self.actual_offsets_per_sdfg[sdfg][
                    node.name_target.parent_ref.name + "_" + node.name_target.part_ref.parent_ref.name + "_" + node.name_target.part_ref.part_ref.name]

                for i in shapenames:
                    self.replace_names[str(i)] = str(target.shape[shapenames.index(i)])
                for i in offsetnames:
                    self.replace_names[str(i)] = str(actual_offsets[offsetnames.index(i)])
            else:
                self.name_mapping[sdfg][node.name_pointer.name] = self.name_mapping[sdfg][
                    node.name_target.parent_ref.name + "_" + node.name_target.part_ref.name]
                self.replace_names[node.name_pointer.name] = self.name_mapping[sdfg][
                    node.name_target.parent_ref.name + "_" + node.name_target.part_ref.name]
                target = sdfg.arrays[
                    self.name_mapping[sdfg][node.name_target.parent_ref.name + "_" + node.name_target.part_ref.name]]
                actual_offsets = self.actual_offsets_per_sdfg[sdfg][
                    node.name_target.parent_ref.name + "_" + node.name_target.part_ref.name]
                for i in shapenames:
                    self.replace_names[str(i)] = str(target.shape[shapenames.index(i)])
                for i in offsetnames:
                    self.replace_names[str(i)] = str(actual_offsets[offsetnames.index(i)])

        elif isinstance(node.name_pointer, ast_internal_classes.Data_Ref_Node):
            raise ValueError("Not imlemented yet")

        else:
            if node.name_target.name not in self.name_mapping[sdfg]:
                raise ValueError("Unknown variable " + node.name_target.name)
            found = False
            for i in self.unallocated_arrays:
                if i[0] == node.name_pointer.name:
                    if found:
                        raise ValueError("Multiple unallocated arrays with the same name")
                    fount = True
                    self.unallocated_arrays.remove(i)
            self.name_mapping[sdfg][node.name_pointer.name] = self.name_mapping[sdfg][node.name_target.name]

    def derivedtypedef2sdfg(self, node: ast_internal_classes.Derived_Type_Def_Node, sdfg: SDFG,
                            cfg: ControlFlowRegion):
        """
        This function is responsible for registering Fortran derived type declarations into a SDFG as nested data types.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        :param cfg: The control flow region into which the node should be translated
        """
        name = node.name.name
        if node.component_part is None:
            components = []
        else:
            components = node.component_part.component_def_stmts
        dict_setup = {}
        for i in components:
            j = i.vars
            for k in j.vardecl:
                complex_datatype = False
                datatype = self.get_dace_type(k.type)
                if isinstance(datatype, dat.Structure):
                    complex_datatype = True
                if k.sizes:
                    sizes = []
                    offset = []
                    offset_value = 0 if self.normalize_offsets else -1
                    for i in k.sizes:
                        tw = ast_utils.TaskletWriter([], [], sdfg, self.name_mapping, placeholders=self.placeholders,
                                                     placeholders_offsets=self.placeholders_offsets,
                                                     rename_dict=self.replace_names)
                        text = tw.write_code(i)
                        sizes.append(sym.pystr_to_symbolic(text))
                        offset.append(offset_value)
                    strides = [dat._prod(sizes[:i]) for i in range(len(sizes))]
                    if not complex_datatype:
                        dict_setup[k.name] = dat.Array(
                            datatype,
                            sizes,
                            strides=strides,
                            offset=offset,
                        )
                    else:
                        dict_setup[k.name] = dat.ContainerArray(datatype, sizes, strides=strides, offset=offset)

                else:
                    if not complex_datatype:
                        dict_setup[k.name] = dat.Scalar(datatype)
                    else:
                        dict_setup[k.name] = datatype

        structure_obj = Structure(dict_setup, name)
        self.registered_types[name] = structure_obj

    def basicblock2sdfg(self, node: ast_internal_classes.Execution_Part_Node, sdfg: SDFG, cfg: ControlFlowRegion):
        """
        This function is responsible for translating Fortran basic blocks into a SDFG.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        :param cfg: The control flow region into which the node should be translated
        """

        for i in node.execution:
            self.translate(i, sdfg, cfg)

    def allocate2sdfg(self, node: ast_internal_classes.Allocate_Stmt_Node, sdfg: SDFG,
                      cfg: ControlFlowRegion):
        """
        This function is responsible for translating Fortran allocate statements into a SDFG.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        :param cfg: The control flow region into which the node should be translated
        :note: We pair the allocate with a list of unallocated arrays.
        """
        for i in node.allocation_list:
            for j in self.unallocated_arrays:
                if j[0] == i.name.name and sdfg == j[2]:
                    datatype = j[1]
                    transient = j[3]
                    self.unallocated_arrays.remove(j)
                    offset_value = 0 if self.normalize_offsets else -1
                    sizes = []
                    offset = []
                    for j in i.shape.shape_list:
                        tw = ast_utils.TaskletWriter([], [], sdfg, self.name_mapping, placeholders=self.placeholders,
                                                     placeholders_offsets=self.placeholders_offsets,
                                                     rename_dict=self.replace_names)
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

    def write2sdfg(self, node: ast_internal_classes.Write_Stmt_Node, sdfg: SDFG, cfg: ControlFlowRegion):
        # TODO implement
        print("Uh oh")
        # raise NotImplementedError("Fortran write statements are not implemented yet")


    def ifstmt2sdfg(self, node: ast_internal_classes.If_Stmt_Node, sdfg: SDFG, cfg: ControlFlowRegion):
        """
        This function is responsible for translating Fortran if statements into a SDFG.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        :param cfg: The control flow region into which the node should be translated
        """
        name = f"Conditional_l_{str(node.line_number[0])}_c_{str(node.line_number[1])}"

        prev_block = None if cfg not in self.last_sdfg_states else self.last_sdfg_states[cfg]
        is_start = prev_block is None

        condition = ast_utils.ProcessedWriter(sdfg, self.name_mapping, self.placeholders, self.placeholders_offsets,
                                              self.replace_names).write_code(node.cond)

        cond_block = ConditionalBlock(name)
        cfg.add_node(cond_block, ensure_unique_name=True, is_start_block=is_start)
        if not is_start:
            cfg.add_edge(self.last_sdfg_states[cfg], cond_block, InterstateEdge())
        self.last_sdfg_states[cfg] = cond_block

        if_body = ControlFlowRegion(cond_block.label + '_if_body')
        cond_block.add_branch(CodeBlock(condition), if_body)
        self.translate(node.body, sdfg, if_body)
        if len(if_body.nodes()) == 0:
            # If there's nothing inside the branch, add a noop state to get a valid SDFG and let simplify take care of
            # the rest.
            if_body.add_state('noop', is_start_block=True)

        if len(node.body_else.execution) > 0:
            else_body = ControlFlowRegion(cond_block.label + '_else_body')
            cond_block.add_branch(None, else_body)
            self.translate(node.body_else, sdfg, else_body)

            if len(else_body.nodes()) == 0:
                else_body.add_state('noop', is_start_block=True)


    def whilestmt2sdfg(self, node: ast_internal_classes.While_Stmt_Node, sdfg: SDFG, cfg: ControlFlowRegion):
        """
        This function is responsible for translating Fortran while statements into a SDFG.
        :param node: The while statement node to be translated
        :param sdfg: The SDFG to which the node should be translated
        :param cfg: The control flow region to which the node should be translated
        """
        name = "While_l_" + str(node.line_number[0]) + "_c_" + str(node.line_number[1])

        condition = ast_utils.ProcessedWriter(sdfg,
                                              self.name_mapping,
                                              placeholders=self.placeholders,
                                              placeholders_offsets=self.placeholders_offsets,
                                              rename_dict=self.replace_names).write_code(node.cond)

        loop_region = LoopRegion(name, condition, inverted=False, sdfg=sdfg)

        is_start = cfg not in self.last_sdfg_states or self.last_sdfg_states[cfg] is None
        cfg.add_node(loop_region, ensure_unique_name=True, is_start_block=is_start)
        if not is_start:
            cfg.add_edge(self.last_sdfg_states[cfg], loop_region, InterstateEdge())
        self.last_sdfg_states[cfg] = loop_region
        self.last_sdfg_states[loop_region] = loop_region.add_state('BeginLoop_' + loop_region.label,
                                                                   is_start_block=True)

        self.translate(node.body, sdfg, loop_region)


    def forstmt2sdfg(self, node: ast_internal_classes.For_Stmt_Node, sdfg: SDFG, cfg: ControlFlowRegion):
        """
        This function is responsible for translating Fortran for statements into a SDFG.
        :param node: The for statement node to be translated
        :param sdfg: The SDFG to which the node should be translated
        :param cfg: The control flow region to which the node should be translated
        """
        name = 'FOR_l_' + str(node.line_number[0]) + '_c_' + str(node.line_number[1])
        decl_node = node.init
        init_expr = None
        if isinstance(decl_node, ast_internal_classes.BinOp_Node):
            if sdfg.symbols.get(decl_node.lval.name) is not None:
                iter_name = decl_node.lval.name
            elif self.name_mapping[sdfg].get(decl_node.lval.name) is not None:
                iter_name = self.name_mapping[sdfg][decl_node.lval.name]
            else:
                raise ValueError("Unknown variable " + decl_node.lval.name)
            init_assignment = ast_utils.ProcessedWriter(sdfg,
                                                        self.name_mapping,
                                                        placeholders=self.placeholders,
                                                        placeholders_offsets=self.placeholders_offsets,
                                                        rename_dict=self.replace_names).write_code(decl_node.rval)
            init_expr = f'{iter_name} = {init_assignment}'

        condition = ast_utils.ProcessedWriter(sdfg,
                                              self.name_mapping,
                                              placeholders=self.placeholders,
                                              placeholders_offsets=self.placeholders_offsets,
                                              rename_dict=self.replace_names).write_code(node.cond)

        increment_expr = 'i+0+1'
        if isinstance(node.iter, ast_internal_classes.BinOp_Node):
            increment_rhs = ast_utils.ProcessedWriter(sdfg,
                                                      self.name_mapping,
                                                      placeholders=self.placeholders,
                                                      placeholders_offsets=self.placeholders_offsets,
                                                      rename_dict=self.replace_names).write_code(node.iter.rval)
            increment_expr = f'{iter_name} = {increment_rhs}'

        loop_region = LoopRegion(name, condition, iter_name, init_expr, increment_expr, inverted=False, sdfg=sdfg)

        is_start = cfg not in self.last_sdfg_states or self.last_sdfg_states[cfg] is None
        cfg.add_node(loop_region, ensure_unique_name=True, is_start_block=is_start)
        if not is_start:
            cfg.add_edge(self.last_sdfg_states[cfg], loop_region, InterstateEdge())
        self.last_sdfg_states[cfg] = loop_region
        self.last_sdfg_states[loop_region] = loop_region.add_state('BeginLoop_' + loop_region.label,
                                                                   is_start_block=True)

        self.translate(node.body, sdfg, loop_region)


    def symbol2sdfg(self, node: ast_internal_classes.Symbol_Decl_Node, sdfg: SDFG, cfg: ControlFlowRegion):
        """
        This function is responsible for translating Fortran symbol declarations into a SDFG.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        :param cfg: The control flow region to which the node should be translated
        """
        if node.name == "modname": return

        if node.name.startswith("__f2dace_A_"):
            # separate name by removing the prefix and the suffix starting with _d_
            array_name = node.name[11:]
            array_name = array_name[:array_name.index("_d_")]
            if array_name in sdfg.arrays:
                return  # already declared
        if node.name.startswith("__f2dace_OA_"):
            # separate name by removing the prefix and the suffix starting with _d_
            array_name = node.name[12:]
            array_name = array_name[:array_name.index("_d_")]
            if array_name in sdfg.arrays:
                return

        if self.contexts.get(sdfg.name) is None:
            self.contexts[sdfg.name] = ast_utils.Context(name=sdfg.name)
        if self.contexts[sdfg.name].constants.get(node.name) is None:
            if isinstance(node.init, ast_internal_classes.Int_Literal_Node) or isinstance(
                    node.init, ast_internal_classes.Real_Literal_Node):
                self.contexts[sdfg.name].constants[node.name] = node.init.value
            elif isinstance(node.init, ast_internal_classes.Name_Node):
                self.contexts[sdfg.name].constants[node.name] = self.contexts[sdfg.name].constants[node.init.name]
            else:
                tw = ast_utils.TaskletWriter([], [], sdfg, self.name_mapping, placeholders=self.placeholders,
                                             placeholders_offsets=self.placeholders_offsets,
                                             rename_dict=self.replace_names)
                if node.init is not None:
                    text = tw.write_code(node.init)
                    self.contexts[sdfg.name].constants[node.name] = sym.pystr_to_symbolic(text)

        datatype = self.get_dace_type(node.type)
        if node.name not in sdfg.symbols:
            sdfg.add_symbol(node.name, datatype)
            if cfg not in self.last_sdfg_states or self.last_sdfg_states[cfg] is None:
                bstate = cfg.add_state("SDFGbegin", is_start_block=True)
                self.last_sdfg_states[cfg] = bstate
            if node.init is not None:
                substate = cfg.add_state(f"Dummystate_{node.name}")
                increment = ast_utils.TaskletWriter([], [],
                                                    sdfg,
                                                    self.name_mapping,
                                                    placeholders=self.placeholders,
                                                    placeholders_offsets=self.placeholders_offsets,
                                                    rename_dict=self.replace_names).write_code(node.init)

                entry = {node.name: increment}
                cfg.add_edge(self.last_sdfg_states[sdfg], substate, InterstateEdge(assignments=entry))
                self.last_sdfg_states[cfg] = substate

    def symbolarray2sdfg(self, node: ast_internal_classes.Symbol_Array_Decl_Node, sdfg: SDFG,
                         cfg: ControlFlowRegion):

        return NotImplementedError(
            "Symbol_Decl_Node not implemented. This should be done via a transformation that itemizes the constant array."
        )

    def subroutine2sdfg(self, node: ast_internal_classes.Subroutine_Subprogram_Node, sdfg: SDFG,
                        cfg: ControlFlowRegion):
        """
        This function is responsible for translating Fortran subroutine declarations into a SDFG.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        :param cfg: The control flow region to which the node should be translated
        """

        if node.execution_part is None:
            return
        if len(node.execution_part.execution) == 0:
            return

        print("TRANSLATE SUBROUTINE", node.name.name)

        # First get the list of read and written variables
        inputnodefinder = ast_transforms.FindInputs()
        inputnodefinder.visit(node)
        input_vars = inputnodefinder.nodes
        outputnodefinder = ast_transforms.FindOutputs(thourough=True)
        outputnodefinder.visit(node)
        output_vars = outputnodefinder.nodes
        write_names = list(dict.fromkeys([i.name for i in output_vars]))
        read_names = list(dict.fromkeys([i.name for i in input_vars]))

        # Collect the parameters and the function signature to comnpare and link
        parameters = node.args.copy()
        my_name_sdfg = node.name.name + str(self.sdfgs_count)
        new_sdfg = SDFG(my_name_sdfg)
        self.sdfgs_count += 1
        self.actual_offsets_per_sdfg[new_sdfg] = {}
        self.names_of_object_in_parent_sdfg[new_sdfg] = {}
        substate = self._add_simple_state_to_cfg(cfg, "state" + my_name_sdfg)

        variables_in_call = []
        if self.last_call_expression.get(sdfg) is not None:
            variables_in_call = self.last_call_expression[sdfg]

        # Sanity check to make sure the parameter numbers match
        if not ((len(variables_in_call) == len(parameters)) or
                (len(variables_in_call) == len(parameters) + 1
                 and not isinstance(node.result_type, ast_internal_classes.Void))):
            print("Subroutine", node.name.name)
            print('Variables in call', len(variables_in_call))
            print('Parameters', len(parameters))
            # for i in variables_in_call:
            #    print("VAR CALL: ", i.name)
            # for j in parameters:
            #    print("LOCAL TO UPDATE: ", j.name)
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
        to_fix = []
        symbol_arguments = []

        # First we need to check if the parameters are literals or variables
        for arg_i, variable in enumerate(variables_in_call):
            if isinstance(variable, ast_internal_classes.Actual_Arg_Spec_Node):
                keyword, variable = variable.arg_name, variable.arg

            if isinstance(variable, ast_internal_classes.Name_Node):
                varname = variable.name
            elif isinstance(variable, ast_internal_classes.Array_Subscript_Node):
                varname = variable.name.name
            elif isinstance(variable, ast_internal_classes.Data_Ref_Node):
                varname = ast_utils.get_name(variable)
            elif isinstance(variable, ast_internal_classes.BinOp_Node):
                varname = variable.rval.name

            if isinstance(variable, ast_internal_classes.Literal) or varname == "LITERAL":
                literals.append(parameters[arg_i])
                literal_values.append(variable)
                continue
            elif varname in sdfg.symbols:
                symbol_arguments.append((parameters[arg_i], variable))
                continue

            par2.append(parameters[arg_i])
            var2.append(variable)

        # This handles the case where the function is called with literals
        variables_in_call = var2
        parameters = par2
        assigns = []
        symbol_assigns = []
        sym_dict = {}
        self.local_not_transient_because_assign[my_name_sdfg] = []
        for lit, litval in zip(literals, literal_values):
            local_name = lit
            self.local_not_transient_because_assign[my_name_sdfg].append(local_name.name)
            # FIXME: Dirty hack to let translator create clean SDFG state names
            if node.line_number == -1:
                node.line_number = (0, 0)
            if isinstance(litval, ast_internal_classes.Int_Literal_Node):    
                sym_dict[local_name.name] = litval.value
                new_sdfg.add_symbol(local_name.name, dtypes.int32)
            else:    
                assigns.append(
                    ast_internal_classes.BinOp_Node(lval=ast_internal_classes.Name_Node(name=local_name.name),
                                                rval=litval,
                                                op="=",
                                                line_number=node.line_number))
        
        # This handles the case where the function is called with symbols
        for parameter, symbol in symbol_arguments:
            sym_dict[parameter.name] = symbol.name
            if parameter.name != symbol.name:
                self.local_not_transient_because_assign[my_name_sdfg].append(parameter.name)

                new_sdfg.add_symbol(parameter.name, dtypes.int32)
                # FIXME: Dirty hack to let translator create clean SDFG state names
                if node.line_number == -1:
                    node.line_number = (0, 0)
                symbol_assigns.append(
                    ast_internal_classes.BinOp_Node(lval=ast_internal_classes.Name_Node(name=parameter.name),
                                                    rval=ast_internal_classes.Name_Node(name=symbol.name),
                                                    op="=",
                                                    line_number=node.line_number))
        names_list = []
        if node.specification_part is not None:
            if node.specification_part.specifications is not None:
                namefinder = ast_transforms.FindDefinedNames()
                for i in node.specification_part.specifications:
                    namefinder.visit(i)
                names_list = namefinder.names
        # This handles the case where the function is called with variables starting with the case that the variable is local to the calling SDFG
        needs_replacement = {}
        for variable_in_call in variables_in_call:
            local_name = parameters[variables_in_call.index(variable_in_call)]
            
            local_definition = namefinder.specs.get(local_name.name)
            if local_definition is None:
                raise ValueError("Variable " + local_name.name + " is not defined in the function")
            self.name_mapping[new_sdfg][local_name.name] = new_sdfg._find_new_name(local_name.name)
            self.all_array_names.append(self.name_mapping[new_sdfg][local_name.name])
            read=False
            if local_name.name in read_names:
                ins_in_new_sdfg.append(self.name_mapping[new_sdfg][local_name.name])
                read=True
            write=False
            if local_name.name in write_names:
                outs_in_new_sdfg.append(self.name_mapping[new_sdfg][local_name.name])
                write=True
            ret,view=self.process_variable_call(variable_in_call,local_name, sdfg, new_sdfg,substate,read,write,local_definition)
            if ret:
                view[3]=variables_in_call.index(variable_in_call)
                views.append(view)


        # Preparing symbol dictionary for nested sdfg

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
        cached_names = [a[0] for a in self.module_vars]
        for i in not_found_read_names:
            if i in names_list:
                continue
            if i in cached_names:
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
                    elif (hasattr(array_in_global, 'type') and array_in_global.type == "Array") or isinstance(
                            array_in_global, dat.Array):
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
                    elif (hasattr(array_in_global, 'type') and array_in_global.type == "Array") or isinstance(
                            array_in_global, dat.Array):
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
            if i in names_list:
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

                    array_in_global = sdfg.arrays[self.name_mapping[sdfg][i]]
                    if isinstance(array_in_global, Scalar):
                        new_sdfg.add_scalar(self.name_mapping[new_sdfg][i], array_in_global.dtype, transient=False)
                    elif (hasattr(array_in_global, 'type') and array_in_global.type == "Array") or isinstance(
                            array_in_global, dat.Array):
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
                    elif (hasattr(array_in_global, 'type') and array_in_global.type == "Array") or isinstance(
                            array_in_global, dat.Array):
                        new_sdfg.add_array(self.name_mapping[new_sdfg][i],
                                           array_in_global.shape,
                                           array_in_global.dtype,
                                           array_in_global.storage,
                                           transient=False,
                                           strides=array_in_global.strides,
                                           offset=array_in_global.offset)
        all_symbols = new_sdfg.free_symbols
        missing_symbols = [s for s in all_symbols if s not in sym_dict]
        for i in missing_symbols:
            if i in sdfg.arrays:
                sym_dict[i] = i
                print("Force adding symbol to nested sdfg: ", i)
            else:
                print("Symbol not found in sdfg arrays: ", i)
        memlet_skip = []
        new_sdfg.parent_sdfg=sdfg
        self.temporary_sym_dict[new_sdfg.name]=sym_dict
        self.temporary_link_to_parent[new_sdfg.name]=substate
        if self.multiple_sdfgs == False:
            # print("Adding nested sdfg", new_sdfg.name, "to", sdfg.name)
            # print(sym_dict)
            if node.execution_part is not None:
                if node.specification_part is not None and node.specification_part.uses is not None:
                    for j in node.specification_part.uses:
                        for k in j.list:
                            if self.contexts.get(new_sdfg.name) is None:
                                self.contexts[new_sdfg.name] = ast_utils.Context(name=new_sdfg.name)
                            if self.contexts[new_sdfg.name].constants.get(
                                    ast_utils.get_name(k)) is None and self.contexts[
                                self.globalsdfg.name].constants.get(
                                ast_utils.get_name(k)) is not None:
                                self.contexts[new_sdfg.name].constants[ast_utils.get_name(k)] = self.contexts[
                                    self.globalsdfg.name].constants[ast_utils.get_name(k)]

                            pass

                    old_mode = self.transient_mode
                    # print("For ",sdfg_name," old mode is ",old_mode)
                    self.transient_mode = True
                    for j in node.specification_part.symbols:
                        if isinstance(j, ast_internal_classes.Symbol_Decl_Node):
                            self.symbol2sdfg(j, new_sdfg, new_sdfg)
                        else:
                            raise NotImplementedError("Symbol not implemented")

                    for j in node.specification_part.specifications:
                        self.declstmt2sdfg(j, new_sdfg, new_sdfg)
                    self.transient_mode = old_mode

                for i in new_sdfg.symbols:
                    if i in new_sdfg.arrays:
                        new_sdfg.arrays.pop(i)
                        if i in ins_in_new_sdfg:
                            for var in variables_in_call:
                                if i == ast_utils.get_name(parameters[variables_in_call.index(var)]):
                                    sym_dict[i] = ast_utils.get_name(var)
                                    memlet_skip.append(ast_utils.get_name(var))
                            ins_in_new_sdfg.remove(i)

                        if i in outs_in_new_sdfg:
                            outs_in_new_sdfg.remove(i)
                            for var in variables_in_call:
                                if i == ast_utils.get_name(parameters[variables_in_call.index(var)]):
                                    sym_dict[i] = ast_utils.get_name(var)
                                    memlet_skip.append(ast_utils.get_name(var))

            internal_sdfg = substate.add_nested_sdfg(new_sdfg,
                                                     sdfg,
                                                     ins_in_new_sdfg,
                                                     outs_in_new_sdfg,
                                                     symbol_mapping=self.temporary_sym_dict[new_sdfg.name])
        else:
            internal_sdfg = substate.add_nested_sdfg(None,
                                                     sdfg,
                                                     ins_in_new_sdfg,
                                                     outs_in_new_sdfg,
                                                     symbol_mapping=self.temporary_sym_dict[new_sdfg.name],
                                                     name="External_nested_" + new_sdfg.name)
            # if self.multiple_sdfgs==False:
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
            if ast_utils.get_name(i) in memlet_skip:
                continue
            local_name = parameters[variables_in_call.index(i)]
            if self.name_mapping.get(sdfg).get(ast_utils.get_name(i)) is not None:
                var = sdfg.arrays.get(self.name_mapping[sdfg][ast_utils.get_name(i)])
                mapped_name = self.name_mapping[sdfg][ast_utils.get_name(i)]
                if needs_replacement.get(mapped_name) is not None:
                    mapped_name = needs_replacement[mapped_name]
                    var = sdfg.arrays[mapped_name]
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
                memlet = ast_utils.generate_memlet(i, sdfg, self, self.normalize_offsets)

            found = False
            for elem in views:
                if mapped_name == elem[0] and elem[3] == variables_in_call.index(i):
                    found = True
                    recursive_view_check_done = False
                    while not recursive_view_check_done:
                        recursive_view_check_done = True
                        for elem2 in views:
                            if elem!=elem2 and elem[1].label == elem2[0] and elem2[3] == variables_in_call.index(i):
                                recursive_view_check_done=False
                                elem = elem2

                    # check variable type, if data ref, check lowest level array indices.
                    tmp_var = i
                    was_data_ref = False
                    while isinstance(tmp_var, ast_internal_classes.Data_Ref_Node):
                        was_data_ref = True
                        tmp_var = tmp_var.part_ref

                    #memlet = ast_utils.generate_memlet_view(
                    #    tmp_var, sdfg, self, self.normalize_offsets, mapped_name, elem[1].label, was_data_ref)

                    if local_name.name in write_names:
                        # memlet = subs.Range([(0, s - 1, 1) for s in sdfg.arrays[elem[2].label].shape])
                        #substate.add_memlet_path(
                        #    internal_sdfg, elem[2], src_conn=self.name_mapping[new_sdfg][local_name.name],
                        #    memlet=Memlet(expr=elem[2].label, subset=memlet))
                        shape= sdfg.arrays[elem[2].label].shape
                        if len(shape)==1:
                            memlet = "0"
                            substate.add_memlet_path(
                                internal_sdfg, elem[2], src_conn=self.name_mapping[new_sdfg][local_name.name],
                                memlet=Memlet(expr=elem[2].label, subset=memlet))
                        elif len(shape)==0:
                            memlet = "0"
                            substate.add_memlet_path(
                                internal_sdfg, elem[2], src_conn=self.name_mapping[new_sdfg][local_name.name],
                                memlet=Memlet())
                        else:

                            substate.add_memlet_path(
                                internal_sdfg, elem[2], src_conn=self.name_mapping[new_sdfg][local_name.name],
                                memlet=Memlet.from_array(elem[2].label,sdfg.arrays[elem[2].label]))
                    if local_name.name in read_names:
                        # memlet = subs.Range([(0, s - 1, 1) for s in sdfg.arrays[elem[1].label].shape])
                        shape= sdfg.arrays[elem[1].label].shape
                        if len(shape)==1:
                            memlet = "0"
                            substate.add_memlet_path(
                                elem[1], internal_sdfg, dst_conn=self.name_mapping[new_sdfg][local_name.name],
                                memlet=Memlet(expr=elem[1].label, subset=memlet))
                        elif len(shape)==0:
                            memlet = "0"
                            substate.add_memlet_path(
                                elem[1], internal_sdfg, dst_conn=self.name_mapping[new_sdfg][local_name.name],
                                memlet=Memlet())
                        else:
                            substate.add_memlet_path(
                                elem[1], internal_sdfg, dst_conn=self.name_mapping[new_sdfg][local_name.name],
                                memlet=Memlet.from_array(elem[1].label,sdfg.arrays[elem[1].label]))
                    if found:
                        break

            if not found:
                if local_name.name in write_names:
                    ast_utils.add_memlet_write(substate, mapped_name, internal_sdfg,
                                               self.name_mapping[new_sdfg][local_name.name], memlet)
                if local_name.name in read_names:
                    ast_utils.add_memlet_read(substate, mapped_name, internal_sdfg,
                                              self.name_mapping[new_sdfg][local_name.name], memlet)

        for i in addedmemlets:
            local_name = ast_internal_classes.Name_Node(name=i)
            memlet = ast_utils.generate_memlet(ast_internal_classes.Name_Node(name=i), sdfg, self,
                                               self.normalize_offsets)
            if local_name.name in write_names:
                ast_utils.add_memlet_write(substate, self.name_mapping[sdfg][i], internal_sdfg,
                                           self.name_mapping[new_sdfg][i], memlet)
            if local_name.name in read_names:
                ast_utils.add_memlet_read(substate, self.name_mapping[sdfg][i], internal_sdfg,
                                          self.name_mapping[new_sdfg][i], memlet)
        for i in globalmemlets:
            local_name = ast_internal_classes.Name_Node(name=i)
            found = False
            parent_sdfg = sdfg
            nested_sdfg = new_sdfg
            first = True
            while not found and parent_sdfg is not None:
                if self.name_mapping.get(parent_sdfg).get(i) is not None:
                    found = True
                else:
                    self.name_mapping[parent_sdfg][i] = parent_sdfg._find_new_name(i)
                    self.all_array_names.append(self.name_mapping[parent_sdfg][i])
                    array_in_global = self.globalsdfg.arrays[self.name_mapping[self.globalsdfg][i]]
                    if isinstance(array_in_global, Scalar):
                        parent_sdfg.add_scalar(self.name_mapping[parent_sdfg][i], array_in_global.dtype,
                                               transient=False)
                    elif (hasattr(array_in_global, 'type') and array_in_global.type == "Array") or isinstance(
                            array_in_global, dat.Array):
                        parent_sdfg.add_array(self.name_mapping[parent_sdfg][i],
                                              array_in_global.shape,
                                              array_in_global.dtype,
                                              array_in_global.storage,
                                              transient=False,
                                              strides=array_in_global.strides,
                                              offset=array_in_global.offset)

                if first:
                    first = False
                else:
                    if local_name.name in write_names:
                        nested_sdfg.parent_nsdfg_node.add_out_connector(self.name_mapping[parent_sdfg][i], force=True)
                    if local_name.name in read_names:
                        nested_sdfg.parent_nsdfg_node.add_in_connector(self.name_mapping[parent_sdfg][i], force=True)

                memlet = ast_utils.generate_memlet(ast_internal_classes.Name_Node(name=i), parent_sdfg, self,
                                                   self.normalize_offsets)
                if local_name.name in write_names:
                    ast_utils.add_memlet_write(nested_sdfg.parent, self.name_mapping[parent_sdfg][i],
                                               nested_sdfg.parent_nsdfg_node,
                                               self.name_mapping[nested_sdfg][i], memlet)
                if local_name.name in read_names:
                    ast_utils.add_memlet_read(nested_sdfg.parent, self.name_mapping[parent_sdfg][i],
                                              nested_sdfg.parent_nsdfg_node,
                                              self.name_mapping[nested_sdfg][i], memlet)
                if not found:
                    nested_sdfg = parent_sdfg
                    parent_sdfg = parent_sdfg.parent_sdfg

        if self.multiple_sdfgs == False:

            prev_block = None if new_sdfg not in self.last_sdfg_states else self.last_sdfg_states[new_sdfg]
            is_start = prev_block is None
            if is_start:
                first_substate = new_sdfg.add_state("start_state", is_start_block=True)
                self.last_sdfg_states[new_sdfg] = first_substate
                
            substate = new_sdfg.add_state("dummy_state_for_symbol_init")
            entries={}
            for i in symbol_assigns:
                entries[i.lval.name]=i.rval.name
            new_sdfg.add_edge(self.last_sdfg_states[new_sdfg], substate, InterstateEdge(assignments=entries))
            self.last_sdfg_states[new_sdfg] = substate

            for i in assigns:
                self.translate(i, new_sdfg, new_sdfg)
            self.translate(node.execution_part, new_sdfg, new_sdfg)
            # import copy
            #
            new_sdfg.reset_cfg_list()
            #new_sdfg.validate()
            #tmp_sdfg=copy.deepcopy(new_sdfg)
            new_sdfg.apply_transformations_repeated(IntrinsicSDFGTransformation)
            from dace.transformation.dataflow import RemoveSliceView
            new_sdfg.apply_transformations_repeated([RemoveSliceView])
            from dace.transformation.passes.lift_struct_views import LiftStructViews
            from dace.transformation.pass_pipeline import FixedPointPipeline
            FixedPointPipeline([LiftStructViews()]).apply_pass(new_sdfg, {})
            #new_sdfg.validate()
            # tmp_sdfg=copy.deepcopy(new_sdfg)
            new_sdfg.simplify()
            #new_sdfg.validate()
            #sdfg.validate()

        if self.multiple_sdfgs == True:
            internal_sdfg.path = self.sdfg_path + new_sdfg.name + ".sdfg"
            # new_sdfg.save(path.join(self.sdfg_path, new_sdfg.name + ".sdfg"))

    def compute_array_shape(self, node: ast_internal_classes.Array_Subscript_Node, sdfg: SDFG,array:dat.Array):
        """
        This function is responsible for computing the shape,offset and strides of an array access.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        :return: The shape of the array
        """
        changed_indices = 0
        index_list=[]
        shape=[]
        strides = list(array.strides)
        offsets = list(array.offset)
        mysize=1
        indices=0
        for i in node.indices:
            if isinstance(i, ast_internal_classes.ParDecl_Node):
                if i.type == "ALL":
                    shape.append(array.shape[indices])
                    mysize = mysize * array.shape[indices]
                    index_list.append(None)
                else:
                    start = i.range[0]
                    stop = i.range[1]
                    text_start = ast_utils.ProcessedWriter(sdfg, self.name_mapping,
                                                            placeholders=self.placeholders,
                                                            placeholders_offsets=self.placeholders_offsets,
                                                            rename_dict=self.replace_names).write_code(
                        start)
                    text_stop = ast_utils.ProcessedWriter(sdfg, self.name_mapping,
                                                            placeholders=self.placeholders,
                                                            placeholders_offsets=self.placeholders_offsets,
                                                            rename_dict=self.replace_names).write_code(
                        stop)
                    symb_size = sym.pystr_to_symbolic(text_stop + " - ( " + text_start + " )+1")
                    shape.append(symb_size)
                    mysize = mysize * symb_size
                    index_list.append(
                        [sym.pystr_to_symbolic(text_start), sym.pystr_to_symbolic(text_stop)])
                    # raise NotImplementedError("Index in ParDecl should be ALL")
            else:
                text = ast_utils.ProcessedWriter(sdfg, self.name_mapping,
                                                    placeholders=self.placeholders,
                                                    placeholders_offsets=self.placeholders_offsets,
                                                    rename_dict=self.replace_names).write_code(i)
                index_list.append([sym.pystr_to_symbolic(text), sym.pystr_to_symbolic(text)])
                strides.pop(indices - changed_indices)
                offsets.pop(indices - changed_indices)
                changed_indices += 1
            indices = indices + 1

        from dace import subsets

        all_indices = [None] * (len(array.shape) - len(index_list)) + index_list
        if self.normalize_offsets:
            subset = subsets.Range([(i[0], i[1], 1) if i is not None else (0, s - 1, 1)
                                    for i, s in zip(all_indices, array.shape)])
        else:
            subset = subsets.Range([(i[0], i[1], 1) if i is not None else (1, s, 1)
                                    for i, s in zip(all_indices, array.shape)])



        return shape,offsets,strides,subset

    def add_full_object(self, new_sdfg: SDFG,sdfg:SDFG, array: dat.Array, local_name: ast_internal_classes.FNode,local_shape:List,local_offsets:List,local_strides:List,sdfg_name,substate,read,write):
        """
        This function is responsible for adding a full array to the SDFG.
        :param new_sdfg: The SDFG to which the array should be added
        :param sdfg: The parent SDFG from which the array should be added
        :param array: The array to be added (can also be scalar or derived type/sruct)
        :param name: The name of the array
        """
        if hasattr(array, "name") and array.name in self.registered_types:
                datatype = self.get_dace_type(array.name)
                datatype_to_add = copy.deepcopy(array)
                datatype_to_add.transient = False
                if new_sdfg.arrays.get(self.name_mapping[new_sdfg][local_name.name]) is None:
                    new_sdfg.add_datadesc(self.name_mapping[new_sdfg][local_name.name], datatype_to_add)
                else:
                    #raise warning that array already exists in sdfg
                    print(f"Array {self.name_mapping[new_sdfg][local_name.name]} already exists in SDFG {new_sdfg.name}")


                if self.struct_views.get(new_sdfg) is None:
                    self.struct_views[new_sdfg] = {}
                add_views_recursive(new_sdfg, local_name.name, datatype_to_add,
                                    self.struct_views[new_sdfg], self.name_mapping[new_sdfg],
                                    self.registered_types, [], self.actual_offsets_per_sdfg[new_sdfg],
                                    self.names_of_object_in_parent_sdfg[new_sdfg],
                                    self.actual_offsets_per_sdfg[sdfg])
                return False, None

        else:
            shape= array.shape
            offset = array.offset
            strides=array.strides
            dtype = array.dtype
            print(f"Array: {local_name.name} shape: {shape}" )
            if len(shape)!=len(local_shape):
                is_scalar=(len(shape)==0) or (len(shape)==1 and shape[0]==1)
                is_local_scalar=(len(local_shape)==0) or (len(local_shape)==1 and local_shape[0]==1)
                if is_scalar and is_local_scalar:
                    pass
                else:
                    #raise ValueError("Shape of array does not match")
                    reshape_viewname, reshape_view = sdfg.add_view(sdfg_name + "_view_reshape_" + str(self.views),
                                                local_shape,
                                                dtype,
                                                storage=array.storage,
                                                strides=local_strides,
                                                offset=local_offsets)
                    memlet=Memlet.from_array(sdfg_name, sdfg.arrays[sdfg_name])
                    wv = None
                    rv = None
                    if read:
                        r = substate.add_read(sdfg_name)
                        wv = substate.add_write(reshape_viewname)
                        substate.add_edge(r, None, wv, 'views', dpcp(memlet))
                    if write:
                        rv = substate.add_read(reshape_viewname)
                        w = substate.add_write(sdfg_name)
                        substate.add_edge(rv, 'views', w, None, dpcp(memlet))
                    new_sdfg.add_array(self.name_mapping[new_sdfg][local_name.name],
                                            local_shape,
                                            dtype,
                                            array.storage,
                                            strides=local_strides,
                                            offset=local_offsets)    

                    return True, (wv, rv)    
            if new_sdfg.arrays.get(self.name_mapping[new_sdfg][local_name.name]) is None:
                if shape == []:
                    new_sdfg.add_scalar(self.name_mapping[new_sdfg][local_name.name], array.dtype,
                                                    array.storage)
                else:
                    new_sdfg.add_array(self.name_mapping[new_sdfg][local_name.name],
                                            shape,
                                            dtype,
                                            array.storage,
                                            strides=array.strides,
                                            offset=offset)
                return False, None    
            else:
                #raise warning that array already exists in sdfg
                print(f"Array {self.name_mapping[new_sdfg][local_name.name]} already exists in SDFG {new_sdfg.name}")


    def add_simple_array_to_element_view_pair_in_tower(self, sdfg: SDFG, array: dat.Array, name_chain: List[str], member: ast_internal_classes.FNode, substate: SDFGState, last_read: nd.AccessNode, last_written: nd.AccessNode, read: bool, write: bool,shape,offsets,strides,subset):

        dtype=array.dtype
        offsets_zero = [0]*len(offsets)
        concatenated_name = "_".join(name_chain)
        view_name=concatenated_name + "_" + ast_utils.get_name(member) + "_m_" + str(
                            self.struct_view_count)
        if len(shape)==0:
                shape=[1]
                offsets_zero=[0]
                strides=[1]
        viewname, view = sdfg.add_view(view_name,
                                            shape,
                                            array.dtype,
                                            storage=array.storage,
                                            strides=strides,
                                            offset=offsets_zero)

        memlet=Memlet.simple(concatenated_name + "_" + ast_utils.get_name(member) + "_" + str(
                            self.struct_view_count), subset)

        return self.add_accesses_and_edges(sdfg,view_name,view, array, substate, last_read, last_written, read, write,memlet)


    def add_array_to_element_view_pair_in_tower(self, sdfg: SDFG, array: dat.Array, name_chain: List[str], member: ast_internal_classes.FNode, substate: SDFGState, last_read: nd.AccessNode, last_written: nd.AccessNode, read: bool, write: bool,subset):

        stype=array.stype
        view_to_member = dat.View.view(stype)
        concatenated_name = "_".join(name_chain)
        view_name=concatenated_name + "_" + ast_utils.get_name(member) + "_m_" + str(
                            self.struct_view_count)

        memlet=Memlet.simple(concatenated_name + "_" + ast_utils.get_name(member) + "_" + str(
                            self.struct_view_count), subset)

        return self.add_accesses_and_edges(sdfg,view_name,view_to_member, array, substate, last_read, last_written, read, write,memlet)

    def add_basic_view_pair_in_tower(self, sdfg: SDFG, array: dat.Array, name_chain: List[str], member: ast_internal_classes.FNode, substate: SDFGState, last_read: nd.AccessNode, last_written: nd.AccessNode, read: bool, write: bool):
        view_to_member = dat.View.view(array)
        concatenated_name = "_".join(name_chain)
        view_name=concatenated_name + "_" + ast_utils.get_name(member) + "_" + str(
                            self.struct_view_count)
        if last_read is not None:
            name=last_read.label
        if last_written is not None:
            name=last_written.label
        if last_read is not None and last_written is not None:
            if last_read.label != last_written.label:
                raise ValueError("Last read and last written are not the same")
        memlet=Memlet.from_array(name + "." + ast_utils.get_name(member), array)
        return self.add_accesses_and_edges(sdfg,view_name,view_to_member, array, substate, last_read, last_written, read, write,memlet)


    def add_accesses_and_edges(self,sdfg: SDFG,view_name:str,view_to_member:dat.View, array: dat.Array, substate: SDFGState, last_read: nd.AccessNode, last_written: nd.AccessNode, read: bool, write: bool,memlet:Memlet):
        sdfg.arrays[view_name] = view_to_member
        if read:
            new_read=substate.add_read(view_name)
            substate.add_edge(last_read, None, new_read, None, dpcp(memlet))
            last_read=new_read
        if write:
            new_written=substate.add_write(view_name)
            substate.add_edge( new_written, None,last_written, None, dpcp(memlet))
            last_written=new_written

        return last_read, last_written

    def get_local_shape(self, sdfg:SDFG,local_definition:ast_internal_classes.Var_Decl_Node):
        shape=[]
        if hasattr(local_definition,"sizes"):
            if local_definition.sizes==None:
                return [1],[0],[0],[1]
            sizes=[]
            offsets=[]
            actual_offsets=[]
            offset_value=0
            for i in local_definition.sizes:
                tw = ast_utils.TaskletWriter([], [], sdfg, self.name_mapping, placeholders=self.placeholders,
                                                placeholders_offsets=self.placeholders_offsets,
                                                rename_dict=self.replace_names)
                text = tw.write_code(i)
                actual_offset_value = local_definition.offsets[local_definition.sizes.index(i)]
                if isinstance(actual_offset_value, int):
                    actual_offset_value = ast_internal_classes.Int_Literal_Node(value=str(actual_offset_value))
                aotext = tw.write_code(actual_offset_value)
                actual_offsets.append(str(sym.pystr_to_symbolic(aotext)))
                # otext = tw.write_code(offset_value)
                sizes.append(sym.pystr_to_symbolic(text))
                offsets.append(offset_value)
            if len(sizes)==0:
                return [1],[0],[0],[1]
            strides = [dat._prod(sizes[:i]) for i in range(len(sizes))]    
            return sizes, offsets, actual_offsets,strides
        else:
            return [1],[0],[0],[1]
        

    def process_variable_call(self, variable_in_calling_context: ast_internal_classes.FNode, local_name:ast_internal_classes.FNode,  sdfg: SDFG, new_sdfg: SDFG, substate:SDFGState, read:bool,write:bool,local_definition:ast_internal_classes.Var_Decl_Node):
        # We need to first check and have separate handling for:
        # 1. Scalars
        # 2. Arrays
        # 3. Derived types

        # The steps are
        # 1. to first generate towers of views for derived types
        # 2. to generate views for arrays and views of arrays coming out of towers of views if the subset is not the whole array
        # 3. this will allow the "final" memlets to the inconnectors to be "simple"

        # Get name of variable in SDFG of calling context or globalSDFG if that fails


        sdfg_name = self.name_mapping.get(sdfg).get(ast_utils.get_name(variable_in_calling_context))
        if sdfg_name is None:
            globalsdfg_name = self.name_mapping.get(self.globalsdfg).get(ast_utils.get_name(variable_in_calling_context))

        # Get array reference in SDFG
        if sdfg_name is not None:
            array = sdfg.arrays.get(sdfg_name)
            self.names_of_object_in_parent_sdfg[new_sdfg][local_name.name] = sdfg_name
        elif globalsdfg_name is not None:
            array = self.globalsdfg.arrays.get(globalsdfg_name)
        else:
            raise ValueError("Variable not found in SDFG or globalSDFG")

        #Get the shape, offset, and type of the array in the local context
        local_shape, local_offsets, local_actual_offsets,local_strides = self.get_local_shape(sdfg,local_definition)



        #this can be a scalar, a full array, or a full derived type object
        if isinstance(variable_in_calling_context, ast_internal_classes.Name_Node):
            views_needed,views= self.add_full_object(new_sdfg,sdfg,array,local_name,local_shape,local_offsets,local_strides,sdfg_name,substate,read,write)
            if views_needed:
                return True, [sdfg_name, views[0], views[1], variable_in_calling_context]
            else:
                return False , None


        #this can be an array slice or a derived type object member slice
        elif isinstance(variable_in_calling_context, ast_internal_classes.Array_Subscript_Node):
            print("Array Subscript node")
            shape,offsets,strides,subset=self.compute_array_shape(variable_in_calling_context,sdfg,array)
            offsets_zero = [0]*len(offsets)
            memlet = Memlet(f'{sdfg_name}[{subset}]')
            if len(shape)==0:
                shape=[1]
                offsets_zero=[0]
                strides=[1]
            viewname, view = sdfg.add_view(sdfg_name + "_view_" + str(self.views),
                                            shape,
                                            array.dtype,
                                            storage=array.storage,
                                            strides=strides,
                                            offset=offsets_zero)

            wv = None
            rv = None
            if read:
                r = substate.add_read(sdfg_name)
                wv = substate.add_write(viewname)
                substate.add_edge(r, None, wv, 'views', dpcp(memlet))
            if write:
                rv = substate.add_read(viewname)
                w = substate.add_write(sdfg_name)
                substate.add_edge(rv, 'views', w, None, dpcp(memlet))

            self.views = self.views + 1
            is_scalar=(len(shape)==0) or (len(shape)==1 and shape[0]==1)
            is_local_scalar=(len(local_shape)==0) or (len(local_shape)==1 and local_shape[0]==1)
            
            if local_shape!=shape and (not(is_scalar and is_local_scalar)):
                #we must add an extra view reshaping the access to the local shape
                if len(shape)==len(local_shape):
                    print("Shapes are not equal, but the same size. We hope that the symbolic sizes evaluate to the same values")
                    recompute_strides=False
                    for i,local in enumerate(local_shape):
                        if local.name.startswith("__f2dace"):
                            local_shape[i]=shape[i]
                            print(f"replacing local shape: {local_shape[i]}")
                            local_offsets[i]=offsets[i]
                            recompute_strides=True
                    if recompute_strides:
                        local_strides = [dat._prod(local_shape[:i]) for i in range(len(local_shape))]        

                            
                else:    
                    if len(local_shape)!=1:
                        raise NotImplementedError("Local shape not 1")
                    reshape_viewname, reshape_view = sdfg.add_view(sdfg_name + "_view_reshape_" + str(self.views),
                                                local_shape,
                                                array.dtype,
                                                storage=array.storage,
                                                strides=local_strides,
                                                offset=local_offsets)
                    
                    
                    memlet=Memlet.from_array(viewname, sdfg.arrays[viewname])
                    if write:
                        res_v_read = substate.add_read(reshape_viewname)
                        substate.add_edge(res_v_read, None, rv, None, dpcp(memlet))
                        rv=res_v_read
                    if read:
                        res_v_write = substate.add_write(reshape_viewname)
                        substate.add_edge(wv, None, res_v_write, None, dpcp(memlet))
                        wv=res_v_write
                    


            new_sdfg.add_array(self.name_mapping[new_sdfg][local_name.name],
                                local_shape,
                                array.dtype,
                                array.storage,
                                strides=local_strides,
                                offset=local_offsets)
            return True, [sdfg_name, wv, rv, variable_in_calling_context]
        #this is an access to a (potentially nested) derived type object member
        elif isinstance(variable_in_calling_context, ast_internal_classes.Data_Ref_Node):
            self.struct_view_count = self.struct_view_count + 1
            print("Data Ref node")
            intermediate_step=variable_in_calling_context
            top_structure_name=self.name_mapping[sdfg][ast_utils.get_name(variable_in_calling_context.parent_ref)]
            top_structure=sdfg.arrays[top_structure_name]
            current_structure=top_structure
            name_chain=[]
            if read:
                last_read=substate.add_read(top_structure_name)
            else:
                last_read=None
            if write:
                last_written=substate.add_write(top_structure_name)
            else:
                last_written=None

            while True:
                member=intermediate_step.part_ref
                parent=intermediate_step.parent_ref
                if isinstance(parent,ast_internal_classes.Array_Subscript_Node):
                    #this means that there is an array access in the chain before the end
                    #such accesses must always collapse to elements
                    shape,offsets,strides,subset=self.compute_array_shape(parent,sdfg,current_structure)
                    print("Array Subscript node")
                    raise NotImplementedError("Array Subscript node in Data Ref parent not implemented")
                elif isinstance(parent,ast_internal_classes.Name_Node):
                    #this is the simpler case - no extra work necessary
                    name_chain.append(ast_utils.get_name(parent))

                else:
                    raise ValueError("Unsupported parent node type")

                if isinstance(member,ast_internal_classes.Name_Node):
                    #this is the end of the chain
                    array=current_structure.members[ast_utils.get_name(member)]
                    last_read, last_written=self.add_basic_view_pair_in_tower(sdfg,array,name_chain,member,substate,last_read,last_written,read,write)

                    views_needed,views=self.add_full_object(new_sdfg,sdfg,array,local_name,local_shape,local_offsets,local_strides,sdfg_name,substate,read,write)
                    if views_needed:
                        return True, [sdfg_name, views[0], views[1], variable_in_calling_context]
                    else:
                
                        return True, [sdfg_name,last_read, last_written, variable_in_calling_context]
                elif isinstance(member,ast_internal_classes.Array_Subscript_Node):

                    print("Array Subscript node in Data Ref as last level")
                    array=current_structure.members[ast_utils.get_name(member)]
                    shape,offsets,strides,subset=self.compute_array_shape(member,sdfg,array)

                    if isinstance(array, dat.ContainerArray):
                        #this is a derived type object, must have first view to Array, then view to subset if necessary
                        last_read, last_written=self.add_basic_view_pair_in_tower(sdfg,array,name_chain,member,substate,last_read,last_written,read,write)
                        if len(shape)==0:
                            #this is exactly one element of the array of structures
                            stype=array.stype
                            if isinstance(stype, dat.ContainerArray):
                                raise NotImplementedError("Array of structures of array of structures not implemented")
                            else:
                                last_read, last_written=self.add_array_to_element_view_pair_in_tower(sdfg,array,name_chain,member,substate,last_read,last_written,read,write,subset)
                                views_needed,views=self.add_full_object(new_sdfg,sdfg,stype,local_name,local_shape,local_offsets,local_strides,sdfg_name,substate,read,write)
                                if views_needed:
                                    return True, [sdfg_name, views[0], views[1], variable_in_calling_context]
                                else:
                
                                    return True, [sdfg_name,last_read, last_written, variable_in_calling_context]

                        else:
                            raise NotImplementedError("Array of structures slice not implemented")

                    else:
                        #this is a simple array, but must still have first view to Array and then to subset.
                        last_read, last_written=self.add_basic_view_pair_in_tower(sdfg,array,name_chain,member,substate,last_read,last_written,read,write)
                        last_read, last_written=self.add_simple_array_to_element_view_pair_in_tower(sdfg,array,name_chain,member,substate,last_read,last_written,read,write,shape,offsets,strides,subset)
                        if len(shape)==0:
                            shape=[1]
                            offsets=[0]
                            strides=[1]
                        is_scalar=(len(shape)==0) or (len(shape)==1 and shape[0]==1)
                        is_local_scalar=(len(local_shape)==0) or (len(local_shape)==1 and local_shape[0]==1)
                        if local_shape!=shape and (not(is_scalar and is_local_scalar)):  
                            if len(shape)==len(local_shape):
                                print("Shapes are not equal, but the same size. We hope that the symbolic sizes evaluate to the same values")
                                #this is not necessary, as here we use the outside sizes for some reason???
                                # recompute_strides=False
                                # for i,local in enumerate(local_shape):
                                #     if local.name.startswith("__f2dace"):
                                #         local_shape[i]=shape[i]
                                #         print(f"replacing local shape: {local_shape[i]}")
                                #         local_offsets[i]=offsets[i]
                                #         recompute_strides=True
                                # if recompute_strides:
                                #     local_strides = [dat._prod(local_shape[:i]) for i in range(len(local_shape))]        
                            else:    
                                raise NotImplementedError("Local shape not the same as outside shape")  
                        new_sdfg.add_array(self.name_mapping[new_sdfg][local_name.name],
                                shape,
                                array.dtype,
                                array.storage,
                                strides=strides,
                                offset=offsets)
                        return True, [sdfg_name,last_read, last_written, variable_in_calling_context]

                elif isinstance(member,ast_internal_classes.Data_Ref_Node):
                    #this is a member access
                    array=current_structure.members[ast_utils.get_name(member.parent_ref)]
                    last_read, last_written=self.add_basic_view_pair_in_tower(sdfg,array,name_chain,member.parent_ref,substate,last_read,last_written,read,write)

                    current_structure=current_structure.members[ast_utils.get_name(member.parent_ref)]
                    intermediate_step=member
        else:
            raise ValueError("Unsupported variable type")



    def binop2sdfg(self, node: ast_internal_classes.BinOp_Node, sdfg: SDFG, cfg: ControlFlowRegion):
        """
        This parses binary operations to tasklets in a new state or creates a function call with a nested SDFG if the
        operation is a function call rather than a simple assignment.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        :param cfg: The control flow region to which the node should be translated
        """

        calls = ast_transforms.FindFunctionCalls()
        calls.visit(node)
        if len(calls.nodes) == 1:
            augmented_call = calls.nodes[0]
            from dace.frontend.fortran.intrinsics import FortranIntrinsics
            if augmented_call.name.name not in ["pow", "atan2", "tanh", "__dace_epsilon",
                                                *FortranIntrinsics.retained_function_names()]:
                augmented_call.args.append(node.lval)
                augmented_call.hasret = True
                self.call2sdfg(augmented_call, sdfg, cfg)
                return

        outputnodefinder = ast_transforms.FindOutputs(thourough=False)
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

        substate = self._add_simple_state_to_cfg(
            cfg, "_state_l" + str(node.line_number[0]) + "_c" + str(node.line_number[1]))

        output_names_changed = [o_t + "_out" for o_t in output_names]

        tasklet = self._add_tasklet(substate, "_l" + str(node.line_number[0]) + "_c" + str(node.line_number[1]),
                                    input_names_tasklet, output_names_changed, "text", node.line_number, self.file_name)

        for i, j in zip(input_names, input_names_tasklet):
            memlet_range = self.get_memlet_range(sdfg, input_vars, i, j)
            src = ast_utils.add_memlet_read(substate, i, tasklet, j, memlet_range)
            # if self.struct_views.get(sdfg) is not None:
            #   if self.struct_views[sdfg].get(i) is not None:
            #     chain= self.struct_views[sdfg][i]
            #     access_parent=substate.add_access(chain[0])
            #     name=chain[0]
            #     for i in range(1,len(chain)):
            #         view_name=name+"_"+chain[i]
            #         access_child=substate.add_access(view_name)
            #         substate.add_edge(access_parent, None,access_child, 'views',  Memlet.simple(name+"."+chain[i],subs.Range.from_array(sdfg.arrays[view_name])))
            #         name=view_name
            #         access_parent=access_child

            #     substate.add_edge(access_parent, None,src,'views',  Memlet(data=name, subset=memlet_range))

        for i, j, k in zip(output_names, output_names_tasklet, output_names_changed):
            memlet_range = self.get_memlet_range(sdfg, output_vars, i, j)
            ast_utils.add_memlet_write(substate, i, tasklet, k, memlet_range)
        tw = ast_utils.TaskletWriter(output_names, output_names_changed, sdfg, self.name_mapping, input_names,
                                     input_names_tasklet, placeholders=self.placeholders,
                                     placeholders_offsets=self.placeholders_offsets, rename_dict=self.replace_names)

        text = tw.write_code(node)
        # print(sdfg.name,node.line_number,output_names,output_names_changed,input_names,input_names_tasklet)
        tasklet.code = CodeBlock(text, lang.Python)

    def call2sdfg(self, node: ast_internal_classes.Call_Expr_Node, sdfg: SDFG, cfg: ControlFlowRegion):
        """
        This parses function calls to a nested SDFG 
        or creates a tasklet with an external library call.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        :param cfg: The control flow region to which the node should be translated
        """

        self.last_call_expression[sdfg] = node.args
        match_found = False
        rettype = "INTEGER"
        hasret = False
        for fsname in self.functions_and_subroutines:
            if fsname.name == node.name.name:

                for i in self.top_level.function_definitions:
                    if i.name.name == node.name.name:
                        self.function2sdfg(i, sdfg, cfg)
                        return
                for i in self.top_level.subroutine_definitions:
                    if i.name.name == node.name.name:
                        self.subroutine2sdfg(i, sdfg, cfg)
                        return
                for j in self.top_level.modules:
                    for i in j.function_definitions:
                        if i.name.name == node.name.name:
                            self.function2sdfg(i, sdfg, cfg)
                            return
                    for i in j.subroutine_definitions:
                        if i.name.name == node.name.name:
                            self.subroutine2sdfg(i, sdfg, cfg)
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
                node for node in ast_utils.mywalk(node) if isinstance(node, ast_internal_classes.Name_Node)
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
                                         self.name_mapping, placeholders=self.placeholders,
                                         placeholders_offsets=self.placeholders_offsets, rename_dict=self.replace_names)
            if not isinstance(rettype, ast_internal_classes.Void) and hasret:
                if isinstance(retval, ast_internal_classes.Name_Node):
                    special_list_in[retval.name] = pointer(self.get_dace_type(rettype))
                    special_list_out.append(retval.name + "_out")
                elif isinstance(retval, ast_internal_classes.Array_Subscript_Node):
                    special_list_in[retval.name.name] = pointer(self.get_dace_type(rettype))
                    special_list_out.append(retval.name.name + "_out")
                else:
                    raise NotImplementedError("Return type not implemented")

                text = tw.write_code(
                    ast_internal_classes.BinOp_Node(lval=retval, op="=", rval=node, line_number=node.line_number))

            else:
                text = tw.write_code(node)
            substate = self._add_simple_state_to_cfg(cfg, "_state" + str(node.line_number[0]))

            tasklet = self._add_tasklet(substate, str(node.line_number[0]), {
                **input_names_tasklet,
                **special_list_in
            }, output_names_changed + special_list_out, "text", node.line_number, self.file_name)
            if libstate is not None:
                ast_utils.add_memlet_read(substate, self.name_mapping[sdfg][libstate], tasklet,
                                          self.name_mapping[sdfg][libstate] + "_task", "0")

                ast_utils.add_memlet_write(substate, self.name_mapping[sdfg][libstate], tasklet,
                                           self.name_mapping[sdfg][libstate] + "_task_out", "0")
            if not isinstance(rettype, ast_internal_classes.Void) and hasret:
                if isinstance(retval, ast_internal_classes.Name_Node):
                    ast_utils.add_memlet_read(substate, self.name_mapping[sdfg][retval.name], tasklet, retval.name, "0")

                    ast_utils.add_memlet_write(substate, self.name_mapping[sdfg][retval.name], tasklet,
                                               retval.name + "_out", "0")
                if isinstance(retval, ast_internal_classes.Array_Subscript_Node):
                    ast_utils.add_memlet_read(substate, self.name_mapping[sdfg][retval.name.name], tasklet,
                                              retval.name.name, "0")

                    ast_utils.add_memlet_write(substate, self.name_mapping[sdfg][retval.name.name], tasklet,
                                               retval.name.name + "_out", "0")

            for i, j in zip(input_names, input_names_tasklet):
                memlet_range = self.get_memlet_range(sdfg, used_vars, i, j)
                ast_utils.add_memlet_read(substate, i, tasklet, j, memlet_range)

            for i, j, k in zip(output_names, output_names_tasklet, output_names_changed):
                memlet_range = self.get_memlet_range(sdfg, used_vars, i, j)
                ast_utils.add_memlet_write(substate, i, tasklet, k, memlet_range)

            setattr(tasklet, "code", CodeBlock(text, lang.Python))

    def declstmt2sdfg(self, node: ast_internal_classes.Decl_Stmt_Node, sdfg: SDFG, cfg: ControlFlowRegion):
        """
        This function translates a variable declaration statement to an access node on the sdfg
        :param node: The node to translate
        :param sdfg: The sdfg to attach the access node to
        :param cfg: The control flow region to which the node should be translated
        :note This function is the top level of the declaration, most implementation is in vardecl2sdfg
        """
        for i in node.vardecl:
            self.translate(i, sdfg, cfg)

    def vardecl2sdfg(self, node: ast_internal_classes.Var_Decl_Node, sdfg: SDFG, cfg: ControlFlowRegion):
        """
        This function translates a variable declaration to an access node on the sdfg
        :param node: The node to translate
        :param sdfg: The sdfg to attach the access node to
        :param cfg: The control flow region to which the node should be translated
        """
        if node.name == "modname": return

        # if the sdfg is the toplevel-sdfg, the variable is a global variable
        is_arg = False
        if isinstance(node.parent,
                      (ast_internal_classes.Subroutine_Subprogram_Node, ast_internal_classes.Function_Subprogram_Node)):
            if hasattr(node.parent, "args"):
                for i in node.parent.args:
                    name = ast_utils.get_name(i)
                    if name == node.name:
                        is_arg = True
                        if self.local_not_transient_because_assign.get(sdfg.name) is not None:
                            if name in self.local_not_transient_because_assign[sdfg.name]:
                                is_arg = False
                        break

        # if this is a variable declared in the module,
        # then we will not add it unless it is used by the functions.
        # It would be sufficient to check the main entry function,
        # since it must pass this variable through call
        # to other functions.
        # However, I am not completely sure how to determine which function is the main one.
        #
        # we ignore the variable that is not used at all in all functions
        # this is a module variaable that can be removed
        if not is_arg:
            if self.subroutine_used_names is not None:

                if node.name not in self.subroutine_used_names:
                    print(
                        f"Ignoring module variable {node.name} because it is not used in the the top level subroutine")
                    return

        if is_arg:
            transient = False
        else:
            transient = self.transient_mode
        # find the type
        datatype = self.get_dace_type(node.type)
        # if hasattr(node, "alloc"):
        #    if node.alloc:
        #        self.unallocated_arrays.append([node.name, datatype, sdfg, transient])
        #        return
        # get the dimensions
        # print(node.name)
        if node.sizes is not None and len(node.sizes) > 0:
            sizes = []
            offset = []
            actual_offsets = []
            offset_value = 0 if self.normalize_offsets else -1
            for i in node.sizes:
                stuff = [ii for ii in ast_utils.mywalk(i) if isinstance(ii, ast_internal_classes.Data_Ref_Node)]
                if len(stuff) > 0:
                    count = self.count_of_struct_symbols_lifted
                    sdfg.add_symbol("tmp_struct_symbol_" + str(count), dtypes.int32)
                    symname = "tmp_struct_symbol_" + str(count)
                    if sdfg.parent_sdfg is not None:
                        sdfg.parent_sdfg.add_symbol("tmp_struct_symbol_" + str(count), dtypes.int32)
                        self.temporary_sym_dict[sdfg.name]["tmp_struct_symbol_" + str(count)] = "tmp_struct_symbol_" + str(count)
                        parent_state=self.temporary_link_to_parent[sdfg.name]
                        for edge in parent_state.parent_graph.in_edges(parent_state):
                            assign = ast_utils.ProcessedWriter(sdfg.parent_sdfg, self.name_mapping,
                                                               placeholders=self.placeholders,
                                                               placeholders_offsets=self.placeholders_offsets,
                                                               rename_dict=self.replace_names).write_code(i)
                            edge.data.assignments["tmp_struct_symbol_" + str(count)] = assign
                            # print(edge)
                    else:
                        assign = ast_utils.ProcessedWriter(sdfg, self.name_mapping, placeholders=self.placeholders,
                                                           placeholders_offsets=self.placeholders_offsets,
                                                           rename_dict=self.replace_names).write_code(i)

                        sdfg.append_global_code(f"{dtypes.int32.ctype} {symname};\n")
                        sdfg.append_init_code(
                            "tmp_struct_symbol_" + str(count) + "=" + assign.replace(".", "->") + ";\n")
                    tw = ast_utils.TaskletWriter([], [], sdfg, self.name_mapping, placeholders=self.placeholders,
                                                 placeholders_offsets=self.placeholders_offsets,
                                                 rename_dict=self.replace_names)
                    text = tw.write_code(
                        ast_internal_classes.Name_Node(name="tmp_struct_symbol_" + str(count), type="INTEGER",
                                                       line_number=node.line_number))
                    sizes.append(sym.pystr_to_symbolic(text))
                    actual_offset_value = node.offsets[node.sizes.index(i)]
                    if isinstance(actual_offset_value, ast_internal_classes.Array_Subscript_Node):
                        # print(node.name,actual_offset_value.name.name)
                        raise NotImplementedError("Array subscript in offset not implemented")
                    if isinstance(actual_offset_value, int):
                        actual_offset_value = ast_internal_classes.Int_Literal_Node(value=str(actual_offset_value))
                    aotext = tw.write_code(actual_offset_value)
                    actual_offsets.append(str(sym.pystr_to_symbolic(aotext)))

                    self.actual_offsets_per_sdfg[sdfg][node.name] = actual_offsets
                    # otext = tw.write_code(offset_value)

                    # TODO: shouldn't this use node.offset??
                    offset.append(offset_value)
                    self.count_of_struct_symbols_lifted += 1
                else:
                    tw = ast_utils.TaskletWriter([], [], sdfg, self.name_mapping, placeholders=self.placeholders,
                                                 placeholders_offsets=self.placeholders_offsets,
                                                 rename_dict=self.replace_names)
                    text = tw.write_code(i)
                    actual_offset_value = node.offsets[node.sizes.index(i)]
                    if isinstance(actual_offset_value, int):
                        actual_offset_value = ast_internal_classes.Int_Literal_Node(value=str(actual_offset_value))
                    aotext = tw.write_code(actual_offset_value)
                    actual_offsets.append(str(sym.pystr_to_symbolic(aotext)))
                    # otext = tw.write_code(offset_value)
                    sizes.append(sym.pystr_to_symbolic(text))
                    offset.append(offset_value)
                    self.actual_offsets_per_sdfg[sdfg][node.name] = actual_offsets

        else:
            sizes = None
        # create and check name - if variable is already defined (function argument and defined in declaration part) simply stop
        if self.name_mapping[sdfg].get(node.name) is not None:
            # here we must replace local placeholder sizes that have already made it to tasklets via size and ubound calls
            if sizes is not None:
                actual_sizes = sdfg.arrays[self.name_mapping[sdfg][node.name]].shape
                # print(node.name,sdfg.name,self.names_of_object_in_parent_sdfg.get(sdfg).get(node.name))
                # print(sdfg.parent_sdfg.name,self.actual_offsets_per_sdfg[sdfg.parent_sdfg].get(self.names_of_object_in_parent_sdfg[sdfg][node.name]))
                # print(sdfg.parent_sdfg.arrays.get(self.name_mapping[sdfg.parent_sdfg].get(self.names_of_object_in_parent_sdfg.get(sdfg).get(node.name))))
                if self.actual_offsets_per_sdfg[sdfg.parent_sdfg].get(
                        self.names_of_object_in_parent_sdfg[sdfg][node.name]) is not None:
                    actual_offsets = self.actual_offsets_per_sdfg[sdfg.parent_sdfg][
                        self.names_of_object_in_parent_sdfg[sdfg][node.name]]
                else:
                    actual_offsets = [1] * len(actual_sizes)

                index = 0
                for i in node.sizes:
                    if isinstance(i, ast_internal_classes.Name_Node):
                        if i.name.startswith("__f2dace_A"):
                            self.replace_names[i.name] = str(actual_sizes[index])
                            # node.parent.execution_part=ast_transforms.RenameVar(i.name,str(actual_sizes[index])).visit(node.parent.execution_part)
                    index += 1
                index = 0
                for i in node.offsets:
                    if isinstance(i, ast_internal_classes.Name_Node):
                        if i.name.startswith("__f2dace_OA"):
                            self.replace_names[i.name] = str(actual_offsets[index])
                            # node.parent.execution_part=ast_transforms.RenameVar(i.name,str(actual_offsets[index])).visit(node.parent.execution_part)
                    index += 1
            elif sizes is None:
                if isinstance(datatype, Structure):
                    datatype_to_add = copy.deepcopy(datatype)
                    datatype_to_add.transient = transient
                    # if node.name=="p_nh":
                    # print("Adding local struct",self.name_mapping[sdfg][node.name],datatype_to_add)
                    if self.struct_views.get(sdfg) is None:
                        self.struct_views[sdfg] = {}
                    add_views_recursive(sdfg, node.name, datatype_to_add, self.struct_views[sdfg],
                                        self.name_mapping[sdfg], self.registered_types, [],
                                        self.actual_offsets_per_sdfg[sdfg], self.names_of_object_in_parent_sdfg[sdfg],
                                        self.actual_offsets_per_sdfg[sdfg.parent_sdfg])

            return

        if node.name in sdfg.symbols:
            return

        self.name_mapping[sdfg][node.name] = sdfg._find_new_name(node.name)

        if sizes is None:
            if isinstance(datatype, Structure):
                datatype_to_add = copy.deepcopy(datatype)
                datatype_to_add.transient = transient
                # if node.name=="p_nh":
                # print("Adding local struct",self.name_mapping[sdfg][node.name],datatype_to_add)
                sdfg.add_datadesc(self.name_mapping[sdfg][node.name], datatype_to_add)
                if self.struct_views.get(sdfg) is None:
                    self.struct_views[sdfg] = {}
                add_views_recursive(sdfg, node.name, datatype_to_add, self.struct_views[sdfg], self.name_mapping[sdfg],
                                    self.registered_types, [], self.actual_offsets_per_sdfg[sdfg], {}, {})
                # for i in datatype_to_add.members:
                #     current_dtype=datatype_to_add.members[i].dtype
                #     for other_type in self.registered_types:
                #         if current_dtype.dtype==self.registered_types[other_type].dtype:
                #             other_type_obj=self.registered_types[other_type]
                #             for j in other_type_obj.members:
                #                 sdfg.add_view(self.name_mapping[sdfg][node.name] + "_" + i +"_"+ j,other_type_obj.members[j].shape,other_type_obj.members[j].dtype)
                #                 self.name_mapping[sdfg][node.name + "_" + i +"_"+ j] = self.name_mapping[sdfg][node.name] + "_" + i +"_"+ j
                #                 self.struct_views[sdfg][self.name_mapping[sdfg][node.name] + "_" + i+"_"+ j]=[self.name_mapping[sdfg][node.name],j]
                #     sdfg.add_view(self.name_mapping[sdfg][node.name] + "_" + i,datatype_to_add.members[i].shape,datatype_to_add.members[i].dtype)
                #     self.name_mapping[sdfg][node.name + "_" + i] = self.name_mapping[sdfg][node.name] + "_" + i
                #     self.struct_views[sdfg][self.name_mapping[sdfg][node.name] + "_" + i]=[self.name_mapping[sdfg][node.name],i]

            else:

                sdfg.add_scalar(self.name_mapping[sdfg][node.name], dtype=datatype, transient=transient)
        else:
            strides = [dat._prod(sizes[:i]) for i in range(len(sizes))]

            if isinstance(datatype, Structure):
                datatype.transient = transient
                arr_dtype = datatype[sizes]
                arr_dtype.offset = [offset_value for _ in sizes]
                container = dat.ContainerArray(stype=datatype, shape=sizes, offset=offset, transient=transient)
                # print("Adding local container array",self.name_mapping[sdfg][node.name],sizes,datatype,offset,strides,transient)
                sdfg.arrays[self.name_mapping[sdfg][node.name]] = container
                if self.struct_views.get(sdfg) is None:
                    self.struct_views[sdfg] = {}
                add_views_recursive(sdfg, node.name, arr_dtype, self.struct_views[sdfg], self.name_mapping[sdfg],
                                    self.registered_types, [], self.actual_offsets_per_sdfg[sdfg], {}, {})
                # sdfg.add_datadesc(self.name_mapping[sdfg][node.name], arr_dtype)

            else:
                # print("Adding local array",self.name_mapping[sdfg][node.name],sizes,datatype,offset,strides,transient)
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

        if hasattr(node, "init") and node.init is not None:
            if isinstance(node.init, ast_internal_classes.Array_Constructor_Node):
                new_exec = ast_transforms.ReplaceArrayConstructor().visit(
                    ast_internal_classes.BinOp_Node(
                        lval=ast_internal_classes.Name_Node(name=node.name, type=node.type),
                        op="=", rval=node.init, line_number=node.line_number, parent=node.parent, type=node.type))
                self.translate(new_exec, sdfg, cfg)
            else:
                self.translate(
                    ast_internal_classes.BinOp_Node(
                        lval=ast_internal_classes.Name_Node(name=node.name, type=node.type),
                        op="=", rval=node.init, line_number=node.line_number, parent=node.parent, type=node.type), sdfg,
                        cfg)

    def break2sdfg(self, node: ast_internal_classes.Break_Node, sdfg: SDFG, cfg: ControlFlowRegion):
        break_block = BreakBlock(f'Break_l_{str(node.line_number[0])}_c_{str(node.line_number[1])}')
        is_start = cfg not in self.last_sdfg_states or self.last_sdfg_states[cfg] is None
        cfg.add_node(break_block, ensure_unique_name=True, is_start_block=is_start)
        if not is_start:
            cfg.add_edge(self.last_sdfg_states[cfg], break_block, InterstateEdge())

    def continue2sdfg(self, node: ast_internal_classes.Continue_Node, sdfg: SDFG, cfg: ControlFlowRegion):
        continue_block = ContinueBlock(f'Continue_l_{str(node.line_number[0])}_c_{str(node.line_number[1])}')
        is_start = cfg not in self.last_sdfg_states or self.last_sdfg_states[cfg] is None
        cfg.add_node(continue_block, ensure_unique_name=True, is_start_block=is_start)
        if not is_start:
            cfg.add_edge(self.last_sdfg_states[cfg], continue_block, InterstateEdge())


def create_ast_from_string(
        source_string: str,
        sdfg_name: str,
        transform: bool = False,
        normalize_offsets: bool = False,
        multiple_sdfgs: bool = False
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
    own_ast = ast_components.InternalFortranAst()
    program = own_ast.create_ast(ast)

    structs_lister = ast_transforms.StructLister()
    structs_lister.visit(program)
    struct_dep_graph = nx.DiGraph()
    for i, name in zip(structs_lister.structs, structs_lister.names):
        if name not in struct_dep_graph.nodes:
            struct_dep_graph.add_node(name)
        struct_deps_finder = ast_transforms.StructDependencyLister(structs_lister.names)
        struct_deps_finder.visit(i)
        struct_deps = struct_deps_finder.structs_used
        for j, pointing, point_name in zip(struct_deps, struct_deps_finder.is_pointer,
                                           struct_deps_finder.pointer_names):
            if j not in struct_dep_graph.nodes:
                struct_dep_graph.add_node(j)
            struct_dep_graph.add_edge(name, j, pointing=pointing, point_name=point_name)

    program.structures = ast_transforms.Structures(structs_lister.structs)

    functions_and_subroutines_builder = ast_transforms.FindFunctionAndSubroutines()
    functions_and_subroutines_builder.visit(program)

    if transform:
        program = ast_transforms.functionStatementEliminator(program)
        program = ast_transforms.CallToArray(functions_and_subroutines_builder).visit(program)
        program = ast_transforms.CallExtractor(program).visit(program)
        program = ast_transforms.SignToIf().visit(program)
        program = ast_transforms.ArrayToLoop(program).visit(program)

        for transformation in own_ast.fortran_intrinsics().transformations():
            transformation.initialize(program)
            program = transformation.visit(program)

        program = ast_transforms.ForDeclarer().visit(program)
        program = ast_transforms.IndexExtractor(program, normalize_offsets).visit(program)
        program = ast_transforms.optionalArgsExpander(program)

    return program, own_ast


class ParseConfig:
    def __init__(self,
                 main: Union[None, Path, str] = None,
                 sources: Union[None, List[Path], Dict[str, str]] = None,
                 includes: Union[None, List[Path], Dict[str, str]] = None,
                 entry_points: Union[None, SPEC, List[SPEC]] = None,
                 config_injections: Optional[List[ConstInjection]] = None):
        # Make the configs canonical, by processing the various types upfront.
        if isinstance(main, Path):
            main = main.read_text()
        main = FortranStringReader(main)
        if not sources:
            sources: Dict[str, str] = {}
        elif isinstance(sources, list):
            sources: Dict[str, str] = {str(p): p.read_text() for p in sources}
        if not includes:
            includes: List[Path] = []
        if not entry_points:
            entry_points = []
        elif isinstance(entry_points, tuple):
            entry_points = [entry_points]

        self.main = main
        self.sources = sources
        self.includes = includes
        self.entry_points = entry_points
        self.config_injections = config_injections or []


def create_fparser_ast(cfg: ParseConfig) -> Program:
    parser = ParserFactory().create(std="f2008")
    ast = parser(cfg.main)
    ast = recursive_ast_improver(ast, cfg.sources, cfg.includes, parser)
    ast = lower_identifier_names(ast)
    assert isinstance(ast, Program)
    return ast


def create_internal_ast(cfg: ParseConfig) -> Tuple[ast_components.InternalFortranAst, FNode]:
    ast = create_fparser_ast(cfg)

    if not cfg.entry_points:
        # Keep all the possible entry points.
        cfg.entry_points = [ident_spec(ast_utils.singular(children_of_type(c, NAMED_STMTS_OF_INTEREST_CLASSES)))
                            for c in ast.children if isinstance(c, ENTRY_POINT_OBJECT_CLASSES)]

    ast = run_fparser_transformations(ast, cfg)
    assert isinstance(ast, Program)

    iast = ast_components.InternalFortranAst()
    prog = iast.create_ast(ast)
    assert isinstance(prog, FNode)
    prog.module_declarations = ast_utils.parse_module_declarations(prog)
    iast.finalize_ast(prog)
    return iast, prog


class SDFGConfig:
    def __init__(self,
                 entry_points: Dict[str, Union[str, List[str]]],
                 config_injections: Optional[List[ConstTypeInjection]] = None,
                 normalize_offsets: bool = True,
                 multiple_sdfgs: bool = False):
        for k in entry_points:
            if isinstance(entry_points[k], str):
                entry_points[k] = [entry_points[k]]
        self.entry_points = entry_points
        self.config_injections = config_injections or []
        self.normalize_offsets = normalize_offsets
        self.multiple_sdfgs = multiple_sdfgs


def run_fparser_transformations(ast: Program, cfg: ParseConfig):
    print("FParser Op: Removing indirections from AST...")
    ast = deconstruct_enums(ast)
    ast = deconstruct_associations(ast)
    ast = remove_access_statements(ast)
    ast = correct_for_function_calls(ast)
    ast = deconstruct_statement_functions(ast)
    ast = deconstruct_procedure_calls(ast)
    ast = deconstruct_interface_calls(ast)
    ast = correct_for_function_calls(ast)

    print("FParser Op: Inject configs & prune...")
    ast = inject_const_evals(ast, cfg.config_injections)
    ast = const_eval_nodes(ast)
    ast = convert_data_statements_into_assignments(ast)

    print("FParser Op: Fix global vars & prune...")
    # Prune things once after fixing global variables.
    # NOTE: Global vars fixing has to be done before any pruning, because otherwise some assignment may get lost.
    ast = make_practically_constant_global_vars_constants(ast)
    ast = const_eval_nodes(ast)
    ast = prune_branches(ast)
    ast = prune_unused_objects(ast, cfg.entry_points)

    print("FParser Op: Fix arguments & prune...")
    # Another round of pruning after fixing the practically constant arguments, just in case.
    ast = make_practically_constant_arguments_constants(ast, cfg.entry_points)
    ast = const_eval_nodes(ast)
    ast = prune_branches(ast)
    ast = prune_unused_objects(ast, cfg.entry_points)

    print("FParser Op: Fix local vars & prune...")
    # Another round of pruning after fixing the locally constant variables, just in case.
    ast = exploit_locally_constant_variables(ast)
    ast = const_eval_nodes(ast)
    ast = prune_branches(ast)
    ast = prune_unused_objects(ast, cfg.entry_points)

    print("FParser Op: Create global initializers & rename uniquely...")
    ast = create_global_initializers(ast, cfg.entry_points)
    ast = assign_globally_unique_subprogram_names(ast, set(cfg.entry_points))
    # TODO: Disabled because some other transforms rely on the naming scheme of variables.
    ast = assign_globally_unique_variable_names(ast, set(cfg.entry_points))
    ast = consolidate_uses(ast)

    return ast


def run_ast_transformations(own_ast: ast_components.InternalFortranAst, program: FNode, cfg: SDFGConfig, normalize_offsets: bool = True):

    functions_and_subroutines_builder = ast_transforms.FindFunctionAndSubroutines()
    functions_and_subroutines_builder.visit(program)

    program = ast_transforms.functionStatementEliminator(program)
    #program = ast_transforms.StructConstructorToFunctionCall(
    #    ast_transforms.FindFunctionAndSubroutines.from_node(program).names).visit(program)
    #program = ast_transforms.CallToArray(ast_transforms.FindFunctionAndSubroutines.from_node(program)).visit(program)
    program = ast_transforms.IfConditionExtractor().visit(program)
    program = ast_transforms.CallExtractor(program).visit(program)

    program = ast_transforms.FunctionCallTransformer().visit(program)
    program = ast_transforms.FunctionToSubroutineDefiner().visit(program)
    program = ast_transforms.PointerRemoval().visit(program)
    for i in program.modules:
        count = 0
        for j in i.function_definitions:
            if isinstance(j, ast_internal_classes.Subroutine_Subprogram_Node):
                i.subroutine_definitions.append(j)
                count += 1
        if count != len(i.function_definitions):
            raise NameError("Not all functions were transformed to subroutines")
        i.function_definitions = []
    program.function_definitions = []
    count = 0
    for i in program.function_definitions:
        if isinstance(i, ast_internal_classes.Subroutine_Subprogram_Node):
            program.subroutine_definitions.append(i)
            count += 1
    if count != len(program.function_definitions):
        raise NameError("Not all functions were transformed to subroutines")
    program.function_definitions = []

    program = ast_transforms.SignToIf(own_ast).visit(program)
    # run it again since signtoif might introduce patterns that have to be extracted
    # example: ABS call inside an UnOpNode
    program = ast_transforms.CallExtractor(program).visit(program)
    program = ast_transforms.ReplaceStructArgsLibraryNodes(program).visit(program)

    program = ast_transforms.ArgumentExtractor(program).visit(program)

    program = ast_transforms.TypeInference(program, assert_voids=False).visit(program)
    program = ast_transforms.ElementalIntrinsicExpander(
        ast_transforms.FindFunctionAndSubroutines.from_node(program).names,
        program
    ).visit(program)

    prior_exception: Optional[NeedsTypeInferenceException] = None
    for transformation in own_ast.fortran_intrinsics().transformations():
        while True:
            try:
                transformation.initialize(program)
                program = transformation.visit(program)
                break
            except NeedsTypeInferenceException as e:

                if prior_exception is not None:
                    if e.line_number == prior_exception.line_number and e.func_name == prior_exception.func_name:
                        print("Running additional type inference didn't help! VOID type in the same place.")
                        raise RuntimeError()
                else:
                    prior_exception = e
                print("Running additional type inference")
                # FIXME: optimize func
                program = ast_transforms.TypeInference(program, assert_voids=False).visit(program)

    array_dims_info = ast_transforms.ArrayDimensionSymbolsMapper()
    array_dims_info.visit(program)
    program = ast_transforms.ArrayDimensionConfigInjector(array_dims_info, cfg.config_injections).visit(program)

    program = ast_transforms.ParDeclNonContigArrayExpander(program).visit(program)

    program = ast_transforms.ArrayToLoop(program).visit(program)
    program = ast_transforms.ForDeclarer().visit(program)
    program = ast_transforms.IndexExtractor(program, normalize_offsets).visit(program)
    program = ast_transforms.optionalArgsExpander(program)
    #program = ast_transforms.ParDeclOffsetNormalizer(program).visit(program)
    program = ast_transforms.allocatableReplacer(program)
    program = ast_transforms.ParDeclOffsetNormalizer(program).visit(program)

    structs_lister = ast_transforms.StructLister()
    structs_lister.visit(program)
    struct_dep_graph = nx.DiGraph()
    for i, name in zip(structs_lister.structs, structs_lister.names):
        if name not in struct_dep_graph.nodes:
            struct_dep_graph.add_node(name)
        struct_deps_finder = ast_transforms.StructDependencyLister(structs_lister.names)
        struct_deps_finder.visit(i)
        struct_deps = struct_deps_finder.structs_used
        # print(struct_deps)
        for j, pointing, point_name in zip(struct_deps, struct_deps_finder.is_pointer,
                                           struct_deps_finder.pointer_names):
            if j not in struct_dep_graph.nodes:
                struct_dep_graph.add_node(j)
            struct_dep_graph.add_edge(name, j, pointing=pointing, point_name=point_name)
    cycles = nx.algorithms.cycles.simple_cycles(struct_dep_graph)
    has_cycles = list(cycles)
    cycles_we_cannot_ignore = []
    for cycle in has_cycles:
        print(cycle)
        for i in cycle:
            is_pointer = struct_dep_graph.get_edge_data(i, cycle[(cycle.index(i) + 1) % len(cycle)])["pointing"]
            point_name = struct_dep_graph.get_edge_data(i, cycle[(cycle.index(i) + 1) % len(cycle)])["point_name"]
            # print(i,is_pointer)
            if is_pointer:
                actually_used_pointer_node_finder = ast_transforms.StructPointerChecker(i, cycle[
                    (cycle.index(i) + 1) % len(cycle)], point_name)
                actually_used_pointer_node_finder.visit(program)
                # print(actually_used_pointer_node_finder.nodes)
                if len(actually_used_pointer_node_finder.nodes) == 0:
                    print("We can ignore this cycle")
                    program = ast_transforms.StructPointerEliminator(i, cycle[(cycle.index(i) + 1) % len(cycle)],
                                                                     point_name).visit(program)
                else:
                    cycles_we_cannot_ignore.append(cycle)
    if len(cycles_we_cannot_ignore) > 0:
        raise NameError("Structs have cyclic dependencies")

    # TODO: `ArgumentPruner` does not cleanly remove arguments -> disable until fixed.
    # Check before rerunning CloudSC
    # ast_transforms.ArgumentPruner(functions_and_subroutines_builder.nodes).visit(program)

    # TODO: Enable permanently after the tests pass.
    # ast_utils.validate_internal_ast(program)
    return program

def create_sdfg_from_internal_ast(own_ast: ast_components.InternalFortranAst, program: FNode, cfg: SDFGConfig):
    # Repeated!
    # We need that to know in transformations what structures are used.
    # The actual structure listing is repeated later to resolve cycles.
    # Not sure if we can actually do it earlier.

    program = run_ast_transformations(own_ast, program, cfg, True)

    gmap = {}
    for ep, ep_spec in cfg.entry_points.items():
        # Find where to look for the entry point.
        assert ep_spec
        mod, pt = ep_spec[:-1], ep_spec[-1]
        assert len(mod) <= 1, f"currently only one level of entry point search is supported, got: {ep_spec}"
        ep_box = program  # This is where we will search for our entry point.
        if mod:
            mod = mod[0]
            mod = [m for m in program.modules if m.name.name == mod]
            assert len(mod) <= 1, f"found multiple modules with the same name: {mod}"
            if not mod:
                # Could not even find the module, so skip.
                continue
            ep_box = mod[0]

        # Find the actual entry point.
        fn = [f for f in ep_box.subroutine_definitions if f.name.name == pt]
        if not mod and program.main_program and program.main_program.name.name.name == pt:
            # The main program can be a valid entry point, so include that when appropriate.
            fn.append(program.main_program)
        assert len(fn) <= 1, f"found multiple subroutines with the same name {ep}"
        if not fn:
            continue
        fn = fn[0]

        # Do the actual translation.
        ast2sdfg = AST_translator(__file__, multiple_sdfgs=cfg.multiple_sdfgs, startpoint=fn, toplevel_subroutine=None,
                                  normalize_offsets=cfg.normalize_offsets, do_not_make_internal_variables_argument=True)
        g = SDFG(ep)
        ast2sdfg.functions_and_subroutines = ast_transforms.FindFunctionAndSubroutines.from_node(program).names
        ast2sdfg.structures = program.structures
        ast2sdfg.placeholders = program.placeholders
        ast2sdfg.placeholders_offsets = program.placeholders_offsets
        ast2sdfg.actual_offsets_per_sdfg[g] = {}
        ast2sdfg.top_level = program
        ast2sdfg.globalsdfg = g
        ast2sdfg.translate(program, g, g)
        g.reset_cfg_list()
        from dace.transformation.passes.lift_struct_views import LiftStructViews
        from dace.transformation.pass_pipeline import FixedPointPipeline
        FixedPointPipeline([LiftStructViews()]).apply_pass(g, {})
        g.apply_transformations_repeated(IntrinsicSDFGTransformation)
        g.expand_library_nodes()
        gmap[ep] = g

    return gmap

def create_singular_sdfg_from_string(
        sources: Dict[str, str],
        entry_point: str,
        normalize_offsets: bool = True,
        config_injections: Optional[List[ConstTypeInjection]] = None) -> SDFG:
    entry_point = entry_point.split('.')

    cfg = ParseConfig(main=sources['main.f90'], sources=sources, entry_points=tuple(entry_point),
                      config_injections=config_injections)
    own_ast, program = create_internal_ast(cfg)

    cfg = SDFGConfig({entry_point[-1]: entry_point}, config_injections=config_injections,
                     normalize_offsets=normalize_offsets, multiple_sdfgs=False)
    gmap = create_sdfg_from_internal_ast(own_ast, program, cfg)
    assert gmap.keys() == {entry_point[-1]}
    g = list(gmap.values())[0]

    return g

def create_sdfg_from_string(
        source_string: str,
        sdfg_name: str,
        normalize_offsets: bool = True,
        multiple_sdfgs: bool = False,
        sources: List[str] = None,
):
    """
    Creates an SDFG from a fortran file in a string
    :param source_string: The fortran file as a string
    :param sdfg_name: The name to be given to the resulting SDFG
    :return: The resulting SDFG

    """
    cfg = ParseConfig(main=source_string, sources=sources)
    own_ast, program = create_internal_ast(cfg)

    cfg = SDFGConfig(
        {sdfg_name: f"{sdfg_name}_function"}, 
        config_injections=None,
        normalize_offsets=normalize_offsets,
        multiple_sdfgs=False
    )
    gmap = create_sdfg_from_internal_ast(own_ast, program, cfg)
    assert gmap.keys() == {sdfg_name}
    g = list(gmap.values())[0]

    return g

def compute_dep_graph(ast: Program, start_point: Union[str, List[str]]) -> nx.DiGraph:
    """
    Compute a dependency graph among all the top level objects in the program.
    """
    if isinstance(start_point, str):
        start_point = [start_point]

    dep_graph = nx.DiGraph()
    exclude = set()
    to_process = start_point
    while to_process:
        item_name, to_process = to_process[0], to_process[1:]
        item = ast_utils.atmost_one(c for c in ast.children if find_name_of_node(c) == item_name)
        if not item:
            print(f"Could not find: {item}")
            continue

        fandsl = ast_utils.FunctionSubroutineLister()
        fandsl.get_functions_and_subroutines(item)
        dep_graph.add_node(item_name, info_list=fandsl)

        used_modules, objects_in_modules = ast_utils.get_used_modules(item)
        for mod in used_modules:
            if mod not in dep_graph.nodes:
                dep_graph.add_node(mod)
            obj_list = []
            if dep_graph.has_edge(item_name, mod):
                edge = dep_graph.get_edge_data(item_name, mod)
                if 'obj_list' in edge:
                    obj_list = edge.get('obj_list')
                    assert isinstance(obj_list, list)
            if mod in objects_in_modules:
                ast_utils.extend_with_new_items_from(obj_list, objects_in_modules[mod])
            dep_graph.add_edge(item_name, mod, obj_list=obj_list)
            if mod not in exclude:
                to_process.append(mod)
                exclude.add(mod)

    return dep_graph


def recursive_ast_improver(ast: Program, source_list: Dict[str, str], include_list, parser):
    exclude = set()

    NAME_REPLACEMENTS = {
        'mo_restart_nml_and_att': 'mo_restart_nmls_and_atts',
        'yomhook': 'yomhook_dummy',
    }

    def _recursive_ast_improver(_ast: Base):
        defined_modules = ast_utils.get_defined_modules(_ast)
        used_modules, objects_in_modules = ast_utils.get_used_modules(_ast)

        modules_to_parse = [mod for mod in used_modules if mod not in chain(defined_modules, exclude)]
        added_modules = []
        for mod in modules_to_parse:
            name = mod.lower()
            if name in NAME_REPLACEMENTS:
                name = NAME_REPLACEMENTS[name]

            mod_file = [srcf for srcf in source_list if os.path.basename(srcf).lower() == f"{name}.f90"]
            assert len(mod_file) <= 1, f"Found multiple files for the same module `{mod}`: {mod_file}"
            if not mod_file:
                print(f"Ignoring error: cannot find a file for `{mod}`")
                continue
            mod_file = mod_file[0]

            reader = fsr(source_list[mod_file], include_dirs=include_list)
            try:
                next_ast = parser(reader)
            except Exception as e:
                raise RuntimeError(f"{mod_file} could not be parsed: {e}") from e

            _recursive_ast_improver(next_ast)

            for c in reversed(next_ast.children):
                if c in added_modules:
                    added_modules.remove(c)
                added_modules.insert(0, c)
                c_stmt = c.children[0]
                c_name = ast_utils.singular(ast_utils.children_of_type(c_stmt, Name)).string
                exclude.add(c_name)

        for mod in reversed(added_modules):
            if mod not in _ast.children:
                _ast.children.append(mod)

    _recursive_ast_improver(ast)

    # Add all the free-floating subprograms from other source files in case we missed them.
    ast = collect_floating_subprograms(ast, source_list, include_list, parser)
    # Sort the modules in the order of their dependency.
    ast = sort_modules(ast)

    return ast


def collect_floating_subprograms(ast: Program, source_list: Dict[str, str], include_list, parser) -> Program:
    known_names: Set[str] = {nm.string for nm in walk(ast, Name)}

    known_floaters: Set[str] = set()
    for esp in ast.children:
        name = find_name_of_node(esp)
        if name:
            known_floaters.add(name)

    known_sub_asts: Dict[str, Program] = {}
    for src, content in source_list.items():

        # TODO: Should be fixed in FParser.
        # FParser cannot handle `convert=...` argument in the `open()` statement.
        content = content.replace(',convert="big_endian"', '')

        reader = fsr(content, include_dirs=include_list)
        try:
            sub_ast = parser(reader)
        except Exception as e:
            print(f"Ignoring {src} due to error: {e}")
            continue
        known_sub_asts[src] = sub_ast

    # Since the order is not topological, we need to incrementally find more connected floating subprograms.
    changed = True
    while changed:
        changed = False
        new_floaters = []
        for src, sub_ast in known_sub_asts.items():
            # Find all the new floating subprograms that are known to be needed so far.
            for esp in sub_ast.children:
                name = find_name_of_node(esp)
                if name and name in known_names and name not in known_floaters:
                    # We have found a new floating subprogram that's needed.
                    known_floaters.add(name)
                    known_names.update({nm.string for nm in walk(esp, Name)})
                    new_floaters.append(esp)
        if new_floaters:
            # Append the new floating subprograms to our main AST.
            append_children(ast, new_floaters)
            changed = True
    return ast


def name_and_rename_dict_creator(parse_order: list, dep_graph: nx.DiGraph) \
        -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, str]]]:
    name_dict = {}
    rename_dict = {}
    for i in parse_order:
        local_rename_dict = {}
        edges = list(dep_graph.in_edges(i))
        names = []
        for j in edges:
            list_dict = dep_graph.get_edge_data(j[0], j[1])
            if (list_dict['obj_list'] is not None):
                for k in list_dict['obj_list']:
                    if not k.__class__.__name__ == "Name":
                        if k.__class__.__name__ == "Rename":
                            if k.children[2].string not in names:
                                names.append(k.children[2].string)
                            local_rename_dict[k.children[2].string] = k.children[1].string
                        # print("Assumption failed: Object list contains non-name node")
                    else:
                        if k.string not in names:
                            names.append(k.string)
        rename_dict[i] = local_rename_dict
        name_dict[i] = names
    return name_dict, rename_dict


@dataclass
class FindUsedFunctionsConfig:
    root: str
    needed_functions: List[str]
    skip_functions: List[str]


def create_sdfg_from_fortran_file_with_options(
        cfg: ParseConfig,
        ast: Program,
        sdfgs_dir,
        subroutine_name: Optional[str] = None,
        normalize_offsets: bool = True,
        propagation_info=None,
        enum_propagator_files: Optional[List[str]] = None,
        enum_propagator_ast=None,
        used_functions_config: Optional[FindUsedFunctionsConfig] = None,
        already_parsed_ast=False,
        config_injections: Optional[List[ConstTypeInjection]] = None,
):
    """
    Creates an SDFG from a fortran file
    :param source_string: The fortran file name
    :return: The resulting SDFG

    """
    if not already_parsed_ast:
        ast = run_fparser_transformations(ast, cfg)
    else:
        ast = correct_for_function_calls(ast)

    dep_graph = compute_dep_graph(ast, 'radiation_interface')
    parse_order = list(reversed(list(nx.topological_sort(dep_graph))))

    what_to_parse_list = {}
    name_dict, rename_dict = name_and_rename_dict_creator(parse_order, dep_graph)

    tables = SymbolTable
    partial_ast = ast_components.InternalFortranAst()
    partial_modules = {}
    partial_ast.symbols["c_int"] = ast_internal_classes.Int_Literal_Node(value=4)
    partial_ast.symbols["c_int8_t"] = ast_internal_classes.Int_Literal_Node(value=1)
    partial_ast.symbols["c_int64_t"] = ast_internal_classes.Int_Literal_Node(value=8)
    partial_ast.symbols["c_int32_t"] = ast_internal_classes.Int_Literal_Node(value=4)
    partial_ast.symbols["c_size_t"] = ast_internal_classes.Int_Literal_Node(value=4)
    partial_ast.symbols["c_long"] = ast_internal_classes.Int_Literal_Node(value=8)
    partial_ast.symbols["c_signed_char"] = ast_internal_classes.Int_Literal_Node(value=1)
    partial_ast.symbols["c_char"] = ast_internal_classes.Int_Literal_Node(value=1)
    partial_ast.symbols["c_null_char"] = ast_internal_classes.Int_Literal_Node(value=1)
    functions_to_rename = {}

    # Why would you ever name a file differently than the module? Especially just one random file out of thousands???
    # asts["mo_restart_nml_and_att"]=asts["mo_restart_nmls_and_atts"]
    partial_ast.to_parse_list = what_to_parse_list
    asts = {find_name_of_stmt(m).lower(): m for m in walk(ast, Module_Stmt)}
    for i in parse_order:
        partial_ast.current_ast = i

        partial_ast.unsupported_fortran_syntax[i] = []
        if i in ["mtime", "ISO_C_BINDING", "iso_c_binding", "mo_cdi", "iso_fortran_env", "netcdf"]:
            continue

        # try:
        partial_module = partial_ast.create_ast(asts[i.lower()])
        partial_modules[partial_module.name.name] = partial_module
        # except Exception as e:
        #    print("Module " + i + " could not be parsed ", partial_ast.unsupported_fortran_syntax[i])
        #    print(e, type(e))
        # print(partial_ast.unsupported_fortran_syntax[i])
        #    continue
        tmp_rename = rename_dict[i]
        for j in tmp_rename:
            # print(j)
            if partial_ast.symbols.get(j) is None:
                # raise NameError("Symbol " + j + " not found in partial ast")
                if functions_to_rename.get(i) is None:
                    functions_to_rename[i] = [j]
                else:
                    functions_to_rename[i].append(j)
            else:
                partial_ast.symbols[tmp_rename[j]] = partial_ast.symbols[j]

        print("Parsed successfully module: ", i, " ", partial_ast.unsupported_fortran_syntax[i])
        # print(partial_ast.unsupported_fortran_syntax[i])
    # try:
    partial_ast.current_ast = "top level"

    program = partial_ast.create_ast(ast)
    program.module_declarations = ast_utils.parse_module_declarations(program)
    structs_lister = ast_transforms.StructLister()
    structs_lister.visit(program)
    struct_dep_graph = nx.DiGraph()
    for i, name in zip(structs_lister.structs, structs_lister.names):
        if name not in struct_dep_graph.nodes:
            struct_dep_graph.add_node(name)
        struct_deps_finder = ast_transforms.StructDependencyLister(structs_lister.names)
        struct_deps_finder.visit(i)
        struct_deps = struct_deps_finder.structs_used
  
        for j, pointing, point_name in zip(struct_deps, struct_deps_finder.is_pointer,
                                           struct_deps_finder.pointer_names):
            if j not in struct_dep_graph.nodes:
                struct_dep_graph.add_node(j)
            struct_dep_graph.add_edge(name, j, pointing=pointing, point_name=point_name)
    # program = ast_transforms.PropagateEnums().visit(program)
    # program = ast_transforms.Flatten_Classes(structs_lister.structs).visit(program)
    program.structures = ast_transforms.Structures(structs_lister.structs)
    program = run_ast_transformations(partial_ast, program, cfg, True)

    

    # functions_and_subroutines_builder = ast_transforms.FindFunctionAndSubroutines()
    # functions_and_subroutines_builder.visit(program)
    # listnames = [i.name for i in functions_and_subroutines_builder.names]
    # for i in functions_and_subroutines_builder.iblocks:
    #     if i not in listnames:
    #         functions_and_subroutines_builder.names.append(ast_internal_classes.Name_Node(name=i, type="VOID"))
    # program.iblocks = functions_and_subroutines_builder.iblocks
    # partial_ast.functions_and_subroutines = functions_and_subroutines_builder.names
    # program = ast_transforms.functionStatementEliminator(program)

    # program = ast_transforms.IfConditionExtractor().visit(program)

    # program = ast_transforms.TypeInference(program, assert_voids=False).visit(program)
    # program = ast_transforms.CallExtractor().visit(program)
    # program = ast_transforms.ArgumentExtractor(program).visit(program)
    # program = ast_transforms.FunctionCallTransformer().visit(program)
    # program = ast_transforms.FunctionToSubroutineDefiner().visit(program)

    # program = ast_transforms.optionalArgsExpander(program)

    # count = 0
    # for i in program.function_definitions:
    #     if isinstance(i, ast_internal_classes.Subroutine_Subprogram_Node):
    #         program.subroutine_definitions.append(i)
    #         partial_ast.functions_and_subroutines.append(i.name)
    #         count += 1
    # if count != len(program.function_definitions):
    #     raise NameError("Not all functions were transformed to subroutines")
    # for i in program.modules:
    #     count = 0
    #     for j in i.function_definitions:
    #         if isinstance(j, ast_internal_classes.Subroutine_Subprogram_Node):
    #             i.subroutine_definitions.append(j)
    #             partial_ast.functions_and_subroutines.append(j.name)
    #             count += 1
    #     if count != len(i.function_definitions):
    #         raise NameError("Not all functions were transformed to subroutines")
    #     i.function_definitions = []
    # program.function_definitions = []


    # program = ast_transforms.SignToIf().visit(program)
    # program = ast_transforms.ReplaceStructArgsLibraryNodes(program).visit(program)
    # program = ast_transforms.ReplaceArrayConstructor().visit(program)
    # program = ast_transforms.ArrayToLoop(program).visit(program)
    # program = ast_transforms.optionalArgsExpander(program)
    # program = ast_transforms.TypeInference(program, assert_voids=False).visit(program)
    # program = ast_transforms.ArgumentExtractor(program).visit(program)
    # program = ast_transforms.ReplaceStructArgsLibraryNodes(program).visit(program)
    # program = ast_transforms.ArrayToLoop(program).visit(program)
    # print("Before intrinsics")

    # prior_exception: Optional[NeedsTypeInferenceException] = None
    # for transformation in partial_ast.fortran_intrinsics().transformations():
    #     while True:
    #         try:
    #             transformation.initialize(program)
    #             program = transformation.visit(program)
    #             break
    #         except NeedsTypeInferenceException as e:

    #             if prior_exception is not None:
    #                 if e.line_number == prior_exception.line_number and e.func_name == prior_exception.func_name:
    #                     print("Running additional type inference didn't help! VOID type in the same place.")
    #                     raise RuntimeError()
    #             else:
    #                 prior_exception = e
    #             print("Running additional type inference")
    #             # FIXME: optimize func
    #             program = ast_transforms.TypeInference(program, assert_voids=False).visit(program)

    # print("After intrinsics")

    # program = ast_transforms.TypeInference(program).visit(program)
    # program = ast_transforms.ReplaceInterfaceBlocks(program, functions_and_subroutines_builder).visit(program)
    # program = ast_transforms.optionalArgsExpander(program)
    # program = ast_transforms.ArgumentExtractor(program).visit(program)
    # program = ast_transforms.ElementalFunctionExpander(
    #     functions_and_subroutines_builder.names, ast=program).visit(program)
   
    # program = ast_transforms.ForDeclarer().visit(program)
    # program = ast_transforms.PointerRemoval().visit(program)
    # program = ast_transforms.IndexExtractor(program, normalize_offsets).visit(program)

    # array_dims_info = ast_transforms.ArrayDimensionSymbolsMapper()
    # array_dims_info.visit(program)
    # program = ast_transforms.ArrayDimensionConfigInjector(array_dims_info, cfg.config_injections).visit(program)

    # structs_lister = ast_transforms.StructLister()
    # structs_lister.visit(program)
    # struct_dep_graph = nx.DiGraph()
    # for i, name in zip(structs_lister.structs, structs_lister.names):
    #     if name not in struct_dep_graph.nodes:
    #         struct_dep_graph.add_node(name)
    #     struct_deps_finder = ast_transforms.StructDependencyLister(structs_lister.names)
    #     struct_deps_finder.visit(i)
    #     struct_deps = struct_deps_finder.structs_used
    #     for j, pointing, point_name in zip(struct_deps, struct_deps_finder.is_pointer,
    #                                        struct_deps_finder.pointer_names):
    #         if j not in struct_dep_graph.nodes:
    #             struct_dep_graph.add_node(j)
    #         struct_dep_graph.add_edge(name, j, pointing=pointing, point_name=point_name)
    # cycles = nx.algorithms.cycles.simple_cycles(struct_dep_graph)
    # has_cycles = list(cycles)
    # cycles_we_cannot_ignore = []
    # for cycle in has_cycles:
    #     print(cycle)
    #     for i in cycle:
    #         is_pointer = struct_dep_graph.get_edge_data(i, cycle[(cycle.index(i) + 1) % len(cycle)])["pointing"]
    #         point_name = struct_dep_graph.get_edge_data(i, cycle[(cycle.index(i) + 1) % len(cycle)])["point_name"]
    #         # print(i,is_pointer)
    #         if is_pointer:
    #             actually_used_pointer_node_finder = ast_transforms.StructPointerChecker(i, cycle[
    #                 (cycle.index(i) + 1) % len(cycle)], point_name, structs_lister, struct_dep_graph, "simple")
    #             actually_used_pointer_node_finder.visit(program)
    #             # print(actually_used_pointer_node_finder.nodes)
    #             if len(actually_used_pointer_node_finder.nodes) == 0:
    #                 print("We can ignore this cycle")
    #                 program = ast_transforms.StructPointerEliminator(i, cycle[(cycle.index(i) + 1) % len(cycle)],
    #                                                                  point_name).visit(program)
    #             else:
    #                 cycles_we_cannot_ignore.append(cycle)
    # if len(cycles_we_cannot_ignore) > 0:
    #     raise NameError("Structs have cyclic dependencies")
    # print("Deleting struct members...")
    # struct_members_deleted = 0
    # for struct, name in zip(structs_lister.structs, structs_lister.names):
    #     struct_member_finder = ast_transforms.StructMemberLister()
    #     struct_member_finder.visit(struct)
    #     for member, is_pointer, point_name in zip(struct_member_finder.members, struct_member_finder.is_pointer,
    #                                               struct_member_finder.pointer_names):
    #         if is_pointer:
    #             actually_used_pointer_node_finder = ast_transforms.StructPointerChecker(name, member, point_name,
    #                                                                                     structs_lister,
    #                                                                                     struct_dep_graph, "full")
    #             actually_used_pointer_node_finder.visit(program)
    #             found = False
    #             for i in actually_used_pointer_node_finder.nodes:
    #                 nl = ast_transforms.FindNames()
    #                 nl.visit(i)
    #                 if point_name in nl.names:
    #                     found = True
    #                     break
    #             # print("Struct Name: ",name," Member Name: ",point_name, " Found: ", found)
    #             if not found:
    #                 # print("We can delete this member")
    #                 struct_members_deleted += 1
    #                 program = ast_transforms.StructPointerEliminator(name, member, point_name).visit(program)
    # print("Deleted " + str(struct_members_deleted) + " struct members.")
    # structs_lister = ast_transforms.StructLister()
    # structs_lister.visit(program)
    # struct_dep_graph = nx.DiGraph()
    # for i, name in zip(structs_lister.structs, structs_lister.names):
    #     if name not in struct_dep_graph.nodes:
    #         struct_dep_graph.add_node(name)
    #     struct_deps_finder = ast_transforms.StructDependencyLister(structs_lister.names)
    #     struct_deps_finder.visit(i)
    #     struct_deps = struct_deps_finder.structs_used
    #     for j, pointing, point_name in zip(struct_deps, struct_deps_finder.is_pointer,
    #                                        struct_deps_finder.pointer_names):
    #         if j not in struct_dep_graph.nodes:
    #             struct_dep_graph.add_node(j)
    #         struct_dep_graph.add_edge(name, j, pointing=pointing, point_name=point_name)

    program.structures = ast_transforms.Structures(structs_lister.structs)
    program.tables = partial_ast.symbols
    program.placeholders = partial_ast.placeholders
    program.placeholders_offsets = partial_ast.placeholders_offsets
    program.functions_and_subroutines = partial_ast.functions_and_subroutines
    unordered_modules = program.modules
    functions_and_subroutines_builder = ast_transforms.FindFunctionAndSubroutines()
    functions_and_subroutines_builder.visit(program)
    # arg_pruner = ast_transforms.ArgumentPruner(functions_and_subroutines_builder.nodes)
    # arg_pruner.visit(program)

    for j in program.subroutine_definitions:

        if subroutine_name is not None:
            if not subroutine_name + "_decon" in j.name.name:
                print("Skipping 1 ", j.name.name)
                continue

        if j.execution_part is None:
            continue

        print(f"Building SDFG {j.name.name}")
        startpoint = j
        ast2sdfg = AST_translator(__file__, multiple_sdfgs=False, startpoint=startpoint, sdfg_path=sdfgs_dir,
                                  normalize_offsets=normalize_offsets)
        sdfg = SDFG(j.name.name)
        ast2sdfg.functions_and_subroutines = functions_and_subroutines_builder.names
        ast2sdfg.structures = program.structures
        ast2sdfg.placeholders = program.placeholders
        ast2sdfg.placeholders_offsets = program.placeholders_offsets
        ast2sdfg.actual_offsets_per_sdfg[sdfg] = {}
        ast2sdfg.top_level = program
        ast2sdfg.globalsdfg = sdfg

        ast2sdfg.translate(program, sdfg, sdfg)

        print(f'Saving SDFG {os.path.join(sdfgs_dir, sdfg.name + "_raw_before_intrinsics_full.sdfgz")}')
        sdfg.save(os.path.join(sdfgs_dir, sdfg.name + "_raw_before_intrinsics_full.sdfgz"), compress=True)

        sdfg.apply_transformations_repeated(IntrinsicSDFGTransformation)

        try:
            sdfg.expand_library_nodes()
        except:
            print("Expansion failed for ", sdfg.name)
            continue

        sdfg.validate()
        print(f'Saving SDFG {os.path.join(sdfgs_dir, sdfg.name + "_validated_f.sdfgz")}')
        sdfg.save(os.path.join(sdfgs_dir, sdfg.name + "_validated_f.sdfgz"), compress=True)

        sdfg.simplify(verbose=True)
        print(f'Saving SDFG {os.path.join(sdfgs_dir, sdfg.name + "_simplified_tr.sdfgz")}')
        sdfg.save(os.path.join(sdfgs_dir, sdfg.name + "_simplified_f.sdfgz"), compress=True)

        print(f'Compiling SDFG {os.path.join(sdfgs_dir, sdfg.name + "_simplifiedf.sdfgz")}')
        sdfg.compile()

    for i in program.modules:

        # for path in source_list:

        #    if path.lower().find(i.name.name.lower()) != -1:
        #        mypath = path
        #        break

        for j in i.subroutine_definitions:

            if subroutine_name is not None:
                # special for radiation
                # if j.name.name!='cloud_generator_2139':
                # if j.name.name!='solver_mcica_lw_3321':
                # if "gas_optics_3057" not in j.name.name:
                #     print("Skipping 2 ", j.name.name)
                #     continue

                #   continue
                if subroutine_name == 'radiation':
                    if not 'radiation' == j.name.name:
                        print("Skipping ", j.name.name)
                        continue

                # elif not subroutine_name in j.name.name :
                #    print("Skipping ", j.name.name)
                #    continue

            if j.execution_part is None:
                continue
            print(f"Building SDFG {j.name.name}")
            startpoint = j
            ast2sdfg = AST_translator(
                __file__,
                multiple_sdfgs=False,
                startpoint=startpoint,
                sdfg_path=sdfgs_dir,
                # toplevel_subroutine_arg_names=arg_pruner.visited_funcs[toplevel_subroutine],
                # subroutine_used_names=arg_pruner.used_in_all_functions,
                normalize_offsets=normalize_offsets
            )
            sdfg = SDFG(j.name.name)
            ast2sdfg.functions_and_subroutines = functions_and_subroutines_builder.names
            ast2sdfg.structures = program.structures
            ast2sdfg.placeholders = program.placeholders
            ast2sdfg.placeholders_offsets = program.placeholders_offsets
            ast2sdfg.actual_offsets_per_sdfg[sdfg] = {}
            ast2sdfg.top_level = program
            ast2sdfg.globalsdfg = sdfg
            ast2sdfg.translate(program, sdfg, sdfg)
            sdfg.validate()
            sdfg.save(os.path.join(sdfgs_dir, sdfg.name + "_raw_before_intrinsics_full.sdfgz"), compress=True)
            sdfg.validate()
            sdfg.apply_transformations_repeated(IntrinsicSDFGTransformation)
            sdfg.validate()
            try:
                sdfg.expand_library_nodes()
            except:
                print("Expansion failed for ", sdfg.name)
                continue

            sdfg.validate()
            sdfg.save(os.path.join(sdfgs_dir, sdfg.name + "_validated_dbg.sdfgz"), compress=True)
            sdfg.validate()
            sdfg.simplify(verbose=True)
            print(f'Saving SDFG {os.path.join(sdfgs_dir, sdfg.name + "_simplified_tr.sdfgz")}')
            sdfg.save(os.path.join(sdfgs_dir, sdfg.name + "_simplified_dbg.sdfgz"), compress=True)
            sdfg.validate()
            print(f'Compiling SDFG {os.path.join(sdfgs_dir, sdfg.name + "_simplifiedf.sdfgz")}')
            sdfg.compile()

    # return sdfg
