# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

import copy
from venv import create
import warnings

from dace.data import Scalar, Structure

import dace.frontend.fortran.ast_components as ast_components
import dace.frontend.fortran.ast_transforms as ast_transforms
import dace.frontend.fortran.ast_utils as ast_utils
import dace.frontend.fortran.ast_internal_classes as ast_internal_classes
from dace.frontend.fortran.intrinsics import IntrinsicSDFGTransformation
from typing import Dict, List, Optional, Tuple, Set
from dace import dtypes
from dace import subsets as subs
from dace import Language as lang
from dace import data as dat
from dace import SDFG, InterstateEdge, Memlet, pointer, nodes, SDFGState
from dace import symbolic as sym
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from copy import deepcopy as dpcp

from dace.properties import CodeBlock
from fparser.two import Fortran2003 as f03
from fparser.two.parser import ParserFactory as pf
from fparser.common.readfortran import FortranStringReader as fsr
from fparser.common.readfortran import FortranFileReader as ffr
from fparser.two.symbol_table import SymbolTable

import os
from os import path
from shutil import copyfile
import networkx as nx

global_struct_instance_counter=0

def find_access_in_destinations(substate,substate_destinations,name):
                                wv=None
                                already_there=False
                                for i in substate_destinations:
                                    if i.data==name:
                                        wv=i
                                        already_there=True
                                        break
                                if not already_there:
                                    wv = substate.add_write(name)
                                return wv,already_there    
def find_access_in_sources(substate,substate_sources,name):
    re=None
    already_there=False
    for i in substate_sources:
        if i.data==name:
            re=i
            already_there=True
            break
    if not already_there:
        re=substate.add_read(name)  
    return re,already_there      

def add_views_recursive(sdfg,name,datatype_to_add,struct_views,name_mapping,registered_types,chain,actual_offsets_per_sdfg,names_of_object_in_parent_sdfg,actual_offsets_per_parent_sdfg): 
    if not isinstance(datatype_to_add,dat.Structure):
        #print("Not adding: ", str(datatype_to_add))
        if isinstance(datatype_to_add,dat.ContainerArray):
            datatype_to_add=datatype_to_add.stype
    for i in datatype_to_add.members:
        current_dtype=datatype_to_add.members[i].dtype
        for other_type in registered_types:
            if current_dtype.dtype==registered_types[other_type].dtype:
                other_type_obj=registered_types[other_type]
                add_views_recursive(sdfg,name,datatype_to_add.members[i],struct_views,name_mapping,registered_types,chain+[i],actual_offsets_per_sdfg,names_of_object_in_parent_sdfg,actual_offsets_per_parent_sdfg)
                #for j in other_type_obj.members:
                #    sdfg.add_view(name_mapping[name] + "_" + i +"_"+ j,other_type_obj.members[j].shape,other_type_obj.members[j].dtype)
                #    name_mapping[name + "_" + i +"_"+ j] = name_mapping[name] + "_" + i +"_"+ j
                #    struct_views[name_mapping[name] + "_" + i+"_"+ j]=[name_mapping[name],i,j]
        if len(chain)>0:
            join_chain="_"+"_".join(chain)
        else:
            join_chain="" 
        current_member=datatype_to_add.members[i]    
        
        if str(datatype_to_add.members[i].dtype.base_type) in registered_types: 


            view_to_member = dat.View.view(datatype_to_add.members[i])
            if sdfg.arrays.get(name_mapping[name]+ join_chain + "_" + i) is None:
                sdfg.arrays[name_mapping[name]+ join_chain + "_" + i] = view_to_member           
        else:
            if sdfg.arrays.get(name_mapping[name]+ join_chain + "_" + i) is None:    
                sdfg.add_view(name_mapping[name]+ join_chain + "_" + i,datatype_to_add.members[i].shape,datatype_to_add.members[i].dtype,strides=datatype_to_add.members[i].strides)
        if names_of_object_in_parent_sdfg.get(name_mapping[name]) is not None:
            if actual_offsets_per_parent_sdfg.get(names_of_object_in_parent_sdfg[name_mapping[name]]+ join_chain + "_" + i) is not None:
                actual_offsets_per_sdfg[name_mapping[name]+ join_chain + "_" + i]= actual_offsets_per_parent_sdfg[names_of_object_in_parent_sdfg[name_mapping[name]]+ join_chain + "_" + i]
            else:
                #print("No offsets in sdfg: ",sdfg.name ," for: ",names_of_object_in_parent_sdfg[name_mapping[name]]+ join_chain + "_" + i)    
                actual_offsets_per_sdfg[name_mapping[name]+ join_chain + "_" + i]=[1]*len(datatype_to_add.members[i].shape)
        name_mapping[name_mapping[name]+ join_chain + "_" + i] = name_mapping[name] + join_chain + "_" + i
        struct_views[name_mapping[name]+  join_chain+ "_" + i]=[name_mapping[name]]+chain+[i]

def add_deferred_shape_assigns_for_structs(structures: ast_transforms.Structures, decl:ast_internal_classes.Var_Decl_Node, sdfg: SDFG, assign_state: SDFGState,name:str,name_:str,placeholders,placeholders_offsets,object,names_to_replace,actual_offsets_per_sdfg):
                            if not structures.is_struct(decl.type):
                                #print("Not adding defferred shape assigns for: ", decl.type,decl.name)
                                return

                            if isinstance(object,dat.ContainerArray):
                                struct_type = object.stype
                            else:
                                struct_type = object
                            global global_struct_instance_counter
                            local_counter=global_struct_instance_counter
                            global_struct_instance_counter+=1
                            overall_ast_struct_type=structures.get_definition(decl.type)
                            counter=0
                            listofmember=list(struct_type.members)
                            #print("Struct: "+decl.name +" Struct members: "+ str(len(listofmember))+ " Definition members: "+str(len(list(overall_ast_struct_type.vars.items()))))
                            
                            for ast_struct_type in overall_ast_struct_type.vars.items():
                                ast_struct_type=ast_struct_type[1]
                                var=struct_type.members[ast_struct_type.name]
                                
                                if isinstance(var,dat.ContainerArray):
                                    var_type=var.stype
                                else:
                                    var_type=var
                                
                                #print(ast_struct_type.name,var_type.__class__)
                                if isinstance(object.members[ast_struct_type.name],dat.Structure):
                                    
                                        add_deferred_shape_assigns_for_structs(structures, ast_struct_type, sdfg, assign_state, f"{name}->{ast_struct_type.name}",f"{ast_struct_type.name}_{name_}",placeholders,placeholders_offsets,object.members[ast_struct_type.name],names_to_replace,actual_offsets_per_sdfg)
                                elif isinstance(var_type,dat.Structure)       :
                                    add_deferred_shape_assigns_for_structs(structures, ast_struct_type, sdfg, assign_state, f"{name}->{ast_struct_type.name}",f"{ast_struct_type.name}_{name_}",placeholders,placeholders_offsets,var_type,names_to_replace,actual_offsets_per_sdfg)
                                #print(ast_struct_type)
                                #print(ast_struct_type.__class__)
                              
                                if ast_struct_type.sizes is None or len(ast_struct_type.sizes) == 0:
                                    continue    
                                offsets_to_replace=[]
                                sanity_count=0

                                for offset in ast_struct_type.offsets:
                                    if isinstance(offset,ast_internal_classes.Name_Node):
                                        if hasattr(offset, "name"):
                                            if sdfg.symbols.get(offset.name) is None:
                                                sdfg.add_symbol(offset.name, dtypes.int32)
                                        sanity_count+=1        
                                        if offset.name.startswith('__f2dace_SOA'):
                                            newoffset=offset.name+"_"+name_+"_"+str(local_counter)
                                            sdfg.append_global_code(f"{dtypes.int32.ctype} {newoffset};\n")
                                            #prog hack
                                            if name.endswith("prog"):
                                                sdfg.append_init_code(f"{newoffset} = {name}[0]->{offset.name};\n")
                                            else:
                                                sdfg.append_init_code(f"{newoffset} = {name}->{offset.name};\n")
                                            
                                            sdfg.add_symbol(newoffset, dtypes.int32)
                                            offsets_to_replace.append(newoffset)
                                            names_to_replace[offset.name]=newoffset
                                        else:
                                            #print("not replacing",offset.name)
                                            offsets_to_replace.append(offset.name)    
                                    else:
                                        sanity_count+=1 
                                        #print("not replacing not namenode",offset)
                                        offsets_to_replace.append(offset)        
                                if sanity_count==len(ast_struct_type.offsets):
                                    #print("adding offsets for: "+name.replace("->","_")+"_"+ast_struct_type.name)
                                    actual_offsets_per_sdfg[name.replace("->","_")+"_"+ast_struct_type.name]=offsets_to_replace
                                        
                                # for assumed shape, all vars starts with the same prefix
                                for size in ast_struct_type.sizes:
                                    if isinstance(size, ast_internal_classes.Name_Node):# and  size.name.startswith('__f2dace_A'):
                                    
                                        
                                        if hasattr(size, "name"):
                                            if sdfg.symbols.get(size.name) is None:
                                                #new_name=sdfg._find_new_name(size.name)
                                                sdfg.add_symbol(size.name, dtypes.int32)
                        
                                            if  size.name.startswith('__f2dace_SA'):  
                                                #newsize=ast_internal_classes.Name_Node(name=size.name+"_"+str(local_counter),parent=size.parent,type=size.type)                                             
                                                newsize=  size.name+"_"+name_+"_"+str(local_counter)
                                                names_to_replace[size.name]=newsize
                                                #var_type.sizes[var_type.sizes.index(size)]=newsize
                                                sdfg.append_global_code(f"{dtypes.int32.ctype} {newsize};\n") 
                                                if name.endswith("prog"):
                                                    sdfg.append_init_code(f"{newsize} = {name}[0]->{size.name};\n")
                                                else:
                                                    sdfg.append_init_code(f"{newsize} = {name}->{size.name};\n")
                                                sdfg.add_symbol(newsize, dtypes.int32)
                                                if isinstance(object,dat.Structure):
                                                    shape2=dpcp(object.members[ast_struct_type.name].shape)
                                                else:
                                                    shape2=dpcp(object.stype.members[ast_struct_type.name].shape)
                                                shapelist=list(shape2)
                                                shapelist[ast_struct_type.sizes.index(size)]= sym.pystr_to_symbolic(newsize)
                                                shape_replace=tuple(shapelist)
                                                viewname= f"{name}->{ast_struct_type.name}"

                                                viewname=viewname.replace("->","_")
                                                #view=sdfg.arrays[viewname]
                                                strides = [dat._prod(shapelist[:i]) for i in range(len(shapelist))]
                                                if isinstance(object.members[ast_struct_type.name],dat.ContainerArray):
                                                    tmpobject=dat.ContainerArray(object.members[ast_struct_type.name].stype,shape_replace,strides=strides)
                                                    
                                                
                                                elif isinstance(object.members[ast_struct_type.name],dat.Array):  
                                                    tmpobject=dat.Array(object.members[ast_struct_type.name].dtype,shape_replace,strides=strides)
                                                        
                                                else:
                                                    raise ValueError("Unknown type"+str(tmpobject.__class__))
                                                object.members.pop(ast_struct_type.name)
                                                object.members[ast_struct_type.name]=tmpobject
                                                tmpview=dat.View.view(object.members[ast_struct_type.name])
                                                if sdfg.arrays.get(viewname) is not None:
                                                    del sdfg.arrays[viewname]
                                                sdfg.arrays[viewname]=tmpview    
                                                #if placeholders.get(size.name) is not None:
                                                #    placeholders[newsize]=placeholders[size.name]
                                                



class AST_translator:
    """  
    This class is responsible for translating the internal AST into a SDFG.
    """
    def __init__(
        self,
        ast: ast_components.InternalFortranAst,
        source: str,
        use_experimental_cfg_blocks: bool = False,
        multiple_sdfgs: bool = False,
        startpoint=None,
        sdfg_path=None,
        toplevel_subroutine: Optional[str] = None,
        normalize_offsets = False
    ):
        """
        :ast: The internal fortran AST to be used for translation
        :source: The source file name from which the AST was generated
        """
        self.sdfg_path=sdfg_path
        self.count_of_struct_symbols_lifted=0
        self.registered_types = {}
        self.transient_mode=True
        self.tables = ast.tables
        self.startpoint = startpoint
        self.top_level = None
        self.globalsdfg = None
        self.multiple_sdfgs = multiple_sdfgs
        self.functions_and_subroutines = ast.functions_and_subroutines
        self.name_mapping = ast_utils.NameMap()
        self.actual_offsets_per_sdfg={}
        self.names_of_object_in_parent_sdfg={}
        self.contexts = {}
        self.views = 0
        self.libstates = []
        self.file_name = source
        self.unallocated_arrays = []
        self.all_array_names = []
        self.last_sdfg_states = {}
        self.last_loop_continues = {}
        self.last_loop_continues_stack = {}
        self.already_has_edge_back_continue = {}
        self.last_loop_breaks = {}
        self.last_returns = {}
        self.module_vars = []
        self.sdfgs_count=0
        self.libraries = {}
        self.local_not_transient_because_assign={}
        self.struct_views={}
        self.last_call_expression = {}
        self.struct_view_count = 0
        self.structures = ast.structures
        self.placeholders = ast.placeholders
        self.placeholders_offsets = ast.placeholders_offsets
        #self.iblocks=ast.iblocks
        self.replace_names = {}
        self.toplevel_subroutine = toplevel_subroutine
        self.normalize_offsets = normalize_offsets
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
        }
        self.use_experimental_cfg_blocks = use_experimental_cfg_blocks

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
                #TODO: This is bandaid.
                if type=="VOID":
                    return ast_utils.fortrantypes2dacetypes["DOUBLE"]
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

    def translate(self, node: ast_internal_classes.FNode, sdfg: SDFG, cfg: Optional[ControlFlowRegion] = None):
        """
        This function is responsible for translating the AST into a SDFG.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        :note: This function is recursive and will call itself for all child nodes
        :note: This function will call the appropriate function for the node type
        :note: The dictionary ast_elements, part of the class itself contains all functions that are called for the different node types
        """
        if not cfg:
            cfg = sdfg
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
        :note: This function is recursive and will call itself for all child nodes
        :note: This function will call the appropriate function for the node type
        :note: The dictionary ast_elements, part of the class itself contains all functions that are called for the different node types
        """
        self.globalsdfg = sdfg
        for i in node.modules:
            structs_lister=ast_transforms.StructLister()
            if i.specification_part is not None:
                structs_lister.visit(i.specification_part)
            struct_dep_graph=nx.DiGraph()
            for ii,name in zip(structs_lister.structs,structs_lister.names):
                if name not in struct_dep_graph.nodes:
                    struct_dep_graph.add_node(name)
                struct_deps_finder=ast_transforms.StructDependencyLister(structs_lister.names)
                struct_deps_finder.visit(ii)
                struct_deps=struct_deps_finder.structs_used
                #print(struct_deps)
                for j,pointing,point_name in zip(struct_deps,struct_deps_finder.is_pointer,struct_deps_finder.pointer_names):
                    if j not in struct_dep_graph.nodes:
                        struct_dep_graph.add_node(j)
                    struct_dep_graph.add_edge(name,j,pointing=pointing,point_name=point_name)
            parse_order = list(reversed(list(nx.topological_sort(struct_dep_graph))))        
            for jj in parse_order:
              for j in i.specification_part.typedecls:
                    if j.name.name==jj:
                        self.translate(j, sdfg, cfg)
                        if j.__class__.__name__ != "Derived_Type_Def_Node":
                            for k in j.vardecl:
                                self.module_vars.append((k.name, i.name))
            if i.specification_part is not None:                                
              for j in i.specification_part.symbols:
                self.translate(j, sdfg, cfg)
                if  isinstance(j, ast_internal_classes.Symbol_Array_Decl_Node):
                    self.module_vars.append((j.name, i.name))
                elif isinstance(j, ast_internal_classes.Symbol_Decl_Node):
                    self.module_vars.append((j.name, i.name))
                else:
                    raise ValueError("Unknown symbol type")
              for j in i.specification_part.specifications:
                self.translate(j, sdfg, cfg)
                for k in j.vardecl:
                    self.module_vars.append((k.name, i.name))
        ast_utils.add_simple_state_to_sdfg(self, sdfg, "GlobalDefEnd")            
        if node.main_program is not None:

            for i in node.main_program.specification_part.typedecls:
                self.translate(i, sdfg, cfg)
            for i in node.main_program.specification_part.symbols:
                self.translate(i, sdfg, cfg)
               
            for i in node.main_program.specification_part.specifications:
                self.translate(i, sdfg, cfg)
            for i in node.main_program.specification_part.specifications:

                        ast_utils.add_simple_state_to_sdfg(self, sdfg, "start_struct_size")
                        assign_state = ast_utils.add_simple_state_to_sdfg(self, sdfg, "assign_struct_sizes")

                        for decl in i.vardecl:
                            add_deferred_shape_assigns_for_structs(self.structures,decl, sdfg, assign_state, decl.name,decl.name,self.placeholders,self.placeholders_offsets,sdfg.arrays[self.name_mapping[sdfg][decl.name]],self.replace_names,self.actual_offsets_per_sdfg[sdfg])

            self.translate(node.main_program.execution_part.execution, sdfg, cfg)
        else: 
            if self.startpoint is None:
                raise ValueError("No main program or start point found")
            else:

                if self.startpoint.specification_part is not None:
                    self.transient_mode=True

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
                            add_deferred_shape_assigns_for_structs(self.structures,decl, sdfg, assign_state, decl.name,decl.name,self.placeholders,self.placeholders_offsets,sdfg.arrays[self.name_mapping[sdfg][decl.name]],self.replace_names,self.actual_offsets_per_sdfg[sdfg])

                self.transient_mode=True
                self.translate(self.startpoint.execution_part.execution, sdfg, cfg)

    def pointerassignment2sdfg(self, node: ast_internal_classes.Pointer_Assignment_Stmt_Node, sdfg: SDFG, cfg: ControlFlowRegion):
        """
        This function is responsible for translating Fortran pointer assignments into a SDFG.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        """
        if self.name_mapping[sdfg][node.name_pointer.name] in sdfg.arrays:
            shapenames = [sdfg.arrays[self.name_mapping[sdfg][node.name_pointer.name]].shape[i] for i in range(len(sdfg.arrays[self.name_mapping[sdfg][node.name_pointer.name]].shape))]
            offsetnames = self.actual_offsets_per_sdfg[sdfg][node.name_pointer.name]
            [sdfg.arrays[self.name_mapping[sdfg][node.name_pointer.name]].offset[i] for i in range(len(sdfg.arrays[self.name_mapping[sdfg][node.name_pointer.name]].offset))]
            #for i in shapenames:
            #    if str(i) in sdfg.symbols:
            #        sdfg.symbols.pop(str(i))
                #if sdfg.parent_nsdfg_node is not None:     
                    #if str(i) in sdfg.parent_nsdfg_node.symbol_mapping:
                        #sdfg.parent_nsdfg_node.symbol_mapping.pop(str(i))   

            #for i in offsetnames:
                #if str(i) in sdfg.symbols:
                #    sdfg.symbols.pop(str(i))
                #if sdfg.parent_nsdfg_node is not None:     
                    #if str(i) in sdfg.parent_nsdfg_node.symbol_mapping:
                        #sdfg.parent_nsdfg_node.symbol_mapping.pop(str(i))            
            sdfg.arrays.pop(self.name_mapping[sdfg][node.name_pointer.name])
        if isinstance(node.name_target, ast_internal_classes.Data_Ref_Node):
            if node.name_target.parent_ref.name not in self.name_mapping[sdfg]: 
                raise ValueError("Unknown variable " + node.name_target.name)
            if isinstance(node.name_target.part_ref,ast_internal_classes.Data_Ref_Node):
                self.name_mapping[sdfg][node.name_pointer.name] = self.name_mapping[sdfg][node.name_target.parent_ref.name+"_"+node.name_target.part_ref.parent_ref.name+"_"+node.name_target.part_ref.part_ref.name]
                #self.replace_names[node.name_pointer.name]=self.name_mapping[sdfg][node.name_target.parent_ref.name+"_"+node.name_target.part_ref.parent_ref.name+"_"+node.name_target.part_ref.part_ref.name]
                target=sdfg.arrays[self.name_mapping[sdfg][node.name_target.parent_ref.name+"_"+node.name_target.part_ref.parent_ref.name+"_"+node.name_target.part_ref.part_ref.name]]        
                #for i in self.actual_offsets_per_sdfg[sdfg]:
                #    print(i)
                actual_offsets=self.actual_offsets_per_sdfg[sdfg][node.name_target.parent_ref.name+"_"+node.name_target.part_ref.parent_ref.name+"_"+node.name_target.part_ref.part_ref.name]
                
                for i in shapenames:
                    self.replace_names[str(i)]=str(target.shape[shapenames.index(i)])
                for i in offsetnames:
                    self.replace_names[str(i)]=str(actual_offsets[offsetnames.index(i)])
            else:    
                self.name_mapping[sdfg][node.name_pointer.name] = self.name_mapping[sdfg][node.name_target.parent_ref.name+"_"+node.name_target.part_ref.name]
                self.replace_names[node.name_pointer.name]=self.name_mapping[sdfg][node.name_target.parent_ref.name+"_"+node.name_target.part_ref.name]
                target=sdfg.arrays[self.name_mapping[sdfg][node.name_target.parent_ref.name+"_"+node.name_target.part_ref.name]]
                actual_offsets=self.actual_offsets_per_sdfg[sdfg][node.name_target.parent_ref.name+"_"+node.name_target.part_ref.name]
                for i in shapenames:
                    self.replace_names[str(i)]=str(target.shape[shapenames.index(i)])
                for i in offsetnames:
                    self.replace_names[str(i)]=str(actual_offsets[offsetnames.index(i)])

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

    def derivedtypedef2sdfg(self, node: ast_internal_classes.Derived_Type_Def_Node, sdfg: SDFG, cfg: ControlFlowRegion):
        """
        This function is responsible for registering Fortran derived type declarations into a SDFG as nested data types.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        """
        name = node.name.name
        if node.component_part is None:
            components=[]
        else:     
            components = node.component_part.component_def_stmts
        dict_setup = {}
        for i in components:
            j = i.vars
            for k in j.vardecl:
                complex_datatype=False
                datatype = self.get_dace_type(k.type)
                if isinstance(datatype, dat.Structure):
                    complex_datatype=True
                if k.sizes is not None:
                    sizes = []
                    offset = []
                    offset_value = 0 if self.normalize_offsets else -1
                    for i in k.sizes:
                        tw = ast_utils.TaskletWriter([], [], sdfg, self.name_mapping,placeholders=self.placeholders,placeholders_offsets=self.placeholders_offsets,rename_dict=self.replace_names)
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
        """

        for i in node.execution:
            self.translate(i, sdfg, cfg)

    def allocate2sdfg(self, node: ast_internal_classes.Allocate_Stmt_Node, sdfg: SDFG, cfg: ControlFlowRegion):
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
                    offset_value = 0 if self.normalize_offsets else -1
                    sizes = []
                    offset = []
                    for j in i.shape.shape_list:
                        tw = ast_utils.TaskletWriter([], [], sdfg, self.name_mapping,placeholders=self.placeholders,placeholders_offsets=self.placeholders_offsets,rename_dict=self.replace_names)
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
        #TODO implement
        raise NotImplementedError("Fortran write statements are not implemented yet")

    def ifstmt2sdfg(self, node: ast_internal_classes.If_Stmt_Node, sdfg: SDFG, cfg: ControlFlowRegion):
        """
        This function is responsible for translating Fortran if statements into a SDFG.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        """

        name = f"If_l_{str(node.line_number[0])}_c_{str(node.line_number[1])}"
        begin_state = ast_utils.add_simple_state_to_sdfg(self, cfg, f"Begin{name}")
        guard_substate = cfg.add_state(f"Guard{name}")
        cfg.add_edge(begin_state, guard_substate, InterstateEdge())

        condition = ast_utils.ProcessedWriter(sdfg, self.name_mapping,self.placeholders,self.placeholders_offsets,self.replace_names).write_code(node.cond)

        body_ifstart_state = cfg.add_state(f"BodyIfStart{name}")
        self.last_sdfg_states[cfg] = body_ifstart_state
        self.translate(node.body, sdfg, cfg)
        final_substate = cfg.add_state(f"MergeState{name}")

        cfg.add_edge(guard_substate, body_ifstart_state, InterstateEdge(condition))

        if self.last_sdfg_states[cfg] not in [
                self.last_loop_breaks.get(cfg),
                self.last_loop_continues.get(cfg),
                self.last_returns.get(cfg),
                self.already_has_edge_back_continue.get(cfg)
        ]:
            body_ifend_state = ast_utils.add_simple_state_to_sdfg(self, cfg, f"BodyIfEnd{name}")
            cfg.add_edge(body_ifend_state, final_substate, InterstateEdge())

        if len(node.body_else.execution) > 0:
            name_else = f"Else_l_{str(node.line_number[0])}_c_{str(node.line_number[1])}"
            body_elsestart_state = cfg.add_state("BodyElseStart" + name_else)
            self.last_sdfg_states[cfg] = body_elsestart_state
            self.translate(node.body_else, sdfg, cfg)
            body_elseend_state = ast_utils.add_simple_state_to_sdfg(self, cfg, f"BodyElseEnd{name_else}")
            cfg.add_edge(guard_substate, body_elsestart_state, InterstateEdge("not (" + condition + ")"))
            cfg.add_edge(body_elseend_state, final_substate, InterstateEdge())
        else:
            cfg.add_edge(guard_substate, final_substate, InterstateEdge("not (" + condition + ")"))
        self.last_sdfg_states[cfg] = final_substate

    def forstmt2sdfg(self, node: ast_internal_classes.For_Stmt_Node, sdfg: SDFG, cfg: ControlFlowRegion):
        """
        This function is responsible for translating Fortran for statements into a SDFG.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        """

        if not self.use_experimental_cfg_blocks:
            declloop = False
            name = "FOR_l_" + str(node.line_number[0]) + "_c_" + str(node.line_number[1])
            begin_state = ast_utils.add_simple_state_to_sdfg(self, cfg, "Begin" + name)
            guard_substate = cfg.add_state("Guard" + name)
            final_substate = cfg.add_state("Merge" + name)
            self.last_sdfg_states[cfg] = final_substate
            decl_node = node.init
            entry = {}
            if isinstance(decl_node, ast_internal_classes.BinOp_Node):
                if sdfg.symbols.get(decl_node.lval.name) is not None:
                    iter_name = decl_node.lval.name
                elif self.name_mapping[sdfg].get(decl_node.lval.name) is not None:
                    iter_name = self.name_mapping[sdfg][decl_node.lval.name]
                else:
                    raise ValueError("Unknown variable " + decl_node.lval.name)
                entry[iter_name] = ast_utils.ProcessedWriter(sdfg, self.name_mapping,placeholders=self.placeholders,placeholders_offsets=self.placeholders_offsets,rename_dict=self.replace_names).write_code(decl_node.rval)

            cfg.add_edge(begin_state, guard_substate, InterstateEdge(assignments=entry))

            condition = ast_utils.ProcessedWriter(sdfg, self.name_mapping,placeholders=self.placeholders,placeholders_offsets=self.placeholders_offsets,rename_dict=self.replace_names).write_code(node.cond)

            increment = "i+0+1"
            if isinstance(node.iter, ast_internal_classes.BinOp_Node):
                increment = ast_utils.ProcessedWriter(sdfg, self.name_mapping,placeholders=self.placeholders,placeholders_offsets=self.placeholders_offsets,rename_dict=self.replace_names).write_code(node.iter.rval)
            entry = {iter_name: increment}

            begin_loop_state = cfg.add_state("BeginLoop" + name)
            end_loop_state = cfg.add_state("EndLoop" + name)
            self.last_sdfg_states[cfg] = begin_loop_state
            self.last_loop_continues[cfg] = end_loop_state
            if self.last_loop_continues_stack.get(cfg) is None:
                self.last_loop_continues_stack[cfg] = []
            self.last_loop_continues_stack[cfg].append(end_loop_state)
            self.translate(node.body, sdfg, cfg)

            cfg.add_edge(self.last_sdfg_states[cfg], end_loop_state, InterstateEdge())
            cfg.add_edge(guard_substate, begin_loop_state, InterstateEdge(condition))
            cfg.add_edge(end_loop_state, guard_substate, InterstateEdge(assignments=entry))
            cfg.add_edge(guard_substate, final_substate, InterstateEdge(f"not ({condition})"))
            self.last_sdfg_states[cfg] = final_substate
            self.last_loop_continues_stack[cfg].pop()
            if len(self.last_loop_continues_stack[cfg]) > 0:
                self.last_loop_continues[cfg] = self.last_loop_continues_stack[cfg][-1]
            else:
                self.last_loop_continues[cfg] = None
        else:
            name = "FOR_l_" + str(node.line_number[0]) + "_c_" + str(node.line_number[1])
            decl_node = node.init
            entry = {}
            if isinstance(decl_node, ast_internal_classes.BinOp_Node):
                if sdfg.symbols.get(decl_node.lval.name) is not None:
                    iter_name = decl_node.lval.name
                elif self.name_mapping[sdfg].get(decl_node.lval.name) is not None:
                    iter_name = self.name_mapping[sdfg][decl_node.lval.name]
                else:
                    raise ValueError("Unknown variable " + decl_node.lval.name)
                entry[iter_name] = ast_utils.ProcessedWriter(sdfg, self.name_mapping, placeholders=self.placeholders, placeholders_offsets=self.placeholders_offsets, rename_dict=self.replace_names).write_code(decl_node.rval)

            condition = ast_utils.ProcessedWriter(sdfg, self.name_mapping,placeholders=self.placeholders,placeholders_offsets=self.placeholders_offsets,rename_dict=self.replace_names).write_code(node.cond)

            increment = "i+0+1"
            if isinstance(node.iter, ast_internal_classes.BinOp_Node):
                increment = ast_utils.ProcessedWriter(sdfg, self.name_mapping, placeholders=self.placeholders, placeholders_offsets=self.placeholders_offsets, rename_dict=self.replace_names).write_code(node.iter.rval)

            loop_region = LoopRegion(name, condition, iter_name, f"{iter_name} = {entry[iter_name]}",
                                     f"{iter_name} = {increment}")
            is_start = self.last_sdfg_states.get(cfg) is None
            cfg.add_node(loop_region, is_start_block=is_start)
            if not is_start:
                cfg.add_edge(self.last_sdfg_states[cfg], loop_region, InterstateEdge())
            self.last_sdfg_states[cfg] = loop_region

            begin_loop_state = loop_region.add_state("BeginLoop" + name, is_start_block=True)
            self.last_sdfg_states[loop_region] = begin_loop_state

            self.translate(node.body, sdfg, loop_region)

    def symbol2sdfg(self, node: ast_internal_classes.Symbol_Decl_Node, sdfg: SDFG, cfg: ControlFlowRegion):
        """
        This function is responsible for translating Fortran symbol declarations into a SDFG.
        :param node: The node to be translated
        :param sdfg: The SDFG to which the node should be translated
        """
        if node.name=="modname": return
        if self.contexts.get(sdfg.name) is None:
            self.contexts[sdfg.name] = ast_utils.Context(name=sdfg.name)
        if self.contexts[sdfg.name].constants.get(node.name) is None:
            if isinstance(node.init, ast_internal_classes.Int_Literal_Node) or isinstance(
                    node.init, ast_internal_classes.Real_Literal_Node):
                self.contexts[sdfg.name].constants[node.name] = node.init.value
            elif isinstance(node.init, ast_internal_classes.Name_Node):
                self.contexts[sdfg.name].constants[node.name] = self.contexts[sdfg.name].constants[node.init.name]
            else:
                tw = ast_utils.TaskletWriter([], [], sdfg, self.name_mapping,placeholders=self.placeholders,placeholders_offsets=self.placeholders_offsets,rename_dict=self.replace_names)
                if node.init is not None:
                    text = tw.write_code(node.init)
                    self.contexts[sdfg.name].constants[node.name] = sym.pystr_to_symbolic(text)
           
        datatype = self.get_dace_type(node.type)
        if node.name not in sdfg.symbols:
            sdfg.add_symbol(node.name, datatype)
            if self.last_sdfg_states.get(cfg) is None:
                bstate = cfg.add_state("SDFGbegin", is_start_state=True)
                self.last_sdfg_states[cfg] = bstate
            if node.init is not None:
                substate = cfg.add_state(f"Dummystate_{node.name}")
                increment = ast_utils.TaskletWriter([], [], sdfg, self.name_mapping,placeholders=self.placeholders,placeholders_offsets=self.placeholders_offsets,rename_dict=self.replace_names).write_code(node.init)

                entry = {node.name: increment}
                cfg.add_edge(self.last_sdfg_states[cfg], substate, InterstateEdge(assignments=entry))
                self.last_sdfg_states[cfg] = substate

    def symbolarray2sdfg(self, node: ast_internal_classes.Symbol_Array_Decl_Node, sdfg: SDFG, cfg: ControlFlowRegion):

        return NotImplementedError(
            "Symbol_Decl_Node not implemented. This should be done via a transformation that itemizes the constant array."
        )

    def subroutine2sdfg(self, node: ast_internal_classes.Subroutine_Subprogram_Node, sdfg: SDFG,
                        cfg: ControlFlowRegion):
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
        outputnodefinder = ast_transforms.FindOutputs(thourough=True)
        outputnodefinder.visit(node)
        output_vars = outputnodefinder.nodes
        write_names = list(dict.fromkeys([i.name for i in output_vars]))
        read_names = list(dict.fromkeys([i.name for i in input_vars]))

        # Collect the parameters and the function signature to comnpare and link
        parameters = node.args.copy()
        my_name_sdfg=node.name.name+str(self.sdfgs_count)
        new_sdfg = SDFG(my_name_sdfg)
        self.sdfgs_count+=1
        self.actual_offsets_per_sdfg[new_sdfg]={}
        self.names_of_object_in_parent_sdfg[new_sdfg]={}
        substate = ast_utils.add_simple_state_to_sdfg(self, cfg, "state" + my_name_sdfg)

        
        variables_in_call = []
        if self.last_call_expression.get(sdfg) is not None:
            variables_in_call = self.last_call_expression[sdfg]

        # Sanity check to make sure the parameter numbers match
        if not ((len(variables_in_call) == len(parameters)) or
                (len(variables_in_call) == len(parameters) + 1
                 and not isinstance(node.result_type, ast_internal_classes.Void))):
            print('Variables in call', len(variables_in_call))
            print('Parameters', len(parameters))
            #for i in variables_in_call:
            #    print("VAR CALL: ", i.name)
            #for j in parameters:
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
            if isinstance(variable, ast_internal_classes.Name_Node):
                varname = variable.name
            elif isinstance(variable, ast_internal_classes.Actual_Arg_Spec_Node):
                varname = variable.arg_name.name
            elif isinstance(variable, ast_internal_classes.Array_Subscript_Node):
                varname = variable.name.name
            elif isinstance(variable, ast_internal_classes.Data_Ref_Node):
                varname = ast_utils.get_name(variable)

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
        self.local_not_transient_because_assign[my_name_sdfg]=[]  
        for lit, litval in zip(literals, literal_values):
            local_name = lit
            self.local_not_transient_because_assign[my_name_sdfg].append(local_name.name)
            assigns.append(
                ast_internal_classes.BinOp_Node(lval=ast_internal_classes.Name_Node(name=local_name.name),
                                                rval=litval,
                                                op="=",
                                                line_number=node.line_number))

        # This handles the case where the function is called with symbols
        for parameter, symbol in symbol_arguments:
            if parameter.name != symbol.name:
                self.local_not_transient_because_assign[my_name_sdfg].append(parameter.name)
                assigns.append(
                    ast_internal_classes.BinOp_Node(lval=ast_internal_classes.Name_Node(name=parameter.name),
                                                    rval=ast_internal_classes.Name_Node(name=symbol.name),
                                                    op="=",
                                                    line_number=node.line_number))

        # This handles the case where the function is called with variables starting with the case that the variable is local to the calling SDFG
        needs_replacement={}
        substate_sources = []
        substate_destinations = []
        for variable_in_call in variables_in_call:
            all_arrays = self.get_arrays_in_context(sdfg)
            
            sdfg_name = self.name_mapping.get(sdfg).get(ast_utils.get_name(variable_in_call))
            globalsdfg_name = self.name_mapping.get(self.globalsdfg).get(ast_utils.get_name(variable_in_call))
            matched = False
            for array_name, array in all_arrays.items():

                if array_name in [sdfg_name]:
                    matched = True
                    local_name = parameters[variables_in_call.index(variable_in_call)]
                    self.names_of_object_in_parent_sdfg[new_sdfg][local_name.name] = sdfg_name
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


                    if isinstance(variable_in_call, ast_internal_classes.Data_Ref_Node):
                        done=False
                        bonus_step=False
                        depth=0
                        tmpvar = variable_in_call
                        local_name = parameters[variables_in_call.index(variable_in_call)]
                        top_structure_name=self.name_mapping[sdfg][ast_utils.get_name(tmpvar.parent_ref)]
                        top_structure=sdfg.arrays[top_structure_name]
                        current_parent_structure=top_structure
                        current_parent_structure_name=top_structure_name
                        name_chain=[top_structure_name]    
                        while not done:
                            if isinstance(tmpvar.part_ref, ast_internal_classes.Data_Ref_Node):
                                
                                tmpvar=tmpvar.part_ref
                                depth+=1
                                current_member_name=ast_utils.get_name(tmpvar.parent_ref)
                                if isinstance(tmpvar.parent_ref, ast_internal_classes.Array_Subscript_Node):
                                    print("Array Subscript Node")
                                if bonus_step==True:
                                    print("Bonus Step")
                                current_member=current_parent_structure.members[current_member_name]
                                concatenated_name="_".join(name_chain)
                                local_shape=current_member.shape
                                new_shape=[]
                                local_indices=0
                                local_strides=list(current_member.strides)
                                local_offsets=list(current_member.offset)
                                local_index_list=[]
                                local_size =1
                                if isinstance(tmpvar.parent_ref, ast_internal_classes.Array_Subscript_Node):
                                    changed_indices = 0
                                    for i in tmpvar.parent_ref.indices:
                                        if isinstance(i, ast_internal_classes.ParDecl_Node):
                                            if i.type == "ALL":
                                                new_shape.append(local_shape[local_indices])
                                                local_size = local_size * local_shape[local_indices]
                                                local_index_list.append(None)
                                            else:
                                                raise NotImplementedError("Index in ParDecl should be ALL")
                                        else:
                                            
                                            text = ast_utils.ProcessedWriter(sdfg, self.name_mapping,placeholders=self.placeholders,placeholders_offsets=self.placeholders_offsets,rename_dict=self.replace_names).write_code(i)
                                            local_index_list.append(sym.pystr_to_symbolic(text))
                                            local_strides.pop(local_indices - changed_indices)
                                            local_offsets.pop(local_indices - changed_indices)
                                            changed_indices += 1
                                        local_indices = local_indices + 1
                                local_all_indices = [None] * (len(local_shape) - len(local_index_list)) + local_index_list
                                if self.normalize_offsets:
                                    subset = subs.Range([(i, i, 1) if i is not None else (0, s-1, 1)
                                                                for i, s in zip(local_all_indices, local_shape)])
                                else:
                                    subset = subs.Range([(i, i, 1) if i is not None else (1, s, 1)
                                                                for i, s in zip(local_all_indices, local_shape)])
                                smallsubset = subs.Range([(0, s - 1, 1) for s in new_shape])
                                bonus_step=False
                                if isinstance(current_member,dat.ContainerArray):
                                    if len(new_shape)==0:
                                        stype=current_member.stype
                                        view_to_container = dat.View.view(current_member)
                                        sdfg.arrays[concatenated_name+"_"+current_member_name+"_"+str(self.struct_view_count)] = view_to_container
                                        while isinstance(stype, dat.ContainerArray):
                                            stype=stype.stype
                                        bonus_step=True
                                        #sdfg.add_view(concatenated_name+"_"+current_member_name+"_"+str(self.struct_view_count),current_member.shape,current_member.dtype)
                                        view_to_member = dat.View.view(stype)
                                        sdfg.arrays[concatenated_name+"_"+current_member_name+"_m_"+str(self.struct_view_count)] = view_to_member
                                        #sdfg.add_view(concatenated_name+"_"+current_member_name+"_"+str(self.struct_view_count),current_member.stype.dtype)
                                else:    
                                    view_to_member = dat.View.view(current_member)
                                    sdfg.arrays[concatenated_name+"_"+current_member_name+"_"+str(self.struct_view_count)] = view_to_member
                                    #sdfg.add_view(concatenated_name+"_"+current_member_name+"_"+str(self.struct_view_count),current_member.shape,current_member.dtype,strides=current_member.strides,offset=current_member.offset)  
                                
                                already_there_1=False
                                already_there_2=False
                                already_there_22=False
                                already_there_3=False
                                already_there_33=False
                                already_there_4=False
                                re=None
                                wv=None
                                wr=None
                                rv=None
                                wv2=None
                                wr2=None
                                if current_parent_structure_name==top_structure_name:
                                    top_level=True
                                else:
                                    top_level=False
                                if local_name.name in read_names:
                                    
                                    re,already_there_1=find_access_in_sources(substate,substate_sources,current_parent_structure_name)
                                    wv,already_there_2=find_access_in_destinations(substate,substate_destinations,concatenated_name+"_"+current_member_name+"_"+str(self.struct_view_count))
                                                                        
                                    if not bonus_step:                                                                        
                                        mem=Memlet.simple(current_parent_structure_name + "." + current_member_name, subset)
                                        substate.add_edge(re,None,wv,"views",dpcp(mem))
                                    else:
                                        firstmem=Memlet.simple(current_parent_structure_name + "." + current_member_name,subs.Range.from_array(sdfg.arrays[concatenated_name+"_"+current_member_name+"_"+str(self.struct_view_count)]))
                                        wv2,already_there_22=find_access_in_destinations(substate,substate_destinations,concatenated_name+"_"+current_member_name+"_m_"+str(self.struct_view_count))
                                        mem     =Memlet.simple(concatenated_name+"_"+current_member_name+"_"+str(self.struct_view_count), subset)
                                        substate.add_edge(re,None,wv,"views",dpcp(firstmem))
                                        substate.add_edge(wv,None,wv2,"views",dpcp(mem))


                                if local_name.name in write_names:
                                    
                                    wr,already_there_3=find_access_in_destinations(substate,substate_destinations,current_parent_structure_name)
                                    rv,already_there_4=find_access_in_sources(substate,substate_sources,concatenated_name+"_"+current_member_name+"_"+str(self.struct_view_count))
                                    
                                    if not bonus_step: 
                                        mem2=Memlet.simple(current_parent_structure_name + "." + current_member_name, subset)
                                        substate.add_edge(rv,"views",wr,None,dpcp(mem2))
                                    else:
                                        firstmem=Memlet.simple(current_parent_structure_name + "." + current_member_name,subs.Range.from_array(sdfg.arrays[concatenated_name+"_"+current_member_name+"_"+str(self.struct_view_count)]))
                                        wr2,already_there_33=find_access_in_sources(substate,substate_sources,concatenated_name+"_"+current_member_name+"_m_"+str(self.struct_view_count))
                                        mem2     =Memlet.simple(concatenated_name+"_"+current_member_name+"_"+str(self.struct_view_count), subset)
                                        substate.add_edge(wr2,"views",rv,None,dpcp(mem2))    
                                        substate.add_edge(rv,"views",wr,None,dpcp(firstmem))
                                        
    
                                if not already_there_1:
                                    if re is not None:
                                        if not top_level:
                                            substate_sources.append(re)
                                        else:
                                            substate_destinations.append(re)
                                           
                                if not already_there_2:
                                    if wv is not None:
                                        substate_destinations.append(wv)
                                
                                if not already_there_3:
                                    if wr is not None:
                                        if not top_level:
                                            substate_destinations.append(wr)
                                        else:
                                            substate_sources.append(wr)
                                if not already_there_4:
                                    if rv is not None:
                                        substate_sources.append(rv)   

                                if bonus_step==True:
                                    if not already_there_22:
                                        if wv2 is not None:
                                            substate_destinations.append(wv2)     
                                    if not already_there_33:
                                        if wr2 is not None:
                                            substate_sources.append(wr2)           
                                    
                                if not bonus_step:
                                    current_parent_structure_name=concatenated_name+"_"+current_member_name+"_"+str(self.struct_view_count)
                                else:
                                    current_parent_structure_name=concatenated_name+"_"+current_member_name+"_m_"+str(self.struct_view_count)
                                current_parent_structure=current_parent_structure.members[current_member_name]
                                self.struct_view_count+=1
                                name_chain.append(current_member_name)
                            else:
                                done=True    
                                tmpvar=tmpvar.part_ref
                                concatenated_name="_".join(name_chain)
                                array_name=ast_utils.get_name(tmpvar)
                                member_name=ast_utils.get_name(tmpvar)
                                if bonus_step==True:
                                    print("Bonus Step")
                                    last_view_name=concatenated_name+"_m_"+str(self.struct_view_count-1)
                                else:    
                                    if depth>0:
                                        last_view_name=concatenated_name+"_"+str(self.struct_view_count-1)
                                    else:
                                        last_view_name=concatenated_name    
                                if isinstance(current_parent_structure,dat.ContainerArray):
                                    stype=current_parent_structure.stype
                                    while isinstance(stype, dat.ContainerArray):
                                        stype=stype.stype   
                                    
                                    array=stype.members[ast_utils.get_name(tmpvar)]
                                    
                                else:
                                    array=current_parent_structure.members[ast_utils.get_name(tmpvar)]    # FLAG
                                
                                if isinstance(array,dat.ContainerArray):
                                    view_to_member = dat.View.view(array)
                                    sdfg.arrays[concatenated_name+"_"+array_name+"_"+str(self.struct_view_count)] = view_to_member
                                    
                                else:    
                                    view_to_member = dat.View.view(array)
                                    sdfg.arrays[concatenated_name+"_"+array_name+"_"+str(self.struct_view_count)] = view_to_member
                                    
                                    #sdfg.add_view(concatenated_name+"_"+array_name+"_"+str(self.struct_view_count),array.shape,array.dtype,strides=array.strides,offset=array.offset)
                                last_view_name_read=None
                                re=None
                                wv=None
                                wr=None
                                rv=None
                                already_there_1=False
                                already_there_2=False
                                already_there_3=False
                                already_there_4=False
                                if current_parent_structure_name==top_structure_name:
                                    top_level=True
                                else:
                                    top_level=False
                                if local_name.name in read_names:
                                    for i in substate_destinations:
                                        if i.data==last_view_name:
                                            re=i
                                            already_there_1=True
                                            break
                                    if not already_there_1:
                                        re=substate.add_read(last_view_name)
                                    
                                    for i in substate_sources:
                                        if i.data==concatenated_name+"_"+array_name+"_"+str(self.struct_view_count):
                                            wv=i
                                            already_there_2=True
                                            break
                                    if not already_there_2:    
                                        wv = substate.add_write(concatenated_name+"_"+array_name+"_"+str(self.struct_view_count))
                                    
                                        
                                    mem=Memlet.from_array(last_view_name + "." + member_name,array )
                                    substate.add_edge(re,None,wv,"views",dpcp(mem))
                                    last_view_name_read=concatenated_name+"_"+array_name+"_"+str(self.struct_view_count)
                                last_view_name_write=None
                                if local_name.name in write_names:
                                    for i in substate_sources:
                                        if i.data==last_view_name:
                                            wr=i
                                            already_there_3=True
                                            break
                                    if not already_there_3:
                                        wr=substate.add_write(last_view_name)
                                    for i in substate_destinations:
                                        if i.data==concatenated_name+"_"+array_name+"_"+str(self.struct_view_count):
                                            rv=i
                                            already_there_4=True
                                            break   
                                    if not already_there_4:        
                                        rv = substate.add_read(concatenated_name+"_"+array_name+"_"+str(self.struct_view_count))
                                    

                                         
                                    mem2=Memlet.from_array(last_view_name + "." + member_name,array )
                                    substate.add_edge(rv,"views",wr,None,dpcp(mem2))
                                    last_view_name_write=concatenated_name+"_"+array_name+"_"+str(self.struct_view_count)
                                if not already_there_1:
                                    if re is not None:
                                        if not top_level:
                                            substate_sources.append(re)
                                        else:
                                            substate_destinations.append(re)
                                if not already_there_2:
                                    if wv is not None:
                                        substate_destinations.append(wv)
                                if not already_there_3:
                                    if wr is not None:
                                        if not top_level:
                                            substate_destinations.append(wr)
                                        else:
                                            substate_sources.append(wr)
                                if not already_there_4:
                                    if rv is not None:
                                        substate_sources.append(rv)   
                                mapped_name_overwrite=concatenated_name+"_"+array_name
                                self.views = self.views + 1
                                views.append([mapped_name_overwrite, wv, rv, variables_in_call.index(variable_in_call)])
                                    

                                if last_view_name_write is not None and last_view_name_read is not None:
                                    if last_view_name_read!=last_view_name_write:
                                        raise NotImplementedError("Read and write views should be the same")
                                    else:
                                        last_view_name=last_view_name_read
                                if last_view_name_read is not None and last_view_name_write is None:
                                    last_view_name=last_view_name_read
                                if last_view_name_write is not None and last_view_name_read is None:
                                    last_view_name=last_view_name_write    
                                mapped_name_overwrite=concatenated_name+"_"+array_name    
                                strides = list(array.strides)
                                offsets = list(array.offset)
                                self.struct_view_count+=1
                                
                                if isinstance(array,dat.ContainerArray) and isinstance(tmpvar,ast_internal_classes.Array_Subscript_Node):
                                    current_member_name=ast_utils.get_name(tmpvar)
                                    current_member=current_parent_structure.members[current_member_name]
                                    concatenated_name="_".join(name_chain)
                                    local_shape=current_member.shape
                                    new_shape=[]
                                    local_indices=0
                                    local_strides=list(current_member.strides)
                                    local_offsets=list(current_member.offset)
                                    local_index_list=[]
                                    local_size =1
                                    changed_indices = 0
                                    for i in tmpvar.indices:
                                        if isinstance(i, ast_internal_classes.ParDecl_Node):
                                            if i.type == "ALL":
                                                new_shape.append(local_shape[local_indices])
                                                local_size = local_size * local_shape[local_indices]
                                                local_index_list.append(None)
                                            else:
                                                raise NotImplementedError("Index in ParDecl should be ALL")
                                        else:
                                            text = ast_utils.ProcessedWriter(sdfg, self.name_mapping,placeholders=self.placeholders,placeholders_offsets=self.placeholders_offsets,rename_dict=self.replace_names).write_code(i)
                                            local_index_list.append(sym.pystr_to_symbolic(text))
                                            local_strides.pop(local_indices - changed_indices)
                                            local_offsets.pop(local_indices - changed_indices)
                                            changed_indices += 1
                                        local_indices = local_indices + 1
                                    local_all_indices = [None] * (len(local_shape) - len(local_index_list)) + local_index_list
                                    if self.normalize_offsets:
                                        subset = subs.Range([(i, i, 1) if i is not None else (0, s-1, 1)
                                                                    for i, s in zip(local_all_indices, local_shape)])
                                    else:
                                        subset = subs.Range([(i, i, 1) if i is not None else (1, s, 1)
                                                                    for i, s in zip(local_all_indices, local_shape)])
                                    smallsubset = subs.Range([(0, s - 1, 1) for s in new_shape])
                                    if isinstance(current_member,dat.ContainerArray):
                                        if len(new_shape)==0:
                                            stype=current_member.stype
                                            while isinstance(stype, dat.ContainerArray):
                                                stype=stype.stype
                                            bonus_step=True
                                            #sdfg.add_view(concatenated_name+"_"+current_member_name+"_"+str(self.struct_view_count),current_member.shape,current_member.dtype)
                                            view_to_member = dat.View.view(stype)
                                            sdfg.arrays[concatenated_name+"_"+current_member_name+"_"+str(self.struct_view_count)] = view_to_member
                                            #sdfg.add_view(concatenated_name+"_"+current_member_name+"_"+str(self.struct_view_count),current_member.stype.dtype)
                                        else:    
                                            view_to_member = dat.View.view(current_member)
                                            sdfg.arrays[concatenated_name+"_"+current_member_name+"_"+str(self.struct_view_count)] = view_to_member
                                    
                                            #sdfg.add_view(concatenated_name+"_"+current_member_name+"_"+str(self.struct_view_count),current_member.shape,current_member.dtype,strides=current_member.strides,offset=current_member.offset)
                                    already_there_1=False
                                    already_there_2=False
                                    already_there_3=False
                                    already_there_4=False  
                                    re=None
                                    wv=None
                                    wr=None
                                    rv=None      
                                    if current_parent_structure_name==top_structure_name:
                                        top_level=True
                                    else:
                                        top_level=False
                                    if local_name.name in read_names:
                                        for i in substate_destinations:
                                            if i.data==last_view_name:
                                                re=i
                                                already_there_1=True
                                                break
                                        if not already_there_1:
                                            re=substate.add_read(last_view_name)  
                                        
                                        for i in substate_sources:
                                            if i.data==concatenated_name+"_"+current_member_name+"_"+str(self.struct_view_count):
                                                wv=i
                                                already_there_2=True
                                                break
                                        if not already_there_2:
                                            wv = substate.add_write(concatenated_name+"_"+current_member_name+"_"+str(self.struct_view_count))
                                         
                                            
                                        if isinstance(current_member,dat.ContainerArray):
                                            mem=Memlet.simple(last_view_name, subset)
                                        else:
                                            mem=Memlet.simple(current_parent_structure_name + "." + current_member_name, subset)    
                                        substate.add_edge(re,None,wv,"views",dpcp(mem))

                                    if local_name.name in write_names:
                                        for i in substate_sources:
                                            if i.data==last_view_name:
                                                wr=i
                                                already_there_3=True
                                                break
                                        if not already_there_3:
                                            wr=substate.add_write(last_view_name)
                                        
                                        for i in substate_destinations:
                                            if i.data==concatenated_name+"_"+current_member_name+"_"+str(self.struct_view_count):
                                                rv=i
                                                already_there_4=True
                                                break   
                                        if not already_there_4:    
                                            rv = substate.add_read(concatenated_name+"_"+current_member_name+"_"+str(self.struct_view_count))
                                   
                                          
                                        if isinstance(current_member,dat.ContainerArray):
                                            mem2=Memlet.simple(last_view_name, subset)
                                        else:
                                            mem2=Memlet.simple(current_parent_structure_name + "." + current_member_name, subset)    
                                        
                                        substate.add_edge(rv,"views",wr,None,dpcp(mem2))
                                    if not already_there_1:
                                        if re is not None:
                                            if not top_level:
                                                substate_sources.append(re)
                                            else:
                                                substate_destinations.append(re)
                                    if not already_there_2:
                                        if wv is not None:
                                            substate_destinations.append(wv)
                                    if not already_there_3:
                                        if wr is not None:
                                            if not top_level:
                                                substate_destinations.append(wr)
                                            else:
                                                substate_sources.append(wr)
                                    if not already_there_4:
                                        if rv is not None:
                                            substate_sources.append(rv)         
                                    last_view_name=concatenated_name+"_"+current_member_name+"_"+str(self.struct_view_count)
                                    if not isinstance(current_member,dat.ContainerArray):
                                        mapped_name_overwrite=concatenated_name+"_"+current_member_name   
                                        needs_replacement[mapped_name_overwrite]=last_view_name
                                    else:
                                        mapped_name_overwrite=concatenated_name+"_"+current_member_name   
                                        needs_replacement[mapped_name_overwrite]=last_view_name
                                        mapped_name_overwrite=concatenated_name+"_"+current_member_name+"_"+str(self.struct_view_count)
                                    self.views = self.views + 1
                                    views.append([mapped_name_overwrite, wv, rv, variables_in_call.index(variable_in_call)])
                                    
                                    strides = list(view_to_member.strides)
                                    offsets = list(view_to_member.offset)
                                    self.struct_view_count+=1
                                    


                                
                               



                     
                        if isinstance(tmpvar,ast_internal_classes.Array_Subscript_Node):
                            
                            changed_indices = 0
                            for i in tmpvar.indices:
                                if isinstance(i, ast_internal_classes.ParDecl_Node):
                                    if i.type == "ALL":
                                        shape.append(array.shape[indices])
                                        mysize = mysize * array.shape[indices]
                                        index_list.append(None)
                                    else:
                                        raise NotImplementedError("Index in ParDecl should be ALL")
                                else:
                                    text = ast_utils.ProcessedWriter(sdfg, self.name_mapping,placeholders=self.placeholders,placeholders_offsets=self.placeholders_offsets,rename_dict=self.replace_names).write_code(i)
                                    index_list.append(sym.pystr_to_symbolic(text))
                                    strides.pop(indices - changed_indices)
                                    offsets.pop(indices - changed_indices)
                                    changed_indices += 1
                                indices = indices + 1
                        

                           
                        elif isinstance(tmpvar,ast_internal_classes.Name_Node):
                            shape = list(array.shape)
                        else:
                            raise NotImplementedError("Unknown part_ref type")
                        
                        if shape == () or shape == (1, ) or shape == [] or shape == [1]:
                            #FIXME 6.03.2024
                            #print(array,array.__class__.__name__)
                            if isinstance(array,dat.ContainerArray):
                                if isinstance(array.stype, dat.ContainerArray):
                                    if isinstance(array.stype.stype,dat.Structure):
                                        element_type=array.stype.stype
                                    else:
                                        element_type=array.stype.stype.dtype 

                                elif isinstance(array.stype,dat.Structure):
                                    element_type=array.stype
                                else:
                                    element_type=array.stype.dtype    
                                #print(element_type,element_type.__class__.__name__)
                                #print(array.base_type,array.base_type.__class__.__name__)
                            elif isinstance(array,dat.Structure):
                                element_type=array
                            elif isinstance(array,pointer):
                                if hasattr(array,"stype"):
                                    if hasattr(array.stype,"free_symbols"):
                                        element_type=array.stype
                                        #print("get stype")
                                
                            else:
                                if hasattr(array,"dtype"):
                                    if hasattr(array.dtype,"free_symbols"):
                                        element_type=array.dtype
                                        #print("get dtype")
                                
                            if isinstance(element_type,pointer):
                                #print("pointer-ized")
                                found=False
                                if hasattr(element_type,"dtype"):
                                    if hasattr(element_type.dtype,"free_symbols"):
                                        element_type=element_type.dtype
                                        found=True
                                        #print("get dtype")
                                if hasattr(element_type,"stype"):
                                    if hasattr(element_type.stype,"free_symbols"):
                                        element_type=element_type.stype
                                        found=True
                                        #print("get stype")
                                if hasattr(element_type,"base_type"):        
                                    if hasattr(element_type.base_type,"free_symbols"):
                                        element_type=element_type.base_type
                                        found=True
                                        #print("get base_type")        
                                #if not found:
                                #    print(dir(element_type))
                            #print("array info: "+str(array),array.__class__.__name__)
                            #print(element_type,element_type.__class__.__name__)    
                            if element_type.name in self.registered_types:
                                datatype=self.get_dace_type(str(element_type))
                                datatype_to_add=copy.deepcopy(element_type)
                                datatype_to_add.transient = False
                                #print(datatype_to_add,datatype_to_add.__class__.__name__)
                                new_sdfg.add_datadesc(self.name_mapping[new_sdfg][local_name.name], datatype_to_add)
                                
                                if self.struct_views.get(new_sdfg) is None:
                                    self.struct_views[new_sdfg] = {}
                                
                                add_views_recursive(new_sdfg,local_name.name,datatype_to_add,self.struct_views[new_sdfg],self.name_mapping[new_sdfg],self.registered_types,[],self.actual_offsets_per_sdfg[new_sdfg],self.names_of_object_in_parent_sdfg[new_sdfg],self.actual_offsets_per_sdfg[sdfg])    
                                
                            else:
                                new_sdfg.add_scalar(self.name_mapping[new_sdfg][local_name.name], array.dtype, array.storage)            
                        else:
                            element_type=array.dtype.base_type    
                            if element_type in self.registered_types:
                                raise NotImplementedError("Nested derived types not implemented")
                                datatype_to_add=copy.deepcopy(element_type)
                                datatype_to_add.transient = False
                                new_sdfg.add_datadesc(self.name_mapping[new_sdfg][local_name.name], datatype_to_add)
                                #arr_dtype = datatype[sizes]
                                #arr_dtype.offset = [offset_value for _ in sizes]
                                #sdfg.add_datadesc(self.name_mapping[sdfg][node.name], arr_dtype)    
                            else:
                                
                                new_sdfg.add_array(self.name_mapping[new_sdfg][local_name.name],
                                            shape,
                                            array.dtype,
                                            array.storage,
                                            strides=strides,
                                            offset=offsets)   
                    else:
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
                                    text = ast_utils.ProcessedWriter(sdfg, self.name_mapping,placeholders=self.placeholders,placeholders_offsets=self.placeholders_offsets,rename_dict=self.replace_names).write_code(i)
                                    index_list.append(sym.pystr_to_symbolic(text))
                                    strides.pop(indices - changed_indices)
                                    offsets.pop(indices - changed_indices)
                                    changed_indices += 1
                                indices = indices + 1
                           
                                

                        if isinstance(variable_in_call, ast_internal_classes.Name_Node):
                            shape = list(array.shape)
                        
                        #print("Data_Ref_Node")    
                    # Functionally, this identifies the case where the array is in fact a scalar
                        if shape == () or shape == (1, ) or shape == [] or shape == [1]:
                            if hasattr(array,"name") and array.name in self.registered_types:
                                datatype=self.get_dace_type(array.name)
                                datatype_to_add=copy.deepcopy(array)
                                datatype_to_add.transient = False
                                new_sdfg.add_datadesc(self.name_mapping[new_sdfg][local_name.name], datatype_to_add)
                                
                                if self.struct_views.get(new_sdfg) is None:
                                    self.struct_views[new_sdfg] = {}
                                add_views_recursive(new_sdfg,local_name.name,datatype_to_add,self.struct_views[new_sdfg],self.name_mapping[new_sdfg],self.registered_types,[],self.actual_offsets_per_sdfg[new_sdfg],self.names_of_object_in_parent_sdfg[new_sdfg],self.actual_offsets_per_sdfg[sdfg])        
                                
                            else:
                                new_sdfg.add_scalar(self.name_mapping[new_sdfg][local_name.name], array.dtype, array.storage)
                        else:
                            # This is the case where the array is not a scalar and we need to create a view
                            if not (shape == () or shape == (1, ) or shape == [] or shape == [1]):
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
                                if self.normalize_offsets:
                                    subset = subsets.Range([(i, i, 1) if i is not None else (0, s-1, 1)
                                                            for i, s in zip(all_indices, array.shape)])
                                else:
                                    subset = subsets.Range([(i, i, 1) if i is not None else (1, s, 1)
                                                            for i, s in zip(all_indices, array.shape)])
                                smallsubset = subsets.Range([(0, s - 1, 1) for s in shape])

                                #memlet = Memlet(f'{array_name}[{subset}]->{smallsubset}')
                                #memlet2 = Memlet(f'{viewname}[{smallsubset}]->{subset}')
                                memlet = Memlet(f'{array_name}[{subset}]')
                                memlet2 = Memlet(f'{array_name}[{subset}]')
                                wv = None
                                rv = None
                                if local_name.name in read_names:
                                    r = substate.add_read(array_name)
                                    wv = substate.add_write(viewname)
                                    substate.add_edge(r, None, wv, 'views', dpcp(memlet))
                                if local_name.name in write_names:
                                    rv = substate.add_read(viewname)
                                    w = substate.add_write(array_name)
                                    substate.add_edge(rv, 'views', w, None, dpcp(memlet2))

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
        names_list=[]
        if node.specification_part is not None:
            if node.specification_part.specifications is not None:
                namefinder=ast_transforms.FindDefinedNames()
                for i in node.specification_part.specifications:
                    namefinder.visit(i)   
                names_list=namefinder.names
        # This handles the case where the function is called with read variables found in a module
        for i in not_found_read_names:
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
                    elif (hasattr(array_in_global, 'type') and array_in_global.type == "Array") or isinstance(array_in_global, dat.Array):
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
                    elif (hasattr(array_in_global, 'type') and array_in_global.type == "Array") or isinstance(array_in_global, dat.Array):
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

                    array = sdfg.arrays[self.name_mapping[sdfg][i]]
                    if isinstance(array_in_global, Scalar):
                        new_sdfg.add_scalar(self.name_mapping[new_sdfg][i], array_in_global.dtype, transient=False)
                    elif (hasattr(array_in_global, 'type') and array_in_global.type == "Array") or isinstance(array_in_global, dat.Array):
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
                    elif (hasattr(array_in_global, 'type') and array_in_global.type == "Array") or isinstance(array_in_global, dat.Array):
                        new_sdfg.add_array(self.name_mapping[new_sdfg][i],
                                           array_in_global.shape,
                                           array_in_global.dtype,
                                           array_in_global.storage,
                                           transient=False,
                                           strides=array_in_global.strides,
                                           offset=array_in_global.offset)
        if self.multiple_sdfgs==False:
            #print("Adding nested sdfg", new_sdfg.name, "to", sdfg.name)
            #print(sym_dict)
            internal_sdfg = substate.add_nested_sdfg(new_sdfg,
                                                 sdfg,
                                                 ins_in_new_sdfg,
                                                 outs_in_new_sdfg,
                                                 symbol_mapping=sym_dict)
        else:
            internal_sdfg = substate.add_nested_sdfg(None,
                                                 sdfg,
                                                 ins_in_new_sdfg,
                                                 outs_in_new_sdfg,
                                                 symbol_mapping=sym_dict,
                                                 name="External_nested_" + new_sdfg.name)    
        #if self.multiple_sdfgs==False:
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
                if needs_replacement.get(mapped_name) is not None:
                        mapped_name = needs_replacement[mapped_name]
                        var=sdfg.arrays[mapped_name]
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

                    if local_name.name in write_names:
                        memlet = subs.Range([(0, s - 1, 1) for s in sdfg.arrays[elem[2].label].shape])
                        substate.add_memlet_path(internal_sdfg,
                                                elem[2],
                                                src_conn=self.name_mapping[new_sdfg][local_name.name],
                                                memlet=Memlet(expr=elem[2].label, subset=memlet))
                    if local_name.name in read_names:
                        memlet = subs.Range([(0, s - 1, 1) for s in sdfg.arrays[elem[1].label].shape])
                        substate.add_memlet_path(elem[1],
                                                internal_sdfg,
                                                dst_conn=self.name_mapping[new_sdfg][local_name.name],
                                                memlet=Memlet(expr=elem[1].label, subset=memlet))
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
            local_name=ast_internal_classes.Name_Node(name=i)
            memlet = ast_utils.generate_memlet(ast_internal_classes.Name_Node(name=i), sdfg, self, self.normalize_offsets)
            if local_name.name in write_names:
                ast_utils.add_memlet_write(substate, self.name_mapping[sdfg][i], internal_sdfg,
                                        self.name_mapping[new_sdfg][i], memlet)
            if local_name.name in read_names:
                ast_utils.add_memlet_read(substate, self.name_mapping[sdfg][i], internal_sdfg,
                                        self.name_mapping[new_sdfg][i], memlet)
        for i in globalmemlets:
            local_name=ast_internal_classes.Name_Node(name=i)
            found=False
            parent_sdfg=sdfg
            nested_sdfg=new_sdfg
            first=True
            while not found and parent_sdfg is not None:
                if self.name_mapping.get(parent_sdfg).get(i) is not None:
                    found=True
                else:
                    self.name_mapping[parent_sdfg][i] = parent_sdfg._find_new_name(i)
                    self.all_array_names.append(self.name_mapping[parent_sdfg][i])
                    array_in_global = self.globalsdfg.arrays[self.name_mapping[self.globalsdfg][i]]
                    if isinstance(array_in_global, Scalar):
                        parent_sdfg.add_scalar(self.name_mapping[parent_sdfg][i], array_in_global.dtype, transient=False)
                    elif (hasattr(array_in_global, 'type') and array_in_global.type == "Array") or isinstance(array_in_global, dat.Array):
                        parent_sdfg.add_array(self.name_mapping[parent_sdfg][i],
                                           array_in_global.shape,
                                           array_in_global.dtype,
                                           array_in_global.storage,
                                           transient=False,
                                           strides=array_in_global.strides,
                                           offset=array_in_global.offset)

                if first:
                    first=False
                else:
                    if local_name.name in write_names:
                        nested_sdfg.parent_nsdfg_node.add_out_connector(self.name_mapping[parent_sdfg][i], force=True)
                    if local_name.name in read_names:
                        nested_sdfg.parent_nsdfg_node.add_in_connector(self.name_mapping[parent_sdfg][i], force=True)

                memlet = ast_utils.generate_memlet(ast_internal_classes.Name_Node(name=i), parent_sdfg, self, self.normalize_offsets)
                if local_name.name in write_names:
                    ast_utils.add_memlet_write(nested_sdfg.parent, self.name_mapping[parent_sdfg][i], nested_sdfg.parent_nsdfg_node,
                                            self.name_mapping[nested_sdfg][i], memlet)
                if local_name.name in read_names:
                    ast_utils.add_memlet_read(nested_sdfg.parent, self.name_mapping[parent_sdfg][i], nested_sdfg.parent_nsdfg_node,
                                            self.name_mapping[nested_sdfg][i], memlet)
                if not found:
                    nested_sdfg=parent_sdfg    
                    parent_sdfg=parent_sdfg.parent_sdfg
                    
        
        if self.multiple_sdfgs==False:
            if node.execution_part is not None:
                if node.specification_part is not None and node.specification_part.uses is not None:
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
                        
                    old_mode=self.transient_mode
                    print("For ",sdfg_name," old mode is ",old_mode)
                    self.transient_mode=True
                    for j in node.specification_part.specifications:
                        
                    
                        self.declstmt2sdfg(j, new_sdfg, new_sdfg)
                    self.transient_mode=old_mode
                   
    
                for i in assigns:
                        self.translate(i, new_sdfg, new_sdfg)
                self.translate(node.execution_part, new_sdfg, new_sdfg)

        if self.multiple_sdfgs==True:
            internal_sdfg.path=self.sdfg_path+ new_sdfg.name + ".sdfg"
            #new_sdfg.save(path.join(self.sdfg_path, new_sdfg.name + ".sdfg"))        


    def binop2sdfg(self, node: ast_internal_classes.BinOp_Node, sdfg: SDFG, cfg: ControlFlowRegion):
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
            from dace.frontend.fortran.intrinsics import FortranIntrinsics
            if augmented_call.name.name not in ["pow", "atan2", "tanh", "__dace_epsilon", *FortranIntrinsics.retained_function_names()]:
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

        substate = ast_utils.add_simple_state_to_sdfg(
            self, cfg, "_state_l" + str(node.line_number[0]) + "_c" + str(node.line_number[1]))

        output_names_changed = [o_t + "_out" for o_t in output_names]

        tasklet = ast_utils.add_tasklet(substate, "_l" + str(node.line_number[0]) + "_c" + str(node.line_number[1]),
                                        input_names_tasklet, output_names_changed, "text", node.line_number,
                                        self.file_name)

        for i, j in zip(input_names, input_names_tasklet):
            memlet_range = self.get_memlet_range(sdfg, input_vars, i, j)
            src= ast_utils.add_memlet_read(substate, i, tasklet, j, memlet_range)
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
                                     input_names_tasklet,placeholders=self.placeholders,placeholders_offsets=self.placeholders_offsets,rename_dict=self.replace_names)

        text = tw.write_code(node)
        #print(sdfg.name,node.line_number,output_names,output_names_changed,input_names,input_names_tasklet)
        tasklet.code = CodeBlock(text, lang.Python)

    def call2sdfg(self, node: ast_internal_classes.Call_Expr_Node, sdfg: SDFG, cfg: ControlFlowRegion):
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
                                         self.name_mapping,placeholders=self.placeholders,placeholders_offsets=self.placeholders_offsets,rename_dict=self.replace_names)
            if not isinstance(rettype, ast_internal_classes.Void) and hasret:
                special_list_in[retval.name] = pointer(self.get_dace_type(rettype))
                special_list_out.append(retval.name + "_out")
                text = tw.write_code(
                    ast_internal_classes.BinOp_Node(lval=retval, op="=", rval=node, line_number=node.line_number))

            else:
                text = tw.write_code(node)
            substate = ast_utils.add_simple_state_to_sdfg(self, cfg, "_state" + str(node.line_number[0]))

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

    def declstmt2sdfg(self, node: ast_internal_classes.Decl_Stmt_Node, sdfg: SDFG, cfg: ControlFlowRegion):
        """
        This function translates a variable declaration statement to an access node on the sdfg
        :param node: The node to translate
        :param sdfg: The sdfg to attach the access node to
        :note This function is the top level of the declaration, most implementation is in vardecl2sdfg
        """
        for i in node.vardecl:
            self.translate(i, sdfg, cfg)

    def vardecl2sdfg(self, node: ast_internal_classes.Var_Decl_Node, sdfg: SDFG, cfg: ControlFlowRegion):
        """
        This function translates a variable declaration to an access node on the sdfg
        :param node: The node to translate
        :param sdfg: The sdfg to attach the access node to

        """
        if node.name=="modname": return

        #if the sdfg is the toplevel-sdfg, the variable is a global variable
        is_arg = False
        if isinstance(node.parent,ast_internal_classes.Subroutine_Subprogram_Node) or isinstance(node.parent,ast_internal_classes.Function_Subprogram_Node):
            if hasattr(node.parent,"args"):
                for i in node.parent.args:
                    name =ast_utils.get_name(i)
                    if name == node.name:
                        is_arg = True
                        if self.local_not_transient_because_assign.get(sdfg.name) is not None:
                            if name in self.local_not_transient_because_assign[sdfg.name]:
                                is_arg=False
                        break
        if is_arg:
            transient=False
        else:
            transient = self.transient_mode
        # find the type
        datatype = self.get_dace_type(node.type)
        #if hasattr(node, "alloc"):
        #    if node.alloc:
        #        self.unallocated_arrays.append([node.name, datatype, sdfg, transient])
        #        return
        # get the dimensions
        #print(node.name)
        if node.sizes is not None:
            sizes = []
            offset = []
            actual_offsets = []
            offset_value = 0 if self.normalize_offsets else -1
            for i in node.sizes:
                stuff=[ii for ii in ast_transforms.mywalk(i) if isinstance(ii, ast_internal_classes.Data_Ref_Node)]
                if len(stuff)>0:
                    count=self.count_of_struct_symbols_lifted
                    sdfg.add_symbol("tmp_struct_symbol_"+str(count),dtypes.int32)
                    symname="tmp_struct_symbol_"+str(count)
                    if sdfg.parent_sdfg is not None:
                        sdfg.parent_sdfg.add_symbol("tmp_struct_symbol_"+str(count),dtypes.int32)
                        sdfg.parent_nsdfg_node.symbol_mapping["tmp_struct_symbol_"+str(count)]="tmp_struct_symbol_"+str(count)
                        for edge in sdfg.parent.parent_graph.in_edges(sdfg.parent):
                            assign= ast_utils.ProcessedWriter(sdfg.parent_sdfg, self.name_mapping,placeholders=self.placeholders,placeholders_offsets=self.placeholders_offsets,rename_dict=self.replace_names).write_code(i)
                            edge.data.assignments["tmp_struct_symbol_"+str(count)]=assign
                            #print(edge)
                    else:
                        assign= ast_utils.ProcessedWriter(sdfg, self.name_mapping,placeholders=self.placeholders,placeholders_offsets=self.placeholders_offsets,rename_dict=self.replace_names).write_code(i)
                        
                        sdfg.append_global_code(f"{dtypes.int32.ctype} {symname};\n")
                        sdfg.append_init_code("tmp_struct_symbol_"+str(count)+"="+assign.replace(".","->")+";\n")       
                    tw = ast_utils.TaskletWriter([], [], sdfg, self.name_mapping,placeholders=self.placeholders,placeholders_offsets=self.placeholders_offsets,rename_dict=self.replace_names)
                    text = tw.write_code(ast_internal_classes.Name_Node(name="tmp_struct_symbol_"+str(count),type="INTEGER",line_number=node.line_number))
                    sizes.append(sym.pystr_to_symbolic(text))
                    actual_offset_value=node.offsets[node.sizes.index(i)]
                    if isinstance(actual_offset_value,ast_internal_classes.Array_Subscript_Node):
                        print(node.name,actual_offset_value.name.name)
                        raise NotImplementedError("Array subscript in offset not implemented")
                    if isinstance(actual_offset_value,int):
                        actual_offset_value=ast_internal_classes.Int_Literal_Node(value=str(actual_offset_value))
                    aotext=tw.write_code(actual_offset_value)    
                    actual_offsets.append(str(sym.pystr_to_symbolic(aotext)))
                    
                    self.actual_offsets_per_sdfg[sdfg][node.name]=actual_offsets
                    #otext = tw.write_code(offset_value)

                    #TODO: shouldn't this use node.offset??
                    offset.append(offset_value)
                    self.count_of_struct_symbols_lifted+=1
                else:
                    tw = ast_utils.TaskletWriter([], [], sdfg, self.name_mapping,placeholders=self.placeholders,placeholders_offsets=self.placeholders_offsets,rename_dict=self.replace_names)
                    text = tw.write_code(i)
                    actual_offset_value=node.offsets[node.sizes.index(i)]
                    if isinstance(actual_offset_value,int):
                        actual_offset_value=ast_internal_classes.Int_Literal_Node(value=str(actual_offset_value))
                    aotext=tw.write_code(actual_offset_value)    
                    actual_offsets.append(str(sym.pystr_to_symbolic(aotext)))
                    #otext = tw.write_code(offset_value)
                    sizes.append(sym.pystr_to_symbolic(text))
                    offset.append(offset_value)
                    self.actual_offsets_per_sdfg[sdfg][node.name]=actual_offsets

        else:
            sizes = None
        # create and check name - if variable is already defined (function argument and defined in declaration part) simply stop
        if self.name_mapping[sdfg].get(node.name) is not None:
            #here we must replace local placeholder sizes that have already made it to tasklets via size and ubound calls
            if sizes is not None:
                actual_sizes=sdfg.arrays[self.name_mapping[sdfg][node.name]].shape
                print(node.name,sdfg.name,self.names_of_object_in_parent_sdfg.get(sdfg).get(node.name))
                print(sdfg.parent_sdfg.name,self.actual_offsets_per_sdfg[sdfg.parent_sdfg].get(self.names_of_object_in_parent_sdfg[sdfg][node.name]))
                print(sdfg.parent_sdfg.arrays.get(self.name_mapping[sdfg.parent_sdfg].get(self.names_of_object_in_parent_sdfg.get(sdfg).get(node.name))))
                if self.actual_offsets_per_sdfg[sdfg.parent_sdfg].get(self.names_of_object_in_parent_sdfg[sdfg][node.name]) is not None:
                    actual_offsets=self.actual_offsets_per_sdfg[sdfg.parent_sdfg][self.names_of_object_in_parent_sdfg[sdfg][node.name]]
                else:
                    actual_offsets=[1] * len(actual_sizes)   

                index=0
                for i in node.sizes:
                    if isinstance(i,ast_internal_classes.Name_Node):
                        if i.name.startswith("__f2dace_A"):
                            self.replace_names[i.name]=str(actual_sizes[index])
                            #node.parent.execution_part=ast_transforms.RenameVar(i.name,str(actual_sizes[index])).visit(node.parent.execution_part)
                    index+=1
                index=0
                for i in node.offsets:
                    if isinstance(i,ast_internal_classes.Name_Node):
                        if i.name.startswith("__f2dace_OA"):
                            self.replace_names[i.name]=str(actual_offsets[index])
                            #node.parent.execution_part=ast_transforms.RenameVar(i.name,str(actual_offsets[index])).visit(node.parent.execution_part)
                    index+=1        
            elif sizes is None:
                if isinstance(datatype, Structure):
                    datatype_to_add=copy.deepcopy(datatype)
                    datatype_to_add.transient = transient
                    #if node.name=="p_nh":
                        #print("Adding local struct",self.name_mapping[sdfg][node.name],datatype_to_add)
                    if self.struct_views.get(sdfg) is None:
                        self.struct_views[sdfg] = {}
                    add_views_recursive(sdfg,node.name,datatype_to_add,self.struct_views[sdfg],self.name_mapping[sdfg],self.registered_types,[],self.actual_offsets_per_sdfg[sdfg],self.names_of_object_in_parent_sdfg[sdfg],self.actual_offsets_per_sdfg[sdfg.parent_sdfg])        
                

            return

        if node.name in sdfg.symbols:
            return

        self.name_mapping[sdfg][node.name] = sdfg._find_new_name(node.name)

        if sizes is None:
            if isinstance(datatype, Structure):
                datatype_to_add=copy.deepcopy(datatype)
                datatype_to_add.transient = transient
                #if node.name=="p_nh":
                    #print("Adding local struct",self.name_mapping[sdfg][node.name],datatype_to_add)
                sdfg.add_datadesc(self.name_mapping[sdfg][node.name], datatype_to_add)
                if self.struct_views.get(sdfg) is None:
                    self.struct_views[sdfg] = {}
                add_views_recursive(sdfg,node.name,datatype_to_add,self.struct_views[sdfg],self.name_mapping[sdfg],self.registered_types,[],self.actual_offsets_per_sdfg[sdfg],{},{})        
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
                container=dat.ContainerArray(stype=datatype,shape=sizes,offset=offset,transient=transient)
                #print("Adding local container array",self.name_mapping[sdfg][node.name],sizes,datatype,offset,strides,transient)
                sdfg.arrays[self.name_mapping[sdfg][node.name]]=container
                #sdfg.add_datadesc(self.name_mapping[sdfg][node.name], arr_dtype)
                
            else:    
                #print("Adding local array",self.name_mapping[sdfg][node.name],sizes,datatype,offset,strides,transient)
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

        if hasattr(node,"init") and node.init is not None:
            self.translate(ast_internal_classes.BinOp_Node(lval=ast_internal_classes.Name_Node(name=node.name,type=node.type),op="=",rval=node.init,line_number=node.line_number), sdfg)    

    def break2sdfg(self, node: ast_internal_classes.Break_Node, sdfg: SDFG, cfg: ControlFlowRegion):

        self.last_loop_breaks[cfg] = self.last_sdfg_states[cfg]
        cfg.add_edge(self.last_sdfg_states[cfg], self.last_loop_continues.get(cfg), InterstateEdge())

    def continue2sdfg(self, node: ast_internal_classes.Continue_Node, sdfg: SDFG, cfg: ControlFlowRegion):
        cfg.add_edge(self.last_sdfg_states[cfg], self.last_loop_continues.get(cfg), InterstateEdge())  
        self.already_has_edge_back_continue[cfg] = self.last_sdfg_states[cfg]

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
    own_ast = ast_components.InternalFortranAst(ast, tables)
    program = own_ast.create_ast(ast)

    structs_lister=ast_transforms.StructLister()
    structs_lister.visit(program)
    struct_dep_graph=nx.DiGraph()
    for i,name in zip(structs_lister.structs,structs_lister.names):
        if name not in struct_dep_graph.nodes:
            struct_dep_graph.add_node(name)
        struct_deps_finder=ast_transforms.StructDependencyLister(structs_lister.names)
        struct_deps_finder.visit(i)
        struct_deps=struct_deps_finder.structs_used
        for j,pointing,point_name in zip(struct_deps,struct_deps_finder.is_pointer,struct_deps_finder.pointer_names):
            if j not in struct_dep_graph.nodes:
                struct_dep_graph.add_node(j)
            struct_dep_graph.add_edge(name,j,pointing=pointing,point_name=point_name)

    program.structures = ast_transforms.Structures(structs_lister.structs)

    functions_and_subroutines_builder = ast_transforms.FindFunctionAndSubroutines()
    functions_and_subroutines_builder.visit(program)
    functions_and_subroutines = functions_and_subroutines_builder.names

    if transform:
        program = ast_transforms.functionStatementEliminator(program)
        program = ast_transforms.CallToArray(functions_and_subroutines_builder).visit(program)
        program = ast_transforms.CallExtractor().visit(program)
        program = ast_transforms.SignToIf().visit(program)
        program = ast_transforms.ArrayToLoop(program).visit(program)

        for transformation in own_ast.fortran_intrinsics().transformations():
            transformation.initialize(program)
            program = transformation.visit(program)

        program = ast_transforms.ForDeclarer().visit(program)
        program = ast_transforms.IndexExtractor(program, normalize_offsets).visit(program)

        program = ast_transforms.optionalArgsExpander(program)

    return (program, own_ast)

def create_sdfg_from_string(
    source_string: str,
    sdfg_name: str,
    normalize_offsets: bool = False,
    multiple_sdfgs: bool = False,
    use_experimental_cfg_blocks: bool = False,
    sources: List[str] = None
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

    exclude_list = []
    missing_modules = []
    dep_graph = nx.DiGraph()
    asts = {}
    actually_used_in_module={}
    interface_blocks = {}
    ast = recursive_ast_improver(ast,
                                 sources,
                                 [],
                                 parser,
                                 interface_blocks,
                                 exclude_list=exclude_list,
                                 missing_modules=missing_modules,
                                 dep_graph=dep_graph,
                                 asts=asts)

    for mod, blocks in interface_blocks.items():

        # get incoming edges
        for in_mod, _, data in dep_graph.in_edges(mod, data=True):

            weights = data.get('obj_list')
            if weights is None:
                continue

            new_weights = []
            for weight in weights:
                name = weight.string
                if name in blocks:
                    new_weights.extend(blocks[name])
                else:
                    new_weights.append(weight)

            dep_graph[in_mod][mod]['obj_list'] = new_weights

    complete_interface_blocks = {}
    for mod, blocks in interface_blocks.items():
        complete_interface_blocks.update(blocks)

    for node, node_data in dep_graph.nodes(data=True):

        objects = node_data.get('info_list')

        if objects is None:
            continue

        new_names_in_subroutines = {}
        for subroutine, names in objects.names_in_subroutines.items():

            new_names_list = []
            for name in names:
                if name in complete_interface_blocks:
                    for replacement in complete_interface_blocks[name]:
                        new_names_list.append(replacement.string)
                else:
                    new_names_list.append(name)
            new_names_in_subroutines[subroutine] = new_names_list
        objects.names_in_subroutines = new_names_in_subroutines
    
    parse_order = list(reversed(list(nx.topological_sort(dep_graph))))
    simple_graph,actually_used_in_module=ast_utils.eliminate_dependencies(dep_graph)
    
    changed=True
    while changed:
        
        simpler_graph=simple_graph.copy()
        simple_graph,actually_used_in_module=ast_utils.eliminate_dependencies(simpler_graph)
        if simple_graph.number_of_nodes()==simpler_graph.number_of_nodes() and simple_graph.number_of_edges()==simpler_graph.number_of_edges(): 
            changed=False


    parse_order = list(reversed(list(nx.topological_sort(simple_graph))))
      
    parse_list={}
    what_to_parse_list={}
    type_to_parse_list={}
    for i in parse_order:
        edges=simple_graph.in_edges(i)
        parse_list[i]=[]
        fands_list=[]
        type_list=[]
        res=simple_graph.nodes.get(i).get("info_list")
        for j in edges:
            deps=simple_graph.get_edge_data(j[0],j[1]).get("obj_list")
            if deps is None:   
                continue
            for k in deps:
                if k.string not in parse_list[i]:
                    parse_list[i].append(k.string)
            
            
            if res is not None:
                for jj in parse_list[i]:
                    if jj in res.list_of_functions:
                        if jj not in fands_list:
                            fands_list.append(jj)
                    if jj in res.list_of_subroutines:
                        if jj not in fands_list:
                            fands_list.append(jj)
                    if jj in res.list_of_types:
                        if jj not in type_list:
                            type_list.append(jj)        
        print("Module " + i + " used names: " + str(parse_list[i]))
        if len(fands_list)>0:
            print("Module " + i + " used fands: " + str(fands_list))
            print("Actually used: "+str(actually_used_in_module[i]))
        for j in actually_used_in_module[i]:
            if res is not None:
                if j in res.list_of_functions:
                    if j not in fands_list:
                        fands_list.append(j)
                if j in res.list_of_subroutines:
                    if j not in fands_list:
                        fands_list.append(j)  
                if j in res.list_of_types:
                    if j not in type_list:
                        type_list.append(j)        

        what_to_parse_list[i]=fands_list  
        type_to_parse_list[i]=type_list 
    
    if len(parse_order)==0:
        top_level_ast=ast
    else:
        top_level_ast = parse_order.pop()
    changes=True
    new_children=[]
    if len(parse_order)>0:
        for i in ast.children:
        
            if i.children[0].children[1].string not in parse_order and i.children[0].children[1].string!=top_level_ast:
                print("Module " + i.children[0].children[1].string + " not needing parsing")
            else:
                types=[]
                subroutinesandfunctions=[]
                new_spec_children=[]
                for j in i.children[1].children:
                    if j.__class__.__name__=="Type_Declaration_Stmt":
                        if j.children[0].__class__.__name__!="Declaration_Type_Spec":
                            new_spec_children.append(j) 
                            continue   
                        else:
                            entity_decls=[]
                            for k in j.children[2].children:
                                if k.__class__.__name__=="Entity_Decl":
                                    if k.children[0].string in actually_used_in_module[i.children[0].children[1].string]:
                                        entity_decls.append(k)
                            if entity_decls==[]:
                                continue            
                            if j.children[2].children.__class__.__name__=="tuple":
                                #print("Assumption failed: Tuple not expected")
                                new_spec_children.append(j)
                                continue
                            j.children[2].children.clear()
                            for k in entity_decls:
                                j.children[2].children.append(k)            
                            new_spec_children.append(j)
                    elif j.__class__.__name__=="Derived_Type_Def":
                        if j.children[0].children[1].string in type_to_parse_list[i.children[0].children[1].string]:
                            new_spec_children.append(j)
                    else:
                        new_spec_children.append(j)
                i.children[1].children.clear()
                for j in new_spec_children:
                    i.children[1].children.append(j)        
                if i.children[2].__class__.__name__=="End_Module_Stmt":
                    new_children.append(i)
                    continue
                if i.children[0].children[1].string!=top_level_ast:
                    for j in i.children[2].children:
                        if j.__class__.__name__!="Contains_Stmt":

                            if j.children[0].children[1].string in what_to_parse_list[i.children[0].children[1].string]:
                                subroutinesandfunctions.append(j)        
                    i.children[2].children.clear()
                    for j in subroutinesandfunctions:
                        i.children[2].children.append(j)        
                new_children.append(i)

        ast.children.clear()
        for i in new_children:
            ast.children.append(i)  
        name_dict = {}
        rename_dict = {}
        for i in parse_order:
            local_rename_dict = {}
            edges = list(simple_graph.in_edges(i))
            names = []
            for j in edges:
                list_dict = simple_graph.get_edge_data(j[0], j[1])
                if (list_dict['obj_list'] is not None):
                    for k in list_dict['obj_list']:
                        if not k.__class__.__name__ == "Name":
                            if k.__class__.__name__ == "Rename":
                                if k.children[2].string not in names:
                                    names.append(k.children[2].string)
                                local_rename_dict[k.children[2].string] = k.children[1].string
                            #print("Assumption failed: Object list contains non-name node")
                        else:
                            if k.string not in names:
                                names.append(k.string)
            rename_dict[i] = local_rename_dict
            name_dict[i] = names
                    

    tables = SymbolTable
    own_ast = ast_components.InternalFortranAst(ast, tables)
    functions_to_rename={}
    for i in parse_order:
        own_ast.current_ast=i
        
        own_ast.unsupported_fortran_syntax[i]=[]
        if i in ["mtime","ISO_C_BINDING", "iso_c_binding", "mo_cdi","iso_fortran_env"]:
            continue
       
        own_ast.add_name_list_for_module(i, name_dict[i])
        try:
            partial_module = own_ast.create_ast(asts[i])
        except:
            print("Module " + i + " could not be parsed ",own_ast.unsupported_fortran_syntax[i])
            print(own_ast.unsupported_fortran_syntax[i])
            continue
        tmp_rename=rename_dict[i]
        for j in tmp_rename:
            #print(j)
            if own_ast.symbols.get(j) is None:
                #raise NameError("Symbol " + j + " not found in partial ast")
                if functions_to_rename.get(i) is None:
                    functions_to_rename[i]=[j]
                else:
                    functions_to_rename[i].append(j)    
            else:
                own_ast.symbols[tmp_rename[j]]=own_ast.symbols[j]
    program = own_ast.create_ast(ast)

    # Repeated!
    # We need that to know in transformations what structures are used.
    # The actual structure listing is repeated later to resolve cycles.
    # Not sure if we can actually do it earlier.

    structs_lister=ast_transforms.StructLister()
    structs_lister.visit(program)
    struct_dep_graph=nx.DiGraph()
    for i,name in zip(structs_lister.structs,structs_lister.names):
        if name not in struct_dep_graph.nodes:
            struct_dep_graph.add_node(name)
        struct_deps_finder=ast_transforms.StructDependencyLister(structs_lister.names)
        struct_deps_finder.visit(i)
        struct_deps=struct_deps_finder.structs_used
        #print(struct_deps)
        for j,pointing,point_name in zip(struct_deps,struct_deps_finder.is_pointer,struct_deps_finder.pointer_names):
            if j not in struct_dep_graph.nodes:
                struct_dep_graph.add_node(j)
            struct_dep_graph.add_edge(name,j,pointing=pointing,point_name=point_name)

    program.structures = ast_transforms.Structures(structs_lister.structs)
    own_ast.structures = ast_transforms.Structures(structs_lister.structs)

    program.placeholders = own_ast.placeholders
    program.placeholders_offsets = own_ast.placeholders_offsets
    functions_and_subroutines_builder = ast_transforms.FindFunctionAndSubroutines()
    functions_and_subroutines_builder.visit(program)
    own_ast.functions_and_subroutines = functions_and_subroutines_builder.names
    own_ast.iblocks=functions_and_subroutines_builder.iblocks
    program = ast_transforms.functionStatementEliminator(program)
    program = ast_transforms.StructConstructorToFunctionCall(functions_and_subroutines_builder.names).visit(program)
    program = ast_transforms.CallToArray(functions_and_subroutines_builder).visit(program)
    program = ast_transforms.CallExtractor().visit(program)
    program = ast_transforms.ArgumentExtractor(program).visit(program)

    program = ast_transforms.FunctionCallTransformer().visit(program)
    program = ast_transforms.FunctionToSubroutineDefiner().visit(program)
    program = ast_transforms.PointerRemoval().visit(program)
    program = ast_transforms.ElementalFunctionExpander(functions_and_subroutines_builder.names).visit(program)
    for i in program.modules:
        count=0
        for j in i.function_definitions:
            if isinstance(j, ast_internal_classes.Subroutine_Subprogram_Node):
                i.subroutine_definitions.append(j)
                own_ast.functions_and_subroutines.append(j.name)
                count+=1
        if count!=len(i.function_definitions):
            raise NameError("Not all functions were transformed to subroutines")
        i.function_definitions=[]
    program.function_definitions=[]
    count=0
    for i in program.function_definitions:
        if isinstance(i, ast_internal_classes.Subroutine_Subprogram_Node):
            program.subroutine_definitions.append(i)
            own_ast.functions_and_subroutines.append(i.name)
            count+=1
    if count!=len(program.function_definitions):
        raise NameError("Not all functions were transformed to subroutines")
    program.function_definitions=[]
    program = ast_transforms.SignToIf().visit(program)
    program = ast_transforms.ArrayToLoop(program).visit(program)

    for transformation in own_ast.fortran_intrinsics().transformations():
        transformation.initialize(program)
        program = transformation.visit(program)

    program = ast_transforms.ForDeclarer().visit(program)
    program = ast_transforms.IndexExtractor(program, normalize_offsets).visit(program)
    program = ast_transforms.optionalArgsExpander(program)
    structs_lister=ast_transforms.StructLister()
    structs_lister.visit(program)
    struct_dep_graph=nx.DiGraph()
    for i,name in zip(structs_lister.structs,structs_lister.names):
        if name not in struct_dep_graph.nodes:
            struct_dep_graph.add_node(name)
        struct_deps_finder=ast_transforms.StructDependencyLister(structs_lister.names)
        struct_deps_finder.visit(i)
        struct_deps=struct_deps_finder.structs_used
        #print(struct_deps)
        for j,pointing,point_name in zip(struct_deps,struct_deps_finder.is_pointer,struct_deps_finder.pointer_names):
            if j not in struct_dep_graph.nodes:
                struct_dep_graph.add_node(j)
            struct_dep_graph.add_edge(name,j,pointing=pointing,point_name=point_name)
    cycles=nx.algorithms.cycles.simple_cycles(struct_dep_graph)
    has_cycles=list(cycles)
    cycles_we_cannot_ignore=[]
    for cycle in has_cycles:
        print(cycle)
        for i in cycle:
            is_pointer=struct_dep_graph.get_edge_data(i,cycle[(cycle.index(i)+1)%len(cycle)])["pointing"]
            point_name=struct_dep_graph.get_edge_data(i,cycle[(cycle.index(i)+1)%len(cycle)])["point_name"]
            #print(i,is_pointer)
            if is_pointer:
                actually_used_pointer_node_finder=ast_transforms.StructPointerChecker(i,cycle[(cycle.index(i)+1)%len(cycle)],point_name)
                actually_used_pointer_node_finder.visit(program)
                #print(actually_used_pointer_node_finder.nodes)
                if len(actually_used_pointer_node_finder.nodes)==0:
                    print("We can ignore this cycle")
                    program=ast_transforms.StructPointerEliminator(i,cycle[(cycle.index(i)+1)%len(cycle)],point_name).visit(program)
                else:
                    cycles_we_cannot_ignore.append(cycle)    
    if len(cycles_we_cannot_ignore)>0:
        raise NameError("Structs have cyclic dependencies")
    own_ast.tables = own_ast.symbols

    ast2sdfg = AST_translator(own_ast, __file__, multiple_sdfgs=multiple_sdfgs, toplevel_subroutine=sdfg_name, normalize_offsets=normalize_offsets, use_experimental_cfg_blocks=use_experimental_cfg_blocks)
    sdfg = SDFG(sdfg_name)
    ast2sdfg.actual_offsets_per_sdfg[sdfg]={}
    ast2sdfg.top_level = program
    ast2sdfg.globalsdfg = sdfg
    ast2sdfg.translate(program, sdfg)

    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.NestedSDFG):
            if node.sdfg is not None:
                if 'test_function' in node.sdfg.name:
                    sdfg = node.sdfg
                    break
    sdfg.parent = None
    sdfg.parent_sdfg = None
    sdfg.parent_nsdfg_node = None
    sdfg.reset_cfg_list()

    sdfg.apply_transformations(IntrinsicSDFGTransformation)
    sdfg.expand_library_nodes()

    sdfg.using_experimental_blocks = use_experimental_cfg_blocks
    return sdfg


def create_sdfg_from_fortran_file(source_string: str, use_experimental_cfg_blocks: bool = False):
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
    own_ast.functions_and_subroutines = functions_and_subroutines_builder.names
    program = ast_transforms.functionStatementEliminator(program)
    program = ast_transforms.CallToArray(functions_and_subroutines_builder).visit(program)
    program = ast_transforms.CallExtractor().visit(program)
    program = ast_transforms.SignToIf().visit(program)
    program = ast_transforms.ArrayToLoop().visit(program)
    program = ast_transforms.SumToLoop().visit(program)
    program = ast_transforms.ForDeclarer().visit(program)
    program = ast_transforms.IndexExtractor().visit(program)
    program = ast_transforms.optionalArgsExpander(program)
    ast2sdfg = AST_translator(own_ast, __file__, use_experimental_cfg_blocks)
    sdfg = SDFG(source_string)
    ast2sdfg.top_level = program
    ast2sdfg.globalsdfg = sdfg
    ast2sdfg.translate(program, sdfg)
    sdfg.apply_transformations(IntrinsicSDFGTransformation)
    sdfg.expand_library_nodes()
    sdfg.using_experimental_blocks = use_experimental_cfg_blocks
    return sdfg

def recursive_ast_improver(ast,
                           source_list,
                           include_list,
                           parser,
                           interface_blocks,
                           exclude_list=[],
                           missing_modules=[],
                           dep_graph=nx.DiGraph(),
                           asts={}):
    dfl = ast_utils.DefModuleLister()
    dfl.get_defined_modules(ast)
    defined_modules = dfl.list_of_modules
    main_program_mode=False
    if len(defined_modules) != 1:
        #print("Defined modules: ", defined_modules)
        print("Assumption failed: Only one module per file")
        if len(defined_modules)==0 and ast.__class__.__name__ == "Program":
            main_program_mode=True
    ufl = ast_utils.UseModuleLister()
    ufl.get_used_modules(ast)
    objects_in_modules = ufl.objects_in_use
    used_modules = ufl.list_of_modules
    
    fandsl=ast_utils.FunctionSubroutineLister()
    fandsl.get_functions_and_subroutines(ast)
    functions_and_subroutines=fandsl.list_of_functions+fandsl.list_of_subroutines #+ list(fandsl.interface_blocks.keys())
    if len(fandsl.interface_blocks) > 0:
        interface_blocks[ast.children[0].children[0].children[1].string.lower()] = fandsl.interface_blocks

    #print("Functions and subroutines: ", functions_and_subroutines)
    if not main_program_mode:
        parent_module = defined_modules[0]
    else:
        parent_module = ast.children[0].children[0].children[1].string
    for i in defined_modules:
        if i not in exclude_list:
            exclude_list.append(i)
        #if i not in dep_graph.nodes:
        dep_graph.add_node(i,info_list=fandsl)
    for i in used_modules:
        if i not in dep_graph.nodes:
            dep_graph.add_node(i)
        weight = None
        if i in objects_in_modules:
            weight=[]

            for j in objects_in_modules[i].children:
                weight.append(j)

        dep_graph.add_edge(parent_module, i, obj_list=weight)

    #print("It's turtles all the way down: ", len(exclude_list))
    modules_to_parse = []
    for i in used_modules:
        if i not in defined_modules and i not in exclude_list:
            #print("Module " + i + " not defined")
            modules_to_parse.append(i)
    added_modules = []
    for i in modules_to_parse:
        found = False
        name=i
        if i=="mo_restart_nml_and_att": 
            name="mo_restart_nmls_and_atts"
        if i=="yomhook":
            name="yomhook_dummy"    
        for j in source_list:
            if name in j:
                fname = j.split("/")
                fname = fname[len(fname) - 1]
                if fname == name + ".f90" or fname == name + ".F90":
                    found = True
                    next_file = j
                    break

        if not found:
            #print("Module " + i + " not found in source list! This is bad!")
            if i not in missing_modules:
                missing_modules.append(i)
            #raise Exception("Module " + i + " not found in source list")
            continue
        if isinstance(source_list,dict):
            reader = fsr(source_list[next_file])
            next_ast = parser(reader)
            
        else:
            next_reader = ffr(file_candidate=next_file, include_dirs=include_list, source_only=source_list)
            next_ast = parser(next_reader)

        next_ast = recursive_ast_improver(next_ast,
                                          source_list,
                                          include_list,
                                          parser,
                                          exclude_list=exclude_list,
                                          missing_modules=missing_modules,
                                          interface_blocks=interface_blocks,
                                          dep_graph=dep_graph,
                                          asts=asts)
        for mod in next_ast.children:
            added_modules.append(mod)
            if mod.children[0].children[1].string not in exclude_list:
                exclude_list.append(mod.children[0].children[1].string)

    for i in added_modules:
        if ast.children.count(i) == 0:
            ast.children.append(i)
        asts[i.children[0].children[1].string.lower()] = i
    return ast

def create_sdfg_from_fortran_file_with_options(source_string: str, source_list, include_list, icon_sources_dir, icon_sdfgs_dir, normalize_offsets: bool = False):
    """
    Creates an SDFG from a fortran file
    :param source_string: The fortran file name
    :return: The resulting SDFG

    """
    parser = pf().create(std="f2008")
    reader = ffr(file_candidate=source_string, include_dirs=include_list, source_only=source_list)
    ast = parser(reader)
    exclude_list = []
    missing_modules = []
    dep_graph = nx.DiGraph()
    asts = {}
    actually_used_in_module={}
    interface_blocks = {}
    ast = recursive_ast_improver(ast,
                                 source_list,
                                 include_list,
                                 parser,
                                 interface_blocks,
                                 exclude_list=exclude_list,
                                 missing_modules=missing_modules,
                                 dep_graph=dep_graph,
                                 asts=asts)

    for mod, blocks in interface_blocks.items():

        # get incoming edges
        for in_mod, _, data in dep_graph.in_edges(mod, data=True):

            weights = data.get('obj_list')
            if weights is None:
                continue

            new_weights = []
            for weight in weights:
                name = weight.string
                if name in blocks:
                    new_weights.extend(blocks[name])
                else:
                    new_weights.append(weight)

            dep_graph[in_mod][mod]['obj_list'] = new_weights

    complete_interface_blocks = {}
    for mod, blocks in interface_blocks.items():
        complete_interface_blocks.update(blocks)

    for node, node_data in dep_graph.nodes(data=True):

        objects = node_data.get('info_list')

        if objects is None:
            continue

        new_names_in_subroutines = {}
        for subroutine, names in objects.names_in_subroutines.items():

            new_names_list = []
            for name in names:
                if name in complete_interface_blocks:
                    for replacement in complete_interface_blocks[name]:
                        new_names_list.append(replacement.string)
                else:
                    new_names_list.append(name)
            new_names_in_subroutines[subroutine] = new_names_list
        objects.names_in_subroutines = new_names_in_subroutines
    
    #print(dep_graph)
    parse_order = list(reversed(list(nx.topological_sort(dep_graph))))
    simple_graph,actually_used_in_module=ast_utils.eliminate_dependencies(dep_graph)
    
    changed=True
    while changed:
        
        simpler_graph=simple_graph.copy()
        simple_graph,actually_used_in_module=ast_utils.eliminate_dependencies(simpler_graph)
        if simple_graph.number_of_nodes()==simpler_graph.number_of_nodes() and simple_graph.number_of_edges()==simpler_graph.number_of_edges(): 
            changed=False

    parse_order = list(reversed(list(nx.topological_sort(simple_graph))))

    parse_list={}
    what_to_parse_list={}
    type_to_parse_list={}
    for i in parse_order:
        edges=simple_graph.in_edges(i)
        parse_list[i]=[]
        fands_list=[]
        type_list=[]
        res=simple_graph.nodes.get(i).get("info_list")
        for j in edges:
            deps=simple_graph.get_edge_data(j[0],j[1]).get("obj_list")
            if deps is None:   
                continue
            for k in deps:
                if k.string not in parse_list[i]:
                    parse_list[i].append(k.string)
            
            
            if res is not None:
                for jj in parse_list[i]:

                    if jj in res.list_of_functions:
                        if jj not in fands_list:
                            fands_list.append(jj)
                    if jj in res.list_of_subroutines:
                        if jj not in fands_list:
                            fands_list.append(jj)
                    if jj in res.list_of_types:
                        if jj not in type_list:
                            type_list.append(jj)        
        print("Module " + i + " used names: " + str(parse_list[i]))
        if len(fands_list)>0:
            print("Module " + i + " used fands: " + str(fands_list))
            print("ACtually used: "+str(actually_used_in_module[i]))
        for j in actually_used_in_module[i]:
            if res is not None:
                if j in res.list_of_functions:

                    if j not in fands_list:
                        fands_list.append(j)

                if j in res.list_of_subroutines:
                    if j not in fands_list:
                        fands_list.append(j)  
                if j in res.list_of_types:
                    if j not in type_list:
                        type_list.append(j)        

        what_to_parse_list[i]=fands_list  
        type_to_parse_list[i]=type_list   
    top_level_ast = parse_order.pop()
    changes=True
    new_children=[]
    name_dict = {}
    rename_dict = {}
    for i in parse_order:
        local_rename_dict = {}
        edges = list(simple_graph.in_edges(i))
        names = []
        for j in edges:
            list_dict = simple_graph.get_edge_data(j[0], j[1])
            if (list_dict['obj_list'] is not None):
                for k in list_dict['obj_list']:
                    if not k.__class__.__name__ == "Name":
                        if k.__class__.__name__ == "Rename":
                            if k.children[2].string not in names:
                                names.append(k.children[2].string)
                            local_rename_dict[k.children[2].string] = k.children[1].string
                        #print("Assumption failed: Object list contains non-name node")
                    else:
                        if k.string not in names:
                            names.append(k.string)
        rename_dict[i] = local_rename_dict
        name_dict[i] = names

    for i in ast.children:
    
        if i.children[0].children[1].string not in parse_order and i.children[0].children[1].string!=top_level_ast:
            print("Module " + i.children[0].children[1].string + " not needing parsing")
        elif  i.children[0].children[1].string==top_level_ast:
            new_children.append(i)
        else:
            types=[]
            subroutinesandfunctions=[]
            new_spec_children=[]
            for j in i.children[1].children:
                if j.__class__.__name__=="Type_Declaration_Stmt":
                    #if j.children[0].__class__.__name__!="Declaration_Type_Spec":
                    #    new_spec_children.append(j)    
                    #else:
                        entity_decls=[]
                        for k in j.children[2].children:
                            if k.__class__.__name__=="Entity_Decl":
                                if k.children[0].string in actually_used_in_module[i.children[0].children[1].string]:
                                    entity_decls.append(k)
                                elif rename_dict[i.children[0].children[1].string].get(k.children[0].string) in actually_used_in_module[i.children[0].children[1].string]:
                                    entity_decls.append(k)    
                        if entity_decls==[]:
                            continue            
                        if j.children[2].children.__class__.__name__=="tuple":
                            #("Assumption failed: Tuple not expected")
                            new_spec_children.append(j)
                            continue
                        j.children[2].children.clear()
                        for k in entity_decls:
                            j.children[2].children.append(k)            
                        new_spec_children.append(j)
                elif j.__class__.__name__=="Derived_Type_Def":
                    if j.children[0].children[1].string in type_to_parse_list[i.children[0].children[1].string]:
                        new_spec_children.append(j)
                else:
                    new_spec_children.append(j)
            i.children[1].children.clear()
            for j in new_spec_children:
                i.children[1].children.append(j)        
            if i.children[2].__class__.__name__=="End_Module_Stmt":
                new_children.append(i)
                continue
            if i.children[0].children[1].string!=top_level_ast:
                for j in i.children[2].children:
                    if j.__class__.__name__!="Contains_Stmt":

                        if j.children[0].children[1].string in what_to_parse_list[i.children[0].children[1].string]:
                            subroutinesandfunctions.append(j)        
                i.children[2].children.clear()
                for j in subroutinesandfunctions:
                    i.children[2].children.append(j)        
            new_children.append(i)

    ast.children.clear()
    for i in new_children:
        ast.children.append(i)  
    name_dict = {}
    rename_dict = {}
    for i in parse_order:
        local_rename_dict = {}
        edges = list(simple_graph.in_edges(i))
        names = []
        for j in edges:
            list_dict = simple_graph.get_edge_data(j[0], j[1])
            if (list_dict['obj_list'] is not None):
                for k in list_dict['obj_list']:
                    if not k.__class__.__name__ == "Name":
                        if k.__class__.__name__ == "Rename":
                            if k.children[2].string not in names:
                                names.append(k.children[2].string)
                            local_rename_dict[k.children[2].string] = k.children[1].string
                        #print("Assumption failed: Object list contains non-name node")
                    else:
                        if k.string not in names:
                            names.append(k.string)
        rename_dict[i] = local_rename_dict
        name_dict[i] = names
    # for i in parse_order:
    #     edges = list(simple_graph.in_edges(i))
    #     for j in edges:
    #         list_dict = simple_graph.get_edge_data(j[0], j[1]) 
    #         names_in_edge = []
    #         if (list_dict['obj_list'] is not None):
    #             for k in list_dict['obj_list']:
    #                     names_in_edge.append(k.string)

    #         changes=True
    #         while changes:
    #             changes=False
    #             if asts.get(i) is None:
    #                 continue
    #             for k in asts[i].children[2].children:
    #                 if k.__class__.__name__ == "Contains_Stmt":
    #                     asts[i].children[2].children.remove(k)
    #                     changes=True
    #                 elif k.__class__.__name__ == "Subroutine_Subprogram":
    #                     if k.children[0].children[1].string not in names_in_edge:
    #                         asts[i].children[2].children.remove(k)
    #                         changes=True
    #                 elif k.__class__.__name__ == "Function_Subprogram":
    #                     if k.children[0].children[1].string not in names_in_edge :
    #                         asts[i].children[2].children.remove(k)
    #                         changes=True
                    
    tables = SymbolTable
    partial_ast = ast_components.InternalFortranAst(top_level_ast, tables)
    partial_modules = {}
    partial_ast.symbols["c_int"]=ast_internal_classes.Int_Literal_Node(value=4)
    partial_ast.symbols["c_int8_t"]=ast_internal_classes.Int_Literal_Node(value=1)
    partial_ast.symbols["c_int64_t"]=ast_internal_classes.Int_Literal_Node(value=8)
    partial_ast.symbols["c_int32_t"]=ast_internal_classes.Int_Literal_Node(value=4)
    partial_ast.symbols["c_size_t"]=ast_internal_classes.Int_Literal_Node(value=4)
    partial_ast.symbols["c_long"]=ast_internal_classes.Int_Literal_Node(value=8)
    partial_ast.symbols["c_signed_char"]=ast_internal_classes.Int_Literal_Node(value=1)
    partial_ast.symbols["c_char"]=ast_internal_classes.Int_Literal_Node(value=1)
    partial_ast.symbols["c_null_char"]=ast_internal_classes.Int_Literal_Node(value=1)
    functions_to_rename={}

    #Why would you ever name a file differently than the module? Especially just one random file out of thousands???
    #asts["mo_restart_nml_and_att"]=asts["mo_restart_nmls_and_atts"]
    partial_ast.to_parse_list=what_to_parse_list
    for i in parse_order:
        partial_ast.current_ast=i
        
        partial_ast.unsupported_fortran_syntax[i]=[]
        if i in ["mtime","ISO_C_BINDING", "iso_c_binding", "mo_cdi","iso_fortran_env"]:
            continue
       
        partial_ast.add_name_list_for_module(i, name_dict[i])
        try:
            partial_module = partial_ast.create_ast(asts[i.lower()])
            partial_modules[partial_module.name.name] = partial_module
        except Exception as e:
            print("Module " + i + " could not be parsed ",partial_ast.unsupported_fortran_syntax[i])
            print(e, type(e))
            #print(partial_ast.unsupported_fortran_syntax[i])
            continue
        tmp_rename=rename_dict[i]
        for j in tmp_rename:
            #print(j)
            if partial_ast.symbols.get(j) is None:
                #raise NameError("Symbol " + j + " not found in partial ast")
                if functions_to_rename.get(i) is None:
                    functions_to_rename[i]=[j]
                else:
                    functions_to_rename[i].append(j)    
            else:
                partial_ast.symbols[tmp_rename[j]]=partial_ast.symbols[j]

        print("Parsed successfully module: ", i," ",partial_ast.unsupported_fortran_syntax[i])
        #print(partial_ast.unsupported_fortran_syntax[i])
    #try:
    partial_ast.current_ast="top level"

    program = partial_ast.create_ast(ast)
    program.module_declarations = ast_utils.parse_module_declarations(program)
    #except:
        
    #        print(" top level module could not be parsed ", partial_ast.unsupported_fortran_syntax["top level"])
            #print(partial_ast.unsupported_fortran_syntax["top level"])
    #        return

    
    structs_lister=ast_transforms.StructLister()
    structs_lister.visit(program)
    struct_dep_graph=nx.DiGraph()
    for i,name in zip(structs_lister.structs,structs_lister.names):
        if name not in struct_dep_graph.nodes:
            struct_dep_graph.add_node(name)
        struct_deps_finder=ast_transforms.StructDependencyLister(structs_lister.names)
        struct_deps_finder.visit(i)
        struct_deps=struct_deps_finder.structs_used
        #print(struct_deps)
        for j,pointing,point_name in zip(struct_deps,struct_deps_finder.is_pointer,struct_deps_finder.pointer_names):
            if j not in struct_dep_graph.nodes:
                struct_dep_graph.add_node(j)
            struct_dep_graph.add_edge(name,j,pointing=pointing,point_name=point_name)

    program.structures = ast_transforms.Structures(structs_lister.structs)

    functions_and_subroutines_builder = ast_transforms.FindFunctionAndSubroutines()
    functions_and_subroutines_builder.visit(program)
    listnames=[i.name for i in functions_and_subroutines_builder.names]
    for i in functions_and_subroutines_builder.iblocks:
        if i not in listnames:
            functions_and_subroutines_builder.names.append(ast_internal_classes.Name_Node(name=i,type="VOID"))
    program.iblocks=functions_and_subroutines_builder.iblocks
    partial_ast.functions_and_subroutines = functions_and_subroutines_builder.names
    program = ast_transforms.functionStatementEliminator(program)
    program = ast_transforms.StructConstructorToFunctionCall(functions_and_subroutines_builder.names).visit(program)
    program = ast_transforms.CallToArray(functions_and_subroutines_builder).visit(program)
    #program = ast_transforms.TypeInterference(program).visit(program)
    #program = ast_transforms.ReplaceInterfaceBlocks(program, functions_and_subroutines_builder).visit(program)
    program = ast_transforms.CallExtractor().visit(program)
    program = ast_transforms.ArgumentExtractor(program).visit(program)
    program = ast_transforms.FunctionCallTransformer().visit(program)
    program = ast_transforms.FunctionToSubroutineDefiner().visit(program)
    
    #program = ast_transforms.optionalArgsExpander(program)
    #program = ast_transforms.ArgumentExtractor(program).visit(program)

    
    count=0
    for i in program.function_definitions:
        if isinstance(i, ast_internal_classes.Subroutine_Subprogram_Node):
            program.subroutine_definitions.append(i)
            partial_ast.functions_and_subroutines.append(i.name)
            count+=1
    if count!=len(program.function_definitions):
        raise NameError("Not all functions were transformed to subroutines")
    for i in program.modules:
        count=0
        for j in i.function_definitions:
            if isinstance(j, ast_internal_classes.Subroutine_Subprogram_Node):
                i.subroutine_definitions.append(j)
                partial_ast.functions_and_subroutines.append(j.name)
                count+=1
        if count!=len(i.function_definitions):
            raise NameError("Not all functions were transformed to subroutines")
        i.function_definitions=[]
    program.function_definitions=[]
    program = ast_transforms.SignToIf().visit(program)
    program = ast_transforms.ArrayToLoop(program).visit(program)
    program = ast_transforms.optionalArgsExpander(program)
    program = ast_transforms.ArgumentExtractor(program).visit(program)
    program = ast_transforms.TypeInference(program, assert_voids = False).visit(program)

    print("Before intrinsics")
    for transformation in partial_ast.fortran_intrinsics().transformations():
        transformation.initialize(program)
        program = transformation.visit(program)
    print("After intrinsics")

    program = ast_transforms.TypeInference(program).visit(program)
    program = ast_transforms.ReplaceInterfaceBlocks(program, functions_and_subroutines_builder).visit(program)
    program = ast_transforms.optionalArgsExpander(program)
    program = ast_transforms.ArgumentExtractor(program).visit(program)
    program = ast_transforms.ElementalFunctionExpander(functions_and_subroutines_builder.names).visit(program)
    print("Before intrinsics")
    for transformation in partial_ast.fortran_intrinsics().transformations():
        transformation.initialize(program)
        program = transformation.visit(program)
    print("After intrinsics")
    program = ast_transforms.ForDeclarer().visit(program)
    program = ast_transforms.PointerRemoval().visit(program)
    program = ast_transforms.IndexExtractor(program, normalize_offsets).visit(program)
    structs_lister=ast_transforms.StructLister()
    structs_lister.visit(program)
    struct_dep_graph=nx.DiGraph()
    for i,name in zip(structs_lister.structs,structs_lister.names):
        if name not in struct_dep_graph.nodes:
            struct_dep_graph.add_node(name)
        struct_deps_finder=ast_transforms.StructDependencyLister(structs_lister.names)
        struct_deps_finder.visit(i)
        struct_deps=struct_deps_finder.structs_used
        for j,pointing,point_name in zip(struct_deps,struct_deps_finder.is_pointer,struct_deps_finder.pointer_names):
            if j not in struct_dep_graph.nodes:
                struct_dep_graph.add_node(j)
            struct_dep_graph.add_edge(name,j,pointing=pointing,point_name=point_name)
    cycles=nx.algorithms.cycles.simple_cycles(struct_dep_graph)
    has_cycles=list(cycles)
    cycles_we_cannot_ignore=[]
    for cycle in has_cycles:
        print(cycle)
        for i in cycle:
            is_pointer=struct_dep_graph.get_edge_data(i,cycle[(cycle.index(i)+1)%len(cycle)])["pointing"]
            point_name=struct_dep_graph.get_edge_data(i,cycle[(cycle.index(i)+1)%len(cycle)])["point_name"]
            #print(i,is_pointer)
            if is_pointer:
                actually_used_pointer_node_finder=ast_transforms.StructPointerChecker(i,cycle[(cycle.index(i)+1)%len(cycle)],point_name,structs_lister,struct_dep_graph,"simple")
                actually_used_pointer_node_finder.visit(program)
                #print(actually_used_pointer_node_finder.nodes)
                if len(actually_used_pointer_node_finder.nodes)==0:
                    print("We can ignore this cycle")
                    program=ast_transforms.StructPointerEliminator(i,cycle[(cycle.index(i)+1)%len(cycle)],point_name).visit(program)
                else:
                    cycles_we_cannot_ignore.append(cycle)    
    if len(cycles_we_cannot_ignore)>0:
        raise NameError("Structs have cyclic dependencies")
    print("Deleting struct members...")
    struct_members_deleted=0
    for struct,name in zip(structs_lister.structs,structs_lister.names):
        struct_member_finder=ast_transforms.StructMemberLister()
        struct_member_finder.visit(struct)
        for member,is_pointer,point_name in zip(struct_member_finder.members,struct_member_finder.is_pointer,struct_member_finder.pointer_names):
            if is_pointer:    
                actually_used_pointer_node_finder=ast_transforms.StructPointerChecker(name,member,point_name,structs_lister,struct_dep_graph,"full")
                actually_used_pointer_node_finder.visit(program)
                found=False
                for i in actually_used_pointer_node_finder.nodes:
                    nl=ast_transforms.FindNames()
                    nl.visit(i)
                    if point_name in nl.names:
                        found=True
                        break
                #print("Struct Name: ",name," Member Name: ",point_name, " Found: ", found)
                if not found:
                    #print("We can delete this member")
                    struct_members_deleted+=1
                    program=ast_transforms.StructPointerEliminator(name,member,point_name).visit(program)
    print("Deleted "+ str(struct_members_deleted)+" struct members.")            
    structs_lister=ast_transforms.StructLister()
    structs_lister.visit(program)
    struct_dep_graph=nx.DiGraph()
    for i,name in zip(structs_lister.structs,structs_lister.names):
        if name not in struct_dep_graph.nodes:
            struct_dep_graph.add_node(name)
        struct_deps_finder=ast_transforms.StructDependencyLister(structs_lister.names)
        struct_deps_finder.visit(i)
        struct_deps=struct_deps_finder.structs_used
        for j,pointing,point_name in zip(struct_deps,struct_deps_finder.is_pointer,struct_deps_finder.pointer_names):
            if j not in struct_dep_graph.nodes:
                struct_dep_graph.add_node(j)
            struct_dep_graph.add_edge(name,j,pointing=pointing,point_name=point_name)

    program.structures = ast_transforms.Structures(structs_lister.structs)
    program.tables=partial_ast.symbols
    program.placeholders=partial_ast.placeholders
    program.placeholders_offsets=partial_ast.placeholders_offsets
    program.functions_and_subroutines=partial_ast.functions_and_subroutines
    unordered_modules=program.modules
    program.modules=[]
    for i in parse_order:
        for j in unordered_modules:
            if j.name.name==i.lower():
                program.modules.append(j)
    for j in unordered_modules:
        if j.name.name==top_level_ast:
            program.modules.append(j)            
    
    for i in program.modules:
        for path in source_list:
            
            if path.lower().find(i.name.name.lower())!=-1:
                mypath=path
                break
        #copyfile(mypath, os.path.join(icon_sources_dir, i.name.name.lower()+".f90"))
        for j in i.subroutine_definitions:
            #if j.name.name!="cloudscouter":
            if j.name.name!="solve_nh":
            #if j.name.name!="rot_vertex_ri" and j.name.name!="cells2verts_scalar_ri" and j.name.name!="get_indices_c" and j.name.name!="get_indices_v" and j.name.name!="get_indices_e" and j.name.name!="velocity_tendencies":
            #if j.name.name!="rot_vertex_ri":
            #if j.name.name!="velocity_tendencies":
            #if j.name.name!="cells2verts_scalar_ri":
            #if j.name.name!="get_indices_c":
                continue
            if j.execution_part is None:
                continue
            print(f"Building SDFG {j.name.name}")
            startpoint = j
            ast2sdfg = AST_translator(program, __file__,multiple_sdfgs=False,startpoint=startpoint,sdfg_path=icon_sdfgs_dir, normalize_offsets=normalize_offsets)
            sdfg = SDFG(j.name.name)
            ast2sdfg.actual_offsets_per_sdfg[sdfg]={}
            ast2sdfg.top_level = program
            ast2sdfg.globalsdfg = sdfg
            
            ast2sdfg.translate(program, sdfg)
            


            sdfg.save(os.path.join(icon_sdfgs_dir, sdfg.name + "_raw_before_intrinsics_full.sdfgz"),compress=True)
            # for sd in sdfg.all_sdfgs_recursive():
            #     free_symbols = sd.free_symbols
            #     for i in ['__f2dace_OA_iblk_d_0_s_4140', '__f2dace_OA_iidx_d_1_s_4138', '__f2dace_OA_iidx_d_2_s_4139', '__f2dace_OA_iblk_d_2_s_4142', '__f2dace_OA_iblk_d_1_s_4141', '__f2dace_OA_iidx_d_0_s_4137','__f2dace_A_iidx_d_1_s_5395', '__f2dace_A_iblk_d_0_s_5397', '__f2dace_A_iidx_d_2_s_5396', '__f2dace_A_iblk_d_1_s_5398', '__f2dace_A_iblk_d_2_s_5399', '__f2dace_A_iidx_d_0_s_5394','__f2dace_A_iidx_d_0_s_4137', '__f2dace_A_iblk_d_2_s_4142', '__f2dace_A_iblk_d_0_s_4140', '__f2dace_A_iidx_d_1_s_4138', '__f2dace_A_iidx_d_2_s_4139', '__f2dace_A_iblk_d_1_s_4141','__f2dace_OA_opt_out2_d_1_s_8111', '__f2dace_A_opt_out2_d_0_s_8110', '__f2dace_OA_opt_out2_d_2_s_8112', '__f2dace_OA_opt_out2_d_0_s_8110', '__f2dace_A_opt_out2_d_2_s_8112', '__f2dace_A_opt_out2_d_1_s_8111','__f2dace_A_opt_out2_d_1_s_8111', '__f2dace_A_ieidx_d_2_s_8121', '__f2dace_OA_opt_out2_d_1_s_8111', '__f2dace_A_ieblk_d_1_s_8123', '__f2dace_A_inidx_d_1_s_8114', '__f2dace_A_inblk_d_2_s_8118', '__f2dace_A_ieidx_d_0_s_8119', '__f2dace_A_opt_out2_d_2_s_8112', '__f2dace_A_ieidx_d_1_s_8120', '__f2dace_OA_opt_out2_d_0_s_8110', '__f2dace_A_inblk_d_0_s_8116', '__f2dace_OA_opt_out2_d_2_s_8112', '__f2dace_A_ieblk_d_2_s_8124', '__f2dace_A_inblk_d_1_s_8117', '__f2dace_A_ieblk_d_0_s_8122', '__f2dace_A_opt_out2_d_0_s_8110', '__f2dace_A_inidx_d_2_s_8115', '__f2dace_A_inidx_d_0_s_8113','__f2dace_A_iidx_d_1_s_8159', '__f2dace_A_iidx_d_0_s_8158', '__f2dace_A_iblk_d_0_s_8161', '__f2dace_A_iblk_d_1_s_8162', '__f2dace_A_iblk_d_2_s_8163', '__f2dace_A_iidx_d_2_s_8160','__f2dace_A_iblk_d_0_s_6698', '__f2dace_A_iblk_d_1_s_6699', '__f2dace_A_iidx_d_2_s_6697', '__f2dace_A_iidx_d_1_s_6696', '__f2dace_A_iblk_d_2_s_6700', '__f2dace_A_iidx_d_0_s_6695','__f2dace_A_incidx_d_2_s_8055', '__f2dace_A_iqblk_d_0_s_8044', '__f2dace_A_incblk_d_1_s_8057', '__f2dace_A_iqblk_d_1_s_8045', '__f2dace_A_iqidx_d_2_s_8043', '__f2dace_A_icidx_d_2_s_8031', '__f2dace_A_ivblk_d_2_s_8052', '__f2dace_A_incidx_d_0_s_8053', '__f2dace_A_icidx_d_0_s_8029', '__f2dace_A_incblk_d_2_s_8058', '__f2dace_A_incblk_d_0_s_8056', '__f2dace_A_ividx_d_2_s_8049', '__f2dace_A_icblk_d_0_s_8032', '__f2dace_A_ieidx_d_0_s_8035', '__f2dace_A_ivblk_d_0_s_8050', '__f2dace_A_ieidx_d_1_s_8036', '__f2dace_A_icidx_d_1_s_8030', '__f2dace_A_incidx_d_1_s_8054', '__f2dace_A_iqidx_d_0_s_8041', '__f2dace_A_icblk_d_2_s_8034', '__f2dace_A_ieblk_d_0_s_8038', '__f2dace_A_ividx_d_0_s_8047', '__f2dace_A_ieblk_d_2_s_8040', '__f2dace_A_ividx_d_1_s_8048', '__f2dace_A_icblk_d_1_s_8033', '__f2dace_A_iqidx_d_1_s_8042', '__f2dace_A_ieblk_d_1_s_8039', '__f2dace_A_iqblk_d_2_s_8046', '__f2dace_A_ieidx_d_2_s_8037', '__f2dace_A_ivblk_d_1_s_8051']:
            #         #print("I want to remove:", i)
            #         if(i in sd.symbols):
            #             sd.symbols.pop(i)
            #             print("Removed from symbols ",i)
            #             if sd.parent_nsdfg_node is not None:
            #                 if i in sd.parent_nsdfg_node.symbol_mapping:
            #                     print("Removed from symbol mapping ",i)
            #                     sd.parent_nsdfg_node.symbol_mapping.pop(i)
            sdfg.apply_transformations(IntrinsicSDFGTransformation)    

            # for sd in sdfg.all_sdfgs_recursive():
            #     free_symbols = sd.free_symbols
            #     for i in free_symbols:
            #         #print("I want to remove:", i)
            #         if(i in sd.symbols):
            #             sd.symbols.pop(i)
            #             #print("Removed from symbols ",i)
            #             if sd.parent_nsdfg_node is not None:
            #                 if i in sd.parent_nsdfg_node.symbol_mapping:
            #                     #print("Removed from symbol mapping ",i)
            #                     sd.parent_nsdfg_node.symbol_mapping.pop(i)
            #try:
            
            #    sdfg.save(os.path.join(icon_sdfgs_dir, sdfg.name + "_raw.sdfg"))
            #except:
            #    print("Intrinsics failed for ", sdfg.name)    
            #    continue
            
            try:
                sdfg.expand_library_nodes()
            except:
                print("Expansion failed for ", sdfg.name)    
                continue
            
            sdfg.validate()
            sdfg.save(os.path.join(icon_sdfgs_dir, sdfg.name + "_validated_f.sdfgz"),compress=True)
            #try:    
            sdfg.simplify(verbose=True)
            print(f'Saving SDFG {os.path.join(icon_sdfgs_dir, sdfg.name + "_simplified_tr.sdfgz")}')
            sdfg.save(os.path.join(icon_sdfgs_dir, sdfg.name + "_simplified_f.sdfgz"),compress=True)
            #except Exception as e:
            #    print("Simplification failed for ", sdfg.name)    
            #    print(e)
            #    continue
            #sdfg.save(os.path.join(icon_sdfgs_dir, sdfg.name + "_simplified.sdfg"))
            #try:  
            print(f'Compiling SDFG {os.path.join(icon_sdfgs_dir, sdfg.name + "_simplifiedf.sdfgz")}')
            sdfg.compile()
            #except Exception as e:
            #    print("Compilation failed for ", sdfg.name)
            #    print(e)
            #    continue

    #return sdfg
