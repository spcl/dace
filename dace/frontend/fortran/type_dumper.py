# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import List, Optional
from dace.frontend.fortran.ast_internal_classes import Program_Node, Derived_Type_Def_Node, Var_Decl_Node, Name_Node
from dace.frontend.fortran.ast_components import InternalFortranAst
from fparser.two.parser import ParserFactory as pf, ParserFactory
from fparser.common.readfortran import FortranFileReader as ffr, FortranStringReader, FortranFileReader
import networkx as nx
import sys
from dace.frontend.fortran.ast_desugaring import SPEC, ENTRY_POINT_OBJECT_TYPES, find_name_of_stmt, find_name_of_node, \
    identifier_specs, append_children, correct_for_function_calls, remove_access_statements, sort_modules, \
    deconstruct_enums, deconstruct_interface_calls, deconstruct_procedure_calls, prune_unused_objects, \
    deconstruct_associations, assign_globally_unique_subprogram_names, assign_globally_unique_variable_names, \
    consolidate_uses, prune_branches, const_eval_nodes, lower_identifier_names, \
    remove_access_statements, ident_spec, NAMED_STMTS_OF_INTEREST_TYPES
from dace.frontend.fortran.fortran_parser import compute_dep_graph, name_and_rename_dict_creator
from fparser.two.Fortran2003 import Program, Name, Subroutine_Subprogram, Module_Stmt
from fparser.two.utils import Base, walk

from dace.frontend.fortran import ast_utils
from dace.frontend.fortran import ast_transforms
from dace.frontend.fortran import ast_internal_classes


class FortranTypeDumper:

    def __init__(self, struct_list: list[ast_internal_classes.Derived_Type_Def_Node], internal_ast: InternalFortranAst):
        self.structs = struct_list
        self.internal_ast = internal_ast

    def generate_dump_code(self, type_name: str) -> str:
        # Find the type definition in the AST
        type_def = self._find_type_definition(type_name)
        if not type_def and type_name not in self.internal_ast.types:
            raise ValueError(f"Type '{type_name}' not found in the AST")

        # Generate Fortran code to dump the content of the type

        if isinstance(type_def, Derived_Type_Def_Node):
            lines = []
            lines.append(f"subroutine dump_{type_def.name.name}({type_def.name.name}_obj, filename)")
            lines.append("use radiation_config,         only : config_type")

            lines.append("use radiation_single_level,   only : single_level_type")
            lines.append("use radiation_thermodynamics, only : thermodynamics_type")
            lines.append("use radiation_gas,            only : gas_type")
            lines.append("use radiation_cloud,          only : cloud_type")
            lines.append("use radiation_aerosol,        only : aerosol_type")
            lines.append("use radiation_flux,           only : flux_type")
            lines.append(f"  type({type_def.name.name}), intent(in) :: {type_def.name.name}_obj")
            lines.append(f"  character(len=*), intent(in) :: filename")
            lines.append(f"  integer :: unit")
            lines.append(f"  open(newunit=unit, file=filename, status='replace')")
            lines.append(self._generate_derived_type_dump_code(type_def, f"{type_def.name.name}_obj"))
            lines.append(f"  close(unit)")
            lines.append(f"end subroutine dump_{type_def.name.name}")
            return "\n".join(lines)
        else:
            lines = []
            lines.append(f"subroutine dump_{type_name}(obj, filename)")
            lines.append(f"  {type_name}, intent(in) :: obj")
            lines.append(f"  character(len=*), intent(in) :: filename")
            lines.append(f"  integer :: unit")
            lines.append(f"  open(newunit=unit, file=filename, status='replace')")
            lines.append(f"  write(unit, *) 'obj = ', obj")
            lines.append(f"  close(unit)")
            lines.append(f"end subroutine dump_{type_name}")
            return "\n".join(lines)

    def _find_type_definition(self, type_name: str) -> Optional[Derived_Type_Def_Node]:
        for structure in self.structs:
            if structure.name.name == type_name:
                return structure
        return None

    def _generate_derived_type_dump_code(self, type_def: Derived_Type_Def_Node, prefix_name: str) -> str:
        lines = []

        for component in type_def.component_part.component_def_stmts:
            for var in component.vars.vardecl:
                var_type = var.type
                var_size = var.sizes
                type_def = self._find_type_definition(var_type)
                if isinstance(type_def, Derived_Type_Def_Node):
                    self._generate_derived_type_dump_code(type_def, f"{prefix_name}%{var.name}")
                else:
                    if var.name.startswith("__f2dace_"):
                        continue
                    if var_type.lower() not in ["integer", "real", "logical", "char", "double"]:
                        lines.append(f"  ! {var.name} is of type {var_type} and size {var_size}")
                    if var_size is not None:
                        if var.alloc == True:
                            lines.append(
                                f"  write(unit, *) '{prefix_name}.{var.name}_a = ', ALLOCATED({prefix_name}%{var.name})"
                            )

                        for i in range(len(var_size)):

                            lines.append(
                                f"  write(unit, *) '{prefix_name}.{var.name}_d{i}_s = ', SIZE({prefix_name}%{var.name},{i+1})"
                            )
                            lines.append(
                                f"  write(unit, *) '{prefix_name}.{var.name}_o{i}_s = ', LBOUND({prefix_name}%{var.name},{i+1})"
                            )
                        lines.append(f"  !write(unit, *) '{prefix_name}.{var.name} = ', {prefix_name}%{var.name}")
                    else:
                        if var.name.lower() == "i_gas_model_sw" or var.name.lower() == "i_gas_model_lw":
                            lines.append(f"  write(unit, *) '{prefix_name}.{var.name} = ', {prefix_name}%i_gas_model")
                        else:
                            lines.append(f"  write(unit, *) '{prefix_name}.{var.name} = ', {prefix_name}%{var.name}")

        return "\n".join(lines)


# Example usage
# Assuming `program_ast` is an instance of `Program_Node` and `internal_ast` is an instance of `InternalFortranAst`
mini_parser = pf().create(std="f2008")
ecrad_ast = mini_parser(ffr(file_candidate=sys.argv[4]))
ast = correct_for_function_calls(ecrad_ast)
dep_graph = compute_dep_graph(ast, 'radiation_interface')
parse_order = list(reversed(list(nx.topological_sort(dep_graph))))
partial_ast = InternalFortranAst()
partial_ast.to_parse_list = {}
asts = {find_name_of_stmt(m).lower(): m for m in walk(ast, Module_Stmt)}
partial_modules = {}
functions_to_rename = {}
name_dict, rename_dict = name_and_rename_dict_creator(parse_order, dep_graph)
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
# except:

#        print(" top level module could not be parsed ", partial_ast.unsupported_fortran_syntax["top level"])
# print(partial_ast.unsupported_fortran_syntax["top level"])
#        return

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
    for j, pointing, point_name in zip(struct_deps, struct_deps_finder.is_pointer, struct_deps_finder.pointer_names):
        if j not in struct_dep_graph.nodes:
            struct_dep_graph.add_node(j)
        struct_dep_graph.add_edge(name, j, pointing=pointing, point_name=point_name)

dumper = FortranTypeDumper(structs_lister.structs, partial_ast)
fortran_code = []
for i in [
        "config_type", "single_level_type", "thermodynamics_type", "gas_type", "cloud_type", "aerosol_type", "flux_type"
]:
    fortran_code.append(dumper.generate_dump_code(i))
output_filename = "dump_all.f90"
fortran_code_all = "\n".join(fortran_code)
with open(output_filename, "w") as f:
    f.write(fortran_code_all)
