import argparse
import os
from itertools import chain
from pathlib import Path
from typing import Generator

from fparser.api import get_reader
from fparser.two.Fortran2003 import Module, Derived_Type_Stmt, Module_Subprogram_Part, Data_Component_Def_Stmt, \
    Procedure_Stmt, Component_Decl, Function_Subprogram, Interface_Block, Program
from fparser.two.utils import walk

from dace.frontend.fortran.ast_desugaring import identifier_specs, append_children, set_children
from dace.frontend.fortran.ast_utils import singular
from dace.frontend.fortran.fortran_parser import ParseConfig, create_fparser_ast


def generate_serde_module(serde_base: Module, ast: Program) -> Module:
    serde_mod = serde_base
    proc_names = []
    impls = singular(sp for sp in walk(serde_mod, Module_Subprogram_Part))

    for tspec, dt in identifier_specs(ast).items():
        if not isinstance(dt, Derived_Type_Stmt):
            continue
        _, dtname, _ = dt.children
        dtdef = dt.parent

        ops = []
        for cdef in walk(dtdef, Data_Component_Def_Stmt):
            for cdecl in walk(cdef, Component_Decl):
                cname, _, _, _ = cdecl.children
                exprs = ['s'] if ops else []
                exprs.extend([f"'# {cname}'", 'new_line("A")', f"serialize(x%{cname})", 'new_line("A")'])
                ops.append(f"s = {'//'.join(exprs)}")

        ops = '\n'.join(ops)
        impl_fn = Function_Subprogram(get_reader(f"""
    function {dtname}_2s(x) result(s)
      use {tspec[0]}, only: {dtname}
      type({dtname}), intent(in) :: x
      character(len=:), allocatable :: s
      {ops}
    end function {dtname}_2s
    """.strip()))
        proc_names.append(f"{dtname}_2s")
        append_children(impls, impl_fn)

    iface = singular(p for p in walk(serde_mod, Interface_Block))
    proc_names = Procedure_Stmt(f"module procedure {', '.join(proc_names)}")
    set_children(iface, iface.children[:-1] + [proc_names] + iface.children[-1:])

    return serde_mod


def find_all_f90_files(root: Path) -> Generator[Path, None, None]:
    for f in chain(root.rglob('*.f90'), root.rglob('*.F90')):
        yield f


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument('--icon_dir', type=str,
                      help='The directory containing ICON source code (absolute path or relative to CWD).')
    argp.add_argument('--ecrad_dir', type=str,
                      default='externals/ecrad',
                      help='The directory containing ECRAD source code (path relative to `icon_dir`).')
    argp.add_argument('--ecrad_entry_file', type=str,
                      default='radiation/radiation_interface.F90',
                      help='The file containing the entry point (path relative to `ecrad_dir`).')
    argp.add_argument('--single_file_ast', type=str,
                      help='A file containing a self contained fortran program (absolute path or relative to CWD).')
    argp.add_argument('--serde_f90', type=str,
                      default=Path(os.path.dirname(os.path.realpath(__file__))).joinpath('conf_files/serde_base.f90'),
                      help='A file containing the basis of the generated functions (absolute path or relative to CWD).')
    argp.add_argument('--out_f90', type=str,
                      default=Path(os.path.dirname(os.path.realpath(__file__))).joinpath('conf_files/serde_out.f90'),
                      help='A file to write the generated functions into (absolute path or relative to CWD).')
    args = argp.parse_args()

    # Construct the primary ECRad AST.
    if args.single_file_ast:
        parse_cfg = ParseConfig(main=Path(args.single_file_ast))
    else:
        icon_dir = Path(args.icon_dir)
        ecrad_dir = icon_dir.joinpath(args.ecrad_dir)
        ecrad_entry_file = ecrad_dir.joinpath(args.ecrad_entry_file)
        parse_cfg = ParseConfig(main=ecrad_entry_file, sources=list(find_all_f90_files(ecrad_dir)))
    ast = create_fparser_ast(parse_cfg)
    # Do further FParser processing here if needed.

    serde_base = Module(get_reader(Path(args.serde_f90).read_text()))
    serde_mod = generate_serde_module(serde_base, ast)

    with open(args.out_f90, 'w') as f:
        f.write(serde_mod.tofortran())


if __name__ == "__main__":
    main()
