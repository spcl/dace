import argparse
import os
from itertools import chain
from pathlib import Path
from typing import Generator, Dict, Tuple, List, Optional

from fparser.api import get_reader
from fparser.two.Fortran2003 import Module, Derived_Type_Stmt, Module_Subprogram_Part, Data_Component_Def_Stmt, \
    Procedure_Stmt, Function_Subprogram, Interface_Block, Program, Intrinsic_Type_Spec, \
    Function_Stmt, Dimension_Component_Attr_Spec, Declaration_Type_Spec
from fparser.two.utils import walk

from dace.frontend.fortran.ast_desugaring import identifier_specs, append_children, set_children
from dace.frontend.fortran.ast_utils import singular, children_of_type, atmost_one
from dace.frontend.fortran.fortran_parser import ParseConfig, create_fparser_ast


def gen_base_type_serializer(typ: str, kind: Optional[int] = None) -> Function_Subprogram:
    assert typ in {'logical', 'integer', 'real'}
    assert kind is None or kind in {1, 2, 4, 8}
    fn_name = f"{typ}{kind or ''}_2s"
    kind = f"(kind={kind})" if kind else ''
    return Function_Subprogram(get_reader(f"""
function {fn_name}(x) result(s)
{typ}{kind}, intent(in) :: x
character(len=:), allocatable :: s
allocate (character(len=50) :: s)
write (s, *) x
s = trim(s)
end function {fn_name}
"""))


def generate_array_meta(arr: str, rank: int) -> List[str]:
    # Assumes there is `arr` is an array in local scope with rank `rank`.
    # Also assumes there is a serialization sink `s` and an integer `kmeta` that can be used as an iterator.
    return f"""
s = s // "# rank" // new_line('A') // serialize({rank}) // new_line('A')
s = s // "# size" // new_line('A')
do kmeta = 1, {rank}
  s = s // serialize(size({arr}, kmeta)) // new_line('A')
end do
s = s // "# lbound" // new_line('A')
do kmeta = 1, {rank}
  s = s // serialize(lbound({arr}, kmeta)) // new_line('A')
end do
""".strip().split('\n')


def generate_array_serializer(dtyp: str, rank: int, tag: str, use: Optional[str] = None) -> Function_Subprogram:
    iter_vars = ', '.join([f"k{k}" for k in range(1, rank+1)])
    decls = f"""
{dtyp}, intent(in) :: a({', '.join([':']*rank)})
character(len=:), allocatable :: s
integer :: k, {iter_vars}
"""
    loop_ops = []
    for k in range(1, rank+1):
        loop_ops.append(f"do k{k} = lbound(a, {k}), ubound(a, {k})")
    loop_ops.append(f"s = s // serialize(a({iter_vars})) // new_line('A')")
    loop_ops.extend(['end do'] * rank)
    loop = '\n'.join(loop_ops)

    return Function_Subprogram(get_reader(f"""
function {tag}_2s{rank}(a) result(s)
{use or ''}
{decls}
s = "# entries" // new_line('A')
{loop}
end function {tag}_2s{rank}
""".strip()))


def generate_serde_module(serde_base: Module, ast: Program) -> Module:
    serde_mod = serde_base
    proc_names = []
    impls = singular(sp for sp in walk(serde_mod, Module_Subprogram_Part))

    base_serializers: List[Function_Subprogram] = [
        gen_base_type_serializer('logical'),
        gen_base_type_serializer('integer', 1),
        gen_base_type_serializer('integer', 2),
        gen_base_type_serializer('integer', 4),
        gen_base_type_serializer('integer', 8),
        gen_base_type_serializer('real', 4),
        gen_base_type_serializer('real', 8),
    ]
    array_serializers: Dict[Tuple[str, int], Function_Subprogram] = {}

    ident_map = identifier_specs(ast)
    for tspec, dt in ident_map.items():
        if not isinstance(dt, Derived_Type_Stmt):
            continue
        assert len(tspec) == 2
        _, dtname, _ = dt.children
        dtdef = dt.parent

        ops = []
        for cdef in walk(dtdef, Data_Component_Def_Stmt):
            ctyp, attrs, cdecls = cdef.children
            ptr = 'POINTER' in f"{attrs}" if attrs else False
            alloc = 'ALLOCATABLE' in f"{attrs}" if attrs else False
            dims = atmost_one(a for a in attrs.children
                              if isinstance(a, Dimension_Component_Attr_Spec)) if attrs else None
            if dims:
                _, dims = dims.children

            for cdecl in cdecls.children:
                cname, shape, _, _ = cdecl.children
                if not shape:
                    # In case the shape was descirbed earlier with `dimension`.
                    shape = dims

                rank = 0
                if shape:
                    rank = len(shape.children)
                if rank:
                    if isinstance(ctyp, Intrinsic_Type_Spec):
                        dtyp, kind = ctyp.children
                        dtyp = f"{dtyp}".lower()
                        if dtyp == 'DOUBLE PRECISION':
                            dtyp, kind = 'REAL', '(KIND = 8)'
                        dtyp = f"{dtyp}{kind or ''}"
                        use = None
                    else:
                        assert isinstance(ctyp, Declaration_Type_Spec)
                        _, dtyp = ctyp.children
                        dtyp = f"{dtyp}"
                        # TODO: Don't rely on unique names.
                        if (tspec[0], dtyp) in ident_map:
                            use = f"use {tspec[0]}, only: {dtyp}"
                        else:
                            mod = singular(k[0] for k in ident_map.keys() if len(k) == 2 and k[-1] == dtyp)
                            use = f"use {mod}, only: {dtyp}"
                        dtyp = f"TYPE({dtyp})"
                    tag = (dtyp
                           .replace('TYPE(', 'dt_')
                           .replace('(KIND =', '_')
                           .replace(' ', '_')
                           .replace(')', '')
                           .lower())
                    if (tag, rank) not in array_serializers:
                        array_serializers[(tag, rank)] = generate_array_serializer(dtyp, rank, tag, use)

                ops.append(f"s = s // '# {cname}' // new_line('A')")
                if ptr:
                    # TODO: pointer types have a whole bunch of different, best-effort strategies. For our purposes, we
                    #  will only populate this when it points to a different component of the same structure.
                    ops.append(
                        f"s = s // '# assoc' // new_line('A') // serialize(associated(x%{cname})) // new_line('A')")
                else:
                    if alloc:
                        ops.append(
                            f"s = s // '# alloc' // new_line('A') // serialize(allocated(x%{cname})) // new_line('A')")
                        ops.append(f"if (allocated(x%{cname})) then")
                    if rank:
                        ops.extend(generate_array_meta(f"x%{cname}", rank))
                    ops.append(f"s = s // new_line('A') // serialize(x%{cname}) // new_line('A')")

                    if alloc:
                        ops.append("end if")

        ops = '\n'.join(ops)
        impl_fn = Function_Subprogram(get_reader(f"""
    function {dtname}_2s(x) result(s)
      use {tspec[0]}, only: {dtname}
      type({dtname}), intent(in) :: x
      character(len=:), allocatable :: s
      integer :: kmeta
      {ops}
    end function {dtname}_2s
    """.strip()))
        proc_names.append(f"{dtname}_2s")
        append_children(impls, impl_fn)

    for fn in chain(array_serializers.values(), base_serializers):
        _, name, _, _ = singular(children_of_type(fn, Function_Stmt)).children
        proc_names.append(f"{name}")
        append_children(impls, fn)

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
