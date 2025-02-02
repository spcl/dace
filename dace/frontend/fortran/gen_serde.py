import argparse
from itertools import chain, combinations
from pathlib import Path
from typing import Generator, Dict, Tuple, List, Optional

from fparser.api import get_reader
from fparser.two.Fortran2003 import Module, Derived_Type_Stmt, Module_Subprogram_Part, Data_Component_Def_Stmt, \
    Procedure_Stmt, Function_Subprogram, Interface_Block, Program, Intrinsic_Type_Spec, \
    Function_Stmt, Dimension_Component_Attr_Spec, Declaration_Type_Spec, Private_Components_Stmt, Component_Part
from fparser.two.utils import walk

from dace.frontend.fortran.ast_desugaring import identifier_specs, append_children, set_children, \
    correct_for_function_calls, const_eval_nodes, deconstruct_enums, deconstruct_associations, \
    deconstruct_statement_functions, deconstruct_procedure_calls, deconstruct_interface_calls
from dace.frontend.fortran.ast_utils import singular, children_of_type, atmost_one
from dace.frontend.fortran.fortran_parser import ParseConfig, create_fparser_ast


def gen_serde_module_skeleton() -> Module:
    return Module(get_reader("""
module serde
  implicit none

  interface serialize
    module procedure :: character_2s
  end interface serialize
contains

  ! Given a string `s`, writes it to a file `path`.
  subroutine write_to(path, s)
    character(len=*), intent(in) :: path
    character(len=*), intent(in) ::  s
    integer :: io
    open (NEWUNIT=io, FILE=path, STATUS="replace", ACTION="write")
    write (io, *) s
    close (UNIT=io)
  end subroutine write_to

  ! Given a string `x`, returns a string where `x` has been serialised
  function character_2s(x) result(s)
    character(len = *), intent(in) :: x
    character(len = :), allocatable :: s
    allocate(character(len = len(x))::s)
    write (s, *) x
    s = trim(s)
  end function character_2s
end module serde
"""))


def gen_base_type_serializer(typ: str, kind: Optional[int] = None) -> Function_Subprogram:
    assert typ in {'logical', 'integer', 'real'}
    if typ == 'logical':
        assert kind is None
    elif typ == 'integer':
        assert kind in {1, 2, 4, 8}
    elif type == 'real':
        assert kind in {4, 8}
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


def generate_pointer_meta(ptr: str, rank: int, candidates: Dict[str, Tuple]) -> List[str]:
    # Assumes there is `ptr` is a pointer to an array in local scope with rank `rank`.
    # Also assumes there is a serialization sink `s` and integers `kmeta` and `kmeta_n` that can be used as iterators.
    cand_checks: List[str] = []
    for c, c_shape in candidates.items():
        c_rank = len(c_shape)
        assert rank <= c_rank
        sub_checks: List[str] = []
        for subsc in combinations(range(c_rank), c_rank - rank):
            ops: List[str] = []
            subsc_str, subsc_str_serialized_ops = [], []
            for k in range(c_rank):
                if k not in subsc:
                    subsc_str.append(':')
                    subsc_str_serialized_ops.append(f"s = s // ':'")
                    continue
                subsc_str.append(f"kmeta_{k}")
                subsc_str_serialized_ops.append(f"s = s // serialize(kmeta_{k})")
                ops.append(f"do kmeta_{k} = lbound({c}, {k + 1}), ubound({c}, {k + 1})")
            end_dos = ['end do'] * len(ops)
            subsc_str = ', '.join(subsc_str)
            subsc_str_serialized_ops = '\n'.join(subsc_str_serialized_ops)
            ops.append(f"""
if (associated({ptr}, {c}({subsc_str}))) then
kmeta = 1
s = s // "=> {c}("
{subsc_str_serialized_ops}
s = s // "))" // new_line('A')
end if
""")
            ops.extend(end_dos)
            sub_checks.append('\n'.join(ops))
        cand_checks.append('\n'.join(sub_checks))

    cand_checks: str = '\n'.join(cand_checks)
    return f"""
if (associated({ptr})) then
    kmeta = 0
    {cand_checks}
    if (kmeta == 0) then
    s = s // "=> missing" // new_line('A')
    end if
end if
""".strip().split('\n')


def generate_array_serializer(dtyp: str, rank: int, tag: str, use: Optional[str] = None) -> Function_Subprogram:
    iter_vars = ', '.join([f"k{k}" for k in range(1, rank + 1)])
    decls = f"""
{dtyp}, intent(in) :: a({', '.join([':'] * rank)})
character(len=:), allocatable :: s
integer :: k, {iter_vars}
"""
    loop_ops = []
    for k in range(1, rank + 1):
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

        # We need to make a table to find pointer associations later.
        array_map: Dict = {}
        for cdef in walk(dtdef, Data_Component_Def_Stmt):
            ctyp, attrs, cdecls = cdef.children
            ptr = 'POINTER' in f"{attrs}" if attrs else False
            dims = atmost_one(a for a in attrs.children
                              if isinstance(a, Dimension_Component_Attr_Spec)) if attrs else None
            if dims:
                _, dims = dims.children

            # Convert to canonical types.
            if isinstance(ctyp, Intrinsic_Type_Spec):
                dtyp, kind = ctyp.children
                dtyp = f"{dtyp}".lower()
                if dtyp == 'DOUBLE PRECISION':
                    dtyp, kind = 'REAL', '(KIND = 8)'
                dtyp = f"{dtyp}{kind or ''}"
            else:
                assert isinstance(ctyp, Declaration_Type_Spec)
                _, dtyp = ctyp.children
                dtyp = f"TYPE({dtyp})"

            for cdecl in cdecls.children:
                cname, shape, _, _ = cdecl.children
                if not shape:
                    # In case the shape was descirbed earlier with `dimension`.
                    shape = dims
                if ptr or not shape:
                    # We are looking for arrays.
                    continue
                rank = len(shape.children)

                array_map[(dtyp, rank)] = (cname, shape)

        ops = []

        private_access = False
        for c in dtdef.children:
            if isinstance(c, Private_Components_Stmt):
                private_access = True
            assert 'PUBLIC' != f"{c}".strip(), \
                f"We need to access public access statements in derived type defintions; got one in: {dtdef}"
            if not isinstance(c, Component_Part):
                # We care only about serialising components.
                continue
            if private_access:
                # We cannot serialise private components, since it won't compile.
                continue
            for cdef in walk(c, Data_Component_Def_Stmt):
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
                               .replace('(LEN =', '_')
                               .replace(' ', '_')
                               .replace(')', '')
                               .lower())
                        if (tag, rank) not in array_serializers:
                            array_serializers[(tag, rank)] = generate_array_serializer(dtyp, rank, tag, use)

                    ops.append(f"s = s // '# {cname}' // new_line('A')")
                    if ptr:
                        # TODO: pointer types have a whole bunch of different, best-effort strategies. For our purposes,
                        #  we will only populate this when it points to a different component of the same structure.
                        ops.append(
                            f"s = s // '# assoc' // new_line('A') // serialize(associated(x%{cname})) // new_line('A')")
                        candidates = {f"x%{v[0]}": v[1].children for k, v in array_map.items()
                                      if k[0] == dtyp and rank <= k[1]}
                        ops.extend(generate_pointer_meta(f"x%{cname}", rank, candidates))
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
        kmetas = ', '.join(f"kmeta_{k}" for k in range(10))
        impl_fn = Function_Subprogram(get_reader(f"""
    function {dtname}_2s(x) result(s)
      use {tspec[0]}, only: {dtname}
      type({dtname}), target, intent(in) :: x
      character(len=:), allocatable :: s
      integer :: kmeta, {kmetas}
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
    if root.is_file():
        yield root
    else:
        for f in chain(root.rglob('*.f90'), root.rglob('*.F90')):
            yield f


def minimal_preprocessing(ast: Program) -> Program:
    """
    We need to remove some minimal processing on the AST. Namely, we must resolve the constant parameters of the
    component shapes and types.
    """
    print("FParser Op: Removing indirections from AST...")
    ast = deconstruct_enums(ast)
    ast = deconstruct_associations(ast)
    ast = correct_for_function_calls(ast)
    ast = deconstruct_statement_functions(ast)
    ast = deconstruct_procedure_calls(ast)
    ast = deconstruct_interface_calls(ast)
    ast = correct_for_function_calls(ast)
    ast = const_eval_nodes(ast)
    return ast


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument('-i', '--in_src', type=str, required=True, action='append', default=[],
                      help='The files or directories containing Fortran source code (absolute path or relative to CWD).'
                           'Can be repeated to include multiple files and directories.')
    argp.add_argument('-o', '--out_f90', type=str, required=False, default=None,
                      help='A file to write the generated functions into (absolute path or relative to CWD).')
    args = argp.parse_args()

    input_dirs = [Path(p) for p in args.in_src]
    input_f90s = [f for p in input_dirs for f in find_all_f90_files(p)]
    print(f"Will be reading from {len(input_f90s)} Fortran files in directories: {input_dirs}")

    cfg = ParseConfig(sources=input_f90s)
    ast = create_fparser_ast(cfg)
    ast = minimal_preprocessing(ast)

    serde_base = gen_serde_module_skeleton()
    serde_mod = generate_serde_module(serde_base, ast)
    f90 = serde_mod.tofortran()

    if args.out_f90:
        with open(args.out_f90, 'w') as f:
            f.write(f90)
    else:
        print(f90)


if __name__ == "__main__":
    main()
