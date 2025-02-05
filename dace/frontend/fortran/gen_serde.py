import argparse
from dataclasses import dataclass
from itertools import chain, combinations
from pathlib import Path
from typing import Generator, Dict, Tuple, List, Optional, Union, Any

from fparser.api import get_reader
from fparser.two.Fortran2003 import Module, Derived_Type_Stmt, Module_Subprogram_Part, Data_Component_Def_Stmt, \
    Procedure_Stmt, Function_Subprogram, Interface_Block, Program, Intrinsic_Type_Spec, \
    Function_Stmt, Dimension_Component_Attr_Spec, Declaration_Type_Spec, Private_Components_Stmt, Component_Part, \
    Derived_Type_Def
from fparser.two.utils import walk

import dace
from dace import SDFG
from dace.frontend.fortran.ast_desugaring import identifier_specs, append_children, set_children, \
    SPEC_TABLE, SPEC
from dace.frontend.fortran.ast_utils import singular, children_of_type, atmost_one
from dace.frontend.fortran.fortran_parser import ParseConfig, create_fparser_ast, run_fparser_transformations

NEW_LINE = "new_line('A')"


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

  ! Given a string `r`, adds a line it to with the content `l`, and returns as `r`.
  function add_line(r, l) result(s)
    character(len=*), intent(in) :: r
    character(len=*), intent(in) :: l
    character(len=:), allocatable :: s
    s = r // trim(l) // NEW_LINE('A')
  end function add_line

  ! Given a string `x`, returns a string where `x` has been serialised
  function character_2s(x) result(s)
    character(len=*), intent(in) :: x
    character(len=:), allocatable :: s
    allocate(character(len = len(x) + 1)::s)
    write (s, *) trim(x)
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
s = add_line(s, "# rank")
s = add_line(s, serialize({rank}))
s = add_line(s, "# size")
do kmeta = 1, {rank}
  s = add_line(s, serialize(size({arr}, kmeta)))
end do
s = add_line(s, "# lbound")
do kmeta = 1, {rank}
  s = add_line(s, serialize(lbound({arr}, kmeta)))
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
s = s // "))" // {NEW_LINE}
end if
""")
            ops.extend(end_dos)
            sub_checks.append('\n'.join(ops))
        cand_checks.append('\n'.join(sub_checks))

    cand_checks: str = '\n'.join(cand_checks)
    # TODO: We are essentially disabling all slice detection with this line. Enable back when it's performant enough.
    cand_checks: str = ''
    return f"""
if (associated({ptr})) then
    kmeta = 0
    {cand_checks}
    if (kmeta == 0) then
    s = add_line(s, "=> missing")
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
    loop_ops.append(f"s = add_line(s, serialize(a({iter_vars})))")
    loop_ops.extend(['end do'] * rank)
    loop = '\n'.join(loop_ops)

    return Function_Subprogram(get_reader(f"""
function {tag}_2s{rank}(a) result(s)
  {use or ''}
  {decls}
  s = ""  ! Start with an empty string.
  s = add_line(s, "# entries")
  {loop}
end function {tag}_2s{rank}
""".strip()))


@dataclass(frozen=True)
class DerivedTypeInfo:
    name: str
    tdef: Derived_Type_Def
    spec: SPEC


def iterate_over_derived_types(ident_map: SPEC_TABLE) -> Generator[DerivedTypeInfo, None, None]:
    """Iterate over the derived types in the given map of identifiers to their AST nodes."""
    for tspec, dt in ident_map.items():
        if not isinstance(dt, Derived_Type_Stmt):
            continue
        assert len(tspec) == 2
        _, dtname, _ = dt.children
        dtdef = dt.parent
        yield DerivedTypeInfo(f"{dtname}", dtdef, tspec)


@dataclass(frozen=True)
class ComponentInfo:
    name: str
    type: Union[Intrinsic_Type_Spec, Declaration_Type_Spec]
    ptr: bool
    alloc: bool
    rank: int
    shape: Any


def iterate_over_public_components(dt: DerivedTypeInfo) \
        -> Generator[ComponentInfo, None, None]:
    private_access = False
    for c in dt.tdef.children:
        if isinstance(c, Private_Components_Stmt):
            private_access = True
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
                if isinstance(ctyp, Intrinsic_Type_Spec):
                    dtyp, kind = ctyp.children
                    dtyp = f"{dtyp}"
                    if dtyp == 'DOUBLE PRECISION':
                        dtyp, kind = 'REAL', '(KIND = 8)'
                    if kind is None:
                        assert dtyp in {'INTEGER', 'REAL', 'LOGICAL', 'CHAR'}, f"Unexpected type: {dtyp}"
                        DEFAULT_KINDS = {
                            'INTEGER': '(KIND = 4)',
                            'REAL': '(KIND = 4)',
                        }
                        kind = DEFAULT_KINDS.get(dtyp)
                    ctyp = Intrinsic_Type_Spec(f"{dtyp}{kind or ''}")
                else:
                    assert isinstance(ctyp, Declaration_Type_Spec)

                yield ComponentInfo(f"{cname}", ctyp, ptr, alloc, rank, shape)


@dataclass(frozen=True)
class SerdeCode:
    f90_serializer: str
    cpp_deserializer: str


def generate_serde_code(ast: Program, g: SDFG) -> SerdeCode:
    # F90 Serializer related data structures.
    f90_mod = gen_serde_module_skeleton()
    proc_names = []
    impls = singular(sp for sp in walk(f90_mod, Module_Subprogram_Part))
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

    # C++ Deserializer related data structures.
    sdfg_structs: Dict[str, dace.data.Structure] = {v.name: v for k, v in g.arrays.items()
                                                    if isinstance(v, dace.data.Structure)}
    sdfg_structs: Dict[str, List[Tuple[str, dace.data.Data]]] = {k: [(kk, vv) for kk, vv in v.members.items()]
                                                                 for k, v in sdfg_structs.items()}

    def _real_ctype(v: dace.data.Data):
        if isinstance(v, dace.data.Scalar):
            return f"{v.ctype}"
        elif isinstance(v, dace.data.Array):
            return f"{v.ctype}*"
        elif isinstance(v, dace.data.Structure):
            return f"{v.ctype}"
        else:
            raise NotImplementedError

    sdfg_structs: Dict[str, Dict[str, str]] = {
        k: {kk: _real_ctype(vv) for kk, vv in v}
        for k, v in sdfg_structs.items()}
    deserializer_fns: List[str] = [f"""
void deserialize(float* x, std::istream& s) {{
    read_scalar(*x, s);
}}
void deserialize(double* x, std::istream& s) {{
    read_scalar(*x, s);
}}
void deserialize(long double* x, std::istream& s) {{
    read_scalar(*x, s);
}}
void deserialize(int* x, std::istream& s) {{
    read_scalar(*x, s);
}}
void deserialize(long* x, std::istream& s) {{
    read_scalar(*x, s);
}}
void deserialize(long long* x, std::istream& s) {{
    read_scalar(*x, s);
}}
void deserialize(bool* x, std::istream& s) {{
    char c;
    read_scalar(c, s);
    assert (c == 'T' or c == 'F');
    *x = (c == 'T');
}}
"""]

    # Actual code generation begins here.
    ident_map = identifier_specs(ast)
    for dt in iterate_over_derived_types(ident_map):
        if dt.name not in sdfg_structs:
            # The type is not present in the final SDFG, so we don't care for it.
            continue
        # We need to make a table to find pointer associations later.
        array_map: Dict = {}
        for z in iterate_over_public_components(dt):
            if z.ptr or not z.shape:
                # We are looking for arrays.
                continue
            array_map[(f"{z.type}", z.rank)] = (z.name, z.shape)

        f90ops: List[str] = []
        cppops: List[str] = []
        for z in iterate_over_public_components(dt):
            if z.name not in sdfg_structs[dt.name]:
                # The component is not present in the final SDFG, so we don't care for it.
                continue
            f90ops.append(f"s = add_line(s , '# {z.name}')")
            cppops.append(f"read_line(s);  // Should contain '# {z.name}'")

            # This is some intermediate calculation for F90 code, but no actual code is generated here (i.e., in ops).
            if z.rank:
                if isinstance(z.type, Intrinsic_Type_Spec):
                    use = None
                elif isinstance(z.type, Declaration_Type_Spec):
                    _, ctyp = z.type.children
                    ctyp = f"{ctyp}"
                    # TODO: Don't rely on unique names.
                    if (dt.spec[0], ctyp) in ident_map:
                        use = f"use {dt.spec[0]}, only: {ctyp}"
                    else:
                        mod = singular(k[0] for k in ident_map.keys() if len(k) == 2 and k[-1] == ctyp)
                        use = f"use {mod}, only: {ctyp}"
                tag = (f"{z.type}"
                       .replace('TYPE(', 'dt_')
                       .replace('(KIND =', '_')
                       .replace('(LEN =', '_')
                       .replace(' ', '_')
                       .replace(')', '')
                       .lower())
                if (tag, z.rank) not in array_serializers:
                    array_serializers[(tag, z.rank)] = generate_array_serializer(f"{z.type}", z.rank, tag, use)

            if z.ptr:
                # TODO: pointer types have a whole bunch of different, best-effort strategies. For our purposes,
                #  we will only populate this when it points to a different component of the same structure.
                f90ops.append(f"""
s = add_line(s, '# assoc')
s = add_line(s, serialize(associated(x%{z.name})))""")
                candidates = {f"x%{v[0]}": v[1].children for k, v in array_map.items()
                              if k[0] == f"{z.type}" and z.rank <= k[1]}
                f90ops.extend(generate_pointer_meta(f"x%{z.name}", z.rank, candidates))

                # TODO: pointer types have a whole bunch of different, best-effort strategies. For our purposes, we will
                #  only populate this when it points to a different component of the same structure.
                cppops.append(f"""
read_line(s);  // Should contain '# assoc'
deserialize(&yep, s);
""")
                # TODO: Currenly we do nothing but this is the flag of associated values, so `nullptr` anyway.
                cppops.append(f"x->{z.name} = nullptr;")
            else:
                if z.alloc:
                    f90ops.append(f"""
s = add_line(s, '# alloc')
s = add_line(s, serialize(allocated(x%{z.name})))
if (allocated(x%{z.name})) then
""")
                    cppops.append(f"""
read_line(s);  // Should contain '# alloc'
deserialize(&yep, s);
if (yep) {{  // BEGINING IF
""")
                if z.rank:
                    f90ops.extend(generate_array_meta(f"x%{z.name}", z.rank))
                    f90ops.append(f"s = add_line(s, serialize(x%{z.name}))")
                    assert '***' not in sdfg_structs[dt.name][z.name]
                    ptrptr = '**' in sdfg_structs[dt.name][z.name]
                    if ptrptr:
                        cppops.append(f"""
m = read_array_meta(s);
read_line(s);  // Should contain '# entries'
// We only need to allocate a volume of contiguous memory, and let DaCe interpret (assuming it follows the same protocol 
// as us).
x ->{z.name} = new std::remove_pointer<decltype(x ->{z.name})>::type[m.volume()];
for (int i=0; i<m.volume(); ++i) {{
  x->{z.name}[i] = new std::remove_pointer<std::remove_reference<decltype(x->{z.name}[i])>::type>::type;
  deserialize(x->{z.name}[i], s);
}}
""")
                    else:
                        cppops.append(f"""
m = read_array_meta(s);
read_line(s);  // Should contain '# entries'
// We only need to allocate a volume of contiguous memory, and let DaCe interpret (assuming it follows the same protocol 
// as us).
x ->{z.name} = new std::remove_pointer<decltype(x ->{z.name})>::type[m.volume()];
for (int i=0; i<m.volume(); ++i) {{
  deserialize(&(x->{z.name}[i]), s);
}}
""")
                elif '*' in sdfg_structs[dt.name][z.name]:
                    f90ops.append(f"s = add_line(s, serialize(x%{z.name}))")
                    cppops.append(f"""
x ->{z.name} = new std::remove_pointer<decltype(x ->{z.name})>::type;
deserialize(x->{z.name}, s);
""")
                else:
                    f90ops.append(f"s = add_line(s, serialize(x%{z.name}))")
                    cppops.append(f"""
deserialize(&(x->{z.name}), s);
""")

                if z.alloc:
                    f90ops.append("end if")
                    cppops.append(f"""}} // CONCLUDING IF""")

        # Conclude the serializer of the type.
        f90ops: str = '\n'.join(f90ops)
        kmetas = ', '.join(f"kmeta_{k}" for k in range(10))
        impl_fn = Function_Subprogram(get_reader(f"""
function {dt.name}_2s(x) result(s)
  use {dt.spec[0]}, only: {dt.name}
  type({dt.name}), target, intent(in) :: x
  character(len=:), allocatable :: s
  integer :: kmeta, {kmetas}
  s = ""  ! Start with an empty string.
  {f90ops}
end function {dt.name}_2s
""".strip()))
        proc_names.append(f"{dt.name}_2s")
        append_children(impls, impl_fn)

        # Conclude the deserializer of the type.
        cppops: str = '\n'.join(cppops)
        deserializer_fns.append(f"""
void deserialize({dt.name}* x, std::istream& s) {{
    bool yep;
    char bin[101];  // We are assuming that our comment lines won't be too long
    array_meta m;
    {cppops}
}}
""")

    # Conclude the serializer code.
    for fn in chain(array_serializers.values(), base_serializers):
        _, name, _, _ = singular(children_of_type(fn, Function_Stmt)).children
        proc_names.append(f"{name}")
        append_children(impls, fn)
    iface = singular(p for p in walk(f90_mod, Interface_Block))
    proc_names = Procedure_Stmt(f"module procedure {', '.join(proc_names)}")
    set_children(iface, iface.children[:-1] + [proc_names] + iface.children[-1:])
    f90_code = f90_mod.tofortran()

    # Conclude the deserializer code.
    forward_decls: str = '\n'.join(f"struct {k};" for k in sdfg_structs.keys())
    struct_defs: Dict[str, str] = {k: '\n'.join(f"{typ} {comp};" for comp, typ in sorted(v.items()))
                                   for k, v in sdfg_structs.items()}
    struct_defs: str = '\n'.join(f"""
struct {name} {{
{comps}
}};
""" for name, comps in struct_defs.items())
    deserializer_fns: str = '\n'.join(deserializer_fns)
    cpp_code = f"""
#ifndef __DACE_SERDE__
#define __DACE_SERDE__

#include <cassert>
#include <istream>
#include <iostream>

// Forward declarations of structs.
{forward_decls}

// (Re-)definitions of structs.
{struct_defs}

namespace serde {{
    struct array_meta {{
      int rank = 0;
      std::vector<int> size, lbound;

      int volume() const {{  return std::reduce(size.begin(), size.end(), 1, std::multiplies<int>()) ; }}
    }};

    void scroll_space(std::istream& s) {{
        while (!s.eof() && isspace(s.peek())) s.get();
    }}

    void read_line(std::istream& s) {{
        if (s.eof()) return;
        scroll_space(s);
        char bin[101];
        s.getline(bin, 100);
    }}

    template<typename T>
    void read_scalar(T& x, std::istream& s) {{
        if (s.eof()) return;
        scroll_space(s);
        s >> x;
    }}

    array_meta read_array_meta(std::istream& s) {{
        array_meta m;
        read_line(s);  // Should contain '# rank'
        read_scalar(m.rank, s);
        m.size.resize(m.rank);
        m.lbound.resize(m.rank);
        read_line(s);  // Should contain '# size'
        for (int i=0; i<m.rank; ++i) {{
            read_scalar(m.size[i], s);
        }}
        read_line(s);  // Should contain '# lbound'
        for (int i=0; i<m.rank; ++i) {{
            read_scalar(m.lbound[i], s);
        }}
        return m;
    }}

    {deserializer_fns}
}}  // namesepace serde

#endif // __DACE_SERDE__
"""

    return SerdeCode(f90_serializer=f90_code.strip(), cpp_deserializer=cpp_code.strip())


def find_all_f90_files(root: Path) -> Generator[Path, None, None]:
    if root.is_file():
        yield root
    else:
        for f in chain(root.rglob('*.f90'), root.rglob('*.F90')):
            yield f


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument('-i', '--in_src', type=str, required=True, action='append', default=[],
                      help='The files or directories containing Fortran source code (absolute path or relative to CWD).'
                           'Can be repeated to include multiple files and directories.')
    argp.add_argument('-g', '--in_sdfg', type=str, required=True, default=None,
                      help='The SDFG file containing the final product of DaCe. We need this to know the structures '
                           'and their members that are present in the end (aftre further pruning etc.)')
    argp.add_argument('-f', '--out_f90', type=str, required=False, default=None,
                      help='A file to write the generated F90 functions into (absolute path or relative to CWD).')
    argp.add_argument('-c', '--out_cpp', type=str, required=False, default=None,
                      help='A file to write the generated C++ functions into (absolute path or relative to CWD).')
    args = argp.parse_args()

    input_dirs = [Path(p) for p in args.in_src]
    input_f90s = [f for p in input_dirs for f in find_all_f90_files(p)]
    print(f"Will be reading from {len(input_f90s)} Fortran files in directories: {input_dirs}")

    print(f"Will be using the SDFG as the deserializer target: {args.in_sdfg}")
    g = SDFG.from_file(args.in_sdfg)

    cfg = ParseConfig(sources=input_f90s)
    ast = create_fparser_ast(cfg)
    ast = run_fparser_transformations(ast, cfg)
    serde_code = generate_serde_code(ast, g)

    if args.out_f90:
        with open(args.out_f90, 'w') as f:
            f.write(serde_code.f90_serializer)
    else:
        print(f"=== SERIALIZER CODE BEGINS ===")
        print(serde_code.f90_serializer)
        print(f"=== SERIALIZER CODE ENDS ===")

    if args.out_cpp:
        with open(args.out_cpp, 'w') as f:
            f.write(serde_code.cpp_deserializer)
    else:
        print(f"=== DESERIALIZER CODE BEGINS ===")
        print(serde_code.cpp_deserializer)
        print(f"=== DESERIALIZER CODE ENDS ===")


if __name__ == "__main__":
    main()
