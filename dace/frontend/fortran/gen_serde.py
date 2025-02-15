import argparse
from dataclasses import dataclass
from itertools import chain, combinations
from pathlib import Path
from typing import Generator, Dict, Tuple, List, Optional, Union, Any

from fparser.api import get_reader
from fparser.two.Fortran2003 import Module, Derived_Type_Stmt, Module_Subprogram_Part, Data_Component_Def_Stmt, \
    Procedure_Stmt, Interface_Block, Program, Intrinsic_Type_Spec, \
    Dimension_Component_Attr_Spec, Declaration_Type_Spec, Private_Components_Stmt, Component_Part, \
    Derived_Type_Def, Subroutine_Subprogram, Subroutine_Stmt
from fparser.two.utils import walk

import dace
from dace import SDFG
from dace.frontend.fortran.ast_desugaring import identifier_specs, append_children, set_children, \
    SPEC_TABLE, SPEC, ConstTypeInjection, find_name_of_node
from dace.frontend.fortran.ast_utils import singular, children_of_type, atmost_one
from dace.frontend.fortran.fortran_parser import ParseConfig, create_fparser_ast, run_fparser_transformations

NEW_LINE = "NEW_LINE('A')"


def gen_f90_serde_module_skeleton() -> Module:
    return Module(get_reader(f"""
module serde
  implicit none

  ! ALWAYS: First argument should be an integer `io` that is an **opened** writeable stream.
  ! ALWAYS: Second argument should be a **specialization type** to be serialized.
  ! ALWAYS: Third argument should be an optional logical about whether to close `io` afterward (default true).
  ! ALWAYS: Fourth argument should be an optional logical about whether add a new line afterward (default true).
  interface serialize
    module procedure :: W_string
  end interface serialize

  ! A counter for versioning data files.
  integer :: generation = 0
contains

  ! Call `tic()` to switch to a new version of data files.
  subroutine tic()
    generation = generation + 1
  end subroutine tic

  ! Constructs a versioned file name for data file.
  function cat(prefix, asis) result(path)
    character(len=*), intent(in) :: prefix
    character(len=:), allocatable :: path
    character(len=50) :: gen
    logical, optional, intent(in) :: asis
    logical :: asis_local
    asis_local = .false.
    if (present(asis)) asis_local = asis
    if (asis_local) then
      path = prefix
    else
      ! NOTE: SINCE WE ARE WRITING TO STRING, WE DON'T ADVANCE.
      write (gen, '(g0)') generation
      path = prefix // '.' // trim(gen) // ".data"
    endif
  end function cat

  ! Constructs a versioned file name for data file, opens it for (over)writing, and returns the handler.
  function at(prefix, asis) result(io)
    character(len=*), intent(in) :: prefix
    integer :: io
    logical, optional, intent(in) :: asis
    logical :: asis_local = .false.
    if (present(asis)) asis_local = asis
    open (NEWUNIT=io, FILE=cat(prefix, asis_local), STATUS="replace", ACTION="write")
  end function at

  subroutine W_string(io, x, cleanup, nline)
    integer :: io
    character(len=*), intent(in) :: x
    logical, optional, intent(in) :: cleanup, nline
    logical :: cleanup_local, nline_local
    cleanup_local = .true.
    nline_local = .true.
    if (present(cleanup)) cleanup_local = cleanup
    if (present(nline)) nline_local = nline
    write (io, '(g0)', advance='no') trim(x)
    if (nline_local)  write (io, '(g0)', advance='no') {NEW_LINE}
    if (cleanup_local) close(UNIT=io)
  end subroutine W_string
end module serde
"""))


def gen_base_type_serializer(typ: str, kind: Optional[int] = None) -> Subroutine_Subprogram:
    assert typ in {'logical', 'integer', 'real'}
    if typ == 'logical':
        assert kind is None
    elif typ == 'integer':
        assert kind in {1, 2, 4, 8}
    elif typ == 'real':
        assert kind in {4, 8}
    fn_name = f"W_{typ}{kind or ''}"
    kind = f"(kind={kind})" if kind else ''
    if typ == 'logical':
        op = '\n'.join(['y = merge(1, 0, x)', "write (io, '(g0)', advance='no') y"])
    else:
        op = "write (io, '(g0)', advance='no') x"

    return Subroutine_Subprogram(get_reader(f"""
subroutine {fn_name}(io, x, cleanup, nline)
  integer :: io
  {typ}{kind}, intent(in) :: x
  integer :: y
  logical, optional, intent(in) :: cleanup, nline
  logical :: cleanup_local, nline_local
  cleanup_local = .true.
  nline_local = .true.
  if (present(cleanup)) cleanup_local = cleanup
  if (present(nline)) nline_local = nline
  {op}
  if (nline_local)  write (io, '(g0)', advance='no') {NEW_LINE}
  if (cleanup_local) close(UNIT=io)
end subroutine {fn_name}
"""))


def generate_array_meta_f90(arr: str, rank: int) -> List[str]:
    # Assumes there is `arr` is an array in local scope with rank `rank`.
    # Also assumes there is a serialization sink `io` and an integer `kmeta` that can be used as an iterator.
    return f"""
call serialize(io, "# rank", cleanup=.false.)
call serialize(io, {rank}, cleanup=.false.)
call serialize(io, "# size", cleanup=.false.)
do kmeta = 1, {rank}
  call serialize(io, size({arr}, kmeta), cleanup=.false.)
end do
call serialize(io, "# lbound", cleanup=.false.)
do kmeta = 1, {rank}
  call serialize(io, lbound({arr}, kmeta), cleanup=.false.)
end do
""".strip().split('\n')


def generate_pointer_meta_f90(ptr: str, rank: int, candidates: Dict[str, Tuple]) -> List[str]:
    # Assumes there is `ptr` is a pointer to an array in local scope with rank `rank`.
    # Also assumes there is a serialization sink `io` and integers `kmeta` and `kmeta_n` that can be used as iterators.
    cand_checks: List[str] = []
    for c, c_shape in candidates.items():
        c_rank = len(c_shape)
        assert rank <= c_rank
        sub_checks: List[str] = []
        for subsc in combinations(range(c_rank), c_rank - rank):
            ops: List[str] = []
            subsc_str, subsc_str_serialized = [], []
            for k in range(c_rank):
                if k not in subsc:
                    subsc_str.append(':')
                    subsc_str_serialized.append('call serialize(io, ":", cleanup=.false., nline=.false.)')
                    continue
                subsc_str.append(f"kmeta_{k}")
                subsc_str_serialized.append(f"call serialize(io, kmeta_{k}, cleanup=.false., , nline=.false.)")
                ops.append(f"do kmeta_{k} = lbound({c}, {k + 1}), ubound({c}, {k + 1})")
            end_dos = ['end do'] * len(ops)
            subsc_str = ', '.join(subsc_str)
            subsc_str_serialized = '\n call serialize(io, ",", cleanup=.false., , nline=.false.) \n'.join(subsc_str_serialized)
            ops.append(f"""
if (associated({ptr}, {c}({subsc_str}))) then
  kmeta = 1
  call serialize(io, "=> {c}(", cleanup=.false., , nline=.false.)
  {subsc_str_serialized}
  call serialize(io, "))", cleanup=.false.)
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
    call serialize(io, "=> missing", cleanup=.false.)
  end if
end if
""".strip().split('\n')


def generate_array_serializer_f90(dtyp: str, rank: int, tag: str, use: Optional[str] = None) -> Subroutine_Subprogram:
    iter_vars = ', '.join([f"k{k}" for k in range(1, rank + 1)])
    decls = f"""
{dtyp}, intent(in) :: x({', '.join([':'] * rank)})
integer :: k, kmeta, {iter_vars}
"""
    loop_ops = []
    for k in reversed(range(1, rank + 1)):
        loop_ops.append(f"do k{k} = lbound(x, {k}), ubound(x, {k})")
    loop_ops.append(f"call serialize(io, x({iter_vars}), cleanup=.false.)")
    loop_ops.extend(['end do'] * rank)
    loop = '\n'.join(loop_ops)
    meta_ops = generate_array_meta_f90('x', rank)
    meta = '\n'.join(meta_ops)
    fn_name = f"W_{tag}_R_{rank}"

    return Subroutine_Subprogram(get_reader(f"""
subroutine {fn_name}(io, x, cleanup, nline, meta)
  {use or ''}
  integer :: io
  {decls}
  logical, optional, intent(in) :: cleanup, nline, meta
  logical :: cleanup_local, nline_local, meta_local
  cleanup_local = .true.
  nline_local = .true.
  meta_local = .true.
  if (present(cleanup)) cleanup_local = cleanup
  if (present(nline)) nline_local = nline
  if (present(meta)) meta_local = meta
  if (meta_local) then
    {meta}
  endif
  call serialize(io, "# entries", cleanup=.false.)
  {loop}
  ! NOTE: THIS CONDITIONAL IS INTENTIONALLY COMMENTED OUT, BECAUSE EACH ELEMENT ADD NEW LINE ANYWAY.
  ! if (nline_local)  write (io, '(g0)', advance='no') {NEW_LINE}
  if (cleanup_local) close(UNIT=io)
end subroutine {fn_name}
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
    cpp_serde: str


def generate_serde_code(ast: Program, g: SDFG) -> SerdeCode:
    # F90 Serializer related data structures.
    f90_mod = gen_f90_serde_module_skeleton()
    proc_names = []
    impls = singular(sp for sp in walk(f90_mod, Module_Subprogram_Part))
    base_serializers: List[Subroutine_Subprogram] = [
        gen_base_type_serializer('logical'),
        gen_base_type_serializer('integer', 1),
        gen_base_type_serializer('integer', 2),
        gen_base_type_serializer('integer', 4),
        gen_base_type_serializer('integer', 8),
        gen_base_type_serializer('real', 4),
        gen_base_type_serializer('real', 8),
    ]
    array_serializers: Dict[Tuple[str, int], Subroutine_Subprogram] = {}
    # Generate basic array serializers for ranks 1 to 4.
    for rank in range(1, 5):
        typez = ['LOGICAL', 'INTEGER(KIND = 1)', 'INTEGER(KIND = 2)', 'INTEGER(KIND = 4)', 'INTEGER(KIND = 8)',
                 'REAL(KIND = 4)', 'REAL(KIND = 8)']
        for t in typez:
            tag = (f"{t}"
                   .replace('TYPE(', 'dt_')
                   .replace('(KIND =', '_')
                   .replace('(LEN =', '_')
                   .replace(' ', '_')
                   .replace(')', '')
                   .lower())
            array_serializers[(tag, rank)] = generate_array_serializer_f90(t, rank, tag)

    # C++ SerDe related data structures.
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
    cpp_deserializer_fns: List[str] = [f"""
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
    read_scalar(*x, s);
}}
"""]
    cpp_serializer_fns: List[str] = [f"""
template<typename T>
void add_line(const T& x, std::ostream& s, bool trailing_newline=true) {{
    s << x;
    if (trailing_newline) s << std::endl;
}}
void add_line(long long x, std::ostream& s, bool trailing_newline=true) {{
    s << x;
    if (trailing_newline) s << std::endl;
}}
void add_line(long double x, std::ostream& s, bool trailing_newline=true) {{
    s << x;
    if (trailing_newline) s << std::endl;
}}
void add_line(bool x, std::ostream& s, bool trailing_newline=true) {{
    add_line(int(x), s, trailing_newline);
}}
template<typename T>
std::string serialize(const T* x) {{
    std::stringstream s;
    add_line(*x, s, false);
    return s.str();
}}
std::string serialize(int x) {{
    return std::to_string(x);
}}
std::string serialize(long x) {{
    return std::to_string(x);
}}
std::string serialize(long long x) {{
    return std::to_string(x);
}}
std::string serialize(float x) {{
    return std::to_string(x);
}}
std::string serialize(double x) {{
    return std::to_string(x);
}}
std::string serialize(long double x) {{
    return std::to_string(x);
}}
std::string serialize(bool x) {{
    return serialize(int(x));
}}
"""]
    config_injection_fns: List[str] = []

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

        def __strip_last_int(x: str) -> str:
            return '_'.join(x.split('_')[:-1]) if x.startswith("__f2dace_") else x

        all_sa_vars: Dict[str, str] = {__strip_last_int(z): z for z in sdfg_structs[dt.name].keys()
                                       if z.startswith("__f2dace_SA_")}
        all_soa_vars: Dict[str, str] = {__strip_last_int(z): z for z in sdfg_structs[dt.name].keys()
                                        if z.startswith("__f2dace_SOA_")}
        all_cinjops: Dict[str, Tuple[str, str]] = {__strip_last_int(z): ('.'.join(dt.spec), z)
                                                   for z, t in sdfg_structs[dt.name].items() if '*' not in t}
        for z in iterate_over_public_components(dt):
            if z.alloc:
                all_cinjops[f"{z.name}_a"] = ('.'.join(dt.spec), z.name)

        f90_ser_ops: List[str] = []
        cpp_ser_ops: List[str] = []
        cpp_deser_ops: List[str] = []
        for z in iterate_over_public_components(dt):
            if z.name not in sdfg_structs[dt.name]:
                # The component is not present in the final SDFG, so we don't care for it.
                continue
            f90_ser_ops.append(f"call serialize(io , '# {z.name}', cleanup=.false.)")
            cpp_ser_ops.append(f"""add_line("# {z.name}", s);""")
            cpp_deser_ops.append(f"""read_line(s, {{"# {z.name}"}});  // Should contain '# {z.name}'""")

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
                    array_serializers[(tag, z.rank)] = generate_array_serializer_f90(f"{z.type}", z.rank, tag, use)

            if z.ptr:
                # TODO: pointer types have a whole bunch of different, best-effort strategies. For our purposes,
                #  we will only populate this when it points to a different component of the same structure.
                f90_ser_ops.append(f"""
call serialize(io, '# assoc', cleanup=.false.)
call serialize(io, associated(x%{z.name}), cleanup=.false.)
""")
                cpp_ser_ops.append(f"""
add_line("# assoc", s);
add_line(serialize(x->{z.name} != nullptr), s);
""")
                candidates = {f"x%{v[0]}": v[1].children for k, v in array_map.items()
                              if k[0] == f"{z.type}" and z.rank <= k[1]}
                f90_ser_ops.extend(generate_pointer_meta_f90(f"x%{z.name}", z.rank, candidates))
                cpp_ser_ops.append(f"""add_line("=> missing", s);""")

                # TODO: pointer types have a whole bunch of different, best-effort strategies. For our purposes, we will
                #  only populate this when it points to a different component of the same structure.
                cpp_deser_ops.append(f"""
read_line(s, {{"# assoc"}});  // Should contain '# assoc'
deserialize(&yep, s);
""")
                # TODO: Currenly we do nothing but this is the flag of associated values, so `nullptr` anyway.
                cpp_deser_ops.append(f"""
read_line(s, {{"=>"}});  // Should contain '=> ...'
x->{z.name} = nullptr;
""")
            else:
                if z.alloc:
                    f90_ser_ops.append(f"""
call serialize(io, '# alloc', cleanup=.false.)
call serialize(io, allocated(x%{z.name}), cleanup=.false.)
if (allocated(x%{z.name})) then  ! BEGINNING IF
""")
                    cpp_ser_ops.append(f"""
add_line("# alloc", s);
add_line(serialize(x->{z.name} != nullptr), s);
if (x->{z.name}) {{    // BEGINING IF
""")
                    cpp_deser_ops.append(f"""
read_line(s, {{"# alloc"}});  // Should contain '# alloc'
deserialize(&yep, s);
if (yep) {{  // BEGINING IF
""")
                if z.rank:
                    f90_ser_ops.extend(generate_array_meta_f90(f"x%{z.name}", z.rank))
                    f90_ser_ops.append(f"call serialize(io, x%{z.name}, cleanup=.false., nline=.true., meta=.false.)")
                    assert '***' not in sdfg_structs[dt.name][z.name]
                    ptrptr = '**' in sdfg_structs[dt.name][z.name]
                    if z.alloc:
                        sa_vars = [all_sa_vars[f"__f2dace_SA_{z.name}_d_{dim}_s"] for dim in range(z.rank)]
                        sa_vars = '\n'.join([f"x->{v} = m.size[{dim}];" for dim, v in enumerate(sa_vars)])
                        soa_vars = [all_soa_vars[f"__f2dace_SOA_{z.name}_d_{dim}_s"] for dim in range(z.rank)]
                        soa_vars = '\n'.join([f"x->{v} = m.lbound[{dim}];" for dim, v in enumerate(soa_vars)])
                    else:
                        sa_vars, soa_vars = '', ''
                    cpp_ser_ops.append(f"""
{{
    const array_meta& m = (*ARRAY_META_DICT())[x->{z.name}];
    add_line("# rank", s);
    add_line(m.rank, s);
    add_line("# size", s);
    for (auto i : m.size) add_line(i, s);
    add_line("# lbound", s);
    for (auto i : m.lbound) add_line(i, s);
    add_line("# entries", s);
    for (int i=0; i<m.volume(); ++i) {{
        add_line(serialize(x->{z.name}[i]), s);
    }}
}}
""")
                    if ptrptr:
                        cpp_deser_ops.append(f"""
m = read_array_meta(s);
{sa_vars}
{soa_vars}
// TODO: THIS IS POTENTIALLY BUGGY, BECAUSE IT IS NOT REALLY TESTED.
// We only need to allocate a volume of contiguous memory, and let DaCe interpret (assuming it follows the same protocol 
// as us).
x ->{z.name} = m.read<std::remove_pointer<decltype(x ->{z.name})>::type>(s);
""")
                    else:
                        cpp_deser_ops.append(f"""
m = read_array_meta(s);
{sa_vars}
{soa_vars}
// We only need to allocate a volume of contiguous memory, and let DaCe interpret (assuming it follows the same protocol 
// as us).
x ->{z.name} = m.read<std::remove_pointer<decltype(x ->{z.name})>::type>(s);
""")
                elif '*' in sdfg_structs[dt.name][z.name]:
                    f90_ser_ops.append(f"call serialize(io, x%{z.name}, cleanup=.false.)")
                    cpp_ser_ops.append(f"add_line(serialize(x->{z.name}), s);")
                    cpp_deser_ops.append(f"""
x ->{z.name} = new std::remove_pointer<decltype(x ->{z.name})>::type;
deserialize(x->{z.name}, s);
""")
                else:
                    f90_ser_ops.append(f"call serialize(io, x%{z.name}, cleanup=.false.)")
                    cpp_ser_ops.append(f"add_line(serialize(x->{z.name}), s);")
                    cpp_deser_ops.append(f"""
deserialize(&(x->{z.name}), s);
""")

                if z.alloc:
                    f90_ser_ops.append("end if  ! CONCLUDING IF")
                    cpp_ser_ops.append(f"""}}  // CONCLUDING IF""")
                    cpp_deser_ops.append(f"""}}  // CONCLUDING IF""")

        # Conclude the F90 serializer of the type.
        f90_ser_ops: str = '\n'.join(f90_ser_ops)
        kmetas = ', '.join(f"kmeta_{k}" for k in range(10))
        impl_fn = Subroutine_Subprogram(get_reader(f"""
subroutine W_{dt.name}(io, x, cleanup, nline)
  use {dt.spec[0]}, only: {dt.name}
  integer :: io
  type({dt.name}), target, intent(in) :: x
  logical, optional, intent(in) :: cleanup, nline
  integer :: kmeta, {kmetas}
  logical :: cleanup_local, nline_local
  cleanup_local = .true.
  nline_local = .true.
  if (present(cleanup)) cleanup_local = cleanup
  if (present(nline)) nline_local = nline
  {f90_ser_ops}
  if (nline_local)  write (io, '(g0)', advance='no') {NEW_LINE}
  if (cleanup_local) close(UNIT=io)
end subroutine W_{dt.name}
""".strip()))
        proc_names.append(f"W_{dt.name}")
        append_children(impls, impl_fn)

        # Conclude the C++ serializer of the type.
        cpp_ser_ops: str = '\n'.join(cpp_ser_ops)
        cpp_serializer_fns.append(f"""
std::string serialize(const {dt.name}* x) {{
    std::stringstream s;
    {cpp_ser_ops}
    std::string out = s.str();
    if (out.length() > 0) out.pop_back();
    return out;
}}
""")

        # Conclude the C++ deserializer of the type.
        cpp_deser_ops: str = '\n'.join(cpp_deser_ops)
        cpp_deserializer_fns.append(f"""
void deserialize({dt.name}* x, std::istream& s) {{
    bool yep;
    array_meta m;
    {cpp_deser_ops}
}}
""")
        # Conclude the config injection representation of the type.
        all_cinjops = {k: (a, f"""(x.{b} ? "true" : "false")""" if k.endswith('_a') else f"x.{b}")
                       for k, (a, b) in all_cinjops.items()}
        all_cinjops: List[str] = [f"""
out << "{{";
out << "\\"type\\": \\"ConstTypeInjection\\", ";
out << "\\"scope\\": null, ";
out << "\\"root\\": \\"{a}\\", ";
out << "\\"component\\": \\"{k}\\", ";
out << "\\"value\\": \\"" << {b} << "\\"}}" << std::endl;
""".strip() for k, (a, b) in all_cinjops.items()]
        all_cinjops: str = '\n'.join(all_cinjops)
        config_injection_fns.append(f"""
std::string config_injection(const {dt.name}& x) {{
    std::stringstream out;
    {all_cinjops}
    return out.str();
}}
""")

    # Conclude the F90 serializer code.
    for fn in chain(array_serializers.values(), base_serializers):
        _, name, _, _ = singular(children_of_type(fn, Subroutine_Stmt)).children
        proc_names.append(f"{name}")
        append_children(impls, fn)
    iface = singular(p for p in walk(f90_mod, Interface_Block) if find_name_of_node(p) == 'serialize')
    proc_names = Procedure_Stmt(f"module procedure {', '.join(proc_names)}")
    set_children(iface, iface.children[:-1] + [proc_names] + iface.children[-1:])
    f90_code = f90_mod.tofortran()

    # Conclude the C++ serde code.
    forward_decls: str = '\n'.join(f"struct {k};" for k in sdfg_structs.keys())
    struct_defs: Dict[str, str] = {k: '\n'.join(f"{typ} {comp} = {{}};" for comp, typ in sorted(v.items()))
                                   for k, v in sdfg_structs.items()}
    struct_defs: str = '\n'.join(f"""
struct {name} {{
{comps}
}};
""" for name, comps in struct_defs.items())
    # TODO: We don't generate our own structure definitions if the header file contains them (which they should). We can
    #  discard the code to generate them after we are sure that will happen permanently.
    struct_defs: str = f"""
// Forward declarations of structs.
{forward_decls}

// (Re-)definitions of structs.
{struct_defs}
"""
    cpp_deserializer_fns: str = '\n'.join(cpp_deserializer_fns)
    cpp_serializer_fns: str = '\n'.join(cpp_serializer_fns)
    config_injection_fns: str = '\n'.join(config_injection_fns)

    cpp_code = f"""
#ifndef __DACE_SERDE__
#define __DACE_SERDE__

#include <cassert>
#include <istream>
#include <iostream>
#include <sstream>
#include <optional>

#include "{g.name}.h"

namespace serde {{
    std::string scroll_space(std::istream& s) {{
        std::string out;
        while (!s.eof() && (!s.peek() || isspace(s.peek()))) {{
            out += s.get();
            assert(s.good());
        }}
        return out;
    }}

    std::string read_line(std::istream& s, const std::optional<std::string>& should_contain = {{}}) {{
        if (s.eof()) return "<eof>";
        scroll_space(s);
        char bin[101];
        s.getline(bin, 100);
        assert(s.good());
        if (should_contain) {{
            bool ok = (std::string(bin).find(*should_contain) != std::string::npos);
            if (!ok) {{
                std::cerr << "Expected: '" << *should_contain << "'; got: '" << bin << "'" << std::endl;
                exit(EXIT_FAILURE);
            }}
        }}
        return {{bin}};
    }}

    struct array_meta;
    std::map<void*, array_meta>* ARRAY_META_DICT();

    struct array_meta {{
        int rank = 0;
        std::vector<int> size, lbound;

        int volume() const {{  return std::reduce(size.begin(), size.end(), 1, std::multiplies<int>()) ; }}

        template<typename T> T* read(std::istream& s) const;
    }};
    std::map<void*, array_meta>* ARRAY_META_DICT() {{
        static auto* M = new std::map<void*, array_meta>();
        return M;
    }}

    template<typename T>
    void read_scalar(T& x, std::istream& s) {{
        if (s.eof()) return;
        scroll_space(s);
        s >> x;
    }}

    void read_scalar(float& x, std::istream& s) {{
        if (s.eof()) return;
        scroll_space(s);
        long double y;
        s >> y;
        x = y;
    }}

    void read_scalar(double& x, std::istream& s) {{
        if (s.eof()) return;
        scroll_space(s);
        long double y;
        s >> y;
        x = y;
    }}

    void read_scalar(bool& x, std::istream& s) {{
        char c;
        read_scalar(c, s);
        assert (c == '1' or c == '0');
        x = (c == '1');
    }}

    array_meta read_array_meta(std::istream& s) {{
        array_meta m;
        read_line(s, {{"# rank"}});  // Should contain '# rank'
        read_scalar(m.rank, s);
        m.size.resize(m.rank);
        m.lbound.resize(m.rank);
        read_line(s, {{"# size"}});  // Should contain '# size'
        for (int i=0; i<m.rank; ++i) {{
            read_scalar(m.size[i], s);
        }}
        read_line(s, {{"# lbound"}});  // Should contain '# lbound'
        for (int i=0; i<m.rank; ++i) {{
            read_scalar(m.lbound[i], s);
        }}
        return m;
    }}

    template<typename T>
    std::pair<array_meta, T*> read_array(std::istream& s) {{
        auto m = serde::read_array_meta(s);
        auto* y = m.read<T>(s);
        return {{m, y}};
    }}

    {cpp_deserializer_fns}
    {cpp_serializer_fns}
    {config_injection_fns}

    template<typename T>
    T* array_meta::read(std::istream& s) const {{
        read_line(s, {{"# entries"}});
        auto* buf = new T[volume()];
        for (int i=0; i<volume(); ++i) {{
            deserialize(&buf[i], s);
        }}
        (*ARRAY_META_DICT())[buf] = *this;
        return buf;
    }}
}}  // namesepace serde

#endif // __DACE_SERDE__
"""

    return SerdeCode(f90_serializer=f90_code.strip(), cpp_serde=cpp_code.strip())


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
    argp.add_argument('-e', '--entry_points', type=str, required=False, action='append', default=[],
                      help='The dot-delimited entry points for the SDFG (empty means all possible entry points).')
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

    print(f"Will be using the following entry points for pruning (empty means all): {args.entry_points}")
    entry_points = [tuple(ep.split('.')) for ep in args.entry_points]

    cfg = ParseConfig(sources=input_f90s, entry_points=entry_points)
    ast = create_fparser_ast(cfg)
    ast = run_fparser_transformations(ast, cfg)
    serde_code = generate_serde_code(ast, g)

    if args.out_f90:
        with open(args.out_f90, 'w') as f:
            f.write(serde_code.f90_serializer)
    else:
        print(f"=== F90 SERIALIZER CODE BEGINS ===")
        print(serde_code.f90_serializer)
        print(f"=== F90 SERIALIZER CODE ENDS ===")

    if args.out_cpp:
        with open(args.out_cpp, 'w') as f:
            f.write(serde_code.cpp_serde)
    else:
        print(f"=== C++ SERDE CODE BEGINS ===")
        print(serde_code.cpp_serde)
        print(f"=== C++ SERDE CODE ENDS ===")


if __name__ == "__main__":
    main()
