import re
import subprocess
from dataclasses import dataclass
from itertools import chain, combinations
from pathlib import Path
from typing import Generator, Dict, Tuple, List, Optional, Union, Any

from fparser.api import get_reader
from fparser.two.Fortran2003 import (
    Module,
    Derived_Type_Stmt,
    Module_Subprogram_Part,
    Data_Component_Def_Stmt,
    Procedure_Stmt,
    Interface_Block,
    Program,
    Intrinsic_Type_Spec,
    Dimension_Component_Attr_Spec,
    Declaration_Type_Spec,
    Private_Components_Stmt,
    Component_Part,
    Derived_Type_Def,
    Subroutine_Subprogram,
    Subroutine_Stmt,
    Main_Program,
    Function_Subprogram,
    Use_Stmt,
    Name,
    Only_List,
    Rename,
    Generic_Spec,
    Specification_Part,
    Entity_Decl,
)
from fparser.two.utils import walk

import dace
from dace import SDFG
from dace.frontend.fortran.ast_desugaring import (
    identifier_specs,
    append_children,
    set_children,
    SPEC_TABLE,
    SPEC,
    find_name_of_node,
    alias_specs,
    remove_self,
    find_scope_spec,
    GLOBAL_DATA_OBJ_NAME,
    GLOBAL_DATA_TYPE_NAME,
    find_type_of_entity,
)
from dace.frontend.fortran.ast_utils import singular, children_of_type, atmost_one

NEW_LINE = "NEW_LINE('A')"


@dataclass(frozen=True)
class SerdeCode:
    f90_serializer: str
    cpp_serde: str


@dataclass(frozen=True)
class DerivedTypeInfo:
    name: str
    tdef: Derived_Type_Def
    spec: SPEC


@dataclass(frozen=True)
class ComponentInfo:
    name: str
    type: Union[Intrinsic_Type_Spec, Declaration_Type_Spec]
    ptr: bool
    alloc: bool
    rank: int
    shape: Any


def gen_f90_serde_module_skeleton(mod_name: str = "serde") -> Module:
    return Module(
        get_reader(f"""
module {mod_name}
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
    logical, intent(in) :: asis
    if (asis) then
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
    logical :: asis_local
    asis_local = .false.
    if (present(asis)) asis_local = asis
    open (NEWUNIT=io, FILE=cat(prefix, asis_local), STATUS="replace", ACTION="write")
  end function at

  subroutine W_string(io, x, cleanup, nline)
    integer :: io
    character(len=*), intent(in) :: x
    integer :: i, xend
    logical, optional, intent(in) :: cleanup, nline
    logical :: cleanup_local, nline_local
    cleanup_local = .true.
    nline_local = .true.
    if (present(cleanup)) cleanup_local = cleanup
    if (present(nline)) nline_local = nline
    xend = len(x)
    do i = 1, len(x)
      if (x(i:i) == char(0)) then
        xend = i - 1
        exit
      end if
    end do
    write (io, '(A)', advance='no') trim(x(1:xend))
    if (nline_local)  write (io, '(g0)', advance='no') {NEW_LINE}
    if (cleanup_local) close(UNIT=io)
  end subroutine W_string
end module {mod_name}
""")
    )


def gen_base_type_serializer(
    typ: str, kind: Optional[int] = None
) -> Subroutine_Subprogram:
    assert typ in {"logical", "integer", "real"}
    if typ == "logical":
        assert kind is None
    elif typ == "integer":
        assert kind in {1, 2, 4, 8}
    elif typ == "real":
        assert kind in {4, 8}
    fn_name = f"W_{typ}{kind or ''}"
    kind = f"(kind={kind})" if kind else ""
    if typ == "logical":
        op = "\n".join(["y = merge(1, 0, x)", "write (io, '(g0)', advance='no') y"])
    elif typ == "real":
        op = "write (buf, '(e28.20)') x; write (io, '(A)', advance='no') trim(adjustl(buf))"
    else:
        op = "write (io, '(g0)', advance='no') x"

    return Subroutine_Subprogram(
        get_reader(f"""
subroutine {fn_name}(io, x, cleanup, nline)
  character(len=50) :: buf
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
""")
    )


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
""".strip().split("\n")


def generate_pointer_meta_f90(
    ptr: str, rank: int, candidates: Dict[str, Tuple]
) -> List[str]:
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
                    subsc_str.append(":")
                    subsc_str_serialized.append(
                        'call serialize(io, ":", cleanup=.false., nline=.false.)'
                    )
                    continue
                subsc_str.append(f"kmeta_{k}")
                subsc_str_serialized.append(
                    f"call serialize(io, kmeta_{k}, cleanup=.false., , nline=.false.)"
                )
                ops.append(f"do kmeta_{k} = lbound({c}, {k + 1}), ubound({c}, {k + 1})")
            end_dos = ["end do"] * len(ops)
            subsc_str = ", ".join(subsc_str)
            subsc_str_serialized = (
                '\n call serialize(io, ",", cleanup=.false., , nline=.false.) \n'.join(
                    subsc_str_serialized
                )
            )
            ops.append(f"""
if (associated({ptr}, {c}({subsc_str}))) then
  kmeta = 1
  call serialize(io, "=> {c}(", cleanup=.false., , nline=.false.)
  {subsc_str_serialized}
  call serialize(io, "))", cleanup=.false.)
end if
""")
            ops.extend(end_dos)
            sub_checks.append("\n".join(ops))
        cand_checks.append("\n".join(sub_checks))

    cand_checks: str = "\n".join(cand_checks)
    return f"""
if (associated({ptr})) then
  kmeta = 0
  {cand_checks}
  call serialize(io, "# missing", cleanup=.false.)
  call serialize(io, (kmeta == 0), cleanup=.false.)
  ! We are dumping the pointer content anyway for now.
  call serialize(io, {ptr}, cleanup=.false.)
end if
""".strip().split("\n")


def generate_array_serializer_f90(
    dtyp: str, rank: int, tag: str, use: Optional[str] = None
) -> Subroutine_Subprogram:
    iter_vars = ", ".join([f"k{k}" for k in range(1, rank + 1)])
    decls = f"""
{dtyp}, intent(in) :: x({", ".join([":"] * rank)})
integer :: k, kmeta, {iter_vars}
"""
    loop_ops = []
    for k in reversed(range(1, rank + 1)):
        loop_ops.append(f"do k{k} = lbound(x, {k}), ubound(x, {k})")
    loop_ops.append(f"call serialize(io, x({iter_vars}), cleanup=.false.)")
    loop_ops.extend(["end do"] * rank)
    loop = "\n".join(loop_ops)
    meta_ops = generate_array_meta_f90("x", rank)
    meta = "\n".join(meta_ops)
    fn_name = f"W_{tag}_R_{rank}"

    return Subroutine_Subprogram(
        get_reader(
            f"""
subroutine {fn_name}(io, x, cleanup, nline, meta)
  {use or ""}
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
""".strip()
        )
    )


def iterate_over_derived_types(
    ident_map: SPEC_TABLE,
) -> Generator[DerivedTypeInfo, None, None]:
    """Iterate over the derived types in the given map of identifiers to their AST nodes."""
    for tspec, dt in ident_map.items():
        if not isinstance(dt, Derived_Type_Stmt):
            continue
        assert len(tspec) == 2
        _, dtname, _ = dt.children
        dtdef = dt.parent
        yield DerivedTypeInfo(f"{dtname}", dtdef, tspec)


def iterate_over_public_components(
    dt: DerivedTypeInfo,
) -> Generator[ComponentInfo, None, None]:
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
            ptr = "POINTER" in f"{attrs}" if attrs else False
            alloc = "ALLOCATABLE" in f"{attrs}" if attrs else False
            dims = (
                atmost_one(
                    a
                    for a in attrs.children
                    if isinstance(a, Dimension_Component_Attr_Spec)
                )
                if attrs
                else None
            )
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
                    if dtyp == "DOUBLE PRECISION":
                        dtyp, kind = "REAL", "(KIND = 8)"
                    if kind is None:
                        assert dtyp in {"INTEGER", "REAL", "LOGICAL", "CHAR"}, (
                            f"Unexpected type: {dtyp}"
                        )
                        DEFAULT_KINDS = {
                            "INTEGER": "(KIND = 4)",
                            "REAL": "(KIND = 4)",
                        }
                        kind = DEFAULT_KINDS.get(dtyp)
                    ctyp = Intrinsic_Type_Spec(f"{dtyp}{kind or ''}")
                else:
                    assert isinstance(ctyp, Declaration_Type_Spec)

                yield ComponentInfo(f"{cname}", ctyp, ptr, alloc, rank, shape)


def _keep_only_derived_types(ast: Program) -> Program:
    for x in reversed(
        walk(
            ast,
            (Main_Program, Interface_Block, Subroutine_Subprogram, Function_Subprogram),
        )
    ):
        remove_self(x)
    ident_map = identifier_specs(ast)
    aliases = set(ident_map.keys())
    for olist in walk(ast, Only_List):
        use = olist.parent
        assert isinstance(use, Use_Stmt)
        mod_name = singular(children_of_type(use, Name)).string
        mod_spec = (mod_name,)
        scope_spec = find_scope_spec(use)
        for c in olist.children:
            assert isinstance(c, (Name, Rename, Generic_Spec))
            if isinstance(c, (Name, Generic_Spec)):
                src, tgt = c, c
            else:
                assert isinstance(c, Rename)
                _, src, tgt = c.children
            src, tgt = f"{src}", f"{tgt}"
            src_spec, tgt_spec = scope_spec + (src,), mod_spec + (tgt,)
            if tgt_spec in aliases:
                aliases.add(src_spec)
            else:
                remove_self(c)
    return ast


def _real_ctype(v: dace.data.Data):
    if isinstance(v, dace.data.Scalar):
        return f"{v.ctype}"
    elif isinstance(v, dace.data.Array):
        return f"{v.ctype}*"
    elif isinstance(v, dace.data.Structure):
        return f"{v.ctype}"
    else:
        raise NotImplementedError


def _get_basic_serializers() -> List[Subroutine_Subprogram]:
    return [
        gen_base_type_serializer("logical"),
        gen_base_type_serializer("integer", 1),
        gen_base_type_serializer("integer", 2),
        gen_base_type_serializer("integer", 4),
        gen_base_type_serializer("integer", 8),
        gen_base_type_serializer("real", 4),
        gen_base_type_serializer("real", 8),
    ]


def strip_last_int(x: str) -> str:
    return "_".join(x.split("_")[:-1]) if x.startswith("__f2dace_") else x


def _generate_serde_for_pointer_component(
    z: ComponentInfo,
    array_map: Dict,
    all_sa_vars: Dict[str, str],
    all_soa_vars: Dict[str, str],
) -> Tuple[List[str], List[str], List[str]]:
    f90_ser_ops: List[str] = []
    cpp_ser_ops: List[str] = []
    cpp_deser_ops: List[str] = []

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
    candidates = {
        f"x%{v[0]}": v[1].children
        for k, v in array_map.items()
        if k[0] == f"{z.type}" and z.rank <= k[1]
    }
    f90_ser_ops.extend(generate_pointer_meta_f90(f"x%{z.name}", z.rank, candidates))
    cpp_ser_ops.append(f"""
if (x->{z.name}) add_line(serialize_array(x->{z.name}), s);
""")

    # TODO: pointer types have a whole bunch of different, best-effort strategies. For our purposes, we will
    #  only populate this when it points to a different component of the same structure.
    cpp_deser_ops.append(f"""
read_line(s, {{"# assoc"}});  // Should contain '# assoc'
deserialize(&yep, s);
""")
    if z.rank:
        sa_vars = [
            all_sa_vars[f"__f2dace_SA_{z.name}_d_{dim}_s"] for dim in range(z.rank)
        ]
        sa_vars = "\n".join(
            [f"x->{v} = m.size.at({dim});" for dim, v in enumerate(sa_vars)]
        )
        soa_vars = [
            all_soa_vars[f"__f2dace_SOA_{z.name}_d_{dim}_s"] for dim in range(z.rank)
        ]
        soa_vars = "\n".join(
            [f"x->{v} = m.lbound.at({dim});" for dim, v in enumerate(soa_vars)]
        )
    else:
        sa_vars, soa_vars = [], []
    cpp_deser_ops.append(f"""
if (yep) {{
    auto [m, arr] = read_pointer<std::remove_pointer<decltype(x ->{z.name})>::type>(s);
    {sa_vars}
    {soa_vars}
    x->{z.name} = arr;
}}
""")
    return f90_ser_ops, cpp_ser_ops, cpp_deser_ops


def _generate_serde_for_array_component(
    z: ComponentInfo,
    dt: DerivedTypeInfo,
    sdfg_structs: Dict[str, Dict[str, str]],
    all_sa_vars: Dict[str, str],
    all_soa_vars: Dict[str, str],
) -> Tuple[List[str], List[str], List[str]]:
    f90_ser_ops: List[str] = []
    cpp_ser_ops: List[str] = []
    cpp_deser_ops: List[str] = []

    f90_ser_ops.extend(generate_array_meta_f90(f"x%{z.name}", z.rank))
    f90_ser_ops.append(
        f"call serialize(io, x%{z.name}, cleanup=.false., nline=.true., meta=.false.)"
    )
    assert "***" not in sdfg_structs[dt.name][z.name]
    ptrptr = "**" in sdfg_structs[dt.name][z.name]
    if z.alloc:
        sa_vars = [
            all_sa_vars[f"__f2dace_SA_{z.name}_d_{dim}_s"] for dim in range(z.rank)
        ]
        sa_vars = "\n".join(
            [f"x->{v} = m.size[{dim}];" for dim, v in enumerate(sa_vars)]
        )
        soa_vars = [
            all_soa_vars[f"__f2dace_SOA_{z.name}_d_{dim}_s"] for dim in range(z.rank)
        ]
        soa_vars = "\n".join(
            [f"x->{v} = m.lbound[{dim}];" for dim, v in enumerate(soa_vars)]
        )
    else:
        sa_vars, soa_vars = "", ""
    cpp_ser_ops.append(f"""
{{
    const array_meta& m = ARRAY_META_DICT_AT(x->{z.name});
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
    return f90_ser_ops, cpp_ser_ops, cpp_deser_ops


def _generate_serde_for_scalar_or_struct_component(
    z: ComponentInfo, dt: DerivedTypeInfo, sdfg_structs: Dict[str, Dict[str, str]]
) -> Tuple[List[str], List[str], List[str]]:
    f90_ser_ops: List[str] = []
    cpp_ser_ops: List[str] = []
    cpp_deser_ops: List[str] = []
    if "*" in sdfg_structs[dt.name][z.name]:
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
    return f90_ser_ops, cpp_ser_ops, cpp_deser_ops


def _generate_serde_for_non_pointer_component(
    z: ComponentInfo,
    dt: DerivedTypeInfo,
    sdfg_structs: Dict[str, Dict[str, str]],
    all_sa_vars: Dict[str, str],
    all_soa_vars: Dict[str, str],
) -> Tuple[List[str], List[str], List[str]]:
    f90_ser_ops: List[str] = []
    cpp_ser_ops: List[str] = []
    cpp_deser_ops: List[str] = []

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
        comp_f90_ser_ops, comp_cpp_ser_ops, comp_cpp_deser_ops = (
            _generate_serde_for_array_component(
                z, dt, sdfg_structs, all_sa_vars, all_soa_vars
            )
        )
    else:
        comp_f90_ser_ops, comp_cpp_ser_ops, comp_cpp_deser_ops = (
            _generate_serde_for_scalar_or_struct_component(z, dt, sdfg_structs)
        )

    f90_ser_ops.extend(comp_f90_ser_ops)
    cpp_ser_ops.extend(comp_cpp_ser_ops)
    cpp_deser_ops.extend(comp_cpp_deser_ops)

    if z.alloc:
        f90_ser_ops.append("end if  ! CONCLUDING IF")
        cpp_ser_ops.append(f"""}}  // CONCLUDING IF""")
        cpp_deser_ops.append(f"""}}  // CONCLUDING IF""")

    return f90_ser_ops, cpp_ser_ops, cpp_deser_ops


def _ensure_array_serializer_exists(
    z: ComponentInfo,
    dt: DerivedTypeInfo,
    array_serializers: Dict[Tuple[str, int], Subroutine_Subprogram],
    ident_map: SPEC_TABLE,
):
    """
    Checks if a Fortran serializer for an array of this component's type
    and rank exists, and creates it if it doesn't.
    """
    if not z.rank:
        return

    if isinstance(z.type, Intrinsic_Type_Spec):
        use = None
    elif isinstance(z.type, Declaration_Type_Spec):
        _, ctyp = z.type.children
        ctyp = f"{ctyp}"
        # TODO: Don't rely on unique names.
        if (dt.spec[0], ctyp) in ident_map:
            use = f"use {dt.spec[0]}, only: {ctyp}"
        else:
            mod = singular(
                k[0] for k in ident_map.keys() if len(k) == 2 and k[-1] == ctyp
            )
            use = f"use {mod}, only: {ctyp}"

    tag = (
        f"{z.type}".replace("TYPE(", "dt_")
        .replace("(KIND =", "_")
        .replace("(LEN =", "_")
        .replace(" ", "_")
        .replace(")", "")
        .lower()
    )

    if (tag, z.rank) not in array_serializers:
        array_serializers[(tag, z.rank)] = generate_array_serializer_f90(
            f"{z.type}", z.rank, tag, use
        )


def _generate_serde_for_component(
    z: ComponentInfo,
    dt: DerivedTypeInfo,
    sdfg_structs: Dict[str, Dict[str, str]],
    array_serializers: Dict[Tuple[str, int], Subroutine_Subprogram],
    ident_map: SPEC_TABLE,
    array_map: Dict,
    all_sa_vars: Dict[str, str],
    all_soa_vars: Dict[str, str],
) -> Optional[Tuple[List[str], List[str], List[str]]]:
    if z.name not in sdfg_structs[dt.name]:
        # The component is not present in the final SDFG, so we don't care for it.
        return None

    f90_ser_ops: List[str] = [f"call serialize(io , '# {z.name}', cleanup=.false.)"]
    cpp_ser_ops: List[str] = [f"""add_line("# {z.name}", s);"""]
    cpp_deser_ops: List[str] = [
        f"""read_line(s, {{"# {z.name}"}});  // Should contain '# {z.name}'"""
    ]

    # Ensure a Fortran array serializer exists for this component if it's an array.
    _ensure_array_serializer_exists(z, dt, array_serializers, ident_map)
    if z.ptr:
        comp_f90_ser_ops, comp_cpp_ser_ops, comp_cpp_deser_ops = (
            _generate_serde_for_pointer_component(
                z, array_map, all_sa_vars, all_soa_vars
            )
        )
    else:
        comp_f90_ser_ops, comp_cpp_ser_ops, comp_cpp_deser_ops = (
            _generate_serde_for_non_pointer_component(
                z, dt, sdfg_structs, all_sa_vars, all_soa_vars
            )
        )

    f90_ser_ops.extend(comp_f90_ser_ops)
    cpp_ser_ops.extend(comp_cpp_ser_ops)
    cpp_deser_ops.extend(comp_cpp_deser_ops)

    return f90_ser_ops, cpp_ser_ops, cpp_deser_ops


def _get_global_data_serde_code(ast: Program, g: SDFG) -> SerdeCode:
    alias_map = alias_specs(ast)
    uses, ser_ops, ser_ops_cpp, des_ops, consistent_ops_cpp = [], [], [], [], []
    if GLOBAL_DATA_OBJ_NAME in g.arrays:
        sdfg_structs: Dict[str, dace.data.Structure] = {
            v.name: v for k, v in g.arrays.items() if isinstance(v, dace.data.Structure)
        }
        all_sa_vars: Dict[str, str] = {
            strip_last_int(z): z
            for z in sdfg_structs[GLOBAL_DATA_TYPE_NAME].keys()
            if z.startswith("__f2dace_SA_")
        }
        all_soa_vars: Dict[str, str] = {
            strip_last_int(z): z
            for z in sdfg_structs[GLOBAL_DATA_TYPE_NAME].keys()
            if z.startswith("__f2dace_SOA_")
        }
        gdata = g.arrays[GLOBAL_DATA_OBJ_NAME].members
        for mod in walk(ast, Module):
            mname = find_name_of_node(mod)
            spart = atmost_one(children_of_type(mod, Specification_Part))
            if not spart:
                continue
            for var in walk(spart, Entity_Decl):
                vname = find_name_of_node(var)
                assert vname
                if vname not in gdata:
                    continue
                renamed: re.Match = re.match(r"^([a-zA-Z0-9_]+)_var_[0-9]+$", vname)
                ogvname = renamed.group(1) if renamed else vname

                uses.append(f"use {mname}, only : {vname} => {ogvname}")
                ser_ops.append(f"""
call serialize(io, "# {vname}", cleanup=.false.)
call serialize(io, {vname}, cleanup=.false.)
""")
                vtype = find_type_of_entity(var, alias_map)
                vctyp = _real_ctype(gdata[vname])
                if isinstance(gdata[vname], dace.data.Array):
                    ser_ops_cpp.append(f"""
add_line(serialize_array(g->{vname}), s);
""")
                    # NOTE: Arrays are not to be included in the `consistent_ops`.
                    rank = len(vtype.shape)
                    sa_vars = [
                        all_sa_vars.get(f"__f2dace_SA_{vname}_d_{dim}_s")
                        for dim in range(rank)
                    ]
                    sa_vars = "\n".join(
                        [
                            f"g->{v} = m.size.at({dim});"
                            for dim, v in enumerate(sa_vars)
                            if v
                        ]
                    )
                    soa_vars = [
                        all_soa_vars.get(f"__f2dace_SOA_{vname}_d_{dim}_s")
                        for dim in range(rank)
                    ]
                    soa_vars = "\n".join(
                        [
                            f"g->{v} = m.lbound.at({dim});"
                            for dim, v in enumerate(soa_vars)
                            if v
                        ]
                    )
                    des_ops.append(f"""
{{
    read_line(s, "# {vname}");
    auto [m, arr] = read_array<{vctyp[:-1]}>(s);
    g->{vname} = arr;
    {sa_vars}
    {soa_vars}
}}
""")
                else:
                    ser_ops_cpp.append(f"""
add_line(serialize(g->{vname}), s);
""")
                    if vtype.spec == ("LOGICAL",):
                        vname_val = f"""(g->{vname} ? ".true." : ".false.")"""
                    else:
                        vname_val = f"""serialize(g->{vname})"""
                    consistent_ops_cpp.append(f"""
consistent["{mname}.{vname}"].insert({vname_val});
""")
                    des_ops.append(f"""
read_line(s, "# {vname}");
deserialize(g->{vname}, s);
""")

    uses = "\n".join(uses)
    ser_ops = "\n".join(ser_ops)
    f90_code = f"""
subroutine serialize_global_data(io)
{uses}
integer :: io
{ser_ops}
close(UNIT=io)
end subroutine serialize_global_data
"""

    des_ops = "\n".join(des_ops)
    ser_ops_cpp = "\n".join(ser_ops_cpp)
    consistent_ops_cpp = "\n".join(consistent_ops_cpp)
    cpp_code = f"""
void deserialize_global_data({GLOBAL_DATA_TYPE_NAME}* g, std::istream& s) {{
    {des_ops}
}}

std::string serialize_global_data(const {GLOBAL_DATA_TYPE_NAME}* g) {{
    std::stringstream s;
    {ser_ops_cpp}
    return s.str();
}}

enum class SerializationType {{ INVALID, PLAIN, CONST_INJECTION, F90_MODULE }};

std::string serialize_consistent_global_data(
    std::vector<const {GLOBAL_DATA_TYPE_NAME}*>& gs,
    SerializationType serialization_type = SerializationType::INVALID)
{{
    assert(serialization_type != SerializationType::INVALID);
    if (gs.empty()) return "";

    std::map<std::string, std::set<std::string>> consistent;
    for (const auto* g : gs) {{
        {consistent_ops_cpp}
    }}

    std::stringstream s;
    if (serialization_type == SerializationType::F90_MODULE) {{
        s << R"(
module global_data_assertion
contains
subroutine assert_global_data()
)";
        for (const auto& [k, vs] : consistent) {{
            std::vector<std::string_view> parts = split(k, '.');
            assert(parts.size() == 2);
            const auto mname = std::string_view(parts[0]);
            s << "use " << mname << std::endl;
        }}
        s << R"(
implicit none
)";
    }}
    for (const auto& [k, vs] : consistent) {{
        if (vs.size() != 1) continue;
        const auto& v = *vs.begin();
        if (serialization_type == SerializationType::PLAIN) {{
            s << k << " = " << v << std::endl;
        }} else if (serialization_type == SerializationType::CONST_INJECTION) {{
            const std::string vval = (v == ".true." ? "1" : (v == ".false." ? "0" : v));
            s << R"({{ "type": "ConstInstanceInjection", "scope": null, )";
            s << R"("root": ")" << k << R"(", "component": null, "value": ")" << vval << R"(" }})" << std::endl;
        }} else if (serialization_type == SerializationType::F90_MODULE) {{
            std::vector<std::string_view> parts = split(k, '.');
            assert(parts.size() == 2);
            const auto vname = std::string_view(parts[1]);
            const auto neqop = (v == ".true." || v == ".false.") ? " .neqv. " : " .ne. ";
            s << "if (" << vname << neqop << v << ") then" << std::endl;
            s << R"(\tprint *, "mismatched )" << vname << "; want " << v << R"(, got: ", )" << vname << std::endl;
            s << "\tcall abort"<< std::endl << "endif" << std::endl;
        }}
    }}
    if (serialization_type == SerializationType::F90_MODULE) {{
        s << R"(
end subroutine assert_global_data
end module global_data_assertion
)";
    }}
    return s.str();
}}
"""
    return SerdeCode(f90_serializer=f90_code, cpp_serde=cpp_code)


def _get_sdfg_structs(g: SDFG) -> Dict[str, Dict[str, str]]:
    """
    Recursively finds all DaCe struct definitions in the SDFG and returns a
    map of struct names to their members (name and C-type).
    """
    sdfg_structs: Dict[str, dace.data.Structure] = {
        v.name: v for k, v in g.arrays.items() if isinstance(v, dace.data.Structure)
    }
    sdfg_structs_from_arrays: Dict[str, dace.data.Structure] = {
        v.stype.name: v.stype
        for k, v in g.arrays.items()
        if isinstance(v, dace.data.ContainerArray)
        and isinstance(v.stype, dace.data.Structure)
    }
    sdfg_structs.update(sdfg_structs_from_arrays)

    while True:
        new_sdfg_structs: Dict[str, dace.data.Structure] = {
            m.name: m
            for _, v in sdfg_structs.items()
            for _, m in v.members.items()
            if isinstance(m, dace.data.Structure) and m.name not in sdfg_structs
        }
        new_sdfg_structs_from_arrays: Dict[str, dace.data.Structure] = {
            m.stype.name: m.stype
            for _, v in sdfg_structs.items()
            for _, m in v.members.items()
            if isinstance(m, dace.data.ContainerArray)
            and isinstance(m.stype, dace.data.Structure)
            and m.stype.name not in sdfg_structs
        }
        new_sdfg_structs.update(new_sdfg_structs_from_arrays)
        if not new_sdfg_structs:
            break
        sdfg_structs.update(new_sdfg_structs)

    sdfg_structs_with_members: Dict[str, List[Tuple[str, dace.data.Data]]] = {
        k: [(kk, vv) for kk, vv in v.members.items()] for k, v in sdfg_structs.items()
    }

    return {
        k: {kk: _real_ctype(vv) for kk, vv in v}
        for k, v in sdfg_structs_with_members.items()
    }


def _get_cpp_base_deserializers() -> List[str]:
    return [
        f"""
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
void deserialize(float& x, std::istream& s) {{
    read_scalar(x, s);
}}
void deserialize(double& x, std::istream& s) {{
    read_scalar(x, s);
}}
void deserialize(long double& x, std::istream& s) {{
    read_scalar(x, s);
}}
void deserialize(int& x, std::istream& s) {{
    read_scalar(x, s);
}}
void deserialize(long& x, std::istream& s) {{
    read_scalar(x, s);
}}
void deserialize(long long& x, std::istream& s) {{
    read_scalar(x, s);
}}
void deserialize(bool& x, std::istream& s) {{
    read_scalar(x, s);
}}
"""
    ]


def _get_cpp_base_serializers() -> List[str]:
    return [
        f"""
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
    s << std::setprecision(20) << x;
    if (trailing_newline) s << std::endl;
}}
void add_line(bool x, std::ostream& s, bool trailing_newline=true) {{
    add_line(int(x), s, trailing_newline);
}}
template<typename T>
std::string serialize(const T* x) {{
    if constexpr (std::is_pointer_v<T>) {{
        return serialize(*x);
    }} else {{
        std::stringstream s;
        add_line(*x, s, false);
        return s.str();
    }}
}}
std::string serialize(int x) {{
    std::stringstream s;
    s << x;
    return s.str();
}}
std::string serialize(long x) {{
    std::stringstream s;
    s << x;
    return s.str();
}}
std::string serialize(long long x) {{
    std::stringstream s;
    s << x;
    return s.str();
}}
std::string serialize(float x) {{
    std::stringstream s;
    s << std::setprecision(20) << x;
    return s.str();
}}
std::string serialize(double x) {{
    std::stringstream s;
    s << std::setprecision(20) << x;
    return s.str();
}}
std::string serialize(long double x) {{
    std::stringstream s;
    s << std::setprecision(20) << x;
    return s.str();
}}
std::string serialize(bool x) {{
    return serialize(int(x));
}}
"""
    ]


def _generate_f90_c_glue_code(
    ast: Program, sdfg_structs: Dict[str, Dict[str, str]], mod_name: str
) -> Module:
    glue_uses = []
    ident_map = identifier_specs(ast)
    for dt in iterate_over_derived_types(ident_map):
        if dt.name not in sdfg_structs:
            # The type is not present in the final SDFG, so we don't care for it.
            continue
        use = f"use {dt.spec[0]}, only: {dt.name}"
        if use not in glue_uses:
            glue_uses.append(use)
        for z in iterate_over_public_components(dt):
            if z.name not in sdfg_structs[dt.name]:
                # The component is not present in the final SDFG, so we don't care for it.
                continue
            if z.rank:
                if isinstance(z.type, Declaration_Type_Spec):
                    _, ctyp = z.type.children
                    ctyp = f"{ctyp}"
                    # TODO: Don't rely on unique names.
                    if (dt.spec[0], ctyp) in ident_map:
                        use = f"use {dt.spec[0]}, only: {ctyp}"
                    else:
                        mod = singular(
                            k[0]
                            for k in ident_map.keys()
                            if len(k) == 2 and k[-1] == ctyp
                        )
                        use = f"use {mod}, only: {ctyp}"
                    if use not in glue_uses:
                        glue_uses.append(use)
    glue_uses = "\n".join(glue_uses)

    def f90_type(tx: str):
        if tx == "int":
            return f"INTEGER(c_int)"
        elif tx == "float":
            return f"REAL(c_float)"
        elif tx == "double":
            return f"REAL(c_double)"
        elif tx.endswith("*"):
            return f"TYPE(c_ptr)"
        else:
            return f"TYPE({tx})"

    f90_struct_defs: Dict[str, str] = {
        k: "\n".join(f"{f90_type(typ)} :: m_{comp}" for comp, typ in sorted(v.items()))
        for k, v in sdfg_structs.items()
    }
    f90_struct_defs: str = "\n".join(
        f"""
type, bind(C) :: glue_{name}
{comps}
end type glue_{name}
"""
        for name, comps in f90_struct_defs.items()
    )

    def f90_to_c_glue(tx: str, nx: str, dtname: str):
        if nx.startswith("__f2dace_"):
            return "", ""
        elif tx in {"int", "float", "double"}:
            return f"out % m_{nx} = inp % {nx}", ""
        elif tx.endswith("*"):
            nx_sas: Dict[str, str] = {
                strip_last_int(z).removesuffix("_s").removeprefix("__f2dace_SA_"): z
                for z in sdfg_structs[dtname].keys()
                if z.startswith("__f2dace_SA_")
            }
            nx_soas: Dict[str, str] = {
                strip_last_int(z).removesuffix("_s").removeprefix("__f2dace_SOA_"): z
                for z in sdfg_structs[dtname].keys()
                if z.startswith("__f2dace_SOA_")
            }
            nx_sas: Dict[tuple, str] = {
                tuple(k.split("_d_")): v
                for k, v in nx_sas.items()
                if k.split("_d_")[0] == nx
            }
            nx_soas: Dict[tuple, str] = {
                tuple(k.split("_d_")): v
                for k, v in nx_soas.items()
                if k.split("_d_")[0] == nx
            }
            basetx = tx.removesuffix("*")
            if basetx in sdfg_structs:
                declx = f"{f90_type('glue_' + basetx)}, allocatable, target :: a_{nx}"
                initx = [f"if (initalloc) allocate(a_{nx})"]
            elif nx_sas:
                declx = f"{f90_type(basetx)}, allocatable, target :: a_{nx}({','.join([':'] * len(nx_sas))})"
                initx = [
                    f"if (initalloc) allocate(a_{nx}({','.join(f'size(inp % {nx}, {i + 1})' for i in range(len(nx_sas)))}))"
                ]
            else:
                declx = f"{f90_type(basetx)}, allocatable, target :: a_{nx}(:)"
                initx = [f"if (initalloc) allocate(a_{nx}(size(inp % {nx}))"]
            initx.append(f"out % m_{nx} = c_loc(a_{nx})")
            if basetx.endswith("*"):
                basetx = basetx.removesuffix("*")
                assert basetx in sdfg_structs, (
                    f"We assume only `T**` for now, got {basetx}"
                )
                assert len(nx_sas) == 3, (
                    f"We assume 3D only for now, got {basetx} / {len(nx_sas)}"
                )
                declx = f"""
{declx}
{f90_type(basetx)}, pointer :: pt_{nx}
integer :: i_{nx}, j_{nx}, k_{nx}
"""
                initx.append(f"""
do i_{nx}=lbound(a_{nx}, 1), ubound(a_{nx}, 1)
do j_{nx}=lbound(a_{nx}, 1), ubound(a_{nx}, 1)
do k_{nx}=lbound(a_{nx}, 1), ubound(a_{nx}, 1)
if (initalloc) allocate(pt_{nx})
call c_tor{basetx}(inp % {nx}(i_{nx}, j_{nx}, k_{nx}), a_{nx}(i_{nx}, j_{nx}, k_{nx}))
end do
end do
end do
""")
                pass
            elif basetx in sdfg_structs:
                initx.append(f"call ctor_{basetx}(inp % {nx}, a_{nx}, initalloc)")
            elif nx_sas:
                idx = ",".join([":"] * len(nx_sas))
                initx.append(f"a_{nx}({idx}) = inp % {nx}({idx})")
            else:
                initx.append(f"a_{nx}(:) = inp % {nx}(:)")
            for k, v in nx_sas.items():
                initx.append(f"out % m_{v} = size(inp % {nx}, {int(k[1]) + 1})")
            for k, v in nx_soas.items():
                initx.append(f"out % m_{v} = lbound(inp % {nx}, {int(k[1]) + 1})")
            initx = "\n".join(initx)
            return initx, declx
        else:
            raise ValueError(f"Unknown type: {tx}")

    f90_struct_glue: Dict[str, list[tuple[str, str]]] = {
        k: [f90_to_c_glue(typ, comp, k) for comp, typ in sorted(v.items())]
        for k, v in sdfg_structs.items()
    }
    f90_struct_glue: Dict[str, str] = {
        k: f"""
{"\n".join(x[1] for x in v)}
{"\n".join(x[0] for x in v)}
"""
        for k, v in f90_struct_glue.items()
    }
    f90_glue_ctor_ifcs: str = "\n".join(
        f"module procedure :: ctor_{name}"
        for name in f90_struct_glue.keys()
        if name != "global_data_type"
    )
    f90_glue_ctor_defs: str = "\n".join(
        f"""
subroutine ctor_{name}(inp, out, initalloc)
type({name}), intent(in) :: inp
type(glue_{name}), intent(inout) :: out
logical, intent(in) :: initalloc
{code}
end subroutine ctor_{name}
"""
        for name, code in f90_struct_glue.items()
        if name != "global_data_type"
    )

    f90_glue_mod = Module(
        get_reader(f"""
module f90_glue_{mod_name}
use, intrinsic :: iso_c_binding
{glue_uses}
implicit none
{f90_struct_defs}
interface ctor
{f90_glue_ctor_ifcs}
end interface ctor
contains
{f90_glue_ctor_defs}
end module f90_glue_{mod_name}
""")
    )
    return f90_glue_mod


def _get_basic_array_serializers() -> Dict[Tuple[str, int], Subroutine_Subprogram]:
    array_serializers: Dict[Tuple[str, int], Subroutine_Subprogram] = {}
    # Generate basic array serializers for ranks 1 to 4.
    for rank in range(1, 5):
        typez = [
            "LOGICAL",
            "INTEGER(KIND = 1)",
            "INTEGER(KIND = 2)",
            "INTEGER(KIND = 4)",
            "INTEGER(KIND = 8)",
            "REAL(KIND = 4)",
            "REAL(KIND = 8)",
        ]
        for t in typez:
            tag = (
                f"{t}".replace("TYPE(", "dt_")
                .replace("(KIND =", "_")
                .replace("(LEN =", "_")
                .replace(" ", "_")
                .replace(")", "")
                .lower()
            )
            array_serializers[(tag, rank)] = generate_array_serializer_f90(t, rank, tag)
    return array_serializers


def _generate_serde_for_one_derived_type(
    dt: DerivedTypeInfo,
    sdfg_structs: Dict[str, Dict[str, str]],
    array_serializers: Dict[Tuple[str, int], Subroutine_Subprogram],
    ident_map: SPEC_TABLE,
) -> Tuple[Optional[Subroutine_Subprogram], Optional[str], Optional[str]]:
    if dt.name not in sdfg_structs:
        # The type is not present in the final SDFG, so we don't care for it.
        return None, None, None

    # We need to make a table to find pointer associations later.
    array_map: Dict = {}
    for z in iterate_over_public_components(dt):
        if z.ptr or not z.shape:
            # We are looking for arrays.
            continue
        array_map[(f"{z.type}", z.rank)] = (z.name, z.shape)

    all_sa_vars: Dict[str, str] = {
        strip_last_int(z): z
        for z in sdfg_structs[dt.name].keys()
        if z.startswith("__f2dace_SA_")
    }
    all_soa_vars: Dict[str, str] = {
        strip_last_int(z): z
        for z in sdfg_structs[dt.name].keys()
        if z.startswith("__f2dace_SOA_")
    }

    f90_ser_ops: List[str] = []
    cpp_ser_ops: List[str] = []
    cpp_deser_ops: List[str] = []
    for z in iterate_over_public_components(dt):
        comp_serde = _generate_serde_for_component(
            z,
            dt,
            sdfg_structs,
            array_serializers,
            ident_map,
            array_map,
            all_sa_vars,
            all_soa_vars,
        )
        if comp_serde is None:
            continue

        f90_ser_ops.extend(comp_serde[0])
        cpp_ser_ops.extend(comp_serde[1])
        cpp_deser_ops.extend(comp_serde[2])

    # Conclude the F90 serializer of the type.
    f90_ser_ops_str: str = "\n".join(f90_ser_ops)
    kmetas = ", ".join(f"kmeta_{k}" for k in range(10))
    impl_fn = Subroutine_Subprogram(
        get_reader(
            f"""
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
  {f90_ser_ops_str}
  if (nline_local)  write (io, '(g0)', advance='no') {NEW_LINE}
  if (cleanup_local) close(UNIT=io)
end subroutine W_{dt.name}
""".strip()
        )
    )

    # Conclude the C++ serializer of the type.
    cpp_ser_ops_str: str = "\n".join(cpp_ser_ops)
    cpp_serializer_fn = f"""
std::string serialize(const {dt.name}* x) {{
    std::stringstream s;
    {cpp_ser_ops_str}
    std::string out = s.str();
    if (out.length() > 0) out.pop_back();
    return out;
}}
"""

    # Conclude the C++ deserializer of the type.
    cpp_deser_ops_str: str = "\n".join(cpp_deser_ops)
    cpp_deserializer_fn = f"""
void deserialize({dt.name}* x, std::istream& s) {{
    bool yep;
    array_meta m;
    {cpp_deser_ops_str}
}}
"""
    return impl_fn, cpp_serializer_fn, cpp_deserializer_fn


def _generate_serde_for_derived_types(
    ast: Program,
    sdfg_structs: Dict[str, Dict[str, str]],
    array_serializers: Dict[Tuple[str, int], Subroutine_Subprogram],
    impls: Module_Subprogram_Part,
) -> Tuple[List[str], List[str], List[str]]:
    proc_names = []
    cpp_serializer_fns = []
    cpp_deserializer_fns = []
    ident_map = identifier_specs(ast)

    for dt in iterate_over_derived_types(ident_map):
        f90_sub, cpp_ser, cpp_deser = _generate_serde_for_one_derived_type(
            dt, sdfg_structs, array_serializers, ident_map
        )
        if f90_sub:
            proc_names.append(f"W_{dt.name}")
            append_children(impls, f90_sub)
            cpp_serializer_fns.append(cpp_ser)
            cpp_deserializer_fns.append(cpp_deser)

    return proc_names, cpp_serializer_fns, cpp_deserializer_fns


def _generate_cpp_serde_code(
    mod_name: str,
    sdfg_structs: Dict[str, Dict[str, str]],
    cpp_deserializer_fns: List[str],
    cpp_serializer_fns: List[str],
    gdata: SerdeCode,
) -> str:
    forward_decls: str = "\n".join(f"struct {k};" for k in sdfg_structs.keys())
    struct_defs: Dict[str, str] = {
        k: "\n".join(f"{typ} {comp} = {{}};" for comp, typ in sorted(v.items()))
        for k, v in sdfg_structs.items()
    }
    struct_defs: str = "\n".join(
        f"""
struct {name} {{
{comps}
}};
"""
        for name, comps in struct_defs.items()
    )
    # TODO: We don't generate our own structure definitions if the header file contains them (which they should). We can
    #  discard the code to generate them after we are sure that will happen permanently.
    struct_defs: str = f"""
// Forward declarations of structs.
{forward_decls}

// (Re-)definitions of structs.
{struct_defs}
"""
    cpp_deserializer_fns_str: str = "\n".join(cpp_deserializer_fns)
    cpp_serializer_fns_str: str = "\n".join(cpp_serializer_fns)

    cpp_code = f"""
#ifndef __DACE_{mod_name.upper()}_SERDE__
#define __DACE_{mod_name.upper()}_SERDE__

#include <cassert>
#include <istream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <optional>
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <ranges>
#include <string_view>
#include <numeric>

#include <cxxabi.h>

namespace {mod_name} {{
    template<typename T>
    std::string type_name() {{
        int status = 0;
        std::unique_ptr<char, void(*)(void*)> res {{
            abi::__cxa_demangle(typeid(T).name(), NULL, NULL, &status),
            std::free
        }};
        if (status != 0) throw status; // stub
        return res.get();
    }}

    std::vector<std::string_view> split(std::string_view s, char delim) {{
        std::vector<std::string_view> parts;
        for (int start_pos = 0, next_pos; start_pos < s.length(); start_pos = next_pos + 1) {{
            next_pos = s.find(delim, start_pos);
            if (next_pos == s.npos) {{
                parts.push_back({{s.begin()+start_pos, s.length()-start_pos}});
                break;
            }}
            parts.push_back({{s.begin()+start_pos, static_cast<size_t>(next_pos-start_pos)}});
        }}
        return parts;
    }}

    std::string scroll_space(std::istream& s) {{
        std::string out;
        while (!s.eof() && (!s.peek() || isspace(s.peek()))) {{
            out += s.get();
            assert(s.good());
        }}
        return out;
    }}

    std::string read_line(std::istream& s, const std::optional<std::string>& should_contain = {{}}) {{
        if (s.eof()) {{
            std::cerr << "Got unexpected EOF" << std::endl;
            exit(EXIT_FAILURE);
        }}
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

    std::string read_until(std::istream& s, const std::string& should_contain) {{
        while (!s.eof()) {{
            scroll_space(s);
            char bin[101];
            s.getline(bin, 100);
            assert(s.good());
            bool ok = (std::string(bin).find(should_contain) != std::string::npos);
            if (ok) return {{bin}};
        }}
        std::cerr << "Expected: '" << should_contain << "'; got EOF" << std::endl;
        exit(EXIT_FAILURE);
    }}

    struct array_meta;
    std::pair<std::map<void *, array_meta>*, std::unique_lock<std::mutex>> ARRAY_META_DICT();

    struct array_meta {{
        int rank = 0;
        std::vector<int> size, lbound;

        int volume() const {{  return std::reduce(size.begin(), size.end(), 1, std::multiplies<int>()) ; }}

        template<typename T> T* read(std::istream& s) const;
    }};
    std::pair<std::map<void *, array_meta>*, std::unique_lock<std::mutex>> ARRAY_META_DICT() {{
      static auto *M = new std::map<void *, array_meta>();
      static std::mutex mu;
      std::unique_lock<std::mutex> lock(mu);
      return std::make_pair(M, std::move(lock));
    }}
    template <typename T>
    const array_meta& ARRAY_META_DICT_AT(T* a) {{
        auto [M, lock] = ARRAY_META_DICT();
        if (M->find(a) == M->end()) {{
          std::cerr << "Array meta not found for: " << type_name<T>() << std::endl;
          exit(EXIT_FAILURE);
        }}
        return M->at(a);
    }}

    void read_scalar(long double& x, std::istream& s) {{
        if (s.eof()) {{
            std::cerr << "Got unexpected EOF" << std::endl;
            exit(EXIT_FAILURE);
        }}
        scroll_space(s);

        std::string line;
        assert(std::getline(s, line));
        assert(!line.empty());
        if (line == "NaN") return;

        // Find the position to insert 'E' if needed (looking for exponent sign from
        // right)
        for (int i = line.length() - 1; i >= 0; --i) {{
            char current_char = line[i];
            if (current_char == '+' || current_char == '-') {{
                // Found a potential exponent sign. Check preceding character.
                if (i > 0 && (std::isdigit(line[i - 1]) || line[i - 1] == '.') &&
                    !(line[i - 1] == 'E' || line[i - 1] == 'e' || line[i - 1] == 'D' ||
                    line[i - 1] == 'd')) {{
                line.insert(i, "E"); // Insert 'E' and break
                break;
              }}
            }} else if (current_char == 'E' || current_char == 'e' ||
                        current_char == 'D' || current_char == 'd') {{
              // Already standard scientific notation, no insertion needed.
              break; // Exit loop
            }}
        }}

        // Parse the (potentially modified) string
        std::istringstream iss(line);
        iss >> x;
        if(iss.fail()) {{
            std::cerr << "Could not read long double: " << line << std::endl;
            exit(EXIT_FAILURE);
        }}
    }}

    void read_scalar(float& x, std::istream& s) {{
        long double xx;
        read_scalar(xx, s);
        x = static_cast<float>(xx);
    }}

    void read_scalar(double& x, std::istream& s) {{
        long double xx;
        read_scalar(xx, s);
        x = static_cast<double>(xx);
    }}

    template<typename T>
    void read_scalar(T& x, std::istream& s) {{
        if (s.eof()) {{
            std::cerr << "Got unexpected EOF" << std::endl;
            exit(EXIT_FAILURE);
        }}
        scroll_space(s);
        s >> x;
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
        auto m = read_array_meta(s);
        auto* y = m.read<T>(s);
        return {{m, y}};
    }}

    template<typename T>
    std::pair<array_meta, T*> read_pointer(std::istream& s) {{
        read_line(s, {{"# missing"}});  // Should contain '# missing'
        int missing;
        read_scalar(missing, s);
        assert(missing == 1);
        return read_array<T>(s);
    }}

    template<typename T> std::string serialize_array(T* arr);

    {cpp_deserializer_fns_str}
    {cpp_serializer_fns_str}

    template<typename T>
    T* array_meta::read(std::istream& s) const {{
        auto* buf = new T[volume()];
        if constexpr (std::is_pointer_v<T>) {{
            auto* bufc = read<std::remove_pointer_t<T>>(s);
            for (int i = 0; i < volume(); ++i) {{
                buf[i] = &bufc[i];
            }}
        }} else {{
            read_line(s, {{"# entries"}});
            for (int i = 0; i < volume(); ++i) {{
                deserialize(&buf[i], s);
            }}
        }}
        auto [M, lock] = ARRAY_META_DICT();
        (*M)[buf] = *this;
        return buf;
    }}

    template<typename T>
    std::string serialize_array(T* arr) {{
        const auto m = ARRAY_META_DICT_AT(static_cast<void*>(arr));
        std::stringstream s;
        add_line("# rank", s);
        add_line(m.rank, s);
        add_line("# size", s);
        for (auto i : m.size) add_line(i, s);
        add_line("# lbound", s);
        for (auto i : m.lbound) add_line(i, s);
        add_line("# entries", s);
        for (int i=0; i<m.volume(); ++i) add_line(serialize(arr[i]), s);
        return s.str();
    }}

    {gdata.cpp_serde}
}}  // namesepace {mod_name}

#endif // __DACE_{mod_name.upper()}_SERDE__
"""
    result = subprocess.run(
        ["clang-format"], input=cpp_code.strip(), text=True, capture_output=True
    )
    if not (result.returncode or result.stderr):
        cpp_code = result.stdout

    return cpp_code.strip()


def _finalize_f90_serde_module(
    f90_mod: Module,
    proc_names: List[str],
    array_serializers: Dict[Tuple[str, int], Subroutine_Subprogram],
    base_serializers: List[Subroutine_Subprogram],
    gdata: SerdeCode,
    ast: Program,
    sdfg_structs: Dict[str, Dict[str, str]],
    mod_name: str,
) -> str:
    impls = singular(sp for sp in walk(f90_mod, Module_Subprogram_Part))

    # Add serializers to the module and collect their names.
    for fn in chain(array_serializers.values(), base_serializers):
        _, name, _, _ = singular(children_of_type(fn, Subroutine_Stmt)).children
        proc_names.append(f"{name}")
        append_children(impls, fn)

    # Update the 'serialize' interface with all procedure names, including the base 'W_string'.
    proc_names.append("W_string")
    proc_names_stmt = Procedure_Stmt(
        f"module procedure {', '.join(sorted(list(set(proc_names))))}"
    )
    iface = singular(
        p for p in walk(f90_mod, Interface_Block) if find_name_of_node(p) == "serialize"
    )
    # The children are (Interface_Stmt, Procedure_Stmt, End_Interface_Stmt), replace the middle one.
    set_children(iface, (iface.children[0], proc_names_stmt, iface.children[2]))

    # Add global data serializer.
    append_children(impls, Subroutine_Subprogram(get_reader(gdata.f90_serializer)))

    # Generate and prepend the F90-C glue code.
    f90_glue_mod = _generate_f90_c_glue_code(ast, sdfg_structs, mod_name)
    return f"{f90_glue_mod.tofortran()}\n{f90_mod.tofortran()}".strip()


def _setup_serde_generation(
    g: SDFG, mod_name: str
) -> Tuple[
    Module,
    Module_Subprogram_Part,
    List[Subroutine_Subprogram],
    Dict[Tuple[str, int], Subroutine_Subprogram],
    Dict[str, Dict[str, str]],
]:
    """Initializes and returns the main data structures for code generation."""
    # F90 Serializer related data structures.
    f90_mod = gen_f90_serde_module_skeleton(mod_name)
    impls = singular(sp for sp in walk(f90_mod, Module_Subprogram_Part))
    base_serializers = _get_basic_serializers()
    array_serializers = _get_basic_array_serializers()

    # C++ SerDe related data structures.
    sdfg_structs = _get_sdfg_structs(g)
    return f90_mod, impls, base_serializers, array_serializers, sdfg_structs


def generate_serde_code(ast: Program, g: SDFG, mod_name: str = "serde") -> SerdeCode:
    ast = _keep_only_derived_types(ast)

    # Global data.
    gdata = _get_global_data_serde_code(ast, g)

    # F90 Serializer related data structures.
    (
        f90_mod,
        impls,
        base_serializers,
        array_serializers,
        sdfg_structs,
    ) = _setup_serde_generation(g, mod_name)

    # Generate ser/de for derived types
    proc_names, cpp_ser_fns, cpp_deser_fns = _generate_serde_for_derived_types(
        ast, sdfg_structs, array_serializers, impls
    )

    cpp_deserializer_fns: List[str] = _get_cpp_base_deserializers() + cpp_deser_fns
    cpp_serializer_fns: List[str] = _get_cpp_base_serializers() + cpp_ser_fns

    # Finalize the F90 serializer code.
    f90_code = _finalize_f90_serde_module(
        f90_mod,
        proc_names,
        array_serializers,
        base_serializers,
        gdata,
        ast,
        sdfg_structs,
        mod_name,
    )

    # Conclude the C++ serde code.
    cpp_code = _generate_cpp_serde_code(
        mod_name, sdfg_structs, cpp_deserializer_fns, cpp_serializer_fns, gdata
    )
    return SerdeCode(f90_serializer=f90_code, cpp_serde=cpp_code)


def _make_type_injection_entry_op(typ: SPEC, component: SPEC, expr: str) -> str:
    return f"""
call serialize(io, '{{ "type": "ConstTypeInjection", "scope": null, "root": "{".".join(typ)}", "component": "{".".join(component)}", "value": "', cleanup=.false., nline=.false.)
call serialize(io, {expr}, cleanup=.false., nline=.false.)
call serialize(io, '" }}', cleanup=.false., nline=.true.)
"""


def type_injection_leaf_ops(
    alias_map: SPEC_TABLE,
    all_derived_types: Dict[SPEC, DerivedTypeInfo],
    dt: DerivedTypeInfo,
    root_tspec: SPEC,
    trail: SPEC = tuple(),
) -> Generator[str, None, None]:
    for z in iterate_over_public_components(dt):
        # Prepare array ops that go together.
        siz_ops, off_ops = [], []
        if z.rank:
            comp = trail + (z.name,)
            for dim in range(z.rank):
                siz = trail + (f"__f2dace_SA_{z.name}_d_{dim}_s",)
                siz_ops.append(
                    _make_type_injection_entry_op(
                        root_tspec, siz, f"size(x%{'%'.join(comp)}, {dim + 1})"
                    )
                )
                off = trail + (f"__f2dace_SOA_{z.name}_d_{dim}_s",)
                off_ops.append(
                    _make_type_injection_entry_op(
                        root_tspec, off, f"lbound(x%{'%'.join(comp)}, {dim + 1})"
                    )
                )

        if z.alloc:
            comp = trail + (z.name,)
            alloc = trail + (f"{z.name}_a",)
            siz_ops = "\n".join(siz_ops)
            off_ops = "\n".join(off_ops)
            yield _make_type_injection_entry_op(
                root_tspec, alloc, f"merge(true, false, allocated(x%{'%'.join(comp)}))"
            )
            yield f"""
if (allocated(x%{"%".join(comp)})) then
  {siz_ops}
  {off_ops}
end if
"""
        elif z.rank:
            for op in chain(siz_ops, off_ops):
                yield op
        elif z.ptr:
            comp = trail + (f"__f2dace_{z.name}_POINTERTO",)
            yield _make_type_injection_entry_op(root_tspec, comp, f"'=> MISSING'")
        elif isinstance(z.type, Intrinsic_Type_Spec):
            comp = trail + (z.name,)
            yield _make_type_injection_entry_op(root_tspec, comp, f"x%{'%'.join(comp)}")
        elif isinstance(z.type, Declaration_Type_Spec):
            _, typ_name = z.type.children
            comp_typ_alias = dt.spec[:-1] + (typ_name.string,)
            assert comp_typ_alias in alias_map
            cdt = singular(
                cdt
                for k, cdt in all_derived_types.items()
                if alias_map[comp_typ_alias].parent is cdt.tdef
            )
            yield from type_injection_leaf_ops(
                alias_map, all_derived_types, cdt, root_tspec, trail + (z.name,)
            )
        else:
            raise NotImplementedError(
                f"Do not know how to process for type-injection: {dt}/{z}"
            )


def generate_type_injection_code(ast: Program, mod_name: str = "type_injection") -> str:
    ast = _keep_only_derived_types(ast)

    f90_mod = Module(
        get_reader(
            f"""
module {mod_name}
  use serde
  implicit none

  ! ALWAYS: First argument should be an integer `io` that is an **opened** writeable stream.
  ! ALWAYS: Second argument should be a **specialization type** object to be injected.
  interface type_inject
  end interface type_inject

  ! Some convenience constants
  character(6), parameter :: true = 'true', false = 'false'
contains
  ! A placeholder so that FParser does not remove the module subprogram part.
  subroutine noop()
  end subroutine noop
end module {mod_name}
""".strip()
        )
    )
    impls = singular(sp for sp in walk(f90_mod, Module_Subprogram_Part))

    # Actual code generation begins here.
    type_injection_serializers: Dict[SPEC, Subroutine_Subprogram] = {}
    ident_map = identifier_specs(ast)
    alias_map = alias_specs(ast)
    derived_type_map = {dt.spec: dt for dt in iterate_over_derived_types(ident_map)}
    for idx, dt in enumerate(iterate_over_derived_types(ident_map)):
        # Add config injectors from this type regardless whether they are present in the final SDFG.
        ti_ops = "\n".join(
            type_injection_leaf_ops(alias_map, derived_type_map, dt, dt.spec)
        )
        ti_fn_name = f"TI_{dt.name}_{idx}"
        type_injection_serializers[dt.spec] = Subroutine_Subprogram(
            get_reader(
                f"""
subroutine {ti_fn_name}(io, x)
  use {dt.spec[0]}, only: {dt.name}
  integer :: io
  type({dt.name}), intent(in) :: x
  {ti_ops}
  close(UNIT=io)
end subroutine {ti_fn_name}
""".strip()
            )
        )

    # Conclude the F90 type injection code.
    ti_procs = []
    for fn in type_injection_serializers.values():
        _, name, _, _ = singular(children_of_type(fn, Subroutine_Stmt)).children
        ti_procs.append(f"{name}")
        append_children(impls, fn)
    iface = singular(
        p
        for p in walk(f90_mod, Interface_Block)
        if find_name_of_node(p) == "type_inject"
    )
    proc_names = (
        [Procedure_Stmt(f"module procedure {', '.join(ti_procs)}")] if ti_procs else []
    )
    set_children(iface, iface.children[:-1] + proc_names + iface.children[-1:])

    return f90_mod.tofortran()


def find_all_f90_files(root: Path) -> Generator[Path, None, None]:
    if root.is_file():
        yield root
    else:
        for f in chain(root.rglob("*.f90"), root.rglob("*.F90"), root.rglob("*.incf")):
            yield f
