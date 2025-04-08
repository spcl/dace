# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
"""This module contains classes and functions that implement the grid-strided map tiling
transformation."""

import ast
import copy
import math
import re
from typing import Any, Dict, List
import dace
from dace.codegen.control_flow import ConditionalBlock
from dace.codegen.targets import cpp
from dace.libraries.standard import CodeLibraryNode
from dace.sdfg import SDFG, NestedDict, SDFGState
from dace.properties import make_properties
from dace.sdfg import nodes
from dace.data import ContainerArray, Structure, View
from dace.sdfg.graph import MultiConnectorEdge
from dace.transformation import pass_pipeline as ppl

from collections import OrderedDict
from typing import Set, Union
import typing
from dace import InterstateEdge, data
from dace.data import Data
from dace.memlet import Memlet
from dace import serialize, symbolic
from dace.properties import OrderedDictProperty, Property, make_properties
from enum import Enum

import numpy
import sympy

from dace.transformation.passes.array_elimination import ArrayElimination
from dace.sdfg import utils as sdutil
from dace.transformation.passes.duplicate_const_arrays import DuplicateConstArrays


class ContainerGroupFlatteningMode(Enum):
    ArrayOfStructs = 1
    StructOfArrays = 2


def _members_to_json(members):
    if members is None:
        return None
    return [(k, serialize.to_json(v)) for k, v in members.items()]


def _members_from_json(obj, context=None):
    if obj is None:
        return {}
    return OrderedDict((k, serialize.from_json(v, context)) for k, v in obj)


@make_properties
class ContainerGroup:
    name = Property(dtype=str, default="", allow_none=False)
    members = OrderedDictProperty(
        default=OrderedDict(),
        desc="Dictionary of structure members",
        from_json=_members_from_json,
        to_json=_members_to_json,
    )
    is_cg = Property(dtype=bool, default=False, allow_none=False)
    is_ca = Property(dtype=bool, default=False, allow_none=False)
    shape = Property(dtype=tuple, default=(1,), allow_none=False)
    strides = Property(dtype=tuple, default=(1,), allow_none=False)

    def __init__(self, name, is_cg, is_ca, shape, strides):
        self.name = name
        self.members = OrderedDict()
        self.is_cg = is_cg
        self.is_ca = is_ca
        self.shape = shape
        self.strides = strides
        self._validate()

    def add_member(self, name: str, member: Union[Data, "ContainerGroup"]):
        if name is None or name == "":
            name = len(self.members)
        self.members[name] = member

    @property
    def free_symbols(self) -> Set[symbolic.SymbolicType]:
        """Returns a set of undefined symbols in this data descriptor."""
        result = set()
        for k, v in self.members.items():
            result |= v.free_symbols
        return result

    def __call__(self):
        return self

    def validate(self):
        self._validate()

    def _validate(self):
        return True

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj["type"] != "ContainerGroup":
            raise TypeError("Invalid data type")

        ret = ContainerGroup({})
        serialize.set_properties_from_json(ret, json_obj, context=context)

        return ret

    def to_json(self):
        attrs = serialize.all_properties_to_json(self)
        retdict = {"type": type(self).__name__, "attributes": attrs}
        return retdict

    def is_equivalent(self, other):
        raise NotImplementedError

    def __eq__(self, other):
        return serialize.dumps(self) == serialize.dumps(other)

    def __hash__(self):
        return hash(serialize.dumps(self))

    def __repr__(self):
        members_repr = ", ".join(
            f"{k}: {v.__repr__()}" for k, v in self.members.items()
        )
        return f"ContainerGroup(name='{self.name}', is_cg={self.is_cg}, is_ca={self.is_ca}, shape={self.shape}, strides={self.strides}, members={{ {members_repr} }})"

    def __str__(self):
        return self.__repr__()

    def _soa_from_struct(self, name, structure):
        self._add_members(name, structure, acc_shape=(), acc_strides=())

    @classmethod
    def from_struct(
        cls,
        name: str,
        struct_or_container_array: typing.Union[data.Structure, data.ContainerArray],
        is_cg: bool,
        is_ca: bool,
        shape: tuple,
        strides: tuple,
    ) -> "ContainerGroup":
        dg = cls(name=name, is_cg=is_cg, is_ca=is_ca, shape=shape, strides=strides)
        assert is_cg ^ is_ca

        if isinstance(struct_or_container_array, data.Structure) and not isinstance(
            struct_or_container_array, data.View
        ):
            struct = struct_or_container_array
            for member_name, member in struct.members.items():
                new_member = None
                if isinstance(member, data.Structure):
                    new_member = cls.from_struct(
                        name=member_name,
                        struct_or_container_array=member,
                        is_cg=True,
                        is_ca=False,
                        shape=(1,),
                        strides=(1,),
                    )
                elif isinstance(member, data.ContainerArray):
                    new_member = cls.from_struct(
                        name=member_name,
                        struct_or_container_array=member,
                        is_cg=False,
                        is_ca=True,
                        shape=member.shape,
                        strides=member.strides,
                    )
                elif isinstance(member, (data.Array, data.Scalar)):
                    new_member = member
                elif isinstance(
                    member, (sympy.Basic, symbolic.SymExpr, int, numpy.integer)
                ):
                    new_member = data.Scalar(symbolic.symtype(member))
                else:
                    raise TypeError(
                        f"Unsupported member type in Structure: {type(member)}"
                    )

                dg.add_member(name=f"{member_name}", member=new_member)
        elif isinstance(
            struct_or_container_array, data.ContainerArray
        ) and not isinstance(struct_or_container_array, data.View):
            assert isinstance(struct_or_container_array, data.ContainerArray)
            container_array = struct_or_container_array
            member = container_array.stype
            member_name = None
            new_member = None

            if isinstance(member, data.Structure):
                # Recursively convert nested Structures
                member_name = member.name
                new_member = cls.from_struct(
                    name=member.name,
                    struct_or_container_array=member,
                    is_cg=True,
                    is_ca=False,
                    shape=(1,),
                    strides=(1,),
                )
            elif isinstance(member, data.ContainerArray):
                raise Exception(
                    "Two container arrays in a row is currently not supported"
                )
            elif isinstance(member, (data.Array, data.Scalar)):
                new_member = member
            elif isinstance(
                member, (sympy.Basic, symbolic.SymExpr, int, numpy.integer)
            ):
                new_member = data.Scalar(symbolic.symtype(member))
            else:
                raise TypeError(f"Unsupported member type in Structure: {type(member)}")
            dg.add_member(
                name=member_name if member_name is not None else "Leaf",
                member=new_member,
            )
        else:
            # If it is not a Struct, ContainerArray, or it is a View, do nothing
            pass

        return dg


# ===================================================================================================
# Functionality to Register Container Groups to an SDFG
# ===================================================================================================
def register_container_group_members(
    sdfg: dace.SDFG,
    flattening_mode: ContainerGroupFlatteningMode,
    register_as_transient: bool,
):
    registered = []
    for name, dg in sdfg.container_groups.items():
        registered += _register_container_group_members(
            sdfg=sdfg,
            flattening_mode=flattening_mode,
            container_group_or_array=dg,
            prefix_name=f"__CG_{name}",
            acc_shape=(),
            acc_strides=(),
            register_as_transient=register_as_transient,
        )
    return registered


def _get_fused_strides(member, acc_shape):
    strides = []
    member.strides
    tot_stride = math.prod(member.strides)
    cur_acc = 1
    for i in acc_shape:
        ns = tot_stride * cur_acc
        strides.insert(0, ns)
        cur_acc *= i
    strides += member.strides
    return strides


def _register_container_group_members(
    sdfg,
    flattening_mode,
    container_group_or_array: typing.Union["ContainerGroup", dace.data.ContainerArray],
    prefix_name: str,
    acc_shape: tuple,
    acc_strides: tuple,
    register_as_transient: bool,
):

    assert len(acc_shape) == len(acc_strides)
    added_descriptors = []
    if flattening_mode == ContainerGroupFlatteningMode.StructOfArrays:
        if isinstance(container_group_or_array, ContainerGroup):
            container_group = container_group_or_array
            for name, member in container_group.members.items():

                if isinstance(member, ContainerGroup):
                    if member.is_cg:
                        dg_prefix = prefix_name + f"__CG_{member.name}"
                        acc_shape2 = acc_shape
                        acc_strides2 = acc_strides
                    else:
                        dg_prefix = prefix_name + f"__CA_{member.name}"
                        acc_shape2 =tuple( list(acc_shape) + list(copy.deepcopy(member.shape)))
                        # lets say strides is [A, 1] and new shape is [B, 1] (shape MxN)
                        # then acc strides will be [A*B*N, B*M, B, 1]
                        acc_strides2 = tuple(
                            [
                                v * (member.strides[0] * member.shape[0])
                                for v in acc_strides
                            ]
                        ) + copy.deepcopy(member.strides)
                    added_descriptors += _register_container_group_members(
                        sdfg=sdfg,
                        flattening_mode=flattening_mode,
                        container_group_or_array=member,
                        prefix_name=dg_prefix,
                        acc_shape=acc_shape2,
                        acc_strides=acc_strides2,
                        register_as_transient=register_as_transient,
                    )
                elif isinstance(member, dace.data.ContainerArray):
                    assert False
                else:
                    # Add the dimensions accumulated while iterating from root to the leaf node of the trees
                    member_demangled_name = prefix_name + f"__m_{name}"
                    if isinstance(member, dace.data.Scalar):
                        if acc_shape == ():
                            datadesc = dace.data.Array(
                                dtype=member.dtype,
                                shape=(1,),
                                strides=(1,),
                                transient=(
                                    member.transient
                                    if not register_as_transient
                                    else True
                                ),
                                allow_conflicts=member.allow_conflicts,
                                storage=member.storage,
                                location=member.location,
                                lifetime=member.lifetime,
                                debuginfo=member.debuginfo,
                                may_alias=False,
                            )
                        else:
                            # if acc_strides != ():
                            #    raise Exception(f"TODO {acc_strides}")
                            datadesc = dace.data.Array(
                                dtype=member.dtype,
                                shape=acc_shape,
                                strides=acc_strides,
                                transient=(
                                    member.transient
                                    if not register_as_transient
                                    else True
                                ),
                                allow_conflicts=member.allow_conflicts,
                                storage=member.storage,
                                location=member.location,
                                may_alias=member.may_alias,
                                lifetime=member.lifetime,
                                debuginfo=member.debuginfo,
                                start_offset=member.start_offset,
                            )
                    elif isinstance(member, dace.data.Array):
                        _acc_shape =tuple( list(acc_shape) + list(member.shape))
                        _acc_strides = (
                            tuple(
                                [
                                    v * (member.strides[0] * member.shape[0])
                                    for v in acc_strides
                                ]
                            )
                            + member.strides
                        )
                        assert len(acc_shape) == len(acc_strides)
                        datadesc = dace.data.Array(
                            dtype=member.dtype,
                            shape=_acc_shape,
                            strides=_acc_strides,
                            transient=(
                                member.transient if not register_as_transient else True
                            ),
                            allow_conflicts=member.allow_conflicts,
                            storage=member.storage,
                            location=member.location,
                            may_alias=member.may_alias,
                            lifetime=member.lifetime,
                            debuginfo=member.debuginfo,
                        )
                    else:
                        raise Exception(
                            "Leaf member in a container group needs to be scalar or array"
                        )
                    sdfg.add_datadesc(
                        name=member_demangled_name,
                        datadesc=datadesc,
                        find_new_name=False,
                    )
                    added_descriptors += [(member_demangled_name, datadesc)]
        elif isinstance(container_group_or_array, dace.data.ContainerArray):
            raise Exception("Top level should not be a ContainerArray")
        else:
            raise Exception("?")
    elif flattening_mode == ContainerGroupFlatteningMode.ArrayOfStructs:
        raise Exception("TODO Support for ArrayOfStructs Flattening")
    else:
        raise Exception("Unsupported Flattening Mode")
    return added_descriptors


def get_demangled_container_group_member_name(
    sdfg, name_hierarchy: typing.List[typing.Type[str]]
):
    current_dg = None
    demangled_name = ""
    for i, name in enumerate(name_hierarchy):
        if current_dg is None:
            current_dg = sdfg.container_groups[name]
            demangled_name += f"__CG_{current_dg.name}"
        elif name in current_dg.members:
            if isinstance(current_dg.members[name], ContainerGroup):
                current_dg = current_dg.members[name]
                if current_dg.is_cg:
                    demangled_name += f"__CG_{current_dg.name}"
                else:
                    demangled_name += f"__CA_{current_dg.name}"
            elif isinstance(current_dg.members[name], dace.data.ContainerArray):
                assert False
            else:
                assert isinstance(current_dg.members[name], dace.data.Data)
                assert i == len(name_hierarchy) - 1
                demangled_name += f"__m_{name}"
                return demangled_name
        else:
            # if we are at last element and it is a "Leaf" (data had no name) it is not an error
            if (
                i == len(name_hierarchy) - 1
                and len(current_dg.members) == 1
                and "Leaf" in current_dg.members
            ):
                demangled_name += f"__m_Leaf"
                return demangled_name
            raise Exception(f"Name Hierarchy {name_hierarchy} Not in ContainerGroups 1")

    if (
        i == len(name_hierarchy) - 1
        and len(current_dg.members) == 1
        and "Leaf" in current_dg.members
    ):
        demangled_name += f"__m_Leaf"
        return demangled_name


    raise Exception(f"Name Hierarchy {name_hierarchy} Not in ContainerGroups 2")


def generate_container_groups_from_structs(
    sdfg: dace.SDFG, flattening_mode: ContainerGroupFlatteningMode
):
    added_cgs = 0
    sdfg.container_groups = NestedDict()
    for arr_name, arr in sdfg._arrays.items():
        if isinstance(arr, dace.data.Structure):
            dg_name = arr_name
            dg = ContainerGroup.from_struct(
                name=dg_name,
                struct_or_container_array=arr,
                is_cg=True,
                is_ca=False,
                shape=(1,),
                strides=(1,),
            )
            sdfg.container_groups[dg_name] = dg
            added_cgs += 1
    return added_cgs


def clean_container_groups(sdfg: dace.SDFG):
    # assert hasattr(
    #    sdfg, "container_groups"
    # ), "SDFG does not have a field named 'container_groups member, clean container groups called'"
    if hasattr(sdfg, "container_groups"):
        sdfg.container_groups = None


def add_container_group_desc(
    sdfg: dace.SDFG,
    name: str,
    container_group_desc: ContainerGroup,
    find_new_name=False,
) -> str:
    if not isinstance(name, str):
        raise TypeError(
            "Data descriptor name must be a string. Got %s" % type(name).__name__
        )

    if find_new_name:
        name = sdfg._find_new_name(name)
        name = name.replace(".", "_")
        if sdfg.is_name_used(name):
            name = sdfg._find_new_name(name)
    else:
        if name in sdfg.arrays:
            raise FileExistsError(
                f'Data group descriptor "{name}" already exists in SDFG'
            )
        if name in sdfg.symbols:
            raise FileExistsError(
                f'Can not create data group descriptor "{name}", the name is used by a symbol.'
            )
        if name in sdfg._subarrays:
            raise FileExistsError(
                f'Can not create data group descriptor "{name}", the name is used by a subarray.'
            )
        if name in sdfg._rdistrarrays:
            raise FileExistsError(
                f'Can not create data group descriptor "{name}", the name is used by a RedistrArray.'
            )
        if name in sdfg._pgrids:
            raise FileExistsError(
                f'Can not create data group descriptor "{name}", the name is used by a ProcessGrid.'
            )

    def _add_symbols(sdfg: SDFG, desc: dace.data.Data):
        if isinstance(desc, dace.data.Structure):
            for v in desc.members.values():
                if isinstance(v, dace.data.Data):
                    _add_symbols(sdfg, v)
        for sym in desc.free_symbols:
            if sym.name not in sdfg.symbols:
                sdfg.add_symbol(sym.name, sym.dtype)

    # Add the data descriptor to the SDFG and all symbols that are not yet known.
    sdfg.container_groups[name] = container_group_desc
    _add_symbols(sdfg, container_group_desc)

    return name


def add_container_group(
    sdfg: dace.SDFG, name: str, find_new_name: bool = False
) -> typing.Tuple[str, "ContainerGroup"]:
    dg_desc = ContainerGroup(name)
    return (
        add_container_group_desc(sdfg, name, dg_desc, find_new_name=find_new_name),
        dg_desc,
    )


def _get_name_hierarchy_from_name(demangled_name: str):
    # Split the string while keeping the prefixes
    parts = re.split(r"(?=__(?:CG|CA|m))", demangled_name)

    name_hierarchy = []
    type_hierarchy = []

    for p in parts:
        if not p or p == "":
            continue
        if p.startswith("__CG_"):
            name_hierarchy.append(p[5:])
            type_hierarchy.append("CG")
        elif p.startswith("__CA_"):
            name_hierarchy.append(p[5:])
            type_hierarchy.append("CA")
        elif p.startswith("__m_"):
            name_hierarchy.append(p[4:])
            type_hierarchy.append("m")

    return name_hierarchy, type_hierarchy


# ===================================================================================================


@make_properties
class Flattener(CodeLibraryNode):
    code = Property(dtype=str, default="", allow_none=False)
    shallow = Property(dtype=bool, default=False, allow_none=False)

    def __init__(self, name, input_names, output_names, code, shallow=False):
        super().__init__(name=name, input_names=input_names, output_names=output_names)
        self.code = code
        self.shallow = shallow

    def generate_code(self, inputs, outputs):
        if not self.shallow:
            all_code = f"""
// Start {self.name}
#pragma omp parallel
{{
#pragma omp single
{{
{{
{self.code}
}}
#pragma omp taskwait
}}
}}
// End {self.name}
"""
        else:
            all_code = f"""
// Start {self.name}
{{
{self.code}
}}
// End {self.name}
"""
        return all_code


@make_properties
class StructToContainerGroups(ppl.Pass):
    clean_trivial_views = Property(
        dtype=bool,
        default=False,
        desc="Clean Trivial Views",
        allow_none=False,
    )
    def __init__(
        self,
        flattening_mode: ContainerGroupFlatteningMode = ContainerGroupFlatteningMode.StructOfArrays,
        simplify: bool = True,
        validate: bool = True,
        validate_all: bool = False,
        clean_container_grous: bool = True,
        save_steps: bool = False,
        interface_with_struct_copy: bool = True,
        interface_to_gpu: bool = False,
        verbose: bool = False,
        clean_trivial_views: bool = False,
        shallow_copy: bool = False,
        shallow_copy_to_gpu: bool = False
    ):
        if shallow_copy:
            assert shallow_copy_to_gpu is False and interface_to_gpu is False and interface_with_struct_copy is True
        if shallow_copy_to_gpu:
            assert shallow_copy is False and interface_to_gpu is True and interface_with_struct_copy is True
        if flattening_mode != ContainerGroupFlatteningMode.StructOfArrays:
            raise Exception("Only StructOfArrays is supported")
        super().__init__()
        self._simplify = simplify
        self._validate = validate
        self._validate_all = validate_all
        self._clean_container_grous = clean_container_grous
        self._flattening_mode = flattening_mode
        self._call = 0
        self._save_steps = save_steps
        self._struct_to_view_reconn_map = dict()
        self._view_to_struct_reconn_map = dict()
        self._interface_with_struct_copy = interface_with_struct_copy
        self._verbose = verbose
        self._struct_replacements = dict()
        self._flattener_codestr = ""
        self._deflattener_codestr = ""
        self._interface_to_gpu = interface_to_gpu
        self.clean_trivial_views = clean_trivial_views
        self._shallow_copy = shallow_copy
        self._shallow_copy_to_gpu = shallow_copy_to_gpu

    def modifies(self) -> ppl.Modifies:
        return (
            ppl.Modifies.Nodes
            | ppl.Modifies.Edges
            | ppl.Modifies.AccessNodes
            | ppl.Modifies.Memlets
            | ppl.Modifies.Descriptors
        )

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _generate_shallow_flattener(
        self,
        sdfg: SDFG,
        name: str,
        desc: dace.data.Structure,
        registered_members: typing.List[typing.Tuple[str, dace.data.Data]],
        verbose:bool=False,
        gpu_prefix:bool=False,
        host_list:List[str] = [
            "__CG_global_data__m_nrdmax",
            "__CG_global_data__m_nflatlev",
            "__CG_p_patch__CG_verts__m_start_block",
            "__CG_p_patch__CG_verts__m_end_block",
            "__CG_p_patch__CG_verts__m_start_index",
            "__CG_p_patch__CG_verts__m_end_index",
            "__CG_p_patch__CG_cells__m_start_block",
            "__CG_p_patch__CG_cells__m_end_block",
            "__CG_p_patch__CG_cells__m_start_index",
            "__CG_p_patch__CG_cells__m_end_index",
            "__CG_p_patch__CG_edges__m_start_block",
            "__CG_p_patch__CG_edges__m_end_block",
            "__CG_p_patch__CG_edges__m_start_index",
            "__CG_p_patch__CG_edges__m_end_index",
        ],
    ):
        def _gen_loop(
            sdfg: SDFG,
            structname: str,
            struct: dace.data.Structure,
            arrname: str,
            arr: dace.data.Data,
            name_hierarchy: List[str],
            name_hierarchy_types: List[str],
        ):

            _cstr = ""
            _cstr_rev = ""
            current_member = struct
            src_access = f"{name_hierarchy[0]}"

            for i, (name, _type) in enumerate(
                zip(name_hierarchy[1:], name_hierarchy_types[1:])
            ):
                prev_name = name_hierarchy[i]
                prev_type = name_hierarchy_types[i]  # i is alread 0 while we are at 1

                if prev_type == "CG":
                    if _type == "CA":
                        src_access += f"->{name}"
                    else:
                        assert _type == "m" or _type == "CG"
                        src_access += f"->{name}"
                elif prev_type == "CA":
                    if isinstance(current_member, ContainerArray):
                        raise Exception("Not implemented on shallow copy yet.")
                    else:
                        raise Exception("Should not happen")
                elif prev_type == "m":
                    raise Exception("Should not happen")
                else:
                    raise Exception("Unsupported type")

                if isinstance(current_member, ContainerArray):
                    current_member = current_member.stype
                else:
                    current_member= current_member.members[name]

            member_arr = struct
            for member_name, prev_type, member_type in zip(
                name_hierarchy[1:], name_hierarchy_types[:-1], name_hierarchy_types[1:]
            ):
                if member_type == "m":
                    if prev_type == "CG":
                        member_arr = member_arr.members[member_name]
                    elif prev_type == "CA":
                        member_arr = member_arr.stype
                else:
                    if prev_type == "CG":
                        member_arr = member_arr.members[member_name]
                    elif prev_type == "CA":
                        member_arr = member_arr.stype

            if isinstance(sdfg.arrays[arrname], dace.data.Scalar):
                access = f"{arrname} = {src_access};\n"
                rev_access = f"{src_access} = {arrname};\n"
            else:
                assert isinstance(sdfg.arrays[arrname], dace.data.Array)
                if arrname in host_list:
                    access = f"{arrname} = {src_access};\n"
                    rev_access = f"{src_access} = {arrname};\n"
                else:
                    access = f"gpu_{arrname} = {src_access};\n"
                    rev_access = f"{src_access} = gpu_{arrname};\n"
            _cstr += access
            _cstr_rev += rev_access
            return _cstr, _cstr_rev

        ll = [
            _gen_loop(
                sdfg,
                name,
                desc,
                arr_name,
                arr_desc,
                *_get_name_hierarchy_from_name(arr_name),
            )
            for (arr_name, arr_desc) in registered_members
            if _get_name_hierarchy_from_name(arr_name)[0][0] == name
        ]

        copy_strs_list, copy_strs_reverse_list = zip(*ll) if ll != [] else ([], [])
        if verbose:
           print("Shallow copy_str:", copy_strs_list, "=", copy_strs_reverse_list)
        #raise Exception(copy_strs_list, copy_strs_reverse_list)
        copy_strs = "\n".join(copy_strs_list)
        copy_strs_reverse = "\n".join(copy_strs_reverse_list)

        flattener_codestr = f"""
{copy_strs}
"""

        deflattener_codestr = f"""
{copy_strs_reverse}
"""

        self._flattener_codestr += flattener_codestr
        self._deflattener_codestr += deflattener_codestr


    def _generate_flattener(
        self,
        sdfg: SDFG,
        name: str,
        desc: dace.data.Structure,
        registered_members: typing.List[typing.Tuple[str, dace.data.Data]],
    ):
        def _gen_loop(
            sdfg: SDFG,
            structname: str,
            struct: dace.data.Structure,
            arrname: str,
            arr: dace.data.Data,
            name_hierarchy: List[str],
            name_hierarchy_types: List[str],
        ):

            _cstr = ""
            _cstr_reverse = ""
            letter = "i"
            used_letters = []

            # If arr.shape is (1,) DaCe generates a scalar, no need to loop + [i] access will be wrong
            if (arr.shape != (1,)):
                fl = []
                for dim in arr.shape:
                    if len(fl) == 0:
                        fl.append(f"#pragma omp simd\nfor (auto {letter} = 0; {letter} < {dim}; {letter}++){{")
                    else:
                        fl.append(f"for (auto {letter} = 0; {letter} < {dim}; {letter}++){{")
                    used_letters.append(letter)
                    letter = chr(ord(letter) + 1)
                # Fortran Order High change first dimension has stride 1
                _cstr += "\n".join(list(reversed(fl))) + "\n"
                _cstr_reverse += "\n".join(list(reversed(fl))) + "\n"

            access_cpp_str = " + ".join(
                [
                    f"({letter} * ({cpp.sym2cpp(stride)}))"
                    for letter, dim, stride in zip(used_letters, arr.shape, arr.strides)
                ]
            )
            current_member = struct
            current_stride = 1
            src_access = f"{name_hierarchy[0]}"
            remaining_letters = copy.deepcopy(used_letters)
            for i, (name, _type) in enumerate(
                zip(name_hierarchy[1:], name_hierarchy_types[1:])
            ):
                prev_name = name_hierarchy[i]
                prev_type = name_hierarchy_types[i]  # i is alread 0 while we are at 1

                if prev_type == "CG":
                    if _type == "CA":
                        src_access += f"->{name}"
                    else:
                        assert _type == "m" or _type == "CG"
                        src_access += f"->{name}"
                elif prev_type == "CA":
                    if isinstance(current_member, ContainerArray):
                        src_access += "["
                        first= True
                        for i in range(len(current_member.shape)):
                            if first:
                                first = False
                            else:
                                src_access += " + "
                            src_access += f"{remaining_letters.pop(0)} * {current_stride}"
                            current_stride *= current_member.strides[i]
                        src_access += "]"
                    else:
                        raise Exception("Should not happen")
                elif prev_type == "m":
                    raise Exception("Should not happen")
                else:
                    raise Exception("Unsupported type")

                if isinstance(current_member, ContainerArray):
                    current_member = current_member.stype
                else:
                    current_member= current_member.members[name]

            # assert len(remaining_letters) == len(arr.shape)
            member_arr = struct
            for member_name, prev_type, member_type in zip(
                name_hierarchy[1:], name_hierarchy_types[:-1], name_hierarchy_types[1:]
            ):
                if member_type == "m":
                    if prev_type == "CG":
                        member_arr = member_arr.members[member_name]
                    elif prev_type == "CA":
                        member_arr = member_arr.stype
                else:
                    if prev_type == "CG":
                        member_arr = member_arr.members[member_name]
                    elif prev_type == "CA":
                        member_arr = member_arr.stype

            if not isinstance(member_arr, dace.data.Scalar):
                if len(arr.shape) > 2:
                    ompfor = f"#pragma omp taskloop\n"
                elif len(arr.shape) == 2:
                    #ompfor = f"#pragma omp taskloop\n"
                    ompfor = f"#pragma omp task\n"
                else:
                    ompfor = ""
                _cstr = ompfor + _cstr
                _cstr_reverse = ompfor + _cstr_reverse

            if (
                isinstance(member_arr, dace.data.Scalar) or
                (isinstance(member_arr, dace.data.Array) and member_arr.shape == (1,))
                ):
                assert len(remaining_letters) == 1 or len(remaining_letters) == 0
                endaccess = ""
            else:
                assert len(remaining_letters) == len(
                    member_arr.shape
                ), f"{member_arr}, {remaining_letters}"
                endaccess = " + ".join(
                    [
                        f"({letter} * ({cpp.sym2cpp(stride)}))"
                        for letter, dim, stride in zip(
                            remaining_letters, member_arr.shape, member_arr.strides
                        )
                    ]
                )


            if endaccess != "":
                src_access = f"{src_access}[{endaccess}]"
            else:
                src_access = f"{src_access}"
            if access_cpp_str != "":
                access_cpp_str = f"[{access_cpp_str}]"

            access = f"{arrname.lower()}{access_cpp_str} = {src_access};\n"
            access_reverse = f"{src_access} = {arrname.lower()}{access_cpp_str};\n"
            _cstr += access
            _cstr_reverse += access_reverse

            if (arr.shape != (1,)):
                for dim in arr.shape:
                    fe = f"}}\n"
                    _cstr += fe
                    _cstr_reverse += fe

            return _cstr, _cstr_reverse

        ll = [
            _gen_loop(
                sdfg,
                name,
                desc,
                arr_name,
                arr_desc,
                *_get_name_hierarchy_from_name(arr_name),
            )
            for (arr_name, arr_desc) in registered_members
            if _get_name_hierarchy_from_name(arr_name)[0][0] == name
        ]

        copy_strs_list, copy_strs_reverse_list = zip(*ll) if ll != [] else ([], [])
        copy_strs = "\n".join(copy_strs_list)
        copy_strs_reverse = "\n".join(copy_strs_reverse_list)

        """
        // Flatten:
        // From:
        // {name}, {desc}
        // To:
        {main_comment}
        """

        flattener_codestr = f"""
{copy_strs}
"""
        """
        // Deflatten: 
        // Form:
        {main_comment}
        // To:
        // {name}, {desc}
        """

        deflattener_codestr = f"""
{copy_strs_reverse}
"""

        self._flattener_codestr += flattener_codestr
        self._deflattener_codestr += deflattener_codestr

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> int:
        new_cgs = generate_container_groups_from_structs(sdfg, self._flattening_mode)
        if new_cgs == 0:
            print("No container groups were registered as no structs were found")
            return
        registered_members = register_container_group_members(
            sdfg,
            self._flattening_mode,
            register_as_transient=True if self._interface_with_struct_copy else False,
        )

        # Preprocess chains as needed

        for state in sdfg.states():

            nodes_to_rm = set()
            edges_to_add = set()
            nodes = state.nodes()
            skip=False
            for node in nodes:
                if isinstance(node, dace.nodes.NestedSDFG):
                    skip=True
                    break
            if skip:
                continue
            for node in nodes:

                if isinstance(node, dace.nodes.AccessNode) and not isinstance(
                    sdfg.arrays[node.data], dace.data.View
                ):
                    sv = False
                    for oe in state.out_edges(node):
                        if isinstance(oe.dst, dace.nodes.AccessNode) and isinstance(
                            sdfg.arrays[oe.dst.data], dace.data.View
                        ):
                            sv = True
                            break
                    if sv:
                        _nodes_to_rm, _edges_to_add = (
                            self._process_and_pad_struct_to_view_view_chain(
                                state, sdfg, node
                            )
                        )
                        nodes_to_rm = nodes_to_rm.union(_nodes_to_rm)
                        edges_to_add = edges_to_add.union(_edges_to_add)
                    vs = False
                    for ie in state.in_edges(node):
                        if isinstance(ie.src, dace.nodes.AccessNode) and isinstance(
                            sdfg.arrays[ie.src.data], dace.data.View
                        ):
                            vs = True
                            break
                    if vs:
                        _nodes_to_rm, _edges_to_add = (
                            self._process_and_pad_view_to_struct_view_chain(
                                state, sdfg, node
                            )
                        )
                        nodes_to_rm = nodes_to_rm.union(_nodes_to_rm)
                        edges_to_add = edges_to_add.union(_edges_to_add)
            for node in nodes_to_rm:
                state.remove_node(node)
            for edge in edges_to_add:
                state.add_edge(*edge)

        if self._save_steps:
            sdfg.save("preprocessed.sdfgz", compress=True)

        # A -> B both access nodes, this should trigger the further check whether we can apply
        i = 0
        name_replacements = dict()
        for state in sdfg.states():

            nodes = state.nodes()
            skip=False
            for node in nodes:
                if isinstance(node, dace.nodes.NestedSDFG):
                    skip=True
                    break
            if skip:
                continue
            removed_nodes = set()
            for node in nodes:
                if node in removed_nodes:
                    continue
                if isinstance(node, dace.nodes.AccessNode):
                    out_edges = state.out_edges(node)
                    for oe in out_edges:
                        if oe.dst in removed_nodes:
                            continue
                        if isinstance(oe.dst, dace.nodes.AccessNode):
                            src_access = node
                            dst_access = oe.dst
                            pattern_found = self._can_be_applied(
                                state, sdfg, src_access, dst_access
                            )
                            if pattern_found:
                                i += 1
                                newly_removed_nodes, (oldname, newname) = self._apply(
                                    state, sdfg, src_access, dst_access
                                )
                                if self._save_steps:
                                    sdfg.save(f"apply{i}.sdfgz")
                                removed_nodes = removed_nodes.union(newly_removed_nodes)
                                name_replacements[oldname] = newname

        if self._save_steps:
            sdfg.save("replaced.sdfgz", compress=True)

        registered_names, registered_descs = map(list, zip(*registered_members))

        # Collect unused arrays
        used_containers = set()
        def collect_used_containers(sdfg):
            for state in sdfg.states():
                for node in state.nodes():
                    if isinstance(node, dace.nodes.AccessNode):
                        if node.data is not None and node.data in registered_names:
                            used_containers.add(node.data)
                    if isinstance(node, dace.nodes.NestedSDFG):
                        collect_used_containers(node.sdfg)

        collect_used_containers(sdfg)

        unused_containers = set(registered_names) - used_containers

        for container in unused_containers:
            desc = sdfg.arrays[container]
            sdfg.remove_data(container, validate=True)
            registered_names.remove(container)
            registered_members.remove((container, desc))
            registered_descs.remove(desc)

        # Do not simplify until flattener is generated it will remove things
        replace_length_one_arrays_with_scalars(sdfg)
        registered_array_names = set(v for v in registered_names if  isinstance(sdfg.arrays[v], dace.data.Array))
        registered_scalar_names = set(v for v in registered_names if  isinstance(sdfg.arrays[v], dace.data.Scalar))

        if self.clean_trivial_views:
            clean_trivial_views(sdfg)


        self._flattener_codestr = ""
        self._deflattener_codestr = ""
        for name, desc in sdfg.arrays.items():
            if isinstance(desc, dace.data.Structure) and not isinstance(
                desc, dace.data.View
            ):
                self._generate_shallow_flattener(sdfg, name, desc, registered_members, self._verbose)

        with open(f"{sdfg.name}_flattener_code.cpp", "w") as f:
            self._generate_shallow_flattener(sdfg, name, desc, registered_members, self._verbose)
            f.write(self._flattener_codestr)
        with open(f"{sdfg.name}_deflattener_code.cpp", "w") as f:
            f.write(self._deflattener_codestr)

        self._flattener_codestr = ""
        self._deflattener_codestr = ""
        # Generate the flattener functions (one per struct)
        for name, desc in sdfg.arrays.items():
            if isinstance(desc, dace.data.Structure) and not isinstance(
                desc, dace.data.View
            ):
                if self._shallow_copy:
                    assert self._interface_with_struct_copy
                    self._generate_shallow_flattener(sdfg, name, desc, registered_members, self._verbose)
                elif self._shallow_copy_to_gpu:
                    self._interface_with_struct_copy

                elif self._interface_with_struct_copy:
                    self._generate_flattener(sdfg, name, desc, registered_members)

        if self._shallow_copy:
            assert self._interface_with_struct_copy
            assert not self._interface_to_gpu
            if self._interface_with_struct_copy:
                flatten_lib_node = Flattener(
                    name="flatten",
                    code=self._flattener_codestr,
                    input_names=[],  # [k.lower() for k, v in sdfg.arrays.items() if isinstance(v, dace.data.Structure) and not
                    # isinstance(v, dace.data.View)],
                    output_names=[n.lower() for n in registered_names],
                )
                deflatten_lib_node = Flattener(
                    name="deflatten",
                    code=self._deflattener_codestr,
                    input_names=[n.lower() for n in registered_names],
                    output_names=[],  # [k.lower() for k, v in sdfg.arrays.items() if (isinstance(v, dace.data.Structure)
                    # or isinstance(v, dace.data.ContainerArray)) and not
                    # isinstance(v, dace.data.View)],
                )
                flatten_lib_node.schedule = dace.dtypes.ScheduleType.CPU_Multicore
                deflatten_lib_node.schedule = dace.dtypes.ScheduleType.CPU_Multicore
                # entry_interface = sdfg.add_state("entry_interface")
                # exit_interface = sdfg.add_state("exit_interface")
                start_state = sdfg.start_state
                entry_interface = sdfg.add_state_before(
                    sdfg.start_state,
                    "entry_interface",
                    is_start_block=True,
                    is_start_state=True,
                )
                # sdfg.add_edge(entry_interface, start_state, dace.sdfg.InterstateEdge())

                assert sdfg.start_state == entry_interface
                assert sdfg.start_block == entry_interface

                entry_interface.add_node(flatten_lib_node)
                for inname in set(
                    [
                        k
                        for k, v in sdfg.arrays.items()
                        if (
                            isinstance(v, dace.data.Structure)
                            or isinstance(v, dace.data.ContainerArray)
                        )
                        and not isinstance(v, dace.data.View)
                    ]
                ):
                    an = dace.nodes.AccessNode(inname)
                    entry_interface.add_node(an)
                    entry_interface.add_edge(
                        an, None, flatten_lib_node, None, dace.Memlet.from_array(inname, sdfg.arrays[inname])
                    )
                    sdfg.arrays[inname].storage = dace.StorageType.CPU_Heap

                for outname in set(registered_names):
                    if self._interface_to_gpu:
                        an = dace.nodes.AccessNode(outname)
                        entry_interface.add_node(an)
                        arr = sdfg.arrays[outname]

                        entry_interface.add_edge(
                            flatten_lib_node,
                            outname.lower(),
                            an,
                            None,
                            dace.Memlet.from_array(outname, sdfg.arrays[outname]),
                        )

                        if outname.lower() not in flatten_lib_node.out_connectors:
                            flatten_lib_node.add_out_connector(outname.lower())

                        sdfg.arrays[outname].storage = dace.StorageType.CPU_Heap

                        if outname.lower() not in flatten_lib_node.out_connectors:
                            flatten_lib_node.add_out_connector(outname.lower())

                exit_interface = sdfg.add_state("exit_interface")
                end_nodes = set()
                for cfg in sdfg.nodes():
                    if sdfg.out_degree(cfg) == 0 and cfg != exit_interface:
                        end_nodes.add(cfg)
                assert len(end_nodes) == 1, f"End nodes: {end_nodes}"
                for end_node in end_nodes:
                    sdfg.add_edge(end_node, exit_interface, dace.sdfg.InterstateEdge())

                exit_interface.add_node(deflatten_lib_node)
                for inname in set(registered_names):
                    an = exit_interface.add_access(inname)
                    exit_interface.add_edge(
                        an, None, deflatten_lib_node, inname.lower(),dace.Memlet.from_array(inname, sdfg.arrays[inname])
                    )

                    if inname.lower() not in deflatten_lib_node.in_connectors:
                        deflatten_lib_node.add_in_connector(inname.lower())

                for outname in set(
                    [
                        k
                        for k, v in sdfg.arrays.items()
                        if (
                            isinstance(v, dace.data.Structure)
                            or isinstance(v, dace.data.ContainerArray)
                        )
                        and not isinstance(v, dace.data.View)
                    ]
                ):
                    an = dace.nodes.AccessNode(outname)
                    exit_interface.add_node(an)
                    exit_interface.add_edge(
                        an, None, deflatten_lib_node, None, dace.Memlet()
                    )
        elif self._shallow_copy_to_gpu:

            assert self._interface_with_struct_copy
            assert self._interface_to_gpu
            if self._interface_with_struct_copy:
                flatten_lib_node = Flattener(
                    name="flatten",
                    code=self._flattener_codestr,
                    input_names=[],  # [k.lower() for k, v in sdfg.arrays.items() if isinstance(v, dace.data.Structure) and not
                    # isinstance(v, dace.data.View)],
                    output_names=["gpu_" + n.lower() for n in registered_array_names] +
                        [n.lower() for n in registered_scalar_names],
                )
                deflatten_lib_node = Flattener(
                    name="deflatten",
                    code=self._deflattener_codestr,
                    input_names=["gpu_" + n.lower() for n in registered_names] +
                        [n.lower() for n in registered_scalar_names],
                    output_names=[],  # [k.lower() for k, v in sdfg.arrays.items() if (isinstance(v, dace.data.Structure)
                    # or isinstance(v, dace.data.ContainerArray)) and not
                    # isinstance(v, dace.data.View)],
                )
                flatten_lib_node.schedule = dace.dtypes.ScheduleType.CPU_Multicore
                deflatten_lib_node.schedule = dace.dtypes.ScheduleType.CPU_Multicore
                # entry_interface = sdfg.add_state("entry_interface")
                # exit_interface = sdfg.add_state("exit_interface")
                start_state = sdfg.start_state
                entry_interface = sdfg.add_state_before(
                    sdfg.start_state,
                    "entry_interface",
                    is_start_block=True,
                    is_start_state=True,
                )
                # sdfg.add_edge(entry_interface, start_state, dace.sdfg.InterstateEdge())

                assert sdfg.start_state == entry_interface
                assert sdfg.start_block == entry_interface

                entry_interface.add_node(flatten_lib_node)
                for inname in set(
                    [
                        k
                        for k, v in sdfg.arrays.items()
                        if (
                            isinstance(v, dace.data.Structure)
                            or isinstance(v, dace.data.ContainerArray)
                        )
                        and not isinstance(v, dace.data.View)
                    ]
                ):
                    an = dace.nodes.AccessNode(inname)
                    entry_interface.add_node(an)
                    entry_interface.add_edge(
                        an, None, flatten_lib_node, None, dace.Memlet.from_array(inname, sdfg.arrays[inname])
                    )
                    sdfg.arrays[inname].storage = dace.StorageType.CPU_Heap

                for outname in set(registered_names):
                    if self._interface_to_gpu:
                        oname = "gpu_" + outname if outname in registered_array_names else outname
                        an = dace.nodes.AccessNode(oname)
                        entry_interface.add_node(an)
                        desc = copy.deepcopy(sdfg.arrays[outname])
                        desc.storage = dace.StorageType.GPU_Global
                        sdfg.remove_data(outname, validate=False)
                        sdfg.add_datadesc(oname, desc)
                        arr = desc

                        entry_interface.add_edge(
                            flatten_lib_node,
                            oname.lower(),
                            an,
                            None,
                            dace.Memlet.from_array(oname, sdfg.arrays[oname]),
                        )

                        if oname.lower() not in flatten_lib_node.out_connectors:
                            flatten_lib_node.add_out_connector(oname.lower())

                        sdfg.arrays[oname].storage = dace.StorageType.GPU_Global

                        if oname.lower() not in flatten_lib_node.out_connectors:
                            flatten_lib_node.add_out_connector(oname.lower())

                exit_interface = sdfg.add_state("exit_interface")
                end_nodes = set()
                for cfg in sdfg.nodes():
                    if sdfg.out_degree(cfg) == 0 and cfg != exit_interface:
                        end_nodes.add(cfg)
                assert len(end_nodes) == 1, f"End nodes: {end_nodes}"
                for end_node in end_nodes:
                    sdfg.add_edge(end_node, exit_interface, dace.sdfg.InterstateEdge())

                exit_interface.add_node(deflatten_lib_node)
                for inname in set(registered_names):
                    iname = "gpu_" + inname if inname in registered_array_names else inname
                    an = exit_interface.add_access(iname)
                    exit_interface.add_edge(
                        an, None, deflatten_lib_node, iname.lower(),dace.Memlet.from_array(iname, sdfg.arrays[iname])
                    )

                    if inname.lower() not in deflatten_lib_node.in_connectors:
                        deflatten_lib_node.add_in_connector(iname.lower())

                for outname in set(
                    [
                        k
                        for k, v in sdfg.arrays.items()
                        if (
                            isinstance(v, dace.data.Structure)
                            or isinstance(v, dace.data.ContainerArray)
                        )
                        and not isinstance(v, dace.data.View)
                    ]
                ):
                    an = dace.nodes.AccessNode(outname)
                    exit_interface.add_node(an)
                    exit_interface.add_edge(
                        an, None, deflatten_lib_node, None, dace.Memlet()
                    )
        else:
            if self._interface_with_struct_copy:
                flatten_lib_node = Flattener(
                    name="flatten",
                    code=self._flattener_codestr,
                    input_names=[],  # [k.lower() for k, v in sdfg.arrays.items() if isinstance(v, dace.data.Structure) and not
                    # isinstance(v, dace.data.View)],
                    output_names=[n.lower() for n in registered_names],
                )
                deflatten_lib_node = Flattener(
                    name="deflatten",
                    code=self._deflattener_codestr,
                    input_names=[n.lower() for n in registered_names],
                    output_names=[],  # [k.lower() for k, v in sdfg.arrays.items() if (isinstance(v, dace.data.Structure)
                    # or isinstance(v, dace.data.ContainerArray)) and not
                    # isinstance(v, dace.data.View)],
                )
                flatten_lib_node.schedule = dace.dtypes.ScheduleType.CPU_Multicore
                deflatten_lib_node.schedule = dace.dtypes.ScheduleType.CPU_Multicore
                # entry_interface = sdfg.add_state("entry_interface")
                # exit_interface = sdfg.add_state("exit_interface")
                start_state = sdfg.start_state
                entry_interface = sdfg.add_state_before(
                    sdfg.start_state,
                    "entry_interface",
                    is_start_block=True,
                    is_start_state=True,
                )
                # sdfg.add_edge(entry_interface, start_state, dace.sdfg.InterstateEdge())

                assert sdfg.start_state == entry_interface
                assert sdfg.start_block == entry_interface

                entry_interface.add_node(flatten_lib_node)
                for inname in set(
                    [
                        k
                        for k, v in sdfg.arrays.items()
                        if (
                            isinstance(v, dace.data.Structure)
                            or isinstance(v, dace.data.ContainerArray)
                        )
                        and not isinstance(v, dace.data.View)
                    ]
                ):
                    an = dace.nodes.AccessNode(inname)
                    entry_interface.add_node(an)
                    entry_interface.add_edge(
                        an, None, flatten_lib_node, None, dace.Memlet.from_array(inname, sdfg.arrays[inname])
                    )
                    sdfg.arrays[inname].storage = dace.StorageType.CPU_Heap

                for outname in set(registered_names):
                    if not self._interface_to_gpu:
                        an = dace.nodes.AccessNode(outname)
                        entry_interface.add_node(an)
                        arr = sdfg.arrays[outname]

                        entry_interface.add_edge(
                            flatten_lib_node,
                            outname.lower(),
                            an,
                            None,
                            dace.Memlet.from_array(outname, sdfg.arrays[outname]),
                        )

                        if outname.lower() not in flatten_lib_node.out_connectors:
                            flatten_lib_node.add_out_connector(outname.lower())

                        sdfg.arrays[outname].storage = dace.StorageType.CPU_Heap
                    else:
                        #sdfg.replace(outname, "gpu_" + outname)
                        assert outname in sdfg.arrays, f"{outname} not in {sdfg.arrays.keys()}"
                        if not isinstance(sdfg.arrays[outname], dace.data.Scalar):
                            an0 = entry_interface.add_access(outname)
                            an1 = entry_interface.add_access("gpu_" + outname)

                            arr = sdfg.arrays[outname]
                            arr.storage = dace.dtypes.StorageType.CPU_Heap
                            arr2 = copy.deepcopy(arr)
                            arr2.storage = dace.dtypes.StorageType.GPU_Global
                            sdfg.add_datadesc("gpu_" + outname, arr2, find_new_name=False)

                            entry_interface.add_edge(
                                flatten_lib_node,
                                outname.lower(),
                                an0,
                                None,
                                dace.Memlet.from_array(outname, sdfg.arrays[outname]),
                            )
                            entry_interface.add_edge(
                                an0, None, an1, None, dace.Memlet.from_array(outname, sdfg.arrays[outname])
                            )
                        else:
                            an0 = entry_interface.add_access(outname)
                            entry_interface.add_edge(
                                flatten_lib_node,
                                outname.lower(),
                                an0,
                                None,
                                dace.Memlet.from_array(outname, sdfg.arrays[outname]),
                            )

                        if outname.lower() not in flatten_lib_node.out_connectors:
                            flatten_lib_node.add_out_connector(outname.lower())

                exit_interface = sdfg.add_state("exit_interface")
                end_nodes = set()
                for cfg in sdfg.nodes():
                    if sdfg.out_degree(cfg) == 0 and cfg != exit_interface:
                        end_nodes.add(cfg)
                assert len(end_nodes) == 1, f"End nodes: {end_nodes}"
                for end_node in end_nodes:
                    sdfg.add_edge(end_node, exit_interface, dace.sdfg.InterstateEdge())

                exit_interface.add_node(deflatten_lib_node)
                for inname in set(registered_names):
                    if self._interface_to_gpu:
                        # If scalar skip
                        if not isinstance(sdfg.arrays[inname], dace.data.Scalar):
                            an0 = exit_interface.add_access(inname)
                            an1 = exit_interface.add_access("gpu_" + inname)

                            exit_interface.add_edge(
                                an1, None, an0, None, dace.Memlet.from_array(inname, sdfg.arrays[inname])
                            )
                            exit_interface.add_edge(
                                an0, None, deflatten_lib_node, inname.lower(), dace.Memlet.from_array(inname, sdfg.arrays[inname])
                            )
                        else:
                            an = exit_interface.add_access(inname)
                            exit_interface.add_edge(
                                an, None, deflatten_lib_node, inname.lower(), dace.Memlet.from_array(inname, sdfg.arrays[inname])
                            )
                    else:
                        an = exit_interface.add_access(inname)
                        exit_interface.add_edge(
                            an, None, deflatten_lib_node, inname.lower(),dace.Memlet.from_array(inname, sdfg.arrays[inname])
                        )

                    if inname.lower() not in deflatten_lib_node.in_connectors:
                        deflatten_lib_node.add_in_connector(inname.lower())

                for outname in set(
                    [
                        k
                        for k, v in sdfg.arrays.items()
                        if (
                            isinstance(v, dace.data.Structure)
                            or isinstance(v, dace.data.ContainerArray)
                        )
                        and not isinstance(v, dace.data.View)
                    ]
                ):
                    an = dace.nodes.AccessNode(outname)
                    exit_interface.add_node(an)
                    exit_interface.add_edge(
                        an, None, deflatten_lib_node, None, dace.Memlet()
                    )

                # Add the end replace the host_name stuff

            if not self._interface_with_struct_copy:
                # Remove structs
                to_rm = []
                for name, desc in sdfg.arrays.items():
                    if isinstance(desc, dace.data.Structure):
                        to_rm.insert(0, name)
                for name in to_rm:
                    sdfg.remove_data(name=name, validate=True)

                if self._save_steps:
                    sdfg.save("data_removed.sdfgz", compress=True)

                nd_to_rm = []
                for s in sdfg.states():
                    for n in s.nodes():
                        if s.in_degree(n) == 0 and s.out_degree(n) == 0:
                            nd_to_rm.insert(0, (s, n))
                for s, n in nd_to_rm:
                    s.remove_node(n)


        # After removing views we might have in and out degree 0
        for node,state in sdfg.all_nodes_recursive():
            if (isinstance(node, dace.nodes.AccessNode) and
                state.in_degree(node) == 0 and state.out_degree(node) == 0):
                state.remove_node(node)

        if self._save_steps:
            sdfg.save("nodes_cleand.sdfgz", compress=True)

        if self._validate or self._validate_all:
            if self._verbose:
                print("All flattening transformatiosn completed, validating")
            sdfg.validate()

            if self._verbose:
                if self._simplify:
                    sdfg.simplify(validate=self._validate, validate_all=self._validate_all)

        if self._clean_container_grous:
            clean_container_groups(sdfg)

        if self._save_steps:
            sdfg.save("flattened.sdfgz", compress=True)

        if self._validate or self._validate_all:
            sdfg.validate()

        sdfg.reset_cfg_list()

        if self._verbose:
            print("Validating again after replacing names and resetting CFG list")

        if self._validate or self._validate_all:
            sdfg.validate()
        if self._simplify:
            sdfg.simplify(validate=self._validate, validate_all=self._validate_all)

        sdfg.save("done.sdfgz", compress=True)

    def _can_be_applied(
        self,
        state: SDFGState,
        sdfg: SDFG,
        src_access: nodes.AccessNode,
        dst_access: nodes.AccessNode,
    ):
        if (
            isinstance(sdfg.arrays[src_access.data], dace.data.View)
            and isinstance(sdfg.arrays[dst_access.data], dace.data.View)
        ) or (
            (
                not isinstance(sdfg.arrays[src_access.data], dace.data.View)
                and not isinstance(sdfg.arrays[dst_access.data], dace.data.View)
            )
        ):
            return False

        # Pattern1: A -> B, A struct, B pointer/view or whatever or
        # Pattern2: B -> A, B pointer/view, A struct
        # Condition: DataGroups have been generated (sdfg.generate_container_groups_from_structs())
        (struct_to_view_pattern, view_to_struct_pattern) = self._get_pattern_type(
            state, sdfg, src_access, dst_access
        )
        if (not struct_to_view_pattern) and (not view_to_struct_pattern):
            return False
        if struct_to_view_pattern and view_to_struct_pattern:
            raise Exception(
                "A -> B and B -> A found in structure + view access (impossible cycle)"
            )

        (struct_access, view_access, struct_data, view_data) = (
            self._assign_src_dst_to_struct_view(sdfg, src_access, dst_access)
        )

        if struct_access is None or view_access is None:
            return False

        if not (isinstance(struct_data, Structure)) and not (
            isinstance(view_data, ContainerArray)
        ):
            return False
        if not (isinstance(view_data, View)):
            return False

        return True

    def _assign_src_dst_to_struct_view(
        self, sdfg: SDFG, src_access: nodes.AccessNode, dst_access: nodes.AccessNode
    ):
        struct_access = None
        view_access = None
        struct_data = None
        view_data = None

        src_data = sdfg.arrays[src_access.data]
        dst_data = sdfg.arrays[dst_access.data]

        if (
            isinstance(src_data, Structure) or isinstance(src_data, ContainerArray)
        ) and not isinstance(src_data, View):
            struct_access = src_access
            struct_data = src_data
        elif (
            isinstance(dst_data, Structure) or isinstance(dst_data, ContainerArray)
        ) and not isinstance(dst_data, View):
            struct_access = dst_access
            struct_data = dst_data

        if isinstance(src_data, View):
            view_access = src_access
            view_data = src_data
        elif isinstance(dst_data, View):
            view_access = dst_access
            view_data = dst_data

        return (struct_access, view_access, struct_data, view_data)

    def _get_pattern_type(
        self,
        state: SDFGState,
        sdfg: SDFG,
        src_access: nodes.AccessNode,
        dst_access: nodes.AccessNode,
    ):
        (struct_access, view_access, struct_data, view_data) = (
            self._assign_src_dst_to_struct_view(sdfg, src_access, dst_access)
        )

        struct_to_view_edges = (
            set(
                [
                    v
                    for _, _, v, _, _ in state.out_edges(struct_access)
                    if v == view_access
                ]
            )
            if struct_access
            else set()
        )
        view_to_struct_edges = (
            set(
                [
                    v
                    for _, _, v, _, _ in state.out_edges(view_access)
                    if v == struct_access
                ]
            )
            if view_access
            else set()
        )

        struct_to_view_pattern = False
        view_to_struct_pattern = False

        if len(struct_to_view_edges) == 0 and len(view_to_struct_edges) == 0:
            return (False, False)
        elif len(struct_to_view_edges) != 0 and len(view_to_struct_edges) != 0:
            raise Exception(
                "A -> B and B -> A found in structure + view access (impossible cycle)"
            )
        elif len(struct_to_view_edges) != 0:
            struct_to_view_pattern = True
        elif len(view_to_struct_edges) != 0:
            view_to_struct_pattern = True

        return (struct_to_view_pattern, view_to_struct_pattern)

    def _get_struct_to_view_view_chain(
        self, state: SDFGState, sdfg: SDFG, first_view_access: nodes.AccessNode
    ):
        view_accesses = [first_view_access]
        current_view_access = first_view_access
        while True:
            out_edges = state.out_edges(current_view_access)
            if len(out_edges) > 1:
                for oe in state.out_edges(current_view_access):
                    if isinstance(oe.dst, nodes.AccessNode) and isinstance(
                        sdfg.arrays[oe.dst.data], dace.data.View
                    ):
                        raise Exception(
                            f"Flattening pass should ensure all paths in the struct tree are chains\n"
                            f"it was not the case at state: {state.label}, node: {current_view_access.label}"
                        )
            if len(out_edges) == 1:
                out_edge = out_edges[0]
                u, uc, v, vc, memlet = out_edge
                if isinstance(v, nodes.AccessNode) and isinstance(
                    sdfg.arrays[v.data], View
                ):
                    current_view_access = v
                    view_accesses.append(v)
                else:
                    return view_accesses
            else:
                return view_accesses

    def _process_and_pad_view_chain(
        self,
        state: SDFGState,
        sdfg: SDFG,
        struct_access: nodes.AccessNode,
        reverse: bool,
    ):
        oes = state.out_edges(struct_access)

        root = struct_access
        stack = [(root, [])]  # Stack stores (node, path_so_far)

        complete_paths = []

        while stack:
            node, path = stack.pop()

            # Add children to the stack in reversed order (to maintain left-to-right order)
            if not reverse:
                children = [
                    e.dst
                    for e in state.out_edges(node)
                    if isinstance(e.dst, dace.nodes.AccessNode)
                    and isinstance(sdfg.arrays[e.dst.data], View)
                ]
            else:
                children = [
                    e.src
                    for e in state.in_edges(node)
                    if isinstance(e.src, dace.nodes.AccessNode)
                    and isinstance(sdfg.arrays[e.src.data], View)
                ]

            if len(children) == 0:
                complete_paths.append(path)
            else:
                for child in list(reversed(children)):
                    if not reverse:
                        stack.append(
                            (child, path + [child])
                        )  # Extend the path for the next n
                    else:
                        stack.append((child, path + [child]))

        # We need all leaf nodes to have in degree 1
        # We need all non leaf and non root nodes to have in and out degree 1
        # For this use DFS get all paths, insert new paths to the graph
        # Remove old nodes
        nodes_to_rm = set()
        edges_to_add = set()
        for path in complete_paths:
            if len(path) == 1:
                continue
            assert len(path) > 1

            old_node_to_new_node = dict()

            for node in path:
                if node not in old_node_to_new_node:
                    new_node = state.add_access(node.data)
                    old_node_to_new_node[node] = new_node
                    nodes_to_rm.add(node)

            # Middle edges
            for node in path[1:-1]:
                if not reverse:
                    ies = [
                        ie
                        for ie in state.in_edges(node)
                        if ie.src == path[path.index(node) - 1]
                    ]
                    oes = [
                        oe
                        for oe in state.out_edges(node)
                        if oe.dst == path[path.index(node) + 1]
                    ]
                else:
                    ies = [
                        ie
                        for ie in state.in_edges(node)
                        if ie.src == path[path.index(node) + 1]
                    ]
                    oes = [
                        oe
                        for oe in state.out_edges(node)
                        if oe.dst == path[path.index(node) - 1]
                    ]
                assert len(ies) == 1
                assert len(oes) == 1
                for ie in ies:
                    edges_to_add.add(
                        (
                            old_node_to_new_node[ie.src],
                            ie.src_conn,
                            old_node_to_new_node[ie.dst],
                            ie.dst_conn,
                            copy.deepcopy(ie.data),
                        )
                    )
                for oe in oes:
                    edges_to_add.add(
                        (
                            old_node_to_new_node[oe.src],
                            oe.src_conn,
                            old_node_to_new_node[oe.dst],
                            oe.dst_conn,
                            copy.deepcopy(oe.data),
                        )
                    )

            if not reverse:
                # Last Node (to Non-Views)
                assert len(state.in_edges(path[-1])) <= 1, f"{state.in_edges(path[-1])}"
                for oe in state.out_edges(path[-1]):
                    edges_to_add.add(
                        (
                            old_node_to_new_node[oe.src],
                            oe.src_conn,
                            oe.dst,
                            oe.dst_conn,
                            copy.deepcopy(oe.data),
                        )
                    )
                for e in state.in_edges(path[-1]):
                    edges_to_add.add(
                        (
                            old_node_to_new_node[e.src],
                            e.src_conn,
                            old_node_to_new_node[e.dst],
                            e.dst_conn,
                            copy.deepcopy(e.data),
                        )
                    )

                # First Node (from Struct)
                ies = [ie for ie in state.in_edges(path[0]) if ie.src == struct_access]
                assert len(ies) == 1
                ie = ies[0]
                edges_to_add.add(
                    (
                        ie.src,
                        ie.src_conn,
                        old_node_to_new_node[ie.dst],
                        ie.dst_conn,
                        copy.deepcopy(ie.data),
                    )
                )
            else:
                assert (
                    len(state.out_edges(path[-1])) <= 1
                ), f"{state.out_edges(path[-1])}"
                # Last Node (from Non-Views)
                for e in state.in_edges(path[-1]):
                    edges_to_add.add(
                        (
                            e.src,
                            e.src_conn,
                            old_node_to_new_node[e.dst],
                            e.dst_conn,
                            copy.deepcopy(e.data),
                        )
                    )
                for e in state.out_edges(path[-1]):
                    edges_to_add.add(
                        (
                            old_node_to_new_node[e.src],
                            e.src_conn,
                            old_node_to_new_node[e.dst],
                            e.dst_conn,
                            copy.deepcopy(e.data),
                        )
                    )

                # First node (to Struct)
                es = [e for e in state.out_edges(path[0]) if e.dst == struct_access]
                assert len(es) == 1
                e = es[0]
                edges_to_add.add(
                    (
                        old_node_to_new_node[e.src],
                        e.src_conn,
                        e.dst,
                        e.dst_conn,
                        copy.deepcopy(e.data),
                    )
                )

        return nodes_to_rm, edges_to_add

    def _process_and_pad_struct_to_view_view_chain(
        self, state: SDFGState, sdfg: SDFG, struct_access: nodes.AccessNode
    ):
        return self._process_and_pad_view_chain(state, sdfg, struct_access, False)

    def _process_and_pad_view_to_struct_view_chain(
        self,
        state: SDFGState,
        sdfg: SDFG,
        struct_access: nodes.AccessNode,
    ):
        return self._process_and_pad_view_chain(state, sdfg, struct_access, True)

    def _get_view_to_struct_view_chain(
        self, state: SDFGState, sdfg: SDFG, last_view_access: nodes.AccessNode
    ):
        view_accesses = [last_view_access]
        current_view_access = last_view_access
        while True:
            in_edges = state.in_edges(current_view_access)
            if len(in_edges) != 1:
                sdfg.save("flattened_failing.sdfgz", compress=True)
            assert len(in_edges) == 1
            in_edge = in_edges[0]
            u, uc, v, vc, memlet = in_edge
            if isinstance(u, nodes.AccessNode) and isinstance(
                sdfg.arrays[u.data], View
            ):
                current_view_access = u
                view_accesses.insert(0, u)
            else:
                return view_accesses

    def _process_edges_internal(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        struct_access: dace.nodes.AccessNode,
        view_chain: typing.List[dace.nodes.AccessNode],
        reverse: bool = False,
    ):
        # Need to keep a map of which view node corresponds to which element
        hierarchy_data = []
        hierarchy_names = []

        hierarchy_names.append(struct_access.data)

        if reverse:
            view_chain = list(reversed(view_chain))

        for node in view_chain:
            edges = state.in_edges(node) if not reverse else state.out_edges(node)
            for ie in edges:
                src, src_conn, dst, dst_conn, memlet = ie
                accesses = memlet.data.split(".")
                assert len(accesses) == 2 or len(accesses) == 1
                # vA.B is access to struct
                # v_AB is access to container array or something else
                access = accesses[-1]

                # If struct access add the member
                if len(accesses) == 2:
                    hierarchy_names.append(access)

                # If container array add the struct inside container array
                if isinstance(sdfg.arrays[node.data], dace.data.ContainerView):
                    hierarchy_names.append(sdfg.arrays[node.data].stype.name)

        chain = [sdfg.arrays[struct_access.data]]
        for name in hierarchy_names[1:]:  # Skip the struct access
            data = chain[-1]
            if isinstance(data, dace.data.ContainerArray):
                chain.append(data.stype)
            if isinstance(data, dace.data.Structure):
                chain.append(data.members[name])
        hierarchy_data = chain

        return hierarchy_names, hierarchy_data

    def _process_edges(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        struct_access: dace.nodes.AccessNode,
        view_chain: typing.List[dace.nodes.AccessNode],
    ):
        return self._process_edges_internal(
            sdfg, state, struct_access, view_chain, False
        )

    def _process_edges_reverse(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        struct_access: dace.nodes.AccessNode,
        view_chain: typing.List[dace.nodes.AccessNode],
    ):
        return self._process_edges_internal(
            sdfg, state, struct_access, view_chain, True
        )

    def _apply(
        self,
        state: SDFGState,
        sdfg: SDFG,
        src_access: nodes.AccessNode,
        dst_access: nodes.AccessNode,
    ):
        src_arr = sdfg.arrays[src_access.data]
        dst_arr = sdfg.arrays[dst_access.data]
        if not isinstance(src_arr, dace.data.Structure) and not isinstance(
            dst_arr, dace.data.Structure
        ):
            raise Exception(
                "Neither source nor destination array is not a structure, at least one needs to be, TODO for the case where it is not"
            )

        removed_nodes = set()

        struct_to_view, view_to_struct = self._get_pattern_type(
            state, sdfg, src_access, dst_access
        )
        if not (struct_to_view or view_to_struct):
            raise Exception("StructToDataGroup not applicable")
        assert not (struct_to_view and view_to_struct)

        if struct_to_view:
            struct_access = src_access
            view_access = dst_access
        else:  # view_to_struct
            assert view_to_struct
            view_access = src_access
            struct_access = dst_access

        # View chain is either struct -> view -> view -> ... -> view -> dst (not a view like tasklet)
        # or src -> view -> ... -> view -> struct (src is e.g. tasklet)
        view_chain = (
            self._get_struct_to_view_view_chain(state, sdfg, view_access)
            if struct_to_view
            else self._get_view_to_struct_view_chain(state, sdfg, view_access)
        )

        assert len(view_chain) >= 1
        name_hierarchy = []

        # Create the sequence of accesses A.B[4].C becomes [A, B, C]
        if struct_to_view:
            name_hierarchy, member_hierarchy = self._process_edges(
                sdfg, state, struct_access, view_chain
            )
            struct_to_view_edges = [
                e for e in state.out_edges(struct_access) if e.dst == view_chain[0]
            ]

        if view_to_struct:
            view_to_struct_edges = [
                e for e in state.in_edges(struct_access) if e.src == view_chain[-1]
            ]
            name_hierarchy, member_hierarchy = self._process_edges_reverse(
                sdfg, state, struct_access, view_chain
            )

        # Get the SoA flattened container name
        demangled_name = get_demangled_container_group_member_name(sdfg, name_hierarchy)

        an = nodes.AccessNode(data=demangled_name)

        if self._verbose:
            print(
                "Source/Destination Struct:",
                struct_access,
                "SV:",
                struct_to_view,
                "VS:",
                view_to_struct,
            )
            print(
                "Applying to:",
                (
                    [struct_access] + view_chain
                    if struct_to_view
                    else view_chain + [struct_access]
                ),
            )
            print(
                "Source/Destination struct is: ",
                struct_access.data,
                type(sdfg.arrays[struct_access.data]),
            )

        # If the length is less than 2 then we have only one view and one source struct
        if self._flattening_mode == ContainerGroupFlatteningMode.StructOfArrays:
            memlet_shape = ()
            # We need calculate the dimension of the new array
            # The shape and memlet expression calculations differ
            if struct_to_view:
                assert len(struct_to_view_edges) == 1
                for vc in [struct_access] + view_chain[:-1]:
                    if state.out_degree(vc) > 0:
                        _dst_edge = state.out_edges(vc)[0]
                        if isinstance(
                            sdfg.arrays[vc.data], dace.data.Structure
                        ) and isinstance(
                            sdfg.arrays[_dst_edge.dst.data], dace.data.ContainerArray
                        ):
                            continue
                        if isinstance(sdfg.arrays[vc.data], dace.data.Structure):
                            if _dst_edge.data.subset.ranges == [(0, 0, 1)]:
                                continue
                        if isinstance(sdfg.arrays[vc.data], dace.data.Scalar):
                            continue
                        # If it is an array to array view continue, ti should not be added to the shape
                        if not isinstance(
                            sdfg.arrays[vc.data],
                            (dace.data.Structure, dace.data.ContainerArray),
                        ) and not isinstance(
                            sdfg.arrays[_dst_edge.dst.data],
                            (dace.data.ContainerArray, dace.data.Structure),
                        ):
                            continue
                        memlet_shape += tuple(_dst_edge.data.subset.ranges)

            if view_to_struct:
                assert len(view_to_struct_edges) == 1
                for vc in [struct_access] + list(reversed(view_chain[1:])):
                    if state.in_degree(vc) > 0:
                        _src_edge = state.in_edges(vc)[0]
                        if isinstance(
                            sdfg.arrays[vc.data], dace.data.ContainerArray
                        ) and isinstance(
                            sdfg.arrays[_src_edge.src.data], dace.data.Structure
                        ):
                            continue
                        if isinstance(
                            sdfg.arrays[_src_edge.src.data], dace.data.Structure
                        ):
                            if _src_edge.data.subset.ranges == [(0, 0, 1)]:
                                continue
                        if isinstance(
                            sdfg.arrays[_src_edge.src.data], dace.data.Scalar
                        ):
                            continue
                        # If it is an array to array view continue, ti should not be added to the shape
                        if not isinstance(
                            sdfg.arrays[vc.data],
                            (dace.data.Structure, dace.data.ContainerArray),
                        ) and not isinstance(
                            sdfg.arrays[_src_edge.src.data],
                            (dace.data.ContainerArray, dace.data.Structure),
                        ):
                            continue
                        memlet_shape += tuple(_src_edge.data.subset.ranges)
        else:
            raise Exception("ArrayOfStructs mode is not implemented yet")

        assert isinstance(sdfg.arrays[demangled_name], dace.data.Array)
        if memlet_shape == ():
            assert isinstance(sdfg.arrays[demangled_name], dace.data.Scalar) or (
                isinstance(sdfg.arrays[demangled_name], dace.data.Array)
                and len(sdfg.arrays[demangled_name].shape) == 1
            )
            memlet_shape = [(0, 0, 1)]
        mc = dace.memlet.Memlet(
            subset=dace.subsets.Range(memlet_shape), data=demangled_name
        )

        # If Struct -> View -> Dst:
        # Then Struct (uc) -> (None) \ View \ (None) -> (vc) Dst
        # Becomes NewData (None) -> (vc) Dst

        # If View -> Struct -> Dst:
        # Then Src (uc) -> (None) \ View \ (None) -> (vc) Struct
        # Becomes Src (uc) -> (None) NewData

        # Simplify manages to remove this
        # if a1 -> AB -> a1 and a2 -> AB -> a2
        # we need to use the new data we inserted for a1 in both case and not reenter data
        if struct_access.guid not in self._struct_replacements:
            self._struct_replacements[struct_access.guid] = set()

        if an.data not in [
            a.data for a in self._struct_replacements[struct_access.guid]
        ]:
            state.add_node(an)
            self._struct_replacements[struct_access.guid].add(an)
        else:
            an = [
                a
                for a in self._struct_replacements[struct_access.guid]
                if a.data == an.data
            ][0]

        view_name = None
        if struct_to_view:
            state.add_edge(an, None, view_chain[-1], "views", mc)
        else:  # view_to_struct
            state.add_edge(view_chain[0], "views", an, None, mc)

        # Replace all occurences
        if struct_to_view:
            repl_before = view_chain[-1].data
        if view_to_struct:
            repl_before = view_chain[0].data
        repl_after = demangled_name
        if self._verbose:
            print("Adding to replace list:", repl_before, "->", repl_after)

        # Clean-up
        if struct_to_view:
            for view_node in view_chain[:-1]:
                state.remove_node(view_node)
                removed_nodes.add(view_node)
            for oe in state.out_edges(struct_access):
                if oe.dst == view_chain[0]:
                    state.remove_edge(oe)

            if (len(state.in_edges(struct_access)) == 0) and (
                len(state.out_edges(struct_access)) == 0
            ):
                removed_nodes.add(struct_access)
        elif view_to_struct:
            for view_node in view_chain[1:]:
                state.remove_node(view_node)
                removed_nodes.add(view_node)
            for ie in state.in_edges(struct_access):
                if ie.src == view_chain[-1]:
                    state.remove_edge(ie)
            if (len(state.in_edges(struct_access)) == 0) and (
                len(state.out_edges(struct_access)) == 0
            ):
                removed_nodes.add(struct_access)

        if (len(state.in_edges(struct_access)) == 0) and (
            len(state.out_edges(struct_access)) == 0
        ):
            state.remove_node(struct_access)
        self._call += 1

        return removed_nodes, (repl_before, repl_after)

    def _get_src_dst(self, state: SDFGState, n1: nodes.Any, n2: nodes.Any):
        n1_to_n2 = [e.dst for e in state.out_edges(n1) if e.dst == n2]
        n2_to_n1 = [e.dst for e in state.out_edges(n2) if e.dst == n1]
        if len(n2_to_n1) == 0 and len(n1_to_n2) == 0:
            raise Exception("E1")
        elif len(n2_to_n1) != 0 and len(n1_to_n2) != 0:
            raise Exception("E2")
        elif len(n2_to_n1) == 0:
            assert len(n1_to_n2) > 0
            return (n1, n2)
        else:
            assert len(n2_to_n1) > 0
            return (n2, n1)

    def annotates_memlets():
        return False

# =================================================================================================
# Utility functions

def replace_length_one_arrays_with_scalars(sdfg: dace.SDFG):
    arrays_that_have_become_scalars = dict()
    for name, desc in sdfg.arrays.items():
        if (isinstance(desc, dace.data.Array) and len(desc.shape) == 1 and desc.shape[0] == 1
            and desc.transient and not isinstance(desc, dace.data.Scalar) and
            not isinstance(desc, dace.data.View)):
            scalarized = dace.data.Scalar(
                dtype=desc.dtype,
                transient=True,
                storage=dace.dtypes.StorageType.Register,
                allow_conflicts=desc.allow_conflicts,
                location=desc.location,
                lifetime=desc.lifetime,
                debuginfo=desc.debuginfo,
            )
            arrays_that_have_become_scalars[name] = (name, scalarized)

    for old_name, (new_name, desc) in arrays_that_have_become_scalars.items():
        sdfg.remove_data(old_name, validate=False)
    for old_name, (new_name, desc) in arrays_that_have_become_scalars.items():
        sdfg.add_datadesc(new_name, desc, find_new_name=False)
        assert not isinstance(desc, dace.data.View)

    for state in sdfg.states():
        nodes_to_rm = set()

        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                replace_length_one_arrays_with_scalars(node.sdfg)

            # Update edges per ndoe as in a change it might have an impact
            edges_to_add = set()
            edges_to_rm = set()

            if isinstance(node, dace.nodes.AccessNode):
                if node.data in arrays_that_have_become_scalars:
                    new_name, scalarized = arrays_that_have_become_scalars[node.data]
                    an = state.add_access(new_name)

                    for ie in state.in_edges(node):
                        mem = copy.deepcopy(ie.data)
                        if mem.data == node.data:
                            mem.data = new_name
                        edges_to_add.add((ie.src, ie.src_conn, an, ie.dst_conn, mem))

                    for oe in state.out_edges(node):
                        mem = copy.deepcopy(oe.data)
                        if mem.data == node.data:
                            mem.data = new_name
                        edges_to_add.add((an, oe.src_conn, oe.dst, oe.dst_conn, mem))

                    for ie in state.in_edges(node):
                        edges_to_rm.add(ie)

                    for oe in state.out_edges(node):
                        edges_to_rm.add(oe)

                    nodes_to_rm.add(node)

            for edge in edges_to_rm:
                state.remove_edge(edge)
            for edge in edges_to_add:
                state.add_edge(*edge)
        for node in nodes_to_rm:
            state.remove_node(node)


    # Remove all views coming out of these scalars (but also replace the names everywhere)
    repl_dict = dict()
    for state in sdfg.states():
        edges_to_add = set()
        edges_to_rm = set()
        nodes_to_rm = set()

        for node in state.nodes():
            if isinstance(node, dace.nodes.AccessNode):
                if node.data in arrays_that_have_become_scalars:
                    for oe in state.out_edges(node):
                        if isinstance(sdfg.arrays[oe.dst.data], dace.data.View):
                            assert oe.dst_conn == "views"
                            for oe2 in state.out_edges(oe.dst):
                                edges_to_rm.add(oe2)
                                mem = copy.deepcopy(mem)
                                mem.data = node.data
                                assert oe.src_conn is None
                                edges_to_add.add((node, oe.src_conn, oe2.dst, oe2.dst_conn, mem))
                            edges_to_rm.add(oe)
                            nodes_to_rm.add(oe.dst)
                        if node.data != oe.dst.data:
                            repl_dict[oe.dst.data] = node.data

        for edge in edges_to_add:
            state.add_edge(*edge)
        for edge in edges_to_rm:
            state.remove_edge(edge)
        for node in nodes_to_rm:
            state.remove_node(node)

    for name in repl_dict:
        if name in sdfg.arrays:
            sdfg.remove_data(name, validate=False)

    for edge, _ in sdfg.all_edges_recursive():
        if isinstance(edge, InterstateEdge):
            edge.data.replace_dict(repl_dict)
    for edge in sdfg.all_interstate_edges(recursive=True):
        edge.data.replace_dict(repl_dict)

    # For all scalars remove zero indices
    scalar_names = [name for name, arr in sdfg.arrays.items() if isinstance(arr, dace.data.Scalar)]
    for edge, _ in sdfg.all_edges_recursive():
        if isinstance(edge, InterstateEdge):
            for dst in scalar_names:
                edge.data.replace_complex_iedge(dst, dst, remove_zero_index=True)
    for edge in sdfg.all_interstate_edges(recursive=True):
        for dst in scalar_names:
            edge.data.replace_complex_iedge(dst, dst, remove_zero_index=True)

    for src, dst in repl_dict.items():
        rename_on_if_conds(sdfg, src, dst, recursive=False)

    sdfg.replace_dict(repl_dict)



def rename_on_if_conds(sdfg: dace.SDFG, src: str, dst: str, recursive=False):
    gpu_host_name_map = {src: dst}

    for _, node in enumerate(sdfg.nodes()) if not recursive else sdfg.all_nodes_recursive():
        if not isinstance(node, ConditionalBlock):
            continue

        for b in node.branches:
            if b[0] is None:
                continue
            if isinstance(b[0].code, list):
                for i, el in enumerate(b[0].code):
                    if isinstance(el, str):
                        for src,dst in gpu_host_name_map.items():
                            b[0].code[i] = b[0].code[i].replace(src,dst)
                    else:
                        def replace_x_with_y(expr: ast.Expr, repl_dict) -> ast.Expr:
                            expr_str = ast.unparse(expr).strip()
                            for src, dst in repl_dict.items():
                                modified_str = expr_str.replace(src, dst)
                            return ast.parse(modified_str, mode="eval").body
                        b[0].code[i] = replace_x_with_y(b[0].code[i], gpu_host_name_map)
            else:
                assert isinstance(b[0].code, str)
                for src,dst in gpu_host_name_map.items():
                    b[0].code = b[0].code.replace(src, dst)


def clean_trivial_views(sdfg: dace.SDFG):
    repl_dict = dict()
    for state in sdfg.states():
        edges_to_rm = set()
        edges_to_add = set()
        nodes_to_rm = set()
        for node in state.nodes():
            if isinstance(node, dace.nodes.AccessNode):
                if isinstance(node, dace.nodes.NestedSDFG):
                    clean_container_groups(node.sdfg)

        edges = state.edges()
        while len(edges) > 0:
            edge = edges.pop(0)
            # AccessNode ->(views) AccessNode
            # And the view covers the whole dimension of left hand array

            # A -> B B is view of A
            if isinstance(edge.src, dace.nodes.AccessNode) and isinstance(
                edge.dst, dace.nodes.AccessNode
            ) and edge.src_conn is None and edge.dst_conn == "views":
                subset = edge.data.subset
                shape_as_subset = dace.subsets.Range.from_array(
                    sdfg.arrays[edge.src.data]
                )
                if subset == shape_as_subset:
                    # This is an edge that goes A -> B where B is a view of A
                    # And the view B is reduntant

                    # Remove edge A -> B
                    # Reroute all edges of type B -> C to be A -> C
                    # Update edges

                    repl_dict[edge.dst.data] = edge.src.data

                    # Rm view node
                    for oe in state.out_edges(edge.dst):
                        mem = copy.deepcopy(oe.data)
                        mem.data = edge.src.data
                        state.add_edge(
                            edge.src, edge.src_conn, oe.dst, oe.dst_conn, mem
                        )
                        state.remove_edge(
                            oe
                        )
                    state.remove_edge(edge)
                    state.remove_node(edge.dst)
                    edges = state.edges()

                # A -> B A is view of B
            elif isinstance(edge.src, dace.nodes.AccessNode) and isinstance(
                edge.dst, dace.nodes.AccessNode
            ) and edge.dst_conn is None and edge.src_conn == "views":
                subset = edge.data.subset
                shape_as_subset = dace.subsets.Range.from_array(
                    sdfg.arrays[edge.dst.data]
                )
                if subset == shape_as_subset:

                    repl_dict[edge.src.data] = edge.dst.data

                    # Rm view node
                    for ie in state.in_edges(edge.src):
                        mem = copy.deepcopy(ie.data)
                        mem.data = edge.dst.data
                        state.add_edge(
                            ie.src, ie.src_conn, edge.dst, edge.dst_conn, mem
                        )
                        state.remove_edge(
                            ie
                        )
                    state.remove_edge(edge)
                    state.remove_node(edge.src)
                    edges = state.edges()

        for edge in edges_to_rm:
            state.remove_edge(edge)
        for edge in edges_to_add:
            state.add_edge(*edge)
        for node in nodes_to_rm:
            state.remove_node(node)

        # Some view access ndoes might have become
        # without in or out edges
        for node in state.nodes():
            if (isinstance(node, dace.nodes.AccessNode) and
                state.in_degree(node) == 0 and state.out_degree(node) == 0):
                state.remove_node(node)

    for name in repl_dict:
        if name in sdfg.arrays:
            sdfg.remove_data(name, validate=False)

    for edge, _ in sdfg.all_edges_recursive():
        if isinstance(edge, InterstateEdge):
            edge.data.replace_dict(repl_dict)
    for edge in sdfg.all_interstate_edges(recursive=True):
        edge.data.replace_dict(repl_dict)

    for src, dst in repl_dict.items():
        rename_on_if_conds(sdfg, src, dst, recursive=False)

    sdfg.replace_dict(repl_dict)





# =================================================================================================
