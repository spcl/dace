# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""

import copy
from typing import Any, Dict
import dace
from dace.sdfg import SDFG, NestedDict, SDFGState
from dace.properties import make_properties
from dace.sdfg import nodes
from dace.data import Structure, View
from dace.transformation import pass_pipeline as ppl

from collections import OrderedDict
from typing import Set, Union
import typing
from dace import data
from dace.data import Data
from dace import serialize, symbolic
from dace.properties import OrderedDictProperty, Property, make_properties
from enum import Enum

import numpy
import sympy

from dace.transformation.passes.array_elimination import ArrayElimination


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

    def __init__(self, name, is_cg, is_ca, shape):
        self.name = name
        self.members = OrderedDict()
        self.is_cg = is_cg
        self.is_ca = is_ca
        self.shape = shape
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
        return f"ContainerGroup(name='{self.name}', is_cg={self.is_cg}, is_ca={self.is_ca}, shape={self.shape}, members={{ {members_repr} }})"

    def __str__(self):
        return self.__repr__()

    def _soa_from_struct(self, name, structure, acc_shape):
        self._add_members(name, structure, acc_shape=None)

    @classmethod
    def from_struct(
        cls,
        name: str,
        struct_or_container_array: typing.Union[data.Structure, data.ContainerArray],
        is_cg: bool,
        is_ca: bool,
        shape: tuple,
    ) -> "ContainerGroup":
        dg = cls(name=name, is_cg=is_cg, is_ca=is_ca, shape=shape)
        assert is_cg ^ is_ca

        if isinstance(struct_or_container_array, data.Structure):
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
                    )
                elif isinstance(member, data.ContainerArray):
                    new_member = cls.from_struct(
                        name=member_name,
                        struct_or_container_array=member,
                        is_cg=False,
                        is_ca=True,
                        shape=member.shape,
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
        else:
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

        return dg


# ===================================================================================================
# Functionality to Register Container Groups to an SDFG
# ===================================================================================================
def register_container_group_members(
    sdfg: dace.SDFG, flattening_mode: ContainerGroupFlatteningMode
):
    for name, dg in sdfg.container_groups.items():
        _register_container_group_members(
            sdfg=sdfg,
            flattening_mode=flattening_mode,
            container_group_or_array=dg,
            prefix_name=f"__CG_{name}",
            acc_shape=(),
        )


def _register_container_group_members(
    sdfg,
    flattening_mode,
    container_group_or_array: typing.Union["ContainerGroup", dace.data.ContainerArray],
    prefix_name: str,
    acc_shape: tuple,
):
    if flattening_mode == ContainerGroupFlatteningMode.StructOfArrays:
        if isinstance(container_group_or_array, ContainerGroup):
            container_group = container_group_or_array
            for name, member in container_group.members.items():
                if isinstance(member, ContainerGroup):
                    if member.is_cg:
                        dg_prefix = prefix_name + f"__CG_{member.name}"
                    else:
                        dg_prefix = prefix_name + f"__CA_{member.name}"
                        acc_shape += member.shape
                    _register_container_group_members(
                        sdfg=sdfg,
                        flattening_mode=flattening_mode,
                        container_group_or_array=member,
                        prefix_name=dg_prefix,
                        acc_shape=acc_shape,
                    )
                elif isinstance(member, dace.data.ContainerArray):
                    assert False
                else:
                    # Add the dimensions accumulated while iterating from root to the leaf node of the trees
                    member_demangled_name = prefix_name + f"__m_{name}"
                    if isinstance(member, dace.data.Scalar):
                        datadesc = dace.data.Array(
                            dtype=member.dtype,
                            shape=acc_shape,
                            transient=member.transient,
                            allow_conflicts=member.allow_conflicts,
                            storage=member.storage,
                            location=member.location,
                            may_alias=member.may_alias,
                            lifetime=member.lifetime,
                            debuginfo=member.debuginfo,
                            start_offset=member.start_offset,
                        )
                    elif isinstance(member, dace.data.Array):
                        datadesc = dace.data.Array(
                            dtype=member.dtype,
                            shape=acc_shape + member.shape,
                            transient=member.transient,
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
        elif isinstance(container_group_or_array, dace.data.ContainerArray):
            assert False
        else:
            raise Exception("?")
    elif flattening_mode == ContainerGroupFlatteningMode.ArrayOfStructs:
        raise Exception("TODO Support for ArrayOfStructs Flattening")
    else:
        raise Exception("Unsupported Flattening Mode")


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
            raise Exception(
                f"Name Hierarchy {name_hierarchy} Not in ContainerGroups {sdfg.container_groups}, {sdfg._arrays} 1"
            )

    if (
        i == len(name_hierarchy) - 1
        and len(current_dg.members) == 1
        and "Leaf" in current_dg.members
    ):
        demangled_name += f"__m_Leaf"
        return demangled_name
    raise Exception(
        f"Name Hierarchy {name_hierarchy} Not in ContainerGroups {sdfg.container_groups}, {sdfg._arrays} 2"
    )


def generate_container_groups_from_structs(
    sdfg: dace.SDFG, flattening_mode: ContainerGroupFlatteningMode
):
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
            )
            sdfg.container_groups[dg_name] = dg


def clean_container_groups(sdfg: dace.SDFG):
    #assert hasattr(
    #    sdfg, "container_groups"
    #), "SDFG does not have a field named 'container_groups member, clean container groups called'"
    if hasattr(sdfg, "container_groups"):
        sdfg.container_groups = None


def add_container_group_desc(sdfg: dace.SDFG, name: str, container_group_desc: ContainerGroup, find_new_name=False) -> str:
    if not isinstance(name, str):
        raise TypeError("Data descriptor name must be a string. Got %s" % type(name).__name__)

    if find_new_name:
        name = sdfg._find_new_name(name)
        name = name.replace('.', '_')
        if sdfg.is_name_used(name):
            name = sdfg._find_new_name(name)
    else:
        if name in sdfg.arrays:
            raise FileExistsError(f'Data group descriptor "{name}" already exists in SDFG')
        if name in sdfg.symbols:
            raise FileExistsError(f'Can not create data group descriptor "{name}", the name is used by a symbol.')
        if name in sdfg._subarrays:
            raise FileExistsError(f'Can not create data group descriptor "{name}", the name is used by a subarray.')
        if name in sdfg._rdistrarrays:
            raise FileExistsError(f'Can not create data group descriptor "{name}", the name is used by a RedistrArray.')
        if name in sdfg._pgrids:
            raise FileExistsError(f'Can not create data group descriptor "{name}", the name is used by a ProcessGrid.')

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

def add_container_group(sdfg: dace.SDFG,
                    name: str,
                    find_new_name: bool = False) -> typing.Tuple[str, "ContainerGroup"]:
    dg_desc = ContainerGroup(name)
    return add_container_group_desc(sdfg, name, dg_desc, find_new_name=find_new_name), dg_desc

# ===================================================================================================


@make_properties
class StructToContainerGroups(ppl.Pass):
    def __init__(
        self,
        flattening_mode: ContainerGroupFlatteningMode = ContainerGroupFlatteningMode.StructOfArrays,
        simplify: bool = True,
        validate: bool = True,
        validate_all: bool = False,
        clean_container_grous: bool = True,
        save_steps: bool = False
    ):
        if flattening_mode != ContainerGroupFlatteningMode.StructOfArrays:
            raise Exception("Only StructOfArrays is supported")
        super().__init__()
        self._simplify = simplify
        self._validate = validate
        self._validate_all = validate_all
        self._clean_container_grous = clean_container_grous
        self._access_names_map = dict()
        self._data_connected_to_vsv_struct = dict()
        self._flattening_mode = flattening_mode
        self._call = 0
        self._save_steps = save_steps

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

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> int:
        generate_container_groups_from_structs(sdfg, self._flattening_mode)
        register_container_group_members(sdfg, self._flattening_mode)

        # A -> B both access nodes, this should trigger the further check whether we can apply
        i = 0
        name_replacements = dict()
        for state in sdfg.states():
            nodes = state.nodes()
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
                                newly_removed_nodes, (oldname, (newname, dimstopad, dims)) = self._apply(
                                    state, sdfg, src_access, dst_access
                                )
                                sdfg.save(f"apply{i}.sdfg")
                                removed_nodes = removed_nodes.union(newly_removed_nodes)
                                name_replacements[oldname] = (newname, dimstopad, dims)

        # View -> Struct -> View patterns result with disconnected compenets reconnect them with saved info
        for state in sdfg.states():
            nodes = state.nodes()
            for node in nodes:
                if node not in state.nodes():
                    continue
                for (
                    in_connected_nodes,
                    out_connected_nodes,
                ) in self._data_connected_to_vsv_struct.values():
                    assert (
                        len(in_connected_nodes) <= 1 and len(out_connected_nodes) <= 1
                    )
                    if len(in_connected_nodes) == 1 and len(out_connected_nodes) == 1:
                        if node in in_connected_nodes:
                            src = node
                            dst = out_connected_nodes[0]
                            for oe in state.out_edges(dst):
                                assert oe.src_conn is None
                                state.add_edge(
                                    src,
                                    None,
                                    oe.dst,
                                    oe.dst_conn,
                                    copy.deepcopy(oe.data),
                                )
                            state.remove_node(dst)
                        elif node in out_connected_nodes:
                            continue

        # Remove structs
        to_rm = []
        for name, desc in sdfg.arrays.items():
            if isinstance(desc, dace.data.Structure):
                to_rm.insert(0, name)
        for name in to_rm:
            sdfg.remove_data(name=name, validate=True)

        if self._save_steps:
            sdfg.save("data_removed.sdfg")

        # Map names
        #print("Replace names:", name_replacements)
        # Clean old names if they are in arrays
        #for oldname in name_replacements:
        #    if oldname in sdfg.arrays:
        #        del sdfg.arrays[oldname]
        # Replace names using removed views
        #sdfg.replace_dict(name_replacements)
        #
        #if self._save_steps:
        #    sdfg.save("name_replaced.sdfg")
        for oldname in name_replacements:
            if oldname in sdfg.arrays:
                del sdfg.arrays[oldname]

        # TODO replace ALL occurences of access to oldname with newname + dimensions
        # Including if/for loop conditions, interstate edges (cond, assignment)
        # If access to a previous view then pad their dimensions accorindgly too
        for oldname, (newname, dimstopad, dims) in name_replacements.items():
            pass

        """
        # Clean Mapped Views (Views within data groups)
        for state in sdfg.states():
            for edge in state.edges():
                if edge.data.data in self._access_names_map:
                    data_name = self._access_names_map[edge.data.data]
                    edge.data.data = data_name

        if self._save_steps:
            sdfg.save("mapped_views_cleaned.sdfgz")
        """


        nd_to_rm = []
        for s in sdfg.states():
            for n in s.nodes():
                if s.in_degree(n) == 0 and s.out_degree(n) == 0:
                    nd_to_rm.insert(0,(s,n))
        for (s,n) in nd_to_rm:
            s.remove_node(n)

        if self._save_steps:
            sdfg.save("nodes_cleand.sdfg")

        if self._simplify:
            sdfg.simplify(self._validate, self._validate_all)

        if self._clean_container_grous:
            clean_container_groups(sdfg)

    def _can_be_applied(
        self,
        state: SDFGState,
        sdfg: SDFG,
        src_access: nodes.AccessNode,
        dst_access: nodes.AccessNode,
    ):
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

        if not (isinstance(struct_data, Structure)):
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

        if isinstance(src_data, Structure):
            struct_access = src_access
            struct_data = src_data
        elif isinstance(dst_data, Structure):
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
            assert len(out_edges) <= 1 # If out degree is 0, then it is probably used in the next state
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

    def _get_view_to_struct_view_chain(
        self, state: SDFGState, sdfg: SDFG, last_view_access: nodes.AccessNode
    ):
        view_accesses = [last_view_access]
        current_view_access = last_view_access
        while True:
            in_edges = state.in_edges(current_view_access)
            assert len(in_edges) == 1
            out_edge = in_edges[0]
            u, uc, v, vc, memlet = out_edge
            if isinstance(u, nodes.AccessNode) and isinstance(
                sdfg.arrays[u.data], View
            ):
                current_view_access = u
                view_accesses.insert(0, u)
            else:
                return view_accesses


    def _process_edges(self,
                       sdfg : dace.SDFG,
                       state: dace.SDFGState,
                       view_chain: typing.List[dace.nodes.AccessNode]):
        #(view_chain, view_chain[0])
        first_access = view_chain[0]
        ie = state.in_edges(first_access)[0]
        src, src_conn, dst, dst_conn, memlet = ie
        data = memlet.data
        #print(data)
        if "." in data:
            all_data = data.split(".")
        else:
            all_data = [data]
        ite = 0
        root = None
        member_hierarchy = []
        name_hierarchy = []
        view_to_name_map = dict()
        for access in all_data:
            if ite == 0:
                root = sdfg.arrays[access]
                name_hierarchy.append(access)
                member_hierarchy.append((access, root))
            else:
                last_member = member_hierarchy[-1][1]
                if isinstance(last_member, dace.data.ContainerArray):
                    name_hierarchy.append(last_member.stype.members[access])
                    member_hierarchy.append((last_member.stype.members[access].name, last_member.stype.members[access]))
                elif isinstance(last_member, dace.data.Structure):
                    name_hierarchy.append(access)
                    member_hierarchy.append((access, last_member.members[access]))
                else:
                    raise Exception("This should not happen.")
            ite += 1
        view_to_name_map[first_access.data] = name_hierarchy[-1]

        #print("VC2", view_chain[1:])
        for access_node in view_chain[1:]:
            for ie in state.in_edges(access_node):
                src, src_conn, dst, dst_conn, memlet = ie
                data = memlet.data
                if "." in data:
                    all_data = data.split(".")
                else:
                    all_data = [data]
                if len(all_data) == 2:
                    pass
                if len(all_data) == 1:
                    pass
                #print(member_hierarchy)
                last_member = member_hierarchy[-1][1]
                last_name = name_hierarchy[-1]
                if isinstance(last_member, dace.data.ContainerArray):
                    #print("CG", view_to_name_map, member_hierarchy, access)
                    access = all_data[0]
                    true_name = view_to_name_map[access]
                    assert len(all_data) == 1
                    #print(last_member.stype.name, true_name, access)
                    name_hierarchy.append(last_member.stype.name)
                    member_hierarchy.append((last_member.stype.name, last_member.stype))
                    assert access_node != view_chain[-1]
                    if access_node != view_chain[-1]:
                        view_to_name_map[access_node.data] = last_member.stype.name
                elif isinstance(last_member, dace.data.Structure):
                    assert len(all_data) == 2
                    access1 = all_data[0]
                    access2 = all_data[1]
                    if access_node != view_chain[-1]:
                        name_hierarchy.append(last_member.members[access2].name)
                        member_hierarchy.append((last_member.members[access2].name, last_member.members[access2]))
                        view_to_name_map[access_node.data] = last_member.members[access2].name
                    else:
                        name_hierarchy.append(access2)
                        member_hierarchy.append((access2, last_member.members[access2]))
                        view_to_name_map[access_node.data] = access2
                else:
                    raise Exception("This should not happen.")

        return name_hierarchy, member_hierarchy

    def _process_edges_reverse(self,
                       sdfg : dace.SDFG,
                       state: dace.SDFGState,
                       view_chain: typing.List[dace.nodes.AccessNode]):
        #print(view_chain, view_chain[0])
        first_access = view_chain[-1]
        ie = state.out_edges(first_access)[0]
        src, src_conn, dst, dst_conn, memlet = ie
        data = memlet.data
        #print(data)
        if "." in data:
            all_data = data.split(".")
        else:
            all_data = [data]
        ite = 0
        root = None
        member_hierarchy = []
        name_hierarchy = []
        view_to_name_map = dict()
        for access in all_data:
            if ite == 0:
                root = sdfg.arrays[access]
                name_hierarchy.append(access)
                member_hierarchy.append((access, root))
            else:
                last_member = member_hierarchy[-1][1]
                if isinstance(last_member, dace.data.ContainerArray):
                    name_hierarchy.append(last_member.stype.members[access])
                    member_hierarchy.append((last_member.stype.members[access].name, last_member.stype.members[access]))
                elif isinstance(last_member, dace.data.Structure):
                    name_hierarchy.append(access)
                    member_hierarchy.append((access, last_member.members[access]))
                else:
                    raise Exception("This should not happen.")
            ite += 1
        view_to_name_map[first_access.data] = name_hierarchy[-1]

        #print("VC2", view_chain[1:])
        for access_node in reversed(view_chain[:-1]):
            for ie in state.out_edges(access_node):
                src, src_conn, dst, dst_conn, memlet = ie
                data = memlet.data
                if "." in data:
                    all_data = data.split(".")
                else:
                    all_data = [data]
                if len(all_data) == 2:
                    pass
                if len(all_data) == 1:
                    pass
                #print(member_hierarchy)
                last_member = member_hierarchy[-1][1]
                last_name = name_hierarchy[-1]
                if isinstance(last_member, dace.data.ContainerArray):
                    #print("CG", view_to_name_map, member_hierarchy, access)
                    access = all_data[0]
                    true_name = view_to_name_map[access]
                    assert len(all_data) == 1
                    #print(last_member.stype.name, true_name, access)
                    name_hierarchy.append(last_member.stype.name)
                    member_hierarchy.append((last_member.stype.name, last_member.stype))
                    assert access_node != view_chain[-1]
                    if access_node != view_chain[-1]:
                        view_to_name_map[access_node.data] = last_member.stype.name
                elif isinstance(last_member, dace.data.Structure):
                    assert len(all_data) == 2
                    access1 = all_data[0]
                    access2 = all_data[1]
                    if access_node != view_chain[0]:
                        name_hierarchy.append(last_member.members[access2].name)
                        member_hierarchy.append((last_member.members[access2].name, last_member.members[access2]))
                        view_to_name_map[access_node.data] = last_member.members[access2].name
                    else:
                        name_hierarchy.append(access2)
                        member_hierarchy.append((access2, last_member.members[access2]))
                        view_to_name_map[access_node.data] = access2
                else:
                    raise Exception("This should not happen.")

        return name_hierarchy, member_hierarchy

    def _apply(
        self,
        state: SDFGState,
        sdfg: SDFG,
        src_access: nodes.AccessNode,
        dst_access: nodes.AccessNode,
    ):
        src_arr = sdfg.arrays[src_access.data]
        dst_arr = sdfg.arrays[dst_access.data]
        if not isinstance(src_arr, dace.data.Structure) and not isinstance(dst_arr, dace.data.Structure):
            raise Exception("Neither source nor destination array is not a structure, at least one needs to be, TODO for the case where it is not")

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
            assert(view_to_struct)
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
            name_hierarchy, _ = self._process_edges(sdfg, state, view_chain)
            struct_to_view_edges = [
                e for e in state.out_edges(struct_access) if e.dst == view_chain[0]
            ]

        if view_to_struct:
            view_to_struct_edges = [
                e for e in state.in_edges(struct_access) if e.src == view_chain[-1]
            ]
            name_hierarchy, _ = self._process_edges_reverse(sdfg, state, view_chain)

        # Get the SoA flattened container name
        demangled_name = get_demangled_container_group_member_name(sdfg, name_hierarchy)

        an = nodes.AccessNode(data=demangled_name)

        print("Source/Destination Struct:", struct_access, "SV:", struct_to_view, "VS:", view_to_struct)
        print("Applying to:", [struct_access] + view_chain if struct_to_view else view_chain + [struct_access])
        print("Source/Destination struct is: ", struct_access.data, type(sdfg.arrays[struct_access.data]))

        # If the length is less than 2 then we have only one view and one source struct

        if struct_to_view:
            assert len(state.in_edges(view_chain[0])) == 1
            src_edge = state.in_edges(view_chain[0])[0]
            assert len(state.out_edges(view_chain[-1])) <= 1
            if len(state.out_edges(view_chain[-1])) == 0:
                dst_edge = None # Can also be not directly used, but used on the interstate edge.
            else:
                dst_edge = state.out_edges(view_chain[-1])[0]
        else:  # view_to_struct
            assert view_to_struct
            assert len(state.in_edges(view_chain[0])) == 1
            src_edge = state.in_edges(view_chain[0])[0]
            assert len(state.out_edges(view_chain[-1])) <= 1
            dst_edge = state.out_edges(view_chain[-1])[0]


        if self._flattening_mode == ContainerGroupFlatteningMode.StructOfArrays:
            memlet_shape = ()
            # We need calculate the dimension of the new array
            # The shape and memlet expression calculations differ
            if struct_to_view:
                assert len(struct_to_view_edges) == 1
                for vc in [struct_access] + view_chain:
                    if state.out_degree(vc) > 0:
                        _dst_edge = state.out_edges(vc)[0]
                        if isinstance(
                            sdfg.arrays[vc.data], dace.data.Structure
                        ):
                            continue
                        memlet_shape += tuple(_dst_edge.data.subset.ranges)

            if view_to_struct:
                assert len(view_to_struct_edges) == 1
                for vc in list(reversed(view_chain)) + [struct_access]:
                    if state.in_degree(vc) > 0:
                        _src_edge = state.in_edges(vc)[0]
                        if isinstance(
                            sdfg.arrays[vc.data], dace.data.Structure
                        ):
                            continue
                        memlet_shape += tuple(_src_edge.data.subset.ranges)
        else:
            raise Exception("ArrayOfStructs mode is not implemented yet")

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
        state.add_node(an)
        view_name = None
        if struct_to_view:
            if dst_edge is not None:
                #view_name = "v_" + demangled_name
                #a = sdfg.arrays[dst_edge.data.data]
                #if view_name not in sdfg.arrays:
                #    sdfg.add_view(
                #        name=view_name,
                #        shape=a.shape,
                #        dtype=a.dtype,
                #        storage=a.storage,
                #        strides=a.strides,
                #        offset=a.offset,
                #        allow_conflicts=a.allow_conflicts,
                #        find_new_name=False,
                #        may_alias=a.may_alias,
                #    )

                #view_access = state.add_access(view_name)
                #state.add_edge(an, None, view_access, "views", mc)
                #nm = copy.deepcopy(dst_edge.data)
                #nm.data = view_name

                state.add_edge(an, None, dst_edge.dst, dst_edge.dst_conn, mc)
            else:
                state.add_edge(an, None, view_chain[-1], "views", mc)

            if struct_access.guid in self._data_connected_to_vsv_struct:
                self._data_connected_to_vsv_struct[struct_access.guid][1].append(an)
            else:
                self._data_connected_to_vsv_struct[struct_access.guid] = ([], [an])

        else:  # view_to_struct
            assert dst_edge is not None
            """
            view_name = "v_" + demangled_name
            if view_name not in sdfg.arrays:
                a = sdfg.arrays[dst_edge.data.data]
                sdfg.add_view(
                    name=view_name,
                    shape=a.shape,
                    dtype=a.dtype,
                    storage=a.storage,
                    strides=a.strides,
                    offset=a.offset,
                    allow_conflicts=a.allow_conflicts,
                    find_new_name=False,
                    may_alias=a.may_alias,
                )
            view_access = state.add_access(view_name)
            nm = copy.deepcopy(src_edge.data)
            nm.data = view_name
            state.add_edge(src_edge.src, src_edge.src_conn, view_access, None, nm)
            state.add_edge(view_access, "views", an, None, mc)
            """
            state.add_edge(src_edge.src, src_edge.src_conn, an, None, mc)
            if struct_access.guid in self._data_connected_to_vsv_struct:
                self._data_connected_to_vsv_struct[struct_access.guid][0].append(an)
            else:
                self._data_connected_to_vsv_struct[struct_access.guid] = ([an], [])

        # Replace all occurences
        if struct_to_view:
            repl_before = view_chain[-1].data

        if view_to_struct:
            repl_before = view_chain[0].data
        repl_after = (demangled_name, 0, copy.deepcopy(mc.subset))
        print("REPL later:", repl_before, "->", repl_after)
        # If the memlet has more dimensions, then it means we need to incorporate the missing dimensions from left
        # in the further accessess to the symbol
        if len(mc.subset) >  len(sdfg.arrays[view_chain[-1 if struct_to_view else 0].data].shape):
            print("This replacement will result with some string of form a[1][2] which might not be supported! TODO!!!\n"*5)
            iterlen = len(mc.subset) - len(sdfg.arrays[view_chain[-1 if struct_to_view else 0].data].shape)
            dimstopad = iterlen
            repl_after = (demangled_name, dimstopad, copy.deepcopy(mc.subset))

            # Go through all memlets and interstate edges, and if conditions and replace the name + access
            # TODO: tomorrow
            # After applying everytihng


        # Clean-up
        if struct_to_view:
            for view_node in view_chain:
                state.remove_node(view_node)
                removed_nodes.add(view_node)
                #print("rm", view_node)
        if view_to_struct:
            for view_node in view_chain:
                state.remove_node(view_node)
                removed_nodes.add(view_node)
                #print("rm", view_node)

        if (len(state.in_edges(struct_access)) == 0) and (
            len(state.out_edges(struct_access)) == 0
        ):
            state.remove_node(struct_access)
            removed_nodes.add(struct_access)
        #sdfg.save(f"aa{self._call}.sdfg")
        self._call += 1



        # All acccess from the view need to me mapped to the newly added array
        # The leaf node will not have access to all of the dimensions in the generated array we need to do that
        # missing_dims = memlet_shape[:-len(sdfg.arrays[view_chain[-1 if struct_to_view else 0].data].shape)]
        # if not isinstance(missing_dims, List):
        #    missing_dims = list(missing_dims)
        if view_name is not None:
            self._access_names_map[view_chain[-1 if struct_to_view else 0].data] = view_name
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
