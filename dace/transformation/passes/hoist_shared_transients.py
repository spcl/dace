# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Hoists the transients of nested SDFGs whose storage outlives a single call (Persistent, External and Global)."""

import copy
import re
import warnings
from dataclasses import dataclass

import sympy as sp

from dace import SDFG, Memlet, SDFGState, data, dtypes, properties, symbolic
from dace.sdfg import nodes
from dace.sdfg.replace import replace_properties_dict
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation

#: Lifetimes whose storage outlives a single call of the SDFG that owns the container: allocated once per program
#: (Persistent), managed by the caller (External), or allocated once per run of the top-level SDFG (Global). In all
#: cases one storage is shared by every call of the nested SDFG node.
SHARED_LIFETIMES = (
    dtypes.AllocationLifetime.Persistent,
    dtypes.AllocationLifetime.External,
    dtypes.AllocationLifetime.Global,
)

#: Storage types whose allocation is private to a thread or a processing element. One root container passed by
#: pointer to every invocation would be shared among them, so such containers stay where they are.
_LOCAL_STORAGES = (
    dtypes.StorageType.Register,
    dtypes.StorageType.CPU_ThreadLocal,
    dtypes.StorageType.GPU_Shared,
    dtypes.StorageType.SVE_Register,
    dtypes.StorageType.Snitch_TCDM,
    dtypes.StorageType.Snitch_SSR,
)


def is_shared_transient(desc: data.Data) -> bool:
    """
    Returns True if the given descriptor is a transient whose storage outlives one call of its SDFG.

    :param desc: The data descriptor.
    :return: True for a transient with Persistent, External or Global lifetime.
    """
    return desc.transient and desc.lifetime in SHARED_LIFETIMES


def has_nested_shared_transients(sdfg: SDFG) -> bool:
    """
    Quickly checks whether any SDFG nested (at any depth) in the given SDFG owns a shared transient.

    :param sdfg: The SDFG to scan.
    :return: True if some nested SDFG owns a Persistent, External or Global transient.
    """
    return any(
        is_shared_transient(desc)
        for nsdfg in sdfg.all_sdfgs_recursive()
        if nsdfg is not sdfg
        for desc in nsdfg.arrays.values()
    )


def _uses_outside_dataflow(sdfg: SDFG, name: str) -> bool:
    """
    Conservatively checks whether a container is named outside of dataflow (on interstate edges, in control-flow
    region code such as loop conditions, or in nested SDFG symbol mappings) of a single SDFG level.

    :param sdfg: The SDFG to check (nested SDFGs are not entered).
    :param name: The container name.
    :return: True if the name appears as a token in any of these places.
    """
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    for edge in sdfg.all_interstate_edges():
        if name in edge.data.read_symbols():
            return True
    for region in sdfg.all_control_flow_regions():
        for block in region.get_meta_codeblocks():
            if pattern.search(block.as_string):
                return True
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG) and any(pattern.search(str(v)) for v in node.symbol_mapping.values()):
                return True
    return False


def _access_kind(sdfg: SDFG, name: str) -> tuple[bool, bool]:
    """
    Determines how a single SDFG level uses a container.

    :param sdfg: The SDFG that owns the container (nested SDFGs are reached through its access nodes).
    :param name: The container name.
    :return: A ``(read, written)`` pair. An unused container counts as read, so that it gets an input connector.
    """
    read = written = False
    for state in sdfg.states():
        for node in state.data_nodes():
            if node.data != name:
                continue
            read |= state.out_degree(node) > 0
            written |= state.in_degree(node) > 0
    if not read and _uses_outside_dataflow(sdfg, name):
        read = True
    if not written:
        read = True
    return read, written


def _global_symbols(sdfg: SDFG) -> set[str]:
    """
    Returns the symbols that are defined at the very beginning of a nested SDFG and do not change during it, i.e.,
    the symbols a container allocated once for this SDFG level may use in its shape.

    :param sdfg: The nested SDFG.
    :return: The set of symbol names.
    """
    return set(sdfg.parent_nsdfg_node.symbol_mapping.keys()) | set(sdfg.constants_prop.keys())


def _substitute_in_place(desc: data.Data, mapping: dict[str, symbolic.SymbolicType]) -> None:
    """
    Substitutes symbols in the shape, strides, offsets and total size of a descriptor.

    :param desc: The descriptor (modified in place).
    :param mapping: Maps symbol names to their replacement expressions.
    """
    if not mapping:
        return
    symrepl = {
        symbolic.symbol(k): (v if isinstance(v, sp.Basic) else symbolic.pystr_to_symbolic(str(v)))
        for k, v in mapping.items()
    }
    replace_properties_dict(desc, {}, symrepl)


def _substitute(desc: data.Data, mapping: dict[str, symbolic.SymbolicType]) -> data.Data:
    """
    Returns a copy of a descriptor with symbols substituted in its shape, strides, offsets and total size.

    :param desc: The descriptor.
    :param mapping: Maps symbol names to their replacement expressions.
    :return: The substituted copy.
    """
    result = copy.deepcopy(desc)
    _substitute_in_place(result, mapping)
    return result


def _used_constants(sdfg: SDFG, desc: data.Data) -> dict[str, object]:
    """
    Returns the constants of an SDFG level that a descriptor uses as symbols.

    :param sdfg: The SDFG level.
    :param desc: The descriptor.
    :return: Maps each such constant's name to its value.
    """
    fsyms = {str(s) for s in desc.free_symbols}
    return {k: v for k, v in sdfg.constants.items() if k in fsyms}


def _hoisting_chain(sdfg: SDFG, top: SDFG, top_symbols: set[str], name: str) -> list[data.Data] | None:
    """
    Computes the descriptor of a container as seen from each SDFG level between its owner and the top-level SDFG.

    :param sdfg: The nested SDFG that owns the container.
    :param top: The top-level SDFG.
    :param top_symbols: The symbols defined for the whole run of ``top`` (its free symbols and constants).
    :param name: The container name.
    :return: The descriptors at the levels ``sdfg.parent_sdfg``, its parent, ..., ``top`` (in that order), or None
             if a symbol of the descriptor is not defined for the whole run of one of the levels (e.g., a map
             parameter or an interstate-edge assignment), in which case the container cannot be hoisted.
    """
    desc = sdfg.arrays[name]
    result = []
    cur = sdfg
    while cur is not top:
        fsyms = {str(s) for s in desc.free_symbols}
        if not fsyms <= _global_symbols(cur):
            return None
        desc = _substitute(desc, _used_constants(cur, desc))
        nsdfg_node = cur.parent_nsdfg_node
        mapping = {k: v for k, v in nsdfg_node.symbol_mapping.items() if k in {str(s) for s in desc.free_symbols}}
        desc = _substitute(desc, mapping)
        cur = cur.parent_sdfg
        result.append(desc)
    fsyms = {str(s) for s in desc.free_symbols}
    if not fsyms <= top_symbols:
        return None
    return result


def _connect(state: SDFGState, nsdfg_node: nodes.NestedSDFG, conn: str, outer: str, read: bool, written: bool) -> None:
    """
    Connects a nested SDFG node to the full range of an outer container, routing through all enclosing scopes.

    :param state: The state containing the nested SDFG node.
    :param nsdfg_node: The nested SDFG node.
    :param conn: The connector name (the container's name in the nested SDFG).
    :param outer: The container name in the SDFG that contains ``state``.
    :param read: Whether to add an input connector.
    :param written: Whether to add an output connector.
    """
    outer_desc = state.sdfg.arrays[outer]
    scope_dict = state.scope_dict()
    entries = []
    scope = scope_dict[nsdfg_node]
    while scope is not None:
        entries.append(scope)
        scope = scope_dict[scope]
    if read:
        nsdfg_node.add_in_connector(conn)
        state.add_memlet_path(
            state.add_read(outer),
            *reversed(entries),
            nsdfg_node,
            memlet=Memlet.from_array(outer, outer_desc),
            dst_conn=conn,
        )
    if written:
        exits = [state.exit_node(e) for e in entries]
        nsdfg_node.add_out_connector(conn, force=True)
        state.add_memlet_path(
            nsdfg_node,
            *exits,
            state.add_write(outer),
            memlet=Memlet.from_array(outer, outer_desc),
            src_conn=conn,
        )


@dataclass(unsafe_hash=True)
@properties.make_properties
@transformation.explicit_cf_compatible
class HoistSharedTransients(ppl.Pass):
    """
    Hoists every transient of a nested SDFG, at any depth, whose storage outlives a single call (lifetime
    ``Persistent``, ``External`` or ``Global``) to the top-level SDFG, so that afterwards nested SDFGs own only
    per-call transients (lifetimes ``Scope``, ``State`` and ``SDFG``).

    The code generator allocates such storage of a nested SDFG once per program (``Persistent``), once per run of the
    top-level SDFG (``Global``), or leaves it to the caller (``External``); in every case it is shared by all calls of
    the nested SDFG node (e.g., all iterations of an enclosing map). The pass preserves this: for each nested SDFG
    node and shared transient ``t`` of its SDFG, it adds one transient with the same descriptor and lifetime to
    the top-level SDFG, makes ``t`` a (``Scope``-lifetime) non-transient of the nested SDFG, and threads the new
    container through every intermediate SDFG level as a non-transient argument. Each nested SDFG node on the way
    gets a connector with full-range memlets, routed through all enclosing scopes (every map iteration sees the whole
    container). The node gets an input connector if the nested SDFG reads ``t`` (or does not use it at all) and an
    output connector if it writes ``t``.

    Limitations: a container stays nested (with a warning) if a symbol of its shape is not defined for the whole run
    of some enclosing SDFG level (e.g., it is a map parameter or assigned on an interstate edge), or if its connector
    name is already taken on the nested SDFG node. Views and references (always ``Scope``-lifetime), and containers
    in thread- or processing-element-local storage (registers, thread-local, GPU shared memory), are never hoisted.
    The SDFG the pass is applied to is the top level: if it is itself nested, its own shared transients stay.
    """

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes | ppl.Modifies.NestedSDFGs | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Descriptors | ppl.Modifies.NestedSDFGs))

    def apply_pass(self, sdfg: SDFG, _) -> dict[tuple[int, str], str] | None:
        """
        Hoists the shared transients of all nested SDFGs to ``sdfg``.

        :param sdfg: The top-level SDFG.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: A dictionary mapping ``(cfg_id of the nested SDFG, container name)`` to the name of the hoisted
                 container in ``sdfg``, or None if nothing was hoisted.
        """
        candidates = [
            (nsdfg, name)
            for nsdfg in sdfg.all_sdfgs_recursive()
            if nsdfg is not sdfg
            for name, desc in nsdfg.arrays.items()
            if is_shared_transient(desc)
        ]
        if not candidates:
            return None

        top_symbols = set(sdfg.free_symbols) | set(sdfg.constants_prop.keys())
        result: dict[tuple[int, str], str] = {}
        for nsdfg, name in candidates:
            hoisted = self._hoist(sdfg, top_symbols, nsdfg, name)
            if hoisted is not None:
                result[(nsdfg.cfg_id, name)] = hoisted
        return result or None

    def _hoist(self, top: SDFG, top_symbols: set[str], nsdfg: SDFG, name: str) -> str | None:
        """
        Hoists one shared transient of a nested SDFG to the top-level SDFG.

        :param top: The top-level SDFG.
        :param top_symbols: The symbols defined for the whole run of ``top``.
        :param nsdfg: The nested SDFG that owns the container.
        :param name: The container name.
        :return: The name of the hoisted container in ``top``, or None if the container was left in place.
        """
        desc = nsdfg.arrays[name]
        if isinstance(desc, (data.View, data.Reference)) or desc.storage in _LOCAL_STORAGES:
            return None
        if not isinstance(desc, (data.Array, data.Scalar)) or isinstance(desc, data.ContainerArray):
            warnings.warn(
                f"HoistSharedTransients: {type(desc).__name__} '{name}' in nested SDFG '{nsdfg.name}' is not "
                "supported and stays nested"
            )
            return None
        node = nsdfg.parent_nsdfg_node
        if name in node.in_connectors or name in node.out_connectors:
            warnings.warn(
                f"HoistSharedTransients: connector '{name}' already exists on nested SDFG node '{node.label}'; "
                "the container stays nested"
            )
            return None
        chain = _hoisting_chain(nsdfg, top, top_symbols, name)
        if chain is None:
            warnings.warn(
                f"HoistSharedTransients: the shape of '{name}' in nested SDFG '{nsdfg.name}' uses a symbol that "
                "is not defined for the whole program; the container stays nested"
            )
            return None

        read, written = _access_kind(nsdfg, name)
        lifetime = desc.lifetime
        base_name = f"{nsdfg.name}_{name}"

        # The innermost level: the container becomes an argument. A connector's descriptor may use only symbols of the
        # symbol mapping, so constants are written as their values.
        _substitute_in_place(desc, _used_constants(nsdfg, desc))
        desc.transient = False
        desc.lifetime = dtypes.AllocationLifetime.Scope

        conn = name
        cur = nsdfg
        for outer_desc in chain:
            node = cur.parent_nsdfg_node
            state = cur.parent
            parent = cur.parent_sdfg
            if parent is top:
                outer_desc.transient = True
                outer_desc.lifetime = lifetime
            else:
                outer_desc.transient = False
                outer_desc.lifetime = dtypes.AllocationLifetime.Scope
            outer = parent.add_datadesc(base_name, outer_desc, find_new_name=True)
            _connect(state, node, conn, outer, read, written)
            conn = outer
            cur = parent
        return conn
