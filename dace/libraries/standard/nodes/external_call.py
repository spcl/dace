# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ExternalCall``: a nest that is either expanded from its own SDFG or called in a separately compiled library.

Expansions:

``DaceReference``  Default. A copy of the nest's SDFG (``standalone_sdfg``), so the program runs unchanged.
``ExternCall``     A C++ tasklet calling ``extern "C" void <symbol>(<signature>)``, whose forward declaration
                   is the tasklet's global code. ``lib_path`` (a ``.a`` or ``.so``), then ``link_flags``, go on
                   the program's link line in that order; a shared dependency's ``-Wl,-rpath,<dir>`` belongs in
                   ``link_flags``.

Connectors are the nest's data names behind a prefix: ``_in_<name>`` for an input, ``_out_<name>`` for an
output; data read and written has both. ``signature`` follows DaCe's nested-SDFG convention: read-only inputs
by name, then outputs by name (data read and written once), then symbols by name; a read-only array is
``const T* __restrict__``, a written one ``T* __restrict__``, a read-only scalar ``T`` by value.
"""

import copy
import hashlib
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import dace
from dace import data, dtypes, library, properties, subsets
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

IN_PREFIX = "_in_"
OUT_PREFIX = "_out_"

#: Hex digits of a link environment's name digest.
DIGEST_DIGITS = 16


def in_conn(name: str) -> str:
    return IN_PREFIX + name


def out_conn(name: str) -> str:
    return OUT_PREFIX + name


def data_name(conn: str) -> str:
    return conn.removeprefix(IN_PREFIX).removeprefix(OUT_PREFIX)


def one_element(memlet: dace.Memlet) -> bool:
    subset = memlet.subset
    return isinstance(subset, subsets.Range) and bool(subset) and subset.num_elements() == 1


@dataclass(slots=True)
class CallSite:
    """What the parent state says about the connectors of one ``ExternalCall``."""

    outputs: dict[str, None]
    descriptors: dict[str, data.Data]
    by_value: dict[str, None]
    scalars: dict[str, None]


def call_site(node: "ExternalCall", state: dace.SDFGState) -> CallSite:
    site = CallSite({}, {}, {}, {})
    for edge in state.in_edges(node):
        if edge.dst_conn is None or edge.data.data is None:
            continue
        desc = state.sdfg.arrays[edge.data.data]
        site.descriptors[edge.dst_conn] = desc
        if isinstance(desc, data.Scalar):
            site.scalars[edge.dst_conn] = None
        if one_element(edge.data):
            site.by_value[edge.dst_conn] = None
    for edge in state.out_edges(node):
        if edge.src_conn is None or edge.data.data is None:
            continue
        site.outputs[data_name(edge.src_conn)] = None
        site.descriptors[edge.src_conn] = state.sdfg.arrays[edge.data.data]
        # a dynamic non-WCR output stays a pointer
        if one_element(edge.data) and not (edge.data.dynamic and edge.data.wcr is None):
            site.by_value[edge.src_conn] = None
    return site


def nested_sdfg_order(node: "ExternalCall", state: dace.SDFGState, symbols: Sequence[str]) -> list[str]:
    """DaCe's nested-SDFG argument order: read-only inputs, then outputs, each by name, then ``symbols`` by name."""
    site = call_site(node, state)
    names = dict.fromkeys(data_name(conn) for conn in site.descriptors)
    inputs = sorted(name for name in names if name not in site.outputs)
    return inputs + sorted(site.outputs) + sorted(s for s in symbols if s not in names)


def data_param(node: "ExternalCall", arg: str, site: CallSite) -> tuple[str, str]:
    """``(parameter, call argument)`` of one data argument: a read-only Scalar by value, the rest by pointer."""
    conn = out_conn(arg) if arg in site.outputs else in_conn(arg)
    if conn not in site.descriptors:
        raise ValueError(
            f"ExternalCall {node.name!r}: abi_order names {arg!r}, but no edge reaches connector "
            f"{conn!r}; keep the DaceReference implementation"
        )
    ctype = site.descriptors[conn].dtype.ctype
    if conn in site.scalars:
        return f"{ctype} {arg}", conn
    const = "" if arg in site.outputs else "const "
    return f"{const}{ctype}* __restrict__ {arg}", f"&{conn}" if conn in site.by_value else conn


def symbol_param(arg: str, sdfg: dace.SDFG) -> str:
    if arg not in sdfg.symbols:
        raise ValueError(
            f"ExternalCall argument {arg!r} is neither a connected container nor a symbol of {sdfg.name!r}"
        )
    return f"{sdfg.symbols[arg].ctype} {arg}"


def params_and_args(node: "ExternalCall", state: dace.SDFGState) -> tuple[list[str], list[str]]:
    """The C parameters and the call arguments, in ``node.abi_order``. C linkage matches the name alone, so a
    wrong order links cleanly and swaps buffers."""
    order = list(node.abi_order)
    if not order:
        raise ValueError(
            f"ExternalCall {node.name!r} has no abi_order; ExternCall must call the linked symbol "
            "in the order it was compiled with"
        )
    site = call_site(node, state)
    connected = {data_name(conn) for conn in site.descriptors}
    params: list[str] = []
    call_args: list[str] = []
    for arg in order:
        if arg in connected:
            param, call_arg = data_param(node, arg, site)
        else:
            param, call_arg = symbol_param(arg, state.sdfg), arg
        params.append(param)
        call_args.append(call_arg)
    return params, call_args


def derive_signature(node: "ExternalCall", state: dace.SDFGState) -> str:
    """The parameter list of ``node``'s entry in ``abi_order``, typed by the parent state's descriptors."""
    return ", ".join(params_and_args(node, state)[0])


def parameter_count(signature: str) -> int:
    depth, count = 0, 1 if signature.strip() else 0
    for ch in signature:
        depth += ch in "(<"
        depth -= ch in ")>"
        count += ch == "," and depth == 0
    return count


def link_environment(lib_path: str, link_flags: Sequence[str]) -> type:
    """The environment linking one library, then ``link_flags``; one per distinct pair. A ``.so`` adds its own
    directory as an rpath, a ``.a`` needs none: it is copied into the program."""
    lib = os.path.abspath(lib_path)
    rpath = [] if lib.endswith(".a") else [f"-Wl,-rpath,{os.path.dirname(lib)}"]
    key = "\0".join([lib, *link_flags])
    name = "ExternalCallLink_" + hashlib.sha256(key.encode()).hexdigest()[:DIGEST_DIGITS]
    registered = library._DACE_REGISTERED_ENVIRONMENTS.get(f"{__name__}.{name}")
    if registered is not None:
        return registered
    fields = dict(
        cmake_minimum_version=None,
        cmake_packages=[],
        cmake_variables={},
        cmake_includes=[],
        cmake_libraries=[lib, *link_flags, *rpath],
        cmake_compile_flags=[],
        cmake_link_flags=[],
        cmake_files=[],
        headers=[],
        state_fields=[],
        init_code="",
        finalize_code="",
        dependencies=[],
        __module__=__name__,
    )
    return library.environment(type(name, (), fields))


@library.expansion
class ExpandDaceReference(ExpandTransformation):
    """Expand to a copy of the nest's SDFG."""

    environments = []

    @staticmethod
    def expansion(node: "ExternalCall", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
        if node.standalone_sdfg is None:
            raise ValueError(
                f"ExternalCall {node.name!r} has no standalone SDFG to expand (it is not serialized; "
                "a node loaded from disk can only use ExternCall)"
            )
        return copy.deepcopy(node.standalone_sdfg)


@library.expansion
class ExpandExternCall(ExpandTransformation):
    """Expand to a C++ tasklet calling the linked library's entry, declared in the tasklet's global code."""

    environments = []

    @staticmethod
    def expansion(node: "ExternalCall", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        if not node.lib_path or not node.symbol:
            raise ValueError(f"ExternalCall {node.name!r} needs lib_path and symbol for ExternCall")
        params, call_args = params_and_args(node, parent_state)
        signature = node.signature or ", ".join(params)
        if parameter_count(signature) != len(call_args):
            raise ValueError(
                f"ExternalCall {node.name!r}: signature ({signature}) has {parameter_count(signature)} "
                f"parameters, abi_order {len(call_args)}"
            )
        return nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            f"{node.symbol}({', '.join(call_args)});",
            language=dtypes.Language.CPP,
            code_global=f'extern "C" void {node.symbol}({signature});',
            side_effects=True,
        )

    def apply(self, state: dace.SDFGState, sdfg: dace.SDFG, *args: Any, **kwargs: Any) -> None:
        node = state.node(self.subgraph[type(self)._match_node])
        env = link_environment(node.lib_path, list(node.link_flags))
        before = dict.fromkeys(state.nodes())
        super().apply(state, sdfg, *args, **kwargs)
        for added in state.nodes():
            if added not in before:
                added.environments = {*added.environments, env.full_class_path()}


@library.node
class ExternalCall(nodes.LibraryNode):
    """A nest lowered to a call of a separately compiled kernel. See the module docstring."""

    implementations = {"DaceReference": ExpandDaceReference, "ExternCall": ExpandExternCall}
    default_implementation = "DaceReference"
    #: a deserialized node is built without __init__
    _standalone_sdfg: dace.SDFG | None = None

    numpy_source = properties.Property(dtype=str, default="", desc="NumPy reference source of the nest")
    symbol = properties.Property(dtype=str, default="", desc="extern-C symbol to call")
    abi_order = properties.ListProperty(element_type=str, default=[], desc="parameter order of the linked entry")
    signature = properties.Property(
        dtype=str, default="", desc="C parameter list of the linked entry; derived from the parent when empty"
    )
    lib_path = properties.Property(dtype=str, default="", desc="kernel library, .a or .so")
    link_flags = properties.ListProperty(
        element_type=str,
        default=[],
        desc="link items after the library, in order: -L directories, the "
        "runtimes it needs (-lomp, -lcudart), -Wl,-rpath for shared ones",
    )

    def __init__(
        self,
        name: str,
        inputs: Sequence[str] | None = None,
        outputs: Sequence[str] | None = None,
        standalone_sdfg: dace.SDFG | None = None,
        numpy_source: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, inputs=list(inputs or []), outputs=list(outputs or []), **kwargs)
        self.standalone_sdfg = standalone_sdfg
        self.numpy_source = numpy_source

    @property
    def standalone_sdfg(self) -> dace.SDFG | None:
        """The nest the ``DaceReference`` expansion copies; kept in memory only, never serialized."""
        return self._standalone_sdfg

    @standalone_sdfg.setter
    def standalone_sdfg(self, value: dace.SDFG | None) -> None:
        self._standalone_sdfg = value


def external_calls(sdfg: dace.SDFG) -> list[ExternalCall]:
    """Every ``ExternalCall`` of ``sdfg``, nested SDFGs included."""
    return [node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, ExternalCall)]
