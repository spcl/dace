# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ExternalCall``: a nest that is either expanded from its own SDFG or called in a separately compiled library.

Expansions:

``DaceReference``  Default. A copy of the nest's SDFG (``standalone_sdfg``), so the program runs unchanged.
``ExternCall``     A C++ tasklet that calls ``extern "C" <symbol>(...)`` in ``lib_path``, its parameters in
                   ``abi_order``. The library and its runtimes are linked through ``ExternLibEnv``.

Connectors are the nest's data names behind a prefix: ``_in_<name>`` for an input, ``_out_<name>`` for an
output; data read and written has both. Pointer types and pass-by-value come from the parent's data
descriptors and symbols, so the node carries no type manifest of its own.
"""
import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import dace
from dace import data, dtypes, library, properties, subsets
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

IN_PREFIX = '_in_'
OUT_PREFIX = '_out_'


def in_conn(name: str) -> str:
    return IN_PREFIX + name


def out_conn(name: str) -> str:
    return OUT_PREFIX + name


def one_element(memlet: dace.Memlet) -> bool:
    subset = memlet.subset
    return isinstance(subset, subsets.Range) and bool(subset) and subset.num_elements() == 1


@dataclass(slots=True)
class CallSite:
    """What the parent state says about the connectors of one ``ExternalCall``."""

    outputs: Dict[str, None]
    descriptors: Dict[str, data.Data]
    by_value: Dict[str, None]
    scalars: Dict[str, None]


def call_site(node: 'ExternalCall', state: dace.SDFGState) -> CallSite:
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
        site.outputs[edge.src_conn.removeprefix(OUT_PREFIX)] = None
        site.descriptors[edge.src_conn] = state.sdfg.arrays[edge.data.data]
        # a dynamic non-WCR output stays a pointer
        if one_element(edge.data) and not (edge.data.dynamic and edge.data.wcr is None):
            site.by_value[edge.src_conn] = None
    return site


def data_param(node: 'ExternalCall', arg: str, site: CallSite) -> Tuple[str, str]:
    """``(parameter, call argument)`` of one data argument: a read-only Scalar by value, the rest by pointer."""
    conn = out_conn(arg) if arg in site.outputs else in_conn(arg)
    if conn not in site.descriptors:
        raise ValueError(f'ExternalCall {node.name!r}: abi_order names {arg!r}, but no edge reaches connector '
                         f'{conn!r}; keep the DaceReference implementation')
    ctype = site.descriptors[conn].dtype.ctype
    if conn in site.scalars:
        return f'{ctype} {arg}', conn
    const = '' if arg in site.outputs else 'const '
    return f'{const}{ctype}* {arg}', f'&{conn}' if conn in site.by_value else conn


def symbol_param(arg: str, sdfg: dace.SDFG) -> str:
    if arg not in sdfg.symbols:
        raise ValueError(f'ExternalCall argument {arg!r} is neither a connected container nor a symbol of '
                         f'{sdfg.name!r}')
    return f'{sdfg.symbols[arg].ctype} {arg}'


def proto_and_call(node: 'ExternalCall', state: dace.SDFGState) -> Tuple[str, str]:
    """The ``extern "C"`` prototype and call of the linked entry, in ``node.abi_order``. C linkage matches the
    name alone, so any other order links cleanly and swaps buffers."""
    order = list(node.abi_order)
    if not order:
        raise ValueError(f'ExternalCall {node.name!r} has no abi_order; ExternCall must declare the linked '
                         'symbol in the order it was compiled with')
    site = call_site(node, state)
    connected = {conn.removeprefix(IN_PREFIX).removeprefix(OUT_PREFIX) for conn in site.descriptors}
    params: List[str] = []
    call_args: List[str] = []
    for arg in order:
        if arg in connected:
            param, call_arg = data_param(node, arg, site)
        else:
            param, call_arg = symbol_param(arg, state.sdfg), arg
        params.append(param)
        call_args.append(call_arg)
    proto = f'extern "C" void {node.symbol}({", ".join(params)});'
    call = f'{node.symbol}({", ".join(call_args)});'
    return proto, call


def with_new_items(existing: List[str], items: Sequence[str]) -> List[str]:
    return [*existing, *(item for item in dict.fromkeys(items) if item not in existing)]


@library.environment
class ExternLibEnv:
    """Links every called library and its runtimes into the program. DaCe reads environments as classes, so
    ``configure`` accumulates class attributes during expansion; ``reset`` before expanding a fresh SDFG."""

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []
    headers = []
    state_fields = []
    init_code = ''
    finalize_code = ''
    dependencies = []

    @classmethod
    def reset(cls) -> None:
        cls.cmake_libraries = []
        cls.cmake_link_flags = []

    @classmethod
    def configure(cls, lib_path: str, runtime_libraries: Sequence[str] = ()) -> None:
        lib = os.path.abspath(lib_path)
        cls.cmake_libraries = with_new_items(cls.cmake_libraries, [lib, *runtime_libraries])
        if not lib.endswith('.a'):  # a shared library needs an rpath
            cls.cmake_link_flags = with_new_items(cls.cmake_link_flags, [f'-Wl,-rpath,{os.path.dirname(lib)}'])


@library.expansion
class ExpandDaceReference(ExpandTransformation):
    """Expand to a copy of the nest's SDFG."""

    environments = []

    @staticmethod
    def expansion(node: 'ExternalCall', parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
        if node.standalone_sdfg is None:
            raise ValueError(f'ExternalCall {node.name!r} has no standalone SDFG to expand (it is not serialized; '
                             'a node loaded from disk can only use ExternCall)')
        return copy.deepcopy(node.standalone_sdfg)


@library.expansion
class ExpandExternCall(ExpandTransformation):
    """Expand to a C++ tasklet calling the linked library's entry."""

    environments = [ExternLibEnv]

    @staticmethod
    def expansion(node: 'ExternalCall', parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        if not node.lib_path or not node.symbol:
            raise ValueError(f'ExternalCall {node.name!r} needs lib_path and symbol for ExternCall')
        proto, call = proto_and_call(node, parent_state)
        ExternLibEnv.configure(node.lib_path, list(node.runtime_libraries))
        return nodes.Tasklet(node.name,
                             node.in_connectors,
                             node.out_connectors,
                             call,
                             language=dtypes.Language.CPP,
                             code_global=proto,
                             side_effects=True)


@library.node
class ExternalCall(nodes.LibraryNode):
    """A nest lowered to a call of a separately compiled kernel. See the module docstring."""

    implementations = {'DaceReference': ExpandDaceReference, 'ExternCall': ExpandExternCall}
    #: a deserialized node is built without __init__
    _standalone_sdfg: Optional[dace.SDFG] = None
    default_implementation = 'DaceReference'

    numpy_source = properties.Property(dtype=str, default='', desc='NumPy reference source of the nest')
    symbol = properties.Property(dtype=str, default='', desc='extern-C symbol to call')
    abi_order = properties.ListProperty(element_type=str, default=[], desc='parameter order of the linked entry')
    lib_path = properties.Property(dtype=str, default='', desc='kernel library, .a or .so')
    runtime_libraries = properties.ListProperty(element_type=str,
                                                default=[],
                                                desc='link items for the runtimes the library needs (libomp, '
                                                'cudart), placed after the objects')

    def __init__(self,
                 name: str,
                 inputs: Optional[Sequence[str]] = None,
                 outputs: Optional[Sequence[str]] = None,
                 standalone_sdfg: Optional[dace.SDFG] = None,
                 numpy_source: str = '',
                 **kwargs: Any) -> None:
        super().__init__(name, inputs=list(inputs or []), outputs=list(outputs or []), **kwargs)
        self.standalone_sdfg = standalone_sdfg
        self.numpy_source = numpy_source

    @property
    def standalone_sdfg(self) -> Optional[dace.SDFG]:
        """The nest the ``DaceReference`` expansion copies; kept in memory only, never serialized."""
        return self._standalone_sdfg

    @standalone_sdfg.setter
    def standalone_sdfg(self, value: Optional[dace.SDFG]) -> None:
        self._standalone_sdfg = value


def external_calls(sdfg: dace.SDFG) -> List[ExternalCall]:
    """Every ``ExternalCall`` of ``sdfg``, nested SDFGs included."""
    return [node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, ExternalCall)]
