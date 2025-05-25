# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import ctypes
import functools
import re
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import sympy
from six import StringIO

import dace
from dace import data as dt, Memlet
from dace import dtypes, registry
from dace import subsets, symbolic
from dace.codegen import common, cppunparse
from dace.codegen.codeobject import CodeObject
from dace.codegen.dispatcher import DefinedType
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets import cpp
from dace.codegen.common import update_persistent_desc
from dace.codegen.targets.cpp import (
    codeblock_to_cpp,
    cpp_array_expr,
    memlet_copy_to_absolute_strides,
    sym2cpp,
    synchronize_streams,
    unparse_cr,
    mangle_dace_state_struct_name,
)
from dace.codegen.targets.target import IllegalCopy, TargetCodeGenerator, make_absolute
from dace.config import Config
from dace.frontend import operations
from dace.sdfg import (
    SDFG,
    ScopeSubgraphView,
    SDFGState,
    has_dynamic_map_inputs,
    is_array_stream_view,
    is_devicelevel_gpu,
    nodes,
    scope_contains_scope,
)
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import ControlFlowRegion, StateSubgraphView
from dace.transformation import helpers as xfh
from dace.transformation.passes import analysis as ap
from dace.sdfg.validation import validate_memlet_data
from dace import data

if TYPE_CHECKING:
    from dace.codegen.targets.framecode import DaCeCodeGenerator
    from dace.codegen.targets.cpu import CPUCodeGen


def prod(iterable):
    return functools.reduce(sympy.Mul, iterable, 1)


def _expr(val):
    if isinstance(val, symbolic.SymExpr):
        return val.expr
    return val


@registry.autoregister_params(name="ascendc")
class AscendCCodeGen(TargetCodeGenerator):
    """AscendC code generator."""

    target_name = "ascendc"
    title = "AscendC"
    _in_device_code = False

    @staticmethod
    def _generate_type_mapping():
        mapping = {
            (dtypes.float16, dtypes.StorageType.Ascend_Global): "GM_HALF",
            (dtypes.float32, dtypes.StorageType.Ascend_Global): "GM_FLOAT",
        }
        tensor_storage_types = [
            dtypes.StorageType.Ascend_VECIN,
            dtypes.StorageType.Ascend_VECOUT,
            dtypes.StorageType.Ascend_A1,
            dtypes.StorageType.Ascend_A2,
            dtypes.StorageType.Ascend_B1,
            dtypes.StorageType.Ascend_B2,
            dtypes.StorageType.Ascend_CO1,
            dtypes.StorageType.Ascend_CO2,
        ]
        for dtype, dtype_name in ((dace.dtypes.float16, "dace::float16"), (dace.dtypes.float32, "dace::float32")):
            for storage in tensor_storage_types:
                mapping[(dtype, storage)] = f"AscendC::LocalTensor<{dtype_name}>"

        return mapping

    _c_type_to_ascend_decl_type = _generate_type_mapping.__func__()
    _access_type = {
        "GM_HALF": "*",
        "GM_FLOAT": "*",
        "AscendC::LocalTensor<dace::float16>": "&",
        "AscendC::LocalTensor<dace::float32>": "&",
    }
    _storage_to_ascendc_que_name = {
        dtypes.StorageType.Ascend_VECIN: "VECIN",
        dtypes.StorageType.Ascend_VECOUT: "VECOUT",
        dtypes.StorageType.Ascend_A1: "A1",
        dtypes.StorageType.Ascend_A2: "A2",
        dtypes.StorageType.Ascend_B1: "B1",
        dtypes.StorageType.Ascend_B2: "B2",
        dtypes.StorageType.Ascend_CO1: "CO1",
        dtypes.StorageType.Ascend_CO2: "CO2",
    }

    def _get_access_type(self, type_str):
        return self._access_type.get(type_str, "")

    def _get_ascendc_type(self, data: data.Data, storage: dtypes.StorageType):
        """
        print("Data", data, type(data))
        print(
            "Data2",
            data.dtype,
            ", ",
            data.dtype.ctype,
            ", ",
            data.storage,
            ", ",
            storage,
        )
        """
        return self._c_type_to_ascend_decl_type[(data.dtype, storage)]

    def _get_templated_type(self, data: data.Data, storage: dtypes.StorageType):
        return self._get_ascendc_type(data, storage), self._get_access_type(
            self._get_ascendc_type(data, storage)
        )

    def __init__(self, frame_codegen: "DaCeCodeGenerator", sdfg: SDFG):
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        dispatcher = self._dispatcher

        self.create_grid_barrier = False
        self.extra_nsdfg_args = []
        AscendCCodeGen._in_device_code = False
        self._cpu_codegen: Optional["CPUCodeGen"] = None
        self._localcode = CodeIOStream()
        self._localwrappercode = CodeIOStream()
        self._globalcode = CodeIOStream()
        self._initcode = CodeIOStream()
        self._exitcode = CodeIOStream()
        self._global_sdfg: SDFG = sdfg
        self._toplevel_schedule = None
        self._kernel_map = None
        self._arglists: Dict[nodes.MapEntry, Dict[str, dt.Data]] = {}

        self._acl_streams = []

        self.scope_entry_stream = self._initcode
        self.scope_exit_stream = self._exitcode

        self.has_pool = False
        self.const_params = set()

        self._cpu_codegen = dispatcher.get_generic_node_dispatcher()

        dispatcher.register_map_dispatcher(dtypes.ASCEND_SCHEDULES, self)

        dispatcher.register_node_dispatcher(self, self.node_dispatch_predicate)

        dispatcher.register_state_dispatcher(self, self.state_dispatch_predicate)

        ascend_global_storage = [
            dtypes.StorageType.Ascend_Global,
            dtypes.StorageType.Ascend_VECIN,
            dtypes.StorageType.Ascend_VECOUT,
            dtypes.StorageType.Ascend_A1,
            dtypes.StorageType.Ascend_A2,
            dtypes.StorageType.Ascend_B1,
            dtypes.StorageType.Ascend_B2,
            dtypes.StorageType.Ascend_CO2,
            dtypes.StorageType.Ascend_CO1,
        ]
        dispatcher.register_array_dispatcher(ascend_global_storage, self)

        for storage in ascend_global_storage:
            for other_storage in [
                dtypes.StorageType.CPU_Heap,
                dtypes.StorageType.Register,
            ]:
                dispatcher.register_copy_dispatcher(storage, other_storage, None, self)
                dispatcher.register_copy_dispatcher(other_storage, storage, None, self)

        for storage in ascend_global_storage:
            for other_storage in ascend_global_storage:
                dispatcher.register_copy_dispatcher(storage, other_storage, None, self)
                dispatcher.register_copy_dispatcher(other_storage, storage, None, self)

        # TODO: illegal copies

    def _emit_sync(self, codestream: CodeIOStream):
        codestream.write("DACE_ACL_CHECK(aclrtSynchronizeDevice());")

    def preprocess(self, sdfg: SDFG) -> None:
        # Find Ascend<->Ascend strided copies that cannot be represented by a single copy command
        # TODO
        self.language = "cce"
        self._codeobject = CodeObject(
            sdfg.name + "_" + "ascendc", "", "cpp", AscendCCodeGen, "AscendC"
        )
        self._impl_codeobject = CodeObject(
            sdfg.name + "_kernels" + "_" + "ascendc", "", self.language, AscendCCodeGen, "AscendC"
        )
        # Annotate AscendC streams
        self._acl_streams, _ = self._compute_acl_streams(sdfg)

        # Find points where memory should be released to the memory pool
        # self._compute_pool_release(sdfg)

        # Write GPU context to state structure
        self._frame.statestruct.append("dace::ascendc::Context *acl_context;")

        # Collect all defined symbols and argument lists with one traversal
        shared_transients = {}
        for state, node, defined_syms in sdutil.traverse_sdfg_with_defined_symbols(
            sdfg, recursive=True
        ):
            if isinstance(node, nodes.MapEntry) and node.map.schedule in (
                dtypes.ScheduleType.Ascend_Device,
            ):
                if state.parent not in shared_transients:
                    shared_transients[state.parent] = state.parent.shared_transients()
                self._arglists[node] = state.scope_subgraph(node).arglist(
                    defined_syms, shared_transients[state.parent]
                )

    # Generate final code
    def get_generated_codeobjects(self):
        fileheader = CodeIOStream()

        self._frame.generate_fileheader(self._global_sdfg, fileheader, "ascend")

        initcode = CodeIOStream()
        for sd in self._global_sdfg.all_sdfgs_recursive():
            if None in sd.init_code:
                initcode.write(codeblock_to_cpp(sd.init_code[None]), sd)
            if "acl" in sd.init_code:
                initcode.write(codeblock_to_cpp(sd.init_code["acl"]), sd)
        initcode.write(self._initcode.getvalue())

        exitcode = CodeIOStream()
        for sd in self._global_sdfg.all_sdfgs_recursive():
            if None in sd.exit_code:
                exitcode.write(codeblock_to_cpp(sd.exit_code[None]), sd)
            if "acl" in sd.exit_code:
                exitcode.write(codeblock_to_cpp(sd.exit_code["acl"]), sd)
        exitcode.write(self._exitcode.getvalue())

        params_comma = self._global_sdfg.init_signature(
            free_symbols=self._frame.free_symbols(self._global_sdfg)
        )
        if params_comma:
            params_comma = ", " + params_comma

        self._codeobject.code = """

#include <dace/dace.h>

namespace dace
{{
    using float16 = aclFloat16;
}}

{file_header}


DACE_EXPORTED int __dace_init_ascendc({sdfg_state_name} *__state{params});
DACE_EXPORTED int __dace_exit_ascendc({sdfg_state_name} *__state);

{other_globalcode}

DACE_EXPORTED int __dace_init_ascendc({sdfg_state_name} *__state{params}) {{
    __state->acl_context = new dace::ascendc::Context({nstreams}, {nevents});
    DACE_ACL_CHECK(aclInit({{}}));
    DACE_ACL_CHECK(aclrtSetDevice(0));
    DACE_ACL_CHECK(aclrtCreateContext(&__state->acl_context->aclrt_context, 0));

    // Initialize acl before we run the application
    float *dev_X;
    DACE_ACL_CHECK(aclrtMalloc((void **) &dev_X, 1, ACL_MEM_MALLOC_HUGE_FIRST));
    DACE_ACL_CHECK(aclrtFree(dev_X));

    // Create acl streams and events
    for(int i = 0; i < {nstreams}; ++i) {{
        DACE_ACL_CHECK(aclrtCreateStream(&__state->acl_context->internal_streams[i]));
        __state->acl_context->streams[i] = __state->acl_context->internal_streams[i]; // Allow for externals to modify streams
    }}
    //for(int i = 0; i < {nevents}; ++i) {{
    //    DACE_ACL_CHECK(aclrtEventCreateWithFlags(&__state->acl_context->events[i], aclrtEventDisableTiming));
    //}}

    {initcode}

    return 0;
}}

DACE_EXPORTED int __dace_exit_ascendc({sdfg_state_name} *__state) {{
    {exitcode}
    // Destroy aclrt streams and events
    for(int i = 0; i < {nstreams}; ++i) {{
        DACE_ACL_CHECK(aclrtDestroyStream(__state->acl_context->internal_streams[i]));
    }}
    //for(int i = 0; i < {nevents}; ++i) {{
    //    DACE_ACL_CHECK(aclrtDestroyEvent(__state->acl_context->events[i]));
    //}}

    DACE_ACL_CHECK(aclFinalize());

    delete __state->acl_context;

    return 0;
}}

DACE_EXPORTED bool __dace_acl_set_stream({sdfg_state_name} *__state, int streamid, aclrtStream stream)
{{
    if (streamid < 0 || streamid >= {nstreams}){{
        return false;
    }}

    __state->acl_context->streams[streamid] = stream;

    return true;
}}

DACE_EXPORTED void __dace_acl_set_all_streams({sdfg_state_name} *__state, aclrtStream stream)
{{
    for (int i = 0; i < {nstreams}; ++i){{
        __state->acl_context->streams[i] = stream;
    }}
}}

{localwrappercode}

""".format(
            params=params_comma,
            sdfg_state_name=mangle_dace_state_struct_name(self._global_sdfg),
            initcode=initcode.getvalue(),
            exitcode=exitcode.getvalue(),
            other_globalcode=self._globalcode.getvalue(),
            localwrappercode=self._localwrappercode.getvalue(),
            file_header=fileheader.getvalue(),
            nstreams=max(1, self._acl_streams),
            sdfg=self._global_sdfg,
            nevents=1,
        )

        structstr = '\n'.join(self._frame.statestruct)
        state_struct = f"""struct {mangle_dace_state_struct_name(self._global_sdfg)} {{
    {structstr}
}};"""
        self._impl_codeobject.code = f"""
#include  "kernel_operator.h"

#include <dace/ascendc/devicetypes.h>


// CUBE_BLOCK needs to be 16?
constexpr uint32_t CUBE_BLOCK = 32;
constexpr uint32_t CUBE_BLOCK_SIZE  = 32 * 32;
__aicore__  inline  uint32_t CeilCubeBlock(uint32_t  len)  {{
    return (len + CUBE_BLOCK - 1) / CUBE_BLOCK ;
}}

using AscendC::Add;
using AscendC::Mmad;

{self._localcode.getvalue()}
"""
        return [self._codeobject, self._impl_codeobject]

    def node_dispatch_predicate(self, sdfg, state, node):
        if hasattr(node, "schedule"):  # NOTE: Works on nodes and scopes
            if node.schedule in dtypes.ASCEND_SCHEDULES:
                return True
        if AscendCCodeGen._in_device_code:
            return True
        return False

    def state_dispatch_predicate(self, sdfg, state):
        if self._toplevel_schedule in dtypes.ASCEND_SCHEDULES:
            return True
        for node in state.sink_nodes():
            if hasattr(node, "_acl_stream"):
                return True
            else:
                for e in state.in_edges(node):
                    if hasattr(e.src, "_acl_stream"):
                        return True
        return False

    @property
    def has_initializer(self):
        return True

    @property
    def has_finalizer(self):
        return True

    @staticmethod
    def cmake_options():
        soc_version = Config.get("compiler", "ascendc", "soc_version")
        options = [f"-DDACE_ENABLE_ASCEND=ON",
                   f"-DDACE_ASCEND_SOC_VERSION={soc_version}"]
        return options

    def declare_array(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        node: nodes.AccessNode,
        nodedesc: dt.Data,
        function_stream: CodeIOStream,
        declaration_stream: CodeIOStream,
    ) -> None:
        fsymbols = self._frame.symbols_and_constants(sdfg)
        # NOTE: `dfg` (state) will be None iff `nodedesc` is non-free symbol dependent
        # (see `DaCeCodeGenerator.determine_allocation_lifetime` in `dace.codegen.targets.framecode`).
        # We add the `dfg is not None` check because the `sdutils.is_nonfree_sym_dependent` check will fail if
        # `nodedesc` is a View and `dfg` is None.
        if dfg and not sdutil.is_nonfree_sym_dependent(node, nodedesc, dfg, fsymbols):
            raise NotImplementedError(
                "The declare_array method should only be used for variables "
                "that must have their declaration and allocation separate."
            )

        ptrname = cpp.ptr(node.data, nodedesc, sdfg, self._frame)

        # Check if array is already declared
        if self._dispatcher.declared_arrays.has(ptrname):
            return

        result_decl = StringIO()
        ctypedef = "%s *" % nodedesc.dtype.ctype
        dataname = node.data
        print("Decl", result_decl, ctypedef, dataname)

        # Different types of GPU arrays
        if nodedesc.storage == dtypes.StorageType.Ascend_Global:
            result_decl.write("%s %s;\n" % (ctypedef, dataname))
            self._dispatcher.declared_arrays.add(
                dataname, DefinedType.Pointer, ctypedef
            )
        else:
            raise NotImplementedError(
                "ascend: Unimplemented storage type " + str(nodedesc.storage)
            )

        declaration_stream.write(result_decl.getvalue() + "//8", cfg, state_id, node)

    def allocate_array(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        node: nodes.AccessNode,
        nodedesc: dt.Data,
        function_stream: CodeIOStream,
        declaration_stream: CodeIOStream,
        allocation_stream: CodeIOStream,
    ) -> None:
        dataname = cpp.ptr(node.data, nodedesc, sdfg, self._frame)

        try:
            self._dispatcher.defined_vars.get(dataname)
            return
        except KeyError:
            pass  # The variable was not defined, we can continue

        # Check if array is already declared
        declared = False
        try:
            self._dispatcher.declared_arrays.get(dataname)
            declared = True  # Array was already declared in this or upper scopes
        except KeyError:  # Array not declared yet
            pass

        if isinstance(nodedesc, dace.data.Stream):
            raise Exception("TODO: Stream?")
        elif isinstance(nodedesc, dace.data.View):
            return self._cpu_codegen.allocate_view(
                sdfg,
                cfg,
                dfg,
                state_id,
                node,
                function_stream,
                declaration_stream,
                allocation_stream,
            )
        elif isinstance(nodedesc, dace.data.Reference):
            return self._cpu_codegen.allocate_reference(
                sdfg,
                cfg,
                dfg,
                state_id,
                node,
                function_stream,
                declaration_stream,
                allocation_stream,
            )

        if nodedesc.lifetime in (
            dtypes.AllocationLifetime.Persistent,
            dtypes.AllocationLifetime.External,
        ):
            nodedesc = update_persistent_desc(nodedesc, sdfg)

        result_decl = StringIO()
        result_alloc = StringIO()
        arrsize = nodedesc.total_size
        is_dynamically_sized = symbolic.issymbolic(arrsize, sdfg.constants)
        arrsize_malloc = "%s * sizeof(%s)" % (sym2cpp(arrsize), nodedesc.dtype.ctype)
        ctypedef = "%s *" % nodedesc.dtype.ctype

        # Different types of GPU arrays
        if nodedesc.storage == dtypes.StorageType.Ascend_Global:
            if not declared:
                result_decl.write("%s %s;\n" % (ctypedef, dataname))
            self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, ctypedef)

            # Strides are left to the user's discretion
            result_alloc.write(
                "DACE_ACL_CHECK(aclrtMalloc((void**)&%s, %s, ACL_MEM_MALLOC_HUGE_FIRST));\n"
                % (dataname, arrsize_malloc)
            )

            if node.setzero:
                result_alloc.write(
                    "DACE_ACL_CHECK(aclrtMemset(%s, 0, %s));\n"
                    % (dataname, arrsize_malloc)
                )
            if isinstance(nodedesc, dt.Array) and nodedesc.start_offset != 0:
                result_alloc.write(
                    f"{dataname} += {cpp.sym2cpp(nodedesc.start_offset)};\n"
                )
        elif nodedesc.storage == dtypes.StorageType.Register:
            if is_dynamically_sized:
                raise ValueError("Dynamic allocation of registers not allowed")
            if nodedesc.start_offset != 0:
                raise NotImplementedError("Start offset unsupported for registers")
            szstr = " = {0}" if node.setzero else ""
            result_decl.write(
                "%s %s[%s]%s;\n"
                % (nodedesc.dtype.ctype, dataname, sym2cpp(arrsize), szstr)
            )
            self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, ctypedef)
        elif nodedesc.storage in dace.dtypes.ASCEND_STORAGES:
            ascend_type_decl = self._get_ascendc_type(nodedesc, nodedesc.storage)
            result_decl.write(f"// Declare {dataname}\n")
            result_decl.write(f"{ascend_type_decl} {dataname};\n")
            if nodedesc.storage == dace.dtypes.StorageType.Ascend_CO1:
                result_decl.write(f"{dataname} = queue_{dataname}.AllocTensor<{nodedesc.dtype.ctype}>();\n")
            self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, ctypedef)
        else:
            raise NotImplementedError(
                "AscendC: Unimplemented storage type " + str(nodedesc.storage)
            )

        declaration_stream.write(result_decl.getvalue(), cfg, state_id, node)
        allocation_stream.write(result_alloc.getvalue(), cfg, state_id, node)

    def deallocate_array(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        node: nodes.AccessNode,
        nodedesc: dt.Data,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ) -> None:
        dataname = cpp.ptr(node.data, nodedesc, sdfg, self._frame)

        if isinstance(nodedesc, dt.Array) and nodedesc.start_offset != 0:
            dataname = f"({dataname} - {cpp.sym2cpp(nodedesc.start_offset)})"

        if self._dispatcher.declared_arrays.has(dataname):
            is_global = nodedesc.lifetime in (
                dtypes.AllocationLifetime.Global,
                dtypes.AllocationLifetime.Persistent,
                dtypes.AllocationLifetime.External,
            )
            self._dispatcher.declared_arrays.remove(dataname, is_global=is_global)

        if isinstance(nodedesc, dace.data.Stream):
            raise Exception("TODO, Stream")
            # return self.deallocate_stream(sdfg, cfg, dfg, state_id, node, nodedesc, function_stream, callsite_stream)
        elif isinstance(nodedesc, dace.data.View):
            return

        if nodedesc.storage == dtypes.StorageType.Ascend_Global:
            callsite_stream.write("DACE_ACL_CHECK(aclrtFree(static_cast<void*>(%s)));\n" % (dataname))
        elif nodedesc.storage in [
            dtypes.StorageType.Ascend_VECIN,
            dtypes.StorageType.Ascend_VECOUT,
        ]:
            ascend_type_decl = self._get_ascendc_type(nodedesc, nodedesc.storage)
            callsite_stream.write(f"// Free {dataname} ?\n")
            # result_alloc.write(f"{ascend_type_decl} {dataname} = queue_{dataname}.AllocTensor<{nodedesc.dtype.ctype}> ();\n")
            # TODO
        else:
            callsite_stream.write(f"// Free {dataname} ?\n")
            #raise NotImplementedError

    def _compute_acl_streams(self, sdfg: SDFG, default_stream=0, default_event=0):
        """Annotates an SDFG (and all nested ones) to include a `_acl_stream`
        field. This field is applied to all GPU maps, tasklets, and copies
        that can be executed in parallel.

        :param sdfg: The sdfg to modify.
        :param default_stream: The stream ID to start counting from (used
                               in recursion to nested SDFGs).
        :param default_event: The event ID to start counting from (used
                              in recursion to nested SDFGs).
        :return: 2-tuple of the number of streams, events to create.
        """
        # TODO: improve this
        return 1, 1

    def _emit_copy(
        self,
        state_id: int,
        src_node: nodes.Node,
        src_storage: dtypes.StorageType,
        dst_node: nodes.Node,
        dst_storage: dtypes.StorageType,
        dst_schedule: dtypes.ScheduleType,
        edge: Tuple[nodes.Node, str, nodes.Node, str, Memlet],
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        callsite_stream: CodeIOStream,
    ) -> None:
        u, uconn, v, vconn, memlet = edge
        state_dfg = cfg.state(state_id)

        cpu_storage_types = [dtypes.StorageType.CPU_Heap]
        ascend_storage_types = [
            dtypes.StorageType.Ascend_Global,
            dtypes.StorageType.Ascend_VECIN,
            dtypes.StorageType.Ascend_VECOUT,
            dtypes.StorageType.Ascend_A1,
            dtypes.StorageType.Ascend_A2,
            dtypes.StorageType.Ascend_B1,
            dtypes.StorageType.Ascend_B2,
            dtypes.StorageType.Ascend_CO2,
            dtypes.StorageType.Ascend_CO1,
        ]

        copy_shape = memlet.subset.bounding_box_size()
        copy_shape = [symbolic.overapproximate(s) for s in copy_shape]
        # Determine directionality
        if isinstance(src_node, nodes.AccessNode) and memlet.data == src_node.data:
            outgoing_memlet = True
        elif isinstance(dst_node, nodes.AccessNode) and memlet.data == dst_node.data:
            outgoing_memlet = False
        else:
            raise LookupError("Memlet does not point to any of the nodes")

        """
        if not isinstance(src_node, nodes.AccessNode) or not isinstance(
            dst_node, nodes.AccessNode):
            #raise ValueError(
            #    "Cannot emit copy for non-access nodes: %s -> %s" % (src_node, dst_node)
            #)
            return

        assert isinstance(src_node, nodes.AccessNode), f"{src_node} -> {dst_node}: {memlet}"
        assert isinstance(dst_node, nodes.AccessNode), f"{src_node} -> {dst_node}: {memlet}"
        """

        if (
            isinstance(src_node, nodes.AccessNode)
            and isinstance(dst_node, nodes.AccessNode)
            and not AscendCCodeGen._in_device_code
            and (
                src_storage
                in [dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Heap]
                or dst_storage
                in [dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Heap]
            )
            and not (
                src_storage in cpu_storage_types and dst_storage in cpu_storage_types
            )
        ):
            src_location = (
                "Device" if src_storage == dtypes.StorageType.Ascend_Global else "Host"
            )
            dst_location = (
                "Device" if dst_storage == dtypes.StorageType.Ascend_Global else "Host"
            )

            # Corner case: A stream is writing to an array
            if isinstance(sdfg.arrays[src_node.data], dt.Stream) and isinstance(
                sdfg.arrays[dst_node.data], (dt.Scalar, dt.Array)
            ):
                return  # Do nothing (handled by ArrayStreamView)

            syncwith = {}  # Dictionary of {stream: event}
            is_sync = False
            max_streams = int(Config.get("compiler", "ascendc", "max_concurrent_streams"))

            if hasattr(src_node, "_acl_stream"):
                ascendstream = src_node._acl_stream
                if not hasattr(dst_node, "_acl_stream"):
                    # Copy after which data is needed by the host
                    is_sync = True
                elif dst_node._acl_stream != src_node._acl_stream:
                    syncwith[dst_node._acl_stream] = getattr(edge, "_ascend_event", None)
                else:
                    pass  # Otherwise, no need to synchronize
            elif hasattr(dst_node, "_acl_stream"):
                ascendstream = dst_node._acl_stream
            else:
                if max_streams >= 0:
                    print("WARNING: Undefined stream, reverting to default")
                if dst_location == "Host":
                    is_sync = True
                ascendstream = "nullptr"

            # Handle case of impending kernel/tasklet on another stream
            if max_streams >= 0:
                for e in state_dfg.out_edges(dst_node):
                    if isinstance(e.dst, nodes.AccessNode):
                        continue
                    if not hasattr(e.dst, "_acl_stream"):
                        is_sync = True
                    elif not hasattr(e, "_ascend_event"):
                        is_sync = True
                    elif e.dst._acl_stream != ascendstream:
                        syncwith[e.dst._acl_stream] = e._ascend_event

                if ascendstream != "nullptr":
                    ascendstream = "__state->acl_context->streams[%d]" % ascendstream

            if memlet.wcr is not None:
                raise NotImplementedError(
                    "Accumulate %s to %s not implemented" % (src_location, dst_location)
                )
            #############################

            # Obtain copy information
            copy_shape, src_strides, dst_strides, src_expr, dst_expr = (
                memlet_copy_to_absolute_strides(
                    self._dispatcher,
                    sdfg,
                    state_dfg,
                    edge,
                    src_node,
                    dst_node,
                    self._cpu_codegen._packed_types,
                )
            )
            dims = len(copy_shape)

            dtype = dst_node.desc(sdfg).dtype

            # Handle unsupported copy types
            if dims == 2 and (src_strides[-1] != 1 or dst_strides[-1] != 1):
                # NOTE: Special case of continuous copy
                # Example: dcol[0:I, 0:J, k] -> datacol[0:I, 0:J]
                # with copy shape [I, J] and strides [J*K, K], [J, 1]
                try:
                    is_src_cont = src_strides[0] / src_strides[1] == copy_shape[1]
                    is_dst_cont = dst_strides[0] / dst_strides[1] == copy_shape[1]
                except (TypeError, ValueError):
                    is_src_cont = False
                    is_dst_cont = False
                if is_src_cont and is_dst_cont:
                    dims = 1
                    copy_shape = [copy_shape[0] * copy_shape[1]]
                    src_strides = [src_strides[1]]
                    dst_strides = [dst_strides[1]]
                else:
                    raise NotImplementedError("2D copy only supported with one stride")

            # Currently we only support ND copies when they can be represented
            # as a 1D copy or as a 2D strided copy
            if dims > 2:
                raise NotImplementedError(
                    "Host Device copies are not supported for N-dimensions if they cannot be represented by a strided copy\n"
                    f"  Nodes: src {src_node} ({src_storage}), dst {dst_node}({dst_storage})\n"
                    f"  Strides: src {src_strides}, dst {dst_strides}"
                )

            if dims == 1 and not (src_strides[-1] != 1 or dst_strides[-1] != 1):
                copysize = " * ".join(_topy(copy_shape))
                array_length = copysize
                copysize += " * sizeof(%s)" % dtype.ctype

                callsite_stream.write(
                    "DACE_ACL_CHECK(aclrtMemcpy(reinterpret_cast<void*>(%s), %s, reinterpret_cast<const void*>(%s), %s, ACL_MEMCPY_%s_TO_%s));\n"
                    % (
                        dst_expr,
                        copysize,
                        src_expr,
                        copysize,
                        src_location.upper(),
                        dst_location.upper(),
                    ),
                    cfg,
                    state_id,
                    [src_node, dst_node],
                )
                node_dtype = dst_node.desc(sdfg).dtype
            elif dims == 1 and ((src_strides[-1] != 1 or dst_strides[-1] != 1)):
                raise NotImplementedError("TODO")
            elif dims == 2:
                raise NotImplementedError("TODO")

            # Post-copy synchronization
            if is_sync:
                # Synchronize with host (done at destination)
                pass
            else:
                # Synchronize with other streams as necessary
                for streamid, event in syncwith.items():
                    syncstream = "__state->acl_context->streams[%d]" % streamid
                    callsite_stream.write("//TODO: sync")

            self._emit_sync(callsite_stream)

        # Copy within the Ascend Device
        elif (
            src_storage in ascend_storage_types and dst_storage in ascend_storage_types
        ):
            name = memlet.data
            nodedesc = sdfg.arrays[name]
            src_name = src_node.data
            dst_name = dst_node.data

            ARCH = dace.config.Config.get("compiler", "ascendc", "soc_version")

            if ARCH == "ascend910":
                import dace.codegen.targets.ascendc.ascendc_copy_utils_910a as tensor_copy_impl
            elif ARCH == "ascend910B4":
                import dace.codegen.targets.ascendc.ascendc_copy_utils_910b as tensor_copy_impl
            else:
                raise NotImplementedError(f"Unsupported architecture: {ARCH}")

            tensor_copy_impl.write_tensor_copy(callsite_stream, memlet, src_name, dst_name, src_storage, dst_storage, nodedesc)
        else:
            self.copy_memory_as_ref(
                sdfg,
                cfg,
                dfg,
                state_id,
                src_node,
                dst_node,
                edge,
                None,
                callsite_stream,
            )

    def copy_memory_as_ref(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        src_node: Union[nodes.Tasklet, nodes.AccessNode],
        dst_node: Union[nodes.Tasklet, nodes.AccessNode],
        edge: Tuple[
            nodes.Node, Optional[str], nodes.Node, Optional[str], dace.memlet.Memlet
        ],
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ) -> None:
        assert isinstance(src_node, nodes.Tasklet) or isinstance(
            dst_node, nodes.Tasklet
        )

        if isinstance(src_node, nodes.Tasklet):
            src_storage = dtypes.StorageType.Register
            try:
                src_parent = dfg.entry_node(src_node)
            except KeyError:
                src_parent = None
            dst_schedule = None if src_parent is None else src_parent.map.schedule
        else:
            src_storage = src_node.desc(sdfg).storage

        if isinstance(dst_node, nodes.Tasklet):
            dst_storage = dtypes.StorageType.Register
        else:
            dst_storage = dst_node.desc(sdfg).storage

        try:
            dst_parent = dfg.entry_node(dst_node)
        except KeyError:
            dst_parent = None
        dst_schedule = None if dst_parent is None else dst_parent.map.schedule

        state_dfg = cfg.node(state_id)

        # Emit actual copy
        self._emit_copy_as_ref(
            sdfg,
            cfg,
            state_id,
            src_node,
            src_storage,
            dst_node,
            dst_storage,
            dst_schedule,
            edge,
            state_dfg,
            callsite_stream,
        )

    def _emit_copy_as_ref(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        state_id: int,
        src_node: nodes.Node,
        src_storage: dtypes.StorageType,
        dst_node: nodes.Node,
        dst_storage: dtypes.StorageType,
        dst_schedule: dtypes.ScheduleType,
        edge: Tuple[
            nodes.Node, Optional[str], nodes.Node, Optional[str], dace.memlet.Memlet
        ],
        dfg: StateSubgraphView,
        stream: CodeIOStream,
    ) -> None:
        u, uconn, v, vconn, memlet = edge
        orig_vconn = vconn
        assert isinstance(dst_node, nodes.Tasklet) or isinstance(
            src_node, nodes.Tasklet
        )

        # Determine memlet directionality
        if isinstance(src_node, nodes.AccessNode) and validate_memlet_data(
            memlet.data, src_node.data
        ):
            write = True
        elif isinstance(dst_node, nodes.AccessNode) and validate_memlet_data(
            memlet.data, dst_node.data
        ):
            write = False
        elif isinstance(src_node, nodes.CodeNode) and isinstance(
            dst_node, nodes.CodeNode
        ):
            # Code->Code copy (not read nor write)
            raise RuntimeError(
                "Copying between code nodes is only supported as part of the participating nodes"
            )
        elif (
            uconn is None
            and vconn is None
            and memlet.data is None
            and dst_schedule == dtypes.ScheduleType.Sequential
        ):
            # Sequential dependency edge
            return
        else:
            raise LookupError("Memlet does not point to any of the nodes")

        if isinstance(dst_node, nodes.Tasklet):
            ascend_type = self._get_ascendc_type(
                sdfg.arrays[edge.data.data], src_storage
            )
            # Copy into tasklet
            access_type = self._get_access_type(ascend_type)
            stream.write(
                f"{ascend_type}{access_type} {vconn} = {memlet.data}; // Type wrapped 1",
                cfg,
                state_id,
                [src_node, dst_node],
            )
            return
        elif isinstance(src_node, nodes.Tasklet):
            raise Exception("TODO: Impl")
            return

    def copy_memory(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        src_node: Union[nodes.Tasklet, nodes.AccessNode],
        dst_node: Union[nodes.CodeNode, nodes.AccessNode],
        memlet: Memlet,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ) -> None:
        state = cfg.state(state_id)
        if isinstance(src_node, nodes.Tasklet):
            src_storage = dtypes.StorageType.Register
            src_parent = state.entry_node(src_node)
            dst_schedule = None if src_parent is None else src_parent.map.schedule
        else:
            src_storage = src_node.desc(sdfg).storage

        if isinstance(dst_node, nodes.Tasklet):
            dst_storage = dtypes.StorageType.Register
        else:
            dst_storage = dst_node.desc(sdfg).storage

        dst_parent = state.entry_node(dst_node)
        dst_schedule = None if dst_parent is None else dst_parent.map.schedule

        # Emit actual copy
        self._emit_copy(
            state_id,
            src_node,
            src_storage,
            dst_node,
            dst_storage,
            dst_schedule,
            memlet,
            sdfg,
            cfg,
            dfg,
            callsite_stream,
        )

    def define_out_memlet(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        state_dfg: StateSubgraphView,
        state_id: int,
        src_node: nodes.Node,
        dst_node: nodes.Node,
        edge: MultiConnectorEdge[Memlet],
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ) -> None:
        # LocalTensor<half> a1Local = queueA1.AllocTensor<half>();
        cdtype = src_node.out_connectors[edge.src_conn]
        desc = sdfg.arrays[edge.data.data]
        #print("DDD", desc, type(desc))
        ascend_type = self._get_ascendc_type(desc, desc.storage)
        access_type = self._get_access_type(ascend_type)
        # ???
        #assert isinstance(cdtype, dtypes.pointer), f"Pointer type expected {cdtype}, type {cdtype}"
        #print("Define out memlet", ascend_type, access_type)

        # If reference set, do not emit initial assignment
        is_refset = (
            isinstance(desc, data.Reference)
            and state_dfg.memlet_path(edge)[-1].dst_conn == "set"
        )

        if not is_refset and not isinstance(desc.dtype, dtypes.pointer):
            ptrname = cpp.ptr(edge.data.data, desc, sdfg, self._frame)
            is_global = desc.lifetime in (
                dtypes.AllocationLifetime.Global,
                dtypes.AllocationLifetime.Persistent,
                dtypes.AllocationLifetime.External,
            )
            defined_type, _ = self._dispatcher.defined_vars.get(
                ptrname, is_global=is_global
            )
            callsite_stream.write(
                f"{ascend_type}{access_type} {edge.src_conn} = {edge.data.data}; // Type wrapped 2",
                cfg,
                state_id,
                src_node,
            )
        else:
            raise Exception("UWU TODO")
            # callsite_stream.write(f'{cdtype.as_arg(edge.src_conn)};', cfg, state_id, src_node)

    def process_out_memlets(self, *args, **kwargs):
        # Call CPU implementation with this code generator as callback
        # self._cpu_codegen.process_out_memlets(*args, codegen=self, **kwargs)
        raise Exception("hmm?")

    def _begin_streams(self, sdfg, state):
        result = set()
        for node in state.source_nodes():
            if hasattr(node, "_acl_stream"):
                if isinstance(node, nodes.AccessNode) and isinstance(
                    sdfg.arrays[node.data], dt.View
                ):
                    continue
                result.add(node._acl_stream)
            else:
                # Collect other streams in state start
                for e in state.out_edges(node):
                    if hasattr(e.dst, "_acl_stream"):
                        if isinstance(node, nodes.AccessNode) and isinstance(
                            sdfg.arrays[node.data], dt.View
                        ):
                            continue
                        result.add(e.dst._acl_stream)
        return result

    def generate_state(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        state: SDFGState,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
        generate_state_footer: bool = False,
    ) -> None:
        # Two modes: device-level state and if this state has active streams
        if AscendCCodeGen._in_device_code:
            self.generate_devicelevel_state(
                sdfg, cfg, state, function_stream, callsite_stream
            )
        else:
            # Active streams found. Generate state normally and sync with the
            # streams in the end
            self._frame.generate_state(
                sdfg,
                cfg,
                state,
                function_stream,
                callsite_stream,
                generate_state_footer=False,
            )

            # Reset thread-block-level information
            self._scope_has_collaborative_copy = False

            # Free pooled memory that needs to be released here
            to_remove = set()
            backend = self.backend

            if state.nosync == False:
                streams_to_sync = set()
                for node in state.sink_nodes():
                    if hasattr(node, "_acl_stream") and node._acl_stream != "nullptr":
                        streams_to_sync.add(node._acl_stream)
                    else:
                        # Synchronize sink-node copies at the end of the state
                        for e in state.in_edges(node):
                            if (
                                hasattr(e.src, "_acl_stream")
                                and e.src._acl_stream != "nullptr"
                            ):
                                streams_to_sync.add(e.src._acl_stream)

                # Relaxed condition for skipping synchronization:
                # if ALL the immediately reachable non-empty states (i.e.,
                # ignoring guard states) use ONLY the same streams as the
                # current state does, and there is only one such stream,
                # then we can skip synchronization.
                next_states = sdutil.get_next_nonempty_states(sdfg, state)
                if next_states and len(streams_to_sync) == 1:
                    if all(
                        self._begin_streams(sdfg, ns) == streams_to_sync
                        for ns in next_states
                    ):
                        # Relax synchronization
                        streams_to_sync = set()

                for stream in streams_to_sync:
                    callsite_stream.write(
                        "//TODO stream \n"
                    )

            # After synchronizing streams, generate state footer normally
            callsite_stream.write("\n")

            # Emit internal transient array deallocation
            self._frame.deallocate_arrays_in_scope(
                sdfg, cfg, state, function_stream, callsite_stream
            )

            # Invoke all instrumentation providers
            """
            for instr in self._frame._dispatcher.instrumentation.values():
                if instr is not None:
                    instr.on_state_end(sdfg, state, callsite_stream, function_stream)
            """

    def generate_devicelevel_state(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        state: SDFGState,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ) -> None:
        self._frame.generate_state(sdfg, cfg, state, function_stream, callsite_stream)

    # NOTE: This function is ONLY called from the CPU side. Therefore, any
    # schedule that is out of the ordinary will raise an exception
    def generate_scope(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg_scope: StateSubgraphView,
        state_id: int,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ) -> None:
        scope_entry = dfg_scope.source_nodes()[0]
        scope_exit = dfg_scope.sink_nodes()[0]

        state = cfg.state(state_id)

        # If in device-level code, call appropriate function
        if (
            self._kernel_map is not None
            and self._kernel_map.map.schedule in dtypes.ASCEND_SCHEDULES
        ):
            self.generate_devicelevel_scope(
                sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream
            )
            return

        # If not device-level code, ensure the schedule is correct
        if scope_entry.map.schedule not in (dtypes.ScheduleType.Ascend_Device,):
            raise TypeError(
                "Cannot schedule %s directly from non-Ascend code"
                % str(scope_entry.map.schedule)
            )

        kernel_name = "%s_%d_%d_%d" % (
            scope_entry.map.label,
            sdfg.cfg_id,
            sdfg.node_id(state),
            state.node_id(scope_entry),
        )

        # Comprehend grid/block dimensions from scopes
        version = dace.config.Config.get("compiler", "ascendc", "soc_version")
        if version == "ascend910":
            block_dims = (32, 1, 1)
        else:
            block_dims = (20, 1, 1)

        # Get parameters of subgraph
        kernel_args = self._arglists[scope_entry]

        # Handle dynamic map inputs
        for e in dace.sdfg.dynamic_map_inputs(state, scope_entry):
            kernel_args[str(e.src)] = e.src.desc(sdfg)

        # Add data from nested SDFGs to kernel arguments
        extra_call_args = []
        extra_call_args_typed = []
        extra_kernel_args = []
        extra_kernel_args_typed = []
        extra_call_args_entry_typed = []
        extra_kernel_args_entry_typed = []
        self.extra_nsdfg_args = []
        visited = set()
        for node, parent in dfg_scope.all_nodes_recursive():
            if isinstance(node, nodes.AccessNode):
                nsdfg: SDFG = parent.parent
                desc = node.desc(nsdfg)
                if (nsdfg, node.data) in visited:
                    continue
                visited.add((nsdfg, node.data))
                if (
                    desc.transient
                    and self._frame.where_allocated[(nsdfg, node.data)] is not nsdfg
                ):
                    outer_name = cpp.ptr(node.data, desc, nsdfg, self._frame)

                    # Create name from within kernel
                    oldval = AscendCCodeGen._in_device_code
                    AscendCCodeGen._in_device_code = True
                    inner_name = cpp.ptr(node.data, desc, nsdfg, self._frame)
                    AscendCCodeGen._in_device_code = oldval

                    raise Exception(
                        "TODO, templated types:",
                        desc.as_arg(name=""),
                        desc.as_arg(name=inner_name),
                    )
                    self.extra_nsdfg_args.append(
                        (desc.as_arg(name=""), inner_name, outer_name)
                    )
                    self._dispatcher.defined_vars.add(
                        inner_name,
                        DefinedType.Pointer,
                        desc.dtype.ctype,
                        allow_shadowing=True,
                    )
                    extra_call_args.append(outer_name)
                    extra_call_args_typed.append(desc.as_arg(name=inner_name))
                    extra_kernel_args.append(f"(void *)&{inner_name}")
                    extra_kernel_args_typed.append(desc.as_arg(name=inner_name))
                    extra_call_args_entry_typed.append(desc.as_arg(name=inner_name))
                    extra_kernel_args_entry_typed.append(desc.as_arg(name=inner_name))

        self.const_params = _get_const_params(dfg_scope)
        # make dynamic map inputs constant
        # TODO move this into _get_const_params(dfg_scope)
        self.const_params |= set(
            (str(e.src)) for e in dace.sdfg.dynamic_map_inputs(state, scope_entry)
        )

        # Store init/exit code streams
        old_entry_stream = self.scope_entry_stream
        old_exit_stream = self.scope_exit_stream
        self.scope_entry_stream = CodeIOStream()
        self.scope_exit_stream = CodeIOStream()

        # Instrumentation for kernel scope
        instr = self._dispatcher.instrumentation[scope_entry.map.instrument]
        if instr is not None:
            pre_outer_stream = CodeIOStream()
            # on_scope_entry(self, sdfg, cfg, state, node, outer_stream, inner_stream, global_stream):
            instr.on_scope_entry(
                sdfg=sdfg,
                cfg=cfg,
                state=state,
                node=scope_entry,
                outer_stream=pre_outer_stream,
                inner_stream=self.scope_entry_stream,
                global_stream=self._globalcode,
            )
            outer_stream = CodeIOStream()
            instr.on_scope_exit(
                sdfg=sdfg,
                cfg=cfg,
                state=state,
                node=scope_exit,
                outer_stream=outer_stream,
                inner_stream=self.scope_exit_stream,
                global_stream=self._globalcode,
            )

        # Redefine constant arguments and rename arguments to device counterparts
        # TODO: This (const behavior and code below) is all a hack.
        #       Refactor and fix when nested SDFGs are separate functions.
        self._dispatcher.defined_vars.enter_scope(scope_entry)
        prototype_kernel_args = {}
        for (
            aname,
            arg,
        ) in (
            kernel_args.items()
        ):  # `list` wrapper is used to modify kernel_args within the loop
            if aname in self.const_params:
                defined_type, ctype = None, None
                if aname in sdfg.arrays:
                    data_desc = sdfg.arrays[aname]
                    is_global = data_desc.lifetime in (
                        dtypes.AllocationLifetime.Global,
                        dtypes.AllocationLifetime.Persistent,
                        dtypes.AllocationLifetime.External,
                    )
                    # Non-free symbol dependent Arrays due to their shape
                    dependent_shape = (
                        isinstance(data_desc, dt.Array)
                        and not isinstance(data_desc, dt.View)
                        and any(
                            str(s) not in self._frame.symbols_and_constants(sdfg)
                            for s in self._frame.free_symbols(data_desc)
                        )
                    )
                    try:
                        # NOTE: It is hard to get access to the view-edge here,
                        # so always check the declared-arrays dictionary for
                        # Views.
                        if dependent_shape or isinstance(data_desc, dt.View):
                            defined_type, ctype = self._dispatcher.declared_arrays.get(
                                aname, is_global=is_global
                            )
                    except KeyError:
                        pass
                    ptrname = cpp.ptr(aname, data_desc, sdfg, self._frame)
                    if not defined_type:
                        defined_type, ctype = self._dispatcher.defined_vars.get(
                            ptrname, is_global=is_global
                        )

                    AscendCCodeGen._in_device_code = True
                    inner_ptrname = cpp.ptr(aname, data_desc, sdfg, self._frame)
                    AscendCCodeGen._in_device_code = False

                    self._dispatcher.defined_vars.add(
                        inner_ptrname,
                        defined_type,
                        "const %s" % ctype,
                        allow_shadowing=True,
                    )

                    # Rename argument in kernel prototype as necessary
                    aname = inner_ptrname
            else:
                if aname in sdfg.arrays:
                    data_desc = sdfg.arrays[aname]
                    ptrname = cpp.ptr(aname, data_desc, sdfg, self._frame)
                    is_global = data_desc.lifetime in (
                        dtypes.AllocationLifetime.Global,
                        dtypes.AllocationLifetime.Persistent,
                        dtypes.AllocationLifetime.External,
                    )
                    defined_type, ctype = self._dispatcher.defined_vars.get(
                        ptrname, is_global=is_global
                    )
                    AscendCCodeGen._in_device_code = True
                    inner_ptrname = cpp.ptr(aname, data_desc, sdfg, self._frame)
                    AscendCCodeGen._in_device_code = False
                    self._dispatcher.defined_vars.add(
                        inner_ptrname, defined_type, ctype, allow_shadowing=True
                    )

                    # Rename argument in kernel prototype as necessary
                    aname = inner_ptrname

            prototype_kernel_args[aname] = arg

        #print(prototype_kernel_args)
        for k, v in prototype_kernel_args.items():
            assert(isinstance(v, dace.data.Array) or isinstance(v, dace.data.Scalar))
        # No const-args allowed
        kernel_args_typed = [
            "GM_ADDR " + k if isinstance(v, dace.data.Array) else v.dtype.ctype + " " + k
            for k, v in prototype_kernel_args.items()
        ]
        kernel_args_entry_typed = [
            "uint8_t* " + k if isinstance(v, dace.data.Array) else v.dtype.ctype + " " + k
            for k, v in prototype_kernel_args.items()
        ]

        kernel_stream = CodeIOStream()
        self._kernel_stream = kernel_stream
        self.generate_kernel_scope(
            sdfg,
            cfg,
            dfg_scope,
            state_id,
            scope_entry.map,
            kernel_name,
            block_dims,
            kernel_args_typed,
            self._globalcode,
            kernel_stream,
        )
        #raise Exception(kernel_args_typed)

        self._dispatcher.defined_vars.exit_scope(scope_entry)

        node = dfg_scope.source_nodes()[0]

        # Write kernel prototype
        self._localcode.write(
            "DACE_EXPORTED __global__ __aicore__ void %s(%s) {\n"
            % (kernel_name, ", ".join(kernel_args_typed + extra_kernel_args_typed)),
            sdfg,
            state_id,
            node,
        )

        # Write constant expressions in GPU code
        self._frame.generate_constants(sdfg, self._localcode)

        self._localcode.write(self.scope_entry_stream.getvalue())

        # Assuming kernel can write to global scope (function_stream), we
        # output the kernel last
        self._localcode.write(kernel_stream.getvalue() + "\n")

        self._localcode.write(self.scope_exit_stream.getvalue())

        # Restore init/exit code streams
        self.scope_entry_stream = old_entry_stream
        self.scope_exit_stream = old_exit_stream

        state_param = [f"{mangle_dace_state_struct_name(self._global_sdfg)} *__state"]

        # Write callback function definition
        self._localcode.write(
            """
DACE_EXPORTED void __dace_runkernel_impl_{fname}({fargs});
DACE_EXPORTED void __dace_runkernel_impl_{fname}({fargs})
{{
""".format(
                fname=kernel_name,
                fargs=", ".join(
                    kernel_args_entry_typed + extra_call_args_typed
                ),
            ),
            cfg,
            state_id,
            node,
        )
        self._localwrappercode.write(
            """
DACE_EXPORTED void __dace_runkernel_{fname}({fargs});
DACE_EXPORTED void __dace_runkernel_impl_{fname}({impl_fargs_no_state});
DACE_EXPORTED void __dace_runkernel_{fname}({fargs})
{{
    {instrumentation_beg}
    __dace_runkernel_impl_{fname}({impl_call_fargs_no_state});

""".format(
                fname=kernel_name,
                fargs=", ".join(
                    state_param + kernel_args_entry_typed + extra_call_args_typed
                ),
                impl_call_fargs_no_state=", ".join(
                    list(prototype_kernel_args.keys()) + extra_call_args
                ),
                impl_fargs_no_state =", ".join(
                    kernel_args_entry_typed + extra_call_args_typed
                ),
                instrumentation_beg=pre_outer_stream.getvalue() if instr is not None and pre_outer_stream is not None else "",
            ),
            cfg,
            state_id,
            node,
        )

        aclstream = "nullptr"

        # make sure dynamic map inputs are properly handled
        for e in dace.sdfg.dynamic_map_inputs(state, scope_entry):
            self._localcode.write(
                self._cpu_codegen.memlet_definition(
                    sdfg, e.data, False, e.dst_conn, e.dst.in_connectors[e.dst_conn]
                ),
                cfg,
                state_id,
                scope_entry,
            )


        #print("KARGS", kernel_args)
        kargs = str(
            ", ".join(
                [f"(GM_ADDR)({k})" if isinstance(v, dace.data.Array) else f"{k}" for k,v in kernel_args.items()]

            )
        )  # join makes them into a tuple of strings somehow if one of them is empty
        # The types here can contain GM_ADDR, they are passed as uint8_t*, we need to cast them

        if len(extra_call_args) != 0:
            raise Exception("TODO, support for: extra_call_args not empty")
        #+ [f"(GM_ADDR)({k})" if isinstance(v, dace.data.Array) else f"{k}" for k,v in extra_call_args.items()]

        assert scope_entry.map.schedule ==  dace.dtypes.ScheduleType.Ascend_Device
        tblock_maps = set([m for m in state.all_nodes_between(scope_entry, state.exit_node(scope_entry)) if isinstance(m, nodes.MapEntry) and m.schedule == dace.dtypes.ScheduleType.Ascend_AiCoreGroup])
        assert len(tblock_maps) == 1
        tblock_map : dace.nodes.Map = tblock_maps.pop()
        r: dace.subsets.Range = tblock_map.range
        dim = 1
        for (b, e, s) in r.ranges:
            try:
                dim *= (e+1 - b) // s
            except Exception as ex:
                raise Exception(
                    f"Error in range {r} of map {tblock_map.label}: {ex} (end - beg) // step should be an integer"
                ) from ex

        self._localcode.write(
            f"""
            {kernel_name}<<<{dim}, nullptr, nullptr>>>({kargs});
            """
        )

        self._emit_sync(self._localwrappercode)

        # Close the runkernel function
        self._localcode.write("}")
        #######################
        # Add invocation to calling code (in another file)
        function_stream.write(
            "DACE_EXPORTED void __dace_runkernel_%s(%s);\n"
            % (
                kernel_name,
                ", ".join(
                    state_param + kernel_args_entry_typed + extra_call_args_entry_typed
                ),
            ),
            cfg,
            state_id,
            scope_entry,
        )

        # If there are dynamic Map inputs, put the kernel invocation in its own scope to avoid redefinitions.
        if dace.sdfg.has_dynamic_map_inputs(state, scope_entry):
            callsite_stream.write("{", cfg, state_id, scope_entry)

        # Synchronize all events leading to dynamic map range connectors
        for e in dace.sdfg.dynamic_map_inputs(state, scope_entry):
            callsite_stream.write(
                self._cpu_codegen.memlet_definition(
                    sdfg, e.data, False, e.dst_conn, e.dst.in_connectors[e.dst_conn]
                ),
                cfg,
                state_id,
                node,
            )

        def add_cast(argname, cast_pattern="uint8_t*"):
            return f"reinterpret_cast<{cast_pattern}>({argname})"


        callsite_stream.write(
            "__dace_runkernel_%s(%s);\n"
            % (
                kernel_name,
                ", ".join(
                    ["__state"]
                    + [
                        add_cast(cpp.ptr(aname, arg, sdfg, self._frame)) if isinstance(arg, dace.data.Array) else cpp.ptr(aname, arg, sdfg, self._frame)
                        for aname, arg in kernel_args.items()
                    ]
                    + extra_call_args
                ),
            ),
            cfg,
            state_id,
            scope_entry,
        )

        # If there are dynamic Map inputs, put the kernel invocation in its own scope to avoid redefinitions.
        if dace.sdfg.has_dynamic_map_inputs(state, scope_entry):
            callsite_stream.write("}", cfg, state_id, scope_entry)

        # synchronize_streams(sdfg, cfg, state, state_id, scope_entry, scope_exit, callsite_stream, self)

        # Instrumentation (post-kernel)

        if instr is not None:
            self._localwrappercode.write(outer_stream.getvalue())

        self._localwrappercode.write("}\n")

    def generate_kernel_scope(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg_scope: ScopeSubgraphView,
        state_id: int,
        kernel_map: nodes.Map,
        kernel_name: str,
        block_dims: list,
        kernel_params: list,
        function_stream: CodeIOStream,
        kernel_stream: CodeIOStream,
    ) -> None:
        state = sdfg.find_state(state_id)
        self._used_arr_set = set()
        # TODO: extend this to track multiple uses
        # TODO: only accesses within the kernel

        for node in state.nodes():
            if isinstance(node, nodes.AccessNode):
                self._used_arr_set.add((node.data, sdfg.arrays[node.data]))
        for edge in state.edges():
            mem = edge.data
            if mem is not None and mem.data is not None:
                arr = sdfg.arrays[mem.data]
                self._used_arr_set.add((mem.data, arr))

        kernel_stream.write(f"// Initialization of Global Storage")
        for name, arr in self._used_arr_set:
            if arr.storage in [dtypes.StorageType.Ascend_Global]:
                kernel_stream.write(f"AscendC::GlobalTensor<{arr.dtype}> {name}_GM;")
                kernel_stream.write(f"__gm__ {arr.dtype.ctype}* {name}_typed = reinterpret_cast<__gm__ {arr.dtype.ctype}*>({name});")

        kernel_stream.write("\n")
        kernel_stream.write(f"// Initialization of Pipe and Queues")
        kernel_stream.write("AscendC::TPipe pipe;")
        for name, arr in self._used_arr_set:
            if arr.storage in dtypes.ASCEND_STORAGES:
                # Global storage uses smth different
                if arr.storage == dace.dtypes.StorageType.Ascend_Global:
                    continue
                que_name = self._storage_to_ascendc_que_name[arr.storage]
                kernel_stream.write(
                    f"AscendC::TQue<AscendC::QuePosition::{que_name}, 1> queue_{name};"
                )

        # Need to find access sizes for the buffer initialization, this can be read from the
        # AiCore group map ranges
        aicore_group_map_entry = None
        kernel_entry = None
        map_entries = []
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry) and node.map == kernel_map:
                if kernel_entry is None:
                    kernel_entry = node
                    break
                else:
                    NotImplementedError(
                        "2 Device Maps Within"
                    )
        assert kernel_entry is not None, f"Could not find kernel map entry {kernel_map.label}, {kernel_map}, {type(kernel_map)}"
        for node in sdutil.dfs_topological_sort(state, kernel_entry):
            if (
                isinstance(node, nodes.MapEntry)
                and node.schedule == dtypes.ScheduleType.Ascend_AiCoreGroup
            ):
                aicore_group_map_entry = node
                break
        for node in sdutil.dfs_topological_sort(state, kernel_entry):
            if (
                isinstance(node, nodes.MapEntry)
            ):
                map_entries.append(node)

        used_arrays_set = set()
        assert kernel_entry is not None
        assert kernel_entry in state.nodes()
        for node in state.all_nodes_between(kernel_entry, state.exit_node(kernel_entry)):
            if isinstance(node, dace.nodes.AccessNode):
                used_arrays_set.add(node.data)

        for arr_name in used_arrays_set:
            arr = sdfg.arrays[arr_name]
            total_size_str = " * ".join([str(s) for s in arr.shape])
            kernel_stream.write(
                f"pipe.InitBuffer(queue_{arr_name}, 1, {total_size_str} * sizeof({arr.dtype.ctype}));"
            )


        node = dfg_scope.source_nodes()[0]
        kernel_stream.write("\n{", cfg, state_id, node)
        # Add more opening braces for scope exit to close
        for dim in range(len(node.map.range) - 1):
            kernel_stream.write("{", cfg, state_id, node)

        # Generate all index arguments for kernel grid
        krange = subsets.Range(kernel_map.range[::-1])
        kdims = krange.size()
        dsym = [
            symbolic.symbol("__DAPB%d" % i, nonnegative=True, integer=True)
            for i in range(len(krange))
        ]
        bidx = krange.coord_at(dsym)

        # handle dynamic map inputs
        for e in dace.sdfg.dynamic_map_inputs(
            sdfg.states()[state_id], dfg_scope.source_nodes()[0]
        ):
            kernel_stream.write(
                self._cpu_codegen.memlet_definition(
                    sdfg, e.data, False, e.dst_conn, e.dst.in_connectors[e.dst_conn]
                ),
                cfg,
                state_id,
                dfg_scope.source_nodes()[0],
            )

        # do not generate an index if the kernel map is persistent
        # Ascend only returns 1D block Id with: GetBlockIdx()

        # First three dimensions are evaluated directly
        #if len(krange) > 1:
        #    raise Exception("TODO: Ranges with more than one dimension")

        kernel_stream.write(f"// Ascend Device Map", sdfg, state_id, node)
        for var, (beg, end, step) in zip(kernel_map.params, kernel_map.range):
            kernel_stream.write(
                f"for ({dace.int64.ctype} {var} = {beg}; {var} < {end+1}; {var} += {step})",
                sdfg,
                state_id,
                node,
            )
            kernel_stream.write("{")
            self._dispatcher.defined_vars.add(var, DefinedType.Scalar, dace.int64.ctype)

        # Delinearize beyond the third dimension
        if len(krange) > 3:
            raise Exception("TODO, >3 dimensions")

        # Dispatch internal code
        assert AscendCCodeGen._in_device_code is False
        AscendCCodeGen._in_device_code = True
        self._kernel_map = node
        self._kernel_state = sdfg.node(state_id)
        self._block_dims = block_dims

        # Emit internal array allocation (deallocation handled at MapExit)
        self._frame.allocate_arrays_in_scope(
            sdfg, cfg, node, function_stream, kernel_stream
        )

        scope_entry = dfg_scope.source_nodes()[0]

        self._dispatcher.dispatch_subgraph(
            sdfg,
            cfg,
            dfg_scope,
            state_id,
            function_stream,
            kernel_stream,
            skip_entry_node=True,
        )

        # Close the opened up for loops

        for _ in kernel_map.params:
            kernel_stream.write("}", cfg, state_id, node)
        for dim in range(len(node.map.range) - 1):
            kernel_stream.write("}", cfg, state_id, node)
        kernel_stream.write("}", cfg, state_id, node)

        self._block_dims = None
        self._kernel_map = None
        self._kernel_state = None
        AscendCCodeGen._in_device_code = False
        self._grid_dims = None

    def get_next_scope_entries(self, dfg, scope_entry):
        parent_scope_entry = dfg.entry_node(scope_entry)
        # We're in a nested SDFG, use full graph
        if parent_scope_entry is None:
            parent_scope = dfg
        else:
            parent_scope = dfg.scope_subgraph(parent_scope_entry)

        # Get all non-sequential scopes from the same level
        all_scopes = [
            node
            for node in parent_scope.bfs_nodes(scope_entry)
            if isinstance(node, nodes.EntryNode)
            and node.map.schedule != dtypes.ScheduleType.Sequential
        ]

        # TODO: Fix to include *next* scopes, without concurrent scopes

        return all_scopes[all_scopes.index(scope_entry) + 1 :]

    def generate_devicelevel_scope(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg_scope: StateSubgraphView,
        state_id: int,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ) -> None:
        # Sanity check
        assert AscendCCodeGen._in_device_code == True

        dfg = cfg.state(state_id)
        scope_entry = dfg_scope.source_nodes()[0]
        scope_exit = dfg_scope.sink_nodes()[0]
        scope_map = scope_entry.map
        next_scopes = self.get_next_scope_entries(dfg, scope_entry)
        state = sdfg.state(state_id)
        dev_entry = state.entry_node(scope_entry)
        assert dev_entry.schedule == dtypes.ScheduleType.Ascend_Device

        # Add extra opening brace (dynamic map ranges, closed in MapExit
        # generator)
        callsite_stream.write("{", cfg, state_id, scope_entry)

        if scope_map.schedule == dtypes.ScheduleType.Ascend_Device:
            callsite_stream.write("// Dev Check")
            dfg_kernel = self._kernel_state.scope_subgraph(self._kernel_map)
        else:
            for dim in range(len(scope_map.range)):
                callsite_stream.write("{", cfg, state_id, scope_entry)

        # Emit internal array allocation (deallocation handled at MapExit)
        self._frame.allocate_arrays_in_scope(
            sdfg, cfg, scope_entry, function_stream, callsite_stream
        )

        # Generate all index arguments for block
        if scope_map.schedule == dtypes.ScheduleType.Ascend_AiCoreGroup:
            brange = subsets.Range(scope_map.range[::-1])
            kdims = brange.size()
            dsym = [
                symbolic.symbol("__DAPT%d" % i, nonnegative=True, integer=True)
                for i in range(len(brange))
            ]
            dsym_end = [
                d + (bs * rng[2]) - 1
                for d, bs, rng in zip(dsym, self._block_dims, brange)
            ]
            tidx = brange.coord_at(dsym)

            # First three dimensions are evaluated directly
            if len(brange) > 2:
                raise Exception("TODO, Ranges with more than 2 dimension")

            print(brange, kdims, dsym, dsym_end, tidx)

            num_cores = brange.num_elements_exact()
            dim_el = [((e+1-b)//s) for b, e, s in brange.ranges]
            num_cores = sympy.prod(dim_el)
            #assert isinstance(num_cores, (sympy.Integer, int))
            offsets = [sympy.prod(dim_el[0:i]) for i in range(len(dim_el))]
            #raise Exception(brange, num_cores, offsets)
            print("tidx", tidx)

            conds = []
            coords = []
            for i in range(min(len(brange), 3)):
                varname = scope_map.params[-i - 1]

                if i == 0:
                    if len(offsets) > 1:
                        block_expr = f"(AscendC::GetBlockIdx() % {offsets[i+1]})"
                    else:
                        block_expr = f"(AscendC::GetBlockIdx())"
                else:
                    block_expr = f"(AscendC::GetBlockIdx() / {offsets[i]})"

                expr = _topy(tidx[i]).replace("__DAPT%d" % i, block_expr)
                callsite_stream.write("// AiCore Group Map")
                callsite_stream.write(
                    "int %s = %s;" % (varname, expr), cfg, state_id, scope_entry
                )
                coords.append("int %s = %s;" % (varname, expr))
            #raise Exception(coords)
            self._dispatcher.defined_vars.add(varname, DefinedType.Scalar, "int")

            # Delinearize beyond the third dimension
            if len(brange) > 3:
                raise Exception("TODO, Ranges with more than 3 dimension")

            # Generate conditions for this block's execution using min and max
            # element, e.g. skipping out-of-bounds threads in trailing block
            minels = brange.min_element()
            maxels = brange.max_element()
            for i, (v, minel, maxel) in enumerate(
                zip(scope_map.params[::-1], minels, maxels)
            ):
                condition = ""

                # Optimize conditions if they are always true
                #############################################

                # Block range start
                if i >= 3 or (dsym[i] >= minel) != True:
                    condition += "%s >= %s" % (v, _topy(minel))

                # Special case: block size is exactly the range of the map (0:b)
                if i >= 3:
                    skipcond = False
                else:
                    skipcond = dsym_end[i].subs({dsym[i]: minel}) == maxel

                # Block range end
                if i >= 3 or (not skipcond and (dsym_end[i] < maxel) != True):
                    if len(condition) > 0:
                        condition += " && "
                    condition += "%s < %s" % (v, _topy(maxel + 1))

                # Emit condition in code
                if len(condition) > 0:
                    callsite_stream.write(
                        "if (%s) {" % condition, cfg, state_id, scope_entry
                    )
                else:
                    callsite_stream.write("{", cfg, state_id, scope_entry)

            # Set Global Buffers
            """
            callsite_stream.write("// Set Global Buffers")
            for e in state.out_edges(scope_entry) + state.out_edges(scope_exit):
                name = e.data.data
                arr = sdfg.arrays[name]
                access_range = e.data.subset
                assert arr.storage == dtypes.StorageType.Ascend_Global
                assert access_range.dims() == 1
                assert access_range.strides() == [1]
                print(
                    1,
                    access_range.absolute_strides(arr.shape),
                    access_range.dims(),
                    access_range.num_elements(),
                )
                beg, _ = access_range.string_list()[0].split(":")
                if arr.storage in [dtypes.StorageType.Ascend_Global]:
                    const = ""
                    if name in self.const_params:
                        const = "const "
                    callsite_stream.write(
                        f"{name}_GM.SetGlobalBuffer(&{name}_typed[{beg}], {access_range.num_elements()});"
                    )
            """

        ##########################################################

        # Generate contents normally
        self._dispatcher.dispatch_subgraph(
            sdfg,
            cfg,
            dfg_scope,
            state_id,
            function_stream,
            callsite_stream,
            skip_entry_node=True,
        )

        # If there are any other threadblock maps down the road,
        # synchronize the thread-block / grid
        parent_scope, _ = xfh.get_parent_map(dfg, scope_entry)

        #if parent_scope.map.schedule == dtypes.ScheduleType.Ascend_Device:
        #    callsite_stream.write("pipe_barrier(PIPE_ALL);", cfg, state_id, scope_entry)
        if (
            len(next_scopes) > 0
            or parent_scope.schedule == dtypes.ScheduleType.Sequential
        ):

            # Thread-block synchronization
            if scope_entry.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
                callsite_stream.write("__syncthreads();", cfg, state_id, scope_entry)
            # Grid synchronization (kernel fusion)
            elif (
                scope_entry.map.schedule == dtypes.ScheduleType.GPU_Device
                and self._kernel_map.schedule == dtypes.ScheduleType.GPU_Device
            ):
                # Escape grid conditions
                for _ in self._kernel_grid_conditions:
                    callsite_stream.write("}", cfg, state_id, scope_entry)

                # Synchronize entire grid
                callsite_stream.write("__gbar.Sync();", cfg, state_id, scope_entry)

                # Rewrite grid conditions
                for cond in self._kernel_grid_conditions:
                    callsite_stream.write(cond, cfg, state_id, scope_entry)

    def generate_node(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        node: nodes.Node,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ) -> None:
        if self.node_dispatch_predicate(sdfg, dfg, node):
            # Dynamically obtain node generator according to class name
            gen = getattr(self, "_generate_" + type(node).__name__, False)
            if gen is not False:  # Not every node type has a code generator here
                gen(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)
                return

        if not AscendCCodeGen._in_device_code:
            self._cpu_codegen.generate_node(
                sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream
            )
            return

        if isinstance(node, nodes.ExitNode):
            self._locals.clear_scope(self._code_state.indentation + 1)

        if AscendCCodeGen._in_device_code and isinstance(node, nodes.MapExit):
            return  # skip

        self._cpu_codegen.generate_node(
            sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream
        )

    def generate_nsdfg_header(
        self, sdfg, cfg, state, state_id, node, memlet_references, sdfg_label
    ):
        return "DACE_DFI " + self._cpu_codegen.generate_nsdfg_header(
            sdfg,
            cfg,
            state,
            state_id,
            node,
            memlet_references,
            sdfg_label,
            state_struct=False,
        )

    def generate_nsdfg_call(
        self, sdfg, cfg, state, node, memlet_references, sdfg_label
    ):
        return self._cpu_codegen.generate_nsdfg_call(
            sdfg, cfg, state, node, memlet_references, sdfg_label, state_struct=False
        )

    def generate_nsdfg_arguments(self, sdfg, cfg, dfg, state, node):
        result = self._cpu_codegen.generate_nsdfg_arguments(sdfg, cfg, dfg, state, node)
        if self.create_grid_barrier:
            result.append(("cub::GridBarrier&", "__gbar", "__gbar"))

        # Add data from nested SDFGs to kernel arguments
        result.extend(
            [(atype, aname, aname) for atype, aname, _ in self.extra_nsdfg_args]
        )
        for arg in self.extra_nsdfg_args:
            defined_type, ctype = self._dispatcher.defined_vars.get(arg[1], 1)
            self._dispatcher.defined_vars.add(arg[1], defined_type, ctype)

        return result

    def _generate_NestedSDFG(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        node: nodes.NestedSDFG,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ) -> None:
        old_schedule = self._toplevel_schedule
        self._toplevel_schedule = node.schedule
        old_codegen = self._cpu_codegen.calling_codegen
        self._cpu_codegen.calling_codegen = self

        self._cpu_codegen._generate_NestedSDFG(
            sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream
        )

        self._cpu_codegen.calling_codegen = old_codegen
        self._toplevel_schedule = old_schedule

    def _generate_MapExit(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        node: nodes.MapExit,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ) -> None:
        if node.map.schedule == dtypes.ScheduleType.GPU_Device:
            # Remove grid invocation conditions
            for i in range(len(node.map.params)):
                if self._kernel_grid_conditions:
                    self._kernel_grid_conditions.pop()

        elif node.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
            # Close block invocation conditions
            for i in range(len(node.map.params)):
                callsite_stream.write("}", cfg, state_id, node)

        elif node.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock_Dynamic:
            # Close lambda function
            callsite_stream.write("});", cfg, state_id, node)
            # Close block invocation
            callsite_stream.write("}", cfg, state_id, node)
            return

        self._cpu_codegen._generate_MapExit(
            sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream
        )

    def _get_thread_id(self) -> str:
        return "AscendC::GetBlockIdx()"

    def _get_warp_id(self) -> str:
        raise Exception("DO NOT CALL")

    def _get_block_id(self) -> str:
        raise Exception("DO NOT CALL")

    def _generate_condition_from_location(
        self,
        name: str,
        index_expr: str,
        node: nodes.Tasklet,
        callsite_stream: CodeIOStream,
    ) -> str:
        if name not in node.location:
            return 0

        location: Union[int, str, subsets.Range] = node.location[name]
        if isinstance(location, str) and ":" in location:
            location = subsets.Range.from_string(location)
        elif symbolic.issymbolic(location):
            location = sym2cpp(location)

        if isinstance(location, subsets.Range):
            # Range of indices
            if len(location) != 1:
                raise ValueError(
                    f"Only one-dimensional ranges are allowed for {name} specialization, {location} given"
                )
            begin, end, stride = location[0]
            rb, re, rs = sym2cpp(begin), sym2cpp(end), sym2cpp(stride)
            cond = ""
            cond += f"(({index_expr}) >= {rb}) && (({index_expr}) <= {re})"
            if stride != 1:
                cond += f" && ((({index_expr}) - {rb}) % {rs} == 0)"

            callsite_stream.write(f"if ({cond}) {{")
        else:
            # Single-element
            callsite_stream.write(f"if (({index_expr}) == {location}) {{")

        return 1

    def _generate_Tasklet(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        node: nodes.Tasklet,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ) -> None:
        generated_preamble_scopes = 0
        if self._in_device_code:
            # If location dictionary prescribes that the code should run on a certain group of threads/blocks,
            # add condition
            # generated_preamble_scopes += self._generate_condition_from_location('ascend_thread', self._get_thread_id(),
            #                                                                    node, callsite_stream)
            # generated_preamble_scopes += self._generate_condition_from_location('gpu_warp', self._get_warp_id(), node,
            #                                                                    callsite_stream)
            # generated_preamble_scopes += self._generate_condition_from_location('gpu_block', self._get_block_id(), node,
            #                                                                    callsite_stream)
            pass

        state = sdfg.state(state_id)
        connectors_to_same_data = dict()
        skip_in_conns = set()
        reque_out_conns = set()
        for iconn, ctype in node.in_connectors.items():
            for oconn, ctype in node.out_connectors.items():
                in_edges = state.in_edges_by_connector(node, iconn)
                out_edges = state.out_edges_by_connector(node, oconn)
                for in_edge in in_edges:
                    for out_edge in out_edges:
                        if in_edge.data.data == out_edge.data.data:
                            connectors_to_same_data[(iconn, oconn)] = in_edge.data.data
                            skip_in_conns.add(iconn)
                            reque_out_conns.add(oconn)
        #raise Exception(connectors_to_same_data, skip_in_conns, skip_out_conns)

        for iconn, ctype in node.in_connectors.items():
            #if iconn in skip_in_conns:
            #    continue
            in_edges = state.in_edges_by_connector(node, iconn)
            for in_edge in in_edges:
                callsite_stream.write(f"{in_edge.data.data} = queue_{in_edge.data.data}.DeQue<{sdfg.arrays[in_edge.data.data].dtype.ctype}>();")

        # TODO: Ensure this is not necessary (decleration would handle alloc tensor?)
        #for oconn, ctype in node.out_connectors.items():
        #    out_edges = state.out_edges_by_connector(node, oconn)
        #    for out_edge in out_edges:
        #        callsite_stream.write(f"{out_edge.data.data} = queue_{out_edge.data.data}.AllocTensor<{sdfg.arrays[out_edge.data.data].dtype.ctype}>();")


        # Call standard tasklet generation
        old_codegen = self._cpu_codegen.calling_codegen
        self._cpu_codegen.calling_codegen = self
        self._cpu_codegen._generate_Tasklet(
            sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream
        )
        self._cpu_codegen.calling_codegen = old_codegen

        if generated_preamble_scopes > 0:
            # Generate appropriate postamble
            for i in range(generated_preamble_scopes):
                callsite_stream.write("}", cfg, state_id, node)

        for oconn, ctype in node.out_connectors.items():
            out_edges = state.out_edges_by_connector(node, oconn)
            for out_edge in out_edges:
                callsite_stream.write(f"queue_{out_edge.data.data}.EnQue<{sdfg.arrays[out_edge.data.data].dtype.ctype}>({out_edge.data.data});")

        for iconn, ctype in node.in_connectors.items():
            if iconn in skip_in_conns:
                continue
            in_edges = state.in_edges_by_connector(node, iconn)
            for in_edge in in_edges:
                callsite_stream.write(f"queue_{in_edge.data.data}.FreeTensor({in_edge.data.data});")



    def make_ptr_vector_cast(self, *args, **kwargs):
        return cpp.make_ptr_vector_cast(*args, **kwargs)


########################################################################
########################################################################
########################################################################
########################################################################
# Helper functions and classes


def _topy(arr):
    """Converts an array of symbolic variables (or one) to C++ strings."""
    if not isinstance(arr, list):
        return cppunparse.pyexpr2cpp(symbolic.symstr(arr, cpp_mode=True))
    return [cppunparse.pyexpr2cpp(symbolic.symstr(d, cpp_mode=True)) for d in arr]


def _named_idx(idx):
    """Converts 0 to x, 1 to y, 2 to z, or raises an exception."""
    if idx < 0 or idx > 2:
        raise ValueError("idx must be between 0 and 2, got %d" % idx)
    return ("x", "y", "z")[idx]


def _get_storagename(storage):
    """Returns a string containing the name of the storage location.
    Example: dtypes.StorageType.GPU_Shared will return "Shared"."""
    sname = str(storage)
    return sname[sname.rindex("_") + 1 :]


def _get_const_params(dfg_scope):
    state = dfg_scope.graph
    sdfg = dfg_scope.parent
    scope_entry = dfg_scope.source_nodes()[0]
    scope_exit = dfg_scope.sink_nodes()[0]
    input_params = set(e.data.data for e in state.in_edges(scope_entry))
    output_params = set(e.data.data for e in state.out_edges(scope_exit))
    toplevel_params = set(
        node.data
        for node in dfg_scope.nodes()
        if isinstance(node, nodes.AccessNode) and sdfg.arrays[node.data].toplevel
    )
    dynamic_inputs = set(
        e.data.data for e in dace.sdfg.dynamic_map_inputs(state, scope_entry)
    )
    return input_params - (output_params | toplevel_params | dynamic_inputs)
