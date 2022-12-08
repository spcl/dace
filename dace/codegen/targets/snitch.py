# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import copy
import dace
import itertools
import numpy as np
import sympy as sp

from dace.transformation.dataflow.streaming_memory import _collect_map_ranges

from dace import registry, data, dtypes, config, sdfg as sd, symbolic
from dace.sdfg import nodes, utils as sdutils
from dace.sdfg.scope import ScopeSubgraphView
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets import cpp
from dace.codegen.common import update_persistent_desc
from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.targets.framecode import DaCeCodeGenerator
from dace.codegen.targets.cpp import sym2cpp
from dace.codegen.dispatcher import DefinedType, TargetDispatcher
from sympy.core.symbol import Symbol

MAX_SSR_STREAMERS = 2
# number of snitch cores executing parallel regions
N_THREADS = 8


def dbg(*args, **kwargs):
    if config.Config.get_bool('debugprint'):
        print("[Snitch] " + " ".join(map(str, args)), **kwargs)


@registry.autoregister_params(name='snitch')
class SnitchCodeGen(TargetCodeGenerator):

    def __init__(self, frame_codegen: DaCeCodeGenerator, sdfg: dace.SDFG):
        ################################################################
        # Define some locals:
        # Can be used to call back to the frame-code generator
        self.frame = frame_codegen
        # Can be used to dispatch other code generators for allocation/nodes
        self.dispatcher = frame_codegen.dispatcher
        # ???
        self.packed_types = False
        # Mapping of ssr to ssr_config
        self.ssrs = MAX_SSR_STREAMERS * [None]

        ################################################################
        # Register handlers/hooks through dispatcher: Can be used for
        # nodes, memory copy/allocation, scopes, states, and more.

        # In this case, register scopes
        self.dispatcher.register_map_dispatcher(dace.ScheduleType.Snitch, self)
        self.dispatcher.register_map_dispatcher(dace.ScheduleType.Snitch_Multicore, self)
        # Snitch_TCDM -> Register

        snitch_storage = [
            dace.StorageType.Snitch_TCDM,
            dace.StorageType.Snitch_L2,
            dace.StorageType.Snitch_SSR,
        ]
        snitch_or_cpu_storage = [
            *snitch_storage, dace.StorageType.Register, dace.StorageType.CPU_Heap, dace.StorageType.CPU_ThreadLocal
        ]
        for src_storage, dst_storage in itertools.chain(itertools.product(snitch_storage, snitch_or_cpu_storage),
                                                        itertools.product(snitch_or_cpu_storage, snitch_storage)):
            self.dispatcher.register_copy_dispatcher(src_storage, dst_storage, None, self)

        # for generate_state
        self.dispatcher.register_state_dispatcher(self, self.state_dispatch_predicate)
        # Registers a function that processes data allocation,
        # initialization, and deinitialization (allocate_array)
        self.dispatcher.register_array_dispatcher(dace.StorageType.Snitch_TCDM, self)
        self.dispatcher.register_array_dispatcher(dace.StorageType.Snitch_SSR, self)

    def state_dispatch_predicate(self, sdfg, state):
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode):
                if "Snitch" in sdfg.arrays[node.label].storage.name:
                    return True
            if isinstance(node, nodes.MapEntry):
                if "Snitch" in node.map.schedule.name:
                    return True
        return False

    def emit_ssr_setup(self, sdfg, state, para, global_stream, callsite_stream):
        if sum([x is not None for x in self.ssrs]) == 0:
            return

        def try_simplify(expr):
            try:
                return sp.simplify(expr)
            except Exception as e:
                return expr

        # for SSR spanning parallel maps, load the thread id here and put the ssr setup in a
        # parallel region
        if para:
            callsite_stream.write(f'unsigned tid = omp_get_thread_num();')

        for ssr_id, ssr in enumerate(self.ssrs):
            if not ssr:
                continue
            dbg(f'emitting ssr config for ssr {ssr}')
            node = ssr["data"]
            alloc_name = cpp.ptr(node.data, node.desc(sdfg))
            # emit bound/stride setup
            stride_off = '0'
            for dim_num, dim in enumerate(ssr["dims"]):
                # SSR setup takes the stride relative to the last dimension and
                # in "bytes"
                # stride = sp.simplify(dim["stride"] - stride_off)
                stride = f'{str(dim["stride"])} - ({stride_off})'
                # stride_off += dim["stride"] * (dim["bound"] - 1)
                stride_off = f'{stride_off} + {str(dim["stride"])} * ({dim["bound"]} - 1)'
                # the bound is one less than the actual bound
                bound = f'{dim["bound"]} - 1' if isinstance(dim["bound"], str) else (dim["bound"] - 1)
                # try to simplify expression
                bound, stride = try_simplify(bound), try_simplify(stride)
                s = '''__builtin_ssr_setup_bound_stride_{dim}d({ssr}, {bound}, sizeof({dtype})*({stride}));'''.format(
                    dim=dim_num + 1,
                    ssr=ssr_id,
                    dtype=ssr["dtype"].ctype,
                    bound=cpp.sym2cpp(bound),
                    stride=cpp.sym2cpp(stride))
                callsite_stream.write(s)

            # repetition
            s = '__builtin_ssr_setup_repetition({ssr}, {reps});'.format(ssr=ssr_id, reps=cpp.sym2cpp(ssr["repeat"]))
            callsite_stream.write(s)
            # emit read/write
            s = '__builtin_ssr_{rw}({ssr}, {dims}, {data});'.format(
                rw="write" if ssr["write"] else "read",
                ssr=ssr_id,
                dims=len(ssr["dims"]) - 1,
                data=f'{alloc_name} + {cpp.sym2cpp(ssr["data_offset"])}')
            callsite_stream.write(s)
        # enable ssr only in non-parallel regime, else, do it inside loop body
        if not para:
            callsite_stream.write('__builtin_ssr_enable();')
        # if para:
        #     callsite_stream.write(f'}}')

    def generate_state(self, sdfg, state, global_stream, callsite_stream, generate_state_footer=True):

        sid = sdfg.node_id(state)
        dbg(f'-- generate state "{state}"')

        # analyze memlets for SSR candidates
        self.ssr_configs = self.ssr_analyze(sdfg, state)
        for ssr_config in self.ssr_configs:
            dbg(f'''SSR Config: data: {ssr_config["data"]} off: {ssr_config["data_offset"]} repeat: {ssr_config["repeat"]} write: {ssr_config["write"]} dims: {len(ssr_config["dims"])} map: {ssr_config["map"]} dst_conn: {ssr_config["dst_conn"]}'''
                )
            dbg(f'  {"dim":4} {"bound":40} {"stride":40}')
            [dbg(f'  {str(i["dim"]):4} {str(i["bound"]):40} {str(i["stride"]):40}') for i in ssr_config["dims"]]

        # allocate SSR streamers for this state
        self.alloc_ssr(sdfg, self.ssr_configs)

        # Emit internal transient array allocation
        # Don't allocate transients shared with another state
        data_to_allocate = (set(state.top_level_transients()) - set(sdfg.shared_transients()))
        allocated = set()
        for node in state.data_nodes():
            if node.data not in data_to_allocate or node.data in allocated:
                continue
            allocated.add(node.data)
            self.dispatcher.dispatch_allocate(sdfg, state, sid, node, global_stream, callsite_stream)

        callsite_stream.write('\n')

        # Emit internal transient array allocation for nested SDFGs
        # TODO: Replace with global allocation management
        gpu_persistent_subgraphs = [
            state.scope_subgraph(node) for node in state.nodes()
            if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.ScheduleType.GPU_Persistent
        ]
        nested_allocated = set()
        for sub_graph in gpu_persistent_subgraphs:
            for nested_sdfg in [n.sdfg for n in sub_graph.nodes() if isinstance(n, nodes.NestedSDFG)]:
                nested_shared_transients = set(nested_sdfg.shared_transients())
                for nested_state in nested_sdfg.nodes():
                    nested_sid = nested_sdfg.node_id(nested_state)
                    nested_to_allocate = (set(nested_state.top_level_transients()) - nested_shared_transients)
                    nodes_to_allocate = [
                        n for n in nested_state.data_nodes()
                        if n.data in nested_to_allocate and n.data not in nested_allocated
                    ]
                    for nested_node in nodes_to_allocate:
                        nested_allocated.add(nested_node.data)
                        self.dispatcher.dispatch_allocate(nested_sdfg, nested_state, nested_sid, nested_node,
                                                          global_stream, callsite_stream)

        callsite_stream.write('\n')

        # Invoke all instrumentation providers
        for instr in self.dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_state_begin(sdfg, state, callsite_stream, global_stream)

        #####################
        # Create dataflow graph for state's children.

        # DFG to code scheme: Only generate code for nodes whose all
        # dependencies have been executed (topological sort).
        # For different connected components, run them concurrently.

        components = dace.sdfg.concurrent_subgraphs(state)

        if len(components) == 1:
            self.dispatcher.dispatch_subgraph(sdfg, state, sid, global_stream, callsite_stream, skip_entry_node=False)
        else:
            if config.Config.get_bool('compiler', 'cpu', 'openmp_sections'):
                callsite_stream.write("#pragma omp parallel sections\n{")
            for c in components:
                if config.Config.get_bool('compiler', 'cpu', 'openmp_sections'):
                    callsite_stream.write("#pragma omp section\n{")
                self.dispatcher.dispatch_subgraph(sdfg, c, sid, global_stream, callsite_stream, skip_entry_node=False)
                if config.Config.get_bool('compiler', 'cpu', 'openmp_sections'):
                    callsite_stream.write("} // End omp section")
            if config.Config.get_bool('compiler', 'cpu', 'openmp_sections'):
                callsite_stream.write("} // End omp sections")

        #####################
        # Write state footer

        if generate_state_footer:

            # Emit internal transient array deallocation for nested SDFGs
            # TODO: Replace with global allocation management
            gpu_persistent_subgraphs = [
                state.scope_subgraph(node) for node in state.nodes()
                if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.ScheduleType.GPU_Persistent
            ]
            nested_deallocated = set()
            for sub_graph in gpu_persistent_subgraphs:
                for nested_sdfg in [n.sdfg for n in sub_graph.nodes() if isinstance(n, nodes.NestedSDFG)]:
                    nested_shared_transients = \
                        set(nested_sdfg.shared_transients())
                    for nested_state in nested_sdfg:
                        nested_sid = nested_sdfg.node_id(nested_state)
                        nested_to_allocate = (set(nested_state.top_level_transients()) - nested_shared_transients)
                        nodes_to_deallocate = [
                            n for n in nested_state.data_nodes()
                            if n.data in nested_to_allocate and n.data not in nested_deallocated
                        ]
                        for nested_node in nodes_to_deallocate:
                            nested_deallocated.add(nested_node.data)
                            self.dispatcher.dispatch_deallocate(nested_sdfg, nested_state, nested_sid, nested_node,
                                                                global_stream, callsite_stream)

            # Emit internal transient array deallocation
            deallocated = set()
            for node in state.data_nodes():
                if (node.data not in data_to_allocate or node.data in deallocated
                        or (node.data in sdfg.arrays and sdfg.arrays[node.data].transient == False)):
                    continue
                deallocated.add(node.data)
                self.dispatcher.dispatch_deallocate(sdfg, state, sid, node, global_stream, callsite_stream)

            # Invoke all instrumentation providers
            for instr in self.dispatcher.instrumentation.values():
                if instr is not None:
                    instr.on_state_end(sdfg, state, callsite_stream, global_stream)

    def define_out_memlet(self, sdfg, state_dfg, state_id, src_node, dst_node, edge, function_stream, callsite_stream):
        cdtype = src_node.out_connectors[edge.src_conn]
        if isinstance(sdfg.arrays[edge.data.data], data.Stream):
            pass
        elif isinstance(cdtype, dtypes.pointer):
            # If pointer, also point to output
            defined_type, _ = self.dispatcher.defined_vars.get(edge.data.data)
            base_ptr = cpp.cpp_ptr_expr(sdfg, edge.data, defined_type)
            callsite_stream.write(f'{cdtype.ctype} {edge.src_conn} = {base_ptr};', sdfg, state_id, src_node)
        else:
            callsite_stream.write(f'{cdtype.ctype} {edge.src_conn};', sdfg, state_id, src_node)

    def memlet_definition(self, sdfg, memlet, output, local_name, conntype=None, allow_shadowing=False, codegen=None):
        # TODO: Robust rule set
        if conntype is None:
            raise ValueError('Cannot define memlet for "%s" without '
                             'connector type' % local_name)
        codegen = codegen or self
        # Convert from Data to typeclass
        if isinstance(conntype, data.Data):
            if isinstance(conntype, data.Array):
                conntype = dtypes.pointer(conntype.dtype)
            else:
                conntype = conntype.dtype

        is_scalar = not isinstance(conntype, dtypes.pointer)
        is_pointer = isinstance(conntype, dtypes.pointer)

        # Allocate variable type
        memlet_type = conntype.dtype.ctype

        desc = sdfg.arrays[memlet.data]
        ptr = cpp.ptr(memlet.data, desc)

        var_type, ctypedef = self.dispatcher.defined_vars.get(memlet.data)
        result = ''
        expr = (cpp.cpp_array_expr(sdfg, memlet, with_brackets=False)
                if var_type in [DefinedType.Pointer, DefinedType.StreamArray, DefinedType.ArrayInterface] else ptr)

        # Special case: ArrayInterface, append _in or _out
        _ptr = ptr
        if var_type == DefinedType.ArrayInterface:
            # Views have already been renamed
            if not isinstance(desc, data.View):
                ptr = cpp.array_interface_variable(ptr, output, self.dispatcher)
        if expr != _ptr:
            expr = '%s[%s]' % (ptr, expr)
        # If there is a type mismatch, cast pointer
        expr = cpp.make_ptr_vector_cast(expr, desc.dtype, conntype, is_scalar, var_type)

        defined = None

        if var_type in [DefinedType.Scalar, DefinedType.Pointer, DefinedType.ArrayInterface]:
            if output:
                if is_pointer and var_type == DefinedType.ArrayInterface:
                    result += "{} {} = {};".format(memlet_type, local_name, expr)
                elif not memlet.dynamic or (memlet.dynamic and memlet.wcr is not None):
                    # Dynamic WCR memlets start uninitialized
                    result += "{} {};".format(memlet_type, local_name)
                    defined = DefinedType.Scalar

            else:
                if not memlet.dynamic:
                    if is_scalar:
                        # We can pre-read the value
                        result += "{} {} = {};".format(memlet_type, local_name, expr)
                    else:
                        # Pointer reference
                        result += "{} {} = {};".format(ctypedef, local_name, expr)
                else:
                    # Variable number of reads: get a const reference that can
                    # be read if necessary
                    memlet_type = '%s const' % memlet_type
                    result += "{} &{} = {};".format(memlet_type, local_name, expr)
                defined = (DefinedType.Scalar if is_scalar else DefinedType.Pointer)
        elif var_type in [DefinedType.Stream, DefinedType.StreamArray]:
            if not memlet.dynamic and memlet.num_accesses == 1:
                if not output:
                    result += f'{memlet_type} {local_name} = ({expr}).pop();'
                    defined = DefinedType.Scalar
            else:
                # Just forward actions to the underlying object
                memlet_type = ctypedef
                result += "{} &{} = {};".format(memlet_type, local_name, expr)
                defined = DefinedType.Stream
        else:
            raise TypeError("Unknown variable type: {}".format(var_type))

        if defined is not None:
            self.dispatcher.defined_vars.add(local_name, defined, memlet_type, allow_shadowing=allow_shadowing)

        dbg(f'    memlet definition: "{result}"')
        return result

    def allocate_array(self, sdfg, dfg, state_id, node, global_stream, function_stream, declaration_stream,
                       allocation_stream) -> None:
        dbg('-- allocate_array')
        name = node.data
        nodedesc = node.desc(sdfg)

        # NOTE: The code below fixes symbol-related issues with transient data originally defined in a NestedSDFG scope
        # but promoted to be persistent. These data must have their free symbols replaced with the corresponding
        # top-level SDFG symbols.
        if nodedesc.lifetime == dtypes.AllocationLifetime.Persistent:
            nodedesc = update_persistent_desc(nodedesc, sdfg)

        # Compute array size
        arrsize = nodedesc.total_size
        arrsize_bytes = arrsize * nodedesc.dtype.bytes
        alloc_name = cpp.ptr(name, nodedesc)
        dbg('  arrsize "{}" arrsize_bytes "{}" alloc_name "{}" nodedesc "{}"'.format(
            arrsize, arrsize_bytes, alloc_name, nodedesc))

        if isinstance(nodedesc, data.Array):
            ctypedef = dtypes.pointer(nodedesc.dtype).ctype

            # catch Threadlocal storage
            if nodedesc.storage is dtypes.StorageType.CPU_ThreadLocal:
                # Define pointer once
                # NOTE: OpenMP threadprivate storage MUST be declared globally.
                if not self.dispatcher.defined_vars.has(name):
                    global_stream.write(
                        "{ctype} *{name};\n#pragma omp threadprivate({name})".format(ctype=nodedesc.dtype.ctype,
                                                                                     name=name),
                        sdfg,
                        state_id,
                        node,
                    )
                    self.dispatcher.defined_vars.add_global(name, DefinedType.Pointer, '%s *' % nodedesc.dtype.ctype)
                # Allocate in each OpenMP thread
                allocation_stream.write(
                    """
                    #pragma omp parallel
                    {{
                        #error "malloc is not threadsafe"
                        {name} = new {ctype} [{arrsize}];""".format(ctype=nodedesc.dtype.ctype,
                                                                    name=alloc_name,
                                                                    arrsize=cpp.sym2cpp(arrsize)),
                    sdfg,
                    state_id,
                    node,
                )
                # Close OpenMP parallel section
                allocation_stream.write('}')

            elif not symbolic.issymbolic(arrsize, sdfg.constants):
                # static allocation
                declaration_stream.write(f'// static allocate storage "{nodedesc.storage}"')
                if node.desc(sdfg).lifetime == dace.AllocationLifetime.Persistent:
                    # Don't put a static if it is declared in the state struct for C compliance
                    declaration_stream.write(f'{nodedesc.dtype.ctype} {name}[{cpp.sym2cpp(arrsize)}];\n', sdfg,
                                             state_id, node)
                else:
                    declaration_stream.write(f'static {nodedesc.dtype.ctype} {name}[{cpp.sym2cpp(arrsize)}];\n', sdfg,
                                             state_id, node)
                self.dispatcher.defined_vars.add(name, DefinedType.Pointer, ctypedef)
            else:
                # malloc array
                declaration_stream.write(f'// allocate storage "{nodedesc.storage}"')
                declaration_stream.write(f'{nodedesc.dtype.ctype} *{name};\n', sdfg, state_id, node)
                allocation_stream.write(
                    f'''{alloc_name} = ({nodedesc.dtype.ctype}*)malloc(sizeof({nodedesc.dtype.ctype})*({cpp.sym2cpp(arrsize)}));\n''',
                    sdfg, state_id, node)
                self.dispatcher.defined_vars.add(name, DefinedType.Pointer, ctypedef)
        else:
            if (nodedesc.storage is dtypes.StorageType.CPU_Heap or nodedesc.storage is dtypes.StorageType.Snitch_TCDM):
                ctypedef = dtypes.pointer(nodedesc.dtype).ctype
                declaration_stream.write(f'// allocate scalar storage "{nodedesc.storage}"')
                declaration_stream.write(f'{nodedesc.dtype.ctype} {name}[1];\n', sdfg, state_id, node)
                self.dispatcher.defined_vars.add(name, DefinedType.Pointer, ctypedef)
            else:
                raise NotImplementedError("Unimplemented storage type " + str(nodedesc.storage))

    def deallocate_array(self, sdfg, dfg, state_id, node, nodedesc, function_stream, callsite_stream):
        arrsize = nodedesc.total_size
        alloc_name = cpp.ptr(node.data, nodedesc)
        dbg(f'-- deallocate_array storate="{nodedesc.storage}" arrsize="{arrsize}" alloc_name="{alloc_name}"')

        if isinstance(nodedesc, data.Scalar):
            return
        elif isinstance(nodedesc, data.View):
            return
        elif isinstance(nodedesc, data.Stream):
            return
        elif (nodedesc.storage == dtypes.StorageType.CPU_Heap
              or (nodedesc.storage == dtypes.StorageType.Snitch_TCDM and symbolic.issymbolic(arrsize, sdfg.constants))
              or (nodedesc.storage == dtypes.StorageType.Snitch_SSR and symbolic.issymbolic(arrsize, sdfg.constants))
              or (nodedesc.storage == dtypes.StorageType.Register and symbolic.issymbolic(arrsize, sdfg.constants))):
            # free array
            if nodedesc.storage == dtypes.StorageType.Snitch_SSR:
                dbg(f'Check deallocation of SSR datatypes!!!')
                callsite_stream.write(f"// free of an SSR type\n", sdfg, state_id, node)
            if not symbolic.issymbolic(arrsize, sdfg.constants):
                # don't free static allocations
                return
            callsite_stream.write(f'// storage "{nodedesc.storage}"\n')
            callsite_stream.write(f"free({alloc_name});\n", sdfg, state_id, node)
            return
        elif nodedesc.storage is dtypes.StorageType.CPU_ThreadLocal:
            # Deallocate in each OpenMP thread
            callsite_stream.write(
                """#pragma omp parallel
                {{
                    delete[] {name};
                }}""".format(name=alloc_name),
                sdfg,
                state_id,
                node,
            )
        else:
            return

    def copy_memory(
        self,
        sdfg,
        dfg,
        state_id,
        src_node,
        dst_node,
        edge,
        function_stream,
        callsite_stream,
    ):
        dbg(f'-- Copy dispatcher for {src_node}({type(src_node)})->{dst_node}({type(dst_node)})')

        # get source storage type
        if isinstance(src_node, nodes.Tasklet):
            src_storage = dtypes.StorageType.Register
            try:
                src_parent = dfg.entry_node(src_node)
            except KeyError:
                src_parent = None
            dst_schedule = None if src_parent is None else src_parent.map.schedule
        else:
            src_storage = src_node.desc(sdfg).storage

        # get destination storage type
        if isinstance(dst_node, nodes.Tasklet):
            dst_storage = dtypes.StorageType.Register
        else:
            dst_storage = dst_node.desc(sdfg).storage

        # find destination schedule
        try:
            dst_parent = dfg.entry_node(dst_node)
        except KeyError:
            dst_parent = None
        dst_schedule = None if dst_parent is None else dst_parent.map.schedule

        state_dfg = sdfg.node(state_id)

        dbg(f'  storage type {src_storage}->{dst_storage}')
        callsite_stream.write(f'// storage type {src_storage}->{dst_storage}', sdfg, state_id, [src_node, dst_node])

        u, uconn, v, vconn, memlet = edge

        # Determine memlet directionality
        if isinstance(src_node, nodes.AccessNode) and memlet.data == src_node.data:
            write = True
        elif isinstance(dst_node, nodes.AccessNode) and memlet.data == dst_node.data:
            write = False
        elif isinstance(src_node, nodes.CodeNode) and isinstance(dst_node, nodes.CodeNode):
            # Code->Code copy (not read nor write)
            raise RuntimeError("Copying between code nodes is only supported as"
                               " part of the participating nodes")
        else:
            raise LookupError("Memlet does not point to any of the nodes")

        if isinstance(dst_node, nodes.Tasklet):
            # Copy into tasklet
            dbg('  copy into tasklet')
            # catch SSR
            candidates = []
            if src_storage == dace.StorageType.Snitch_SSR:
                candidates = [
                    i for i, x in enumerate(self.ssrs)
                    if x is not None and x["data"].data == memlet.data and x["dst_conn"] == vconn and x["tasklet"] == v
                ]
            if src_storage == dace.StorageType.Snitch_SSR and len(candidates):
                dbg(f'    memlet.data {memlet.data} candidates {candidates}')
                streamer = candidates[0]
                callsite_stream.write(f'// copy into tasklet SSR{streamer}')
                callsite_stream.write(
                    "{} {} = __builtin_ssr_pop({});".format(dst_node.in_connectors[vconn].dtype.ctype, vconn, streamer),
                    sdfg, state_id, [src_node, dst_node])
            else:
                callsite_stream.write('// copy into tasklet')
                callsite_stream.write(
                    "    " + self.memlet_definition(sdfg, memlet, False, vconn, dst_node.in_connectors[vconn]),
                    sdfg,
                    state_id,
                    [src_node, dst_node],
                )
            return
        elif isinstance(src_node, nodes.Tasklet):
            # Copy out of tasklet
            dbg('  copy cut of tasklet')
            raise NotImplementedError
            # callsite_stream.write(
            #     "    " + self.memlet_definition(sdfg, memlet, True, uconn,
            #                                     src_node.out_connectors[uconn]),
            #     sdfg,
            #     state_id,
            #     [src_node, dst_node],
            # )
            # return
        else:  # Copy array-to-array
            dbg('  copy array-to-array')
            src_nodedesc = src_node.desc(sdfg)
            dst_nodedesc = dst_node.desc(sdfg)
            if write:
                vconn = dst_node.data
            ctype = dst_nodedesc.dtype.ctype
            state_dfg = sdfg.nodes()[state_id]

            #############################################
            # Corner cases ignored
            #############################################

            copy_shape, src_strides, dst_strides, src_expr, dst_expr = \
                cpp.memlet_copy_to_absolute_strides(
                    self.dispatcher, sdfg, state_dfg, edge, src_node, dst_node,
                    self.packed_types)
            dbg(f'  copy_shape = "{copy_shape}", src_strides = "{src_strides}", dst_strides = "{dst_strides}", src_expr = "{src_expr}", dst_expr = "{dst_expr}"'
                )

            # 2D transfer?
            if len(copy_shape) == 2:
                xfer = '''__builtin_sdma_start_twod( 
                        (uint64_t)({src}), (uint64_t)({dst}), 
                        {size}, {sstride}, {dstride}, {nrep}, {cfg});'''.format(
                    src=src_expr,
                    dst=dst_expr,
                    size=f'sizeof({src_nodedesc.dtype.ctype})*({cpp.sym2cpp(copy_shape[1])})',
                    sstride=f'sizeof({src_nodedesc.dtype.ctype})*({cpp.sym2cpp(src_strides[0])})',
                    dstride=f'sizeof({dst_nodedesc.dtype.ctype})*({cpp.sym2cpp(dst_strides[0])})',
                    nrep=cpp.sym2cpp(copy_shape[0]),
                    cfg='0')
            # 1D transfer?
            elif len(copy_shape) == 1:
                # if only a single element, perform a load
                if isinstance(copy_shape[0], int) and copy_shape[0] == 1:
                    # if None:
                    xfer = '''*({dst}) = *({src});'''.format(src=src_expr, dst=dst_expr)
                    callsite_stream.write(xfer, sdfg, state_id, [src_node, dst_node])
                    return
                else:
                    if src_strides[0] == 1 and dst_strides[0] == 1:
                        xfer = '''__builtin_sdma_start_oned( 
                                (uint64_t)({src}), (uint64_t)({dst}), 
                                {size}, {cfg});'''.format(
                            src=src_expr,
                            dst=dst_expr,
                            size=f'sizeof({src_nodedesc.dtype.ctype})*({cpp.sym2cpp(copy_shape[0])})',
                            cfg='0')
                    else:
                        xfer = '''__builtin_sdma_start_twod( 
                                (uint64_t)({src}), (uint64_t)({dst}), 
                                {size}, {sstride}, {dstride}, {nrep}, {cfg});'''.format(
                            src=src_expr,
                            dst=dst_expr,
                            size=f'sizeof({src_nodedesc.dtype.ctype})',
                            sstride=f'sizeof({src_nodedesc.dtype.ctype})*({cpp.sym2cpp(src_strides[0])})',
                            dstride=f'sizeof({dst_nodedesc.dtype.ctype})*({cpp.sym2cpp(dst_strides[0])})',
                            nrep=cpp.sym2cpp(copy_shape[0]),
                            cfg='0')

            else:
                raise NotImplementedError('Unsupported dimnesions')

            # emit transfer
            callsite_stream.write(xfer, sdfg, state_id, [src_node, dst_node])
            # emit wait for idle
            callsite_stream.write('__builtin_sdma_wait_for_idle();', sdfg, state_id, [src_node, dst_node])

    # A scope dispatcher will trigger a method called generate_scope whenever
    # an SDFG has a scope with that schedule
    def generate_scope(self, sdfg: dace.SDFG, scope: ScopeSubgraphView, state_id: int, function_stream: CodeIOStream,
                       callsite_stream: CodeIOStream):
        # The parameters here are:
        # sdfg: The SDFG we are currently generating.
        # scope: The subgraph of the state containing only the scope (map contents)
        #        we want to generate the code for.
        # state_id: The state in the SDFG the subgraph is taken from (i.e.,
        #           `sdfg.node(state_id)` is the same as `scope.graph`)
        # function_stream: A cursor to the global code (which can be used to define
        #                  functions, hence the name).
        # callsite_stream: A cursor to the current location in the code, most of
        #                  the code is generated here.

        # We can get the map entry node from the scope graph
        entry_node = scope.source_nodes()[0]
        dbg(f'-- generate scope entry_node="{entry_node}" type="{type(entry_node)}"')

        # Encapsulate map with a C scope
        callsite_stream.write('{', sdfg, state_id, entry_node)

        ssr_region = sum([x is not None and x["map"] == entry_node for x in self.ssrs]) != 0
        para = entry_node.map.schedule == dace.ScheduleType.Snitch_Multicore

        # in a parallel region, emit SSR in parallel section
        if para and ssr_region:
            callsite_stream.write(f'#pragma omp parallel')
            callsite_stream.write(f'{{')

        # emit the SSR setup calls if this map is is one of the ssrs
        if ssr_region:
            non_null_ssrs = [x for x in self.ssrs if x]
            callsite_stream.write(f'// ssr allocated: {len(non_null_ssrs)}: {[x["data"] for x in non_null_ssrs]}')
            self.emit_ssr_setup(sdfg, sdfg.states()[state_id], para, function_stream, callsite_stream)

        # loop over out edges which are the in edges to the tasklet
        # for e in scope.out_edges(entry_node):
        #     desc = sdfg.arrays[e.data.data]
        #     # if access pattern is suitable for SSR...
        #     if False:
        #         # keep track on which memlet is mapped do which SSR
        #         self.map_memlets_to_ssr[e] = 1

        # decorate woth omp pragma for parallel maps
        if para:
            if ssr_region:
                s = f'#pragma omp for schedule(static)'
            else:
                s = f'#pragma omp parallel for schedule(static)'
            # append private variables
            private_vars = [
                var for var in sdfg.shared_transients() if sdfg.arrays[var].storage == dace.dtypes.StorageType.Register
            ]
            if len(private_vars) > 0:
                s += f' firstprivate({",".join(private_vars)})'
            # emit
            callsite_stream.write(s)

        ################################################################
        # Generate specific code: We will generate a reversed loop with a
        # comment for each dimension of the map. For the sake of simplicity,
        # dynamic map ranges are not supported.

        for param, rng in zip(entry_node.map.params, entry_node.map.range):
            dbg(f'  opening for parameter {param}')
            # We use the sym2cpp function from the cpp support functions
            # to convert symbolic expressions to proper C++
            begin, end, stride = (sym2cpp(r) for r in rng)
            end = sym2cpp(rng[1] + 1)

            # Every write is optionally (but recommended to be) tagged with
            # 1-3 extra arguments, serving as line information to match
            # SDFG, state, and graph nodes/edges to written code.
            callsite_stream.write(
                f'''// Loopy-loop {param}
            for (int {param} = {begin}; {param} < {end}; {param} += {stride}) {{''', sdfg, state_id, entry_node)

            # NOTE: CodeIOStream will automatically take care of indentation for us.

        # enable SSR in loop body if any are enabled and we are in a parallel region
        if ssr_region:
            for ssr_id, ssr in enumerate([x for x in self.ssrs if x]):
                if ssr["map"].schedule == dtypes.ScheduleType.Snitch_Multicore:
                    callsite_stream.write('__builtin_ssr_enable();')
                    break

        # Emit internal transient array allocation
        to_allocate = sdutils.local_transients(sdfg, scope, entry_node)
        dbg(f'  to_allocate:{to_allocate} scope childern: {scope.scope_children()[entry_node]}')
        allocated = set()
        for child in scope.scope_children()[entry_node]:
            if not isinstance(child, nodes.AccessNode):
                continue
            if child.data not in to_allocate or child.data in allocated:
                continue
            allocated.add(child.data)
            dbg(f'  calling allocate for {child.data}')
            self.dispatcher.dispatch_allocate(sdfg, scope, state_id, child, function_stream, callsite_stream)

        # Now that the loops have been defined, use the dispatcher to invoke any
        # code generator (including this one) that is registered to deal with
        # the internal nodes in the subgraph. We skip the MapEntry node.
        self.dispatcher.dispatch_subgraph(sdfg,
                                          scope,
                                          state_id,
                                          function_stream,
                                          callsite_stream,
                                          skip_entry_node=True,
                                          skip_exit_node=True)

        # Emit internal transient array deallocation
        to_allocate = sdutils.local_transients(sdfg, scope, entry_node)
        deallocated = set()
        for child in scope.scope_children()[entry_node]:
            if not isinstance(child, nodes.AccessNode):
                continue
            if child.data not in to_allocate or child.data in deallocated:
                continue
            deallocated.add(child.data)
            self.dispatcher.dispatch_deallocate(sdfg, scope, state_id, child, None, callsite_stream)

        dbg(f'  after dispatch_subgraph')

        # disable SSR in loop body if any are enabled and we are in a parallel region
        if ssr_region:
            for ssr_id, ssr in enumerate([x for x in self.ssrs if x]):
                if ssr["map"].schedule == dtypes.ScheduleType.Snitch_Multicore:
                    callsite_stream.write('__builtin_ssr_enable();')
                    break

        for param, rng in zip(entry_node.map.params, entry_node.map.range):
            dbg(f'  closing for parameter {param}')
            callsite_stream.write(f'''// end loopy-loop
                                    }}''', sdfg, state_id, entry_node)

        if ssr_region:
            # callsite_stream.write(f'// end ssr allocated: {len(self.ssr_configs)}')
            # if there is at least one SSR active, disable the region here
            if para:
                callsite_stream.write(f'}} // omp parallel')
            else:
                callsite_stream.write(f'__builtin_ssr_disable();')
            # deallocate SSRs
            for i, x in enumerate([x for x in self.ssrs if x]):
                if x["map"] == entry_node:
                    self.ssrs[i] = None

        # End-encapsulate map with a C scope
        callsite_stream.write('}', sdfg, state_id, entry_node)

        # postamble code for disabling SSR comes here
        # for param, rng in zip(entry_node.map.params, entry_node.map.range):
        #     # closing brace
        #     callsite_stream.write(f'''// end loopy-loop
        #                         }}''',
        #                           sdfg, state_id, entry_node)

    def alloc_ssr(self, sdfg, ssr_configs):
        """Given a list of ssr configurations, allocate SSR streamers to the data and change
        the corresponding storage types"""

        if len(ssr_configs) < 0:
            return
        allocated = 0
        for streamer, sc in enumerate(ssr_configs):
            if sc["map"] is None:
                continue
            # map streamer to data
            self.ssrs[allocated] = sc
            # change storage type to SSR
            sc["data"].desc(sdfg).storage = dace.StorageType.Snitch_SSR
            allocated += 1
            if allocated == MAX_SSR_STREAMERS:
                break

    def ssr_analyze(self, sdfg, state):
        """Analyze the `state` for possible SSR candidates and generate the SSR config. Returns
        a list of ssr configuration dicts"""

        dbg('-- ssr_analyze for state', state)

        def match_exp(expr, param, ignore):
            """Takes a symbolic expression `expr` and a string parameter `param` and 
            tries to match the expression in the form a*param+b. On success, returns
            the tuple (a,b), else None"""
            ignore = [dace.symbol(i) for i in ignore]
            # a, b = sp.Wild('a'), sp.Wild('b')
            a, b = sp.Wild('a', exclude=ignore), sp.Wild('b', exclude=[param])
            # a -> stride, b -> offset
            dic = expr.match(a * param + b)
            dbg(f'matching expr "{expr}" for param "{param}" ignoring "{ignore}" result "{dic}"')
            # dbg(f'match: {dic}')
            if dic:
                return dic[a], dic[b]
            return None
            # if dic and dic[a].is_integer and dic[b].is_integer:
            #     return dic[a],dic[b]
            # return None

        # iterate over all nodes in the sdfg that have the dace.StorageType.Snitch_SSR
        # storage class
        ssr_configs = []
        for node in [n for n in state.data_nodes() if n.desc(sdfg).storage == dace.StorageType.Snitch_SSR]:
            desc = node.desc(sdfg)

            # ignore non array
            if not isinstance(desc, dace.data.Array):
                continue

            # ignore non double dtype
            if not desc.dtype.as_numpy_dtype() == np.float64:
                continue

            # get acces type: read and write
            w_node, r_node = node.has_writes(state), node.has_reads(state)
            node_strides = node.desc(sdfg).strides

            # get node output edges
            out_edges = state.out_edges(node)
            # dbg(f'  out_edges: {out_edges}')

            # analyze all nodes (read/write/RW)
            # And assume the user knows what they're doing
            if True:
                if len(out_edges) == 0:
                    continue
                dbg(f'\nnode "{node}" with strides {node_strides} desc "{node.desc(sdfg)}"')
                # for each outgoing edge
                memlet_paths = []
                for oe in out_edges:
                    # an out edge can be a tree of memlet paths,
                    # get a list of all memlet paths
                    leafs = list([
                        x.edge for x in state.memlet_tree(oe).traverse_children()
                        if isinstance(x.edge.dst, dace.sdfg.nodes.Tasklet)
                    ])
                    memlet_paths += [state.memlet_path(leaf) for leaf in leafs]

                for memlet_path in memlet_paths:
                    map_ranges = _collect_map_ranges(state, memlet_path)

                    stop = False
                    for mp in memlet_path:
                        if isinstance(mp.src, dace.sdfg.nodes.MapEntry):
                            if len(mp.src.map.params) != 1:
                                # TODO: implement this
                                dbg(f'SSR crossing multi-param maps not supportd. Call MapExpand on the map "{mp.src.map}"'
                                    )
                                stop = True
                    if stop:
                        continue

                    # out edge doesn't traverse at least one map
                    if len(map_ranges) == 0:
                        continue

                    max_dims = 4
                    ssr_config = {
                        "data": node,
                        "repeat": 0,
                        "write": False,
                        "dims": [],
                        "dtype": desc.dtype,
                        "data_offset": 0,
                        "dst_conn": memlet_path[-1].dst_conn,
                        "tasklet": memlet_path[-1].dst
                    }
                    ssr_config["map"] = None
                    # collect all induction variables of the maps to later ignore them
                    ignore_syms = [rng[0] for rng in map_ranges]
                    data_offset = 0
                    prev_map = 0
                    for edge, rng, dim in zip(reversed(memlet_path), reversed(map_ranges), range(max_dims)):

                        # if parent is not a map, stop
                        if not isinstance(edge.src, dace.sdfg.nodes.MapEntry):
                            break

                        # store a reference to the map before which the SSR configutaion
                        # must be emitted
                        ssr_config["map"] = edge.src

                        # if the parent map is scheduled for parallelism, stop
                        if edge.src.schedule == dtypes.ScheduleType.CPU_Multicore:
                            ssr_config["map"] = prev_map
                            break
                        prev_map = edge.src
                        memlet = edge.data
                        param, (map_begin, map_end, map_stride) = rng
                        begin, end, strd = rng[1]
                        ssr_bound = (end - begin + strd) / strd
                        dbg(f'  begin, end, strd {begin} {end} {strd}')

                        # determine omp schedule if so specified
                        if edge.src.schedule == dtypes.ScheduleType.Snitch_Multicore:
                            loopSize = (end - begin) / strd + 1
                            loopSize = loopSize
                            # chunk = sp.floor(loopSize / N_THREADS)
                            chunk = loopSize / N_THREADS
                            leftOver = loopSize - chunk * N_THREADS
                            stride = loopSize
                            thds = []
                            # same as for static scheduling in kmp.c
                            thd_sym = dace.symbol('tid')

                            my_chunk = f'{str(chunk + 1)} if {str(thd_sym)} < {str(leftOver)} else {str(chunk)}'

                            beg_lt = str(thd_sym * strd * (chunk + 1))
                            beg_else = thd_sym * strd * chunk + leftOver
                            beg_else = str((beg_else if not isinstance(loopSize, dace.symbolic.symbol) else beg_else))
                            my_begin = f'{beg_lt} if {str(thd_sym)} < {str(leftOver)} else {beg_else}'

                            dbg(f'  OMP loopSize, chunk, leftOver, begin [{loopSize}, {sym2cpp(my_chunk)}, {leftOver}, {my_begin}]'
                                )

                            # overwrite ssr_bounds, stride stays the same
                            ssr_bound = my_chunk

                        # the index=induction variable of the current map as symbol
                        induction_var = dace.symbol(rng[0])
                        # remove this variable from the ignore list
                        ignore_syms.remove(rng[0])

                        # extract subset access pattern
                        subset_access = memlet.subset.min_element()
                        abs_access = sum([a * b for a, b in zip(subset_access, node_strides)])
                        match_ind = match_exp(abs_access, induction_var, ignore_syms)
                        dbg(f'  subset_access: "{subset_access}"')
                        dbg(f'  match_ind: "{match_ind}"')
                        dbg(f'  abs_access:    "{abs_access}"')
                        # Match same expression for loop variable offset
                        off_corr = 0
                        if begin != 0:
                            if len(begin.free_symbols) == 0:
                                # no free symbols -> const offset
                                off_corr = -begin
                            else:
                                match_begin = match_exp(abs_access, begin, ignore_syms)
                                if not match_begin:
                                    # TODO: Verify
                                    # raise NotImplementedError('This should be equal to begin == 0')
                                    off_corr = 0
                                else:
                                    off_corr = begin * match_begin[0]
                            dbg(f'    found offset correction "{off_corr}" "{type(off_corr)}"')

                        if match_ind:
                            ssr_stride = match_ind[0] * strd
                            if strd != 1:
                                # raise NotImplementedError('Check this case where strd != 1')
                                dbg('WARNING: Check this case where strd != 1')

                            off = sum([a * b for a, b in zip(subset_access, node_strides)])
                            # dbg(f'set offset to ssr_config["data_offset"]={ssr_config["data_offset"] - begin}')
                        else:
                            # didn't find induction variable in subset use -> set stride to 0
                            ssr_stride = 0

                        # omp offset correction based on stride
                        off_omp = 0
                        if edge.src.schedule == dtypes.ScheduleType.Snitch_Multicore:
                            off_omp = f'{str(ssr_stride)} * ({my_begin})'

                        ssr_config["data_offset"] = f'{sym2cpp(sp.simplify(match_ind[1] - off_corr))} + ({off_omp})'
                        dbg(f'    base off "{str(sp.simplify(match_ind[1] - off_corr))}" omp off "{off_omp}"')
                        dbg(f'    set offset to ssr_config["data_offset"]={ssr_config["data_offset"]}')

                        ssr_config["dims"].append({'dim': dim, 'bound': ssr_bound, 'stride': ssr_stride})

                    ssr_configs.append(ssr_config)

        return ssr_configs

    def write_and_resolve_expr(self, sdfg, memlet, nc, outname, inname, indices=None, dtype=None):
        """
        Emits a conflict resolution call from a memlet.
        """
        from dace.frontend import operations
        redtype = operations.detect_reduction_type(memlet.wcr)
        atomic = "_atomic" if not nc else ""
        defined_type, _ = self.dispatcher.defined_vars.get(memlet.data)

        if isinstance(indices, str):
            ptr = '%s + %s' % (cpp.cpp_ptr_expr(sdfg, memlet, defined_type), indices)
        else:
            ptr = cpp.cpp_ptr_expr(sdfg, memlet, defined_type, indices=indices)
        if isinstance(dtype, dtypes.pointer):
            dtype = dtype.base_type
        # If there is a type mismatch, cast pointer
        if isinstance(dtype, dtypes.vector):
            ptr = f'({dtype.ctype} *)({ptr})'

        # catch non-conflicting types
        if nc:
            if redtype == dtypes.ReductionType.Sum:
                op = "+="
                return (f'*({ptr}) {op} {inname}')
            else:
                raise NotImplementedError("Unimplemented reduction type " + str(redtype))
                # fmt_str='inline {t} reduction_{sdfgid}_{stateid}_{nodeid}({t} {arga}, {t} {argb}) {{ {unparse_wcr_result} }}'
                # fmt_str.format(t=dtype.ctype,
                #   sdfgid=sdfg.sdfg_id, stateid=42, nodeid=43, unparse_wcr_result=cpp.unparse_cr_split(sdfg,memlet.wcr)[0],
                #   arga=cpp.unparse_cr_split(sdfg,memlet.wcr)[1][0],argb=cpp.unparse_cr_split(sdfg,memlet.wcr)[1][1])
                # sdfgid=sdfg.sdfg_id
                # stateid=42
                # nodeid=43
                # return (f'reduction_{sdfgid}_{stateid}_{nodeid}(*({ptr}), {inname})')

        # Special call for detected reduction types
        if redtype != dtypes.ReductionType.Custom:
            credtype = "dace::ReductionType::" + str(redtype)[str(redtype).find(".") + 1:]
            return (f'dace::wcr_fixed<{credtype}, {dtype.ctype}>::reduce{atomic}('
                    f'{ptr}, {inname})')

        # General reduction
        custom_reduction = cpp.unparse_cr(sdfg, memlet.wcr, dtype)
        return (f'dace::wcr_custom<{dtype.ctype}>:: template reduce{atomic}('
                f'{custom_reduction}, {ptr}, {inname})')

    @staticmethod
    def gen_code_snitch(sdfg):
        """Take an SDFG an generate Snitch compatible C code and header file"""

        import re
        # Disable parallel sections in frame generation
        config.Config.set('compiler', 'cpu', 'openmp_sections', value=False)

        # Generate the code
        code = sdfg.generate_code()[0]

        # Generate headers
        hdrs = ""
        init_params = (sdfg.name, sdfg.name, sdfg.signature(with_types=True, for_call=False, with_arrays=False))
        call_params = sdfg.signature(with_types=True, for_call=False)
        if len(call_params) > 0:
            call_params = ', ' + call_params
        params = (sdfg.name, sdfg.name, call_params)
        exit_params = (sdfg.name, sdfg.name)
        hdrs += 'typedef void * %sHandle_t;\n' % sdfg.name
        hdrs += '#ifdef __cplusplus\nextern "C" {\n#endif\n'
        hdrs += '%sHandle_t __dace_init_%s(%s);\n' % init_params
        hdrs += 'void __dace_exit_%s(%sHandle_t handle);\n' % exit_params
        hdrs += 'void __program_%s(%sHandle_t handle%s);\n' % params
        hdrs += '#ifdef __cplusplus\n}\n#endif\n'

        # Fixup some includes
        code._code = code._code.replace("#include \"../../include/hash.h\"", '', 1)
        code._code = code._code.replace('<dace/dace.h>', '"dace/dace.h"', 1)
        code._code = code._code.replace('dace::float64', '(double)')
        code._code = code._code.replace('dace::int64', '(int64_t)')
        code._code = code._code.replace('dace::math::pow', 'pow')
        # __unused is reserved in C
        code._code = code._code.replace('__unused', '_unused_var')

        # change new/delete to malloc/free
        code._code = re.sub(r"new (.+) \[(\d*)\];", r"(\1*)malloc(\2*sizeof(\1));", code._code)
        code._code = re.sub(r"new ([a-zA-Z0-9 _]*);", r"(\1*)malloc(sizeof(\1));", code._code)
        code._code = re.sub(r"delete (.*);", r"free(\1);", code._code)
        code._code = re.sub(r"delete\[\] (.*);", r"free(\1);", code._code)

        # prepend all uses of the state struct with `struct`
        ccode = code.clean_code
        state_struct = re.findall(r"struct (\w+) {", ccode)
        if len(state_struct) == 1:
            # match all occurences, except for the one prepended by "struct "
            # dbg(f'found declaration of state struct {state_struct}')
            state_struct = state_struct[0]
            ccode = re.sub(r"(?<!struct )({})".format(state_struct), r"struct {}".format(state_struct), ccode)

        # replace stuff
        replace = [
            ('DACE_EXPORTED', 'DACE_C_EXPORTED'),
            ('nullptr', 'NULL'),
            ('constexpr', 'static const'),
            ('inline ', 'static inline ')  # change to static scope
        ]
        for (i, o) in replace:
            ccode = ccode.replace(i, o)

        return (ccode, hdrs)
