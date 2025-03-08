# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""

import copy
import dace
import typing
from dace.codegen.control_flow import ConditionalBlock, ControlFlowBlock
from dace.data import Property, make_properties
from dace.transformation import pass_pipeline as ppl

@make_properties
class DuplicateConstArrays(ppl.Pass):
    verbose: bool = Property(dtype=bool, default=True, desc="Print debug information")

    def __init__(
        self,
        verbose: bool = True
    ):
        self.verbose = verbose
        super().__init__()

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


    def _set_def_to_gpu_global(self, sdfg: dace.SDFG):
        for arr_name, arr in sdfg.arrays.items():
            if arr.storage == dace.dtypes.StorageType.Default:
                arr.storage = dace.dtypes.StorageType.GPU_Global
        for state in sdfg.states():
            for node in sdfg.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    self._set_def_to_gpu_global(node.sdfg)

    def wrap_tasklets(self, sdfg: dace.SDFG, l):
        # If tasklet has no scope, then:
        # 1. Put it in a GPU_device map
        # 2. All edges that come from GPU storage Access ndoes OK
        # 3. All edges that do not come from GPU storage Access nodes, not OK, copy to GPU
        # 4. For all other edges (coming from tasklets for example)
        # 4.1. If from GPU storage, move it through an access node
        # 4.2  If from CPU storage, move it through an access while copying to GPU

        for s in sdfg.states():
            for node in s.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    ies = s.in_edges(node)
                    oes = s.out_edges(node)
                    try:
                        scope = s.scope_dict()[node]
                    except Exception as e:
                        sdfg.save("wrapped_failing.sdfgz", compress=True)
                        raise e

                    if scope is not None:
                        continue
                    if node.label not in l:
                        continue

                    map_entry, map_exit = s.add_map("wrap", {"i": "0:0"}, dace.ScheduleType.GPU_Device)
                    edges_to_rm = set()
                    edges_to_add = set()
                    for ie in ies:
                        name = ie.data.data
                        edges_to_rm.add(ie)
                        if isinstance(ie.src, dace.nodes.AccessNode):
                            if sdfg.arrays[name].storage == dace.dtypes.StorageType.GPU_Global:
                                mem = copy.deepcopy(ie.data)
                                edges_to_add.add((ie.src, ie.src_conn, map_entry, f"IN_{name}", mem))
                                edges_to_add.add((map_entry, f"OUT_{name}", ie.dst, ie.dst_conn, copy.deepcopy(mem)))
                                map_entry.add_in_connector(f"IN_{name}")
                                map_entry.add_out_connector(f"OUT_{name}")
                            else:
                                if "gpu_" + name not in sdfg.arrays:
                                    newdesc = copy.deepcopy(sdfg.arrays[name])
                                    newdesc.storage = dace.dtypes.StorageType.GPU_Global
                                    sdfg.add_datadesc("gpu_" + name, newdesc)
                                an = s.add_access("gpu_" + name)
                                mem = copy.deepcopy(ie.data)
                                mem.data = "gpu_" + name
                                edges_to_add.add((ie.src, ie.src_conn, an, None, copy.deepcopy(ie.data)))
                                edges_to_add.add((an, None, map_entry, f"IN_gpu_{name}", copy.deepcopy(mem)))
                                edges_to_add.add((map_entry, f"OUT_gpu_{name}", ie.dst, ie.dst_conn, copy.deepcopy(mem)))
                                map_entry.add_in_connector(f"IN_gpu_{name}")
                                map_entry.add_out_connector(f"OUT_gpu_{name}")
                        else:
                            if sdfg.arrays[name].storage == dace.dtypes.StorageType.GPU_Global:
                                an = s.add_access(name)
                                mem = copy.deepcopy(ie.data)
                                edges_to_add.add((ie.src, ie.src_conn, an, None, copy.deepcopy(mem)))
                                edges_to_add.add((an, None, map_entry, f"IN_{name}", mem))
                                edges_to_add.add((map_entry, f"OUT_{name}", ie.dst, ie.dst_conn, copy.deepcopy(mem)))
                                map_exit.add_in_connector(f"IN_{name}")
                                map_exit.add_out_connector(f"OUT_{name}")
                            else:
                                if "gpu_" + name not in sdfg.arrays:
                                    newdesc = copy.deepcopy(sdfg.arrays[name])
                                    newdesc.storage = dace.dtypes.StorageType.GPU_Global
                                    sdfg.add_datadesc("gpu_" + name, newdesc)
                                an0 = s.add_access(name)
                                an = s.add_access("gpu_" + name)
                                mem = copy.deepcopy(ie.data)
                                mem.data = "gpu_" + name
                                edges_to_add.add((ie.src, ie.src_conn, an0, None, copy.deepcopy(ie.data)))
                                edges_to_add.add((an0, None, an, None, copy.deepcopy(ie.data)))
                                edges_to_add.add((an, None, map_entry, f"IN_gpu_{name}", copy.deepcopy(mem)))
                                edges_to_add.add((map_entry, f"OUT_gpu_{name}", ie.dst, ie.dst_conn, copy.deepcopy(mem)))
                                map_entry.add_in_connector(f"IN_gpu_{name}")
                                map_entry.add_out_connector(f"OUT_gpu_{name}")
                    if len(ies) == 0:
                        edges_to_add.add((map_entry, None, node, None, dace.memlet.Memlet(None)))
                    assert len(oes) > 0
                    for oe in oes:
                        name = oe.data.data
                        edges_to_rm.add(oe)
                        if isinstance(oe.dst, dace.nodes.AccessNode):
                            if sdfg.arrays[name].storage == dace.dtypes.StorageType.GPU_Global:
                                #an = s.add_access("host_" + ie.data)
                                mem = copy.deepcopy(oe.data)
                                edges_to_add.add((oe.src, oe.src_conn, map_exit, f"IN_{name}", mem))
                                edges_to_add.add((map_exit, f"OUT_{name}", oe.dst, oe.dst_conn, copy.deepcopy(mem)))
                                map_exit.add_in_connector(f"IN_{name}")
                                map_exit.add_out_connector(f"OUT_{name}")
                            else:
                                if "gpu_" + name not in sdfg.arrays:
                                    newdesc = copy.deepcopy(sdfg.arrays[name])
                                    newdesc.storage = dace.dtypes.StorageType.GPU_Global
                                    sdfg.add_datadesc("gpu_" + name, newdesc)
                                an = s.add_access("gpu_" + name)
                                mem = copy.deepcopy(oe.data)
                                mem.data = "gpu_" + name
                                edges_to_add.add((oe.src, oe.src_conn, map_exit, f"IN_gpu_{name}", copy.deepcopy(mem)))
                                edges_to_add.add((map_exit, f"OUT_gpu_{name}", an, None, copy.deepcopy(mem)))
                                edges_to_add.add((an, None, oe.dst, oe.dst_conn, copy.deepcopy(oe.data)))
                                map_exit.add_in_connector(f"IN_gpu_{name}")
                                map_exit.add_out_connector(f"OUT_gpu_{name}")
                        else:
                            if sdfg.arrays[name].storage == dace.dtypes.StorageType.GPU_Global:
                                an = s.add_access(name)
                                mem = copy.deepcopy(oe.data)
                                edges_to_add.add((oe.src, oe.src_conn, map_exit, f"IN_{name}",  copy.deepcopy(mem)))
                                edges_to_add.add((map_exit, f"OUT_{name}", an, None, copy.deepcopy(mem)))
                                edges_to_add.add((an, None, oe.dst, oe.dst_conn, copy.deepcopy(oe.data)))
                                map_exit.add_in_connector(f"IN_{name}")
                                map_exit.add_out_connector(f"OUT_{name}")
                            else:
                                if "gpu_" + name not in sdfg.arrays:
                                    newdesc = copy.deepcopy(sdfg.arrays[name])
                                    newdesc.storage = dace.dtypes.StorageType.GPU_Global
                                    sdfg.add_datadesc("gpu_" + name, newdesc)
                                an = s.add_access(name)
                                an0 = s.add_access("gpu_" + name)
                                mem = copy.deepcopy(oe.data)
                                mem.data = "gpu_" + name
                                edges_to_add.add((oe.src, oe.src_conn, map_exit, f"IN_gpu_{name}", copy.deepcopy(mem)))
                                edges_to_add.add((map_exit, f"OUT_gpu_{name}", an0, None, copy.deepcopy(mem)))
                                edges_to_add.add((an0, None, an, None, mem))
                                edges_to_add.add((an, None, oe.dst, oe.dst_conn, copy.deepcopy(oe.data)))
                                map_exit.add_in_connector(f"IN_gpu_{name}")
                                map_exit.add_out_connector(f"OUT_gpu_{name}")
                    for e in edges_to_rm:
                        s.remove_edge(e)
                    for e in edges_to_add:
                        s.add_edge(*e)

                if isinstance(node, dace.nodes.NestedSDFG):
                    self.wrap_tasklets(node.sdfg, l)

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: typing.Dict[str, typing.Any]) -> int:
        # Filter or const arrays (nothing is written to them)
        # This means no in edge, or memlet is None on all in edges
        arrays_written_to = {k: 0 for k, v in sdfg.arrays.items() if isinstance(v, dace.data.Array)}
        arrays = set([arr_name for arr_name, arr in sdfg.arrays.items() if isinstance(arr, dace.data.Array)])
        wrap_list = pipeline_results["wrap_list"]

        def collect_writes(sdfg: dace.SDFG):
            for cfg in sdfg.nodes():
                for node in cfg.nodes():
                    if isinstance(node, dace.nodes.AccessNode):
                        if len(cfg.in_edges(node)) > 0:
                            for ie in cfg.in_edges(node):
                                if ie.data is not None and ie.dst_conn != "views":
                                    if isinstance(sdfg.arrays[node.data], dace.data.Array):
                                        arrays_written_to[node.data] += 1
                    if isinstance(node, dace.nodes.NestedSDFG):
                        collect_writes(node.sdfg)


        # Let initialization be ok (writing just once)
        non_const_arrays = set([k for k, v in arrays_written_to.items() if v > 1])
        const_arrays = arrays - non_const_arrays

        if self.verbose:
            print("Non const arrays:")
            print(non_const_arrays)
            print("Const arrays:")
            print(const_arrays)

        self._set_def_to_gpu_global(sdfg)
        #ssdfg.simplify()

        # 1. Copy all these arrays in the first state to GPU
        # R. need to do it after flattening.
        # 1.1 After we detect the first access node, append a copy to GPU
        # 1.2 If we do not detect it do it to the first state
        arrays_to_copy = copy.deepcopy(const_arrays)
        host_gpu_name_map = dict()
        gpu_host_name_map = dict()
        visited_data = set()

        #if cfg.out_degree(node) == 0 and cfg.in_degree(node) == 1:
        for const_arr_name in const_arrays:
            # Add a copy to GPU
            # 1.2 If we do not detect it do it to the first state
            # 1.1 After we detect the first access node, append a copy to GPU
            arr = sdfg.arrays[const_arr_name]
            if arr.storage == dace.dtypes.StorageType.GPU_Global:
                if self.verbose:
                    print(f"Copying {const_arr_name} to host as host_{const_arr_name}")

                # Check if this array is a gpu variant of an array
                if const_arr_name.startswith("gpu_") and const_arr_name[4:] in sdfg.arrays:
                    gpu_host_name_map[const_arr_name[4:]] = "host_" + const_arr_name[4:]
                    host_gpu_name_map["host_" + const_arr_name[4:]] = const_arr_name[4:]
                else:
                    gpu_host_name_map[const_arr_name] = "host_" + const_arr_name
                    host_gpu_name_map["host_" + const_arr_name] = const_arr_name
                    if "host_" + const_arr_name not in sdfg.arrays:
                        newdesc = copy.deepcopy(arr)
                        newdesc.storage = dace.dtypes.StorageType.CPU_Heap
                        sdfg.add_datadesc("host_" + const_arr_name, newdesc)
                        #an = cfg.add_access("host_" + const_arr_name)
                        #cfg.add_edge(node, None, an, None, dace.memlet.Memlet(expr=const_arr_name))
            else:
                assert arr.storage == dace.dtypes.StorageType.CPU_Heap or arr.storage == dace.dtypes.StorageType.Default
                if self.verbose:
                    print(f"Copying {const_arr_name} to GPU as gpu_{const_arr_name}")

                if (not const_arr_name.startswith("gpu_")) and "gpu_" + const_arr_name in sdfg.arrays:
                    gpu_host_name_map["gpu_" + const_arr_name] = const_arr_name
                    host_gpu_name_map[const_arr_name] = "gpu_" + const_arr_name
                else:
                    gpu_host_name_map["gpu_" + const_arr_name] = const_arr_name
                    host_gpu_name_map[const_arr_name] = "gpu_" + const_arr_name
                    if "gpu_" + const_arr_name not in sdfg.arrays:
                        newdesc = copy.deepcopy(arr)
                        newdesc.storage = dace.dtypes.StorageType.GPU_Global
                        sdfg.add_datadesc("gpu_" + const_arr_name, newdesc)

        # TODO initialization

        # 2.1 If host code and accesses GPU, access Host version instead
        # 2.2 If device code and accesses Host, access GPU version instead

        # 2.1.a for interstate edges


        # Only do it top level and hope that the accessing gpu data on interstate edge only occurs at the top level
        def duplicate_views(sdfg: dace.SDFG):
            for state in sdfg.states():
                added_views = dict()
                for node in state.nodes():
                    if isinstance(node, dace.nodes.AccessNode):
                        if len(state.out_edges(node)) == 1:
                            src, srconn, dst, dstconn, memlet = state.out_edges(node)[0]
                            if isinstance(dst, dace.nodes.AccessNode) and dstconn == "views":
                                src_arr = sdfg.arrays[src.data]
                                dst_arr = sdfg.arrays[dst.data]
                                if src.data in const_arrays and dst.data in const_arrays:
                                    if src_arr.storage == dace.dtypes.StorageType.GPU_Global:
                                        an_src = state.add_access(gpu_host_name_map[src.data]) if gpu_host_name_map[src.data] not in added_views else added_views[gpu_host_name_map[src.data]]
                                        an_dst = state.add_access(gpu_host_name_map[dst.data]) if gpu_host_name_map[dst.data] not in added_views else added_views[gpu_host_name_map[dst.data]]
                                    else:
                                        an_src = state.add_access(host_gpu_name_map[src.data]) if host_gpu_name_map[src.data] not in added_views else added_views[host_gpu_name_map[src.data]]
                                        an_dst = state.add_access(host_gpu_name_map[dst.data]) if host_gpu_name_map[dst.data] not in added_views else added_views[host_gpu_name_map[dst.data]]

                                    mem = copy.deepcopy(memlet)
                                    mem.data = gpu_host_name_map[mem.data] if src_arr.storage == dace.dtypes.StorageType.GPU_Global else host_gpu_name_map[mem.data]
                                    state.add_edge(an_src, None, an_dst, "views", mem)
                                    if dst.data not in added_views:
                                        added_views[dst.data] = an_dst
                                    if src.data not in added_views:
                                        added_views[src.data] = an_src

        sdfg.simplify(validate=False)
        duplicate_views(sdfg)

        for e in  sdfg.all_interstate_edges(recursive=False):
            interstate_edge : dace.InterstateEdge = e.data
            interstate_edge.replace_dict(gpu_host_name_map)

        #sdfg.validate()

        self.wrap_tasklets(sdfg, l=wrap_list)
