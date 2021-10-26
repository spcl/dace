# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

from collections import defaultdict
from dace import (data, dtypes, nodes, properties, registry, memlet as mm,
                  subsets, symbolic)
from dace.sdfg import SDFG, SDFGState, utils as sdutil, graph as gr
from typing import Dict, List, Tuple
from dace.transformation import transformation as xf
from dace.transformation.dataflow import streaming_memory as sm
import networkx as nx
import warnings
import copy
from dace.libraries.standard import Gearbox
import dace

from dace.transformation.dataflow.streaming_memory import _canonicalize_memlet, _collect_map_ranges, _do_memlets_correspond, _streamify_recursive


@registry.autoregister_params(singlestate=True)
@properties.make_properties
class MemoryBuffering(sm.StreamingMemory):
    """
     The MemoryBuffering transformation allows reading/writing data from memory
     using a wider data format (e.g. 512 bits), and then convert it
     on the fly to the right data type used by the computation.
     A combination of the StreamingMemory transformation and a gearbox node. 
    """
    # TODO: 512 bits
    # TODO: Replace / with //
    vector_size = properties.Property(dtype=int, default=2, desc='Vector Size')

    def can_be_applied(self,
                       graph: SDFGState,
                       candidate: Dict[xf.PatternNode, int],
                       expr_index: int,
                       sdfg: SDFG,
                       strict: bool = False) -> bool:

        # Same requierements as in the StreamingMemory transformation
        # and the Stride has to be 1
        access = graph.node(candidate[MemoryBuffering.access])
        desc = sdfg.arrays[access.data]

        # print(desc.strides)
        # print(desc.strides[-1])

        # print(desc.strides == (1,))

        # TODO: correct stride check?
        # TODO: make general
        # TODO: Only makes sense if Global array

        return super().can_be_applied(graph, candidate, expr_index, sdfg,
                                      strict) and desc.strides[-1] == 1

    def apply(self, sdfg: SDFG) -> nodes.AccessNode:

        state = sdfg.node(self.state_id)
        dnode: nodes.AccessNode = self.access(sdfg)
        desc = sdfg.arrays[dnode.data]

        print("sdfg.arrays[dnode.data] = ", sdfg.arrays[dnode.data])
        print(self.access(sdfg))
        arrname = str(self.access(sdfg))
        print()


        if self.expr_index == 0:
            print("Read")
            edges = state.out_edges(dnode)
            #  Read
            gearbox_input_type = dtypes.vector(desc.dtype, self.vector_size)
            gearbox_output_type = desc.dtype
        else:
            print("Write")
            # Write
            edges = state.in_edges(dnode)
            gearbox_input_type = desc.dtype
            gearbox_output_type = dtypes.vector(desc.dtype, self.vector_size)

        print("edges: ", edges)

        # To understand how many components we need to create, all map ranges
        # throughout memlet paths must match exactly. We thus create a
        # dictionary of unique ranges
        mapping: Dict[Tuple[subsets.Range],
                      List[gr.MultiConnectorEdge[mm.Memlet]]] = defaultdict(
                          list)
        ranges = {}
        for edge in edges:
            mpath = state.memlet_path(edge)
            ranges[edge] = _collect_map_ranges(state, mpath)
            mapping[tuple(r[1] for r in ranges[edge])].append(edge)

        print("ranges: ", ranges)
        print("mapping: ", mapping)

        # Collect all edges with the same memory access pattern
        components_to_create: Dict[
            Tuple[symbolic.SymbolicType],
            List[gr.MultiConnectorEdge[mm.Memlet]]] = defaultdict(list)
        for edges_with_same_range in mapping.values():
            for edge in edges_with_same_range:
                # Get memlet path and innermost edge
                mpath = state.memlet_path(edge)
                innermost_edge = copy.deepcopy(mpath[-1] if self.expr_index ==
                                               0 else mpath[0])

                # Store memlets of the same access in the same component
                expr = _canonicalize_memlet(innermost_edge.data, ranges[edge])
                components_to_create[expr].append((innermost_edge, edge))
        components = list(components_to_create.values())

        print("components: ", components)

        # Split out components that have dependencies between them to avoid
        # deadlocks
        if self.expr_index == 0:
            ccs_to_add = []
            for i, component in enumerate(components):
                edges_to_remove = set()
                for cedge in component:
                    if any(
                            nx.has_path(state.nx, o[1].dst, cedge[1].dst)
                            for o in component if o is not cedge):
                        ccs_to_add.append([cedge])
                        edges_to_remove.add(cedge)
                if edges_to_remove:
                    components[i] = [
                        c for c in component if c not in edges_to_remove
                    ]
            components.extend(ccs_to_add)
        # End of split

        # Create new streams of shape 1
        streams = {}
        mpaths = {}
        for edge in edges:

            total_size = edge.data.volume

            print("Total_size = ", total_size)

            if self.expr_index == 0:
                print("Read")
                #  Read
                gearbox_read_volume = total_size  // self.vector_size
                gearbox_write_volume = total_size
            else:
                print("Write")
                # Write
                gearbox_read_volume = total_size
                gearbox_write_volume = total_size // self.vector_size

            # TODO: Check correct usage of both streams
            input_gearbox_name, input_gearbox_newdesc = sdfg.add_stream(
                "gearbox_input",
                gearbox_input_type,
                buffer_size=self.buffer_size,
                storage=dtypes.StorageType.FPGA_Local,
                transient=True,
                find_new_name=True)

            output_gearbox_name, output_gearbox_newdesc = sdfg.add_stream(
                "gearbox_output",
                gearbox_output_type,
                buffer_size=self.buffer_size,
                storage=dtypes.StorageType.FPGA_Local,
                transient=True,
                find_new_name=True)

            print("input_gearbox_name = ", input_gearbox_name)
            print("input_gearbox_newdesc = ", input_gearbox_newdesc)
            print("output_gearbox_name = ", output_gearbox_name)
            print("output_gearbox_newdesc = ", output_gearbox_newdesc)

            # Add Gearbox
            # TODO: different name if multiple gearboxes

            print(streams)

            read_to_gearbox = state.add_read(input_gearbox_name)
            write_from_gearbox = state.add_write(output_gearbox_name)

            # gearbox_name = sdfg._find_new_name("gearbox")
            print("Gearbox", sdfg.arrays[arrname].total_size / self.vector_size)

            gearbox = Gearbox(sdfg.arrays[arrname].total_size /
                              self.vector_size,
                              name="gearbox")
            state.add_node(gearbox)

            state.add_memlet_path(read_to_gearbox,
                                  gearbox,
                                  dst_conn="from_memory",
                                  memlet=dace.Memlet(
                                      input_gearbox_name + "[0]",
                                      volume=gearbox_read_volume))
            state.add_memlet_path(gearbox,
                                  write_from_gearbox,
                                  src_conn="to_kernel",
                                  memlet=dace.Memlet(
                                      output_gearbox_name + "[0]",
                                      volume=gearbox_write_volume))

            if self.expr_index == 0:
                # Read
                streams[edge] = input_gearbox_name
                name = output_gearbox_name
                newdesc = output_gearbox_newdesc
            else:
                # Write
                streams[edge] = output_gearbox_name
                name = input_gearbox_name
                newdesc = input_gearbox_newdesc

            mpath = state.memlet_path(edge)
            mpaths[edge] = mpath

            print("mpath = ", mpath)

            # Replace memlets in path with stream access
            for e in mpath:
                print("e = ", e)
                e.data = mm.Memlet(data=name,
                                   subset='0',
                                   other_subset=e.data.other_subset)
                if isinstance(e.src, nodes.NestedSDFG):
                    e.data.dynamic = True
                    _streamify_recursive(e.src, e.src_conn, newdesc)
                if isinstance(e.dst, nodes.NestedSDFG):
                    e.data.dynamic = True
                    _streamify_recursive(e.dst, e.dst_conn, newdesc)

            # Replace access node and memlet tree with one access
            if self.expr_index == 0:
                replacement = state.add_read(output_gearbox_name)
                state.remove_edge(edge)
                state.add_edge(replacement, edge.src_conn, edge.dst,
                               edge.dst_conn, edge.data)
            else:
                replacement = state.add_write(input_gearbox_name)
                state.remove_edge(edge)
                state.add_edge(edge.src, edge.src_conn, replacement,
                               edge.dst_conn, edge.data)

        # Make read/write components
        ionodes = []
        for component in components:
            # Pick the first edge as the edge to make the component from
            innermost_edge, outermost_edge = component[0]
            mpath = mpaths[outermost_edge]
            mapname = streams[outermost_edge]
            innermost_edge.data.other_subset = None

            # Get edge data and streams
            if self.expr_index == 0:
                opname = 'read'
                path = [e.dst for e in mpath[:-1]]
                rmemlets = [(dnode, '__inp', innermost_edge.data)]
                wmemlets = []
                for i, (_, edge) in enumerate(component):
                    name = streams[edge]
                    ionode = state.add_write(name)
                    ionodes.append(ionode)
                    wmemlets.append((ionode, '__out%d' % i,
                                     mm.Memlet(data=input_gearbox_name,
                                               subset='0')))
                code = '\n'.join('__out%d = __inp' % i
                                 for i in range(len(component)))
            else:
                # More than one input stream might mean a data race, so we only
                # address the first one in the tasklet code
                if len(component) > 1:
                    warnings.warn(
                        f'More than one input found for the same index for {dnode.data}'
                    )
                opname = 'write'
                path = [state.entry_node(e.src) for e in reversed(mpath[1:])]
                wmemlets = [(dnode, '__out', innermost_edge.data)]
                rmemlets = []
                for i, (_, edge) in enumerate(component):
                    name = streams[edge]
                    ionode = state.add_read(name)
                    ionodes.append(ionode)
                    rmemlets.append(
                        (ionode, '__inp%d' % i, mm.Memlet(data=name,
                                                          subset='0')))
                code = '__out = __inp0'

            # Vectorize the the global array
            print("Vectorize")
            arrname = str(self.access(sdfg))
            print(arrname)
            print(sdfg.arrays)
            dtype = sdfg.arrays[arrname].dtype

            print(dtype)

            print(dtype)
            if not isinstance(dtype, dtypes.vector):
                sdfg.arrays[arrname].dtype = dtypes.vector(
                    dtype, self.vector_size)
                new_shape = list(sdfg.arrays[arrname].shape)
                contigidx = sdfg.arrays[arrname].strides.index(1)
                new_shape[contigidx] /= self.vector_size
                try:
                    new_shape[contigidx] = int(new_shape[contigidx])
                except TypeError:
                    pass
                sdfg.arrays[arrname].shape = new_shape

                print("Strides:", sdfg.arrays[arrname].strides)

                new_strides: List = list(sdfg.arrays[arrname].strides)
                new_strides.reverse()
                print(new_strides)

                # Divides the stride by vector size
                # divider = 1

                for i in range(len(new_strides)):
                    if i == 0:
                        continue
                    new_strides[i] = int(new_strides[i] // self.vector_size)
                    # divider *= self.vector_size

                new_strides.reverse()

                print(new_strides)
                sdfg.arrays[arrname].strides = new_strides

            print("States: ", sdfg.states())

            # TODO States always sorted?

            for e in sdfg.states()[-1].edges():
                print(e)
                print(e.src)
                print(e.dst)
                print(e.data)
                print(str(e.src) == arrname)
                if (str(e.src) == arrname):
                    print("Change data")
                    print(e.data.subset)
                    new_subset = list(e.data.subset)

                    print(new_subset)

                    new_subset.reverse()

                    i, j, k = new_subset[0]

                    new_subset[0] = (i, j // self.vector_size, k)

                    new_subset.reverse()
                    print(new_subset)

                    print(subsets.Range(new_subset))

                    e.data = mm.Memlet(data=str(e.src),
                                       subset=subsets.Range(new_subset))
                    print(e.data)

            print(sdfg.states()[-1].edges())

            # Create map structure for read/write component
            maps = []
            for entry in path:
                map: nodes.Map = entry.map

                print(map.range)

                for p, r in zip(map.params, map.range):
                    print("p = ", p)
                    print("r = ", r)

                print("Schedule = ", map.schedule)

                ranges = [(p, (r[0], r[1], r[2]))
                          for p, r in zip(map.params, map.range)]

                print(ranges)
                print(ranges[-1][1][1])
                print(ranges[-1][1][1] // self.vector_size)


                # TODO: Not always last dimension ?
                ranges[-1] = (ranges[-1][0],
                              (ranges[-1][1][0],
                               ranges[-1][1][1] // self.vector_size,
                               ranges[-1][1][2]))

                print(ranges)

                print({m[1] for m in rmemlets})

                maps.append(
                    state.add_map(f'__s{opname}_{mapname}', ranges,
                                  map.schedule))
            tasklet = state.add_tasklet(
                f'{opname}_{mapname}',
                {m[1]
                 for m in rmemlets},
                {m[1]
                 for m in wmemlets},
                code,
            )
            for node, cname, memlet in rmemlets:
                state.add_memlet_path(node,
                                      *(me for me, _ in maps),
                                      tasklet,
                                      dst_conn=cname,
                                      memlet=memlet)
            for node, cname, memlet in wmemlets:
                state.add_memlet_path(tasklet,
                                      *(mx for _, mx in reversed(maps)),
                                      node,
                                      src_conn=cname,
                                      memlet=memlet)

        # sdfg.expand_library_nodes()
        print("Done")
        return ionodes