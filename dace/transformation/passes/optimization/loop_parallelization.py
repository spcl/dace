# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Any, Dict, List, Set, Tuple, Type, Union

import sympy
from dace import data, dtypes, properties, symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes, propagation
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import LoopRegion, SDFGState
from dace.subsets import Range, SubsetUnion
from dace.transformation import pass_pipeline as ppl, helpers as xfh
from dace.transformation.passes.analysis import loop_analysis


#TODO: rename to just parallelize loops
@properties.make_properties
class ParallelizeDoacrossLoops(ppl.Pass):
    """
    TODO
    """

    CATEGORY: str = 'Parallelization'

    use_doacross = properties.Property(dtype=bool, default=False,
                                       desc='Parallelize loops with sequential dependencies using doacross parallelism')

    def __init__(self):
        self._non_analyzable_loops = set()
        super().__init__()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.CFG

    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        return {loop_analysis.LoopCarryDependencyAnalysis}

    def _parallelize_loop(self, loop: LoopRegion,
                          doacross_deps: Dict[Memlet, Tuple[Memlet, List[sympy.Basic]]]) -> None:
        body: SDFGState = loop.nodes()[0]
        loop_start = loop_analysis.get_init_assignment(loop)
        loop_end = loop_analysis.get_loop_end(loop)
        loop_stride = loop_analysis.get_loop_stride(loop)
        itvar = symbolic.symbol(loop.loop_variable)

        in_deps_edges: Set[MultiConnectorEdge[Memlet]] = set()
        for idep in doacross_deps.keys():
            if idep._edge:
                in_deps_edges.add(idep._edge)
        out_deps_edges: Set[MultiConnectorEdge[Memlet]] = set()
        for odep, _ in doacross_deps.values():
            if odep._edge:
                out_deps_edges.add(odep._edge)
        handled_in_deps = set()
        handled_out_deps = set()

        source_nodes = body.source_nodes()
        sink_nodes = body.sink_nodes()

        # Check intermediate notes
        intermediate_nodes = []
        for node in body.nodes():
            if isinstance(node, nodes.AccessNode) and body.in_degree(node) > 0 and node not in sink_nodes:
                # Scalars written without WCR must be thread-local
                if (isinstance(node.desc(loop.sdfg), data.Scalar) and
                    any(e.data.wcr is None for e in body.in_edges(node))):
                    continue
                # Arrays written with subsets that do not depend on the loop variable must be thread-local
                map_dependency = False
                for e in body.in_edges(node):
                    subset = e.data.get_dst_subset(e, body)
                    if any(str(s) == loop.loop_variable for s in subset.free_symbols):
                        map_dependency = True
                        break
                if not map_dependency:
                    continue
                intermediate_nodes.append(node)

        if doacross_deps:
            map_entry, map_exit = body.add_map(loop.name + '_doacross',
                                               [[loop.loop_variable, Range([(loop_start, loop_end, loop_stride)])]],
                                               schedule=dtypes.ScheduleType.CPU_Multicore_Doacross)
            map_entry.map.omp_schedule = dtypes.OMPScheduleType.Static
            map_entry.map.omp_chunk_size = 1
        else:
            map_entry, map_exit = body.add_map(loop.name + '_parallel',
                                               [[loop.loop_variable, Range([(loop_start, loop_end, loop_stride)])]],
                                               schedule=dtypes.ScheduleType.Default)

        # If the map uses symbols from data containers, instantiate reads
        containers_to_read = map_entry.free_symbols & loop.sdfg.arrays.keys()
        for rd in containers_to_read:
            anode = body.add_read(rd)
            body.add_memlet_path(anode, map_entry, dst_conn=rd, memlet=Memlet(rd))

        # Direct edges among source and sink access nodes must pass through a tasklet. We first gather them and handle
        # them later.
        direct_edges: Set[MultiConnectorEdge[Memlet]] = set()
        for n1 in source_nodes:
            if not isinstance(n1, nodes.AccessNode):
                continue
            for n2 in sink_nodes:
                if not isinstance(n2, nodes.AccessNode):
                    continue
                for e in body.edges_between(n1, n2):
                    e.data.try_initialize(loop.sdfg, body, e)
                    direct_edges.add(e)
                    body.remove_edge(e)

        # Reroute all memlets through the entry and exit nodes
        for n in source_nodes:
            if isinstance(n, nodes.AccessNode):
                for e in body.out_edges(n):
                    # Fix memlet to contain outer data as subset
                    new_memlet = propagation.align_memlet(body, e, dst=False)

                    body.remove_edge(e)
                    body.add_edge_pair(map_entry, e.dst, n, new_memlet, internal_connector=e.dst_conn)
                    if e in in_deps_edges:
                        leaves = body.memlet_tree(e).leaves()
                        for leaf in leaves:
                            can_skip = False
                            subs = leaf.data.dst_subset or leaf.data.subset
                            # Check if this is not actually a carry dependency but rather depends on i for the iteration
                            # i, in which case it gets skipped.
                            for rng in subs.ranges:
                                if itvar in rng[0].free_symbols or itvar in rng[1].free_symbols:
                                    if rng[0] == rng[1] and rng[0] == itvar:
                                        can_skip = True
                                    else:
                                        can_skip = False
                                        break
                            if not can_skip:
                                leaf.data.schedule = dtypes.MemletScheduleType.Doacross_Sink
                                leaf.data.doacross_dependency_offset = doacross_deps[e.data][1]
                        handled_in_deps.add(e)
            else:
                body.add_nedge(map_entry, n, Memlet())
        for n in sink_nodes:
            if isinstance(n, nodes.AccessNode):
                for e in body.in_edges(n):
                    # Fix memlet to contain outer data as subset
                    new_memlet = propagation.align_memlet(body, e, dst=True)

                    body.remove_edge(e)
                    body.add_edge_pair(map_exit, e.src, n, new_memlet, internal_connector=e.src_conn)
                    if e in out_deps_edges:
                        leaves = body.memlet_tree(e).leaves()
                        for leaf in leaves:
                            leaf.data.schedule = dtypes.MemletScheduleType.Doacross_Source
                        handled_out_deps.add(e)
            else:
                body.add_nedge(n, map_exit, Memlet())
        intermediate_sinks = {}
        for n in intermediate_nodes:
            if isinstance(loop.sdfg.arrays[n.data], data.View):
                continue
            if n.data in intermediate_sinks:
                sink = intermediate_sinks[n.data]
            else:
                sink = body.add_access(n.data)
                intermediate_sinks[n.data] = sink
            xfh.make_map_internal_write_external(loop.sdfg, body, map_exit, n, sink)

        # Here we handle the direct edges among source and sink access nodes.
        for e in direct_edges:
            src = e.src.data
            dst = e.dst.data
            if e.data.subset.num_elements() == 1:
                t = body.add_tasklet(f"{n1}_{n2}", {'__inp'}, {'__out'}, "__out =  __inp")
                src_conn, dst_conn = '__out', '__inp'
            else:
                desc = loop.sdfg.arrays[src]
                tname, _ = loop.sdfg.add_transient('tmp',
                                                   e.data.src_subset.size(),
                                                   desc.dtype,
                                                   desc.storage,
                                                   find_new_name=True)
                t = body.add_access(tname)
                src_conn, dst_conn = None, None
            body.add_memlet_path(n1,
                                 map_entry,
                                 t,
                                 memlet=Memlet(data=src, subset=e.data.src_subset),
                                 dst_conn=dst_conn)
            body.add_memlet_path(t,
                                 map_exit,
                                 n2,
                                 memlet=Memlet(data=dst,
                                                      subset=e.data.dst_subset,
                                                      wcr=e.data.wcr,
                                                      wcr_nonatomic=e.data.wcr_nonatomic),
                                 src_conn=src_conn)

        if not source_nodes and not sink_nodes:
            body.add_nedge(map_entry, map_exit, Memlet())

        # If not all input and output doacross dependencies were handled yet, make sure they are marked here.
        for in_dep_edge in in_deps_edges:
            if in_dep_edge not in handled_in_deps:
                for leaf in body.memlet_tree(in_dep_edge).leaves():
                    can_skip = False
                    subs = leaf.data.dst_subset or leaf.data.subset
                    # Check if this is not actually a carry dependency but rather depends on i for the iteration i, in
                    # which case it gets skipped.
                    for rng in subs.ranges:
                        if itvar in rng[0].free_symbols or itvar in rng[1].free_symbols:
                            if rng[0] == rng[1] and rng[0] == itvar:
                                can_skip = True
                            else:
                                can_skip = False
                                break
                    if not can_skip:
                        leaf.data.schedule = dtypes.MemletScheduleType.Doacross_Sink
                        leaf.data.doacross_dependency_offset = doacross_deps[in_dep_edge.data][1]
                handled_in_deps.add(in_dep_edge)
        for out_dep_edge in out_deps_edges:
            if out_dep_edge not in handled_out_deps:
                for leaf in body.memlet_tree(out_dep_edge).leaves():
                    leaf.data.schedule = dtypes.MemletScheduleType.Doacross_Source
                handled_out_deps.add(out_dep_edge)

        # Ensure internal map schedules are adjusted.
        for n in body.nodes():
            if n != map_entry and isinstance(n, nodes.MapEntry):
                n.map.schedule = dtypes.ScheduleType.Default

        # Get rid of the loop and nest the loop body into the loop parent's graph.
        loop.parent_graph.add_node(body)
        for iedge in loop.parent_graph.in_edges(loop):
            isedge_data = iedge.data
            loop.parent_graph.add_edge(iedge.src, body, isedge_data)
            loop.parent_graph.remove_edge(iedge)
        for oedge in loop.parent_graph.out_edges(loop):
            isedge_data = oedge.data
            loop.parent_graph.add_edge(body, oedge.dst, isedge_data)
            loop.parent_graph.remove_edge(oedge)
        loop.parent_graph.remove_node(loop)

        # Finally, make sure there is only one dependency resolving edge (i.e, doacross source). If there are more than
        # one, set their schedules to not generate a sync call (deferred), and instead synchronize at the end of the map
        # scope.
        if len(handled_out_deps) > 1:
            for oe in handled_out_deps:
                for leaf in body.memlet_tree(oe).leaves():
                    if leaf.data.schedule == dtypes.MemletScheduleType.Doacross_Source:
                        leaf.data.schedule = dtypes.MemletScheduleType.Doacross_Source_Deferred
            map_entry.map.omp_doacross_multi_source = True

        loop.sdfg.reset_cfg_list()

    def apply_pass(self, top_sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Any:
        """
        TODO
        """
        results = {}

        loop_deps_dict = pipeline_results[loop_analysis.LoopCarryDependencyAnalysis.__name__]

        for cfg in top_sdfg.all_control_flow_regions(recursive=True):
            if isinstance(cfg, LoopRegion):
                # We can currently only parallelize a loop with a single state as the loop body.
                if len(cfg.nodes()) > 1 or not isinstance(cfg.nodes()[0], SDFGState):
                    continue

                loop: LoopRegion = cfg
                loop_var = symbolic.symbol(loop.loop_variable)

                start_expr = loop_analysis.get_init_assignment(loop)
                end_expr = loop_analysis.get_loop_end(loop)
                stride_expr = loop_analysis.get_loop_stride(loop)
                for expr in (start_expr, end_expr, stride_expr):
                    if symbolic.contains_sympy_functions(expr):
                        continue

                sdfg = loop.sdfg
                sdfg_id = sdfg.cfg_id
                loop_deps: Dict[Memlet, Set[Memlet]] = loop_deps_dict[sdfg_id][loop]

                if not loop_deps:
                    # No dependencies, this loop can be parallelized as a doall loop.
                    self._parallelize_loop(loop, dict())
                elif self.use_doacross:
                    # Check that no dependencies can not be parallelized via do-across, and that there is at least one
                    # such parallelizable dependency
                    can_parallelize = True
                    doacross_deps: Dict[Memlet, Tuple[Memlet, List[int]]] = dict()
                    for read, writes in loop_deps.items():
                        do_acrossable = False
                        violating_dependency = False
                        read_deps: List[Tuple[Memlet, List[int]]] = []
                        for write in writes:
                            write_subset = write.dst_subset or write.subset
                            if isinstance(write_subset, SubsetUnion):
                                if len(write_subset.subset_list) == 1:
                                    write_subset = write_subset.subset_list[0]
                                else:
                                    # TODO(later): We want a way to handle this too.
                                    can_parallelize = False
                                    break
                            read_subset = read.src_subset or read.subset
                            if read_subset.dims() != write_subset.dims():
                                continue
                            subsets_work_fine = True
                            dep_offsets = []
                            for i in range(read_subset.dims()):
                                if read_subset[i] != write_subset[i]:
                                    # TODO(later): This is currently limiting to where the dependency is not more than
                                    # one element and not strided. This could be expanded later, but is a bit more
                                    # complex.
                                    if read_subset[i][2] != 1 or write_subset[i][2] != 1:
                                        subsets_work_fine = False
                                        break
                                    if not any(loop_var in x.free_symbols for x in write_subset[i]):
                                        subsets_work_fine = False
                                        break
                                    if not any(loop_var in x.free_symbols for x in read_subset[i]):
                                        subsets_work_fine = False
                                        break
                                    dep_start: sympy.Basic = read_subset[i][0] - write_subset[i][0]
                                    dep_end: sympy.Basic = read_subset[i][1] - write_subset[i][1]
                                    if len(dep_start.free_symbols) > 0 or len(dep_end.free_symbols) > 0:
                                        subsets_work_fine = False
                                        break
                                    if dep_end == 0 and not dep_start == 0:
                                        dep_end -= read_subset[i][2]
                                    # TODO(later): This is currently limiting to exactly one dependency, but it may be
                                    # interesting to handle multiple. In that case, the guarantee that needs to be
                                    # fulfilled is just: dep_start > dep_end
                                    if dep_start != dep_end:
                                        subsets_work_fine = False
                                        break
                                    dep_offsets.append(loop_var + dep_start)
                            if not subsets_work_fine:
                                violating_dependency = True
                                break
                            read_deps.append([write, dep_offsets])
                            do_acrossable = True
                        if not do_acrossable or violating_dependency or len(read_deps) == 0:
                            can_parallelize = False
                            break

                        resolving_write = None
                        if len(read_deps) > 1:
                            # TODO: get the last one of the writes to determine which one is the resolving write.
                            raise NotImplementedError()
                        else:
                            resolving_write = read_deps[0]

                        if resolving_write == None:
                            can_parallelize = False
                            break
                        doacross_deps[read] = resolving_write
                    if can_parallelize and doacross_deps:
                        self._parallelize_loop(loop, doacross_deps)

        return results
