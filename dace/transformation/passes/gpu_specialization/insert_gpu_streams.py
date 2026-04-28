# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict, Set, Type, Union

import dace
from dace import SDFG, dtypes, properties
from dace.sdfg import is_devicelevel_gpu
from dace.sdfg.nodes import AccessNode, MapExit, Node
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import NaiveGPUStreamScheduler
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (add_gpu_stream_connector,
                                                                               get_gpu_stream_array_name,
                                                                               is_gpu_relevant_node,
                                                                               is_inside_gpu_device_kernel)


@properties.make_properties
@transformation.explicit_cf_compatible
class InsertGPUStreams(ppl.Pass):
    """Insert the GPU stream array into the top-level SDFG and propagate it to nested SDFGs that need it.

    Every intermediate SDFG along the path to a user gets the array and a matching nested-SDFG
    in-connector, so subsequent passes can rely on its presence everywhere it's needed.
    """

    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        return {NaiveGPUStreamScheduler}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]):
        """Create the GPU stream array at the top level and wire it through every nested SDFG that needs it.

        All relevant SDFGs end up sharing a consistent reference to the same stream array, so
        subsequent pipeline passes can rely on its presence without redefining it.
        """
        # Extract stream array name and number of streams to allocate
        stream_array_name = get_gpu_stream_array_name()
        stream_assignments: Dict[Node, Union[int, str]] = pipeline_results['NaiveGPUStreamScheduler']
        num_assigned_streams = max(stream_assignments.values(), default=0) + 1

        # Add the GPU stream array at the top level. If a re-run of the pass
        # finds the root array already in place, just propagate it deeper.
        if stream_array_name not in sdfg.arrays:
            self._add_stream_array(sdfg, stream_array_name, num_assigned_streams, transient=True)

        for child_sdfg in self.find_child_sdfgs_requiring_gpu_stream(sdfg):
            # Skip if a higher-level call already inserted the array.
            if stream_array_name in child_sdfg.arrays:
                continue
            self._propagate_stream_array_up(child_sdfg, stream_array_name, num_assigned_streams)

        return {}

    @staticmethod
    def _add_stream_array(target_sdfg: SDFG, stream_name: str, num_streams: int, *, transient: bool) -> None:
        """Add the reserved ``gpu_streams`` descriptor to ``target_sdfg``.

        Passes ``_internal_use=True`` so the reservation guard in
        ``SDFG.add_datadesc`` lets this pass through (the pipeline owns
        the name)."""
        desc = dace.data.Array(dtype=dace.dtypes.gpuStream_t,
                               shape=(num_streams, ),
                               transient=transient,
                               storage=dace.dtypes.StorageType.Register)
        target_sdfg.add_datadesc(stream_name, desc, _internal_use=True)

    @classmethod
    def _propagate_stream_array_up(cls, child_sdfg: SDFG, stream_name: str, num_streams: int) -> None:
        """Add ``stream_name`` to ``child_sdfg`` and every parent SDFG up to
        (and including) the first ancestor that already has it, wiring the
        NestedSDFG-node stream connector at each level so the array flows
        from outer to inner via the standard connector pattern."""
        cls._add_stream_array(child_sdfg, stream_name, num_streams, transient=False)

        # Every chain edge carries the explicit ``[0:num_streams]`` slice so
        # downstream codegen can index by stream id without inferring the
        # bound from the descriptor — the implicit-subset form would force
        # the codegen to look up the array shape on every consumer.
        slice_str = f"{stream_name}[0:{num_streams}]"

        # Climb until a parent already has the array; add+wire at every step.
        cur = child_sdfg
        while stream_name not in cur.parent_sdfg.arrays:
            cls._add_stream_array(cur.parent_sdfg, stream_name, num_streams, transient=False)
            _wire_stream_into_parent(cur, stream_name, dace.Memlet(slice_str))
            cur = cur.parent_sdfg

        # Final wire from the first parent that already had it.
        _wire_stream_into_parent(cur, stream_name, dace.Memlet(slice_str))

    def find_child_sdfgs_requiring_gpu_stream(self, sdfg: SDFG) -> Set[SDFG]:
        """Identify all child SDFGs that need the GPU stream array.

        A child SDFG needs the array when it (host-)launches GPU kernels or
        issues stream-bound runtime calls — either through a ``GPU_Device``
        ``MapEntry`` / ``MapExit`` (kernel launch) or through a
        ``CopyLibraryNode`` / ``MemsetLibraryNode`` whose expansion emits
        ``cudaMemcpyAsync`` / ``cudaMemsetAsync``.

        Host- vs. device-level rule: a NestedSDFG inside a ``Sequential`` /
        CPU map with GPU work inside still runs on the host and gets the
        array threaded in. A NestedSDFG inside a ``GPU_Device`` map executes
        as device code (``__device__`` / ``DACE_DFI``); it cannot issue
        ``cudaMemcpyAsync`` / ``cudaLaunchKernel``, so threading streams
        into it would bind a host-only resource into device code and
        produce ill-formed CUDA. We skip those.
        """
        requiring_gpu_stream = set()
        for child_sdfg in sdfg.all_sdfgs_recursive():

            # Skip the root SDFG itself
            if child_sdfg is sdfg:
                continue

            if is_inside_gpu_device_kernel(child_sdfg):
                continue

            for state in child_sdfg.states():
                for node in state.nodes():
                    # MapExit also counts (it's the symmetric end of the
                    # kernel launch); ``is_gpu_relevant_node`` only matches
                    # MapEntry, so check it here.
                    if isinstance(node, MapExit) and node.map.schedule == dtypes.ScheduleType.GPU_Device:
                        requiring_gpu_stream.add(child_sdfg)
                        break
                    # Skip GPU AccessNodes that are inside a device scope
                    # (codegen handles them through register/local paths,
                    # no host-side stream needed).
                    if (isinstance(node, AccessNode) and node.desc(state).storage == dtypes.StorageType.GPU_Global
                            and is_devicelevel_gpu(state.sdfg, state, node)):
                        continue
                    if is_gpu_relevant_node(node, child_sdfg, state):
                        requiring_gpu_stream.add(child_sdfg)
                        break

                # Stop scanning this SDFG once a reason is found
                if child_sdfg in requiring_gpu_stream:
                    break

        return requiring_gpu_stream


def _wire_stream_into_parent(level: SDFG, stream_name: str, memlet: dace.Memlet) -> None:
    """Wire ``stream_name`` from ``level.parent_sdfg`` into ``level``'s
    NestedSDFG node: stream connector + AccessNode + edge.

    Module-level helper called twice in the propagation loop above and
    once after — same call signature, no closure capture, easier to read
    in isolation.
    """
    nsdfg_node = level.parent_nsdfg_node
    parent_state = level.parent
    add_gpu_stream_connector(nsdfg_node, stream_name, single_stream=False)
    src = parent_state.add_access(stream_name)
    parent_state.add_edge(src, None, nsdfg_node, stream_name, memlet)
