# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from ordered_set import OrderedSet

from dace import dtypes
from dace.sdfg import nodes, SDFG
from dace.sdfg.state import SDFGState

from typing import Dict, List, Optional


class SchedulePhase():
    """``GPU_Device`` at a host level, ``Sequential`` below one; a host map keeps its body host-level.

    With no host maps this is the plain rule -- the top-level maps are the kernels, and everything
    inside one of them is sequential device code::

        library node -> GPU
        map -> GPU
            library node -> Seq
            map -> Seq

    A host map is the exception the rule needs for ICON's shape: an ``nblks`` map exists to LAUNCH
    the ``nproma``/``nlev`` work under it, so it stays on the host and its body is still a host
    level, which makes the maps inside it the kernels.
    """

    def apply(self,
              sdfg: SDFG,
              sdfg_scope_dict: Optional[Dict] = None,
              verbose: bool = False,
              host_map_entries: Optional[OrderedSet] = None) -> None:
        self.verbose = verbose
        self.host_map_entries = host_map_entries if host_map_entries is not None else OrderedSet()
        self.assign_schedules(sdfg)

    def assign_schedules(self, sdfg: SDFG, host_level: bool = True, via_host_map: bool = False) -> None:

        def walk(state: SDFGState, scope_children: Dict[Optional[nodes.Node], List[nodes.Node]],
                 entry: Optional[nodes.MapEntry], host_level: bool, via_host_map: bool) -> None:
            for node in scope_children.get(entry, ()):
                if isinstance(node, nodes.MapEntry):
                    is_kernel = host_level and node not in self.host_map_entries
                    node.map.schedule = (dtypes.ScheduleType.GPU_Device
                                         if is_kernel else dtypes.ScheduleType.Sequential)
                    if self.verbose:
                        print(f"Phase1: set map {node} to {'GPU' if is_kernel else 'sequential'} schedule")
                    # A host map does not consume the host level: what it launches is still host code.
                    is_host_map = host_level and not is_kernel
                    walk(state, scope_children, node, is_host_map, via_host_map or is_host_map)

                elif isinstance(node, nodes.LibraryNode):
                    # Sequential specifically: Default can be lowered to CUDA in the wrong places.
                    node.schedule = (dtypes.ScheduleType.GPU_Device if host_level else dtypes.ScheduleType.Sequential)
                    if self.verbose:
                        print(f"Phase1: set libnode {node} to {'GPU' if host_level else 'sequential'} schedule")

                elif isinstance(node, nodes.NestedSDFG):
                    # A nested SDFG is a host level of its own ONLY when a host map put us here: that
                    # map runs on the host to launch what is inside, so the maps inside are the
                    # kernels. Everywhere else its contents stay sequential -- the copy analysis
                    # decides placement for the state around it, and promoting a map it has not
                    # reasoned about turns a nested gather into a kernel reading host memory
                    # (npbench spmv: 'Illegal copy! (from x to indirection)').
                    self.assign_schedules(node.sdfg, host_level and via_host_map, via_host_map)

        for state in sdfg.states():
            walk(state, state.scope_children(), None, host_level, via_host_map)
