# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shrink a transient that only ever lives inside one map body down to the box that body touches."""
import re
from typing import Any, Dict, List, Optional, Tuple

from dace import SDFG, SDFGState, data, dtypes, properties, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg import graph as gr
from dace.sdfg.scope import is_devicelevel_gpu
from dace.memlet import Memlet
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.canonicalize.prune_unreferenced_transients import code_text

#: Storages whose buffer is a plain allocation this pass may resize.
RESIZABLE_STORAGE = (dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap, dtypes.StorageType.Register)


def named_in_text(sdfg: SDFG, name: str) -> bool:
    """Whether ``name`` appears as a word in code a structural walk cannot see.

    :param sdfg: SDFG to read (its own blocks only).
    :param name: Descriptor name.
    :returns: ``True`` when a tasklet body, interstate edge or loop header spells the name.
    """
    word = re.compile(rf'\b{re.escape(name)}\b')
    texts = [code_text(sdfg.init_code), code_text(sdfg.exit_code)]
    for state in sdfg.states():
        texts += [code_text(n.code) for n in state.nodes() if isinstance(n, nodes.Tasklet)]
    for edge in sdfg.all_interstate_edges():
        texts.append(code_text(edge.data.condition))
        texts += [code_text(a) for a in edge.data.assignments.values()]
    return any(word.search(text) for text in texts)


def map_local_accesses(sdfg: SDFG, name: str) -> Optional[Tuple[SDFGState, List[gr.MultiConnectorEdge[Memlet]]]]:
    """The state and edges of ``name`` when every access to it sits strictly inside ONE map body.

    :param sdfg: SDFG owning the descriptor.
    :param name: Descriptor name.
    :returns: ``(state, edges)``, or ``None`` when the accesses are not confined to one map body.
    """
    found: Optional[Tuple[SDFGState, List[gr.MultiConnectorEdge[Memlet]]]] = None
    for state in sdfg.states():
        edges = [e for e in state.edges() if e.data.data == name]
        nodes_here = [n for n in state.data_nodes() if n.data == name]
        if not edges and not nodes_here:
            continue
        if found is not None:
            return None
        sdict = state.scope_dict()
        scopes = {sdict[n] for n in nodes_here}
        scopes.update(sdict[e.src] for e in edges)
        scopes.update(sdict[e.dst] for e in edges)
        if len(scopes) != 1 or not isinstance(next(iter(scopes)), nodes.MapEntry):
            return None
        found = (state, edges)
    return found


def uniform_subset(edges: List[gr.MultiConnectorEdge[Memlet]]) -> Optional[subsets.Range]:
    """The one subset every edge in ``edges`` reads or writes, or ``None`` when they disagree.

    :param edges: Edges naming one descriptor.
    :returns: The shared subset, or ``None``.
    """
    shared: Optional[subsets.Range] = None
    for edge in edges:
        memlet = edge.data
        if memlet.other_subset is not None or memlet.dynamic or memlet.wcr is not None:
            return None
        if not isinstance(memlet.subset, subsets.Range):
            return None
        if shared is None:
            shared = memlet.subset
        elif memlet.subset != shared:
            return None
    return shared


def device_level_accesses(sdfg: SDFG, state: SDFGState, edges: List[gr.MultiConnectorEdge[Memlet]]) -> bool:
    """Whether every edge in ``edges`` runs in device code, under a ``GPU_Device`` map.

    A ``GPU_Global`` transient confined to one map body is only resizable there: shrunk under a
    kernel it becomes thread-private, while under a host map it would still be one device buffer
    that the host loop's iterations share.

    :param sdfg: SDFG owning ``state``.
    :param state: The state holding the edges.
    :param edges: Edges naming the descriptor.
    :returns: ``True`` when every edge is device-level.
    """
    return all(is_devicelevel_gpu(sdfg, state, edge.dst) for edge in edges)


def shrinks_the_buffer(desc: data.Array, size: Tuple[Any, ...]) -> bool:
    """Whether a descriptor of extent ``size`` is provably smaller than ``desc``.

    :param desc: The descriptor as it stands.
    :param size: The candidate extent.
    :returns: ``True`` when the candidate is provably the smaller buffer.
    """
    smaller = desc.total_size
    for dim in size:
        smaller = smaller / dim
    return symbolic.simplify(smaller) != 1 and bool(symbolic.equal(desc.total_size, 1) is not True)


@properties.make_properties
@transformation.explicit_cf_compatible
class ShrinkMapLocalTransients(ppl.Pass):
    """Resize a map-body-local transient to the single box its accesses name.

    A pass that internalizes a computation -- an inline of a nested SDFG, a fusion that pulls a
    producer into its consumer -- can leave a FULL-EXTENT transient whose every access sits inside
    one map body and names one box, e.g. ``rsq[i, j]`` under ``map[i, j, k]``. The descriptor is
    then scope-local, so codegen allocates the whole extent ONCE PER ITERATION, and the scope
    default storage inside a ``CPU_Multicore`` map is ``Register``: an ``N*N`` stack array per
    iteration, which segfaults the moment ``N`` leaves the stack (measured on examinimd, where the
    canonicalized form died in the OpenMP body while the plain lowering ran).

    Only the buffer shrinks. The accesses stay private to the iteration that makes them -- each
    one writes the box and reads it back within its own iteration, which is what makes rewriting
    the subset to the origin an identity on the values.
    """

    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Descriptors | ppl.Modifies.Memlets | ppl.Modifies.Nodes))

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Shrink every map-body-local transient that names one box.

        :param sdfg: SDFG to rewrite in place.
        :param pipeline_results: Unused; the pass resolves its own facts.
        :returns: The number of descriptors resized, or ``None`` when none were.
        """
        shrunk = 0
        for sd in sdfg.all_sdfgs_recursive():
            for name, desc in list(sd.arrays.items()):
                if not self.is_candidate(sd, name, desc):
                    continue
                accesses = map_local_accesses(sd, name)
                if accesses is None:
                    continue
                on_device = desc.storage == dtypes.StorageType.GPU_Global
                if on_device and not device_level_accesses(sd, *accesses):
                    continue
                shared = uniform_subset(accesses[1])
                if shared is None:
                    continue
                size = tuple(shared.size())
                if len(size) != len(desc.shape) or not shrinks_the_buffer(desc, size):
                    continue
                if named_in_text(sd, name):
                    continue
                desc.set_shape(size)
                if on_device:
                    # Thread-private by construction: every access sits in one map body under a
                    # kernel and names one iteration's box. Left in GPU_Global it would be one
                    # buffer shared by every thread.
                    desc.storage = dtypes.StorageType.Register
                for edge in accesses[1]:
                    edge.data.subset = subsets.Range([(0, dim - 1, 1) for dim in size])
                    edge.data.volume = edge.data.subset.num_elements()
                shrunk += 1
        return shrunk or None

    def is_candidate(self, sdfg: SDFG, name: str, desc: data.Data) -> bool:
        """Whether ``name`` is a plain resizable transient array of ``sdfg``.

        :param sdfg: SDFG owning the descriptor.
        :param name: Descriptor name.
        :param desc: The descriptor.
        :returns: ``True`` when the descriptor is one this pass may resize.
        """
        if type(desc) is not data.Array or not desc.transient:
            return False
        if name in sdfg.constants_prop:
            return False
        if desc.lifetime != dtypes.AllocationLifetime.Scope:
            return False
        return desc.storage in RESIZABLE_STORAGE or desc.storage == dtypes.StorageType.GPU_Global
