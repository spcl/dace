# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``PromoteGPUScalarsToArrays`` -- replace GPU-incompatible ``Scalar``
descriptors with length-1 ``Array`` descriptors. Runs after storage/schedule
inference (depends on ``InferDefaultSchedulesAndStorages``).

Two rules: (1) a ``Scalar`` with ``GPU_Global``/``GPU_Shared`` storage is
widened to length-1 keeping its storage; (2) a ``Scalar`` written by a GPU
map's ``MapExit`` (kernel output) is widened and forced to ``GPU_Global``.
Bare-identifier references to a promoted name are subscripted ``name[0]`` in
interstate/loop/branch code slots and ``symbol_mapping`` values; memlets are
left intact since a ``Scalar`` access already carries subset ``[0]``.
"""
import re
from typing import Any, Dict, Optional, Callable

from dace import data, dtypes, properties
from dace.sdfg import SDFG, infer_types, nodes, SDFGState
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import written_by_gpu_map_exit

PatternApplier: type = Callable[[str], str]


def invalidate_array_connectors(sdfg: SDFG):
    """Reset NestedSDFG connectors whose inner descriptor is an ``Array`` to
    ``typeclass(None)`` so a follow-up ``infer_connector_types`` re-derives
    them as pointer-typed.

    A connector typed at construction time as a scalar dtype against an
    ``Array`` inner descriptor produces a wrapper signature ``T name`` that the
    body indexes ``name[0]`` (compile error). Common cause: cuBLAS expansion's
    ``gpu_streams`` connector.
    """
    uninferred = dtypes.typeclass(None)
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for node in state.nodes():
                if not isinstance(node, nodes.NestedSDFG):
                    continue
                for cname in list(node.in_connectors):
                    if cname in node.sdfg.arrays and isinstance(node.sdfg.arrays[cname], data.Array):
                        node.in_connectors[cname] = uninferred
                for cname in list(node.out_connectors):
                    if cname in node.sdfg.arrays and isinstance(node.sdfg.arrays[cname], data.Array):
                        node.out_connectors[cname] = uninferred


@properties.make_properties
@transformation.explicit_cf_compatible
class InferDefaultSchedulesAndStorages(ppl.Pass):
    """Pipeline-shaped wrapper around
    :func:`dace.sdfg.infer_types.set_default_schedule_and_storage_types`.

    Exists so the call can participate in a ``Pipeline`` with a real
    ``depends_on`` edge: ``PromoteGPUScalarsToArrays`` relies on every
    descriptor having a final, non-default storage decision.
    """

    def modifies(self) -> ppl.Modifies:
        # Storage lives on descriptors, schedule on ``Map`` nodes.
        return ppl.Modifies.Descriptors | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    @staticmethod
    def _schedules_and_storages(sdfg: SDFG) -> Dict[Any, Any]:
        """Snapshot every slot ``set_default_schedule_and_storage_types`` can write.

        Those are descriptor storages (keyed by ``(SDFG, name)``) and the schedules of scope
        entry nodes / library nodes. The inference function reports nothing about what it
        resolved, so the pass diffs a before/after snapshot. Exit nodes are skipped: their
        schedule proxies the same :class:`~dace.sdfg.nodes.Map` as the matching entry node.
        """
        snapshot: Dict[Any, Any] = {}
        for nsdfg in sdfg.all_sdfgs_recursive():
            for name, desc in nsdfg.arrays.items():
                snapshot[(nsdfg, name)] = desc.storage
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, (nodes.EntryNode, nodes.LibraryNode)):
                snapshot[node] = node.schedule
        return snapshot

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Resolve every ``Default`` schedule and storage in the SDFG hierarchy.

        :returns: Number of schedule/storage slots whose value changed, or ``None`` if none did.
        """
        before = self._schedules_and_storages(sdfg)
        infer_types.set_default_schedule_and_storage_types(sdfg, None)
        after = self._schedules_and_storages(sdfg)

        changed = sum(1 for key, value in after.items() if before.get(key) != value)
        return changed or None


@properties.make_properties
@transformation.explicit_cf_compatible
class PromoteGPUScalarsToArrays(ppl.Pass):
    """Replace GPU-incompatible ``Scalar`` descriptors with length-1 Arrays."""

    # Register-storage scalars are thread-local; widening would force
    # per-thread ``cudaMalloc`` inside the kernel body.
    _RULE2_EXEMPT_STORAGES = frozenset({dtypes.StorageType.Register})

    non_transient_only = properties.Property(dtype=bool,
                                             default=True,
                                             desc="Rule 2 only promotes non-transient kernel-output scalars. "
                                             "A transient scalar written by a GPU map exit stays a Scalar -- the "
                                             "host never observes the value, so it can live in registers / "
                                             "per-thread stack. Disable to promote every kernel-output scalar.")

    def depends_on(self):
        return {InferDefaultSchedulesAndStorages}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # Adding new GPU-storage Scalars (e.g. via library expansion) re-arms
        # the pass; harmless when nothing matches.
        return bool(modified & (ppl.Modifies.Descriptors | ppl.Modifies.Nodes))

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Promote every GPU-incompatible scalar across the SDFG hierarchy.

        :returns: Number of scalars promoted, or ``None`` if nothing changed.
        """
        promoted = 0
        # Top-down so a parent's promotion is visible when we visit the child's
        # matching descriptor (children inherit the parent's choice).
        for nsdfg in list(sdfg.all_sdfgs_recursive()):
            for name in list(nsdfg.arrays):
                if not self._needs_promotion(nsdfg, name):
                    continue
                self._promote_one(nsdfg, name)
                promoted += 1

        invalidate_array_connectors(sdfg)

        return promoted if promoted > 0 else None

    def _needs_promotion(self, sdfg: SDFG, name: str) -> bool:
        desc = sdfg.arrays[name]
        if not isinstance(desc, data.Scalar):
            return False

        # Rule 1: GPU storage is incompatible with Scalar.
        if desc.storage in (dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared):
            return True

        # Rule 2: kernel output -- written by a GPU map's ``MapExit``.
        if desc.storage in self._RULE2_EXEMPT_STORAGES:
            return False
        if self.non_transient_only and desc.transient:
            return False
        return written_by_gpu_map_exit(sdfg, name)

    def _promote_one(self, sdfg: SDFG, name: str):
        """Replace a Scalar descriptor with a length-1 Array and propagate the
        change, recursing into nested SDFGs that re-declare the same name as a
        Scalar.
        """
        scalar_desc: data.Scalar = sdfg.arrays[name]

        # Rule 2 promotes Default / CPU-side scalars to GPU_Global because
        # the kernel write needs real device memory; rule 1 keeps the
        # pre-existing GPU storage.
        target_storage = scalar_desc.storage
        if target_storage not in (dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared):
            target_storage = dtypes.StorageType.GPU_Global

        array_desc = data.Array(
            dtype=scalar_desc.dtype,
            shape=(1, ),
            transient=scalar_desc.transient,
            storage=target_storage,
            location=scalar_desc.location,
            strides=(1, ),
            lifetime=scalar_desc.lifetime,
            allow_conflicts=scalar_desc.allow_conflicts,
            debuginfo=scalar_desc.debuginfo,
        )

        sdfg.remove_data(name, validate=False)
        sdfg.add_datadesc(name, array_desc)

        compiled_pattern = re.compile(rf'(?<![\w.])({re.escape(name)})(?!\s*\[)\b')
        pattern = lambda s: compiled_pattern.sub(rf'\1[0]', s)

        self._rewrite_state_machine(sdfg, pattern)
        self._rewrite_states(sdfg=sdfg, name=name, pattern=pattern)

    @staticmethod
    def _rewrite_codeblock(pattern: PatternApplier, codeblock: properties.CodeBlock) -> properties.CodeBlock:
        codeblock_str = codeblock.as_string
        new_codeblock_str = pattern(codeblock_str)
        return properties.CodeBlock(new_codeblock_str, codeblock.language)

    def _rewrite_state_machine(self, sdfg: SDFG, pattern: PatternApplier) -> None:
        """Rewrite bare-identifier references on every state-machine code slot:
        interstate edges (assignments + condition), ``LoopRegion`` and
        ``ConditionalBlock`` CodeBlocks. ``InterstateEdge.assignments`` and
        ``.condition`` are class-level properties so they always exist.
        """
        for cfg in sdfg.all_control_flow_regions():
            for edge in cfg.edges():
                ise = edge.data
                if ise is None:
                    continue
                for k, v in list(ise.assignments.items()):
                    if not isinstance(v, str):
                        continue
                    new_v = pattern(v)
                    if new_v != v:
                        ise.assignments[k] = new_v
                ise.condition = self._rewrite_codeblock(pattern, ise.condition)

        # ``ConditionalBlock`` and ``LoopRegion`` carry the only CodeBlock slots
        # a state-machine walk reaches; other blocks embed no user expressions.
        for block in sdfg.all_control_flow_blocks(recursive=True):
            if isinstance(block, ConditionalBlock):
                for branch in block.branches:
                    if branch[0] is not None:
                        branch[0] = self._rewrite_codeblock(pattern, branch[0])
            elif isinstance(block, LoopRegion):
                # init/update are optional -- a ``while`` LoopRegion has only the condition.
                if block.update_statement is not None:
                    block.update_statement = self._rewrite_codeblock(pattern, block.update_statement)
                if block.init_statement is not None:
                    block.init_statement = self._rewrite_codeblock(pattern, block.init_statement)
                if block.loop_condition is not None:
                    block.loop_condition = self._rewrite_codeblock(pattern, block.loop_condition)

    def _rewrite_states(self, sdfg: SDFG, name: str, pattern: PatternApplier) -> None:
        """Apply the promotion in all states."""
        for state in sdfg.states():
            self._rewrite_state(state=state, name=name, pattern=pattern)

    def _rewrite_state(self, state: SDFGState, name: str, pattern: PatternApplier) -> None:
        """Push the rewrite into NestedSDFGs reached from ``state``.

        Memlets are not touched -- a ``Scalar`` access always carries subset
        ``[0]``, identical to a length-1 array's. The two slots that DO need
        attention are (a) the inner descriptor when the NSDFG re-declares it
        as a ``Scalar``, and (b) ``symbol_mapping`` values that bare-reference
        the promoted name (frontend symbol-promotion threads scalars into the
        nested scope this way).
        """
        for node in state.nodes():
            if not isinstance(node, nodes.NestedSDFG):
                continue

            for k, v in list(node.symbol_mapping.items()):
                v_str = v if isinstance(v, str) else str(v)
                new_v = pattern(v_str)
                if new_v != v_str:
                    node.symbol_mapping[k] = new_v

            handled_inner_names: set[str] = set()  # If data is referenced as input and output.
            for iedge in state.in_edges(node):
                if iedge.data.is_empty():
                    continue
                inner_name = iedge.dst_conn
                if iedge.data.data == name and isinstance(node.sdfg.arrays[inner_name], data.Scalar):
                    assert inner_name not in handled_inner_names  # Can only appear once.
                    self._promote_one(node.sdfg, inner_name)
                    handled_inner_names.add(inner_name)

            for oedge in state.out_edges(node):
                if oedge.data.is_empty():
                    continue
                inner_name = oedge.src_conn
                if oedge.data.data == name and inner_name not in handled_inner_names and isinstance(
                        node.sdfg.arrays[inner_name], data.Scalar):
                    self._promote_one(node.sdfg, inner_name)
