# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``PromoteGPUScalarsToArrays`` -- replace GPU-incompatible ``Scalar``
descriptors with length-1 ``Array`` descriptors (after storage/schedule
inference; depends on ``InferDefaultSchedulesAndStorages``).

Three rules: (1) a ``Scalar`` with ``GPU_Global``/``GPU_Shared`` storage keeps
its storage and is widened to length-1; (2) a ``Scalar`` that is a
**kernel output** -- written by a GPU map's ``MapExit`` -- is widened and
forced to ``GPU_Global``; (3) a host ``Scalar`` that is an endpoint of a
direct copy edge whose other endpoint is GPU-resident is widened but **keeps
its host storage** -- the GPU pipeline lifts that edge to a ``CopyLibraryNode``
whose ``cudaMemcpyAsync`` tasklet needs a pointer, and a bare ``Scalar`` source
would codegen as ``const T* = scalar`` (missing ``&``). ``Register`` storage is
exempt from rule 2 (thread-local stack), and the ``non_transient_only`` knob
further restricts rule 2 to non-transient scalars (kernel-local transients then
stay as registers).

Bare-identifier references to a promoted name are subscripted ``name[0]``
in interstate assignments, interstate conditions, ``LoopRegion``
init/update/condition, ``ConditionalBlock`` branch conditions, and
``NestedSDFG.symbol_mapping`` values. Memlets are left intact -- a
``Scalar`` access already has subset ``[0]``, matching the length-1 array's
subset. Nested SDFGs are recursed via the connector that carries the
promoted descriptor (inner name may differ from outer).
"""
import re
from typing import Any, Dict, Optional, Callable

from dace import data, dtypes, properties
from dace.sdfg import SDFG, infer_types, nodes, SDFGState
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation

PatternApplier: type = Callable[[str], str]


def invalidate_array_connectors(sdfg: SDFG):
    """Reset NestedSDFG connectors whose inner descriptor is an ``Array`` so a follow-up
    ``infer_connector_types`` re-derives them as pointer-typed.

    A connector typed at construction time as a scalar dtype against an
    ``Array`` inner descriptor produces a wrapper signature ``T name`` that the
    body indexes ``name[0]`` (compile error); resetting to ``typeclass(None)``
    forces re-inference. Common cause: cuBLAS expansion's ``gpu_streams``
    connector.

    :param sdfg: SDFG whose nested-SDFG connectors are reset in place.
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

    The function itself is the actual implementation -- this class exists
    so the call can participate in a ``Pipeline`` with a real
    ``depends_on`` edge from later passes. ``PromoteGPUScalarsToArrays``
    in particular relies on every descriptor having a final, non-default
    storage decision, which is exactly what this pass establishes.
    """

    def modifies(self) -> ppl.Modifies:
        # Storage and schedule attributes live on descriptors and on
        # ``Map`` instances respectively; both are reachable through
        # ``Modifies.Descriptors | Modifies.Nodes``.
        return ppl.Modifies.Descriptors | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        infer_types.set_default_schedule_and_storage_types(sdfg, None)
        return None


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

        :param sdfg: Root SDFG to promote scalars in (modified in place).
        :param pipeline_results: Results of prior pipeline passes (unused).
        :returns: Number of scalars promoted, or ``None`` if nothing changed.
        """
        promoted = 0
        # Top-down so a parent's promotion is visible when we visit the
        # child's matching descriptor (children inherit the parent's choice
        # -- see ``_promote_one`` for the recursion into nested SDFGs).
        for nsdfg in list(sdfg.all_sdfgs_recursive()):
            for name in list(nsdfg.arrays):
                reason = self._needs_promotion(nsdfg, name)
                if reason is None:
                    continue
                # Rule 3 (host copy endpoint) keeps the scalar's own storage --
                # a host->device copy source must stay on the host. Rules 1/2
                # force GPU_Global for kernel-side scalars.
                self._promote_one(nsdfg, name, force_gpu_global=(reason != 'copy_endpoint'))
                promoted += 1

        # Reset NestedSDFG connectors whose inner descriptor became an Array
        # so ``infer_connector_types`` re-derives them as pointer-typed.
        invalidate_array_connectors(sdfg)

        return promoted if promoted > 0 else None

    def _needs_promotion(self, sdfg: SDFG, name: str) -> Optional[str]:
        """Classify why ``name`` must be widened, or ``None`` if it must not.

        :returns: ``'gpu'`` for a GPU-storage / kernel-output scalar (Rules 1
            and 2 -- widened and forced to ``GPU_Global``), ``'copy_endpoint'``
            for a host scalar that is an endpoint of a cross-boundary copy edge
            (Rule 3 -- widened but keeps its host storage), or ``None``.
        """
        desc = sdfg.arrays[name]
        if not isinstance(desc, data.Scalar):
            return None

        # Rule 1: GPU storage is incompatible with Scalar.
        if desc.storage in (dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared):
            return 'gpu'

        # Rule 2: scalar is a kernel output -- written by a GPU map's ``MapExit``.
        # Transient kernel outputs are skipped under the default knob (the
        # host can never observe the value, so it can live in registers).
        if desc.storage not in self._RULE2_EXEMPT_STORAGES and not (self.non_transient_only and desc.transient):
            for state in sdfg.states():
                for node in state.nodes():
                    if not (isinstance(node, nodes.AccessNode) and node.data == name):
                        continue
                    for in_edge in state.in_edges(node):
                        src = in_edge.src
                        if not isinstance(src, nodes.ExitNode):
                            continue
                        entry = state.entry_node(src)
                        if entry is not None and entry.map.schedule in dtypes.GPU_SCHEDULES:
                            return 'gpu'

        # Rule 3: host scalar that is an endpoint of a direct copy edge whose
        # other endpoint is GPU-resident. The GPU pipeline lifts that edge to a
        # ``CopyLibraryNode`` whose ``cudaMemcpyAsync`` tasklet types the scalar
        # side as a pointer connector; a bare ``Scalar`` source then codegens as
        # ``const T* = scalar`` (missing ``&``). Widening it to a length-1 host
        # array gives the copy a real pointer. See
        # ``dace.transformation.passes.insert_explicit_copies``.
        if self._is_host_gpu_copy_endpoint(sdfg, name):
            return 'copy_endpoint'

        return None

    @staticmethod
    def _is_host_gpu_copy_endpoint(sdfg: SDFG, name: str) -> bool:
        """True if ``name`` is a scalar on one side of a direct ``AccessNode ->
        AccessNode`` copy edge whose other side is a ``GPU_RESIDENT_STORAGES``
        array (a host<->device scalar copy that the GPU pipeline will lift)."""
        for state in sdfg.states():
            for edge in state.edges():
                src, dst = edge.src, edge.dst
                if not (isinstance(src, nodes.AccessNode) and isinstance(dst, nodes.AccessNode)):
                    continue
                if edge.data.is_empty() or edge.data.wcr is not None:
                    continue
                if src.data == name:
                    other = sdfg.arrays[dst.data]
                elif dst.data == name:
                    other = sdfg.arrays[src.data]
                else:
                    continue
                if other.storage in dtypes.GPU_RESIDENT_STORAGES:
                    return True
        return False

    def _promote_one(self, sdfg: SDFG, name: str, force_gpu_global: bool = True):
        """Replace a Scalar descriptor with a length-1 Array and propagate the change.

        Rewrites memlets referencing it and recurses into nested SDFGs that
        re-declare the same name as a Scalar.

        :param sdfg: SDFG owning the descriptor (modified in place).
        :param name: Name of the Scalar descriptor to promote.
        :param force_gpu_global: When ``True`` (Rules 1/2), a non-GPU-storage
            scalar is moved to ``GPU_Global`` because a kernel touches it.
            When ``False`` (Rule 3), the scalar's storage is preserved -- a
            host copy source/destination must stay on the host.
        """
        scalar_desc: data.Scalar = sdfg.arrays[name]

        # Rule 2 promotes Default / CPU-side scalars to GPU_Global because
        # the kernel write needs real device memory; rule 1 keeps the
        # pre-existing GPU storage. Rule 3 (force_gpu_global=False) keeps the
        # host storage of a copy endpoint.
        target_storage = scalar_desc.storage
        if force_gpu_global and target_storage not in (dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared):
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

        # Keep ``constants_prop`` consistent with the widened descriptor. A
        # compile-time constant scalar (e.g. the autodiff gradient seed ``one``
        # = 1.0) is emitted by the frame codegen straight from
        # ``constants_prop`` as ``constexpr T name = v;`` -- ignoring the
        # ``arrays`` descriptor. If such a constant is a copy source it is fed
        # into a pointer-typed connector (``const T* _cpy_in = name;``), which
        # does not compile. Re-storing it as an ``Array`` descriptor with a
        # length-1 value makes the codegen emit ``constexpr T name[1] = {v};``,
        # which decays to a pointer for the copy.
        if name in sdfg.constants_prop:
            import numpy as np
            _, cstval = sdfg.constants_prop[name]
            sdfg.constants_prop[name] = (array_desc, np.reshape(np.asarray(cstval), (1, )))

        # The rewrite patten we need to apply.
        compiled_pattern = re.compile(rf'(?<![\w.])({re.escape(name)})(?!\s*\[)\b')
        pattern = lambda s: compiled_pattern.sub(rf'\1[0]', s)

        self._rewrite_state_machine(sdfg, pattern)
        self._rewrite_states(sdfg=sdfg, name=name, pattern=pattern, force_gpu_global=force_gpu_global)

    @staticmethod
    def _rewrite_codeblock(pattern: PatternApplier, codeblock: properties.CodeBlock) -> properties.CodeBlock:
        codeblock_str = codeblock.as_string
        new_codeblock_str = pattern(codeblock_str)
        return properties.CodeBlock(new_codeblock_str, codeblock.language)

    def _rewrite_state_machine(self, sdfg: SDFG, pattern: PatternApplier) -> None:
        """Rewrite bare-identifier references on every state-machine code
        slot: interstate edges (assignments + condition), ``LoopRegion`` and
        ``ConditionalBlock`` CodeBlocks. ``InterstateEdge.assignments`` and
        ``.condition`` are class-level properties so they always exist.

        :param sdfg: SDFG whose state-machine slots are rewritten.
        :param pattern: The rewrite pattern.
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

        # ``ConditionalBlock`` and ``LoopRegion`` carry the only CodeBlock
        # slots reached from a state-machine walk -- a ``ControlFlowBlock``
        # doesn't otherwise embed user expressions. Both subclass
        # ``ControlFlowBlock`` so they appear via ``all_control_flow_blocks``.
        for block in sdfg.all_control_flow_blocks(recursive=True):
            if isinstance(block, ConditionalBlock):
                for i in range(len(block._branches)):
                    cond, branch_body = block._branches[i]
                    if cond is not None:
                        # ``_branches`` entries are (condition, body) tuples -- rebuild the tuple
                        # rather than assigning into it.
                        block._branches[i] = (self._rewrite_codeblock(pattern, cond), branch_body)
            elif isinstance(block, LoopRegion):
                # ``init_statement`` / ``update_statement`` are optional --
                # a bare ``while`` LoopRegion has only the condition.
                if block.update_statement is not None:
                    block.update_statement = self._rewrite_codeblock(pattern, block.update_statement)
                if block.init_statement is not None:
                    block.init_statement = self._rewrite_codeblock(pattern, block.init_statement)
                if block.loop_condition is not None:
                    block.loop_condition = self._rewrite_codeblock(pattern, block.loop_condition)

    def _rewrite_states(self, sdfg: SDFG, name: str, pattern: PatternApplier, force_gpu_global: bool = True) -> None:
        """Applies the promotion, `Scalar` to 'One-Element-Array', in all states.

        :param sdfg: The SDFG on which we operate.
        :param name: The name of the data that was promoted.
        :param pattern: The rewrite pattern that we need to apply.
        :param force_gpu_global: Propagated to nested-SDFG promotions.
        """
        for state in sdfg.states():
            self._rewrite_state(state=state, name=name, pattern=pattern, force_gpu_global=force_gpu_global)

    def _rewrite_state(self, state: SDFGState, name: str, pattern: PatternApplier,
                       force_gpu_global: bool = True) -> None:
        """Push the rewrite into NestedSDFGs reached from ``state``.

        Memlets are not touched -- a ``Scalar`` access always carries subset
        ``[0]``, identical to a length-1 array's. The two slots that DO need
        attention are (a) the inner descriptor when the NSDFG re-declares it
        as a ``Scalar``, and (b) ``symbol_mapping`` values that bare-reference
        the promoted name (frontend symbol-promotion threads scalars into the
        nested scope this way).

        :param state: The state in which to apply the promotion.
        :param name: The name of the data that was promoted.
        :param pattern: The pattern to apply.
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
                    self._promote_one(node.sdfg, inner_name, force_gpu_global=force_gpu_global)
                    handled_inner_names.add(inner_name)

            for oedge in state.out_edges(node):
                if oedge.data.is_empty():
                    continue
                inner_name = oedge.src_conn
                if oedge.data.data == name and inner_name not in handled_inner_names and isinstance(
                        node.sdfg.arrays[inner_name], data.Scalar):
                    self._promote_one(node.sdfg, inner_name, force_gpu_global=force_gpu_global)
