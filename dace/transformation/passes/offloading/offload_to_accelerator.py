# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Decide what runs on the accelerator, and place the host/device copies that follow from it.

The pass reads the SDFG once into a small control-flow IR (:class:`OffloadingIRNode`), records per
block which arrays are wanted on the CPU and which on the GPU, propagates those sets along the
graph, and only then materializes copies -- so a copy is emitted where the location actually
changes rather than around every kernel.
"""
from copy import deepcopy
from typing import Any, Optional

from dace.ordered import OrderedSet

from dace import dtypes, properties, data, Memlet, subsets, symbolic
from dace.config import Config
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.sdfg import nodes, SDFG
from dace.sdfg.state import (SDFGState, ConditionalBlock, ControlFlowRegion, LoopRegion, ReturnBlock, ContinueBlock,
                             BreakBlock, ControlFlowBlock, AbstractControlFlowRegion)
from dace.sdfg.scope import is_devicelevel_gpu
from dace.sdfg.utils import require_structured_control_flow
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible
from dace.transformation.dataflow import TrivialMapElimination
from dace.transformation.passes import FuseMaps
from dace.transformation.passes.length_one_array_scalar_conversion import (ConvertLengthOneArraysToScalars,
                                                                           ConvertScalarsToLengthOneArrays)
from dace.transformation.passes.offloading.offloading_helpers import (
    enclosing_kernel, get_data_used_by_incoming_access_nodes, get_data_used_by_outgoing_access_nodes,
    get_new_map_identifiers, get_schedule, has_GPU_schedule, is_array, is_array_stored_on_GPU, is_length1_array,
    is_scalar, is_stream, is_view, link_early_returns, register_kernel_local_transients, remove_empty_return_entries,
    separate_early_returns, traverse_IR, traverse_same_level)
from dace.transformation.passes.simplification.control_flow_raising import ControlFlowRaising
from dace.transformation.passes.offloading.taskloop import is_device_wide_libnode, sdfg_only_launches, taskloop_maps
from dace.transformation.passes.offloading.host_maps import HostMapSpec, host_maps, maps_pinned_by_host_loops

from dace.transformation.passes.offloading.offloading_ir_node import OffloadingIRNode


def in_sequential_specialization_arm(block) -> bool:
    """Whether ``block`` is, or is inside, the sequential arm of a guarded specialization.

    Canonicalization emits a loop it can only parallelize under a runtime condition as both arms of
    one ConditionalBlock -- a Map for the parallel case, the original LoopRegion for the fallback,
    which it marks ``pinned_sequential`` (``specialize_loop_under_condition`` and the scatter
    guard's dispatcher). That fallback IS host code, and it owns the copies that bring its inputs
    down; lifting its tasklets into size-1 kernels, or hoisting its copies to the program entry,
    would make every execution pay what only the rarely-taken arm needs.

    Recognized by that marker on a loop at the top level of a branch, not by "a loop under a
    conditional": amg_setup's loops under its source-level ``if n > 100`` are ordinary host code
    around device work, and wavefront skew pins loops that are no fallback at all.
    """
    current = block
    while current is not None:
        # A branch is the arm when its own top level holds the pinned fallback loop; the arm's copy
        # states sit beside that loop, not inside it.
        if isinstance(current, ControlFlowRegion) and isinstance(current.parent_graph, ConditionalBlock) and any(
                isinstance(b, LoopRegion) and b.pinned_sequential for b in current.nodes()):
            return True
        current = current.parent_graph
    return False


def in_a_loop(block) -> bool:
    """Whether a ``LoopRegion`` of ``block``'s own SDFG encloses it, i.e. whether it may run more than once."""
    region = block.parent_graph
    while region is not None and not isinstance(region, SDFG):
        if isinstance(region, LoopRegion):
            return True
        region = region.parent_graph
    return False


@properties.make_properties
@explicit_cf_compatible
class OffloadToAccelerator(ppl.Pass):
    """Move the work an accelerator can take to it, and insert the copies that decision implies.

    The only offloader on this branch, and what ``SDFG.apply_gpu_transformations`` runs. Where the
    transformation it replaced copied in and out around each kernel it made, this one propagates the
    wanted location of every array through the control flow first, so an array that stays on the
    device across a whole loop is copied once rather than per iteration, and a host-only branch pays
    for its copies inside that branch.
    """

    CATEGORY: str = 'Offload To Accelerator'

    taskloop_overrides = properties.DictProperty(
        key_type=str,
        value_type=bool,
        default={},
        desc='Map label -> whether that map is a taskloop, deciding it outright. Final in both '
        'directions and consulted before any heuristic: a caller naming a map has looked at the '
        'kernel, and the rules have not. Maps left unnamed are classified as usual.')

    def __init__(self, taskloop_overrides: dict[str, bool] | None = None, host_maps: HostMapSpec = False):
        self.taskloop_overrides = dict(taskloop_overrides) if taskloop_overrides else {}
        self._host_maps = host_maps
        self._host_map_entries = OrderedSet()
        self._host_pinned = OrderedSet()
        self._host_only_loops: dict = {}
        #: Containers :meth:`place_single_sided_data` duplicated at entry: valid on both sides all run.
        self._read_only_duplicates: OrderedSet = OrderedSet()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> OrderedSet[type[ppl.Pass]]:
        return OrderedSet([ControlFlowRaising])

    #def report(self, pass_retval: Any) -> Optional[str]:
    #    """
    #    Returns a user-readable string report based on the results of this pass.
    #
    #    :param pass_retval: The return value from applying this pass.
    #    :return: A string with the user-readable report, or None if nothing to report.
    #    """
    #    return None

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> Any | None:
        """
        Applies the pass to the given SDFG.

        :param sdfg: The SDFG to apply the pass to.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: Some object if pass was applied, or None if nothing changed.
        """

        # The copy analysis reads each region as a line of blocks; callers run ControlFlowRaising first (depends_on).
        require_structured_control_flow(sdfg, 'OffloadToAccelerator')

        self.taskloop_heuristics = Config.get_bool('optimizer', 'gpu_taskloop_heuristics')
        self.cache_scopes(sdfg)
        self.hybrid_overlap: dict = {}

        # step 1: set schedule of maps and library nodes -> heuristic only!
        self.find_taskloops(sdfg)
        self.assign_schedules(sdfg)

        # Read before any copy is placed: the staging below renames the accesses this looks at.
        self.no_copy_in_needed = self.overwritten_before_any_read(sdfg)

        self.place_single_sided_data(sdfg)
        self.place_and_copy(sdfg)
        self.offload_host_level_bodies(sdfg)
        self.scalarize_locals_of_removed_trivial_maps(sdfg)
        self.refuse_by_value_scalars_the_device_writes(sdfg)
        register_kernel_local_transients(sdfg)

        return self.device_resident(sdfg) or None

    def device_resident(self, sdfg: SDFG) -> OrderedSet[str]:
        """Every container this pass left in a GPU storage, qualified by the SDFG that holds it.

        This is the pass's result, and a Pipeline reads it as "did anything change": an SDFG with
        nothing on the device came back unoffloaded, which is exactly the ``None`` the Pass contract
        asks for. Names are qualified because a nested SDFG may reuse a name the parent also has.
        """
        placed: OrderedSet[str] = OrderedSet()
        for nested in sdfg.all_sdfgs_recursive():
            for name, desc in nested.arrays.items():
                if desc.storage in GPU_RESIDENT_STORAGES:
                    placed.add(f'{nested.cfg_id}.{name}')
        return placed

    def overwritten_before_any_read(self, sdfg: SDFG) -> OrderedSet[str]:
        """Signature arrays whose value on entry cannot be observed, so staging them down is dead work.

        A container that nothing ever READS -- no access node with an out-edge anywhere, no interstate
        edge naming it -- carries no information into the program. Copying it to the device before a
        kernel overwrites it is a full transfer of data that is then discarded, which for a
        write-only output is the whole array on every call.

        The copy is only dead if the device also writes ALL of it. The copy-out sends the entire
        device buffer back, so a partially written one would deliver whatever the allocation held for
        the rest -- and it is precisely the copy-in that makes those elements round-trip unchanged
        today. A single write whose subset covers the descriptor is therefore required, not merely a
        write.

        :param sdfg: SDFG to analyse.
        :returns: Names for which no host-to-device copy is needed.
        """
        dead: OrderedSet[str] = OrderedSet()
        for name, desc in sdfg.arrays.items():
            if desc.transient or isinstance(desc, (data.View, data.Stream)) or is_scalar(name, sdfg):
                continue
            if self.read_anywhere(sdfg, name) or not self.written_in_full(sdfg, name):
                continue
            dead.add(name)
        return dead

    def read_anywhere(self, sdfg: SDFG, name: str) -> bool:
        """True if anything reads ``name``: an access node with an out-edge, or an interstate edge.

        Interstate reads are invisible to an access-node walk -- a condition or an assignment naming
        the container has no node of its own -- so they are asked for separately, the way
        :meth:`read_outside_a_kernel` asks.
        """
        for nested in sdfg.all_sdfgs_recursive():
            for state in nested.states():
                if any(node.data == name and state.out_degree(node) > 0 for node in state.data_nodes()):
                    return True
            if any(name in edge.data.used_arrays(nested.arrays) for edge in nested.all_interstate_edges()):
                return True
        return False

    def written_in_full(self, sdfg: SDFG, name: str) -> bool:
        """True if one write to ``name`` provably touches every element of its descriptor.

        A covering SUBSET is not enough. An indirect write -- ``A[x[i], y[j]]`` -- carries the whole
        array as its subset because that is where it MIGHT land, while its volume says how many
        elements it actually writes; ``tests/transformations/gpu_transform_test.py``'s
        ``write_subset_dynamic`` covers 20x20 and writes 256 of the 400. Trusting the subset alone
        there drops the copy-in and hands the caller uninitialised memory for the rest, so the
        volume has to agree with the descriptor as well.

        The union of several partial writes could also cover the array, but a single covering write
        is what a map over the full range emits and is the only shape worth trusting here: getting
        this wrong is silent, so the analysis declines rather than reasons.
        """
        desc = sdfg.arrays[name]
        whole = subsets.Range.from_array(desc)
        for state in sdfg.states():
            for node in state.data_nodes():
                if node.data != name:
                    continue
                for edge in state.in_edges(node):
                    memlet = edge.data
                    if memlet.dynamic or memlet.wcr is not None:
                        continue
                    written = memlet.get_dst_subset(edge, state)
                    if written is None or not written.covers(whole):
                        continue
                    if symbolic.equal(memlet.volume, desc.total_size, is_length=False):
                        return True
        return False

    def kernel_local_len1_arrays(self, sdfg: SDFG) -> OrderedSet[str]:
        """Length-1 transients written inside a TRIVIAL SEQUENTIAL map that a kernel encloses.

        That map is a loop of one iteration inside device code, and the array is the stack slot its
        body writes through. Both halves of the guard are load-bearing. A map with a real extent
        needs the array, because the iterations are distinct writes. A trivial map at a HOST level,
        or one carrying a GPU schedule, is not device-local at all: the schedule is what makes the
        write a register rather than a kernel argument, and a scalar handed to a kernel goes by value
        and loses the write -- which is why :meth:`place_and_copy` promoted these to arrays to begin
        with.
        """
        found: OrderedSet[str] = OrderedSet()
        for nested in sdfg.all_sdfgs_recursive():
            for state in nested.states():
                scopes = state.scope_dict()
                for entry in state.nodes():
                    if not isinstance(entry, nodes.MapEntry) or entry.map.schedule in dtypes.GPU_SCHEDULES:
                        continue
                    if not all(begin == end for begin, end, _ in entry.map.range):
                        continue
                    if not enclosing_kernel(scopes, entry):
                        continue
                    for node in state.scope_subgraph(entry).nodes():
                        if not isinstance(node, nodes.AccessNode) or node.data not in nested.arrays:
                            continue
                        desc = nested.arrays[node.data]
                        if (desc.transient and is_length1_array(node.data, nested)
                                and desc.storage not in GPU_RESIDENT_STORAGES):
                            found.add(node.data)
        return found

    def data_written_by_device_code(self, sdfg: SDFG) -> OrderedSet[str]:
        """Every descriptor a GPU-scheduled scope writes whose value has to OUTLIVE that scope.

        A kernel writes across TWO boundaries and the analysis used to see only the first:

        * an access node OUTSIDE the scope, reached through the ``MapExit`` -- what
          :meth:`get_data_used_by_outgoing_access_nodes` reports;
        * an access node INSIDE the scope. A size-1 wrapper that pulls a tasklet and the node it
          writes into the same kernel leaves nothing at that exit, so the first test reports
          nothing and the descriptor is never claimed for the device (polybench durbin).

        Whether an inside write counts is decided by SCOPE, not by storage. Storage is still
        ``Default`` when this runs, so durbin's ``alpha`` and the kernel-local
        ``_wcr_priv_set_sum_out_sum`` are indistinguishable by it; what separates them is that
        ``alpha`` is also accessed outside the kernel that writes it. An access under a DIFFERENT
        kernel counts as outside too -- two launches cannot hand a value to each other in a
        register.
        """
        through_the_exit: OrderedSet[str] = OrderedSet()
        written_inside: OrderedSet[str] = OrderedSet()
        kernels_per_data: dict[str, OrderedSet[Optional[nodes.MapEntry]]] = {}
        for state in sdfg.states():
            scopes = state.scope_dict()
            for node in state.nodes():
                if isinstance(node, (nodes.MapExit, nodes.LibraryNode)) and has_GPU_schedule(node):
                    through_the_exit |= get_data_used_by_outgoing_access_nodes(sdfg,
                                                                               state,
                                                                               node,
                                                                               include_scalars=True,
                                                                               ordering=False,
                                                                               through_copies=False)
                if not isinstance(node, nodes.AccessNode) or node.data not in sdfg.arrays:
                    continue
                kernel = enclosing_kernel(scopes, node)
                kernels_per_data.setdefault(node.data, OrderedSet()).add(kernel)
                if kernel is not None and state.in_degree(node) > 0:
                    written_inside.add(node.data)
        return through_the_exit | OrderedSet(name for name in written_inside if len(kernels_per_data[name]) > 1)

    def refuse_by_value_scalars_the_device_writes(self, sdfg: SDFG) -> None:
        """Raise if a Scalar a kernel writes would reach that kernel BY VALUE.

        ``Scalar.as_arg`` renders a pointer for ``GPU_Global`` and a plain ``double x`` parameter
        for every other storage, so a kernel handed one writes its own stack and the write is
        discarded. Nothing downstream objects -- the launch succeeds and the numbers are wrong
        (polybench durbin) -- so the placement is CHECKED here rather than trusted. A descriptor
        this pass failed to claim has to stop the compile, not reach a user as a result.
        """
        offenders = [
            name for name in self.data_written_by_device_code(sdfg) if isinstance(sdfg.arrays[name], data.Scalar)
            and sdfg.arrays[name].storage is not dtypes.StorageType.GPU_Global
        ]
        if offenders:
            raise ValueError(f'device code writes {offenders}, still Scalars in host storage. A kernel takes those '
                             f'BY VALUE, so the write would be discarded and the result silently wrong: '
                             f'{[(name, sdfg.arrays[name].storage.name) for name in offenders]}')

    def scalarize_locals_of_removed_trivial_maps(self, sdfg: SDFG) -> None:
        """Drop single-iteration maps, then scalarize the kernel locals that lost theirs.

        ``TrivialMapElimination`` declines a GPU schedule itself, so a kernel is never the map that
        goes; what goes is a one-iteration loop inside one. The filter is the DIFFERENCE across the
        elimination rather than the census before it: a map the transformation declines (a dynamic
        map range keeps one parameter alive) leaves its array under a map still, and converting that
        one would hand a kernel a by-value scalar and lose the write.
        """
        before = self.kernel_local_len1_arrays(sdfg)
        if not sdfg.apply_transformations_repeated(TrivialMapElimination, validate=False, validate_all=False):
            return
        self.cache_scopes(sdfg)
        freed = before - self.kernel_local_len1_arrays(sdfg)
        if freed:
            ConvertLengthOneArraysToScalars(recursive=True, filter=freed).apply_pass(sdfg, {})
            self.cache_scopes(sdfg)

    def touches_device_code(self, scopes: dict, node: nodes.Node) -> bool:
        """Whether ``node`` is device code, or the boundary of a scope that is."""
        if isinstance(node, (nodes.MapEntry, nodes.MapExit)):
            return get_schedule(node) in dtypes.GPU_SCHEDULES
        # A nested SDFG holds its own scopes, so answer for it the way that cannot be wrong in the
        # direction that matters: calling it device keeps its containers out of the host staging.
        # Under a pinned map it is host code by decision (see ``maps_pinned_by_host_loops``).
        if isinstance(node, nodes.NestedSDFG):
            return not self.under_pinned_map(scopes, node)
        if isinstance(node, nodes.LibraryNode):
            return has_GPU_schedule(node)
        return enclosing_kernel(scopes, node) is not None

    def in_host_only_loop(self, state: SDFGState) -> bool:
        """Whether the innermost loop around ``state`` holds no device work at all.

        Such a loop is a serial host algorithm, and it stays host code: the arrays it touches are
        copied at its boundary, not lifted element by element into size-1 kernels. srad's region
        sum, a host double loop over ``J`` inside the time loop, launched one kernel and made two
        single-element copies per element of the region.
        """
        region = state.parent_graph
        while region is not None and not isinstance(region, SDFG):
            if isinstance(region, LoopRegion):
                if region not in self._host_only_loops:
                    self._host_only_loops[region] = not any(
                        self.is_device_work(node) for body in region.all_states() for node in body.nodes())
                return self._host_only_loops[region]
            region = region.parent_graph
        return False

    def under_pinned_map(self, scopes: dict[nodes.Node, nodes.EntryNode | None], node: nodes.Node) -> bool:
        """Whether a map pinned to the host by :func:`maps_pinned_by_host_loops` encloses ``node``."""
        scope = scopes[node]
        while scope is not None:
            if scope in self._host_pinned:
                return True
            scope = scopes[scope]
        return False

    def data_sides(self, sdfg: SDFG) -> tuple[OrderedSet[str], OrderedSet[str], OrderedSet[str]]:
        """Which side of the machine touches each top-level container, and what is written.

        Read off the graph rather than off the IR: the per-state analysis answers for one state at
        a time and its hybrid resolution rewrites the answer, so by the time the IR exists a
        container that only host code touches already reads as a device one.
        """
        host: OrderedSet[str] = OrderedSet()
        device: OrderedSet[str] = OrderedSet()
        written: OrderedSet[str] = OrderedSet()
        for state in sdfg.states():
            scopes = state.scope_dict()
            # A fallback arm's host reads are conditional, and the arm already copies what it needs
            # inside itself. Counting them here would answer a rarely-taken read with a copy every
            # execution pays for -- the exact cost the arm-local copies exist to avoid.
            fallback = in_sequential_specialization_arm(state)
            # Host means "touched by code that STAYS on the host". A free tasklet sharing its state
            # with device code does not: the hybrid resolution wraps it in a len-1 map, and a
            # container staged on the strength of that use is then written from inside a kernel.
            hybrid = any(self.touches_device_code(scopes, node) for node in state.nodes())
            for node in state.data_nodes():
                if node.data not in sdfg.arrays:
                    continue
                if state.in_degree(node) > 0:
                    written.add(node.data)
                for edge in state.all_edges(node):
                    other = edge.dst if edge.src is node else edge.src
                    if self.touches_device_code(scopes, other):
                        device.add(node.data)
                    elif not fallback and not hybrid:
                        host.add(node.data)
        return host, device, written

    def stage_on_host(self, sdfg: SDFG, name: str, write_back: bool) -> bool:
        """Give ``name`` a host copy, point every use at it, and copy at the SDFG's boundary.

        Declines rather than stage a written container it cannot write back: the host copy would
        hold the answer and the caller's array would not, which is a wrong number rather than a
        broken graph, so it has to be refused where it can still be seen.
        """
        sinks = sdfg.sink_nodes() if write_back else []
        if write_back and not sinks:
            return False

        existing = OrderedSet(sdfg.all_control_flow_blocks())
        self.create_interstate_copy(sdfg, None, sdfg.start_block, OrderedSet([name]), to_gpu=False)

        rename = {name: self._get_host_name(name)}
        for block in existing:
            if isinstance(block, SDFGState):
                self._insert_copy_names_in_state(block, rename)
            else:
                block.replace_meta_accesses(rename)
        for edge in sdfg.all_interstate_edges():
            edge.data.replace(name, self._get_host_name(name))

        # Every exit needs the write-back, not just one: a sink each side of a branch is two ways out.
        for sink in sinks:
            self.create_interstate_copy(sdfg, sink, None, OrderedSet([name]), to_gpu=True)
        return True

    def place_single_sided_data(self, sdfg: SDFG) -> None:
        """Give a container ONE home before the per-state placement runs.

        A signature array is put on the device wholesale so the caller can hand one down, but that
        is an ABI decision, not a placement: a container only host code touches is then device
        memory the host writes, and the per-state analysis cannot undo it because it sees one state
        at a time and its hybrid resolution answers by moving the whole state onto the device --
        which for npbench nbody's ``PE`` means wrapping a scalar accumulation into one kernel launch
        per iteration. Staging it on the host instead is both correct and free.

        Only the single-sided case is decided here. A container both sides WRITE needs coherence and
        stays with the per-state copies; one both sides only read needs none, so a single copy at
        entry serves every host read for the rest of the run.
        """
        host, device, written = self.data_sides(sdfg)
        for name in list(sdfg.arrays):
            desc = sdfg.arrays[name]
            if isinstance(desc, data.Stream):
                continue
            if desc.transient or desc.storage != dtypes.StorageType.GPU_Global or name not in host:
                continue
            if name in device and name in written:
                continue

            if name in device:
                # Read on both sides and written by neither, so the two copies can never disagree:
                # one copy at entry, and the per-state renaming already points the host reads at it.
                self.create_interstate_copy(sdfg, None, sdfg.start_block, OrderedSet([name]), to_gpu=False)
                self._read_only_duplicates.add(name)
            elif not self.stage_on_host(sdfg, name, write_back=name in written):
                continue

            # The copy states are new blocks; every later step reads scopes from the cache.
            self.cache_scopes(sdfg)

    def place_and_copy(self, sdfg: SDFG) -> None:
        """Place every array across ONE host level and copy accordingly; taskloop bodies come later."""
        # step 2:
        # Names already put through a conversion. A SIGNATURE descriptor is not rewritten in place:
        # ``preserve_abi`` stages a transient beside it and copies, so the array itself is still an
        # array on the next scan and would be requested again for ever (TSVC s332's ``result``).
        attempted: OrderedSet[str] = OrderedSet()
        entries = self.separate_returns_and_refresh_scopes(sdfg)
        self.initialize_device_tables_on_the_device(sdfg)

        for _ in range(3):
            # step 2: copy analysis -> IR stores analysis results
            self.hybrid_states = OrderedSet()
            sdfgIR = self.sdfg_to_IR(sdfg)

            # step 3: resolve hybrid states
            new_maps = OrderedSet()
            if self.hybrid_states:
                for state in self.hybrid_states:
                    new_maps |= self.make_size1_map_wrappers(sdfg, state)

            if new_maps:
                # No validation: host reads of device storage are renamed only by ``eval_IR``, so the
                # graph is invalid until the copies exist (CloudSC's ``pap`` on an interstate edge).
                mapfusion_pass = FuseMaps(
                    strict_dataflow=True,
                    perform_vertical_map_fusion=True,
                    perform_horizontal_map_fusion=True,
                    validate=False,
                )
                mapfusion_pipeline = ppl.Pipeline([mapfusion_pass])
                mapfusion_pipeline.apply_pass(sdfg, {})

            # step 4: assign scalars / len1-arrays correctly
            all_scalars: OrderedSet[str] = OrderedSet(data_name for data_name in sdfg.arrays
                                                      if is_scalar(data_name, sdfg))
            all_len1arrays: OrderedSet[str] = {
                data_name
                for data_name in sdfg.arrays if is_length1_array(data_name, sdfg)
            }
            gpu_written = self.data_written_by_device_code(sdfg)

            to_len1_arrays = (all_scalars & gpu_written) - attempted
            # ``__return`` stays by reference: the caller reads the result back through it.
            to_scalars = {
                name
                for name in (all_len1arrays - gpu_written) - attempted if not name.startswith("__return")
            }
            attempted |= to_len1_arrays | to_scalars
            # What the conversions REWROTE, not what they were asked to: both decline a descriptor
            # they cannot express (a View, an opaque handle, a signature array they may not stage),
            # and a request that is declined every round is a fixed point, not progress. Reading the
            # request instead spins until the retry budget runs out and then raises on a graph that
            # had already settled -- TSVC s332, whose non-transient ``result`` is refused each time.
            rewritten: OrderedSet[str] = OrderedSet()
            if to_len1_arrays:
                rewritten |= ConvertScalarsToLengthOneArrays(
                    recursive=True,
                    preserve_abi=True,
                    filter=to_len1_arrays,
                ).apply_pass(sdfg, {}) or OrderedSet()

            if to_scalars:
                rewritten |= ConvertLengthOneArraysToScalars(
                    recursive=True,
                    preserve_abi=True,
                    filter=to_scalars,
                ).apply_pass(sdfg, {}) or OrderedSet()

            # What the wrappers BUILT, not the states that asked: a partition can be legitimately
            # declined (a lone staging node, or one computing only scalars -- covariance's ``N - 1``),
            # and a request declined every round is a fixed point, not progress.
            if rewritten or new_maps:  # sdfg has been changed
                self.cache_scopes(sdfg)
                continue  # repeat phases 2 - 4

            break  # else IR is correct for current sdfg, go on to next step

        else:
            raise RuntimeError("Offloading did not settle: the copy analysis, the hybrid-state "
                               "resolution and the scalar/len-1 assignment kept changing the graph "
                               "for 3 rounds.")

        # TODO: remove eventually
        def assert_no_scalars(node: OffloadingIRNode):
            scalars = OrderedSet(data_name for data_name in node.gpu_set | node.cpu_set if is_scalar(data_name, sdfg))
            assert not scalars, (f"scalars {scalars} found in {node.debug_name}\n"
                                 f"\tgpu: {node.gpu_set}\n\tcpu: {node.cpu_set}")

        traverse_IR(sdfgIR, assert_no_scalars)

        # step 4: insert copies based on IR
        self.eval_IR(sdfg, sdfgIR)
        remove_empty_return_entries(entries)

    def separate_returns_and_refresh_scopes(self, sdfg: SDFG) -> list[tuple[ControlFlowRegion, SDFGState]]:
        """Give each return its own entry state, so the copy-backs an early return needs run on its path."""
        entries = separate_early_returns(sdfg)
        if entries:
            # The analysis reads scopes from the cache, which predates the entry states.
            self.cache_scopes(sdfg)
        return entries

    def initialize_device_tables_on_the_device(self, sdfg: SDFG) -> None:
        """Fill a table kernels read on the device too, as ONE size-1 kernel per state: no copy ships it there.

        Left alone, the copy analysis ships a host-filled table down mid-run, because the hybrid
        resolution only lifts host code that shares a state with a device use of the same data --
        not a fill like CloudSC's ``imelt[0:5] = 2, 3, 4, 3, -99``, which sits beside unrelated kernels
        and is read by kernels in later states. A table the host reads as well keeps its host fill,
        and the kernel fills a device twin that every device-side state reads instead
        (CloudSC's ``iphase`` and ``zvqx``, which interstate assignments also read).
        """
        regions: dict[SDFGState, OrderedSet[nodes.Tasklet]] = {}
        for name, (state, tasklets, twin_states) in self.device_tables(sdfg).items():
            if twin_states is not None:
                tasklets = self.fill_device_twin(sdfg, state, tasklets, name, twin_states)
            regions.setdefault(state, OrderedSet()).update(tasklets)
        for state, region in regions.items():
            self._wrap_region_in_size1_map(state, region)
        if regions:
            self.cache_scopes(sdfg)

    def device_tables(self, sdfg: SDFG) -> dict[str, tuple[SDFGState, OrderedSet[nodes.Tasklet], Optional[list]]]:
        """Tables to fill on the device: ``name -> (fill state, fill tasklets, device-side states or None)``.

        A table is a transient array longer than one element; a length-1 one becomes a by-value
        scalar and needs no copy at all. Some kernel reads it, and:

        * every write is a top-level tasklet with one output that reads only scalars, which a kernel
          takes by value -- so no array is dragged onto the device with it;
        * those writes sit in ONE state outside every loop, so the launch that replaces the copy runs
          once -- not once per iteration, nor once per state a scattered fill touches.

        Nothing else writes it, a kernel included: the host and device copies must stay equal. The
        third entry is None when nothing on the host reads it -- the fill then simply moves. Otherwise
        it lists the states that only kernels touch it in, which read the twin; a state touching it
        from both sides cannot be split by a rename, so such a table is left alone.
        """
        host_read: OrderedSet[str] = OrderedSet(name for edge in sdfg.all_interstate_edges()
                                                for name in edge.data.used_arrays(sdfg.arrays))
        for region in sdfg.all_control_flow_regions():
            host_read |= OrderedSet(memlet.data for memlet in region.get_meta_read_memlets())
        refused: OrderedSet[str] = OrderedSet()
        fills: dict[str, dict[SDFGState, OrderedSet[nodes.Tasklet]]] = {}
        #: Per table and state, the sides touching it there: True for device code, False for host code.
        sides: dict[str, dict[SDFGState, set[bool]]] = {}
        for state in sdfg.states():
            scopes = self.cached_scopes[state]
            looped = in_a_loop(state)
            for node in state.data_nodes():
                desc = sdfg.arrays.get(node.data)
                if type(desc) is not data.Array or not desc.transient or is_length1_array(node.data, sdfg):
                    continue
                touched = sides.setdefault(node.data, {}).setdefault(state, set())
                if enclosing_kernel(scopes, node):
                    touched.add(True)
                    if any(not isinstance(edge.src, nodes.EntryNode) and not edge.data.is_empty()
                           for edge in state.in_edges(node)):
                        refused.add(node.data)
                    continue
                for edge in state.out_edges(node):
                    if isinstance(edge.dst, nodes.MapEntry) and has_GPU_schedule(edge.dst):
                        touched.add(True)
                    elif not edge.data.is_empty():
                        touched.add(False)
                        host_read.add(node.data)
                for edge in state.in_edges(node):
                    if edge.data.is_empty():
                        continue
                    if not looped and self.fills_from_scalars(sdfg, state, scopes, edge.src):
                        fills.setdefault(node.data, {}).setdefault(state, OrderedSet()).add(edge.src)
                        touched.add(False)
                    else:
                        refused.add(node.data)
        tables = {}
        for name, per_state in fills.items():
            if name in refused or len(per_state) != 1 or not any(True in on for on in sides[name].values()):
                continue
            (state, tasklets), = per_state.items()
            if name not in host_read:
                tables[name] = (state, tasklets, None)
            elif all(len(on) < 2 for on in sides[name].values()):
                tables[name] = (state, tasklets, [other for other, on in sides[name].items() if True in on])
        return tables

    def fill_device_twin(self, sdfg: SDFG, state: SDFGState, tasklets: OrderedSet[nodes.Tasklet], name: str,
                         twin_states: list) -> OrderedSet[nodes.Tasklet]:
        """Clone the fill of ``name`` onto a device twin, point ``twin_states`` at it, return the clones.

        The twin takes the name the copy analysis gives a device copy, so it is exactly that copy --
        born filled instead of copied.
        """
        twin = self._get_gpu_name(name)
        desc = deepcopy(sdfg.arrays[name])
        desc.storage = dtypes.StorageType.GPU_Global
        sdfg.add_datadesc(twin, desc)
        for other in twin_states:
            self._insert_copy_names_in_state(other, {name: twin})
        target = state.add_access(twin)
        clones: OrderedSet[nodes.Tasklet] = OrderedSet()
        for tasklet in tasklets:
            clone = deepcopy(tasklet)
            state.add_node(clone)
            for edge in state.in_edges(tasklet):
                state.add_edge(edge.src, edge.src_conn, clone, edge.dst_conn, deepcopy(edge.data))
            write = state.out_edges(tasklet)[0]
            memlet = deepcopy(write.data)
            memlet.data = twin
            state.add_edge(clone, write.src_conn, target, None, memlet)
            clones.add(clone)
        return clones

    def fills_from_scalars(self, sdfg: SDFG, state: SDFGState, scopes: dict, node: nodes.Node) -> bool:
        """A top-level tasklet with one output whose every input is a scalar, or that has none."""
        return (isinstance(node, nodes.Tasklet) and scopes[node] is None and state.out_degree(node) == 1 and all(
            edge.data.is_empty() or (isinstance(edge.src, nodes.AccessNode) and is_scalar(edge.src.data, sdfg))
            for edge in state.in_edges(node)))

    def offload_host_level_bodies(self, sdfg: SDFG) -> None:
        """Place again inside every nested SDFG that is still host code: each body is its own level.

        Its connector-bound descriptors already carry this level's storage, so the body seeds from
        that and copies only what its own control flow needs. Schedules were assigned tree-wide.

        Two kinds of body qualify, and both fail the same way when skipped -- their interstate edges
        read device arrays from host code. A taskloop body, whatever the config says, because
        ``find_taskloops`` records a map enclosing a device-wide library node unconditionally
        (npbench spmv's ``start = A_indptr[i]``); and one sitting at a state's own top level, which
        no map encloses at all (npbench scattering_self_energies' ``neigh_idx``). The level's own
        analysis only asks such a body where it wants its bound arrays -- it never places within it.
        """
        # Copy insertion added states; the cached scopes predate them.
        self.cache_scopes(sdfg)
        for state in sdfg.states():
            for node in self.host_level_nested_sdfgs(state, None):
                self.stage_device_scalar_bindings(sdfg, state, node)
                self.inherit_binding_storage(sdfg, state, node)
                body = type(self)()
                body.taskloop_heuristics = self.taskloop_heuristics
                body.taskloops = self.taskloops
                # The body places its own level, so it needs the state ``apply_pass`` seeds.
                body.hybrid_overlap = {}
                body.no_copy_in_needed = body.overwritten_before_any_read(node.sdfg)
                body.cache_scopes(node.sdfg)
                body.place_and_copy(node.sdfg)
                body.offload_host_level_bodies(node.sdfg)

    def stage_device_scalar_bindings(self, sdfg: SDFG, state: SDFGState, nsdfg_node: nodes.NestedSDFG) -> None:
        """Materialize on the host a device element bound to a scalar connector host code reads.

        The lowering rule is that a scalar connector names ONE element by reference, so the body
        reads it wherever the binding points -- and type inference propagates the outer storage
        inward, which is right. A body that reads it as host code therefore has no valid binding to
        device memory at all: nothing writes the value, so no copy-back exists to place, and the
        placement machinery works on arrays and never sees a scalar. Bringing the element to the
        host at the binding site is the decision the graph is missing (npbench azimint_hist reads
        ``bin_edges[i]`` in a host subtraction).

        Inputs only, and only when the body reads the value outside a kernel: one used on the device
        alone is already where it belongs.
        """
        for edge in state.in_edges(nsdfg_node):
            if edge.data is None or edge.data.is_empty() or edge.dst_conn is None:
                continue
            body = nsdfg_node.sdfg
            if edge.dst_conn not in body.arrays or not is_scalar(edge.dst_conn, body):
                continue
            if sdfg.arrays[edge.data.data].storage not in GPU_RESIDENT_STORAGES:
                continue
            if not self.read_outside_a_kernel(body, edge.dst_conn):
                continue
            desc = body.arrays[edge.dst_conn]
            host_name, _ = sdfg.add_scalar(f"{edge.dst_conn}_host",
                                           desc.dtype,
                                           transient=True,
                                           storage=dtypes.StorageType.Default,
                                           find_new_name=True)
            staged = state.add_access(host_name)
            # Reuse the edge's own source: a second access node for the same data would leave the
            # original isolated once this edge goes, which is not a valid SDFG.
            source = edge.src if isinstance(edge.src, nodes.AccessNode) else state.add_read(edge.data.data)
            state.remove_edge(edge)
            state.add_edge(source, edge.src_conn, staged, None, deepcopy(edge.data))
            state.add_edge(staged, None, nsdfg_node, edge.dst_conn,
                           Memlet.from_array(host_name, sdfg.arrays[host_name]))

    def read_outside_a_kernel(self, sdfg: SDFG, name: str) -> bool:
        """True if ``name`` is read anywhere in ``sdfg`` that a device schedule does not cover."""
        for nested in sdfg.all_sdfgs_recursive():
            for state in nested.states():
                for node in state.data_nodes():
                    if node.data == name and state.out_degree(node) > 0 and not is_devicelevel_gpu(nested, state, node):
                        return True
            for edge in nested.all_interstate_edges():
                if name in edge.data.used_arrays(nested.arrays):
                    return True
        return False

    def host_level_nested_sdfgs(self, state: SDFGState, entry: Optional[nodes.MapEntry]):
        """Nested SDFGs under ``entry`` reached through taskloops only -- one below a kernel is device code."""
        for node in self.cached_scope_children[state].get(entry, ()):
            if isinstance(node, nodes.NestedSDFG):
                yield node
            elif isinstance(node, nodes.MapEntry) and node in self.taskloops:
                yield from self.host_level_nested_sdfgs(state, node)

    def inherit_binding_storage(self, sdfg: SDFG, state: SDFGState, nsdfg_node: nodes.NestedSDFG) -> None:
        """Bind inner descriptors to their outer storage: the body reads these as starting locations."""
        for edge in state.in_edges(nsdfg_node) + state.out_edges(nsdfg_node):
            if edge.data is None or edge.data.is_empty():
                continue
            connector = edge.dst_conn if edge.dst is nsdfg_node else edge.src_conn
            if connector is None or connector not in nsdfg_node.sdfg.arrays:
                continue
            inner = nsdfg_node.sdfg.arrays[connector]
            # ARRAYS only. A scalar connector binds ONE element by reference, so it names no memory
            # of its own and the outer storage says nothing about where the body may read it: giving
            # it the device storage makes every host read of the value invalid (npbench azimint_hist
            # subtracts ``bin_edges[i]`` on the host).
            if isinstance(inner, data.Scalar):
                continue
            inner.storage = sdfg.arrays[edge.data.data].storage

    def cache_scopes(self, sdfg):
        # Nested SDFGs too: a taskloop's body is a host level, read by the walk and the analysis.
        self.cached_scopes = {}
        self.cached_scope_children = {}
        for nested in sdfg.all_sdfgs_recursive():
            for state in nested.states():
                self.cached_scopes[state] = state.scope_dict()
                self.cached_scope_children[state] = state.scope_children()

    # STEP 1
    def find_taskloops(self, sdfg: SDFG) -> None:
        """Record which maps belong on the host.

        A map enclosing a device-wide library node is recorded whatever the config says: that call is
        issued by host code, so a kernel cannot contain it. The launch-only rule is the optional half.
        """
        self.taskloops = taskloop_maps(sdfg, launch_only=self.taskloop_heuristics, overrides=self.taskloop_overrides)
        # Kept apart from ``taskloops``: a host map is only a map that does not become the kernel, so
        # it must not pick up the taskloop-specific handling those carry elsewhere in this pass.
        self._host_map_entries = host_maps(sdfg, self._host_maps)
        # A host map launches the kernels under it; a pinned map keeps its whole subtree host code.
        self._host_pinned = maps_pinned_by_host_loops(sdfg)

    def assign_schedules(self, sdfg: SDFG, host_level: bool = True) -> None:
        """``GPU_Device`` at a host level, ``Sequential`` below one; a taskloop keeps its body host-level.

        With no taskloops this is the old rule: top-level maps are the kernels.
        """

        def walk(state: SDFGState, entry, host_level: bool) -> None:
            for node in self.cached_scope_children[state].get(entry, ()):
                if isinstance(node, nodes.MapEntry):
                    pinned = node in self._host_pinned
                    is_kernel = (host_level and not pinned and node not in self.taskloops
                                 and node not in self._host_map_entries)
                    self.set_schedule(node,
                                      dtypes.ScheduleType.GPU_Device if is_kernel else dtypes.ScheduleType.Sequential)
                    walk(state, node, host_level and not is_kernel and not pinned)

                elif isinstance(node, nodes.LibraryNode):
                    self.set_schedule(node,
                                      dtypes.ScheduleType.GPU_Device if host_level else dtypes.ScheduleType.Sequential)

                elif isinstance(node, nodes.NestedSDFG):
                    # A nested SDFG at a host level is a host level of its own. Under a kernel
                    # ``host_level`` is already False, so no extra gate belongs here.
                    self.assign_schedules(node.sdfg, host_level)

        for state in sdfg.states():
            walk(state, None, host_level)

    def set_schedule(self, node, schedule: dtypes.ScheduleType) -> None:
        # Sequential specifically: Default can be lowered to CUDA in the wrong places.
        if schedule is dtypes.ScheduleType.Sequential and has_GPU_schedule(node):
            raise RuntimeError("Invalid SDFG for OffloadToAccelerator pass. All maps must have default or CPU "
                               f"schedule before pass. Node {node} has schedule type {get_schedule(node)}")
        node.schedule = schedule

    # generic HELPERS

    # STEP 2: copy analysis

    # Helpers to get the set of arrays accessed by specific nodes or edges

    def get_arrays_used_by_edge(self, sdfg: SDFG, state: SDFGState, edge, is_out_edge: bool):
        if edge.data and not edge.data.is_empty():
            data_name = edge.data.data

            if is_array(data_name, sdfg):  # array access on edge
                return {data_name}

            elif is_view(data_name, sdfg):  # view -> we need to find the corresponding view access node by iteration.
                for n in state.data_nodes():
                    if n.data == data_name:
                        if is_out_edge:
                            return get_data_used_by_outgoing_access_nodes(sdfg, state, n)
                        return get_data_used_by_incoming_access_nodes(sdfg, state, n)

            elif is_scalar(data_name, sdfg):  # might be a scalar access of an array slice
                if is_out_edge:
                    if isinstance(edge.dst, nodes.AccessNode):
                        return get_data_used_by_outgoing_access_nodes(sdfg, state, edge.dst)
                else:
                    if isinstance(edge.src, nodes.AccessNode):
                        return get_data_used_by_incoming_access_nodes(sdfg, state, edge.src)

            elif is_stream(data_name, sdfg):
                # A Stream is a queue with its own device-side push/pop protocol, not a buffer whose
                # location this pass decides: there is nothing to place and nothing to copy, and the
                # code generator allocates it where the kernel that pushes into it runs. Invisible to
                # the analysis, which is what the offloading it replaced did with one as well.
                return OrderedSet()

            else:
                raise RuntimeError(f"edge {edge} carries {edge.data}, which is neither an array, a scalar nor a view")

        return OrderedSet()

    def host_preferred_arrays(self, sdfg: SDFG, state: SDFGState, node: nodes.LibraryNode) -> OrderedSet[str]:
        """Single-element INPUTS of a device library node, which are cheaper to leave on the host.

        A vendor call reads a coefficient through a host pointer just as happily as a device one, so
        moving one element to the device buys nothing and costs a transfer before the launch. Only
        the connectors a node declares in ``LibraryNode.host_or_device_connectors`` qualify: an
        expansion that is a device map reads every operand inside the kernel, and a host pointer
        there is a memory access fault at run time (QE vexx_k's ``np.where(match, jv, np.max(jv))``,
        whose reduced ``max_jv`` was copied back to the host only for the select's kernel to read).
        Outputs are excluded -- the call writes those on the device.
        """
        preferred: OrderedSet[str] = OrderedSet()
        for edge in state.in_edges(node):
            if edge.dst_conn not in node.host_or_device_connectors or edge.data is None or edge.data.is_empty():
                continue
            name = edge.data.data
            # Length-1 ARRAYS only: a scalar is never placed at all (the pass asserts as much), so
            # naming one here would put it in a set that must not hold it -- tsvc_2_5
            # ext_break_capture's ``__ff_KFIND``.
            if name in sdfg.arrays and is_length1_array(name, sdfg):
                preferred.add(name)
        return preferred

    def host_pinned_arrays_in_state(self, sdfg: SDFG, state: SDFGState) -> OrderedSet[str]:
        """Every host-pinned array of the library nodes at this state's own level.

        Only that level: a node under a kernel is device code, and a pin there would name memory
        the host cannot reach anyway.
        """
        pinned: OrderedSet[str] = OrderedSet()
        for node in self.cached_scope_children[state].get(None, ()):
            if isinstance(node, nodes.LibraryNode):
                pinned |= self.host_pinned_arrays(sdfg, state, node)
        return pinned

    def host_pinned_arrays(self, sdfg: SDFG, state: SDFGState, node: nodes.LibraryNode) -> OrderedSet[str]:
        """Arrays this library node reaches through a connector it declares HOST-resident.

        A node whose expansion is a device call can still read part of its interface on the host --
        cuBLAS takes alpha and beta through a host pointer, ``ScatterConflictCheck`` reads its flag
        there -- and says so through ``LibraryNode.host_connectors``. Placing one of those on the
        device gives the expansion a device pointer to dereference in host code.
        """
        if not node.host_connectors:
            return OrderedSet()
        pinned: OrderedSet[str] = OrderedSet()
        for edge in state.in_edges(node):
            if edge.dst_conn in node.host_connectors:
                pinned |= self.get_arrays_used_by_edge(sdfg, state, edge, False)
        for edge in state.out_edges(node):
            if edge.src_conn in node.host_connectors:
                pinned |= self.get_arrays_used_by_edge(sdfg, state, edge, True)
        return pinned

    def get_arrays_used_by_node(self, sdfg, state, node):
        arrays: OrderedSet[str] = OrderedSet()

        # edges
        for e in state.in_edges(node):
            arrays |= self.get_arrays_used_by_edge(sdfg, state, e, False)

        for e in state.out_edges(node):
            arrays |= self.get_arrays_used_by_edge(sdfg, state, e, True)

        # neighbouring access nodes
        arrays |= get_data_used_by_incoming_access_nodes(sdfg, state, node)
        arrays |= get_data_used_by_outgoing_access_nodes(sdfg, state, node)

        return arrays

    # Data Analysis: traverse the graph and sort all accessed arrays into gpu and cpu sets

    def get_data_locations_of_map(self, sdfg: SDFG, state: SDFGState, map_entry: nodes.MapEntry):
        """
        finds all arrays accessed by a map, i.e. arrays which are
            - part of the read/write set of an enclosed tasklet
            - data of an enclosed access node
            - accessed by a second, enclosed map
            - the original arrays behind an accessed view

        and decides whether their location should be on gpu or a cpu, i.e.
            - gpu if ANY parent map has a gpu schedule (even if the direct parent has cpu-schedule)
            - cpu else

        returns two sets (gpu_set, cpu_set) with the names of the respective arrays
        """

        # helper to validate data and add it to correct set
        def _add_data(data_name: str,
                      gpu_set: OrderedSet[str],
                      cpu_set: OrderedSet[str],
                      is_gpu: bool,
                      host_level: bool = False) -> tuple[OrderedSet[str], OrderedSet[str]]:
            if data_name in gpu_set:  # has already been accessed on GPU
                if not is_gpu:  # is now accessed on CPU
                    if host_level:
                        # A launcher staging on the host for a kernel is a hybrid state, not an error.
                        cpu_set.add(data_name)
                        return
                    raise RuntimeError("GPU->CPU inside a map: an inner sequential map still runs as a kernel, so data "
                                       "under a GPU map has to stay on the GPU")

            elif data_name in cpu_set:  # has already been accessed on CPU
                if is_gpu:  # is now accessed on GPU
                    gpu_set.add(data_name)
                    #raise RuntimeError("CPU->GPU copy needed within map for " + data_name)

            else:
                assert isinstance(data_name, str), f"{data_name} -> {data_name.__class__.__name__}"
                (gpu_set if is_gpu else cpu_set).add(data_name)

        # main work horse, can recurse to nested maps
        def _recursive_helper(sdfg: SDFG,
                              state: SDFGState,
                              map_entry: nodes.MapEntry,
                              gpu_set: OrderedSet[str],
                              cpu_set: OrderedSet[str],
                              is_gpu: bool,
                              host_level: bool = True):
            is_gpu = is_gpu or map_entry.map.schedule in dtypes.GPU_SCHEDULES  # TODO Q: how not to hardcode?
            is_taskloop = map_entry in self.taskloops
            host_level = host_level and is_taskloop

            # get all nodes within this map's scope
            map_nodes = [n for n, parent in self.cached_scopes[state].items() if parent is map_entry]

            # input & output nodes
            input_and_output = get_data_used_by_incoming_access_nodes(
                sdfg, state, map_entry) | get_data_used_by_outgoing_access_nodes(sdfg, state,
                                                                                 state.exit_node(map_entry))
            if is_taskloop:
                pass  # transparent for now, resolved below once the body has spoken
            elif is_gpu:
                gpu_set |= input_and_output
            else:
                cpu_set |= input_and_output

            # internal nodes
            for node in map_nodes:
                if isinstance(node, nodes.MapEntry):  # recurse on inner map
                    _recursive_helper(sdfg, state, node, gpu_set, cpu_set, is_gpu, host_level)

                elif isinstance(node, nodes.AccessNode):  # find accessed arrays -> add
                    # Staging between a launcher and a kernel says nothing about where data belongs.
                    if not host_level:
                        for name in get_data_used_by_outgoing_access_nodes(sdfg, state, node):
                            _add_data(name, gpu_set, cpu_set, is_gpu, host_level)

                elif isinstance(node, nodes.Tasklet):  # find accessed arrays -> add
                    for name in self.get_arrays_used_by_node(sdfg, state, node):
                        _add_data(name, gpu_set, cpu_set, is_gpu, host_level)

                elif isinstance(node, (ControlFlowRegion)):
                    g, c = self.get_data_locations_of_cfregion(sdfg, node)
                    if not is_gpu:
                        gpu_set |= g
                        cpu_set |= c
                    else:
                        gpu_set |= g | c

                elif isinstance(node, nodes.LibraryNode):
                    # Launched from the host, reading device memory: the schedule around it says nothing.
                    on_gpu = is_gpu or has_GPU_schedule(node)
                    # A host pin is about a HOST-issued call taking a value by value; inside a kernel
                    # the expansion is device code and there is no host to read it from.
                    host_side = OrderedSet() if is_gpu else (self.host_pinned_arrays(sdfg, state, node)
                                                             | self.host_preferred_arrays(sdfg, state, node))
                    for name in self.get_arrays_used_by_node(sdfg, state, node):
                        _add_data(name, gpu_set, cpu_set, on_gpu and name not in host_side, host_level)

                elif isinstance(node, nodes.NestedSDFG):
                    if is_gpu:
                        pass  # inside a kernel everything below is on the device already
                    else:
                        g, c = self.get_data_locations_of_nested_sdfg(sdfg, state, node)
                        gpu_set |= g
                        cpu_set |= c

                elif isinstance(node, nodes.MapExit):
                    pass

                else:
                    raise RuntimeError(f"unhandled node {node.label} of type {type(node).__name__} inside map "
                                       f"{map_entry} in state {state}")

            if is_taskloop:
                # Unclaimed by the body means device data, or every iteration pays for a copy of it.
                gpu_set |= input_and_output - cpu_set

        # function body, calls recursive helper
        gpu_set: OrderedSet[str] = OrderedSet()
        cpu_set: OrderedSet[str] = OrderedSet()
        _recursive_helper(sdfg, state, map_entry, gpu_set, cpu_set, False)
        return gpu_set, cpu_set

    def get_data_locations_of_nested_sdfg(self, sdfg: SDFG, state: SDFGState,
                                          node: nodes.NestedSDFG) -> tuple[OrderedSet[str], OrderedSet[str]]:
        """Where a nested SDFG wants its bound arrays, in the OUTER SDFG's names."""
        # Its hybrid states are resolved when the body is offloaded; wrapping them needs that SDFG.
        outer_hybrid = self.hybrid_states
        self.hybrid_states = OrderedSet()
        inner_gpu, inner_cpu = self.get_data_locations_of_cfregion(node.sdfg, node.sdfg)
        self.hybrid_states = outer_hybrid

        gpu_set: OrderedSet[str] = OrderedSet()
        cpu_set: OrderedSet[str] = OrderedSet()
        for edge in state.in_edges(node) + state.out_edges(node):
            if edge.data is None or edge.data.is_empty():
                continue
            connector = edge.dst_conn if edge.dst is node else edge.src_conn
            name = edge.data.data
            if connector is None or name not in sdfg.arrays or not is_array(name, sdfg):
                continue
            if connector in inner_gpu:
                gpu_set.add(name)
            elif connector in inner_cpu:
                cpu_set.add(name)
        return gpu_set, cpu_set

    def get_data_locations_of_state(self,
                                    sdfg: SDFG,
                                    state: SDFGState,
                                    recursive_call=False) -> tuple[OrderedSet[str], OrderedSet[str]]:
        # iterate through all toplevel nodes of this state
        #  - map entry -> give to get_data_locations_of_map, which handles all nodes inside scope
        #  - control flow (nested) -> recurse
        #  - non-nested toplevel scopes -> add accessed data to cpu set
        gpu_set: OrderedSet[str] = OrderedSet()
        cpu_set: OrderedSet[str] = OrderedSet()

        # The analysis phase never mutates, so the scope map cached for this round is the current one.
        top_level_nodes = self.cached_scope_children[state][None]
        #: What a bare host tasklet touches, which is the one host use a size-1 map can lift.
        free_tasklet_data: OrderedSet[str] = OrderedSet()

        for node in top_level_nodes:

            g, c = OrderedSet(), OrderedSet()

            # process map and all nodes within -> may be on GPU
            if isinstance(node, nodes.MapEntry):
                g, c = self.get_data_locations_of_map(sdfg, state, node)

            elif isinstance(node, nodes.MapExit):
                pass

            # library nodes are usually GPU, can be CPU
            elif isinstance(node, nodes.LibraryNode):
                host_side = (self.host_pinned_arrays(sdfg, state, node) | self.host_preferred_arrays(sdfg, state, node))
                if has_GPU_schedule(node):
                    g = self.get_arrays_used_by_node(sdfg, state, node) - host_side
                    c = host_side
                else:
                    c = self.get_arrays_used_by_node(sdfg, state, node)

            # a nested SDFG at the top of a state is host code holding its own maps
            elif isinstance(node, nodes.NestedSDFG):
                g, c = self.get_data_locations_of_nested_sdfg(sdfg, state, node)

            # recurse if nested
            elif isinstance(node, ControlFlowRegion):
                g, c = self.get_data_locations_of_cfregion(sdfg, node)

            # all else is definitely on CPU
            elif isinstance(node, nodes.Tasklet):  # outside a map scope (else handled by locations_of_map) -> cpu
                c = self.get_arrays_used_by_node(sdfg, state, node)
                free_tasklet_data |= c

            elif isinstance(node, nodes.AccessNode):
                pass  # nothing to do; cannot be classified without context

            else:
                raise RuntimeError(f"unhandled node {node} of type {node.__class__.__name__} in state {state}")

            gpu_set |= g
            cpu_set |= c

        # A name a host-issued call reads BY VALUE cannot be moved, whatever the state around it
        # does -- ``gpucub::DeviceScan``'s seed (tsvc_2_5 fission_dep_then_indep).
        pinned = self.host_pinned_arrays_in_state(sdfg, state)
        if pinned:
            cpu_set |= gpu_set & pinned
            gpu_set -= pinned

        # Check for hybrid state configurations, where arrays are accessed on both CPU and GPU.
        # A free tasklet over an already device-resident array is the same hybrid wearing a shape
        # the overlap cannot see: it is host code reading device memory, and nothing else in the
        # state marks that array as a device use, so the state goes unreconciled and the graph is
        # invalid (npbench nbody writes PE from one such tasklet and never reads it on the device).
        # The sequential arm of a guarded specialization is the exception -- see
        # :func:`in_sequential_specialization_arm`.
        resident: OrderedSet[str] = OrderedSet()
        if not in_sequential_specialization_arm(state) and not self.in_host_only_loop(state):
            resident = OrderedSet(name for name in free_tasklet_data
                                  if name in sdfg.arrays and sdfg.arrays[name].storage == dtypes.StorageType.GPU_Global)
        overlap = (gpu_set & cpu_set) | (resident - pinned)
        if overlap:
            self.hybrid_states.add(state)
            self.hybrid_overlap[state] = OrderedSet(overlap)
            gpu_set |= cpu_set - pinned
            cpu_set &= pinned

        return gpu_set, cpu_set

    def get_data_locations_of_condblock(self, sdfg: SDFG,
                                        block: ConditionalBlock) -> tuple[OrderedSet[str], OrderedSet[str]]:
        gpu_set: OrderedSet[str] = OrderedSet()
        cpu_set: OrderedSet[str] = OrderedSet()

        # get array accesses in condition
        for memlet in block.get_meta_read_memlets():
            if not memlet:
                continue
            data_name = memlet.data
            if memlet.data in sdfg.arrays and is_array(data_name, sdfg):
                cpu_set.add(memlet.data)

        # add array accesses in branches
        sides = [(branch, *self.get_data_locations_of_cfregion(sdfg, branch)) for _, branch in block.branches]
        for branch, g, c in sides:
            if any(isinstance(b, LoopRegion) and b.pinned_sequential for b in branch.nodes()):
                # A fallback arm copies in and out inside itself (:func:`in_sequential_specialization_arm`),
                # so what the other arms keep on the device is device data at this block's boundary too;
                # reported as host, every execution would round-trip it for the rarely-taken arm.
                device_elsewhere = OrderedSet(name for other, og, _oc in sides if other is not branch for name in og)
                gpu_set |= g | (c & device_elsewhere)
                cpu_set |= c - device_elsewhere
            else:
                gpu_set |= g
                cpu_set |= c

        return gpu_set, cpu_set

    def get_data_locations_of_loop(self, sdfg: SDFG, loop: LoopRegion) -> tuple[OrderedSet[str], OrderedSet[str]]:
        # get array accesses in init_statement, update_statement, and loop_condition
        cpu_set: OrderedSet[str] = OrderedSet()
        for memlet in loop.get_meta_read_memlets():
            if not memlet:
                continue
            data_name = memlet.data
            if data_name in sdfg.arrays and is_array(data_name, sdfg):
                cpu_set.add(data_name)

        # add array accesses in loop body
        gpu_set, c = self.get_data_locations_of_cfregion(sdfg, loop)
        cpu_set |= c

        return gpu_set, cpu_set

    def get_data_locations_of_cfblock(self, sdfg: SDFG,
                                      block: ControlFlowBlock) -> tuple[OrderedSet[str], OrderedSet[str]]:
        if isinstance(block, SDFGState):
            return self.get_data_locations_of_state(sdfg, block)

        elif isinstance(block, ConditionalBlock):
            return self.get_data_locations_of_condblock(sdfg, block)

        elif isinstance(block, LoopRegion):
            return self.get_data_locations_of_loop(sdfg, block)

        elif isinstance(block, ControlFlowRegion):
            return self.get_data_locations_of_cfregion(sdfg, block)

        elif isinstance(block, (nodes.NestedSDFG, ReturnBlock, ContinueBlock, BreakBlock)):
            return OrderedSet(), OrderedSet()  # do nothing

        raise RuntimeError(f"Unknown block type: {block} of type {block.__class__.__name__}")

    def get_data_locations_of_cfregion(self, sdfg: SDFG,
                                       cfr: ControlFlowRegion) -> tuple[OrderedSet[str], OrderedSet[str]]:
        gpu_set: OrderedSet[str] = OrderedSet()
        cpu_set: OrderedSet[str] = OrderedSet()

        # An interstate assignment is host code, and only a LoopRegion's or a ConditionalBlock's own
        # meta reads are reported by the blocks below -- a plain state-to-state edge is reported by
        # nobody. Left unclaimed, a taskloop hands the array down to its body as device memory that
        # the body then reads from the host (npbench spmv's ``start = A_row[i]``).
        for edge in cfr.edges():
            for data_name in edge.data.used_arrays(sdfg.arrays):
                if is_array(data_name, sdfg):
                    cpu_set.add(data_name)

        for block in cfr.bfs_nodes():
            g, c = self.get_data_locations_of_cfblock(sdfg, block)
            gpu_set |= g
            cpu_set |= c

        return gpu_set, cpu_set

    # wrapper
    #def get_data_locations(self, sdfg:SDFG) -> tuple[OrderedSet[str], OrderedSet[str]]:
    #    return self.get_data_locations_of_cfregion(sdfg, sdfg)

    # STEP 3: Intermediate Representation
    def written_arrays(self, sdfg: SDFG) -> OrderedSet[str]:
        """Names this SDFG writes: an access node with an incoming edge.

        Not ``SDFG.read_and_write_sets``, which resolves every access node's descriptor: the one
        caller runs between ``_insert_copy_names``, which renames nodes onto the device names, and
        ``create_interstate_copy``, which registers those descriptors -- so the graph names data it
        does not hold yet and resolving raises.
        """
        written: OrderedSet[str] = OrderedSet()
        for state in sdfg.states():
            for node in state.data_nodes():
                if state.in_degree(node) > 0:
                    written.add(node.data)
        return written

    def sdfg_to_IR(self, sdfg: SDFG):

        # remember initial non-transient array locations
        non_transients = {name for name in sdfg.arrays if not sdfg.arrays[name].transient and not is_scalar(name, sdfg)}
        initially_on_gpu = OrderedSet()
        initially_on_cpu = OrderedSet()

        for array_name in non_transients:
            if is_array_stored_on_GPU(sdfg, array_name):
                initially_on_gpu.add(array_name)
            else:
                initially_on_cpu.add(array_name)

        # create inital node (open node)
        IR = OffloadingIRNode.new_open_node(sdfg)
        IR.gpu_set = initially_on_gpu.copy()
        IR.cpu_set = initially_on_cpu.copy()  # no copy -> may cause sideeffects

        # parse entire graph
        end = self._parse_to_IR(sdfg, sdfg, IR)

        # finish graph: tie the final node together with the inital close node
        end.append_node(IR.close)
        link_early_returns(IR)
        # Only what this SDFG WROTE goes back: restoring a read-only input is dead traffic, and
        # inside a nested SDFG it writes an input connector, which is invalid.
        written = self.written_arrays(sdfg)
        IR.close.gpu_set = initially_on_gpu & written
        IR.close.cpu_set = initially_on_cpu & written

        self._propagate_arrays(IR)

        return IR

    def _parse_to_IR(self, sdfg: SDFG, cfr: ControlFlowRegion, curr_node: OffloadingIRNode) -> OffloadingIRNode:
        # NOTE to self: ControlFlowRegion inherits from ControlFlowBlock
        block: ControlFlowBlock
        for block in cfr.bfs_nodes():

            # iterate through all (incoming) interstate edges
            in_edge_arrays = OrderedSet()
            for edge in cfr.in_edges(block):
                arrays = {data_name for data_name in edge.data.used_arrays(sdfg.arrays) if is_array(data_name, sdfg)}
                in_edge_arrays |= arrays

            if in_edge_arrays:
                edge_node = OffloadingIRNode.new_edge_node(block, in_edge_arrays)
                curr_node.append_node(edge_node)
                curr_node = edge_node

            # iterate through all nodes
            # non-nested state
            if isinstance(block, SDFGState):
                state: SDFGState = block
                gpu_set, cpu_set = self.get_data_locations_of_state(sdfg,
                                                                    state)  # beating heart of this entire function
                state_node = OffloadingIRNode.new_state_node(state, cpu_set, gpu_set)
                curr_node.append_node(state_node)
                curr_node = state_node

            # do nothing
            elif isinstance(block, (ReturnBlock, ContinueBlock, BreakBlock)):
                pass

            # container node with outer wrapper
            else:
                # outer node
                outer_node = OffloadingIRNode.new_open_node(block)
                curr_node.append_node(outer_node)
                curr_node = outer_node

                # if else
                if isinstance(block, ConditionalBlock):
                    cond_block: ConditionalBlock = block

                    # branch condition
                    meta_data_node: OffloadingIRNode = None
                    meta_data = {
                        memlet.data
                        for memlet in cond_block.get_meta_read_memlets() if memlet.data in sdfg.arrays
                    }
                    if meta_data:
                        meta_data_node = OffloadingIRNode.new_state_node(block, cpu_set=meta_data, gpu_set=OrderedSet())
                        curr_node.append_node(meta_data_node)
                        curr_node = meta_data_node

                    # parse branches and connect each branch to close node
                    for _, branch in cond_block.branches:
                        branch_end: OffloadingIRNode = self._parse_to_IR(sdfg, branch, curr_node)
                        # TODO: FIND ALL TAILS
                        branch_end.append_node(outer_node.close)

                # loop
                elif isinstance(block, LoopRegion):
                    loop: LoopRegion = block

                    # add meta data node if needed
                    meta_data_node: OffloadingIRNode = None
                    meta_data = OrderedSet(memlet.data for memlet in loop.get_meta_read_memlets()
                                           if memlet.data in sdfg.arrays)
                    if meta_data:
                        meta_data_node = OffloadingIRNode.new_state_node(block, cpu_set=meta_data, gpu_set=OrderedSet())
                        curr_node.append_node(meta_data_node)
                        curr_node = meta_data_node

                    # parse body and connect to loop close node
                    # TODO: FIND ALL TAILS
                    curr_node = self._parse_to_IR(sdfg, loop,
                                                  curr_node)  # linked list representing all internal nodes of loop
                    curr_node.append_node(outer_node.close)

                # nested region -> flatten
                elif isinstance(block, ControlFlowRegion):
                    curr_node = self._parse_to_IR(sdfg, block, curr_node)
                    curr_node.append_node(outer_node.close)

                elif isinstance(block, nodes.NestedSDFG):
                    curr_node = self._parse_to_IR(block.sdfg, block.sdfg, curr_node)
                    curr_node.append_node(outer_node.close)

                else:
                    raise RuntimeError(f"Unknown block type: {block} of type {block.__class__.__name__}")

                # finish container
                self._populate_container_node_sets(outer_node)
                curr_node = outer_node.close

        # TODO: FIND ALL TAILS?
        return curr_node

    def traverse_IR_after_predecessors(self, IR: OffloadingIRNode, method) -> None:
        """Apply ``method`` to each IR node once all of its predecessors have had it applied.

        The walk :meth:`__traverse_IR` makes, held back at every join until its last in-edge
        arrives, so arms are still taken in ``node.next`` order.
        """
        waiting: dict[OffloadingIRNode, int] = {}

        def count(node: OffloadingIRNode) -> None:
            for next in node.next:
                waiting[next] = waiting.get(next, 0) + 1

        traverse_IR(IR, count)
        stack = [IR]
        while stack:
            node = stack.pop()
            method(node)
            ready = []
            for next in node.next:
                waiting[next] -= 1
                if waiting[next] == 0:
                    ready.append(next)
            stack.extend(reversed(ready))
        stuck = [node.debug_name for node, pending in waiting.items() if pending]
        if stuck:
            raise RuntimeError(f'the offloading IR is not a DAG: {stuck} are never reached by all predecessors')

    def _populate_container_node_sets(self, IR: OffloadingIRNode):
        self.__populate_open_node_sets(IR)
        self.__populate_close_node_sets(IR)
        # TODO for both: deal with FIND ALL TAILS

    def __populate_open_node_sets(self, IR: OffloadingIRNode):
        assert IR.is_open_node(), str(IR)

        # Behavior 1:
        # if there are no or multiple direct children, leave the sets empty & simply propagate later
        # there is no good heuristic to choose from here, which copies to make and which not
        # (not without significantly more analysis)
        children = IR.next
        if len(children) != 1:
            return

        # Behavior 2:
        # if there is a single direct child, then analyse the section & find first known location of each used array
        # if the graph splits later, the first of all possible paths is chosen for analysis
        # this can lead to unnecessary copies in the other paths
        location_on_gpu = {}

        def gather_data(node: OffloadingIRNode):
            if isinstance(node.block, nodes.NestedSDFG
                          ):  # Nested SDFGs do not share namespace, array names should not leak to outer scope
                return

            for array_name in node.gpu_set:
                if array_name not in location_on_gpu:
                    location_on_gpu[array_name] = True

            for array_name in node.cpu_set:
                if array_name not in location_on_gpu:
                    location_on_gpu[array_name] = False

        # traverse graph
        traverse_same_level(IR, gather_data)

        # populate IR sets
        IR.gpu_set = OrderedSet(array_name for array_name in location_on_gpu if location_on_gpu[array_name])
        IR.cpu_set = OrderedSet(array_name for array_name in location_on_gpu if not location_on_gpu[array_name])

    def __populate_close_node_sets(self, IR: OffloadingIRNode):
        assert IR.is_open_node(), str(IR)

        tails = IR.get_all_tails()
        assert tails, f"{IR.debug_name} doesn't have any tails! {IR}"

        # Behavior 1:
        # if there is a single route to this section's close node, then analyse the section & find
        # the last known location of each used array. A conditional inside the section is two routes
        # even when its arms meet again before one tail: the walk below steps over it.
        if IR.has_one_route():
            # define data gathering function
            location_on_gpu = {}

            def gather_data(node: OffloadingIRNode):
                if isinstance(node.block, nodes.NestedSDFG
                              ):  # Nested SDFGs do not share namespace, array names should not leak to outer scope
                    return

                for array_name in node.gpu_set:
                    location_on_gpu[array_name] = True

                for array_name in node.cpu_set:
                    location_on_gpu[array_name] = False

            # traverse graph
            traverse_same_level(IR, gather_data)

            # populate IR sets
            IR.close.gpu_set = OrderedSet(array_name for array_name in location_on_gpu if location_on_gpu[array_name])
            IR.close.cpu_set = OrderedSet(array_name for array_name in location_on_gpu
                                          if not location_on_gpu[array_name])

        # Behaviour 2:
        # if there are multiple tail nodes, then mark this node for later.
        # In a second pass, it will assume the gpu&cpu set of its next successor.
        # This means that each branch will have to insert copies individually, usually leading to the least amount of
        # necessary copies.

    def _propagate_arrays(self, IR: OffloadingIRNode):
        # all arrays which aren't used by this state retain their previous status
        # ASSUMPTION: arrays are either gpu or cpu within a state
        def propagate(node):
            # A fallback arm's tail does not decide where the block it closes leaves its data: the arm
            # copies back inside itself, so the other arms' locations hold after the conditional. Taken
            # first-come, an arm listed first made every execution copy to the host and back.
            block = node.open.block if node.type == OffloadingIRNode.CLOSE and node.open else node.block
            arm_tail = isinstance(block, ControlFlowBlock) and in_sequential_specialization_arm(block)
            if node.type == OffloadingIRNode.STATE and isinstance(node.block, SDFGState):
                self.place_copy_destinations(node)
            for next in node.next:
                if arm_tail and next.type == OffloadingIRNode.CLOSE and not in_sequential_specialization_arm(
                        next.open.block):
                    continue
                next_arrays = next.cpu_set | next.gpu_set

                for array in node.cpu_set:
                    if array not in next_arrays:
                        next.cpu_set.add(array)
                for array in node.gpu_set:
                    if array not in next_arrays:
                        next.gpu_set.add(array)

        # A node forwards what it holds when visited, so a join must first hear from every arm.
        self.traverse_IR_after_predecessors(IR, propagate)

    def place_copy_destinations(self, node: OffloadingIRNode) -> None:
        """A top-level container-to-container copy writes its destination on its source's side.

        The state analysis leaves such a copy unplaced, so without this the destination kept the
        location it had before the copy and a later reader on the other side got no copy in
        (cegterg: ``__inl9_a = hc`` on the device, then a host loop over ``__inl9_a_host``).
        """
        state = node.block
        sdfg = state.sdfg
        top = self.cached_scope_children[state][None]
        for edge in state.edges():
            src, dst = edge.src, edge.dst
            if not (isinstance(src, nodes.AccessNode) and isinstance(dst, nodes.AccessNode)) or edge.data.is_empty():
                continue
            if src not in top or dst not in top or dst.data in node.cpu_set or dst.data in node.gpu_set:
                continue
            # A copy between a container and its own twin is this pass's placement, not the program's.
            if self.are_twins(src.data, dst.data):
                continue
            if isinstance(sdfg.arrays[dst.data], data.View) or not is_array(dst.data, sdfg):
                continue
            if src.data in node.gpu_set:
                node.gpu_set.add(dst.data)
            elif src.data in node.cpu_set:
                node.cpu_set.add(dst.data)

    def are_twins(self, a: str, b: str) -> bool:
        return b in (self._get_host_name(a), self._get_gpu_name(a)) or a in (self._get_host_name(b),
                                                                             self._get_gpu_name(b))

    def _insert_copy_names_in_block(self,
                                    sdfg: SDFG,
                                    block: ControlFlowBlock,
                                    rename_dict: dict,
                                    interstate_only: bool = False):
        if block is None:
            return

        cfr = block.parent_graph
        if cfr and isinstance(cfr, AbstractControlFlowRegion):
            for edge in cfr.in_edges(block):
                relevant_edge_arrays = edge.data.used_arrays(rename_dict)
                # Renaming is decided by where the array BEGINS, not by where this edge reads
                # it: one that begins on the GPU is read through its host copy under the host
                # name, one that begins on the CPU keeps its own name. The copies themselves
                # are inserted later.
                for name in relevant_edge_arrays:
                    if sdfg.arrays[name].storage == dtypes.StorageType.GPU_Global:
                        edge.data.replace(name, self._get_host_name(name))

        # An EDGE node stands for the interstate edges REACHING ``block``, and it holds the same block
        # object as the state node that follows it. Its decision is about what those edges read, so
        # letting it fall through here would apply it a second time to dataflow the state node has
        # already decided about -- tsvc s315, where a host tasklet writing ``a`` came out writing
        # ``a_gpu``.
        if interstate_only:
            return

        if isinstance(block, SDFGState):
            self._insert_copy_names_in_state(block, rename_dict)

        elif isinstance(block, ControlFlowBlock):
            # rename meta accesses (control-flow metadata like loop bounds or conditions)
            block.replace_meta_accesses(rename_dict)
            # NOTE: states / blocks within the current block all have their own IRNodes and don't need to be handled
            # recursively here
        else:
            raise NotImplementedError(
                f"in _correct_names_in_block: IR.block unhandled type: {block} is {block.__class__.__name__}")

    def _insert_copy_names_in_state(self, state: SDFGState, rename_dict: dict):
        # rename access nodes
        for access in state.data_nodes():
            if access.data in rename_dict:
                access.data = rename_dict[access.data]

        # rename edge conditions
        for edge in state.edges():
            for e in state.memlet_tree(edge):
                memlet = e.data
                if memlet is not None and not memlet.is_empty() and memlet.data in rename_dict:
                    memlet.data = rename_dict[memlet.data]

    def _insert_copy_names(self, sdfg: SDFG, IR: OffloadingIRNode):
        # make a rename dict for each IR node, then rename all such arrays in the IR.block
        def _insert_copy_names_in_node(node: OffloadingIRNode):
            rename_dict = {}
            # Which SIDE the descriptor lives on, not which storage names that side. ``Default`` is
            # one host storage of several, and a graph that has been through ``auto_optimize`` for
            # the CPU carries ``CPU_Heap`` instead -- against which an identity test is False, so the
            # device accesses kept the HOST name. The kernel then wrote the host array while the
            # copy-back overwrote it with a device buffer nothing had written: vadv came back exactly
            # as it went in.
            for name in node.gpu_set:
                assert name in sdfg.arrays
                if not is_array_stored_on_GPU(sdfg, name):  # starts on CPU, but this access is on GPU
                    rename_dict[name] = self._get_gpu_name(name)

            for name in node.cpu_set:
                assert name in sdfg.arrays
                if is_array_stored_on_GPU(sdfg, name):  # starts on GPU, but this access is on CPU
                    rename_dict[name] = self._get_host_name(name)

            self._insert_copy_names_in_block(sdfg, node.block, rename_dict, node.type == OffloadingIRNode.EDGE)

        traverse_IR(IR, _insert_copy_names_in_node)

    def _correct_transient_storage_locations(self, sdfg: SDFG, IR: OffloadingIRNode):
        seen_transients = OrderedSet()

        def _correct_transients(node: OffloadingIRNode):
            for name in node.gpu_set:
                assert name in sdfg.arrays
                desc = sdfg.arrays[name]
                if desc.transient and name not in seen_transients:
                    desc.storage = dtypes.StorageType.GPU_Global
                    seen_transients.add(name)

            for name in node.cpu_set:
                assert name in sdfg.arrays
                desc = sdfg.arrays[name]
                if desc.transient and name not in seen_transients:
                    desc.storage = dtypes.StorageType.Default
                    seen_transients.add(name)

        traverse_IR(IR, _correct_transients)

    def eval_IR(self, sdfg: SDFG, IR: OffloadingIRNode) -> None:
        # modifies SDFG in place & inserts all necessary copies
        # Filled after the renaming below, where a host-side write takes the host name.
        written: OrderedSet[str] = OrderedSet()
        # Directions each container is already copied in at one program point, keyed (block, side). Loop tails
        # and IR edges resolve to the same block -- every branch of a conditional ends at the ConditionalBlock --
        # and each one stacked another identical copy state there (24 in a row after one CLOUDSC branch).
        placed: dict[tuple[Any, str], dict[str, set[bool]]] = {}
        # Fills of containers nothing writes, by direction: placed ONCE, at the program's entry.
        entry_fills: dict[bool, OrderedSet[str]] = {True: OrderedSet(), False: OrderedSet()}

        def twin_of(name: str) -> str:
            return self._get_host_name(name) if is_array_stored_on_GPU(sdfg, name) else self._get_gpu_name(name)

        def place_copy(before, after, array_names, to_gpu: bool):
            # A copy toward the container's home is about modifications of its twin: a twin nothing
            # writes still holds what the home holds (CloudSC's ``iphase_gpu -> iphase`` after the
            # kernels that only read it). Decided here, for every copy, and not only between two
            # blocks: the end-of-iteration copies of a loop came through without it, and polybench
            # nussinov copied ``seq_host -> seq`` after every ``j`` of its O(N^2) host loop nest.
            array_names = OrderedSet(name for name in array_names
                                     if (is_array_stored_on_GPU(sdfg, name) != to_gpu or twin_of(name) in written)
                                     and name not in self._read_only_duplicates)
            # A fill of the twin of a container that NEITHER side writes copies the same bytes each
            # time, so the first fill is the only one that carries anything. The IR still moves such
            # a container back and forth around a loop -- that is what it records, not whether the
            # data changed -- and the refill then runs every iteration: nussinov read ``seq`` on the
            # host inside its ``j`` loop and refilled ``seq_host`` there, 6.4 million copies per call,
            # and lavamd refilled all of ``box_offsets_host`` once per box, O(n_boxes^2) bytes. Its
            # value on entry is its value throughout, so one fill at the entry serves every reader.
            constant = OrderedSet(name for name in array_names if name not in written and twin_of(name) not in written)
            # Except inside a fallback arm, whose copies exist so the parallel arm pays none.
            if in_sequential_specialization_arm(after if after is not None else before):
                constant = OrderedSet()
            entry_fills[to_gpu] |= constant
            array_names = OrderedSet(name for name in array_names if name not in constant)
            point = (after, 'before') if after is not None else (before, 'after')
            directions = placed.setdefault(point, {})
            fresh = OrderedSet(name for name in array_names if directions.get(name) != {to_gpu})
            for name in fresh:
                directions.setdefault(name, set()).add(to_gpu)
            if fresh:
                self.create_interstate_copy(sdfg, before, after, fresh, to_gpu=to_gpu)

        def insert_copies(node, next, node_block, next_block):
            # Copying BACK to the device is about host-side modifications. A name whose host copy is
            # never written already matches on the device, and when it is a nested SDFG's input
            # connector the copy is not merely wasted -- it writes a container the body may only read
            # (npbench scattering_self_energies' ``neigh_idx``).
            gpu_copies = {name for name in node.cpu_set & next.gpu_set if name not in self.no_copy_in_needed}
            if gpu_copies:
                place_copy(node_block, next_block, gpu_copies, to_gpu=True)

            cpu_copies = node.gpu_set & next.cpu_set
            if cpu_copies:
                place_copy(node_block, next_block, cpu_copies, to_gpu=False)

        def eval(node: OffloadingIRNode):
            for next in node.next:

                if node.cpu_set & node.gpu_set:
                    raise NotImplementedError(
                        f"state {node.debug_name} uses {node.cpu_set & node.gpu_set} on both the CPU and "
                        f"the GPU; this pass cannot place a copy inside a single state")

                # A CLOSE node has no block of its own: a copy after it goes after the region it closes.
                # Before an interstate edge, or between two CLOSEs, there is no next block: copy AFTER the node.
                after = node.open.block if node.type == OffloadingIRNode.CLOSE else node.block
                if next.type == OffloadingIRNode.EDGE or (node.type == OffloadingIRNode.CLOSE
                                                          and next.type == OffloadingIRNode.CLOSE):
                    insert_copies(node, next, after, None)

                else:  # the usual: copies between node -> next
                    insert_copies(node, next, node.block, next.block)

            # loop copies if applicable
            if node.type == OffloadingIRNode.OPEN_LOOP:
                top = node  # INV: top.type == OffloadingIRNode.OPEN_LOOP
                bottom = node.close  # INV: bottom.type == OffloadingIRNode.CLOSE
                tails = OffloadingIRNode.get_all_tails(top)  # INV: all are STATE or CLOSE if there's a nested loop

                gpu_copies = bottom.cpu_set & top.gpu_set

                if gpu_copies:
                    for tail in tails:
                        if tail.type == OffloadingIRNode.CLOSE:  # and bottom.type == OffloadingIRNode.CLOSE:
                            place_copy(tail.open.block, None, gpu_copies, to_gpu=True)
                        else:
                            place_copy(tail.block, None, gpu_copies, to_gpu=True)

                cpu_copies = bottom.gpu_set & top.cpu_set
                if cpu_copies:
                    for tail in tails:
                        if tail.type == OffloadingIRNode.CLOSE:
                            place_copy(tail.open.block, None, cpu_copies, to_gpu=False)
                        else:
                            place_copy(tail.block, None, cpu_copies, to_gpu=False)

        self._correct_transient_storage_locations(sdfg, IR)
        self._insert_copy_names(sdfg, IR)
        written |= self.written_arrays(sdfg)
        traverse_IR(IR, eval)
        for to_gpu, names in entry_fills.items():
            if names:
                self.create_interstate_copy(sdfg, None, sdfg.start_block, names, to_gpu=to_gpu)

    # Step 4: Copy Insertion
    # create ONE copy state for all arrays in array_names

    def create_interstate_copy(self, sdfg, state1, state2, array_names, to_gpu: bool):
        assert state1 is not None or state2 is not None, "invalid: both states are None"

        # 1) insert new state
        copy_state: SDFGState
        joined = '_'.join(sorted(array_names))
        direction = 'to_gpu' if to_gpu else 'to_host'
        label = f"copy_{joined}_{direction}"

        if state2 is not None:
            target_graph = state2.parent_graph
            assert target_graph is not None, "copy insertion requires a parent control-flow graph (s2)"

            copy_state = target_graph.add_state_before(state2, label=label)
            if state2 is target_graph.start_block:
                target_graph.start_block = target_graph.node_id(copy_state)  # copy state becomes new start block

        elif state1 is not None:
            target_graph = state1.parent_graph if state1.parent_graph else state1
            assert target_graph is not None, "copy insertion requires a parent control-flow graph (s1)"

            #copy_state = self.add_state_after(target_graph, state1, label)
            copy_state = target_graph.add_state_after(state1, label=label)

        # 2) create the copy map with correct names
        copy_map = {}
        name: str
        for name in array_names:
            assert name in sdfg.arrays

            if is_array_stored_on_GPU(sdfg, name):  # original array is on GPU
                if not to_gpu:  # copy goes to CPU: A -> A_host
                    copy_map[name] = self._get_host_name(name)

                else:  # copy goes to GPU: A_host -> A
                    copy_map[self._get_host_name(name)] = name

            else:  # original array is on CPU
                if to_gpu:  # copy goes to GPU: A -> A_gpu
                    copy_map[name] = self._get_gpu_name(name)

                else:  # copy goes to CPU: A_gpu -> A
                    copy_map[self._get_gpu_name(name)] = name

        # 3) build all the copies inside the new state
        for old_name, new_name in copy_map.items():

            # a) if first copy of this array: register new copy array with sdfg
            if new_name not in sdfg.arrays:
                self._register_new_copy_transient(sdfg, new_name, old_name)
            elif old_name not in sdfg.arrays:
                self._register_new_copy_transient(
                    sdfg, old_name, new_name
                    # in some cases, e.g. loops, a copy-from can be registered before its copy-to, leading to an unknown
                    # "old_name"
                )

            # b) a view is re-derived from whichever container it aliases on each side, so it needs
            # the descriptor above but no copy of its own: it has no storage, and the container's
            # copy in this same state already carries the data.
            if isinstance(sdfg.arrays[old_name], data.View) or isinstance(sdfg.arrays[new_name], data.View):
                continue

            # c) add (Access Node -> Access Node) to state
            copy_in = copy_state.add_access(old_name)
            copy_out = copy_state.add_access(new_name)

            src_desc = sdfg.arrays[old_name]
            dst_desc = sdfg.arrays[new_name]
            src_subset = subsets.Range.from_array(src_desc)
            dst_subset = subsets.Range.from_array(dst_desc)

            copy_memlet = Memlet(
                data=old_name,
                subset=src_subset,
                other_subset=dst_subset,
            )

            copy_state.add_edge(copy_in, None, copy_out, None, copy_memlet)

    def _register_new_copy_transient(self, sdfg: SDFG, unknown_name: str, known_name: str):
        assert known_name in sdfg.arrays
        desc = sdfg.arrays[known_name]

        new_storage = dtypes.StorageType.Default if is_array_stored_on_GPU(
            sdfg, known_name) else dtypes.StorageType.GPU_Global
        if isinstance(desc, data.View):
            sdfg.add_view(unknown_name, desc.shape, desc.dtype, storage=new_storage)
        else:
            sdfg.add_array(unknown_name, desc.shape, desc.dtype, storage=new_storage, transient=True)

    def _get_host_name(self, name: str) -> str:
        """Host-side copy of ``name``.

        ``__return`` is special-cased because it names the SDFG's return slot: a copy of it is a
        buffer, not another return value.
        """
        if name.startswith("__return"):
            return f"buffer__return{name[8:]}_host"
        return f"{name}_host"

    def _get_gpu_name(self, name: str) -> str:
        """Device-side copy of ``name``; see :meth:`_get_host_name` for ``__return``."""
        if name.startswith("__return"):
            return f"buffer__return{name[8:]}_gpu"
        return f"{name}_gpu"

# OPTIMIZATION

# heuristic: size1 maps are faster than more CPU-GPU copies

    from collections import deque

    def _get_boundary_in_edges(self, state: SDFGState, node, bounded_set: set):
        # A list, not a set: the connector numbering below follows this order.
        return [e for e in state.in_edges(node) if e.src not in bounded_set]

    def _get_boundary_out_edges(self, state: SDFGState, node, bounded_set: set):
        return [e for e in state.out_edges(node) if e.dst not in bounded_set]

    def _get_entry_nodes(self, state: SDFGState, bounded_set: set):
        return OrderedSet(node for node in bounded_set if all(e.src not in bounded_set for e in state.in_edges(node)))

    def _get_exit_nodes(self, state: SDFGState, bounded_set: set):
        return OrderedSet(node for node in bounded_set if all(e.dst not in bounded_set for e in state.out_edges(node)))

    def _wrap_region_in_size1_map(self, state: SDFGState, region_nodes: set) -> tuple[nodes.MapEntry, nodes.MapExit]:
        if not region_nodes:
            return
        map_label, map_param = get_new_map_identifiers(state, "size1_wrap_region", "__wrap_i")
        map_entry, map_exit = state.add_map(name=map_label,
                                            ndrange={map_param: '0:1'},
                                            schedule=dtypes.ScheduleType.GPU_Device)

        # MAP ENTRY
        boundary_in_edges = []
        for node in region_nodes:
            boundary_in_edges += self._get_boundary_in_edges(state, node, region_nodes)

        idx = 0
        for edge in boundary_in_edges:
            src, src_conn, dst, dst_conn = edge.src, edge.src_conn, edge.dst, edge.dst_conn
            ext_memlet = deepcopy(edge.data)
            int_memlet = deepcopy(edge.data)
            state.remove_edge(edge)

            # An empty memlet ORDERS; it carries no data and so may not carry a connector.
            if ext_memlet.is_empty():
                state.add_nedge(src, map_entry, ext_memlet)
                state.add_nedge(map_entry, dst, int_memlet)
                continue

            in_conn = f"IN_REGION_IN_{idx}"
            out_conn = f"OUT_REGION_IN_{idx}"
            map_entry.add_in_connector(in_conn)
            map_entry.add_out_connector(out_conn)
            idx += 1

            state.add_edge(src, src_conn, map_entry, in_conn, ext_memlet)
            state.add_edge(map_entry, out_conn, dst, dst_conn, int_memlet)

        # A region root the rewiring did not reach reads nothing, so no edge put it under the entry
        # and the scope does not contain it -- while whatever it feeds does, which is the invalid
        # inside-to-outside path. Order it after the entry instead. Rewiring a boundary edge does
        # not make the OTHER roots any less dangling, so this cannot be an else-branch of it: tsvc
        # s252 wraps one tasklet reading two arrays beside one reading none.
        for node in region_nodes:
            if state.in_degree(node) == 0:
                state.add_nedge(map_entry, node, Memlet())

        # MAP EXIT
        boundary_out_edges = []
        for node in region_nodes:
            boundary_out_edges += self._get_boundary_out_edges(state, node, region_nodes)

        idx = 0
        for edge in boundary_out_edges:
            src, src_conn, dst, dst_conn = edge.src, edge.src_conn, edge.dst, edge.dst_conn
            int_memlet = deepcopy(edge.data)
            ext_memlet = deepcopy(edge.data)
            state.remove_edge(edge)

            if int_memlet.is_empty():  # ordering edge, see above
                state.add_nedge(src, map_exit, int_memlet)
                state.add_nedge(map_exit, dst, ext_memlet)
                continue

            in_conn = f"IN_REGION_OUT_{idx}"
            out_conn = f"OUT_REGION_OUT_{idx}"
            map_exit.add_in_connector(in_conn)
            map_exit.add_out_connector(out_conn)
            idx += 1

            state.add_edge(src, src_conn, map_exit, in_conn, int_memlet)
            state.add_edge(map_exit, out_conn, dst, dst_conn, ext_memlet)

        # The leaf half of the same rule.
        for node in region_nodes:
            if state.out_degree(node) == 0:
                state.add_nedge(node, map_exit, Memlet())

        return map_entry, map_exit

    def _subgraphs_after_removing_partition_nodes(self,
                                                  state: SDFGState,
                                                  partition_nodes: set,
                                                  scope_entry=None,
                                                  scope_children=None) -> list[OrderedSet[nodes.Node]]:
        """
        Returns connected components (as sets of nodes) after removing partition_nodes
        from ONE SCOPE of a SINGLE SDFG state graph.

        ``scope_entry`` is the map whose body is partitioned, or None for the state's top level, and
        ``scope_children`` the caller's already-computed scope map -- recomputing one walks the whole
        state. Connectivity is treated as undirected (uses both in/out edges) but never leaves the
        scope: an edge out of it lands on the enclosing entry or exit, a boundary and not a member.
        """
        visited = OrderedSet()
        components = []
        if scope_children is None:
            scope_children = state.scope_children()
        members = scope_children[scope_entry]
        scope_nodes = OrderedSet(members)
        remaining_nodes = [n for n in members if n not in partition_nodes]

        for start in remaining_nodes:
            if start in visited:
                continue

            comp = OrderedSet()
            queue = self.deque([start])
            visited.add(start)

            while queue:
                u = queue.popleft()
                comp.add(u)

                neighbors = OrderedSet(e.dst for e in state.out_edges(u)) | OrderedSet(e.src for e in state.in_edges(u))
                for v in neighbors:
                    if v in partition_nodes or v in visited or v not in scope_nodes:
                        continue
                    visited.add(v)
                    queue.append(v)

            components.append(comp)

        return components

    def scope_closed_partition(self, state: SDFGState, region: OrderedSet, boundary: OrderedSet):
        """``region`` grown until every map scope it touches lies wholly inside it, or None.

        A size-1 map around HALF a scope puts a map entry inside the new map and its own exit
        outside it, which is not a scope at all: ``entry_node`` then answers for the wrapping map,
        and validation reports the pair as Map objects that were copied separately. The partition
        arrives that way because the components are cut by dataflow, which a map scope spans.

        None when closing would have to swallow one of the nodes the partitioning deliberately kept
        out -- a GPU map or a device-wide library call. Those are the boundaries the partition
        exists to respect, so the answer there is to leave this group alone rather than to wrap a
        region that reaches across one.
        """
        closed: OrderedSet = OrderedSet(region)
        scope_children = state.scope_children()
        queue = list(region)
        while queue:
            node = queue.pop()
            if isinstance(node, nodes.MapEntry):
                entry = node
            elif isinstance(node, nodes.MapExit):
                entry = state.entry_node(node)
            else:
                continue
            if entry is None:
                continue
            for extra in [entry, state.exit_node(entry), *scope_children[entry]]:
                if extra is None or extra in closed:
                    continue
                if extra in boundary:
                    return None
                closed.add(extra)
                queue.append(extra)
        return closed

    def _remove_all_outer_access_nodes_from_group(self, state: SDFGState, group: set):
        outer_nodes = self._get_entry_nodes(state, group) | self._get_exit_nodes(state, group)
        nodes_to_remove = OrderedSet(node for node in outer_nodes if isinstance(node, nodes.AccessNode))

        while nodes_to_remove:
            group -= nodes_to_remove
            outer_nodes = self._get_entry_nodes(state, group) | self._get_exit_nodes(state, group)
            nodes_to_remove = OrderedSet(node for node in outer_nodes if isinstance(node, nodes.AccessNode))

    def _insert_access_between_adjacent_maps(self, state: SDFGState, map_exit: nodes.MapExit) -> None:
        # avoid illegal direct map-to-map connections by routing through an access node.
        for edge in list(state.out_edges(map_exit)):
            if not isinstance(edge.dst, nodes.MapEntry):
                continue
            if edge.data is None or edge.data.is_empty() or edge.data.data is None:
                continue

            src, src_conn, dst, dst_conn = edge.src, edge.src_conn, edge.dst, edge.dst_conn
            access = state.add_access(edge.data.data)
            out_memlet = deepcopy(edge.data)
            in_memlet = deepcopy(edge.data)

            state.remove_edge(edge)
            state.add_edge(src, src_conn, access, None, out_memlet)
            state.add_edge(access, None, dst, dst_conn, in_memlet)

    def _find_last_access_nodes_in_map_bfs(self, state: SDFGState, map_entry: nodes.MapEntry, map_exit: nodes.MapExit,
                                           data_names: OrderedSet[str]) -> dict[str, nodes.AccessNode]:
        if not data_names:
            return {}
        last_access: dict[str, nodes.AccessNode] = {}
        queue = self.deque([map_entry])
        visited = {map_entry}

        while queue:
            node = queue.popleft()

            if isinstance(node, nodes.AccessNode) and node.data in data_names:
                last_access[node.data] = node

            if node is map_exit:
                continue

            for edge in state.out_edges(node):
                child = edge.dst
                if not child or child in visited:
                    continue
                visited.add(child)
                queue.append(child)

        return last_access

    def _forward_input_only_map_data(self, state: SDFGState, map_entry: nodes.MapEntry,
                                     map_exit: nodes.MapExit) -> None:
        # For map inputs that are not map outputs, route final in-map access through map_exit
        # -> Ensure all map inputs are also outputs to avoid dace erroneusly labeling them as constants

        # get inputs & isolate those without corresponding outputs
        input_memlets = [
            edge.data for edge in state.in_edges(map_entry)
            if edge.data is not None and not edge.data.is_empty() and edge.data.data is not None
        ]
        input_only_data = [
            memlet.data for memlet in input_memlets
            if all(edge.data is None or edge.data.is_empty() or edge.data.data != memlet.data
                   for edge in state.out_edges(map_exit))
        ]
        # find last accesses(ignore data without accesses)
        # INV: dictionary holds ONLY data which goes into the map, is accessed within but does not exit -> if left
        # unchanged this would be detected as a constant and lead to errors
        last_accesses: dict = self._find_last_access_nodes_in_map_bfs(state, map_entry, map_exit, input_only_data)

        # wire the last access through map_exit to a new outside access node
        for input_memlet in input_memlets:
            data_name = input_memlet.data
            if data_name not in last_accesses:
                continue
            last_access = last_accesses[data_name]

            # create unique connectors
            connector_index = 0
            while (f"IN_INPUT_ONLY_{connector_index}" in map_exit.in_connectors
                   or f"OUT_INPUT_ONLY_{connector_index}" in map_exit.out_connectors):
                connector_index += 1
            in_conn = f"IN_INPUT_ONLY_{connector_index}"
            out_conn = f"OUT_INPUT_ONLY_{connector_index}"

            # add new external access node & edges to it
            map_exit.add_in_connector(in_conn)
            map_exit.add_out_connector(out_conn)
            outside_access = state.add_access(data_name)
            internal_memlet = deepcopy(input_memlet)
            external_memlet = deepcopy(input_memlet)
            state.add_edge(last_access, None, map_exit, in_conn, internal_memlet)
            state.add_edge(map_exit, out_conn, outside_access, None, external_memlet)

    def host_level_scopes(self, state: SDFGState, scope_children: dict) -> list:
        """Every scope of ``state`` that runs as host code: the top level, and each taskloop body.

        A taskloop is host code by construction, so free computation left in its body is host code
        too -- and a device-wide library node beside it does not change that. Descend through
        taskloops only: a scope under a kernel is device code and has no free computation to lift.
        """
        found = [None]
        for entry in found:  # grows as taskloops are met; a taskloop under a kernel is never reached
            for node in scope_children[entry]:
                if isinstance(node, nodes.MapEntry) and node in self.taskloops:
                    found.append(node)
        return found

    def make_size1_map_wrappers(self, sdfg: SDFG, state: SDFGState):
        # Wrapping never adds or removes a taskloop, so the scope list is read once here.
        scopes = self.host_level_scopes(state, state.scope_children())
        new_maps = OrderedSet()
        for scope_entry in scopes:
            new_maps |= self.wrap_free_computation(sdfg, state, scope_entry)
        return new_maps

    def is_device_work(self, node: nodes.Node) -> bool:
        """The one rule for what a host scope launches rather than runs: its device work.

        A GPU map or GPU library node is device work, and so is a nested SDFG that launches: one
        whose own states issue a device-wide call, or that computes only inside maps. Anything else
        in a host scope -- a tasklet, a nested SDFG of scalar code -- is host work. A scope is hybrid
        when host work touching device arrays sits beside device work, and only that host work is
        wrapped, one size-1 kernel per connected run of it. A nested SDFG whose device-wide calls
        sit under its own loops is per-element code (npbench cp2k_density_matrix_trs4's row body),
        so it is host work and runs as one kernel; one that issues the call itself (npbench gem's
        body around a Reduce) stays a host level, and wrapping it would run that call in one thread.
        """
        if isinstance(node, (nodes.MapEntry, nodes.MapExit, nodes.LibraryNode)):
            return has_GPU_schedule(node)
        if not isinstance(node, nodes.NestedSDFG):
            return False
        issued = (n for body in node.sdfg.states() for n in body.scope_children()[None])
        return any(is_device_wide_libnode(n) for n in issued) or sdfg_only_launches(node.sdfg)

    def wrap_free_computation(self, sdfg: SDFG, state: SDFGState, scope_entry=None):
        """Lift the host work of ONE hybrid host scope into kernels; :meth:`is_device_work` has the rule.

        A taskloop or host map is host code by decision, and ``host_level_scopes`` lifts from INSIDE
        it, so it bounds the partitions like device work does: left out, closing a partition would
        drag its whole body -- device-wide library calls included -- into a kernel.
        """
        scope_children = state.scope_children()
        members = scope_children[scope_entry]
        partition_nodes = OrderedSet(
            node for node in members if self.is_device_work(node) or (isinstance(node, nodes.MapEntry) and (
                node in self.taskloops or node in self._host_map_entries or node in self._host_pinned)))
        partition_nodes |= OrderedSet(
            state.exit_node(node) for node in partition_nodes if isinstance(node, nodes.MapEntry))
        if scope_entry is not None:
            # The scope's own exit is its boundary, and scope_children lists it beside the body.
            partition_nodes.add(state.exit_node(scope_entry))

        partitions = self._subgraphs_after_removing_partition_nodes(state, partition_nodes, scope_entry, scope_children)
        new_maps = OrderedSet()

        # each partition is wrapped into a map
        ctr = 0
        for partition in partitions:

            # Host work that touches no array needs no kernel. Read the memlets, not the access
            # nodes: inside a scope a tasklet reaches its arrays through the scope's entry and exit.
            array_access = any(edge.data.data and not is_scalar(edge.data.data, sdfg) for node in partition
                               for edge in state.all_edges(node))
            if not array_access:
                continue

            # reduce partition to nodes which need to go into wrap
            self._remove_all_outer_access_nodes_from_group(state, partition)
            ctr += 1

            # A partition is a dataflow component, and a map scope spans one: covariance hands this
            # a lone MapEntry whose body and exit went to another component.
            partition = self.scope_closed_partition(state, partition, partition_nodes)
            if partition is None:
                continue

            # if anything is left, wrap it
            if partition:
                map_entry, map_exit = self._wrap_region_in_size1_map(state, partition)
                new_maps.add((map_entry, map_exit))

                # Avoid illegal direct map-to-map connections by routing through an access node.
                self._insert_access_between_adjacent_maps(state, map_exit)

                # Ensure all map inputs are also outputs to avoid dace erroneusly labeling them as constants
                self._forward_input_only_map_data(state, map_entry, map_exit)

        return new_maps


## Fix Point Iteration Over Lattice                           ##
# A GPU-scheduled map that writes a variable needs it to be a len-1 ARRAY: a scalar is
# passed by value, so the written value is lost. The rule propagates -- if any input or
# output of a tasklet is GPU-written, every output of that tasklet can be too -- so the
# answer is the fixpoint, compared against the current scalars / len-1 arrays with the
# mismatches converted.

    def decide_length1_array_or_scalar_FPI(self, sdfg: SDFG):
        # 1)
        all_scalars: OrderedSet[str] = OrderedSet(data_name for data_name in sdfg.arrays if is_scalar(data_name, sdfg))
        all_len1arrays: OrderedSet[str] = OrderedSet(data_name for data_name in sdfg.arrays
                                                     if is_length1_array(data_name, sdfg))
        vars: OrderedSet[str] = all_scalars | all_len1arrays

        # 2) with current scheduling heuristic, only toplevel can be GPU
        # 3) tasklets within nested sdfgs are not relevant
        gpu_written: OrderedSet[str] = OrderedSet()
        tasklet_dict: dict = {
            # maps tasklet to (inputs, outputs) where both are sets of data names (array & scalar) accessed as
            # input/output
        }
        for state in sdfg.states():
            for node in state.nodes():

                if isinstance(node, (nodes.MapExit, nodes.LibraryNode)) and has_GPU_schedule(node):
                    outputs = get_data_used_by_outgoing_access_nodes(sdfg, state, node, include_scalars=True)
                    gpu_written |= outputs & vars

                elif isinstance(node, nodes.Tasklet):
                    inputs = get_data_used_by_incoming_access_nodes(sdfg, state, node, include_scalars=True)
                    outputs = get_data_used_by_outgoing_access_nodes(sdfg, state, node, include_scalars=True)
                    tasklet_dict[node] = (inputs, outputs)

        # 4)

        if gpu_written:
            new_gpu_written = gpu_written.copy()

            while True:
                for inputs, outputs in tasklet_dict.values():

                    # at least one in- or output var is written to by gpu
                    if inputs & gpu_written or outputs & gpu_written:
                        new_gpu_written |= outputs  # add all outputs as being potentially written to by gpu

                if new_gpu_written == gpu_written:  # fixpoint reached
                    break
                gpu_written = new_gpu_written.copy()

        # 5)
        to_len1_arrays = all_scalars & gpu_written
        to_scalars = all_len1arrays - gpu_written
        to_scalars = {name
                      for name in to_scalars if not name.startswith("__return")
                      }  # is usually very inefficient because __return if mostly used at the end of the graph

        if to_len1_arrays:
            ConvertScalarsToLengthOneArrays(
                recursive=True,
                preserve_abi=True,
                filter=to_len1_arrays,
            ).apply_pass(sdfg, {})

        if to_scalars:
            ConvertLengthOneArraysToScalars(
                recursive=True,
                preserve_abi=True,
                filter=to_scalars,
            ).apply_pass(sdfg, {})
