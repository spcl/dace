# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Dict, Optional

from ordered_set import OrderedSet

from dace import dtypes, Memlet, subsets
from dace.sdfg import nodes, SDFG
from dace.sdfg.state import SDFGState, ControlFlowBlock, AbstractControlFlowRegion

from dace.transformation.passes.offloading.offloading_ir_node import OffloadingIRNode
import dace.transformation.passes.offloading.offloading_helpers as helpers


class CopyInsertionPhase():

    def apply(self,
              sdfg: SDFG,
              IR: OffloadingIRNode,
              sdfg_scope_dict: Optional[Dict] = None,
              verbose: bool = False) -> None:
        self.verbose = verbose
        if sdfg_scope_dict:
            self.sdfg_scope_dict = sdfg_scope_dict
        else:
            self.sdfg_scope_dict = helpers.get_sdfg_scope_dict(sdfg)
        # Both read before the phase rewrites anything: renaming moves the writes onto the staged
        # twins, and the transient correction below rewrites the very storage that says where a
        # container's home is.
        self.written = helpers.containers_written(sdfg)
        self.home_on_gpu = OrderedSet(name for name in sdfg.arrays if helpers.is_array_stored_on_GPU(sdfg, name))

        self.correct_transient_storage_locations(sdfg, IR)
        self.correct_view_storage_locations(sdfg)
        self.insert_copy_names_in_SDFG(sdfg, IR)
        self.eval_IR(sdfg, IR)
        # Last: only now does every container carry the storage the placement gave it, so only now
        # can an alias be pointed at the same side as the container it aliases.
        self.match_view_storage_to_origin(sdfg)

    def match_view_storage_to_origin(self, sdfg: SDFG) -> None:
        """A view lives where the container it aliases lives.

        The placement decides for containers and renames the accesses that follow it; a view owns no
        storage, so nothing in that pass updates it and it keeps what it was declared with. Left
        alone, a view of a device array is read and written by device code while still declared
        host -- npbench correlation hands cuBLAS three ``Default`` views of ``GPU_Global`` arrays,
        and the call is then emitted into the host translation unit, where the stream it wants does
        not exist ('__dace_current_stream was not declared in this scope').

        A view inside a kernel keeps the Register storage :func:`correct_view_storage_locations`
        gave it: GPU_Global on a view is not handled correctly by the code generator.
        """
        for state in sdfg.states():
            scope = self.sdfg_scope_dict.get(state, {})
            for node in state.data_nodes():
                name = node.data
                if not helpers.is_view(name, sdfg):
                    continue
                if sdfg.arrays[name].storage == dtypes.StorageType.Register:
                    continue

                parent = scope.get(node)
                if isinstance(parent, nodes.MapEntry) and helpers.has_GPU_schedule(parent):
                    continue

                origin = helpers.view_origin(state, node)
                if origin is not None and origin in sdfg.arrays:
                    sdfg.arrays[name].storage = sdfg.arrays[origin].storage

    ################################################################
    ### Ensure Correct Storage Locations Before Inserting Copies ###
    ################################################################

    def correct_transient_storage_locations(self, sdfg: SDFG, IR: OffloadingIRNode) -> None:
        seen = OrderedSet()

        def _correct_transients(node: OffloadingIRNode):
            for name in node.gpu_set:
                assert name in sdfg.arrays
                desc = sdfg.arrays[name]
                if desc.transient and not name in seen:
                    desc.storage = dtypes.StorageType.GPU_Global
                    seen.add(name)

            for name in node.cpu_set:
                assert name in sdfg.arrays
                desc = sdfg.arrays[name]
                if desc.transient and not name in seen:
                    desc.storage = dtypes.StorageType.Default
                    seen.add(name)

        helpers.traverse_IR(IR, _correct_transients)

    def correct_view_storage_locations(self, sdfg: SDFG) -> None:
        state: SDFGState
        for state in sdfg.states():
            scope = self.sdfg_scope_dict[state]
            for node in state.data_nodes():
                data_name = node.data
                parent = scope.get(node)
                if not helpers.is_view(data_name, sdfg):
                    continue

                if isinstance(parent, nodes.MapEntry) and helpers.has_GPU_schedule(parent):
                    # if its within a GPU map, set it to register because GPU_Global isn't handled correctly by code gen
                    sdfg.arrays[data_name].storage = dtypes.StorageType.Register
                    continue

                # A view is an alias, so it lives where the container it aliases lives. Placement
                # decides for containers and leaves the alias with what it was declared with, which
                # is host memory: a kernel writing through the view then writes a host array and the
                # dispatcher answers with an illegal copy (npbench mlp, whose reduce binds _out to
                # the view tmp_max_keepdims).
                origin = helpers.view_origin(state, node)
                if origin is not None and origin in sdfg.arrays:
                    sdfg.arrays[data_name].storage = sdfg.arrays[origin].storage

    #################################
    ### Rename Copied Arrays      ###
    ### A -> A_gpu or A -> A_host ###
    #################################

    ### Renaming Conventions ###
    def _get_host_name(self, name: str) -> str:
        if name.startswith("__return"):
            return f"buffer__return{name[8:]}_host"
        return f"{name}_host"

    def _get_gpu_name(self, name: str) -> str:
        # startswith, not equality: a program returning several values names them __return_0,
        # __return_1, ... and "<name>_gpu" still STARTS with __return, which the runtime refuses as a
        # transient ('Used the special array name "__return_0_gpu" as transient'). The buffer_ prefix
        # is what moves the twin out of the reserved namespace, so it has to cover every __return.
        if name.startswith("__return"):
            return f"buffer__return{name[8:]}_gpu"
        return f"{name}_gpu"

    ###

    def insert_copy_names_in_SDFG(self, sdfg: SDFG, IR: OffloadingIRNode) -> None:
        # make a rename dict for each IR node, then rename all such arrays in the IR.block
        def _insert_copy_names_in_node(node: OffloadingIRNode):
            rename_dict = {}
            for name in node.gpu_set:
                assert name in sdfg.arrays
                if sdfg.arrays[name].storage == dtypes.StorageType.Default:  # starts on CPU, but this access is on GPU
                    rename_dict[name] = self._get_gpu_name(name)

            for name in node.cpu_set:
                assert name in sdfg.arrays
                if sdfg.arrays[
                        name].storage == dtypes.StorageType.GPU_Global:  # starts on GPU, but this access is on CPU
                    rename_dict[name] = self._get_host_name(name)

            self.insert_copy_names_in_block(sdfg, node.block, rename_dict)

        helpers.traverse_IR(IR, _insert_copy_names_in_node)

    def stage_views_with_their_origin(self, sdfg: SDFG, state: SDFGState, rename_dict: Dict[str, str]) -> None:
        """Give a view of a staged container a twin of its own, on the same side.

        A view owns no storage, so the placement decides for the container it aliases and treats an
        access through the view as an access to that container. What follows from staging the
        container is that the alias has to follow it: ``C -> C_gpu`` leaves ``C_0``, a view of ``C``,
        naming a buffer on the other side, and a descriptor carries ONE storage, so the same view
        cannot serve a host state and a kernel at once. npbench mandelbrot2 has exactly that shape --
        ``C_0`` is read by a device map in one state and by host code in another -- and the
        dispatcher answers the losing side with ``Illegal copy! (from C_0 to _numpy_add_)``.

        So each side gets its own alias: ``C -> C_gpu`` and ``C_0 -> C_0_gpu``, the twin viewing the
        twin. Registering the name here is all it takes; the alias is pointed at the staged
        container by the edge that renaming rewrites, and :func:`match_view_storage_to_origin` reads
        the storage back off that container once every placement is final.
        """
        for access in state.data_nodes():
            name = access.data
            if name in rename_dict or not helpers.is_view(name, sdfg):
                continue
            origin = helpers.view_origin(state, access)
            if origin is None or origin not in rename_dict:
                continue

            # Read off the NAME, not the descriptor: the staged container is registered later, by
            # the copy insertion itself, so there is nothing to look up yet. The storage set here is
            # a starting value in any case -- ``match_view_storage_to_origin`` reads it back off the
            # container once every placement is final.
            to_gpu = rename_dict[origin] == self._get_gpu_name(origin)
            twin = self._get_gpu_name(name) if to_gpu else self._get_host_name(name)
            if twin not in sdfg.arrays:
                desc = sdfg.arrays[name]
                sdfg.add_view(twin,
                              desc.shape,
                              desc.dtype,
                              storage=dtypes.StorageType.GPU_Global if to_gpu else dtypes.StorageType.Default,
                              strides=desc.strides,
                              offset=desc.offset)
            rename_dict[name] = twin

    def insert_copy_names_in_state(self, sdfg: SDFG, state: SDFGState, rename_dict: Dict[str, str]) -> None:
        # A view follows the container it aliases, so the names it needs are known only here, once
        # this state's renaming is known.
        self.stage_views_with_their_origin(sdfg, state, rename_dict)

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

    def insert_copy_names_in_block(self, sdfg: SDFG, block: ControlFlowBlock, rename_dict: Dict[str, str]) -> None:
        if block is None: return

        cfr = block.parent_graph
        if cfr and isinstance(cfr, AbstractControlFlowRegion):
            for edge in cfr.in_edges(block):
                relevant_edge_arrays = edge.data.used_arrays(rename_dict)
                """
                A begins on CPU
                edge accesses A, which is on CPU at that time -> don't copy, don't rename (A)
                edge accesses A, which is on GPU at that time -> do    copy, don't rename (A)

                A begins on GPU
                edge accesses A, which is on CPU at that time -> don't copy,    do rename (A_host)
                edge accesses A, which is on GPU at that time -> do    copy,    do rename (A_host)

                -> copies are handled later, hence here the arrays are renamed iff they begin on GPU
                """
                for name in relevant_edge_arrays:
                    if sdfg.arrays[name].storage == dtypes.StorageType.GPU_Global:
                        edge.data.replace(name, self._get_host_name(name))

        if isinstance(block, SDFGState):
            self.insert_copy_names_in_state(sdfg, block, rename_dict)

        elif isinstance(block, ControlFlowBlock):
            # rename meta accesses (control-flow metadata like loop bounds or conditions)
            block.replace_meta_accesses(rename_dict)
            # NOTE: states / blocks within the current block all have their own IRNodes and don't need to be handled recursively
        else:
            raise NotImplementedError(
                f"in _correct_names_in_block: IR.block unhandled type: {block} is {block.__class__.__name__}")

    ######################################################
    ### Evaluate the IR to Find Copy Locations in SDFG ###
    ######################################################

    def eval_IR(self, sdfg: SDFG, IR: OffloadingIRNode) -> None:
        # modifies SDFG in place & inserts all necessary copies

        def eval(node: OffloadingIRNode) -> None:
            # loop copies if applicable
            if node.type == OffloadingIRNode.CLOSE and node.open and node.open.type == OffloadingIRNode.OPEN_LOOP:  # CLOSE LOOP
                top: OffloadingIRNode = node.open
                bottom: OffloadingIRNode = node
                tails = OffloadingIRNode.get_all_tails(top)  # INV: all are STATE or CLOSE if there's a nested loop

                gpu_copies = bottom.cpu_set & top.gpu_set
                if gpu_copies:
                    if self.verbose:
                        print(f"Phase 6: LOOP GPU copy for {gpu_copies} at end of interation of loop {node.debug_name}")

                    for tail in tails:
                        if tail.type == OffloadingIRNode.CLOSE:
                            self.create_interstate_copy(sdfg, tail.open.block, None, gpu_copies, to_gpu=True)
                        else:
                            self.create_interstate_copy(sdfg, tail.block, None, gpu_copies, to_gpu=True)

                cpu_copies = bottom.gpu_set & top.cpu_set
                if cpu_copies:
                    if self.verbose:
                        print(f"Phase 6: LOOP CPU copy for {cpu_copies} at end of iteration of loop {node.debug_name}")

                    for tail in tails:
                        if tail.type == OffloadingIRNode.CLOSE:
                            self.create_interstate_copy(sdfg, tail.open.block, None, cpu_copies, to_gpu=False)
                        else:
                            self.create_interstate_copy(sdfg, tail.block, None, cpu_copies, to_gpu=False)

                # copies added at end of loop state, within loop -> modify IR of LOOP_CLOSE to represent that
                node.gpu_set = (node.gpu_set | gpu_copies) - cpu_copies
                node.cpu_set = (node.cpu_set | cpu_copies) - gpu_copies

            for next in node.next:

                if node.cpu_set & node.gpu_set:
                    raise NotImplementedError(
                        f"This pass does not support copies within a single state. State {node.debug_name} uses arrays {node.cpu_set & node.gpu_set} on both cpu and gpu."
                    )

                # edge case: if this condition is true, both blocks are None, can't insert
                if node.type == OffloadingIRNode.CLOSE and next.type == OffloadingIRNode.CLOSE:
                    self.insert_copies(sdfg, node, next, node.open.block, None)

                elif next.type == OffloadingIRNode.EDGE:  # then I want the copy AFTER the node, not before
                    self.insert_copies(sdfg, node, next, node.block, None)

                else:  # the usual: copies between node -> next
                    self.insert_copies(sdfg, node, next, node.block, next.block)

        helpers.traverse_IR(IR, eval)

    ########################################
    ### Insert New Copy States into SDFG ###
    ########################################

    def insert_copies(self, sdfg: SDFG, node: OffloadingIRNode, next: OffloadingIRNode, node_block: ControlFlowBlock,
                      next_block: ControlFlowBlock) -> None:
        gpu_copies = node.cpu_set & next.gpu_set
        if gpu_copies:
            if self.verbose:
                print(f"Phase 6: GPU copy for {gpu_copies} between {node.debug_name} and {next.debug_name}")
            self.create_interstate_copy(sdfg, node_block, next_block, gpu_copies, to_gpu=True)

        cpu_copies = node.gpu_set & next.cpu_set
        if cpu_copies:
            if self.verbose:
                print(f"Phase 6: CPU copy for {cpu_copies} between {node.debug_name} and {next.debug_name}")
            self.create_interstate_copy(sdfg, node_block, next_block, cpu_copies, to_gpu=False)

    def create_interstate_copy(self, sdfg: SDFG, state1: ControlFlowBlock, state2: ControlFlowBlock,
                               array_names: OrderedSet, to_gpu: bool) -> None:
        assert state1 is not None or state2 is not None, "invalid: both states are None"

        array_names = OrderedSet(name for name in array_names if not self.copy_is_redundant(name, to_gpu))
        if not array_names:
            return

        # 1) insert new state
        copy_state: SDFGState
        joined = "_".join(sorted(array_names))
        label = f"copy_{joined}_{'to_gpu' if to_gpu else 'to_host'}"

        if state2 is not None:
            if self.verbose: print("Phase 6: copy placed before", state2)
            target_graph = state2.parent_graph
            assert target_graph is not None, "copy insertion requires a parent control-flow graph (s2)"

            copy_state = target_graph.add_state_before(state2, label=label)
            if state2 is target_graph.start_block:
                target_graph.start_block = target_graph.node_id(copy_state)  # copy state becomes new start block

        elif state1 is not None:
            if self.verbose: print("Phase 6: copy placed after", state2)
            target_graph = state1.parent_graph if state1.parent_graph else state1
            assert target_graph is not None, "copy insertion requires a parent control-flow graph (s1)"
            copy_state = target_graph.add_state_after(state1, label=label)

        # 2) create the copy map with correct names
        copy_map = {}
        name: str
        for name in array_names:
            assert name in sdfg.arrays

            if helpers.is_array_stored_on_GPU(sdfg, name):  # original array is on GPU
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
            if not new_name in sdfg.arrays:
                self._register_new_copy_transient(sdfg, new_name, old_name)
            elif not old_name in sdfg.arrays:
                self._register_new_copy_transient(
                    sdfg, old_name, new_name
                )  # in some cases, e.g. loops, a copy-from can be registered before its copy-to, leading to an unknown "old_name"

            # b) add (Access Node -> Access Node) to state
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

    def copy_is_redundant(self, name: str, to_gpu: bool) -> bool:
        """A copy back into the container's own home carries nothing new when nothing wrote it.

        The container never moves -- the placement stages a twin on the other side and points the
        accesses there -- so the home copy is only stale once something has written the twin. For a
        container the SDFG never writes it never goes stale, and restoring it copies bytes that are
        already in place. Not merely wasted: the copy WRITES the container, which validation refuses
        as soon as this SDFG is nested and the container reaches it through an input connector only
        ("Data descriptor A is written to, but only given to nested SDFG as an input connector"),
        the graph ``GPUTransformMap`` builds around a map that reads one array and writes another.

        Only the direction that restores the home is dropped. Filling the twin is what makes the
        other side readable at all, so it stands whether or not anything writes it.
        """
        restores_the_home = (name in self.home_on_gpu) == to_gpu
        return restores_the_home and name not in self.written

    def _register_new_copy_transient(self, sdfg: SDFG, unknown_name: str, known_name: str):
        assert known_name in sdfg.arrays
        desc = sdfg.arrays[known_name]

        new_storage = dtypes.StorageType.Default if helpers.is_array_stored_on_GPU(
            sdfg, known_name) else dtypes.StorageType.GPU_Global
        if helpers.is_view(known_name, sdfg):
            sdfg.add_view(unknown_name, desc.shape, desc.dtype, storage=new_storage)
        else:
            sdfg.add_array(unknown_name, desc.shape, desc.dtype, storage=new_storage, transient=True)
