# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from dace import SDFG, InterstateEdge
from dace.sdfg import nodes as nd
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes import analysis as ap


@transformation.explicit_cf_compatible
class ScalarFission(ppl.Pass):
    """
    Fission transient scalars or arrays of size 1 that are dominated by a write into separate data containers.
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.AccessNodes

    def depends_on(self):
        return {ap.ScalarWriteShadowScopes}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        """
        Rename scalars and arrays of size 1 based on dominated scopes.

        :param sdfg: The SDFG to modify.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: A dictionary mapping the original name to a set of all new names created for each data container.
        """
        results: Dict[str, Set[str]] = defaultdict(lambda: set())

        shadow_scope_dict: ap.WriteScopeDict = pipeline_results[ap.ScalarWriteShadowScopes.__name__][sdfg.cfg_id]

        for name, write_scope_dict in shadow_scope_dict.items():
            desc = sdfg.arrays[name]

            # If this isn't a scalar or an array of size 1, don't do anything.
            if desc.total_size != 1:
                continue

            # Don't rename anything that's not transient, as it may be used externally.
            if not desc.transient:
                continue

            # Privatize the undominated (``None``) scope -- accesses whose reads are
            # not dominated by a single write, e.g. a scalar written in every
            # branch of a conditional and read after the merge (the cloudsc
            # zcor/zfac/zqe pattern). These are loop-local (no upward-exposed use),
            # so each enclosing loop gets its own copy -- scalar privatization.
            if None in write_scope_dict:
                self._privatize_loop_local_undominated(sdfg, name, write_scope_dict[None], results)

            # If there is only one (dominating) scope, no further fission is needed.
            if len([w for w in write_scope_dict if w is not None]) <= 1:
                continue

            for write, shadowed_reads in write_scope_dict.items():
                if write is not None and len(shadowed_reads) > 0:
                    newdesc = desc.clone()
                    newname = sdfg.add_datadesc(name, newdesc, find_new_name=True)

                    # Replace the write and any connected memlets with writes to the new data container.
                    write_node = write[1]
                    write_node.data = newname
                    for iedge in write[0].in_edges(write_node):
                        if iedge.data.data == name:
                            iedge.data.data = newname
                    for oeade in write[0].out_edges(write_node):
                        if oeade.data.data == name:
                            oeade.data.data = newname

                    # Replace all dominated reads and connected memlets.
                    affected_states: Set[SDFGState] = {write[0]} if isinstance(write[0], SDFGState) else set()
                    for read in shadowed_reads:
                        if isinstance(read[1], nd.AccessNode):
                            read_node = read[1]
                            read_node.data = newname
                            for iedge in read[0].in_edges(read_node):
                                if iedge.data.data == name:
                                    iedge.data.data = newname
                            for oeade in read[0].out_edges(read_node):
                                if oeade.data.data == name:
                                    oeade.data.data = newname
                            if isinstance(read[0], SDFGState):
                                affected_states.add(read[0])
                        elif isinstance(read[1], InterstateEdge):
                            read[1].replace_dict({name: newname})

                    # Propagate the rename across every NestedSDFG boundary
                    # touched by the renamed accesses: connector names, inner
                    # arrays catalog, inner accesses + memlets, symbol_mapping.
                    self._propagate_rename_into_nsdfgs(affected_states, name, newname)
                    results[name].add(newname)
        return results

    def report(self, pass_retval: Any) -> Optional[str]:
        return f'Renamed {len(pass_retval)} scalars: {pass_retval}.'

    # ------------------------------------------------------------------ #
    #  Privatization of undominated (None-scope) loop-local scalars
    # ------------------------------------------------------------------ #

    def _privatize_loop_local_undominated(self, sdfg: SDFG, name: str, accesses: Set[Tuple], results):
        """Give a separate container to each loop's copy of a scalar whose reads
        are not dominated by a single write (the ``None`` write-scope), when it
        is provably loop-local. This is scalar privatization; it is legal only if
        the scalar has **no upward-exposed use** in that loop (it is definitely
        written before any read on every path -- e.g. written in all branches of
        a conditional, read after the merge). A scalar that may be read before it
        is written (a non-exhaustive ``if``) is loop-carried and is left alone.

        :param sdfg: The SDFG being modified.
        :param name: The size-1 transient scalar.
        :param accesses: The ``None``-scope ``(block, node-or-edge)`` accesses.
        :param results: Accumulator mapping the original name to new names.
        """
        by_loop: Dict[LoopRegion, List[Tuple]] = defaultdict(list)
        outside_loop = False
        for block, node in accesses:
            loop = self._innermost_loop(block)
            if loop is None:
                outside_loop = True
                break
            by_loop[loop].append((block, node))
        if outside_loop:
            # An undominated access at non-loop scope is not loop-local; leave the
            # whole scalar alone rather than split a value across scopes.
            return

        for loop, loop_accesses in by_loop.items():
            # Only privatize if every undominated access of ``name`` in this loop
            # is in this group (don't split a single value) and the scalar has no
            # upward-exposed use in the loop (privatization legality).
            if not self._no_upward_exposed_use(loop, name, defined_on_entry=False):
                continue
            newname = sdfg.add_datadesc(name, sdfg.arrays[name].clone(), find_new_name=True)
            affected_states: Set[SDFGState] = set()
            for block, node in loop_accesses:
                if isinstance(node, nd.AccessNode):
                    node.data = newname
                    for e in block.in_edges(node):
                        if e.data.data == name:
                            e.data.data = newname
                    for e in block.out_edges(node):
                        if e.data.data == name:
                            e.data.data = newname
                    if isinstance(block, SDFGState):
                        affected_states.add(block)
                elif isinstance(node, InterstateEdge):
                    node.replace_dict({name: newname})
            self._propagate_rename_into_nsdfgs(affected_states, name, newname)
            results[name].add(newname)

    @staticmethod
    def _propagate_rename_into_nsdfgs(states, old_name: str, new_name: str) -> None:
        """Propagate a scalar rename across every ``NestedSDFG`` boundary
        whose input/output connector matches ``old_name``.

        The outer-side rename (``AccessNode.data``, surrounding memlets) is
        not enough when the renamed scalar crosses into a ``NestedSDFG``:
        the inner SDFG references the scalar by connector name, which
        binds to its own arrays catalog entry of the same name. Without
        propagation the inner SDFG ends up dangling (inner descriptor
        keyed on the old name, outer side rewired to the new name).

        For every NSDFG in ``states`` that has a connector named
        ``old_name``, this helper:

        * renames the IN / OUT connector ``old_name -> new_name``
        * updates the connecting outer memlet's ``data`` if it still
          references ``old_name``
        * runs ``SDFG.replace_dict({old_name: new_name})`` on the inner
          SDFG, which renames the inner arrays catalog entry, every
          inner ``AccessNode.data``, every inner memlet, and any
          symbol references including ``symbol_mapping`` values
        * updates the ``symbol_mapping`` itself if ``old_name`` appears
          as a KEY (the inner symbol name binding)

        :param states: The set of outer ``SDFGState`` instances where
                       the rename happened. Only NSDFGs reachable from
                       these states get the propagation.
        :param old_name: The original scalar name.
        :param new_name: The renamed scalar name.
        """
        if not states or old_name == new_name:
            return
        for state in states:
            for n in state.nodes():
                if not isinstance(n, nd.NestedSDFG):
                    continue
                in_match = old_name in n.in_connectors
                out_match = old_name in n.out_connectors
                if not (in_match or out_match or old_name in n.symbol_mapping):
                    continue
                # 1. Rename the connector.
                if in_match:
                    n.in_connectors[new_name] = n.in_connectors.pop(old_name)
                if out_match:
                    n.out_connectors[new_name] = n.out_connectors.pop(old_name)
                # 2. Update the outer-side memlet on edges that connect to
                #    the renamed connector.
                for e in state.in_edges(n):
                    if e.dst_conn == old_name:
                        e.dst_conn = new_name
                    if e.data is not None and e.data.data == old_name:
                        e.data.data = new_name
                for e in state.out_edges(n):
                    if e.src_conn == old_name:
                        e.src_conn = new_name
                    if e.data is not None and e.data.data == old_name:
                        e.data.data = new_name
                # 3. Propagate INSIDE the NestedSDFG. We do this manually
                #    (no ``SDFG.replace_dict``) to avoid its symbolic-
                #    replacement pathway tripping on the scalar's free-
                #    symbol typeclass coercion.
                inner = n.sdfg
                if old_name in inner.arrays:
                    inner.arrays[new_name] = inner.arrays.pop(old_name)
                    for inner_state in inner.states():
                        for inn in inner_state.data_nodes():
                            if inn.data == old_name:
                                inn.data = new_name
                        for ie in inner_state.edges():
                            if ie.data is not None and ie.data.data == old_name:
                                ie.data.data = new_name
                # Interstate edges in the inner SDFG (rare for a scalar-
                # carrier shape but cheap to cover).
                for ise in inner.all_interstate_edges():
                    if any(str(s) == old_name for s in ise.data.free_symbols):
                        ise.data.replace_dict({old_name: new_name})
                # 4. Symbol_mapping uses inner-side symbol names as keys
                #    (when the scalar appears as a symbol there). If the
                #    inner symbol was named ``old_name``, update the key.
                if old_name in n.symbol_mapping:
                    n.symbol_mapping[new_name] = n.symbol_mapping.pop(old_name)
                # Recurse: inner NestedSDFGs may themselves carry the
                #    scalar across another boundary.
                ScalarFission._propagate_rename_into_nsdfgs(set(inner.states()), old_name, new_name)

    @staticmethod
    def _innermost_loop(block) -> Optional[LoopRegion]:
        """Find the innermost loop enclosing a block.

        :param block: The control-flow block to search from.
        :returns: The innermost enclosing ``LoopRegion``, or ``None`` if the block
                  is not inside any loop.
        """
        region = block.parent_graph
        while region is not None:
            if isinstance(region, LoopRegion):
                return region
            region = region.parent_graph
        return None

    def _no_upward_exposed_use(self, region: ControlFlowRegion, name: str, defined_on_entry: bool) -> bool:
        """Test scalar-privatization legality: ``name`` has no upward-exposed use
        in ``region``, i.e. every read is preceded by a write on every path
        (``name`` is definitely assigned before use). Path-insensitive must-def
        analysis -- a non-exhaustive conditional does not establish a definition.

        :param region: The region (typically a loop) to test.
        :param name: The scalar data container.
        :param defined_on_entry: Whether ``name`` is already defined on entry.
        :returns: ``True`` if ``name`` has no upward-exposed use (privatizable).
        """
        return self._analyze_region(region, name, defined_on_entry)[1]

    def _analyze_region(self, region: ControlFlowRegion, name: str, defined_on_entry: bool) -> Tuple[bool, bool]:
        """Forward must-def analysis of ``name`` over ``region``.

        The fixpoint is seeded pessimistically (``False`` off the entry), so
        cycles and unanalyzable shapes stay conservative -- a possibly-carried
        scalar is never reported as privatizable.

        :param region: The region to analyze.
        :param name: The scalar data container.
        :param defined_on_entry: Whether ``name`` is defined on entry to ``region``.
        :returns: ``(definitely_defined_at_exit, no_upward_exposed_use)``.
        """
        blocks = list(region.nodes())
        if not blocks:
            return defined_on_entry, True
        bdef = {b: self._block_defines(b, name) for b in blocks}
        start = region.start_block
        defn = {b: False for b in blocks}
        for _ in range(len(blocks) + 1):
            changed = False
            for b in blocks:
                if b is start:
                    nv = defined_on_entry
                else:
                    preds = [e.src for e in region.in_edges(b)]
                    nv = all(defn[p] or bdef[p] for p in preds) if preds else False
                if nv != defn[b]:
                    defn[b] = nv
                    changed = True
            if not changed:
                break
        no_ue = all(not self._block_ue(b, name, defn[b]) for b in blocks)
        sinks = [b for b in blocks if region.out_degree(b) == 0]
        must_def_exit = bool(sinks) and all(defn[b] or bdef[b] for b in sinks)
        return must_def_exit, no_ue

    def _block_defines(self, block, name: str) -> bool:
        """Whether ``block`` definitely writes ``name`` on every path through it.
        A conditional defines only if it is exhaustive (has an ``else``) and every
        branch defines; a loop may run zero times and so never definitely defines.

        :param block: The control-flow block.
        :param name: The scalar data container.
        :returns: ``True`` if ``name`` is written on every path through ``block``.
        """
        if isinstance(block, SDFGState):
            return self._state_defines(block, name)
        if isinstance(block, ConditionalBlock):
            has_else = any(cond is None for cond, _ in block.branches)
            return has_else and all(self._analyze_region(br, name, False)[0] for _, br in block.branches)
        if isinstance(block, LoopRegion):
            return False
        if isinstance(block, ControlFlowRegion):
            return self._analyze_region(block, name, False)[0]
        return False

    def _block_ue(self, block, name: str, defined_on_entry: bool) -> bool:
        """Whether ``block`` has an upward-exposed read of ``name``.

        :param block: The control-flow block.
        :param name: The scalar data container.
        :param defined_on_entry: Whether ``name`` is already defined on entry.
        :returns: ``True`` if a read of ``name`` may execute before any write.
        """
        if defined_on_entry:
            return False
        if isinstance(block, SDFGState):
            return self._state_reads_before_write(block, name)
        if isinstance(block, ConditionalBlock):
            return any(not self._analyze_region(br, name, False)[1] for _, br in block.branches)
        if isinstance(block, ControlFlowRegion):
            return not self._analyze_region(block, name, False)[1]
        return True  # unknown block type -> conservative

    @staticmethod
    def _state_defines(state: SDFGState, name: str) -> bool:
        """Whether ``state`` writes ``name`` (has a non-empty write to it).

        :param state: The state to inspect.
        :param name: The scalar data container.
        :returns: ``True`` if ``name`` is written in ``state``.
        """
        return any(n.data == name and any(not e.data.is_empty() for e in state.in_edges(n)) for n in state.data_nodes())

    @staticmethod
    def _state_reads_before_write(state: SDFGState, name: str) -> bool:
        """Whether ``state`` has an upward-exposed read of ``name``: an access
        node that is read (has out-edges) but not written in the state (no
        non-empty in-edge) reads the value coming from before the state.

        :param state: The state to inspect.
        :param name: The scalar data container.
        :returns: ``True`` if a read precedes any write of ``name`` in ``state``.
        """
        for node in state.data_nodes():
            if node.data != name:
                continue
            written = any(not e.data.is_empty() for e in state.in_edges(node))
            if state.out_degree(node) > 0 and not written:
                return True
        return False


@transformation.explicit_cf_compatible
class PrivatizeScalars(ppl.Pipeline):
    """Give every write-before-read transient scalar its own name (scalar privatization).

    A self-contained pipeline that runs :class:`ScalarFission` together with the
    analysis it depends on (``ScalarWriteShadowScopes`` and that pass's own
    dependencies), so it can be applied on its own --
    ``PrivatizeScalars().apply_pass(sdfg, {})`` -- or dropped in as a single
    element of a larger pipeline such as simplify. ``ScalarFission`` declares
    those dependencies but cannot resolve them outside a pipeline.

    A size-1 transient reused as a per-iteration temporary is written before it
    is read on every iteration, so the iterations share only its *name*, not a
    real value. Splitting each dominating write into its own container removes
    that false write-after-write, which is what otherwise makes a shared
    loop-local scalar block ``LoopToMap``.
    """

    def __init__(self):
        super().__init__([ScalarFission()])
