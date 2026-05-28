# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Rewrite a same-write-set ``if/else`` into compute-then/compute-else/merge CFGs.

The two arms become sequential states producing ``_then_<arr>`` /
``_else_<arr>`` temporaries; a final state folds them with the symbolic
``merge`` (see :mod:`dace.runtime.include.dace.merge`) which the
vectorizer lowers to a SIMD blend. Only handles a two-branch
``if/else`` with single-state arms whose shared writes are matching
element subsets and whose bodies are tasklets/access nodes; anything
else raises :class:`NotImplementedError`.
"""
import re
from typing import Optional, Tuple

import dace
from dace import properties, symbolic
from dace.properties import CodeBlock
from dace.sdfg.construction_utils import (
    assert_connector_role_matches_edges,
    copy_state_contents,
)
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation import pass_pipeline as ppl


def _symbol_has_external_consumer(sdfg: dace.SDFG, sym_name: str, defining_edge, skip_cb=None) -> bool:
    """Whether ``sym_name`` is consumed outside its own defining edge.

    Decides if the lift can delete the assignment and drop the symbol.
    Walks interstate assignment RHSs/conditions, ConditionalBlock branch
    conditions, LoopRegion init/cond/update, Tasklet code, and parent
    NestedSDFG ``symbol_mapping``.

    :param sdfg: SDFG to scan.
    :param sym_name: symbol being lifted.
    :param defining_edge: the interstate edge that assigns ``sym_name``
        (its own lhs use does not count).
    :param skip_cb: the ``ConditionalBlock`` being rewritten; its branch
        conditions are the consumer being rewired away, so they do not
        count as external.
    :returns: ``True`` if an external consumer remains.
    """
    from dace.sdfg.state import LoopRegion as _LoopRegion
    from dace.sdfg.state import ConditionalBlock as _ConditionalBlock

    only = {sym_name}
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        for e in cfg.edges():
            for lhs, rhs in (e.data.assignments or {}).items():
                # Skip the about-to-be-deleted assignment of sym_name itself.
                if e is defining_edge and lhs == sym_name:
                    continue
                if symbolic.symbols_in_code(str(rhs), potential_symbols=only):
                    return True
            if e.data.condition is not None:
                cond_text = e.data.condition.as_string
                if symbolic.symbols_in_code(cond_text, potential_symbols=only):
                    return True

    for block in sdfg.all_control_flow_blocks():
        if isinstance(block, _ConditionalBlock):
            if block is skip_cb:
                continue
            for c, _ in block.branches:
                if c is None:
                    continue
                text = c.as_string if isinstance(c, CodeBlock) else str(c)
                if symbolic.symbols_in_code(text, potential_symbols=only):
                    return True

    for region in sdfg.all_control_flow_regions(recursive=True):
        if isinstance(region, _LoopRegion):
            for attr in ("loop_condition", "update_statement", "init_statement"):
                code = getattr(region, attr, None)
                if code is None:
                    continue
                text = code.as_string if isinstance(code, CodeBlock) else str(code)
                if symbolic.symbols_in_code(text, potential_symbols=only):
                    return True

    for state in sdfg.all_states():
        for n in state.nodes():
            if isinstance(n, dace.nodes.Tasklet):
                code = n.code.as_string if isinstance(n.code, CodeBlock) else str(n.code)
                if symbolic.symbols_in_code(code, potential_symbols=only):
                    return True

    if sdfg.parent_nsdfg_node is not None:
        for k, v in sdfg.parent_nsdfg_node.symbol_mapping.items():
            if k == sym_name:
                continue
            if symbolic.symbols_in_code(str(v), potential_symbols=only):
                return True

    return False


@properties.make_properties
class SameWriteSetIfElseToMergeCFG(ppl.Pass):
    """Rewrite same-write-set ``if/else`` blocks into 3-CFG merge form.

    See module docstring for the pass contract.
    """

    CATEGORY: str = "Vectorization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Rewrite every matching same-write-set block.

        :param sdfg: SDFG to transform in place.
        :returns: number of blocks rewritten, or ``None`` if none.
        """
        rewritten = 0
        for cfg in list(sdfg.all_control_flow_regions(recursive=True)):
            for block in list(cfg.nodes()):
                if not isinstance(block, ConditionalBlock):
                    continue
                if not self._matches(block):
                    continue
                self._rewrite(sdfg, block)
                rewritten += 1
        return rewritten or None

    def _matches(self, cb: ConditionalBlock) -> bool:
        """Whether ``cb`` is a rewritable conditional block.

        Two variants match:

        - **Two-arm** ``if/else``: at least one shared element write across
          both single-state arms (the canonical same-write-set case).
        - **Single-arm** ``if`` (no ``else``): the lone arm IS the
          shared-write set (the absent else reads the pre-cb value of the
          target via the merge tasklet's ``else_op = arr``), so any
          element-write arm matches.

        :param cb: candidate conditional block.
        :returns: ``True`` if the block matches one of the variants above.
        """
        if len(cb.branches) == 2:
            (cond0, body0), (cond1, body1) = cb.branches
            if cond0 is None or cond1 is not None:
                return False
            if not (isinstance(body0, ControlFlowRegion) and isinstance(body1, ControlFlowRegion)):
                return False
            if len(body0.nodes()) != 1 or len(body1.nodes()) != 1:
                return False
            s0, s1 = body0.nodes()[0], body1.nodes()[0]
            if not (isinstance(s0, dace.SDFGState) and isinstance(s1, dace.SDFGState)):
                return False
            for s in (s0, s1):
                for n in s.nodes():
                    if isinstance(n, (dace.nodes.AccessNode, dace.nodes.Tasklet)):
                        continue
                    return False
            shared = self._shared_writes(s0, s1)
            return bool(shared)
        if len(cb.branches) == 1:
            (cond0, body0), = cb.branches
            if cond0 is None:
                return False
            if not isinstance(body0, ControlFlowRegion):
                return False
            if len(body0.nodes()) != 1:
                return False
            s0 = body0.nodes()[0]
            if not isinstance(s0, dace.SDFGState):
                return False
            for n in s0.nodes():
                if isinstance(n, (dace.nodes.AccessNode, dace.nodes.Tasklet)):
                    continue
                return False
            try:
                w0 = self._collect_write_subsets(s0)
            except NotImplementedError:
                return False
            return bool(w0)
        return False

    def _collect_write_subsets(self, state: dace.SDFGState):
        """Element-wise writes of a state.

        :param state: arm state to inspect.
        :returns: ``{arr_name: subset}`` for every element-wise write.
        :raises NotImplementedError: if any write is not element-wise
            (the merge rewrite cannot produce per-element tasklets then;
            ``_shared_writes`` swallows this and returns ``{}``).
        """
        from dace.transformation.passes.vectorization.utils.queries import collect_element_write_subsets
        out = collect_element_write_subsets(state)
        if out is None:
            raise NotImplementedError(
                f"SameWriteSetIfElseToMergeCFG: non-element write subset found in state {state.label}")
        return out

    def _shared_writes(self, s0: dace.SDFGState, s1: dace.SDFGState) -> dict:
        """Shared element writes of both arms.

        :param s0: the ``if`` arm state.
        :param s1: the ``else`` arm state.
        :returns: ``{arr_name: subset}`` for arrays written in both arms
            with identical subsets; empty if either arm has a
            non-element-wise write (treated by the caller as a non-match).
        """
        try:
            w0 = self._collect_write_subsets(s0)
            w1 = self._collect_write_subsets(s1)
        except NotImplementedError:
            return {}
        shared = {}
        for k, v in w0.items():
            if k in w1 and str(v) == str(w1[k]):
                shared[k] = v
        return shared

    def _rewrite(self, sdfg: dace.SDFG, cb: ConditionalBlock):
        """Replace ``cb`` with compute-then / compute-else / apply-merge states.

        Clones each arm (redirecting escaping writes to ``_then_<arr>`` /
        ``_else_<arr>`` transients) and emits one ``merge`` tasklet per
        escaping target; arm-local writes stay inline. The single-arm case
        (``if c: ...`` with no ``else``) lowers the same way — the merge
        tasklet's ``else_op`` reads the pre-cb value of the target (``arr``
        itself), so an absent else arm reduces to a per-lane RMW select.

        :param sdfg: SDFG used for name resolution.
        :param cb: the conditional block to rewrite (removed in place).
        """
        parent = cb.parent_graph
        single_arm = (len(cb.branches) == 1)
        if single_arm:
            (cond_block, then_body), = cb.branches
            else_body = None
        else:
            (cond_block, then_body), (_, else_body) = cb.branches
        then_state: dace.SDFGState = then_body.nodes()[0]
        else_state: Optional[dace.SDFGState] = else_body.nodes()[0] if else_body is not None else None
        # ``cb`` may live inside a NestedSDFG; arrays referenced from its arms
        # live on the immediate enclosing SDFG, not the outermost ``sdfg``.
        local_sdfg: dace.SDFG = cb.sdfg

        # Per-arm escape set drives temp allocation and the clone-redirect.
        # An arm-local intermediate that nothing outside reads stays inline.
        from dace.transformation.passes.vectorization.branch_normalization import (  # avoid import cycle
            compute_arm_escape_writes, )
        escape_plan = compute_arm_escape_writes(local_sdfg, cb)
        all_escapes = escape_plan.get(0, set()) | escape_plan.get(1, set())
        if not all_escapes:
            return

        # We need the writing arm's subset to size the merge memlet, and we
        # only allocate ``_<arm>_<arr>`` for arms that actually write ``arr``;
        # the other arm's merge operand is the pre-cb value of ``arr``.
        then_writes = self._collect_write_subsets(then_state)
        else_writes = self._collect_write_subsets(else_state) if else_state is not None else {}

        def _alloc(prefix: str, arr_name: str) -> str:
            base = local_sdfg.arrays[arr_name]
            name, _ = local_sdfg.add_array(name=f"{prefix}_{arr_name}",
                                           shape=base.shape,
                                           dtype=base.dtype,
                                           storage=dace.dtypes.StorageType.Register,
                                           transient=True,
                                           find_new_name=True)
            return name

        temp_then = {arr: _alloc("_then", arr) for arr in sorted(all_escapes) if arr in then_writes}
        temp_else = {arr: _alloc("_else", arr) for arr in sorted(all_escapes) if arr in else_writes}

        # Subset per target: arms must agree when both write it (an
        # element-write convention M3.1b already enforces upstream).
        write_subsets = {}
        for arr in all_escapes:
            t, e = then_writes.get(arr), else_writes.get(arr)
            if t is not None and e is not None and str(t) != str(e):
                raise NotImplementedError(
                    f"SameWriteSetIfElseToMergeCFG: arms write {arr!r} with different subsets ({t} vs {e})")
            write_subsets[arr] = t if t is not None else e

        # New 3-CFG states in the parent graph. The compute-else state is
        # emitted as an empty pass-through for single-arm conditionals (no
        # else body to clone); the apply-merge state then reads the pre-cb
        # value of every target via the ``else_op = arr`` fallback.
        ct_state = parent.add_state(f"compute_then_{cb.label}")
        ce_state = parent.add_state(f"compute_else_{cb.label}")
        am_state = parent.add_state(f"apply_merge_{cb.label}")

        # Clone bodies redirecting only the per-arm escape writes.
        self._clone_with_redirect(then_state, ct_state, temp_then)
        if else_state is not None:
            self._clone_with_redirect(else_state, ce_state, temp_else)

        # Merge tasklets. A non-writing arm contributes the pre-cb value
        # (read the original ``arr``, which is intact because the writing
        # arm now targets its private temp).
        # Resolve cond once so the symbol-lifting side effect (deleting
        # the upstream assignment) fires exactly once even when multiple
        # writes share the same cond.
        cond_text = cond_block.as_string if isinstance(cond_block, CodeBlock) else str(cond_block)
        any_subset_str = str(next(iter(write_subsets.values())))
        resolved = self._resolve_cond_to_array(local_sdfg, am_state, cond_text, any_subset_str, skip_cb=cb)
        if resolved is None:
            cond_array_name, cond_producer = None, None
        else:
            cond_array_name, cond_producer = resolved
        for arr, subset in write_subsets.items():
            then_op = temp_then.get(arr, arr)
            else_op = temp_else.get(arr, arr)
            self._emit_merge_tasklet(local_sdfg,
                                     am_state,
                                     arr,
                                     subset,
                                     then_op,
                                     else_op,
                                     cond_text,
                                     cond_array_name=cond_array_name,
                                     cond_producer=cond_producer)

        # Stitch in/out edges of the ConditionalBlock onto ct_state -> ce_state
        # -> am_state, then drop the original block.
        in_edges = list(parent.in_edges(cb))
        out_edges = list(parent.out_edges(cb))
        was_start = (parent.start_block is cb)
        for e in in_edges + out_edges:
            parent.remove_edge(e)
        parent.remove_node(cb)
        for e in in_edges:
            parent.add_edge(e.src, ct_state, e.data)
        parent.add_edge(ct_state, ce_state, dace.InterstateEdge())
        parent.add_edge(ce_state, am_state, dace.InterstateEdge())
        for e in out_edges:
            parent.add_edge(am_state, e.dst, e.data)
        if was_start:
            parent.start_block = parent.node_id(ct_state)

        # End-of-pass invariant: every emitted state has well-formed connectors.
        for s in (ct_state, ce_state, am_state):
            assert_connector_role_matches_edges(s)

    def _clone_with_redirect(self, src: dace.SDFGState, dst: dace.SDFGState, rename: dict):
        """Deep-copy ``src`` into ``dst``; redirect only the *write* of each
        renamed array to its private temp.

        Only access nodes that are written in the arm (in-degree > 0 in the
        clone) are retargeted to ``rename[old]``; read-only access nodes
        keep the original name. This is load-bearing for read-modify-write
        arms: ``a[i] = a[i] + b[i]*d[i]`` must clone as
        ``_then_a = a + b*d`` (RHS reads the *original* ``a``), not
        ``_then_a = _then_a + b*d`` (RHS would read the uninitialised temp
        and propagate garbage through the merge — TSVC s2710). Memlets are
        rebound only on edges incident to a redirected write node, so the
        RHS read memlet keeps naming the original array.
        """
        node_map = copy_state_contents(src, dst)
        redirected_nodes = set()
        for old, new in node_map.items():
            if not isinstance(new, dace.nodes.AccessNode):
                continue
            if new.data in rename and dst.in_degree(new) > 0:
                new.data = rename[new.data]
                redirected_nodes.add(new)
        for e in dst.edges():
            if (e.src in redirected_nodes or e.dst in redirected_nodes) and e.data.data in rename:
                # Rebind memlet to the new array name; keep subset (writes are
                # element-wise per the slice's restriction).
                e.data.data = rename[e.data.data]

    def _emit_merge_tasklet(self,
                            sdfg: dace.SDFG,
                            state: dace.SDFGState,
                            arr_name: str,
                            subset,
                            then_name: str,
                            else_name: str,
                            cond_text: str,
                            *,
                            cond_array_name: Optional[str] = None,
                            cond_producer: Optional[dace.nodes.AccessNode] = None):
        """Emit ``arr[subset] = merge(_c, _t, _e)`` where ``_c``, ``_t``,
        ``_e`` are wired as 3 in-connectors. ``cond_array_name`` is the
        bool transient already lifted for this cond; when ``None``, the
        cond stays as free-symbol text inside the merge tasklet body.
        ``cond_producer`` is the access node the lift/combine tasklet wrote
        the transient through; reusing it keeps the merge on the same
        connected dataflow path as the producer (else codegen may emit the
        merge before the cond is computed). ``None`` falls back to a fresh
        read node (recipe-1 array, produced elsewhere)."""
        access_then = state.add_access(then_name)
        access_else = state.add_access(else_name)
        access_out = state.add_access(arr_name)
        subset_str = str(subset)

        if cond_array_name is not None:
            cond_access = cond_producer if cond_producer is not None else state.add_access(cond_array_name)
            t = state.add_tasklet(
                name=f"merge_{arr_name}",
                inputs={"_c", "_t", "_e"},
                outputs={"_o"},
                code="_o = merge(_c, _t, _e)",
            )
            cond_subset = "0" if sdfg.arrays[cond_array_name].total_size == 1 else subset_str
            state.add_edge(cond_access, None, t, "_c", dace.Memlet(expr=f"{cond_array_name}[{cond_subset}]"))
        else:
            t = state.add_tasklet(
                name=f"merge_{arr_name}",
                inputs={"_t", "_e"},
                outputs={"_o"},
                code=f"_o = merge({cond_text}, _t, _e)",
            )
        state.add_edge(access_then, None, t, "_t", dace.Memlet(expr=f"{then_name}[{subset_str}]"))
        state.add_edge(access_else, None, t, "_e", dace.Memlet(expr=f"{else_name}[{subset_str}]"))
        state.add_edge(t, "_o", access_out, None, dace.Memlet(expr=f"{arr_name}[{subset_str}]"))

    def _resolve_cond_to_array(self,
                               sdfg: dace.SDFG,
                               state: dace.SDFGState,
                               cond_text: str,
                               subset_str: str,
                               *,
                               skip_cb=None) -> Optional[Tuple[str, Optional[dace.nodes.AccessNode]]]:
        """Resolve ``cond_text`` to a per-lane bool transient usable as the
        ``_c`` source of the merge tasklet.

        :returns: ``None`` when no transient can be produced (the caller
            then keeps the cond as free-symbol text in the merge body), or
            ``(array_name, producer_access)`` where ``producer_access`` is
            the access node the emitted lift/combine tasklet writes the
            transient through in *this* state — consumers must connect their
            read edge from that exact node so the dataflow stays a single
            connected path. ``producer_access`` is ``None`` only for recipe
            1 (the name already designates an array produced elsewhere with
            its own edges); the caller then adds its own read node.

        Recipes tried in order:

        1. ``cond_text`` already names an SDFG array — return the name.
        2. ``cond_text`` is a single symbol that an interstate edge sets;
           lift that assignment into a comparison tasklet.
        3. ``cond_text`` is a compound expression (`(c1 or c2)`, etc.)
           over several such symbols; lift each name recursively and emit
           one combine tasklet over the lifted transients.
        """
        cond_text = cond_text.strip()
        if cond_text in sdfg.arrays:
            return cond_text, None
        direct = self._lift_interstate_cond_to_tasklet(sdfg, state, cond_text, subset_str, skip_cb=skip_cb)
        if direct is not None:
            return direct
        return self._lift_compound_cond_to_tasklet(sdfg, state, cond_text, subset_str, skip_cb=skip_cb)

    def _lift_compound_cond_to_tasklet(self,
                                       sdfg: dace.SDFG,
                                       state: dace.SDFGState,
                                       cond_text: str,
                                       subset_str: str,
                                       *,
                                       skip_cb=None):
        """Handle the case where ``cond_text`` is a Python boolean
        expression over multiple symbols, each set by an interstate-edge
        assignment, e.g. ``(__tmp0 or __tmp1)``. Recursively lifts each
        constituent name and then emits a single combine tasklet whose
        body is the original ``cond_text`` with each name swapped for
        its in-connector."""
        import ast as _ast
        # By the time we see ``cond_text``, upstream simplification may
        # have rewritten the Python boolean operators into their C++
        # equivalents (``||``, ``&&``, ``!``). Normalise back so the AST
        # parser can handle the expression; the substituted form is what
        # the emitted tasklet body uses too.
        py_text = re.sub(r"\|\|", " or ", cond_text)
        py_text = re.sub(r"&&", " and ", py_text)
        py_text = re.sub(r"!\s*\(", "not (", py_text)
        try:
            tree = _ast.parse(py_text, mode="eval").body
        except SyntaxError:
            return None
        names = sorted({n.id for n in _ast.walk(tree) if isinstance(n, _ast.Name)})
        # The direct lift already handles the bare-name case
        # (``cond_text`` literally equals one symbol). Anything that has
        # zero names (a pure constant) can't be lifted here; let the
        # caller bake it inline.
        if not names or cond_text in names:
            return None

        # Recurse, every name must resolve to an array (no partial lifts).
        # ``_resolve_cond_to_array`` returns ``(array_name, producer_access)``
        # where ``producer_access`` is the access node a lift/combine tasklet
        # in *this* state writes the transient through (``None`` when the
        # name was already a pre-existing array produced elsewhere).
        lifted = {}
        for name in names:
            resolved = self._resolve_cond_to_array(sdfg, state, name, subset_str, skip_cb=skip_cb)
            if resolved is None:
                return None
            lifted[name] = resolved

        # Pick the shape from any lifted transient; they all describe the
        # same per-lane bool result.
        any_arr = next(iter(lifted.values()))[0]
        shape = sdfg.arrays[any_arr].shape

        cond_name, _ = sdfg.add_array(name="_cond_compound",
                                      shape=shape,
                                      dtype=dace.bool_,
                                      storage=dace.dtypes.StorageType.Register,
                                      transient=True,
                                      find_new_name=True)

        # Substitute each lifted name in the Python-normalised expression
        # with its in-connector. Word-boundary regex avoids touching names
        # that happen to be substrings of others.
        in_conn_names = {}
        cleaned = py_text
        for i, name in enumerate(names):
            conn = f"_c_{i}"
            in_conn_names[name] = conn
            cleaned = re.sub(rf"\b{re.escape(name)}\b", conn, cleaned)

        out_conn = f"_out_{cond_name}"
        t = state.add_tasklet(name=f"combine_cond_{cond_name}",
                              inputs=set(in_conn_names.values()),
                              outputs={out_conn},
                              code=f"{out_conn} = ({cleaned})")
        for name, conn in in_conn_names.items():
            lifted_name, lifted_producer = lifted[name]
            # Reuse the producing access node so ``lift_cond -> transient ->
            # combine_cond`` is a single connected dataflow path. A fresh
            # ``state.add_access`` would leave the producer in a disconnected
            # component, so codegen is free to emit the combine before the
            # lift (TSVC s271: ``_cond_compound = _cond_b_index>0`` was
            # emitted ahead of ``_cond_b_index = b``). Only the
            # already-an-array case (producer ``None``, written elsewhere
            # with its own edges) needs a fresh read node here.
            lifted_access = lifted_producer if lifted_producer is not None else state.add_access(lifted_name)
            lifted_total = sdfg.arrays[lifted_name].total_size
            lifted_subset = "0" if lifted_total == 1 else subset_str
            state.add_edge(lifted_access, None, t, conn, dace.Memlet(expr=f"{lifted_name}[{lifted_subset}]"))

        cond_access = state.add_access(cond_name)
        cond_subset = "0" if shape == (1, ) else subset_str
        state.add_edge(t, out_conn, cond_access, None, dace.Memlet(expr=f"{cond_name}[{cond_subset}]"))
        return cond_name, cond_access

    def _lift_interstate_cond_to_tasklet(self,
                                         sdfg: dace.SDFG,
                                         state: dace.SDFGState,
                                         cond_sym: str,
                                         subset_str: str,
                                         *,
                                         skip_cb=None):
        """Walk the CFG looking for an interstate-edge assignment to
        ``cond_sym``. If found, emit a tasklet in ``state`` that computes
        the assignment's RHS using array reads as in-connectors and
        writes to a fresh transient. Returns the transient name on
        success, ``None`` if no assignment is found."""
        rhs = None
        defining_edge = None
        for cfg in sdfg.all_control_flow_regions(recursive=True):
            for e in cfg.edges():
                assigns = getattr(e.data, "assignments", None) or {}
                if cond_sym in assigns:
                    rhs = assigns[cond_sym]
                    defining_edge = e
                    break
            if rhs is not None:
                break
        if rhs is None:
            return None
        # Only delete the upstream assignment + drop the symbol when the
        # symbol has no other consumer in the SDFG. With other consumers
        # the kept assignment defines the symbol for them, while the
        # per-lane lift tasklet supplies the vector form for the merge.
        if not _symbol_has_external_consumer(sdfg, cond_sym, defining_edge, skip_cb=skip_cb):
            del defining_edge.data.assignments[cond_sym]
            if cond_sym in sdfg.symbols:
                sdfg.remove_symbol(cond_sym)
            if sdfg.parent_nsdfg_node is not None and cond_sym in sdfg.parent_nsdfg_node.symbol_mapping:
                del sdfg.parent_nsdfg_node.symbol_mapping[cond_sym]

        # Collect the arrays the RHS reads. A subscripted access ``c[i]``
        # is a :class:`Subscript` node whose ``free_symbols`` are only the
        # indices, so ``free_symbols_and_functions`` no longer reports the
        # array head — :func:`symbolic.arrays` is the accessor for that.
        # Union both so a bare array reference (``cond_sym = c``) and a
        # bracketed one (``cond_sym = c[i]``) are each picked up.
        try:
            free_vars = set(symbolic.arrays(rhs)) | set(symbolic.free_symbols_and_functions(rhs))
        except Exception:
            return None
        arr_reads = sorted(v for v in free_vars if v in sdfg.arrays)

        # Build the lifted transient sized to the cond range. The RHS of
        # ``cond_sym``'s assignment is NOT necessarily a boolean
        # comparison — it is frequently a *value* operand that a
        # downstream comparison consumes (e.g. ``cond_sym = b[i]`` feeding
        # ``b[i] > 0.0``). Typing this array ``bool`` truncates that
        # float operand to ``b != 0`` and the comparison becomes wrong
        # (TSVC s271 ``if b[i] > 0.0`` -> all negative-b lanes wrongly
        # taken). Carry the RHS's actual dtype (the first array read's);
        # a genuine bool-valued RHS stored as ``0.0``/``1.0`` is still
        # correct for the downstream comparison/merge. ``_cond_compound``
        # (the final boolean) stays ``bool`` separately.
        if arr_reads:
            template = sdfg.arrays[arr_reads[0]]
            shape = template.shape
            cond_dtype = template.dtype
        else:
            shape = (1, )
            cond_dtype = dace.bool_
        cond_name, _ = sdfg.add_array(name=f"_cond_{cond_sym}",
                                      shape=shape,
                                      dtype=cond_dtype,
                                      storage=dace.dtypes.StorageType.Register,
                                      transient=True,
                                      find_new_name=True)

        # Substitute each array reference with its in-connector name. The
        # interstate-edge RHS may write ``arr[idx]`` (bracketed access) or
        # the bare ``arr`` symbol; ``replace_array_accesses_with_connectors``
        # parses the RHS via :class:`SymExpr` and rewrites both forms
        # structurally so identifier overlaps (``arr`` vs ``arr10``) cannot
        # corrupt the result.
        in_conn_names = {arr: f"_in_{arr}_{i}" for i, arr in enumerate(arr_reads)}
        cleaned_rhs, extracted_subsets = symbolic.replace_array_accesses_with_connectors(
            rhs, in_conn_names, set(sdfg.arrays.keys()))

        out_conn = f"_out_{cond_name}"
        t = state.add_tasklet(name=f"lift_cond_{cond_sym}",
                              inputs=set(in_conn_names.values()),
                              outputs={out_conn},
                              code=f"{out_conn} = ({cleaned_rhs})")
        for arr, conn in in_conn_names.items():
            an = state.add_access(arr)
            # Prefer the subset the RHS itself wrote (``arr[i, j]`` → ``[i, j]``);
            # only fall back to ``[0]`` for shape-1 scalars and ``subset_str`` for
            # bare references where the RHS gave us no per-array subset.
            # A length-1 / Scalar operand is a loop-invariant value (a
            # non-transient scalar source like the kernel arg ``c`` in
            # ``a[i, j] > c``). It must stay a 1-element ``[0]`` read so
            # the vectorizer broadcasts it (array-op-scalar) — never the
            # captured / W-wide per-lane subset, which OOB-reads a
            # 1-element source and cannot be reshaped (it is a parent-fed
            # connector). Otherwise prefer the subset the RHS wrote
            # (``arr[i, j]`` → ``[i, j]``), or ``subset_str``.
            captured = extracted_subsets.get(arr)
            if sdfg.arrays[arr].total_size == 1:
                arr_subset = "0"
            elif captured:
                arr_subset = captured.strip("[]")
            else:
                arr_subset = subset_str
            state.add_edge(an, None, t, conn, dace.Memlet(expr=f"{arr}[{arr_subset}]"))
        cond_access = state.add_access(cond_name)
        cond_subset = "0" if shape == (1, ) else subset_str
        state.add_edge(t, out_conn, cond_access, None, dace.Memlet(expr=f"{cond_name}[{cond_subset}]"))
        return cond_name, cond_access
