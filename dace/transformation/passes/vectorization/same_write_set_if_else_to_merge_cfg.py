# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
``SameWriteSetIfElseToMergeCFG`` — rewrites a same-write-set ``if/else`` into
three sequential CFGs (``compute_then``, ``compute_else``, ``apply_merge``),
where ``apply_merge`` emits one ``merge(cond, _then_<arr>, _else_<arr>)``
tasklet per shared write target.

After this pass runs, the original ``ConditionalBlock`` is gone for any pair
of arms that write to *identical* element subsets; the two arms become
sequentially-executed CFGs that produce temporaries, and a third state folds
those temporaries via the symbolic ``merge`` (see
:mod:`dace.runtime.include.dace.merge`) which the vectorizer later lowers
to a SIMD blend.

Restrictions in this slice (raise :class:`NotImplementedError` otherwise):
- the ``ConditionalBlock`` has exactly two branches, the first with a
  condition and the second with ``None`` (the ``else``);
- each branch is a :class:`ControlFlowRegion` containing exactly one state;
- writes are element subsets (``num_elements_exact() == 1``) with the
  *same* subset in both arms for every shared write target;
- arm bodies contain only ``Tasklet`` and ``AccessNode`` nodes (no
  nested SDFGs, no maps inside the arm; tile-level prep happens elsewhere).
"""
from typing import Optional

import dace
from dace import properties
from dace.properties import CodeBlock
from dace.sdfg.construction_utils import (
    assert_connector_role_matches_edges,
    copy_state_contents,
)
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation import pass_pipeline as ppl


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
        if len(cb.branches) != 2:
            return False
        (cond0, body0), (cond1, body1) = cb.branches
        if cond0 is None or cond1 is not None:
            return False
        if not (isinstance(body0, ControlFlowRegion) and isinstance(body1, ControlFlowRegion)):
            return False
        # Each branch must hold exactly one SDFGState.
        if len(body0.nodes()) != 1 or len(body1.nodes()) != 1:
            return False
        s0, s1 = body0.nodes()[0], body1.nodes()[0]
        if not (isinstance(s0, dace.SDFGState) and isinstance(s1, dace.SDFGState)):
            return False
        # No maps / nested SDFGs / non-tasklet compute in the arms.
        for s in (s0, s1):
            for n in s.nodes():
                if isinstance(n, (dace.nodes.AccessNode, dace.nodes.Tasklet)):
                    continue
                return False
        # The arms must share at least one write target with matching subsets.
        # Each arm may write additional arm-local intermediates (typical
        # frontend shape: per-binop temp plus the shared output), those stay
        # in the arm after the 3-CFG rewrite.
        shared = self._shared_writes(s0, s1)
        return bool(shared)

    def _collect_write_subsets(self, state: dace.SDFGState):
        """Return ``{arr_name: subset}`` for every write. Each write must be a
        single element (matches the restriction documented in the module
        docstring)."""
        out = {}
        for n in state.nodes():
            if not isinstance(n, dace.nodes.AccessNode):
                continue
            for e in state.in_edges(n):
                if e.data.data is None:
                    continue
                if e.data.subset.num_elements_exact() != 1:
                    raise NotImplementedError(
                        f"SameWriteSetIfElseToMergeCFG: non-element write subset {e.data.subset} on {n.data}")
                out[n.data] = e.data.subset
        return out

    def _shared_writes(self, s0: dace.SDFGState, s1: dace.SDFGState) -> dict:
        """Return ``{arr_name: subset}`` for arrays written in both arms with
        identical subsets. Returns an empty dict if no shared write target
        has a matching subset (caller treats that as a non-match)."""
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
        parent = cb.parent_graph
        (cond_block, then_body), (_, else_body) = cb.branches
        then_state: dace.SDFGState = then_body.nodes()[0]
        else_state: dace.SDFGState = else_body.nodes()[0]
        # ``cb`` may live inside a NestedSDFG; arrays referenced from its arms
        # live on the immediate enclosing SDFG, not the outermost ``sdfg``.
        local_sdfg: dace.SDFG = cb.sdfg

        # Per-arm escape set drives temp allocation and the clone-redirect.
        # An arm-local intermediate that nothing outside reads stays inline.
        from dace.transformation.passes.vectorization.branch_normalization import (  # avoid import cycle
            compute_arm_escape_writes,
        )
        escape_plan = compute_arm_escape_writes(local_sdfg, cb)
        all_escapes = escape_plan.get(0, set()) | escape_plan.get(1, set())
        if not all_escapes:
            return

        # We need the writing arm's subset to size the merge memlet, and we
        # only allocate ``_<arm>_<arr>`` for arms that actually write ``arr``;
        # the other arm's merge operand is the pre-cb value of ``arr``.
        then_writes = self._collect_write_subsets(then_state)
        else_writes = self._collect_write_subsets(else_state)

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

        # New 3-CFG states in the parent graph.
        ct_state = parent.add_state(f"compute_then_{cb.label}")
        ce_state = parent.add_state(f"compute_else_{cb.label}")
        am_state = parent.add_state(f"apply_merge_{cb.label}")

        # Clone bodies redirecting only the per-arm escape writes.
        self._clone_with_redirect(then_state, ct_state, temp_then)
        self._clone_with_redirect(else_state, ce_state, temp_else)

        # Merge tasklets. A non-writing arm contributes the pre-cb value
        # (read the original ``arr``, which is intact because the writing
        # arm now targets its private temp).
        cond_text = cond_block.as_string if isinstance(cond_block, CodeBlock) else str(cond_block)
        for arr, subset in write_subsets.items():
            then_op = temp_then.get(arr, arr)
            else_op = temp_else.get(arr, arr)
            self._emit_merge_tasklet(local_sdfg, am_state, arr, subset, then_op, else_op, cond_text)

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
        """Deep-copy ``src`` into ``dst``; access nodes whose ``.data`` is a key
        in ``rename`` get retargeted to ``rename[old]``, and every memlet that
        names the same array follows."""
        node_map = copy_state_contents(src, dst)
        for old, new in node_map.items():
            if not isinstance(new, dace.nodes.AccessNode):
                continue
            if new.data in rename:
                new.data = rename[new.data]
        for e in dst.edges():
            if e.data.data in rename:
                # Rebind memlet to the new array name; keep subset (writes are
                # element-wise per the slice's restriction).
                e.data.data = rename[e.data.data]

    def _emit_merge_tasklet(self, sdfg: dace.SDFG, state: dace.SDFGState, arr_name: str, subset, then_name: str,
                            else_name: str, cond_text: str):
        """Emit ``arr[subset] = merge(_c, _t, _e)`` where ``_c``, ``_t``,
        ``_e`` are wired as 3 in-connectors. ``cond_text`` may be:
        - an array name in the SDFG, used directly as the ``_c`` source,
        - a symbol set by an interstate-edge assignment upstream, lifted
          inline into a comparison tasklet that writes a fresh scalar
          transient and connects it to ``_c``,
        - a free symbol kept as text in the merge tasklet's code (no in-edge)."""
        access_then = state.add_access(then_name)
        access_else = state.add_access(else_name)
        access_out = state.add_access(arr_name)
        subset_str = str(subset)

        cond_resolved = self._resolve_cond_to_array(sdfg, state, cond_text, subset_str)
        if cond_resolved is not None:
            cond_array_name, cond_access = cond_resolved
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

    def _resolve_cond_to_array(self, sdfg: dace.SDFG, state: dace.SDFGState, cond_text: str,
                               subset_str: str):
        """Resolve ``cond_text`` to an ``(array_name, access_node)`` tuple
        usable as the ``_c`` source of the merge tasklet. Tries three
        recipes in order, falls back to ``None`` so the merge tasklet
        keeps the cond as free-symbol text:

        1. ``cond_text`` already names an SDFG array.
        2. ``cond_text`` is a single symbol that an interstate edge sets;
           lift that assignment into a comparison tasklet.
        3. ``cond_text`` is a compound expression (`(c1 or c2)`, etc.)
           over several such symbols; lift each name recursively and emit
           one combine tasklet over the lifted transients.
        """
        cond_text = cond_text.strip()
        if cond_text in sdfg.arrays:
            return cond_text, state.add_access(cond_text)
        direct = self._lift_interstate_cond_to_tasklet(sdfg, state, cond_text, subset_str)
        if direct is not None:
            return direct
        return self._lift_compound_cond_to_tasklet(sdfg, state, cond_text, subset_str)

    def _lift_compound_cond_to_tasklet(self, sdfg: dace.SDFG, state: dace.SDFGState, cond_text: str,
                                       subset_str: str):
        """Handle the case where ``cond_text`` is a Python boolean
        expression over multiple symbols, each set by an interstate-edge
        assignment, e.g. ``(__tmp0 or __tmp1)``. Recursively lifts each
        constituent name and then emits a single combine tasklet whose
        body is the original ``cond_text`` with each name swapped for
        its in-connector."""
        import ast as _ast
        import re as _re
        # By the time we see ``cond_text``, upstream simplification may
        # have rewritten the Python boolean operators into their C++
        # equivalents (``||``, ``&&``, ``!``). Normalise back so the AST
        # parser can handle the expression; the substituted form is what
        # the emitted tasklet body uses too.
        py_text = _re.sub(r"\|\|", " or ", cond_text)
        py_text = _re.sub(r"&&", " and ", py_text)
        py_text = _re.sub(r"!\s*\(", "not (", py_text)
        try:
            tree = _ast.parse(py_text, mode="eval").body
        except SyntaxError:
            return None
        names = sorted({n.id for n in _ast.walk(tree) if isinstance(n, _ast.Name)})
        # Only a compound when there is more than one symbol to combine;
        # the single-symbol case is already covered by the direct lift.
        if len(names) < 2:
            return None

        # Recurse, every name must resolve to an array (no partial lifts).
        lifted = {}
        for name in names:
            resolved = self._resolve_cond_to_array(sdfg, state, name, subset_str)
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
            cleaned = _re.sub(rf"\b{_re.escape(name)}\b", conn, cleaned)

        out_conn = f"_out_{cond_name}"
        t = state.add_tasklet(name=f"combine_cond_{cond_name}",
                              inputs=set(in_conn_names.values()),
                              outputs={out_conn},
                              code=f"{out_conn} = ({cleaned})")
        for name, conn in in_conn_names.items():
            lifted_name, lifted_access = lifted[name]
            lifted_total = sdfg.arrays[lifted_name].total_size
            lifted_subset = "0" if lifted_total == 1 else subset_str
            state.add_edge(lifted_access, None, t, conn, dace.Memlet(expr=f"{lifted_name}[{lifted_subset}]"))

        cond_access = state.add_access(cond_name)
        cond_subset = "0" if shape == (1, ) else subset_str
        state.add_edge(t, out_conn, cond_access, None, dace.Memlet(expr=f"{cond_name}[{cond_subset}]"))
        return cond_name, cond_access

    def _lift_interstate_cond_to_tasklet(self, sdfg: dace.SDFG, state: dace.SDFGState, cond_sym: str,
                                          subset_str: str):
        """Walk the CFG looking for an interstate-edge assignment to
        ``cond_sym``. If found, emit a tasklet in ``state`` that computes
        the assignment's RHS using array reads as in-connectors and
        writes to a fresh transient. Returns the transient name on
        success, ``None`` if no assignment is found."""
        rhs = None
        for cfg in sdfg.all_control_flow_regions(recursive=True):
            for e in cfg.edges():
                assigns = getattr(e.data, "assignments", None) or {}
                if cond_sym in assigns:
                    rhs = assigns[cond_sym]
                    # Drop the assignment so the symbol no longer leaks
                    # downstream as a free reference.
                    del e.data.assignments[cond_sym]
                    break
            if rhs is not None:
                break
        if rhs is None:
            return None
        # Remove the now-stale symbol from this SDFG's symbol table and from
        # the nested-SDFG's symbol_mapping so validation does not flag it as
        # an unmapped symbol reference.
        if cond_sym in sdfg.symbols:
            sdfg.remove_symbol(cond_sym)
        if sdfg.parent_nsdfg_node is not None and cond_sym in sdfg.parent_nsdfg_node.symbol_mapping:
            del sdfg.parent_nsdfg_node.symbol_mapping[cond_sym]

        try:
            rhs_sym = dace.symbolic.SymExpr(rhs)
            free_vars = {str(s) for s in rhs_sym.free_symbols}
        except Exception:
            return None
        arr_reads = sorted(v for v in free_vars if v in sdfg.arrays)

        # Build the new bool transient sized to the cond range. Use the
        # first array read's shape as a template, the bool array tracks
        # the per-lane comparison result.
        if arr_reads:
            template = sdfg.arrays[arr_reads[0]]
            shape = template.shape
        else:
            shape = (1, )
        cond_name, _ = sdfg.add_array(name=f"_cond_{cond_sym}",
                                      shape=shape,
                                      dtype=dace.bool_,
                                      storage=dace.dtypes.StorageType.Register,
                                      transient=True,
                                      find_new_name=True)

        # Substitute each array reference with its in-connector name. The
        # interstate-edge RHS may write ``arr[idx]`` (bracketed access) or
        # the bare ``arr`` symbol, both forms collapse to the per-lane
        # scalar connector inside the tasklet body.
        cleaned_rhs = rhs
        in_conn_names = {}
        import re as _re
        for i, arr in enumerate(arr_reads):
            conn = f"_in_{arr}_{i}"
            in_conn_names[arr] = conn
            # Bracketed: ``arr[anything]`` -> ``conn``.
            cleaned_rhs = _re.sub(rf"\b{_re.escape(arr)}\[[^\]]*\]", conn, cleaned_rhs)
            # Bare name: ``arr`` (as a whole token) -> ``conn``.
            cleaned_rhs = _re.sub(rf"\b{_re.escape(arr)}\b", conn, cleaned_rhs)

        out_conn = f"_out_{cond_name}"
        t = state.add_tasklet(name=f"lift_cond_{cond_sym}",
                              inputs=set(in_conn_names.values()),
                              outputs={out_conn},
                              code=f"{out_conn} = ({cleaned_rhs})")
        for arr, conn in in_conn_names.items():
            an = state.add_access(arr)
            arr_total = sdfg.arrays[arr].total_size
            arr_subset = "0" if arr_total == 1 else subset_str
            state.add_edge(an, None, t, conn, dace.Memlet(expr=f"{arr}[{arr_subset}]"))
        cond_access = state.add_access(cond_name)
        cond_subset = "0" if shape == (1, ) else subset_str
        state.add_edge(t, out_conn, cond_access, None, dace.Memlet(expr=f"{cond_name}[{cond_subset}]"))
        return cond_name, cond_access
