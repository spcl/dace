# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Specialize an SDFG for every combination of finite configuration domains and dispatch to the variants.

A configuration variable (a namelist flag, a mode switch) takes few values in a deployment. Baking each
combination into its own SDFG lets constant propagation prune every branch the combination never takes; the
returned root SDFG keeps the original signature and calls the matching variant, stored as an external SDFG file.
"""

import copy
import itertools
import os
from collections.abc import Sequence
from typing import Any

from dace import SDFG, data, nodes, properties
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState
from dace.sdfg.utils import specialize_scalars, specialize_symbols
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.dead_state_elimination import DeadStateElimination
from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars

Value = int | float | str


def staged_copies(sdfg: SDFG, names: dict[str, Value]) -> dict[str, Value]:
    """Single-element transients of ``sdfg`` whose only writes copy one of ``names``, with that name's value.

    A frontend stages a length-1 argument into a scalar it reads instead; the copy carries the configuration value.
    """
    copied: dict[str, Value] = {}
    other_writes: set[str] = set()
    for state in sdfg.states():
        for node in state.data_nodes():
            for edge in state.in_edges(node):
                src = edge.src
                if isinstance(src, nodes.AccessNode) and src.data in names:
                    copied.setdefault(node.data, names[src.data])
                else:
                    other_writes.add(node.data)
    return {
        name: val
        for name, val in copied.items()
        if name not in other_writes and sdfg.arrays[name].transient and sdfg.arrays[name].total_size == 1
    }


def specialize_config(sdfg: SDFG, values: dict[str, Value]) -> None:
    """Bake ``values`` into ``sdfg`` in place: symbols are substituted, data containers become constants.

    A configuration container stops being an argument: it turns transient, a length-1 array becomes a scalar under
    its own name, and every read folds to the constant, so nothing is left to initialize it. A transient staged
    from it by a copy takes the same value.
    """
    containers = {name: val for name, val in values.items() if name in sdfg.arrays}
    containers.update(staged_copies(sdfg, containers))
    specialize_symbols(sdfg, {name: val for name, val in values.items() if name not in containers})
    if not containers:
        return
    for name in containers:
        sdfg.arrays[name].transient = True
    arrays = {name for name in containers if isinstance(sdfg.arrays[name], data.Array)}
    if arrays:
        ConvertLengthOneArraysToScalars(filter=arrays, single_element=True).apply_pass(sdfg, {})
    specialize_scalars(sdfg, containers)


def config_condition(sdfg: SDFG, values: dict[str, Value]) -> str:
    """The condition selecting ``values``, read from the arguments of ``sdfg``."""

    def ref(name: str) -> str:
        desc = sdfg.arrays.get(name)
        return f"{name}[0]" if isinstance(desc, data.Array) else name

    return " and ".join(f"({ref(name)} == {val})" for name, val in values.items())


@properties.make_properties
class ConfigPropagation(ppl.Pass):
    """Specialize the SDFG for each combination of the configuration domains, saving every variant as an external
    SDFG file, and return a root SDFG with the original signature that dispatches to them.

    The input SDFG is left untouched. A singleton domain is baked into every variant; only the multi-valued domains
    select between variants.
    """

    CATEGORY: str = "Specialization"

    domains = properties.DictProperty(
        key_type=str, value_type=list, desc="Values each configuration symbol or container can take."
    )
    output_dir = properties.Property(dtype=str, default=".", desc="Directory the variant SDFG files are written to.")

    def __init__(self, domains: dict[str, Sequence[Value]] | None = None, output_dir: str = "."):
        super().__init__()
        self.domains = {name: list(vals) for name, vals in (domains or {}).items()}
        self.output_dir = output_dir

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def variant_values(self) -> list[dict[str, Value]]:
        """Every combination of the domains, singletons included."""
        names = list(self.domains)
        return [dict(zip(names, combo)) for combo in itertools.product(*(self.domains[n] for n in names))]

    def build_variant(self, sdfg: SDFG, values: dict[str, Value], selecting: list[str]) -> SDFG:
        variant = copy.deepcopy(sdfg)
        variant.name = "_".join([sdfg.name] + [f"{name}_{values[name]}" for name in selecting])
        specialize_config(variant, values)
        DeadStateElimination().apply_pass(variant, {})
        variant.simplify()
        return variant

    def add_call(self, root: SDFG, state: SDFGState, variant: SDFG, path: str) -> None:
        """Call the external ``variant`` stored at ``path`` from ``state``, passing the arguments of ``root``."""
        args = {name for name, desc in variant.arrays.items() if not desc.transient}
        reads, writes = variant.read_and_write_sets()
        # Ordered, untyped: the connector types are taken from the variant once it is loaded
        inputs = dict.fromkeys(sorted(args & reads))
        outputs = dict.fromkeys(sorted(args & writes))
        symbols = sorted(str(s) for s in variant.free_symbols)
        node = state.add_nested_sdfg(
            None,
            inputs,
            outputs,
            name=variant.name,
            symbol_mapping={s: s for s in symbols},
            external_path=path,
        )
        for name in inputs:
            state.add_edge(state.add_read(name), None, node, name, root.make_array_memlet(name))
        for name in outputs:
            state.add_edge(node, name, state.add_write(name), None, root.make_array_memlet(name))

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> SDFG:
        """:return: The dispatcher SDFG; the variants are written to ``output_dir``."""
        selecting = [name for name, vals in self.domains.items() if len(vals) > 1]
        root = SDFG(sdfg.name)
        for name, desc in sdfg.arrays.items():
            if not desc.transient:
                root.add_datadesc(name, copy.deepcopy(desc))
        for name in sorted(str(s) for s in sdfg.free_symbols):
            if name not in root.symbols:  # a shape may have declared it already
                root.add_symbol(name, sdfg.symbols[name])
        os.makedirs(self.output_dir, exist_ok=True)
        cond = None
        if selecting:
            cond = ConditionalBlock("config_dispatch")
            root.add_node(cond, is_start_block=True)
        for values in self.variant_values():
            variant = self.build_variant(sdfg, values, selecting)
            path = os.path.join(self.output_dir, f"{variant.name}.sdfgz")
            variant.save(path, compress=True)
            if cond is None:  # no choice left: one variant, called unconditionally
                state = root.add_state(variant.name, is_start_block=True)
            else:
                branch = ControlFlowRegion(f"call_{variant.name}", sdfg=root)
                cond.add_branch(CodeBlock(config_condition(sdfg, {n: values[n] for n in selecting})), branch)
                state = branch.add_state(variant.name, is_start_block=True)
            self.add_call(root, state, variant, path)
        return root
