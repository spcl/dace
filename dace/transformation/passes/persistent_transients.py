# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Promotion of transients whose allocation size is fixed for a whole program run to Persistent."""

from typing import Any, Dict, Optional, Set

from dace import data as dt
from dace import dtypes, properties, symbolic
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import LoopRegion
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation

#: Lifetimes that outlive a Persistent allocation, or already are one. Promoting any of them is a
#: demotion: Global memory belongs to the program, External memory to the caller.
STRONGER_THAN_PERSISTENT = (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.Global,
                            dtypes.AllocationLifetime.External)


def persistent_size_symbols(sdfg: SDFG) -> Set[str]:
    """The symbol names an ``__dace_init``-time allocation may size itself by.

    A Persistent descriptor is allocated once, in the init function, from the top-level SDFG's
    arguments -- so only the root's own free symbols and compile-time constants have a value there.
    A name that any control flow region or interstate edge in the tree also ASSIGNS is excluded even
    when it is free at the root: its value changes between executions of the allocation site, so the
    single init-time size would be stale (:func:`~dace.sdfg.sdfg.SDFG.free_symbols` keeps a symbol
    free when it is read before it is written, as in ``i = i + 1``).

    :param sdfg: The top-level SDFG.
    :return: Names that hold one value for the whole run.
    """
    allowed = {str(s) for s in sdfg.free_symbols} | set(sdfg.constants_prop.keys())
    reassigned: Set[str] = set()
    # Collected across the whole tree: a nested assignment to a name the root also owns is a
    # collision this cannot tell apart from a rebinding, and refusing the promotion is the safe read.
    for nested in sdfg.all_sdfgs_recursive():
        for edge in nested.all_interstate_edges():
            reassigned |= set(edge.data.assignments.keys())
        for cfg in nested.all_control_flow_regions(recursive=True):
            if isinstance(cfg, LoopRegion) and cfg.loop_variable:
                reassigned.add(str(cfg.loop_variable))
    return allowed - reassigned


def size_symbols_at_root(nsdfg: SDFG, names: Set[str]) -> Optional[Set[str]]:
    """Rewrite ``names`` from ``nsdfg``'s symbol namespace into the root SDFG's.

    Each nested SDFG renames what it is handed through ``symbol_mapping``, so a nested descriptor's
    size symbols say nothing on their own -- the same name may be bound to an enclosing map parameter
    or loop variable one level up. Walking the mapping chain out to the root is what makes the
    free-symbol test mean "constant for the whole program" rather than "free in this nest".

    :param nsdfg: The SDFG owning ``names``.
    :param names: Symbol names in that SDFG's namespace.
    :return: The equivalent root-namespace names, or None if any name is not bound on the way out.
    """
    current, resolved = nsdfg, set(names)
    while current.parent_nsdfg_node is not None:
        if current.parent_sdfg is None:
            return None
        mapping = current.parent_nsdfg_node.symbol_mapping
        outer: Set[str] = set()
        for name in resolved:
            if name in current.constants_prop:
                continue
            if name not in mapping:
                return None
            outer |= {str(s) for s in symbolic.pystr_to_symbolic(str(mapping[name])).free_symbols}
        resolved, current = outer, current.parent_sdfg
    return resolved


def size_is_program_wide(nsdfg: SDFG, desc: dt.Data, allowed: Set[str]) -> bool:
    """Whether ``desc``'s allocation size holds one value for the whole program run.

    :param nsdfg: The SDFG owning ``desc``.
    :param desc: The descriptor to test.
    :param allowed: Root-namespace names from :func:`persistent_size_symbols`.
    """
    try:
        names = {str(s) for s in desc.total_size.free_symbols}
    except AttributeError:  # total_size is an integer, so the size is a compile-time constant
        return True
    if not names:
        return True
    at_root = size_symbols_at_root(nsdfg, names)
    return at_root is not None and not (at_root - allowed)


@properties.make_properties
@transformation.explicit_cf_compatible
class MakeTransientsPersistent(ppl.Pass):
    """
    Promotes every transient array whose allocation size is fixed for the whole run to
    :attr:`~dace.dtypes.AllocationLifetime.Persistent`.

    Such an array is otherwise malloc'd and freed on every execution of the state or scope holding
    it -- once per iteration of any enclosing loop. Persistent allocation moves the malloc into
    ``__dace_init`` and the free into ``__dace_exit``, so it runs once.

    A descriptor qualifies when its size reads only symbols that are free at the top level and are
    never reassigned (:func:`persistent_size_symbols`, resolved out through every nested SDFG's
    ``symbol_mapping``). Views, references, streams and scalars are skipped -- they are not heap
    allocations to begin with -- as are registers, compile-time constants, and lifetimes that
    already outlive a Persistent one.
    """

    CATEGORY: str = 'Optimization Preparation'

    toplevel_only = properties.Property(
        dtype=bool,
        default=True,
        desc='Only promote access nodes that appear outside every map/consume scope. A descriptor '
        'used inside a scope is allocated per scope entry, which persistent allocation would '
        'serialize across the threads that share the state struct.')

    def __init__(self, toplevel_only: bool = True):
        self.toplevel_only = toplevel_only

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[Dict[int, Set[str]]]:
        """
        :param sdfg: The SDFG to modify in-place.
        :return: cfg_id -> names promoted in it, or None if nothing was promoted.
        """
        allowed = persistent_size_symbols(sdfg)
        result: Dict[int, Set[str]] = {}
        for nsdfg in sdfg.all_sdfgs_recursive():
            promoted = self._promote_in(nsdfg, allowed)
            for name in promoted:
                nsdfg.arrays[name].lifetime = dtypes.AllocationLifetime.Persistent
            result[nsdfg.cfg_id] = promoted
        return result if any(result.values()) else None

    def _promote_in(self, nsdfg: SDFG, allowed: Set[str]) -> Set[str]:
        """The names in ``nsdfg`` that every one of their access nodes allows promoting."""
        persistent: Set[str] = set()
        refused: Set[str] = set()
        for state in nsdfg.states():
            for dnode in state.data_nodes():
                if dnode.data in refused:
                    continue
                desc = dnode.desc(nsdfg)
                # A struct member follows its container: promoting it alone would place it in a
                # state-struct field its parent descriptor does not have.
                if (dnode.root_data != dnode.data
                        and nsdfg.arrays[dnode.root_data].lifetime != dtypes.AllocationLifetime.Persistent):
                    continue
                if (dnode.data in nsdfg.constants_prop or not desc.transient or type(desc) is not dt.Array
                        or desc.storage == dtypes.StorageType.Register or desc.lifetime in STRONGER_THAN_PERSISTENT):
                    refused.add(dnode.data)
                    continue
                if not size_is_program_wide(nsdfg, desc, allowed):
                    refused.add(dnode.data)
                    continue
                if xfh.get_parent_map(state, dnode) is not None:
                    if self.toplevel_only or desc.lifetime == dtypes.AllocationLifetime.Scope:
                        refused.add(dnode.data)
                        continue
                persistent.add(dnode.data)
        return persistent - refused
