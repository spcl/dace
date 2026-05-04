"""
reuse: apply buffer-reuse to a pair of explicitly-allocated SDFG arrays.

_apply_reuse(sdfg, new_arr, donor_arr) rewires the alloc/free annotations
placed by make_explicit so that new_arr receives donor_arr's heap pointer
instead of allocating fresh memory.

After the call:
  - donor_arr: allocated (alloc edge), never freed (free entry removed).
  - new_arr:   not allocated; receives donor pointer via a reuse edge entry.
  - new_arr:   freed normally (free edge retained from make_explicit).

The invariant is safe because the cats trace guarantees donor_arr's lifetime
ends before new_arr's lifetime begins.

buffer_reuse_same_pass(sdfg, symbols, sym_max_vals) is the top-level automated
pass that runs the CATS trace extractor, identifies same-size same-dtype
transient pairs whose lifetimes do not overlap, and applies _apply_reuse for
each discovered pair.
"""

import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union

from dace import dtypes
from dace.sdfg import SDFG

from .make_explicit import make_explicit


# ---------------------------------------------------------------------------
# Layer 1: _apply_reuse
# ---------------------------------------------------------------------------

def _apply_reuse(sdfg: SDFG, new_arr: str, donor_arr: str) -> None:
    """Rewrite the SDFG so that *new_arr* reuses *donor_arr*'s allocation.

    :param sdfg:       The SDFG to modify in-place.
    :param new_arr:    Name of the array that should receive donor's pointer.
    :param donor_arr:  Name of the array whose allocation is reused.
    :raises ValueError: If either array is not a transient in *sdfg*.
    """
    for name in (new_arr, donor_arr):
        if name not in sdfg.arrays:
            raise ValueError(f"'{name}' not found in sdfg.arrays")
        if not sdfg.arrays[name].transient:
            raise ValueError(f"'{name}' is not a transient data container")

    # Step 1: ensure both arrays have Explicit lifetime and alloc/free on edges.
    # Skip arrays that are already Explicit — make_explicit is not idempotent
    # when an array is already in a reuse entry: calling it again would add the
    # array back to alloc, creating an "in both alloc and reuse" contradiction.
    needs_explicit = [
        name for name in (new_arr, donor_arr)
        if sdfg.arrays[name].lifetime != dtypes.AllocationLifetime.Explicit
    ]
    if needs_explicit:
        make_explicit(sdfg, needs_explicit)

    # make_explicit may place alloc/free on interstate edges inside nested
    # control-flow regions (e.g. loop bodies), so iterate recursively.
    all_edges = list(sdfg.all_interstate_edges(recursive=True))

    # Step 2: for new_arr — replace alloc entries with reuse entries
    for edge in all_edges:
        if new_arr in edge.data.alloc:
            edge.data.alloc.remove(new_arr)
            edge.data.reuse.append([new_arr, donor_arr])

    # Step 3: for donor_arr — remove free entries (new_arr's free takes over)
    for edge in all_edges:
        if donor_arr in edge.data.free:
            edge.data.free.remove(donor_arr)


# ---------------------------------------------------------------------------
# Layer 2: CATS-based liveness extraction
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _AllocEntry:
    array_name: str
    size_bytes: int
    dtype: Any          # dace.dtypes.typeclass


@dataclasses.dataclass
class _FreeEntry:
    array_name: str
    size_bytes: int     # carried for convenience — needed to rebuild pool key
    dtype: Any          # dace.dtypes.typeclass
    ua_ratio: float = 1.0   # usage_window / alloc_window ∈ [0.0, 1.0]; 1.0 = tight or unknown


_LivenessEvent = Union[_AllocEntry, _FreeEntry]


def _run_cats(
    sdfg: SDFG,
    symbols: Dict[str, int],
    symbol_max_vals: Dict[str, int],
) -> Tuple[Any, Dict[str, str]]:
    """Run CATS trace extraction on *sdfg* and return the generator plus a
    reverse map from CATS alloc names to DaCe array names.

    The CATS codegen registers its own CPU target (``@autoregister_params``),
    which shadows DaCe's built-in one.  This helper saves and restores the
    'cpu' registry slot around the call so subsequent DaCe compilations use
    the real DaCe CPU codegen.
    """
    from dace.codegen.target import TargetCodeGenerator
    from dace.codegen.targets.cpp import ptr as cpp_ptr

    from cats_dace.cats_trace_extractor import CATSTraceExtractor, _get_codegen_targets
    from cats_dace.codegen.targets.cpu import CPUCodeGen as CATSCPUCodeGen

    # Snapshot the 'cpu' slot AFTER the CATS import so that if the caller already
    # imported cats_dace elsewhere (e.g. from cats_dace.utils), the CATS class is
    # excluded from the restore list — otherwise we would re-register it.
    orig_cpu_entry = {
        cls: kwargs
        for cls, kwargs in TargetCodeGenerator.extensions().items()
        if kwargs.get('name') == 'cpu' and cls is not CATSCPUCodeGen
    }

    generator = CATSTraceExtractor(sdfg, symbols, symbol_max_vals,
                                   transients_only=True)
    _targets = {'cpu': CATSCPUCodeGen(generator, sdfg)}  # noqa: F841
    _get_codegen_targets(sdfg, generator)
    generator.generate_timeline(sdfg, None)

    if CATSCPUCodeGen in TargetCodeGenerator.extensions():
        TargetCodeGenerator.unregister(CATSCPUCodeGen)
    for cls, kwargs in orig_cpu_entry.items():
        if cls not in TargetCodeGenerator.extensions():
            TargetCodeGenerator.register(cls, **kwargs)

    # CATS encodes each allocation as str(sdfg.cfg_id) + '_' + cpp_ptr(...)
    alloc_name_to_array: Dict[str, str] = {}
    for arr_name, desc in sdfg.arrays.items():
        if not desc.transient:
            continue
        ptr_str = cpp_ptr(arr_name, desc, sdfg, generator)
        alloc_name = f'{sdfg.cfg_id}_{ptr_str}'
        alloc_name_to_array[alloc_name] = arr_name

    return generator, alloc_name_to_array


def _extract_liveness(
    sdfg: SDFG,
    symbols: Dict[str, int],
    symbol_max_vals: Dict[str, int],
) -> List[_LivenessEvent]:
    """Run CATS on *sdfg* and return a liveness event list for transients.

    Each array's effective window is ``[first_DataAccessEvent, last_DataAccessEvent]``.
    For arrays with no access events (unused scratch), the window falls back to
    ``[AllocationEvent, DeallocationEvent]`` and ``ua_ratio`` is set to 1.0.

    ``ua_ratio`` on each ``_FreeEntry`` = ``usage_window / alloc_window``, clamped
    to ``[0.0, 1.0]``.  A low ratio means the array sat allocated for much longer
    than it was actually used.
    """
    from cats_dace.utils import AllocationEvent, DeallocationEvent, DataAccessEvent

    generator, alloc_name_to_array = _run_cats(sdfg, symbols, symbol_max_vals)

    alloc_idx: Dict[str, int] = {}
    dealloc_idx: Dict[str, int] = {}
    first_access_idx: Dict[str, int] = {}
    last_access_idx: Dict[str, int] = {}
    size_bytes: Dict[str, int] = {}
    dtype_of: Dict[str, Any] = {}

    for i, event in enumerate(generator.access_timeline):
        if isinstance(event, AllocationEvent):
            for alloc_name, nbytes in event.data:
                arr_name = alloc_name_to_array.get(alloc_name)
                if arr_name is None or nbytes <= 0:
                    continue
                alloc_idx[arr_name] = i
                size_bytes[arr_name] = nbytes
                dtype_of[arr_name] = sdfg.arrays[arr_name].dtype
        elif isinstance(event, DeallocationEvent):
            for alloc_name in event.data:
                arr_name = alloc_name_to_array.get(alloc_name)
                if arr_name is None:
                    continue
                dealloc_idx[arr_name] = i
        elif isinstance(event, DataAccessEvent):
            arr_name = alloc_name_to_array.get(event.alloc_name)
            if arr_name is None:
                continue
            if arr_name not in first_access_idx:
                first_access_idx[arr_name] = i
            last_access_idx[arr_name] = i

    # Build (sort_key, kind, arr_name, ua_ratio) tuples, then emit in order.
    ordered: List[Tuple[float, str, str, float]] = []
    for arr_name in alloc_idx:
        ai = alloc_idx[arr_name]
        da = dealloc_idx.get(arr_name)
        fa = first_access_idx.get(arr_name)
        la = last_access_idx.get(arr_name)

        if fa is not None and la is not None:
            alloc_key = fa - 0.5
            free_key = la + 0.5
            alloc_window = (da - ai) if (da is not None and da != ai) else 0
            usage_window = la - fa
            if alloc_window > 0:
                ua_ratio = max(0.0, min(1.0, usage_window / alloc_window))
            else:
                ua_ratio = 1.0
        else:
            if da is None:
                continue
            alloc_key = float(ai)
            free_key = float(da)
            ua_ratio = 1.0

        ordered.append((alloc_key, 'alloc', arr_name, 0.0))
        ordered.append((free_key, 'free', arr_name, ua_ratio))

    ordered.sort(key=lambda t: t[0])

    events: List[_LivenessEvent] = []
    for _, kind, arr_name, ua_ratio in ordered:
        if kind == 'alloc':
            events.append(_AllocEntry(arr_name, size_bytes[arr_name], dtype_of[arr_name]))
        else:
            events.append(_FreeEntry(arr_name, size_bytes[arr_name], dtype_of[arr_name], ua_ratio))
    return events



# ---------------------------------------------------------------------------
# Shared safety checks (used by both Layer 3 and Layer 4 passes)
#
# _collect_scopes:  reject cross-ControlFlowRegion pairs (necessary for both
#                   same-size and cross-size reuse).
# _edge_order_safe: additionally required by buffer_reuse_cross_pass — checks strict
#                   topological ordering within the shared region.  Needed
#                   because make_explicit coarsens CATS event-level lifetimes
#                   to block-boundary edges; two events that CATS sees as
#                   non-overlapping but inside the same block would alias via
#                   the reuse pointer without this check.
# ---------------------------------------------------------------------------

def _collect_scopes(
    sdfg: SDFG,
    names: set,
) -> Dict[str, Tuple[frozenset, frozenset]]:
    """For each array name in *names*, collect the set of parent
    ControlFlowRegion ids for its alloc edges and for its free edges.

    Returns: name -> (frozenset of alloc-region ids, frozenset of free-region ids).

    Rejects cross-scope pairs: when new_arr's alloc/free regions differ from
    donor_arr's, rebinding via _apply_reuse would free the donor's storage
    across loop iterations and cause a use-after-free.  Insufficient alone
    for the arena pass — see _edge_order_safe.
    """
    scopes: Dict[str, Tuple[set, set]] = {n: (set(), set()) for n in names}
    for ns in sdfg.all_sdfgs_recursive():
        for region in ns.all_control_flow_regions(recursive=True):
            for edge in region.edges():
                data = edge.data
                for n in data.alloc:
                    if n in scopes:
                        scopes[n][0].add(id(region))
                for n in data.free:
                    if n in scopes:
                        scopes[n][1].add(id(region))
    return {n: (frozenset(a), frozenset(f)) for n, (a, f) in scopes.items()}



# ---------------------------------------------------------------------------
# Layer 3: left-to-right greedy scan + top-level pass
# ---------------------------------------------------------------------------

def _greedy_same_size_scan(
    liveness: List[_LivenessEvent],
) -> List[Tuple[str, str]]:
    """Identify reuse pairs by a left-to-right greedy scan.

    Match key: ``(size_bytes, dtype)`` — exact byte size and element type
    must agree.  When an allocation is seen and the free pool has a matching
    donor, the donor is assigned to the new array (LIFO within each bucket).

    :param liveness: Output of :func:`_extract_liveness`.
    :returns: List of ``(new_array_name, donor_array_name)`` pairs.
    """
    # free_pool: (size_bytes, dtype) → stack of freed array names
    free_pool: Dict[Tuple[int, Any], List[str]] = {}
    reuse_plan: List[Tuple[str, str]] = []

    for event in liveness:
        if isinstance(event, _AllocEntry):
            key = (event.size_bytes, event.dtype)
            bucket = free_pool.get(key)
            if bucket:
                donor = bucket.pop()
                if not bucket:
                    del free_pool[key]
                reuse_plan.append((event.array_name, donor))

        elif isinstance(event, _FreeEntry):
            key = (event.size_bytes, event.dtype)
            free_pool.setdefault(key, []).append(event.array_name)

    return reuse_plan


def _greedy_donor_candidates(
    liveness: List[_LivenessEvent],
) -> Dict[str, List[str]]:
    """Return ALL available donors per consumer in LIFO priority order.

    Unlike :func:`_greedy_same_size_scan`, does not commit a donor on the first
    match.  Returns the full ordered list so that callers can fall back to the
    next candidate when the primary donor fails safety checks.

    :returns: Dict mapping each consumer array name to its ordered donor list
              (most-recently-freed first).
    """
    free_pool: Dict[Tuple[int, Any], List[str]] = {}
    candidates: Dict[str, List[str]] = {}

    for event in liveness:
        if isinstance(event, _AllocEntry):
            key = (event.size_bytes, event.dtype)
            bucket = free_pool.get(key, [])
            if bucket:
                candidates[event.array_name] = list(reversed(bucket))
        elif isinstance(event, _FreeEntry):
            key = (event.size_bytes, event.dtype)
            free_pool.setdefault(key, []).append(event.array_name)

    return candidates


def _ua_greedy_same_size_scan(
    liveness: List[_LivenessEvent],
) -> List[Tuple[str, str]]:
    """U/A-ratio variant of :func:`_greedy_same_size_scan`.

    Identical left-to-right scan, but when multiple same-size same-dtype donors
    are available the one with the **lowest** ``ua_ratio`` is selected first —
    i.e. the most-idle original allocation is consumed before tightly-used ones.
    """
    # free_pool: (size_bytes, dtype) → list of (ua_ratio, array_name)
    free_pool: Dict[Tuple[int, Any], List[Tuple[float, str]]] = {}
    reuse_plan: List[Tuple[str, str]] = []

    for event in liveness:
        if isinstance(event, _AllocEntry):
            key = (event.size_bytes, event.dtype)
            bucket = free_pool.get(key)
            if bucket:
                best_idx = min(range(len(bucket)), key=lambda i: bucket[i][0])
                _, donor = bucket.pop(best_idx)
                if not bucket:
                    del free_pool[key]
                reuse_plan.append((event.array_name, donor))
        elif isinstance(event, _FreeEntry):
            key = (event.size_bytes, event.dtype)
            free_pool.setdefault(key, []).append((event.ua_ratio, event.array_name))

    return reuse_plan


def _ua_donor_candidates(
    liveness: List[_LivenessEvent],
) -> Dict[str, List[str]]:
    """Return ALL available donors per consumer sorted by ascending u/a ratio.

    U/A-ratio variant of :func:`_greedy_donor_candidates`: the most-idle donor
    (lowest ratio) is first in each list.

    :returns: Dict mapping each consumer array name to its ordered donor list.
    """
    free_pool: Dict[Tuple[int, Any], List[Tuple[float, str]]] = {}
    candidates: Dict[str, List[str]] = {}

    for event in liveness:
        if isinstance(event, _AllocEntry):
            key = (event.size_bytes, event.dtype)
            bucket = free_pool.get(key, [])
            if bucket:
                candidates[event.array_name] = [
                    name for _, name in sorted(bucket, key=lambda x: x[0])
                ]
        elif isinstance(event, _FreeEntry):
            key = (event.size_bytes, event.dtype)
            free_pool.setdefault(key, []).append((event.ua_ratio, event.array_name))

    return candidates


def buffer_reuse_same_pass(
    sdfg: SDFG,
    symbols: Dict[str, int],
    symbol_max_vals: Dict[str, int],
) -> List[Tuple[str, str]]:
    """Automatically reuse same-size same-dtype transient buffers in *sdfg*.

    1. Extracts a liveness trace via CATS (transients only).
    2. Runs a left-to-right greedy scan to find non-overlapping same-size
       same-dtype pairs.
    3. Applies :func:`_apply_reuse` for each discovered pair.

    :param sdfg:            Root SDFG to optimise in-place.
    :param symbols:         Concrete symbol values for CATS size computation.
    :param symbol_max_vals: Upper-bound values for unresolved symbols.
    :returns: The list of ``(new_array, donor_array)`` reuse pairs applied.
    """
    import copy as _copy

    # CATS compiles with its own CPUCodeGen; running it directly on *sdfg*
    # would poison the build folder, so use a separately-named deep copy.
    probe = _copy.deepcopy(sdfg)
    probe._name = probe._name + "_cats_probe"
    liveness = _extract_liveness(probe, symbols, symbol_max_vals)
    donor_candidates = _greedy_donor_candidates(liveness)

    involved = {n for new in donor_candidates
                for n in [new] + donor_candidates[new] if n in probe.arrays}
    for name in list(involved):
        if probe.arrays[name].lifetime != dtypes.AllocationLifetime.Explicit:
            try:
                make_explicit(probe, [name])
            except ValueError:
                involved.discard(name)
    scopes = _collect_scopes(probe, involved)

    # Try donors in LIFO priority order; fall back if a candidate fails
    # the scope or edge-order safety check.
    applied: List[Tuple[str, str]] = []
    used_donors: set = set()
    for new_arr, donors in donor_candidates.items():
        for donor_arr in donors:
            if donor_arr in used_donors:
                continue
            if new_arr not in scopes or donor_arr not in scopes:
                continue
            if scopes[new_arr] != scopes[donor_arr]:
                continue
            if not _edge_order_safe(probe, new_arr, donor_arr):
                continue
            try:
                _apply_reuse(sdfg, new_arr, donor_arr)
                applied.append((new_arr, donor_arr))
                used_donors.add(donor_arr)
                break
            except ValueError:
                pass

    return applied


def buffer_reuse_same_pass_ua(
    sdfg: SDFG,
    symbols: Dict[str, int],
    symbol_max_vals: Dict[str, int],
) -> List[Tuple[str, str]]:
    """U/A-ratio variant of :func:`buffer_reuse_same_pass`.

    Uses :func:`_extract_liveness` (liveness tightened to each array's
    actual access window) and :func:`_ua_greedy_same_size_scan` (ratio-sorted:
    when multiple same-size same-dtype donors are available, the one with the
    lowest u/a ratio — most-idle allocation — is consumed first).

    :param sdfg:            Root SDFG to optimise in-place.
    :param symbols:         Concrete symbol values for CATS size computation.
    :param symbol_max_vals: Upper-bound values for unresolved symbols.
    :returns: Applied ``(new_array, donor_array)`` reuse pairs.
    """
    import copy as _copy

    probe = _copy.deepcopy(sdfg)
    probe._name = probe._name + "_cats_probe"
    liveness = _extract_liveness(probe, symbols, symbol_max_vals)
    donor_candidates = _ua_donor_candidates(liveness)

    involved = {n for new in donor_candidates
                for n in [new] + donor_candidates[new] if n in probe.arrays}
    for name in list(involved):
        if probe.arrays[name].lifetime != dtypes.AllocationLifetime.Explicit:
            try:
                make_explicit(probe, [name])
            except ValueError:
                involved.discard(name)
    scopes = _collect_scopes(probe, involved)

    applied: List[Tuple[str, str]] = []
    used_donors: set = set()
    for new_arr, donors in donor_candidates.items():
        for donor_arr in donors:
            if donor_arr in used_donors:
                continue
            if new_arr not in scopes or donor_arr not in scopes:
                continue
            if scopes[new_arr] != scopes[donor_arr]:
                continue
            if not _edge_order_safe(probe, new_arr, donor_arr):
                continue
            try:
                _apply_reuse(sdfg, new_arr, donor_arr)
                applied.append((new_arr, donor_arr))
                used_donors.add(donor_arr)
                break
            except ValueError:
                pass
    return applied


# ---------------------------------------------------------------------------
# Layer 4: cross-size chain reuse (buffer_reuse_cross_pass)
# ---------------------------------------------------------------------------

def _greedy_cross_size_scan(
    liveness: List[_LivenessEvent],
) -> List[Tuple[str, str, int]]:
    """Cross-size, cross-dtype best-fit sub-allocation scan (strict bump allocator).

    On each :class:`_AllocEntry`, find the available donor block with the
    smallest *remaining* capacity (``total - used``) that still fits the
    consumer and satisfies the alignment constraint
    (``block.dtype.bytes >= consumer.dtype.bytes``).  Assign the consumer at
    the block's current bump pointer; advance the pointer by
    ``consumer.size_bytes``.  Multiple consumers can share one donor block at
    non-overlapping byte offsets.

    A donor block enters the pool when its :class:`_FreeEntry` is processed.
    Consumer free events are silently dropped — storage is owned by the donor.

    :param liveness: Output of :func:`_extract_liveness`.
    :returns: List of ``(consumer, donor, offset_bytes)`` triples in liveness order.
    """
    # pool: each entry is [total_size, dtype, donor_name, used_bytes]  (mutable)
    pool: List[list] = []
    plan: List[Tuple[str, str, int]] = []
    is_consumer: set = set()

    for event in liveness:
        if isinstance(event, _AllocEntry):
            best_idx: Optional[int] = None
            best_remaining: Optional[int] = None
            for idx, (total, dt, _nm, used) in enumerate(pool):
                remaining = total - used
                if remaining < event.size_bytes:
                    continue
                if dt.bytes < event.dtype.bytes:
                    continue
                if best_remaining is None or remaining < best_remaining:
                    best_idx = idx
                    best_remaining = remaining

            if best_idx is not None:
                _total, _dt, donor, used = pool[best_idx]
                plan.append((event.array_name, donor, used))
                pool[best_idx][3] = used + event.size_bytes
                is_consumer.add(event.array_name)

        elif isinstance(event, _FreeEntry):
            if event.array_name not in is_consumer:
                pool.append([event.size_bytes, event.dtype, event.array_name, 0])

    return plan



def _apply_arena_reuse(
    sdfg: SDFG,
    new_arr: str,
    donor_arr: str,
    offset_bytes: int = 0,
) -> None:
    """Cross-dtype/cross-size chain reuse: rebind *new_arr* into *donor_arr*'s
    heap block at the given byte offset.

    Emits a 3-tuple reuse entry ``[new_arr, donor_arr, offset_bytes]`` which
    the codegen lowers to ``(T_new*)((char*)donor_ptr + offset)``. The donor
    retains ownership of the heap block: its ``free`` entry is moved to the
    edge that previously held the consumer's ``free`` entry, so the typed
    ``delete[] donor_ptr`` fires only after both arrays are done. The
    consumer's ``free`` entry is removed because the consumer never owned
    the storage.
    """
    for name in (new_arr, donor_arr):
        if name not in sdfg.arrays:
            raise ValueError(f"'{name}' not found in sdfg.arrays")
        if not sdfg.arrays[name].transient:
            raise ValueError(f"'{name}' is not a transient data container")

    needs_explicit = [
        name for name in (new_arr, donor_arr)
        if sdfg.arrays[name].lifetime != dtypes.AllocationLifetime.Explicit
    ]
    if needs_explicit:
        make_explicit(sdfg, needs_explicit)

    all_edges = list(sdfg.all_interstate_edges(recursive=True))

    # Remember where consumer's free lived so we can re-home the donor's free.
    new_free_edge = None
    for edge in all_edges:
        if new_arr in edge.data.free:
            new_free_edge = edge
            break

    # Replace consumer's alloc with a 3-tuple reuse entry.
    for edge in all_edges:
        while new_arr in edge.data.alloc:
            edge.data.alloc.remove(new_arr)
            edge.data.reuse.append([new_arr, donor_arr, int(offset_bytes)])

    # Strip consumer's free entries — consumer never owns the heap block.
    for edge in all_edges:
        while new_arr in edge.data.free:
            edge.data.free.remove(new_arr)

    # Re-home donor's free to the consumer's old free site, so the typed
    # delete[] donor_ptr happens after the consumer's last use.
    if new_free_edge is not None:
        for edge in all_edges:
            while donor_arr in edge.data.free:
                edge.data.free.remove(donor_arr)
        if donor_arr not in new_free_edge.data.free:
            new_free_edge.data.free.append(donor_arr)


def _resolve_donor_root(sdfg: SDFG, donor: str) -> Tuple[str, int]:
    """Walk *donor*'s reuse chain to its storage-owning ancestor.

    If a previous arena pair already made *donor* a reuse consumer
    (entry ``[donor, X, off]`` exists somewhere in the SDFG), follow
    the chain to the root allocator and accumulate offsets. Returns
    ``(root, total_offset)``. If *donor* is not a reuse consumer,
    returns ``(donor, 0)``.

    Required because the cross-size apply moves the donor's ``free``
    entry; if donor is itself a view into an earlier donor, the typed
    ``delete[]`` must fire on the original heap block, not on the view.
    """
    seen: set = set()
    current = donor
    total = 0
    while current not in seen:
        seen.add(current)
        next_arr: Optional[str] = None
        next_off = 0
        for edge in sdfg.all_interstate_edges(recursive=True):
            for entry in edge.data.reuse:
                if entry[0] == current:
                    next_arr = entry[1]
                    next_off = int(entry[2]) if len(entry) >= 3 else 0
                    break
            if next_arr is not None:
                break
        if next_arr is None:
            return current, total
        total += next_off
        current = next_arr
    return current, total


def _edge_order_safe(probe: SDFG, new_arr: str, donor_arr: str) -> bool:
    """Return True iff donor's last-use block executes *strictly before*
    consumer's first-use block in topological order, within the same parent
    ControlFlowRegion.

    Strict ordering is required: if both blocks are the same, donor is
    accessed *inside* the block whose entry edge fires the rebind, so the
    donor's typed accesses race against consumer's writes through the
    aliased pointer. This catches the case where ``make_explicit`` placed
    alloc/free edges around a block whose internal accesses CATS knew
    were disjoint, but whose block-level boundary still overlaps.
    """
    import networkx as _nx

    new_alloc: List[Tuple[Any, Any]] = []
    donor_free: List[Tuple[Any, Any]] = []
    for ns in probe.all_sdfgs_recursive():
        for region in ns.all_control_flow_regions(recursive=True):
            for edge in region.edges():
                if new_arr in edge.data.alloc:
                    new_alloc.append((region, edge))
                if donor_arr in edge.data.free:
                    donor_free.append((region, edge))

    if len(new_alloc) != 1 or len(donor_free) != 1:
        return False

    new_region, new_alloc_edge = new_alloc[0]
    donor_region, donor_free_edge = donor_free[0]
    if new_region is not donor_region:
        return False

    g = new_region.nx
    if not _nx.is_directed_acyclic_graph(g):
        return False
    order = {n: i for i, n in enumerate(_nx.topological_sort(g))}

    new_first = new_alloc_edge.dst
    donor_last = donor_free_edge.src
    if new_first not in order or donor_last not in order:
        return False
    return order[donor_last] < order[new_first]


def buffer_reuse_cross_pass(
    sdfg: SDFG,
    symbols: Dict[str, int],
    symbol_max_vals: Dict[str, int],
) -> List[Tuple[str, str]]:
    """Cross-size, cross-dtype buffer reuse with bump-allocator sub-allocation.

    A single donor block may serve multiple consumers at non-overlapping byte
    offsets (strict bump allocator: the offset only increases within one block).
    Consumer ``c`` at offset ``k`` is assigned ``(T_c*)((char*)donor + k)``.

    Donor selection: best-fit by remaining capacity so larger donors are
    preserved for larger future consumers.

    :param sdfg:            Root SDFG to optimise in-place.
    :param symbols:         Concrete symbol values for CATS size computation.
    :param symbol_max_vals: Upper bounds for unresolved symbols.
    :returns: List of ``(consumer, donor)`` pairs actually applied.
    """
    import copy as _copy
    from collections import defaultdict

    # CATS compiles with its own CPUCodeGen; isolate it on a renamed copy
    # so the working SDFG's build folder stays clean.
    probe = _copy.deepcopy(sdfg)
    probe._name = probe._name + "_cats_probe"
    liveness = _extract_liveness(probe, symbols, symbol_max_vals)
    reuse_plan = _greedy_cross_size_scan(liveness)

    # Build a free-event index for ordering consumers of the same donor.
    free_idx: Dict[str, int] = {
        e.array_name: i
        for i, e in enumerate(liveness)
        if isinstance(e, _FreeEntry)
    }

    # Scope and edge-order checks use an unmodified second probe.
    check_probe = _copy.deepcopy(sdfg)
    involved = {n for triple in reuse_plan for n in triple[:2]
                if n in check_probe.arrays}
    for name in list(involved):
        if check_probe.arrays[name].lifetime != dtypes.AllocationLifetime.Explicit:
            try:
                make_explicit(check_probe, [name])
            except ValueError:
                involved.discard(name)
    scopes = _collect_scopes(check_probe, involved)

    # Group consumers by donor; filter on scope/safety.
    consumers_by_donor: Dict[str, List[Tuple[str, int, int]]] = defaultdict(list)
    for consumer, donor, offset in reuse_plan:
        if consumer not in scopes or donor not in scopes:
            continue
        if scopes[consumer] != scopes[donor]:
            continue
        if not _edge_order_safe(check_probe, consumer, donor):
            continue
        consumers_by_donor[donor].append(
            (consumer, offset, free_idx.get(consumer, 0))
        )

    # Apply each donor's consumers sorted by free order (earliest first) so
    # _apply_arena_reuse leaves the donor's free at the latest consumer's site.
    applied: List[Tuple[str, str]] = []
    for donor, consumers in consumers_by_donor.items():
        consumers.sort(key=lambda t: t[2])
        for consumer, offset, _ in consumers:
            actual_donor, chain_off = _resolve_donor_root(sdfg, donor)
            try:
                _apply_arena_reuse(sdfg, consumer, actual_donor,
                                   offset_bytes=chain_off + offset)
                applied.append((consumer, donor))
            except ValueError:
                pass

    return applied
