# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Array-descriptor mutation helpers (add or rebuild ``dace.data.Array`` descriptors on an SDFG)."""
from typing import Any, Iterable, Set, Tuple

import dace
from dace import typeclass


def replace_arrays_with_new_shape(sdfg: dace.SDFG, array_namelist: Set[str], new_shape: Tuple[Any],
                                  new_type: typeclass) -> None:
    """
    Replace existing array descriptors with a new shape (and optionally a new dtype).

    Also rewrites tasklet-adjacent memlets on truly-reshaped arrays so the
    subset matches the new shape: the old subsets were sized against the
    original (typically ``(LEN_1D,)`` or scalar) and become invalid against
    the new typically-W-wide shape — leaving them as-is produces OOB scalar
    accesses (e.g. ``_cond[tile_i]`` on a ``bool[W]`` buffer when ``tile_i``
    is ``0, 8, 16, ...``).  The replacement subset spans every dim of
    ``new_shape`` (``[0 : dim-1]`` per dim), which is the appropriate
    "full vector read/write" the W-wide consumers expect.

    :param sdfg: SDFG to mutate.
    :param array_namelist: Array names to reshape.
    :param new_shape: Target shape.
    :param new_type: New dtype, or ``None`` to preserve the original.
    """
    # Track which arrays *actually* changed shape: rewriting memlets on
    # arrays whose old descriptor already matched ``new_shape`` is a no-op
    # but can clobber existing valid subsets (e.g. point accesses to a
    # connector array that was already ``(W,)``).  Compare via stringified
    # shape so sympy/int mismatches don't false-positive (``(8,) == (8,)``
    # via str even when one side carries sympy Integer wrappers).
    reshaped: Set[str] = set()
    new_shape_str = tuple(str(d) for d in new_shape)
    for arr_name in array_namelist:
        arr = sdfg.arrays[arr_name]
        old_shape_str = tuple(str(d) for d in arr.shape)
        sdfg.remove_data(arr_name, validate=False)
        sdfg.add_array(name=arr_name,
                       shape=new_shape,
                       storage=arr.storage,
                       dtype=arr.dtype if new_type is None else new_type,
                       location=arr.location,
                       transient=arr.transient,
                       lifetime=arr.lifetime,
                       debuginfo=arr.debuginfo,
                       allow_conflicts=arr.allow_conflicts,
                       find_new_name=False,
                       alignment=arr.alignment,
                       may_alias=arr.may_alias)
        if old_shape_str != new_shape_str:
            reshaped.add(arr_name)

    if not reshaped:
        return

    # For arrays whose shape actually changed (typically ``(LEN,)`` or
    # scalar → ``(W,)``), rewrite the memlets whose subset became OOB
    # against the new W-wide shape.  We scope this narrowly:
    # only memlets directly connected to a ``Tasklet`` (either side)
    # need the rewrite, because the codegen reads the subset to decide
    # whether the connector is a scalar reference (``T&``) or a pointer
    # (``T*``) — a scalar subset like ``[tile_i]`` on a now-``T[W]``
    # array generates ``T&`` which then explodes inside the vectorized
    # tasklet body (s1161 ``bool&`` vs ``bool*`` for ``_cond_c_index``).
    # AccessNode↔AccessNode copies and NSDFG connector edges keep their
    # explicit subsets (e.g. ``[i + 1]`` on accumulator passes, point
    # accesses into already-W-wide connectors like ``__tmp_70_18_r``).
    import copy as _copy

    full_subset = dace.subsets.Range([(dace.symbolic.SymExpr(0), dace.symbolic.SymExpr(d) - dace.symbolic.SymExpr(1),
                                       dace.symbolic.SymExpr(1)) for d in new_shape])
    for s in sdfg.all_states():
        for e in s.edges():
            if e.data is None or e.data.data is None:
                continue
            if e.data.data not in reshaped:
                continue
            # Only rewrite memlets touching a Tasklet, where the connector
            # typing depends on the subset shape.
            is_tasklet_edge = (isinstance(e.src, dace.nodes.Tasklet) or isinstance(e.dst, dace.nodes.Tasklet))
            if not is_tasklet_edge:
                continue
            e.data.subset = _copy.deepcopy(full_subset)
            if e.data.other_subset is not None:
                e.data.other_subset = None


def copy_arrays_with_a_new_shape(sdfg: dace.SDFG, array_namelist: Set[str], new_shape: Tuple[Any],
                                 name_suffix: str) -> None:
    """
    Create copies of existing arrays with a new shape and a name suffix.

    :param sdfg: SDFG to mutate.
    :param array_namelist: Source array names.
    :param new_shape: Target shape for the new arrays.
    :param name_suffix: Suffix appended to each new array name.
    """
    for arr_name in array_namelist:
        arr = sdfg.arrays[arr_name]
        sdfg.add_array(name=arr_name + name_suffix,
                       shape=new_shape,
                       storage=arr.storage,
                       dtype=arr.dtype,
                       location=arr.location,
                       transient=arr.transient,
                       lifetime=arr.lifetime,
                       debuginfo=arr.debuginfo,
                       allow_conflicts=arr.allow_conflicts,
                       find_new_name=False,
                       alignment=arr.alignment,
                       may_alias=arr.may_alias)


def add_transient_arrays_from_list(sdfg: dace.SDFG, arr_name_shape_storage_dtype: Iterable[Tuple[str, Any, Any,
                                                                                                 Any]]) -> None:
    """
    Add transient arrays to an SDFG from a list of ``(name, shape, storage, dtype)`` tuples.

    :param sdfg: SDFG to mutate.
    :param arr_name_shape_storage_dtype: Iterable of array specifications.
    """

    for arr_name, shape, storage, dtype in arr_name_shape_storage_dtype:
        sdfg.add_array(
            name=arr_name,
            shape=shape,
            storage=storage,
            dtype=dtype,
            transient=True,
            find_new_name=False,
        )
