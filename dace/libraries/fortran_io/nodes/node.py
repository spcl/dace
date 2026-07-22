# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared base and helpers for the Fortran I/O library nodes."""
from typing import List, Tuple

from dace import data, dtypes
from dace.sdfg import nodes

#: DaCe base type -> (``dace_fio_*`` entry suffix, C scalar type) for the
#: shipped wrappers.  The suffix selects the typed ``read``/``write`` entry; the
#: C type is the pointer cast at the call site (so int64 vs ``long long`` and
#: similar width spellings never trip ``-Werror``).
_FIO_TYPES = {
    dtypes.float64: ("f64", "double"),
    dtypes.float32: ("f32", "float"),
    dtypes.int32: ("i32", "int"),
    dtypes.int64: ("i64", "long long"),
}


def fio_type(dtype) -> Tuple[str, str]:
    """Resolve the wrapper suffix and C cast type for an item ``dtype``.

    :param dtype: the connected descriptor's DaCe data type.
    :returns: ``(suffix, c_type)`` for the matching ``dace_fio_*`` entry.
    :raises NotImplementedError: if no wrapper exists for ``dtype``.
    """
    base = dtype.base_type
    if base not in _FIO_TYPES:
        raise NotImplementedError(f"fortran_io has no wrapper for dtype {dtype}")
    return _FIO_TYPES[base]


class FortranIONode(nodes.LibraryNode):
    """Abstract base for a Fortran external-file I/O library node.

    Fortran I/O has observable side effects (it touches the file system), so
    these nodes report :pyattr:`has_side_effects` and must never be removed as
    dead code even when, like ``WRITE``, they have no output connectors.
    """

    @property
    def has_side_effects(self) -> bool:
        return True

    def _ordered_items(self, sdfg, state, prefix: str, edges_in: bool) -> List[Tuple[str, object, str, bool]]:
        """Resolve the connected I/O items in connector order.

        :param prefix: connector prefix (``"_in_"`` for WRITE, ``"_out_"`` for
            READ).
        :param edges_in: ``True`` to walk input edges (WRITE items), ``False``
            for output edges (READ items).
        :returns: ``(connector, descriptor, count, is_value)`` per item, where
            ``count`` is the element-count expression and ``is_value`` marks a
            connector DaCe emits by value (a true :class:`~dace.data.Scalar` or
            any single-element memlet, so the call site takes its address) vs.
            a multi-element array (already a pointer).
        :raises ValueError: if a declared item connector has no edge.
        """
        if edges_in:
            edges = {e.dst_conn: e for e in state.in_edges(self) if e.dst_conn}
        else:
            edges = {e.src_conn: e for e in state.out_edges(self) if e.src_conn}
        items = []
        for i in range(self.num_items):
            conn = f"{prefix}{i}"
            edge = edges.get(conn)
            if edge is None:
                raise ValueError(f"{type(self).__name__} '{self.name}': item connector '{conn}' is not connected")
            desc = sdfg.arrays[edge.data.data]
            num_elements = edge.data.subset.num_elements()
            is_value = isinstance(desc, data.Scalar) or num_elements == 1
            count = "*".join(str(s) for s in edge.data.subset.size_exact()) or "1"
            items.append((conn, desc, count, is_value))
        return items
