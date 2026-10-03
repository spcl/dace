# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""``WRITE`` library node: list-directed write of its inputs to a file.

Lowers a Fortran ``write`` statement (fused with its ``open``/``close``) to a
C++ tasklet that calls the shipped ``dace_fio_*`` wrappers: open ``filename``
for writing, write each connected item (``_in_0`` ... ``_in_{num_items-1}``) in
order through the real Fortran runtime, then close.
"""
import dace.library
import dace.properties
from dace import dtypes
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

from .node import FortranIONode, fio_type
from .. import environments


def _c_string(text: str) -> str:
    """Render ``text`` as a C string literal body (escaping ``\\`` and ``"``)."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


@dace.library.expansion
class ExpandWriteFortranIO(ExpandTransformation):

    environments = [environments.FortranIO]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        items = node._ordered_items(parent_sdfg, parent_state, "_in_", edges_in=True)
        path = _c_string(node.filename)
        lines = [f'int _u = dace_fio_open("{path}", {len(node.filename.encode())}, 1);']
        for conn, desc, count, is_value in items:
            suffix, ctype = fio_type(desc.dtype)
            if is_value:
                lines.append(f'dace_fio_write_{suffix}(_u, (const {ctype} *)&{conn});')
            else:
                lines.append(f'dace_fio_write_{suffix}_arr(_u, (const {ctype} *){conn}, {count});')
        lines.append("dace_fio_close(_u);")
        return nodes.Tasklet(node.name,
                             node.in_connectors,
                             node.out_connectors,
                             "\n".join(lines),
                             language=dtypes.Language.CPP,
                             side_effects=True)


@dace.library.node
class Write(FortranIONode):
    """Write ``num_items`` connected inputs to ``filename`` (list-directed)."""

    implementations = {"FortranIO": ExpandWriteFortranIO}
    default_implementation = "FortranIO"

    filename = dace.properties.Property(dtype=str, default="", desc="Output file path")
    num_items = dace.properties.Property(dtype=int, default=0, desc="Number of items written")

    def __init__(self, name, filename: str = "", num_items: int = 0, **kwargs):
        super().__init__(name, inputs={f"_in_{i}" for i in range(num_items)}, outputs=set(), **kwargs)
        self.filename = filename
        self.num_items = num_items
