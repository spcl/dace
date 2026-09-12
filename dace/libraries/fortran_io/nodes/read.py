# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""``READ`` library node: list-directed read from a file into its outputs.

Lowers a Fortran ``read`` statement (fused with its ``open``/``close``) to a
C++ tasklet that calls the shipped ``dace_fio_*`` wrappers: open ``filename``
for reading, read each connected item (``_out_0`` ... ``_out_{num_items-1}``)
in order through the real Fortran runtime, then close.
"""
import dace.library
import dace.properties
from dace import dtypes
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

from .node import FortranIONode, fio_type
from .write import _c_string
from .. import environments


@dace.library.expansion
class ExpandReadFortranIO(ExpandTransformation):

    environments = [environments.FortranIO]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        items = node._ordered_items(parent_sdfg, parent_state, "_out_", edges_in=False)
        path = _c_string(node.filename)
        lines = [f'int _u = dace_fio_open("{path}", {len(node.filename.encode())}, 0);']
        for conn, desc, count, is_value in items:
            suffix, ctype = fio_type(desc.dtype)
            if is_value:
                lines.append(f'dace_fio_read_{suffix}(_u, ({ctype} *)&{conn});')
            else:
                lines.append(f'dace_fio_read_{suffix}_arr(_u, ({ctype} *){conn}, {count});')
        lines.append("dace_fio_close(_u);")
        return nodes.Tasklet(node.name,
                             node.in_connectors,
                             node.out_connectors,
                             "\n".join(lines),
                             language=dtypes.Language.CPP,
                             side_effects=True)


@dace.library.node
class Read(FortranIONode):
    """Read ``num_items`` connected outputs from ``filename`` (list-directed)."""

    implementations = {"FortranIO": ExpandReadFortranIO}
    default_implementation = "FortranIO"

    filename = dace.properties.Property(dtype=str, default="", desc="Input file path")
    num_items = dace.properties.Property(dtype=int, default=0, desc="Number of items read")

    def __init__(self, name, filename: str = "", num_items: int = 0, **kwargs):
        super().__init__(name, inputs=set(), outputs={f"_out_{i}" for i in range(num_items)}, **kwargs)
        self.filename = filename
        self.num_items = num_items
