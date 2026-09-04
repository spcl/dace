# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""``NamelistRead`` library node: read named members from a Fortran namelist.

A fixed Fortran ``NAMELIST`` read needs its group declared at compile time, so
this node does not generate Fortran -- it expands (like the other nodes) to a
pure C++ tasklet that calls the shipped generic ``dace_nml_*`` C functions:
open the ``(file, group)``, fetch each member by name into its output
connector, then close.  The member names (in connector order ``_out_0`` ...
``_out_{num_items-1}``) come from :pyattr:`members`, which the bridge fills
from the namelist group descriptor.
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
class ExpandNamelistReadFortranIO(ExpandTransformation):

    environments = [environments.FortranIO]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        items = node._ordered_items(parent_sdfg, parent_state, "_out_", edges_in=False)
        if len(node.members) != len(items):
            raise ValueError(f"NamelistRead '{node.name}': {len(node.members)} member names "
                             f"for {len(items)} connected outputs")
        path, group = _c_string(node.filename), _c_string(node.group)
        lines = [
            f'int _h = dace_nml_open("{path}", {len(node.filename.encode())}, '
            f'"{group}", {len(node.group.encode())});'
        ]
        for (conn, desc, count, is_value), member in zip(items, node.members):
            suffix, ctype = fio_type(desc.dtype)
            name_arg = f'"{_c_string(member)}", {len(member.encode())}'
            if is_value:
                lines.append(f'dace_nml_get_{suffix}(_h, {name_arg}, ({ctype} *)&{conn});')
            else:
                lines.append(f'dace_nml_get_{suffix}_arr(_h, {name_arg}, ({ctype} *){conn}, {count});')
        lines.append("dace_nml_close(_h);")
        return nodes.Tasklet(node.name,
                             node.in_connectors,
                             node.out_connectors,
                             "\n".join(lines),
                             language=dtypes.Language.CPP,
                             side_effects=True)


@dace.library.node
class NamelistRead(FortranIONode):
    """Read ``members`` of namelist ``group`` from ``filename`` into outputs."""

    implementations = {"FortranIO": ExpandNamelistReadFortranIO}
    default_implementation = "FortranIO"

    filename = dace.properties.Property(dtype=str, default="", desc="Namelist file path")
    group = dace.properties.Property(dtype=str, default="", desc="Namelist group name")
    members = dace.properties.ListProperty(element_type=str, default=[], desc="Member names, in output-connector order")

    def __init__(self, name, filename: str = "", group: str = "", members=None, **kwargs):
        members = list(members or [])
        super().__init__(name, inputs=set(), outputs={f"_out_{i}" for i in range(len(members))}, **kwargs)
        self.filename = filename
        self.group = group
        self.members = members

    @property
    def num_items(self) -> int:
        return len(self.members)
