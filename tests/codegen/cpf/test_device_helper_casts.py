# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The helper text a device unit pastes in holds no C-style cast to an arithmetic type.

The helpers are written once and pasted into every unit that calls them, so a ``(long long)i`` or a
``(unsigned)blocks`` in one of them is in every render, where ``-Wold-style-cast`` reports it. Their
narrowings are ``static_cast``, their widenings implicit, and their index type is the unit's ``int64_t``.
"""

import re

from dace import cpf_lowering
from dace.cpf_lowering import Dialect

#: A C-style cast to an arithmetic type, followed by the operand it casts (``sizeof(int64_t)`` is not one).
ARITHMETIC_CAST = re.compile(
    r"\((?:unsigned )?(?:long long|size_t|unsigned|int|int64_t|uint64_t|double|float)\)\s*[\w(]"
)


def test_device_helpers_hold_no_c_style_arithmetic_cast():
    """Every helper a HIP or CUDA unit can paste in: the inline definitions, the optional blocks, and the core."""
    texts = {"DEVICE_CORE": cpf_lowering.DEVICE_CORE}
    for dialect in (Dialect.STANDALONE_HIP, Dialect.STANDALONE_CUDA):
        texts[f"DEVICE_TYPES[{dialect.name}]"] = cpf_lowering.DEVICE_TYPES[dialect]
        texts.update({f"{dialect.name}:{name}": text for name, text in cpf_lowering.DEVICE_BLOCKS[dialect].items()})
        texts.update(
            {f"{dialect.name}:{name}": text for name, text in cpf_lowering.device_definitions(dialect).items()}
        )
    casts = {name: ARITHMETIC_CAST.findall(text) for name, text in texts.items()}
    assert not {name: found for name, found in casts.items() if found}, "C-style casts in device helper text"


if __name__ == "__main__":
    test_device_helpers_hold_no_c_style_arithmetic_cast()
