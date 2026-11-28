# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The helper text a device unit pastes in holds no C-style cast to an arithmetic type.

The helpers are written once and pasted into every unit that calls them, so a ``(long long)i`` or a
``(unsigned)blocks`` in one of them is in every render, where ``-Wold-style-cast`` reports it. Their
narrowings are ``static_cast``, their widenings implicit, and their index type is the unit's ``int64_t``.
"""

import pathlib
import re

import dace
from dace import cpf_lowering
from dace.cpf_lowering import Dialect

#: A C-style cast to an arithmetic type, followed by the operand it casts (``sizeof(int64_t)`` is not one).
ARITHMETIC_CAST = re.compile(
    r"\((?:unsigned )?(?:long long|size_t|unsigned|int|int64_t|uint64_t|double|float)\)\s*[\w(]"
)


#: A ``long long`` pointer parameter: an answer written back through it is not the unit's ``int64_t``.
LONG_LONG_POINTER = re.compile(r"(?<!unsigned )\blong long\s*\*\s*\w+\s*[,)]")


def helper_texts() -> dict[str, str]:
    """Every helper a HIP or CUDA unit can paste in: the inline definitions, the optional blocks, and the core."""
    texts = {"DEVICE_CORE": cpf_lowering.DEVICE_CORE}
    for dialect in (Dialect.STANDALONE_HIP, Dialect.STANDALONE_CUDA):
        texts[f"DEVICE_TYPES[{dialect.name}]"] = cpf_lowering.DEVICE_TYPES[dialect]
        texts.update({f"{dialect.name}:{name}": text for name, text in cpf_lowering.DEVICE_BLOCKS[dialect].items()})
        texts.update(
            {f"{dialect.name}:{name}": text for name, text in cpf_lowering.device_definitions(dialect).items()}
        )
    return texts


def test_device_helpers_hold_no_c_style_arithmetic_cast():
    casts = {name: ARITHMETIC_CAST.findall(text) for name, text in helper_texts().items()}
    assert not {name: found for name, found in casts.items() if found}, "C-style casts in device helper text"


def test_device_helpers_and_runtime_headers_answer_through_int64_t():
    """find_first_index_device, detect_collision_device and arg_reduce write their index or flag through
    ``int64_t *``, in the CPF helper text and in the DaCe runtime headers the library expansions call."""
    texts = helper_texts()
    include = pathlib.Path(dace.__file__).parent / "runtime" / "include" / "dace"
    for header in ("cub_compat.cuh", "cuda/detect.cuh"):
        texts[header] = (include / header).read_text()
    pointers = {name: LONG_LONG_POINTER.findall(text) for name, text in texts.items()}
    assert not {name: found for name, found in pointers.items() if found}, "long long out-parameters"


if __name__ == "__main__":
    test_device_helpers_hold_no_c_style_arithmetic_cast()
    test_device_helpers_and_runtime_headers_answer_through_int64_t()
