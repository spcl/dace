# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Mapping between ``torch.dtype`` and DaCe typeclasses."""
from typing import Dict

import torch

import dace
from dace import dtypes


class UnsupportedDtypeError(NotImplementedError):
    pass


TORCH_TO_DACE: Dict[torch.dtype, dtypes.typeclass] = {
    torch.float32: dace.float32,
    torch.float64: dace.float64,
    torch.float16: dace.float16,
    torch.bfloat16: dace.bfloat16,
    torch.int8: dace.int8,
    torch.int16: dace.int16,
    torch.int32: dace.int32,
    torch.int64: dace.int64,
    torch.uint8: dace.uint8,
    torch.bool: dace.bool_,
    torch.complex64: dace.complex64,
    torch.complex128: dace.complex128,
}
for _tname, _dname in (('float8_e4m3fn', 'float8_e4m3fn'), ('float8_e5m2', 'float8_e5m2'), ('uint16', 'uint16'),
                       ('uint32', 'uint32'), ('uint64', 'uint64')):
    if hasattr(torch, _tname) and hasattr(dace, _dname):
        TORCH_TO_DACE[getattr(torch, _tname)] = getattr(dace, _dname)

DACE_TO_TORCH: Dict[dtypes.typeclass, torch.dtype] = {v: k for k, v in TORCH_TO_DACE.items()}


def to_dace_dtype(dtype: torch.dtype) -> dtypes.typeclass:
    try:
        return TORCH_TO_DACE[dtype]
    except KeyError:
        raise UnsupportedDtypeError(f'torch dtype {dtype} has no DaCe equivalent')


def to_torch_dtype(dtype: dtypes.typeclass) -> torch.dtype:
    try:
        return DACE_TO_TORCH[dtype]
    except KeyError:
        raise UnsupportedDtypeError(f'DaCe dtype {dtype} has no torch equivalent')


def is_floating(dtype: dtypes.typeclass) -> bool:
    return dtype in (dace.float16, dace.bfloat16, dace.float32, dace.float64) or dtype.type.__name__.startswith('float')


def is_boolean(dtype: dtypes.typeclass) -> bool:
    return dtype == dace.bool_
