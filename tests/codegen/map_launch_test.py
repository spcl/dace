from typing import Union
import dace
import numpy as np
import copy
import sys
import time
import pytest

try:
    import cupy as cp
except ImportError:
    cp = None


def _make_sdfg(
    lb: Union[str, int],
    ub: Union[str, int],
    on_gpu: bool,
) -> tuple[dace.SDFG, dace.nodes.MapEntry]:
    sdfg = dace.SDFG("map_test_" + ("gpu" if on_gpu else "cpu") + f"_{int(time.time())}")
    state = sdfg.add_state(is_start_block=True)

    for name in ["i0", "o0"]:
        sdfg.add_array(
            name,
            shape=(10, ),
            dtype=dace.float64,
            transient=False,
        )
        if on_gpu:
            sdfg.arrays[name].storage = dace.dtypes.StorageType.GPU_Global

    for b in [lb, ub]:
        if isinstance(b, str):
            sdfg.add_symbol(b, dace.int32)
        else:
            assert isinstance(b, int)

    _, me, _ = state.add_mapped_tasklet(
        "map",
        map_ranges={"__i": f"{lb}:{ub}"},
        inputs={"__in": dace.Memlet("i0[__i + 5]")},
        code="__out = __in + 10.0",
        outputs={"__out": dace.Memlet("o0[__i + 5]")},
        external_edges=True,
    )
    if on_gpu:
        me.map.schedule = dace.dtypes.ScheduleType.GPU_Device
        # If you change this value, for example to `(32, 1, 1)`, to prove something,
        #  make sure to also modify the bounds below, because you have to make some
        #  changes there as well.
        me.map.gpu_block_size = (1, 1, 1)

    sdfg.validate()
    return sdfg, me


def _run_test(
    lb: Union[str, int],
    ub: Union[str, int],
    on_gpu: bool,
    **kwargs,
):
    xp = cp if on_gpu else np

    if xp is None:
        raise RuntimeError("Could not find cupy")

    sdfg, me = _make_sdfg(lb=lb, ub=ub, on_gpu=on_gpu)
    args = {
        "i0": xp.array(xp.random.rand(10), dtype=xp.float64, copy=True),
        "o0": xp.array(xp.random.rand(10), dtype=xp.float64, copy=True),
    }
    args.update(kwargs)
    org_args = copy.deepcopy(args)

    csdfg = sdfg.compile()
    csdfg(**args)

    # Again because of the nasty selection of the bounds, we should not see any
    #  change in the value.
    assert all(np.allclose(args[k], org_args[k]) for k in ["i0", "o0"])


def test_launch_cpu_zero_sized_map():
    with pytest.warns(match="zero or negative sized range"):
        _run_test(1, 1, on_gpu=False)


def test_launch_cpu_sym_zero_sized_map():
    _run_test("a", "b", on_gpu=False, a=1, b=1)


def test_launch_cpu_negative_sized_map():
    with pytest.warns(match="zero or negative sized range"):
        _run_test(1, 0, on_gpu=False)


def test_launch_cpu_sym_negative_sized_map():
    _run_test("a", "b", on_gpu=False, a=1, b=0)


@pytest.mark.gpu
def test_launch_gpu_zero_sized_map():
    _run_test(1, 1, on_gpu=True)


@pytest.mark.gpu
def test_launch_gpu_sym_zero_sized_map():
    _run_test("a", "b", on_gpu=True, a=1, b=1)


@pytest.mark.gpu
def test_launch_gpu_negative_sized_map():
    _run_test(1, 0, on_gpu=True)


@pytest.mark.gpu
def test_launch_gpu_sym_negative_sized_map():
    _run_test("a", "b", on_gpu=True, a=1, b=0)


if __name__ == "__main__":
    test_launch_cpu_zero_sized_map()
    test_launch_cpu_sym_zero_sized_map()
    test_launch_cpu_negative_sized_map()
    test_launch_cpu_sym_negative_sized_map()
    test_launch_gpu_zero_sized_map()
    test_launch_gpu_sym_zero_sized_map()
    test_launch_gpu_negative_sized_map()
    test_launch_gpu_sym_negative_sized_map()
