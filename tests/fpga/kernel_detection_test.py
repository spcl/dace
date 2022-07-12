# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# Tests for kernels detection

import dace
import numpy as np
from pathlib import Path
import pytest
import re
from dace.sdfg.utils import is_fpga_kernel
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.fpga_testing import fpga_test


def count_kernels(sdfg: dace.SDFG):
    '''
    Test utility functions: Counts the number of generated device kernels
    :param sdfg: Already compiled SDFG to count kernels for.
    :return: number of kernels
    '''

    import csv
    kernels = 0
    with open(Path(sdfg.build_folder) / "dace_files.csv", "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[1] == "device" and (row[-1].endswith("cpp") or row[-1].endswith("cl")):
                kernels = kernels + 1
    return kernels


@fpga_test()
def test_kernels_inside_component_0():
    '''
    Tests for kernels detection inside a single connected component.
    It computes z =(x+y) + (v+w)

    High-level overview:
     ┌───────────┐
     │ Add_Map_0 │
     └──────┬────┘
            │
     ┌───────────┐        ┌───────────┐
     │ Add_Map_1 │        │ Add_Map_2 │
     └──────┬────┘        └──────┬────┘
            │   ┌───────────┐    │
            └─► │ Add_Map_3 │◄───┘
                └───────────┘
    The 4 maps, should belong to three distinct kernels
    :return:
    '''

    @dace.program
    def kernels_inside_component_0(x: dace.float32[8], y: dace.float32[8], v: dace.float32[8], w: dace.float32[8],
                                   z: dace.float32[8]):
        tmp = (x + y) + v
        return tmp + (w + z)

    x = np.random.rand(8).astype(np.float32)
    y = np.random.rand(8).astype(np.float32)
    v = np.random.rand(8).astype(np.float32)
    w = np.random.rand(8).astype(np.float32)
    z = np.random.rand(8).astype(np.float32)

    sdfg = kernels_inside_component_0.to_sdfg()
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])

    for state in sdfg.states():
        if is_fpga_kernel(sdfg, state):
            state.instrument = dace.InstrumentationType.FPGA

    res = sdfg(x=x, y=y, v=v, w=w, z=z)
    assert count_kernels(sdfg) == 3
    assert np.allclose(res, x + y + v + w + z)

    report = sdfg.get_latest_report()
    assert len(re.findall(r"[0-9\.]+\s+[0-9\.]+\s+[0-9\.]+\s+[0-9\.]+", str(report))) == 5
    assert len(re.findall(r"Full FPGA .+ runtime", str(report))) == 2

    return sdfg


@fpga_test()
def test_kernels_inside_component_1():
    '''
    Tests for kernels detection inside a single connected component.
    The program computes:
    - z = alpha*((x+y) + (v+w))
    - t = beta*((x+y) + (v+w))

    High-level overview:
     ┌───────────┐        ┌───────────┐
     │ Add_Map_0 │        │ Add_Map_1 │
     └──────┬────┘        └──────┬────┘
            │   ┌───────────┐    │
            └─► │ Add_Map_2 │◄───┘
            ────└───────────┘────
            │                   │
     ┌──────v────┐        ┌─────v─────┐
     │   Mul_3   │        │   Mul_4   │
     └───────────┘        └───────────┘

    The five Maps should belong to 5 distinct kernels

    '''

    @dace.program
    def kernels_inside_component_1(x: dace.float32[8], y: dace.float32[8], v: dace.float32[8], w: dace.float32[8],
                                   z: dace.float32[8], t: dace.float32[8], alpha: dace.float32, beta: dace.float32):
        tmp1 = x + y
        tmp2 = v + w
        tmp3 = tmp1 + tmp2
        z[:] = alpha * tmp3
        t[:] = beta * tmp3

    x = np.random.rand(8).astype(np.float32)
    y = np.random.rand(8).astype(np.float32)
    v = np.random.rand(8).astype(np.float32)
    w = np.random.rand(8).astype(np.float32)
    z = np.random.rand(8).astype(np.float32)
    t = np.random.rand(8).astype(np.float32)
    alpha = 1.0
    beta = 2.0

    sdfg = kernels_inside_component_1.to_sdfg()
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])
    program = sdfg.compile()
    assert count_kernels(sdfg) == 5
    program(x=x, y=y, v=v, w=w, z=z, t=t, alpha=alpha, beta=beta)
    ref_z = alpha * (x + y + v + w)
    ref_t = beta * (x + y + v + w)
    assert np.allclose(z, ref_z)
    assert np.allclose(t, ref_t)

    return sdfg


@fpga_test()
def test_kernels_inside_component_2():
    '''
    Tests for PEs detection inside a single Component.
    It computes z =(x+y) and t = (y+v)


    High-level overview:

        x            y         v
        │            │         │
     ┌──V────────<───┘────>────V──────┐
     │ Add_Map_0 │        │ Add_Map_1 │
     └───────────┘        └───────────┘

    Map_0 and Map_1 should belong to two distinct kernels

    :return:
    '''

    @dace.program
    def kernels_inside_component_2(x: dace.float32[8], y: dace.float32[8], v: dace.float32[8], z: dace.float32[8],
                                   t: dace.float32[8]):
        z[:] = x + y
        t[:] = y + v

    x = np.random.rand(8).astype(np.float32)
    y = np.random.rand(8).astype(np.float32)
    v = np.random.rand(8).astype(np.float32)
    z = np.random.rand(8).astype(np.float32)
    t = np.random.rand(8).astype(np.float32)

    sdfg = kernels_inside_component_2.to_sdfg()
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])
    program = sdfg.compile()

    # NOTE: here we have only one kernel since subgraph detection already
    # detects two PEs
    assert count_kernels(sdfg) == 1
    program(x=x, y=y, v=v, t=t, z=z)
    assert np.allclose(z, x + y)
    assert np.allclose(t, v + y)

    return sdfg


@fpga_test()
def test_kernels_lns_inside_component():
    '''
    Tests for kernels detection inside a single connected component where we
    have multiple library nodes.

    It computes z =(x+y) + (v+w)

    High-level overview:
     ┌───────────┐        ┌───────────┐
     │  Matmul_0 │        │  Matmul_1 │
     └──────┬────┘        └──────┬────┘
            │   ┌───────────┐    │
            └─► │   Dot_2   │◄───┘
                └───────────┘
    '''

    # (Provisional) Disable unique function
    unique_functions_conf = dace.config.Config.get('compiler', 'unique_functions')
    dace.config.Config.set('compiler', 'unique_functions', value="none")

    @dace.program
    def kernels_lns_inside_component(A: dace.float32[8, 8], x: dace.float32[8], B: dace.float32[8, 8],
                                     y: dace.float32[8]):
        tmp1 = A @ x
        tmp2 = B @ y
        return np.dot(tmp1, tmp2)

    A = np.random.rand(8, 8).astype(np.float32)
    B = np.random.rand(8, 8).astype(np.float32)
    x = np.random.rand(8).astype(np.float32)
    y = np.random.rand(8).astype(np.float32)

    sdfg = kernels_lns_inside_component.to_sdfg()
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])
    program = sdfg.compile()

    assert count_kernels(sdfg) == 3
    z = program(A=A, x=x, B=B, y=y)
    ref = np.dot(A @ x, B @ y)
    assert np.allclose(z, ref)
    dace.config.Config.set('compiler', 'unique_functions', value=unique_functions_conf)

    return sdfg


@fpga_test()
def test_kernels_inside_components_0():
    '''
    Tests for kernels detection in two distinct connected components.
    The program computes:
    z = (x+y) + (v+w)
    zz = (xx+yy) + (vv+ww)

    High-level overview: the two connected components are the same and look
    like  the following
     ┌───────────┐        ┌───────────┐
     │ Add_Map_0 │        │ Add_Map_1 │
     └──────┬────┘        └──────┬────┘
            │   ┌───────────┐    │
            └─► │ Add_Map_2 │◄───┘
                └───────────┘
    The three maps, should belong to three distinct kernels

    '''

    @dace.program
    def kernels_inside_components_0(x: dace.float32[8], y: dace.float32[8], v: dace.float32[8], w: dace.float32[8],
                                    xx: dace.float32[8], yy: dace.float32[8], vv: dace.float32[8], ww: dace.float32[8]):
        z = (x + y) + (v + w)
        zz = (xx + yy) + (vv + ww)
        return z, zz

    x = np.random.rand(8).astype(np.float32)
    y = np.random.rand(8).astype(np.float32)
    v = np.random.rand(8).astype(np.float32)
    w = np.random.rand(8).astype(np.float32)
    xx = np.random.rand(8).astype(np.float32)
    yy = np.random.rand(8).astype(np.float32)
    vv = np.random.rand(8).astype(np.float32)
    ww = np.random.rand(8).astype(np.float32)

    sdfg = kernels_inside_components_0.to_sdfg()
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])
    program = sdfg.compile()

    assert count_kernels(sdfg) == 6
    z, zz = program(x=x, y=y, v=v, w=w, xx=xx, yy=yy, vv=vv, ww=ww)
    assert np.allclose(z, x + y + v + w)
    assert np.allclose(zz, xx + yy + vv + ww)

    return sdfg


@fpga_test()
def test_kernels_inside_components_multiple_states():
    '''
    Tests for kernels detection in two distinct states.
    It computes
    z = (x+y) + (v+w)
    zz = (xx+yy) + (vv+ww)

    High-level overview: the two connected components are the same and look
    like  the following
     ┌───────────┐        ┌───────────┐
     │ Add_Map_0 │        │ Add_Map_1 │
     └──────┬────┘        └──────┬────┘
            │   ┌───────────┐    │
            └─► │ Add_Map_2 │◄───┘
                └───────────┘
    The three maps, should belong to three distinct kernels
    :return:
    '''

    def make_sdfg(dtype=dace.float32):
        sdfg = dace.SDFG("multiple_kernels_multiple_states")
        n = dace.symbol("size")

        input_data = ["x", "y", "v", "w", "xx", "yy", "vv", "ww"]
        output_data = ["z", "zz"]
        device_transient_data = ["device_tmp0", "device_tmp1", "device_tmp2", "device_tmp3"]

        for d in input_data + output_data:
            sdfg.add_array(d, shape=[n], dtype=dtype)
            sdfg.add_array(f"device_{d}",
                           shape=[n],
                           dtype=dtype,
                           storage=dace.dtypes.StorageType.FPGA_Global,
                           transient=True)

        for d in device_transient_data:
            sdfg.add_array(d, shape=[n], dtype=dtype, storage=dace.dtypes.StorageType.FPGA_Global, transient=True)

        ###########################################################################
        # Copy data to FPGA

        copy_in_state = sdfg.add_state("copy_to_device")

        for d in input_data:
            in_host = copy_in_state.add_read(d)
            in_device = copy_in_state.add_read(f"device_{d}")

            copy_in_state.add_memlet_path(in_host, in_device, memlet=dace.Memlet(f"{d}[0:{n}]"))

        ###########################################################################
        # Copy data from FPGA
        copy_out_state = sdfg.add_state("copy_to_host")

        for d in output_data:
            out_host = copy_out_state.add_write(d)
            out_device = copy_out_state.add_read(f"device_{d}")

            copy_out_state.add_memlet_path(out_device, out_host, memlet=dace.Memlet(f"{d}[0:{n}]"))

        ########################################################################
        # FPGA, First State

        fpga_state_0 = sdfg.add_state("fpga_state_0")

        x_in = fpga_state_0.add_read("device_x")
        y_in = fpga_state_0.add_read("device_y")
        v_in = fpga_state_0.add_read("device_v")
        w_in = fpga_state_0.add_read("device_w")
        device_tmp0 = fpga_state_0.add_access("device_tmp0")
        device_tmp1 = fpga_state_0.add_access("device_tmp1")
        z_out = fpga_state_0.add_write("device_z")

        # x + y
        vecMap_entry00, vecMap_exit00 = fpga_state_0.add_map('vecAdd_map00',
                                                             dict(i=f'0:{n}'),
                                                             schedule=dace.dtypes.ScheduleType.FPGA_Device)

        vecAdd_tasklet00 = fpga_state_0.add_tasklet('vec_add_task00', ['x_con', 'y_con'], ['z_con'],
                                                    'z_con = x_con + y_con')

        fpga_state_0.add_memlet_path(x_in,
                                     vecMap_entry00,
                                     vecAdd_tasklet00,
                                     dst_conn='x_con',
                                     memlet=dace.Memlet("device_x[i]"))

        fpga_state_0.add_memlet_path(y_in,
                                     vecMap_entry00,
                                     vecAdd_tasklet00,
                                     dst_conn='y_con',
                                     memlet=dace.Memlet("device_y[i]"))

        fpga_state_0.add_memlet_path(vecAdd_tasklet00,
                                     vecMap_exit00,
                                     device_tmp0,
                                     src_conn='z_con',
                                     memlet=dace.Memlet("device_tmp0[i]"))

        # v + w

        vecMap_entry01, vecMap_exit01 = fpga_state_0.add_map('vecAdd_map01',
                                                             dict(i=f'0:{n}'),
                                                             schedule=dace.dtypes.ScheduleType.FPGA_Device)

        vecAdd_tasklet01 = fpga_state_0.add_tasklet('vec_add_task01', ['x_con', 'y_con'], ['z_con'],
                                                    'z_con = x_con + y_con')

        fpga_state_0.add_memlet_path(v_in,
                                     vecMap_entry01,
                                     vecAdd_tasklet01,
                                     dst_conn='x_con',
                                     memlet=dace.Memlet(f"device_v[i]"))

        fpga_state_0.add_memlet_path(w_in,
                                     vecMap_entry01,
                                     vecAdd_tasklet01,
                                     dst_conn='y_con',
                                     memlet=dace.Memlet(f"device_w[i]"))

        fpga_state_0.add_memlet_path(vecAdd_tasklet01,
                                     vecMap_exit01,
                                     device_tmp1,
                                     src_conn='z_con',
                                     memlet=dace.Memlet(f"device_tmp1[i]"))

        # tmp0 + tmp 1

        vecMap_entry02, vecMap_exit02 = fpga_state_0.add_map('vecAdd_map02',
                                                             dict(i=f'0:{n}'),
                                                             schedule=dace.dtypes.ScheduleType.FPGA_Device)

        vecAdd_tasklet02 = fpga_state_0.add_tasklet('vec_add_task02', ['x_con', 'y_con'], ['z_con'],
                                                    'z_con = x_con + y_con')

        fpga_state_0.add_memlet_path(device_tmp0,
                                     vecMap_entry02,
                                     vecAdd_tasklet02,
                                     dst_conn='x_con',
                                     memlet=dace.Memlet("device_tmp0[i]"))

        fpga_state_0.add_memlet_path(device_tmp1,
                                     vecMap_entry02,
                                     vecAdd_tasklet02,
                                     dst_conn='y_con',
                                     memlet=dace.Memlet("device_tmp1[i]"))

        fpga_state_0.add_memlet_path(vecAdd_tasklet02,
                                     vecMap_exit02,
                                     z_out,
                                     src_conn='z_con',
                                     memlet=dace.Memlet("device_z[i]"))
        ########################################################################
        # FPGA, Second State

        fpga_state_1 = sdfg.add_state("fpga_state_1")

        xx_in = fpga_state_1.add_read("device_xx")
        yy_in = fpga_state_1.add_read("device_yy")
        vv_in = fpga_state_1.add_read("device_vv")
        ww_in = fpga_state_1.add_read("device_ww")
        device_tmp2 = fpga_state_1.add_access("device_tmp2")
        device_tmp3 = fpga_state_1.add_access("device_tmp3")
        zz_out = fpga_state_1.add_write("device_zz")

        # xx + yy
        vecMap_entry10, vecMap_exit10 = fpga_state_1.add_map('vecAdd_map10',
                                                             dict(i=f'0:{n}'),
                                                             schedule=dace.dtypes.ScheduleType.FPGA_Device)

        vecAdd_tasklet10 = fpga_state_1.add_tasklet('vec_add_task10', ['x_con', 'y_con'], ['z_con'],
                                                    'z_con = x_con + y_con')

        fpga_state_1.add_memlet_path(xx_in,
                                     vecMap_entry10,
                                     vecAdd_tasklet10,
                                     dst_conn='x_con',
                                     memlet=dace.Memlet("device_xx[i]"))

        fpga_state_1.add_memlet_path(yy_in,
                                     vecMap_entry10,
                                     vecAdd_tasklet10,
                                     dst_conn='y_con',
                                     memlet=dace.Memlet("device_yy[i]"))

        fpga_state_1.add_memlet_path(vecAdd_tasklet10,
                                     vecMap_exit10,
                                     device_tmp2,
                                     src_conn='z_con',
                                     memlet=dace.Memlet("device_tmp2[i]"))

        # vv + ww
        vecMap_entry11, vecMap_exit11 = fpga_state_1.add_map('vecAdd_map11',
                                                             dict(i=f'0:{n}'),
                                                             schedule=dace.dtypes.ScheduleType.FPGA_Device)

        vecAdd_tasklet11 = fpga_state_1.add_tasklet('vec_add_task11', ['x_con', 'y_con'], ['z_con'],
                                                    'z_con = x_con + y_con')

        fpga_state_1.add_memlet_path(vv_in,
                                     vecMap_entry11,
                                     vecAdd_tasklet11,
                                     dst_conn='x_con',
                                     memlet=dace.Memlet(f"device_vv[i]"))

        fpga_state_1.add_memlet_path(ww_in,
                                     vecMap_entry11,
                                     vecAdd_tasklet11,
                                     dst_conn='y_con',
                                     memlet=dace.Memlet(f"device_ww[i]"))

        fpga_state_1.add_memlet_path(vecAdd_tasklet11,
                                     vecMap_exit11,
                                     device_tmp3,
                                     src_conn='z_con',
                                     memlet=dace.Memlet(f"device_tmp3[i]"))

        # tmp2 + tmp 3

        vecMap_entry12, vecMap_exit12 = fpga_state_1.add_map('vecAdd_map12',
                                                             dict(i=f'0:{n}'),
                                                             schedule=dace.dtypes.ScheduleType.FPGA_Device)

        vecAdd_tasklet12 = fpga_state_1.add_tasklet('vec_add_task12', ['x_con', 'y_con'], ['z_con'],
                                                    'z_con = x_con + y_con')

        fpga_state_1.add_memlet_path(device_tmp2,
                                     vecMap_entry12,
                                     vecAdd_tasklet12,
                                     dst_conn='x_con',
                                     memlet=dace.Memlet("device_tmp2[i]"))

        fpga_state_1.add_memlet_path(device_tmp3,
                                     vecMap_entry12,
                                     vecAdd_tasklet12,
                                     dst_conn='y_con',
                                     memlet=dace.Memlet("device_tmp3[i]"))

        fpga_state_1.add_memlet_path(vecAdd_tasklet12,
                                     vecMap_exit12,
                                     zz_out,
                                     src_conn='z_con',
                                     memlet=dace.Memlet("device_zz[i]"))

        ######################################
        # Interstate edges
        sdfg.add_edge(copy_in_state, fpga_state_0, dace.sdfg.sdfg.InterstateEdge())
        sdfg.add_edge(fpga_state_0, fpga_state_1, dace.sdfg.sdfg.InterstateEdge())
        sdfg.add_edge(fpga_state_1, copy_out_state, dace.sdfg.sdfg.InterstateEdge())

        #########
        # Validate
        sdfg.fill_scope_connectors()
        sdfg.validate()
        return sdfg

    x = np.random.rand(8).astype(np.float32)
    y = np.random.rand(8).astype(np.float32)
    v = np.random.rand(8).astype(np.float32)
    w = np.random.rand(8).astype(np.float32)
    z = np.random.rand(8).astype(np.float32)
    xx = np.random.rand(8).astype(np.float32)
    yy = np.random.rand(8).astype(np.float32)
    vv = np.random.rand(8).astype(np.float32)
    ww = np.random.rand(8).astype(np.float32)
    zz = np.random.rand(8).astype(np.float32)

    sdfg = make_sdfg()
    program = sdfg.compile()
    assert count_kernels(sdfg) == 6
    program(z=z, zz=zz, x=x, y=y, v=v, w=w, xx=xx, yy=yy, vv=vv, ww=ww, size=8)
    assert np.allclose(z, x + y + v + w)
    assert np.allclose(zz, xx + yy + vv + ww)

    return sdfg


if __name__ == "__main__":
    test_kernels_inside_component_0(None)
    test_kernels_inside_component_1(None)
    test_kernels_inside_component_2(None)
    test_kernels_lns_inside_component(None)
    test_kernels_inside_components_0(None)
    test_kernels_inside_components_multiple_states(None)
