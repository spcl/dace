#!/usr/bin/env python3

import dace
from dace.memlet import Memlet
import dace.libraries.blas as blas
import numpy as np
import scipy

vec_width = 4
vtype = dace.vector(dace.float32, vec_width)

n = dace.symbol("n")

# ---------- ----------
# Create Inner Nested SDFG
# ---------- ----------

# ---------- ----------
# SETUP GRAPH
# ---------- ----------
vecAdd_sdfg = dace.SDFG('vecAdd_graph')
vecAdd_state = vecAdd_sdfg.add_state()

# ---------- ----------
# MEMORY LOCATIONS
# ---------- ----------
vlen = n / vec_width
vecAdd_sdfg.add_array('_x', shape=[vlen], dtype=vtype)
vecAdd_sdfg.add_array('_y', shape=[vlen], dtype=vtype)
vecAdd_sdfg.add_array('_res', shape=[vlen], dtype=vtype)

x_in = vecAdd_state.add_read('_x')
y_in = vecAdd_state.add_read('_y')
z_out = vecAdd_state.add_write('_res')

# ---------- ----------
# COMPUTE
# ---------- ----------
vecMap_entry, vecMap_exit = vecAdd_state.add_map(
    'vecAdd_map', dict(i='0:n/{}'.format(vec_width)))

vecAdd_tasklet = vecAdd_state.add_tasklet('vecAdd_task', {'x_con', 'y_con'},
                                          {'z_con'}, 'z_con = x_con + y_con')

vecAdd_state.add_memlet_path(x_in,
                             vecMap_entry,
                             vecAdd_tasklet,
                             dst_conn='x_con',
                             memlet=dace.Memlet.simple(x_in.data, 'i'))

vecAdd_state.add_memlet_path(y_in,
                             vecMap_entry,
                             vecAdd_tasklet,
                             dst_conn='y_con',
                             memlet=dace.Memlet.simple(y_in.data, 'i'))

vecAdd_state.add_memlet_path(vecAdd_tasklet,
                             vecMap_exit,
                             z_out,
                             src_conn='z_con',
                             memlet=dace.Memlet.simple(z_out.data, 'i'))

# ---------- ----------
# Create Outer SDFG
# ---------- ----------

test_sdfg = dace.SDFG("saxpy_test")
test_state = test_sdfg.add_state("test_state")

test_sdfg.add_array('x1', shape=[n], dtype=dace.float32)
test_sdfg.add_array('y1', shape=[n], dtype=dace.float32)
test_sdfg.add_array('z1', shape=[n], dtype=dace.float32)

x_in1 = test_state.add_read('x1')
y_in1 = test_state.add_read('y1')
z_out1 = test_state.add_write('z1')

nested_sdfg = test_state.add_nested_sdfg(vecAdd_sdfg, test_sdfg, {"_x", "_y"},
                                         {"_res"})

test_state.add_memlet_path(x_in1,
                           nested_sdfg,
                           dst_conn='_x',
                           memlet=Memlet.simple(x_in1, "0:n"))
test_state.add_memlet_path(y_in1,
                           nested_sdfg,
                           dst_conn='_y',
                           memlet=Memlet.simple(y_in1, "0:n"))

test_state.add_memlet_path(nested_sdfg,
                           z_out1,
                           src_conn='_res',
                           memlet=Memlet.simple(z_out1, "0:n"))


def test_nested_vectorization():
    # Compile
    compiledSDFG = test_sdfg.compile()

    # Run and verify
    testSize = 96
    a = np.random.randint(testSize, size=testSize).astype(np.float32)
    b = np.random.randint(testSize, size=testSize).astype(np.float32)
    b_ref = np.copy(b)
    scaling = np.float32(1.0)
    c = np.zeros(testSize).astype(np.float32)

    compiledSDFG(x1=a, y1=b, a1=scaling, z1=c, n=np.int32(a.shape[0]))
    ref_result = scipy.linalg.blas.saxpy(a, b, a=scaling)

    diff = np.linalg.norm(c - ref_result)
    print('Difference:', diff)
    assert diff < 1e-8


if __name__ == '__main__':
    test_nested_vectorization()
