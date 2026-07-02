# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy as np


def test_pythonclass_scalar_rebind_and_new_field_codegen():

    class Holder:
        scalar: dace.float64

    PythonHolder = dace.data.PythonClass.from_class(Holder)

    @dace.program
    def prog(holder: PythonHolder, A: dace.float64[4]):
        holder.scalar = A[0]
        holder.new_field = A[1]

    sdfg = prog.to_sdfg(simplify=False)

    assert 'holder.scalar' in sdfg.arrays
    assert 'holder.new_field' in sdfg.arrays
    assert isinstance(sdfg.arrays['holder.scalar'], dace.data.Scalar)
    assert isinstance(sdfg.arrays['holder.new_field'], dace.data.Scalar)

    assignment_targets = {'holder.scalar', 'holder.new_field'}
    assignment_states = []
    for state in sdfg.nodes():
        for edge in state.edges():
            if getattr(edge.dst, 'data', None) in assignment_targets:
                assignment_states.append(state)
                break

    assert len(assignment_states) == 2
    assert all(not any(isinstance(node, dace.nodes.Tasklet) for node in state.nodes()) for state in assignment_states)

    code = sdfg.generate_code()[0].clean_code
    assert 'dace_set_pyobject_attr<double>(holder, "scalar",' in code
    assert 'dace_set_pyobject_attr<double>(holder, "new_field",' in code

    holder = Holder()
    holder.scalar = -1.0
    values = np.array([3.5, 7.25, 0.0, 0.0], dtype=np.float64)

    prog(holder, values)

    assert holder.scalar == values[0]
    assert holder.new_field == values[1]


def test_pythonclass_literal_scalar_assignment_uses_tasklet_output_codegen():

    class Holder:
        scalar: dace.float64

    PythonHolder = dace.data.PythonClass.from_class(Holder)

    @dace.program
    def prog(holder: PythonHolder):
        holder.new_field = 4.25

    sdfg = prog.to_sdfg(simplify=False)

    assert 'holder.new_field' in sdfg.arrays
    assert isinstance(sdfg.arrays['holder.new_field'], dace.data.Scalar)

    assignment_states = []
    for state in sdfg.nodes():
        for edge in state.edges():
            if getattr(edge.dst, 'data', None) == 'holder.new_field':
                assignment_states.append(state)
                break

    assert len(assignment_states) == 1
    assert sum(isinstance(node, dace.nodes.Tasklet) for node in assignment_states[0].nodes()) == 1
    assert next(node for node in assignment_states[0].nodes()
                if isinstance(node, dace.nodes.Tasklet)).language == dace.dtypes.Language.Python
    assert not any(state.label.startswith('pythonclass_attr_barrier_') for state in sdfg.nodes())

    code = sdfg.generate_code()[0].clean_code
    assert 'dace_set_pyobject_attr<double>(holder, "new_field",' in code

    holder = Holder()
    holder.scalar = -1.0

    prog(holder)

    assert holder.new_field == 4.25


def test_pythonclass_array_field_access_codegen():

    class Holder:
        data: dace.float64[4]

    PythonHolder = dace.data.PythonClass.from_class(Holder)

    @dace.program
    def prog(holder: PythonHolder):
        for i in range(4):
            holder.data[i] = holder.data[i] + 1.0

    code = prog.to_sdfg(simplify=False).generate_code()[0].clean_code
    assert 'dace_get_pyobject_attr_ptr<double>(holder, "data")' in code

    holder = Holder()
    holder.data = np.array([1.0, 2.5, -3.0, 0.25], dtype=np.float64)
    expected = holder.data + 1.0

    prog(holder)

    assert np.allclose(holder.data, expected)


def test_pythonclass_nested_array_field_access_codegen():

    class Inner:
        data: dace.float64[4]

    class Outer:
        inner: Inner

    PythonOuter = dace.data.PythonClass.from_class(Outer)

    @dace.program
    def prog(holder: PythonOuter):
        for i in range(4):
            holder.inner.data[i] = holder.inner.data[i] + 1.0

    code = prog.to_sdfg(simplify=False).generate_code()[0].clean_code
    assert 'dace_get_pyobject_attr_ptr<double>(holder, "inner.data")' in code

    class InnerRuntime:
        pass

    class OuterRuntime:
        pass

    holder = OuterRuntime()
    holder.inner = InnerRuntime()
    holder.inner.data = np.array([1.0, 2.5, -3.0, 0.25], dtype=np.float64)
    expected = holder.inner.data + 1.0

    prog(holder)

    assert np.allclose(holder.inner.data, expected)


def test_pythonclass_new_array_field_assignment_uses_reference_set():

    class Holder:
        scalar: dace.float64

    PythonHolder = dace.data.PythonClass.from_class(Holder)

    @dace.program
    def prog(holder: PythonHolder, A: dace.float64[4], out: dace.float64[4]):
        holder.new_data = A
        out[:] = holder.new_data[:]

    sdfg = prog.to_sdfg(simplify=False)

    assert 'holder.new_data' in sdfg.arrays
    assert isinstance(sdfg.arrays['holder.new_data'], dace.data.ArrayReference)

    set_edges = []
    assignment_state = None
    for state in sdfg.nodes():
        for edge in state.edges():
            if getattr(edge.dst, 'data', None) == 'holder.new_data' and edge.dst_conn == 'set':
                set_edges.append(edge)
                assignment_state = state

    assert len(set_edges) == 1
    assert assignment_state is not None
    assert not any(isinstance(node, dace.nodes.Tasklet) for node in assignment_state.nodes())
    assert not any(state.label.startswith('pythonclass_attr_barrier_') for state in sdfg.nodes())

    code = sdfg.generate_code()[0].clean_code
    assert 'dace_get_pyobject_attr_ptr<double>(holder, "new_data") = A;' not in code
    assert 'dace_get_pyobject_attr_ptr<double>(holder, "new_data")' in code
    assert 'dace_set_pyobject_attr_array<double>(holder, "new_data", A, __shape,' in code
    assert '__strides, 1);' in code

    holder = Holder()
    holder.scalar = -1.0
    values = np.array([1.0, 2.5, -3.0, 0.25], dtype=np.float64)
    out = np.zeros_like(values)

    prog(holder, values, out)

    assert isinstance(holder.new_data, np.ndarray)
    assert np.allclose(out, values)
    assert np.allclose(holder.new_data, values)

    values[0] = 9.5
    assert holder.new_data[0] == values[0]


if __name__ == '__main__':
    test_pythonclass_scalar_rebind_and_new_field_codegen()
    test_pythonclass_literal_scalar_assignment_uses_tasklet_output_codegen()
    test_pythonclass_array_field_access_codegen()
    test_pythonclass_nested_array_field_access_codegen()
    test_pythonclass_new_array_field_assignment_uses_reference_set()
