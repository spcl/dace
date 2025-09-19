# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
The tests in this file verify that proper nested SDFG signatures and calls
are generated for different possible data combinations.
"""

import copy
import dace
import pytest

import dace.data as dt

N = 20  # Array size
M = 10  # View size
# Data combinations to test
# Parent SDFG data, Nested SDFG data, Subset
data_combinations = [
    (dt.Scalar, dt.Scalar, None),
    (dt.Scalar, dt.Scalar, dace.subsets.Range.from_string(f"0:1")),
    (dt.Array, dt.Scalar, dace.subsets.Range.from_string(f"{N//4}:{N//4+1}")),
    (dt.Array, dt.Array, dace.subsets.Range.from_string(f"0:{N}")),
    (dt.Array, dt.Array, dace.subsets.Range.from_string(f"{N//4}:{(3*N)//4}")),
    (dt.Structure, dt.Structure, None),
    (dt.Structure, dt.Structure, dace.subsets.Range.from_string("0:1")),
    (dt.ContainerArray, dt.Structure, dace.subsets.Range.from_string(f"{N//4}:{N//4+1}")),
    (dt.ContainerArray, dt.ContainerArray, dace.subsets.Range.from_string(f"0:{N}")),
    (dt.ContainerArray, dt.ContainerArray, dace.subsets.Range.from_string(f"{N//4}:{(3*N)//4}")),
    (dt.ArrayView, dt.Scalar, dace.subsets.Range.from_string(f"{M//4}:{M//4+1}")),
    (dt.ArrayView, dt.Array, dace.subsets.Range.from_string(f"0:{M}")),
    (dt.ArrayView, dt.Array, dace.subsets.Range.from_string(f"{M//4}:{(3*M)//4}")),
    (dt.StructureView, dt.Structure, None),
    (dt.StructureView, dt.Structure, dace.subsets.Range.from_string("0:1")),
    (dt.ContainerView, dt.Structure, dace.subsets.Range.from_string(f"{M//4}:{M//4+1}")),
    (dt.ContainerView, dt.ContainerArray, dace.subsets.Range.from_string(f"0:{M}")),
    (dt.ContainerView, dt.ContainerArray, dace.subsets.Range.from_string(f"{M//4}:{(3*M)//4}")),
]

# Parent SDFG data nesting combinations to test
parent_data_nesting = [
    (),
    (dt.Structure, ),
    (
        dt.Structure,
        dt.Structure,
    ),
    (
        dt.Structure,
        dt.Structure,
        dt.Structure,
    ),
    (
        dt.ContainerArray,
        dt.Structure,
    ),
    (
        dt.ContainerArray,
        dt.Structure,
        dt.Structure,
    ),
    (
        dt.ContainerArray,
        dt.Structure,
        dt.Structure,
        dt.Structure,
    ),
    (
        dt.Structure,
        dt.ContainerArray,
        dt.Structure,
    ),
    (
        dt.Structure,
        dt.ContainerArray,
        dt.Structure,
        dt.Structure,
    ),
    (
        dt.Structure,
        dt.Structure,
        dt.ContainerArray,
        dt.Structure,
    ),
]

FooBar = dt.Structure(name="FooBar",
                      members={
                          "foo": dt.Scalar(dace.float32),
                          "bar": dt.Array(shape=(N, ), dtype=dace.float32)
                      })

# dtypes
dtypes = {
    dt.Scalar: dace.float32,
    dt.Array: dace.float32,
    dt.Structure: FooBar,
    dt.ContainerArray: FooBar,
    dt.ArrayView: dace.float32,
    dt.StructureView: FooBar,
    dt.ContainerView: FooBar,
}

# constructors
constructors = {
    dt.Scalar: lambda dtype: dt.Scalar(dtype=dtype),
    dt.Array: lambda shape, dtype: dt.Array(shape=shape, dtype=dtype),
    dt.Structure: lambda stype: copy.deepcopy(stype),
    dt.ContainerArray: lambda shape, stype: dt.ContainerArray(shape=shape, stype=stype),
    dt.ArrayView: lambda shape, dtype: dt.ArrayView(shape=shape, dtype=dtype),
    dt.StructureView: lambda stype: dt.View.view(stype),
    dt.ContainerView: lambda shape, stype: dt.ContainerView(shape=shape, stype=stype),
}


def make_sdfg(data_combo, parent_nesting):
    parent_dtype, nested_dtype, subset = data_combo

    sdfg = dace.SDFG("nested_sdfg_signature_call_test")
    state = sdfg.add_state("parent_state")

    if issubclass(parent_dtype, dt.View):
        parent_classes = parent_dtype.__bases__
        non_view_class = next(c for c in parent_classes if c is not dt.View)
    else:
        non_view_class = None

    if issubclass(parent_dtype, dt.Array):
        if non_view_class is not None:
            non_view_desc = constructors[non_view_class]((N, ), dtypes[non_view_class])
            view_desc = constructors[parent_dtype]((M, ), dtypes[non_view_class])
            sdfg.add_datadesc("A", non_view_desc)
            sdfg.add_datadesc("Av", view_desc)
            non_view_access = state.add_access("A")
            parent_access = state.add_access("Av")
            state.add_edge(non_view_access, None, parent_access, "views",
                           dace.Memlet(data="A", subset=f"{N//4}:{N//4+M}", other_subset=f"0:{M}"))
        else:
            non_view_desc = constructors[parent_dtype]((N, ), dtypes[parent_dtype])
            sdfg.add_datadesc("A", non_view_desc)
            parent_access = state.add_access("A")
    else:
        desc = constructors[parent_dtype](dtypes[parent_dtype])
        sdfg.add_datadesc("A", desc)
        parent_access = state.add_access("A")

    nested_sdfg = dace.SDFG("nested")
    _ = nested_sdfg.add_state("nested_state")  # Necessary to avoid errors

    if issubclass(nested_dtype, dt.Array):
        nested_desc = constructors[nested_dtype]((*subset.size(), ), dtypes[nested_dtype])
    else:
        nested_desc = constructors[nested_dtype](dtypes[nested_dtype])
    nested_sdfg.add_datadesc("B", nested_desc)

    nested_sdfg_node = state.add_nested_sdfg(nested_sdfg, {"B"}, {})
    state.add_edge(parent_access, None, nested_sdfg_node, "B", dace.Memlet(data="A", subset=subset))

    return sdfg


@pytest.mark.parametrize("data_combo", data_combinations)
@pytest.mark.parametrize("parent_nesting", [None])
def test_nested_sdfg_signature_call(data_combo, parent_nesting):

    sdfg = make_sdfg(data_combo, parent_nesting)

    with dace.config.set_temporary("optimizer", "automatic_simplification", value=False):
        sdfg.compile()

    assert True


if __name__ == "__main__":
    for data_combo in data_combinations:
        parent_nesting = None
        # for parent_nesting in parent_data_nesting:
        print(f"Testing combination: {data_combo} with parent nesting {parent_nesting}...")
        test_nested_sdfg_signature_call(data_combo, parent_nesting)
    print("All combinations passed!")
