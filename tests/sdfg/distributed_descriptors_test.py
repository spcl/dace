import copy

import dace
from dace.data.distributed import ProcessGrid, RedistrArray, SubArray


def test_distributed_descriptors_are_data_descriptors():
    pgrid = ProcessGrid('__pgrid', False, [2, 3])
    subgrid = ProcessGrid('__pgrid_0', True, parent_grid=pgrid, color=[False, True])
    subarray = SubArray('__subarray', dace.int32, [16, 8], [8, 4], pgrid.name)
    redistr = RedistrArray('__rdistrarray', subarray.name, subarray.name)

    for desc in (pgrid, subgrid, subarray, redistr):
        assert isinstance(desc, dace.data.Data)
        assert desc.transient
        assert desc.is_equivalent(copy.deepcopy(desc))

    assert subgrid.parent_grid == pgrid.name
    assert tuple(subgrid.shape) == (3, )
    assert tuple(subarray.shape) == (16, 8)
    assert tuple(subarray.subshape) == (8, 4)


def test_sdfg_distributed_descriptor_storage_and_compatibility_views():
    sdfg = dace.SDFG('distributed_descriptors')

    pgrid = sdfg.add_pgrid([2, 2])
    subgrid = sdfg.add_pgrid(parent_grid=pgrid, color=[False, True])
    subarray = sdfg.add_subarray(dace.int32, [16, 16], [8, 8], pgrid)
    redistr = sdfg.add_rdistrarray(subarray, subarray)

    assert not hasattr(sdfg, '_pgrids')
    assert not hasattr(sdfg, '_subarrays')
    assert not hasattr(sdfg, '_rdistrarrays')

    assert isinstance(sdfg.arrays[pgrid], ProcessGrid)
    assert isinstance(sdfg.arrays[subgrid], ProcessGrid)
    assert isinstance(sdfg.arrays[subarray], SubArray)
    assert isinstance(sdfg.arrays[redistr], RedistrArray)

    assert sdfg.process_grids[pgrid] is sdfg.arrays[pgrid]
    assert sdfg.process_grids[subgrid] is sdfg.arrays[subgrid]
    assert sdfg.subarrays[subarray] is sdfg.arrays[subarray]
    assert sdfg.rdistrarrays[redistr] is sdfg.arrays[redistr]

    assert pgrid not in sdfg.arglist()
    assert subgrid not in sdfg.arglist()
    assert subarray not in sdfg.arglist()
    assert redistr not in sdfg.arglist()


def test_distributed_descriptor_json_roundtrip():
    sdfg = dace.SDFG('distributed_descriptor_roundtrip')
    pgrid = sdfg.add_pgrid([2, 2])
    subarray = sdfg.add_subarray(dace.float64, [16, 16], [8, 8], pgrid)
    redistr = sdfg.add_rdistrarray(subarray, subarray)

    loaded = dace.SDFG.from_json(sdfg.to_json())

    assert isinstance(loaded.arrays[pgrid], ProcessGrid)
    assert isinstance(loaded.arrays[subarray], SubArray)
    assert isinstance(loaded.arrays[redistr], RedistrArray)
    assert pgrid in loaded.process_grids
    assert subarray in loaded.subarrays
    assert redistr in loaded.rdistrarrays
    assert not hasattr(loaded, '_pgrids')
    assert not hasattr(loaded, '_subarrays')
    assert not hasattr(loaded, '_rdistrarrays')


def test_legacy_distributed_side_store_json_migrates_to_arrays():
    pgrid = ProcessGrid('__pgrid', False, [2, 2])
    subarray = SubArray('__subarray', dace.float64, [16, 16], [8, 8], pgrid.name)
    redistr = RedistrArray('__rdistrarray', subarray.name, subarray.name)

    json_obj = dace.SDFG('legacy_distributed_descriptors').to_json()
    json_obj['attributes']['_pgrids'] = {pgrid.name: dace.serialize.to_json(pgrid)}
    json_obj['attributes']['_subarrays'] = {subarray.name: dace.serialize.to_json(subarray)}
    json_obj['attributes']['_rdistrarrays'] = {redistr.name: dace.serialize.to_json(redistr)}

    loaded = dace.SDFG.from_json(json_obj)

    assert isinstance(loaded.arrays[pgrid.name], ProcessGrid)
    assert isinstance(loaded.arrays[subarray.name], SubArray)
    assert isinstance(loaded.arrays[redistr.name], RedistrArray)
    assert loaded.process_grids[pgrid.name] is loaded.arrays[pgrid.name]
    assert loaded.subarrays[subarray.name] is loaded.arrays[subarray.name]
    assert loaded.rdistrarrays[redistr.name] is loaded.arrays[redistr.name]
    assert not hasattr(loaded, '_pgrids')
    assert not hasattr(loaded, '_subarrays')
    assert not hasattr(loaded, '_rdistrarrays')
