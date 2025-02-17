# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import json
import numpy as np

import dace
from dace.properties import Property, make_properties, ListProperty
from dace.serialize import all_properties_to_json, set_properties_from_json


@make_properties
class MyObject(object):
    float_prop = Property(dtype=float, default=0.0)

    def __init__(self, p: float):
        super().__init__()
        self.float_prop = p

    def to_json(self):
        return all_properties_to_json(self)

    @staticmethod
    def from_json(json_obj, context=None):
        ret = MyObject(0.0)
        set_properties_from_json(ret, json_obj, context=context)
        return ret


@make_properties
class MyListObject(object):
    list_prop = ListProperty(element_type=int)

    def __init__(self, p):
        super().__init__()
        self.list_prop = p

    def to_json(self):
        return all_properties_to_json(self)

    @staticmethod
    def from_json(json_obj, context=None):
        ret = MyListObject([])
        set_properties_from_json(ret, json_obj, context=context)
        return ret


def test_serialize_int_float():
    obj = MyObject(1.0)
    assert obj.float_prop == 1.0
    json_obj = obj.to_json()
    # Force casting to int
    json_obj['float_prop'] = int(json_obj['float_prop'])
    obj = MyObject.from_json(json_obj)
    assert obj.float_prop == 1.0


def test_serialize_list_int64():
    obj = MyListObject([])
    obj.list_prop.append(np.int64(2))
    assert obj.list_prop == [2]
    json_obj = obj.to_json()

    obj = MyListObject.from_json(json.loads(json.dumps(json_obj)))
    assert obj.list_prop == [2]


def test_serialize_infinity():

    @dace.program
    def reduction_infinity_1(a: dace.float64[3]):
        return a.max()
    
    sdfg = reduction_infinity_1.to_sdfg()
    json_string = json.dumps(sdfg.to_json())
    assert(json_string.find('Infinity') == -1)

    @dace.program
    def reduction_infinity_2(a: dace.float64[3]):
        return np.max(a)
    
    sdfg = reduction_infinity_1.to_sdfg()
    json_string = json.dumps(sdfg.to_json())
    assert(json_string.find('Infinity') == -1)


if __name__ == '__main__':
    test_serialize_int_float()
    test_serialize_list_int64()
    test_serialize_infinity()
