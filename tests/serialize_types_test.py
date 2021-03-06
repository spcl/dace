# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.properties import Property, make_properties
from dace.serialize import all_properties_to_json, set_properties_from_json
import json


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


def test_serialize_int_float():
    obj = MyObject(1.0)
    assert obj.float_prop == 1.0
    json_obj = obj.to_json()
    # Force casting to int
    json_obj['float_prop'] = int(json_obj['float_prop'])
    obj = MyObject.from_json(json_obj)
    assert obj.float_prop == 1.0


if __name__ == '__main__':
    test_serialize_int_float()
