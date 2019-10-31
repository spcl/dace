import json
import numpy as np


JSON_STORE_METADATA = True


class NumpySerializer:
    """ Helper class to load/store numpy arrays from JSON. """

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj is None:
            return None
        if json_obj['type'] != 'ndarray':
            raise TypeError('Object is not a numpy ndarray')

        if 'dtype' in json_obj:
            return np.array(json_obj['data'], dtype=json_obj['dtype'])

        return np.array(json_obj['data'])

    @staticmethod
    def to_json(obj):
        if obj is None:
            return None
        return {
            'type': 'ndarray',
            'data': obj.tolist(),
            'dtype': str(obj.dtype)
        }


_DACE_SERIALIZE_TYPES = {
    "ndarray": NumpySerializer
    # All classes annotated with the make_properties decorator will register
    # themselves here.
}


def get_serializer(type_name):
    return _DACE_SERIALIZE_TYPES[type_name]


# Decorator for objects that should be serializable, but don't call
# make_properties
def serializable(cls):
    _DACE_SERIALIZE_TYPES[cls.__name__] = cls
    return cls


def to_json(obj):
    if hasattr(obj, "to_json"):
        # If the object knows how to convert itself, let it. By calling the
        # method directly on the type, this works for both static and
        # non-static implementations of to_json.
        return type(obj).to_json(obj)
    elif type(obj) in {bool, int, float, list, dict, str}:
        # Some types are natively understood by JSON
        return obj
    elif isinstance(obj, np.ndarray):
        # Special case for external structures (numpy arrays)
        return NumpySerializer.to_json(obj)
    else:
        # If not available, go for the default str() representation
        return str(obj)


def from_json(obj, context=None):
    if not isinstance(obj, dict):
        return obj
    attr_type = None
    if "attributes" in obj:
        tmp = obj['attributes']
        if isinstance(tmp, dict):
            if "type" in tmp:
                attr_type = tmp['type']
        else:
            # The object was consumed previously
            try:
                obj['type']
            except KeyError:
                return tmp
            # If a type is available, the parent element must also be parsed accordingly

    try:
        t = obj['type']
    except KeyError:
        t = attr_type

    if t in Property.known_types():
        return (Property.known_types()[t]).from_json(
            obj, context=context)

    return obj


def all_properties_to_json(object_with_properties, store_metadata=False):
    retdict = {}
    for x, v in object_with_properties.properties():
        retdict[x.attr_name] = x.to_json(v)

        # Add the meta elements decoupled from key/value to facilitate value usage
        # (The meta is only used when rendering the values)
        if store_metadata:
            retdict['_meta_' + x.attr_name] = json.loads(x.meta_to_json(x))

    return retdict


def set_properties_from_json(object_with_properties, json_obj, context=None):

    try:
        attrs = json_obj['attributes']
    except KeyError:
        attrs = json_obj

    # Apply properties
    ps = dict(object_with_properties.__properties__)
    for prop_name, prop in ps.items():
        try:
            val = attrs[prop_name]
        except KeyError:
            raise KeyError("Missing property for object of type " +
                           type(object_with_propertes).__name__ + ":" +
                           prop_name)

        # Some properties only work when converted back from string.
        stringified = json.dumps(val, default=dump_json)
        val = prop.from_json(stringified, context)
        setattr(object_with_properties, field_name, val)
