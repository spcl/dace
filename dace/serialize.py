# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import aenum
import json
import numpy as np
import warnings
import dace.dtypes


class SerializableObject(object):

    json_obj = {}
    typename = None

    def __init__(self, json_obj={}, typename=None):
        self.json_obj = json_obj
        self.typename = typename

    def to_json(self):
        retval = self.json_obj
        retval['dace_unregistered'] = True
        return retval

    @staticmethod
    def from_json(json_obj, context=None, typename=None):
        return SerializableObject(json_obj, typename)


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
        return {'type': 'ndarray', 'data': obj.tolist(), 'dtype': str(obj.dtype)}


_DACE_SERIALIZE_TYPES = {
    # Define these manually, so dtypes can stay independent
    "pointer": dace.dtypes.pointer,
    "vector": dace.dtypes.vector,
    "callback": dace.dtypes.callback,
    "struct": dace.dtypes.struct,
    "ndarray": NumpySerializer,
    "DebugInfo": dace.dtypes.DebugInfo,
    "string": dace.dtypes.string,
    # All classes annotated with the make_properties decorator will register
    # themselves here.
}
# Also register each of the basic types
_DACE_SERIALIZE_TYPES.update({v.to_string(): v for v in dace.dtypes.DTYPE_TO_TYPECLASS.values()})


def get_serializer(type_name):
    return _DACE_SERIALIZE_TYPES[type_name]


# Decorator for objects that should be serializable, but don't call
# make_properties
def serializable(cls):
    _DACE_SERIALIZE_TYPES[cls.__name__] = cls
    return cls


def to_json(obj):
    if obj is None:
        return None
    elif hasattr(obj, "to_json"):
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
    elif isinstance(obj, aenum.Enum):
        # Store just the name of this key
        return obj._name_
    else:
        # If not available, go for the default str() representation
        return str(obj)


def from_json(obj, context=None, known_type=None):
    if not isinstance(obj, dict):
        if known_type is not None:
            # For enums, resolve using the type if known
            if issubclass(known_type, aenum.Enum) and isinstance(obj, str):
                return known_type[obj]
            # If we can, convert from string
            if isinstance(obj, str):
                if hasattr(known_type, "from_string"):
                    return known_type.from_string(obj)
        if isinstance(obj, list):
            return [from_json(o, context) for o in obj]
        # Otherwise we don't know what to do with this
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

    if known_type is not None and t is not None and t != known_type.__name__:
        raise TypeError("Type mismatch in JSON, found " + t + ", expected " + known_type.__name__)

    if t:
        try:
            deserialized = _DACE_SERIALIZE_TYPES[t].from_json(obj, context=context)
        except Exception as ex:
            warnings.warn(f'Failed to deserialize element, {type(ex).__name__}: {ex}')
            deserialized = SerializableObject.from_json(obj, context=context, typename=t)
        return deserialized

    # No type was found, so treat this as a regular dictionary
    return {from_json(k, context): from_json(v, context) for k, v in obj.items()}


def loads(*args, context=None, **kwargs):
    loaded = json.loads(*args, **kwargs)
    return from_json(loaded, context)


def dumps(*args, **kwargs):
    return json.dumps(*args, default=to_json, indent=2, **kwargs)


def load(*args, context=None, **kwargs):
    loaded = json.load(*args, **kwargs)
    return from_json(loaded, context)


def dump(*args, **kwargs):
    return json.dump(*args, default=to_json, indent=2, **kwargs)


def all_properties_to_json(object_with_properties):
    retdict = {}
    for x, v in object_with_properties.properties():
        retdict[x.attr_name] = x.to_json(v)

    return retdict


def set_properties_from_json(object_with_properties, json_obj, context=None, ignore_properties=None):
    ignore_properties = ignore_properties or set()
    try:
        attrs = json_obj['attributes']
    except KeyError:
        attrs = json_obj

    # Apply properties
    ps = dict(object_with_properties.__properties__)
    source_properties = set(attrs.keys())
    for prop_name, prop in ps.items():
        if prop_name in ignore_properties:
            continue

        try:
            val = attrs[prop_name]
            # Make sure we use all properties
            source_properties.remove(prop_name)
        except KeyError:
            # Allow a property to not be set if it has a default value
            # TODO: is this really the job of serialize?
            if prop.default is not None:
                val = prop.default
            elif prop.allow_none:
                val = None
            else:
                raise KeyError("Missing property for object of type " + type(object_with_properties).__name__ + ": " +
                               prop_name)

        if isinstance(val, dict):
            val = prop.from_json(val, context)
            if val is None and attrs[prop_name] is not None:
                raise ValueError("Unparsed to None from: {}".format(attrs[prop_name]))
        else:
            try:
                val = prop.from_json(val, context)
            except TypeError as err:
                # TODO: This seems to be called both from places where the
                # dictionary has been fully deserialized, and on raw json
                # objects. In the interest of time, we're not failing here, but
                # should untangle this eventually
                print("WARNING: failed to parse object {}"
                      " for property {} of type {}. Error was: {}".format(val, prop_name, prop, err))
                raise

        setattr(object_with_properties, prop_name, val)

    remaining_properties = source_properties - ignore_properties
    # Ignore all metadata "properties" saved for DIODE
    remaining_properties = set(prop for prop in remaining_properties if not prop.startswith('_meta'))
    if len(remaining_properties) > 0:
        # TODO: elevate to error once #28 is fixed.
        print("WARNING: unused properties: {}".format(", ".join(sorted(remaining_properties))))
