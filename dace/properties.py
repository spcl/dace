import ast
import astunparse
from collections import OrderedDict
import copy
from dace.frontend.python.astutils import unparse
import itertools
import pydoc
import re
import sympy as sp
import numpy as np
import dace.subsets as sbs
import dace
from dace.symbolic import pystr_to_symbolic
from dace.types import DebugInfo
import json

###############################################################################
# External interface to guarantee correct usage
###############################################################################


def set_property_from_string(prop, obj, string, sdfg=None, from_json=False):
    """ Interface function that guarantees that a property will always be
    correctly set, if possible, by accepting all possible input arguments to
    from_string. """

    # If the property is a string (property name), obtain it from the object
    if isinstance(prop, str):
        prop = type(obj).__properties__[prop]

    if isinstance(prop, CodeProperty):
        if from_json:
            val = prop.from_json(string)
        else:
            val = prop.from_string(string, obj.language)
    elif isinstance(prop, (ReferenceProperty, DataProperty)):
        if sdfg is None:
            raise ValueError(
                "You cannot pass sdfg=None when editing a ReferenceProperty!")
        if from_json:
            val = prop.from_json(string, sdfg)
        else:
            val = prop.from_string(string, sdfg)
    else:
        if from_json:
            val = prop.from_json(string, sdfg)
        else:
            val = prop.from_string(string)
    setattr(obj, prop.attr_name, val)


###############################################################################
# Property base implementation
###############################################################################


class PropertyError(Exception):
    """Exception type for errors related to internal functionality of
    these properties."""
    pass


class Property:
    """ Class implementing properties of DaCe objects that conform to strong
    typing, and allow conversion to and from strings to be edited. """

    def __init__(
            self,
            getter=None,
            setter=None,
            dtype=None,
            default=None,
            from_string=None,
            to_string=None,
            from_json=None,
            to_json=None,
            meta_to_json=None,
            enum=None,  # Values must be present in this enum
            unmapped=False,  # Don't enforce 1:1 mapping with a member variable
            allow_none=False,
            indirected=False,  # This property belongs to a different class
            category='General',
            desc=""):

        self._getter = getter
        self._setter = setter
        self._dtype = dtype
        self._default = default
        if from_string is not None:
            self._from_string = from_string
        elif enum is not None:
            self._from_string = lambda s: enum[s]
        else:
            self._from_string = self.dtype
        if to_string is not None:
            self._to_string = to_string
        elif enum is not None:
            self._to_string = lambda val: val._name_
        else:
            self._to_string = str
        if from_json is None:
            import json

            def tmp(x, sdfg=None):
                if self.dtype == bool:
                    return json.loads(x)
                elif self.dtype is None:
                    # Return without type cast.
                    return self.from_string(json.loads(x))
                if self.dtype == dict or self.dtype == object:
                    # Treat special types (e.g. dict)
                    return json.loads(x, object_hook=Property.json_loader)

                pre = json.loads(x, object_hook=Property.json_loader)
                if pre is None:
                    return None
                return self.from_string(pre)

            self._from_json = tmp
        else:
            self._from_json = from_json
        if to_json is None:
            import json

            # We have to add an indirection
            # (if just returning the output of to_string, one could not always parse it back)
            def tmp(x):
                if x is not None:
                    if self.dtype == bool:
                        return json.dumps(x)
                    elif self.dtype is None:
                        # Return without type cast.
                        return json.dumps(None
                                          if x is None else self.to_string(x))
                    elif self.dtype == dict:
                        # Treat special types (e.g. dict)
                        typecast = dict(x)
                        return json.dumps(
                            None if x is None else typecast,
                            default=Property.json_dumper)
                    elif self.dtype == tuple:
                        typecast = tuple(x)
                        return json.dumps(
                            None if x is None else typecast,
                            default=Property.json_dumper)
                    elif self.dtype == list:
                        typecast = list(x)
                        return json.dumps(
                            None if x is None else typecast,
                            default=Property.json_dumper)
                    elif self.dtype == object:
                        # Not treating this - go away.
                        return json.dumps(x)

                return json.dumps(None if x is None else self.to_string(x))

            self._to_json = tmp
        else:
            self._to_json = to_json
        if meta_to_json is None:
            import json

            def tmp_func(self):
                typestr = self.typestring()

                _default = self.to_json(self.default)

                mdict = {
                    "type": typestr,
                    "desc": self.desc,
                    "category": self.category,
                    "default": json.loads(_default),
                }
                if self.indirected:
                    mdict['indirected'] = True
                return json.dumps(mdict)

            self._meta_to_json = tmp_func
        else:
            self._meta_to_json = meta_to_json

        self._enum = enum
        self._unmapped = unmapped
        self._allow_none = allow_none
        self._indirected = indirected
        self._desc = desc
        self._category = category

    def __get__(self, obj, objtype=None):
        if obj is None:
            # Called on the class rather than an instance, so return the
            # property object itself
            return self
        # If a custom getter is specified, use it
        if self.getter:
            return self.getter(obj)
        if not hasattr(self, "attr_name"):
            raise RuntimeError("Attribute name not set")
        # Otherwise look for attribute prefixed by "_"
        return getattr(obj, "_" + self.attr_name)

    def __set__(self, obj, val):
        # If custom setter is specified, use it
        if self.setter:
            return self.setter(obj, val)
        if not hasattr(self, "attr_name"):
            raise RuntimeError("Attribute name not set")
        # Fail on None unless explicitly allowed
        if val is None and not self.allow_none:
            raise ValueError(
                "None not allowed for property {} in class {}".format(
                    self.attr_name,
                    type(obj).__name__))

        # Accept all DaCe/numpy typeclasses as Python native types
        if isinstance(val, np.number):
            val = val.item()

        # Check if type matches before setting
        if (self.dtype is not None and not isinstance(val, self.dtype)
                and not (val is None and self.allow_none)):
            if isinstance(val, str):
                raise TypeError(
                    "Received str for property {} of type {}. Use "
                    "dace.properties.set_property_from_string or the "
                    "from_string method of the property.".format(
                        self.attr_name, self.dtype))
            raise TypeError(
                "Invalid type \"{}\" for property {}: expected {}".format(
                    type(val).__name__, self.attr_name, self.dtype.__name__))
        # If the value has not yet been set, we cannot pass it to the enum
        # function. Fail silently if this happens
        if self.enum is not None and isinstance(self.enum, (list, tuple, set)):
            if val not in self.enum:
                raise ValueError("Value {} not present in enum: {}".format(
                    val, self.enum))
        setattr(obj, "_" + self.attr_name, val)

    # Python Properties of this Property class

    @property
    def getter(self):
        return self._getter

    @getter.setter
    def getter(self, val):
        self._getter = val

    @property
    def setter(self):
        return self._setter

    @setter.setter
    def setter(self, val):
        self._setter = val

    @property
    def dtype(self):
        return self._dtype

    def typestring(self):
        typestr = ""
        try:
            typestr = self.dtype.__name__
        except:
            # Try again, it might be an enum
            try:
                typestr = self.enum.__name__
            except:
                try:
                    typestr = type(self).__name__
                except:
                    typestr = 'None'
        return typestr

    @property
    def default(self):
        return self._default

    @property
    def allow_none(self):
        return self._allow_none

    @property
    def desc(self):
        return self._desc

    @property
    def from_string(self):
        return self._from_string

    @property
    def to_string(self):
        return self._to_string

    @property
    def from_json(self):
        return self._from_json

    @property
    def to_json(self):
        return self._to_json

    @property
    def meta_to_json(self):
        """ Returns a function to export meta information (type, description, default value).
        """
        return self._meta_to_json

    @property
    def enum(self):
        return self._enum

    @property
    def unmapped(self):
        return self._unmapped

    @property
    def indirected(self):
        return self._indirected

    @indirected.setter
    def indirected(self, val):
        self._indirected = val

    @property
    def category(self):
        return self._category

    @staticmethod
    def add_none_pair(dict_in):
        dict_in[None] = None
        return dict_in

    @staticmethod
    def all_properties_to_json(object_with_properties,
                               options={"no_meta": False}):
        retdict = {}
        for x, v in object_with_properties.properties():
            # The following loads is intended: This is a nested object.
            # If loads() is not used, every element would be a string, and loading the resulting string is hard.
            t = x.to_json(v)
            retdict[x.attr_name] = json.loads(t)

            # Add the meta elements decoupled from key/value to facilitate value usage
            # (The meta is only used when rendering the values)
            if not options['no_meta']:
                retdict['_meta_' + x.attr_name] = json.loads(x.meta_to_json(x))

        # Stringify using the default interface
        return json.dumps(retdict)

    @staticmethod
    def set_properties_from_json(object_with_properties,
                                 json_obj,
                                 context=None):
        try:
            attrs = json_obj['attributes']
        except:
            attrs = json_obj

        # Apply properties
        ps = dict(object_with_properties.__properties__)
        for tmp, _v in ps.items():
            pname = tmp
            try:
                val = attrs[pname]
            except:
                continue

            #TODO: Do we need to dump again? Answer: Yes.
            # Some properties only work from string.
            try:
                stringified = json.dumps(val, default=Property.json_dumper)
                newval = _v.from_json(stringified, context)
                setattr(object_with_properties, tmp, newval)
            except Exception as e:
                import traceback
                traceback.print_exc()
                # #TODO: Maybe log this...
                raise e

    @staticmethod
    def get_property_element(object_with_properties, name):
        # Get the property object (as it may have more options than available by exposing the value)
        ps = dict(object_with_properties.__properties__)
        for tmp, _v in ps.items():
            pname = tmp
            if pname == name:
                return _v
        return None

    @staticmethod
    def json_dumper(obj):
        try:
            # Try the toJSON-methods by default
            tmp = json.loads(obj.toJSON())
            return tmp
        except:
            # If not available, go for the default str() representation
            return str(obj)

    @staticmethod
    def known_types():
        import dace.data
        return {
            "Array": dace.data.Array,
            "Scalar": dace.data.Scalar,
            "Stream": dace.data.Stream,
            "AccessNode": dace.graph.nodes.AccessNode,
            "MapEntry": dace.graph.nodes.MapEntry,
            "MapExit": dace.graph.nodes.MapExit,
            "Reduce": dace.graph.nodes.Reduce,
            "ConsumeEntry": dace.graph.nodes.ConsumeEntry,
            "ConsumeExit": dace.graph.nodes.ConsumeExit,
            "Tasklet": dace.graph.nodes.Tasklet,
            "EmptyTasklet": dace.graph.nodes.EmptyTasklet,
            "NestedSDFG": dace.graph.nodes.NestedSDFG,
            "Memlet": dace.memlet.Memlet,
            "MultiConnectorEdge": dace.graph.graph.MultiConnectorEdge,
            "InterstateEdge": dace.graph.edges.InterstateEdge,
            "Edge": dace.graph.graph.Edge,
            "SDFG": dace.sdfg.SDFG,
            "SDFGState": dace.sdfg.SDFGState,

            # Data types (Note: Types must be qualified, as properties also have type subelements)
            "subsets.Range": dace.subsets.Range,
            "subsets.Indices": dace.subsets.Indices,
        }

    @staticmethod
    def json_loader(obj, context=None):
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
                    t = obj['type']
                except:
                    return tmp
                # If a type is available, the parent element must also be parsed accordingly

        if "type" in obj:
            try:
                t = obj['type']
            except:
                t = attr_type

            if t in Property.known_types():
                return (Property.known_types()[t]).fromJSON_object(
                    obj, context=context)

        return obj


###############################################################################
# Decorator for objects with properties
###############################################################################


def _property_generator(instance):
    for name, prop in type(instance).__properties__.items():
        yield prop, getattr(instance, name)


def make_properties(cls):
    """ A decorator for objects that adds support and checks for strongly-typed 
        properties (which use the Property class).
    """

    # Extract all Property members of the class
    properties = OrderedDict([(name, prop)
                              for name, prop in cls.__dict__.items()
                              if isinstance(prop, Property)])
    # Set the property name to its field name in the class
    for name, prop in properties.items():
        prop.attr_name = name
    # Grab properties from baseclass(es)
    own_properties = copy.copy(properties)
    for base in cls.__bases__:
        if hasattr(base, "__properties__"):
            duplicates = base.__properties__.keys() & own_properties.keys()
            if len(duplicates) != 0:
                raise AttributeError(
                    "Duplicate properties in class {} deriving from {}: {}".
                    format(cls.__name__, base.__name__, duplicates))
            properties.update(base.__properties__)
    # Add the list of properties to the class
    cls.__properties__ = properties
    # Add an iterator to pairs of property names and their values
    cls.properties = _property_generator

    # Grab old init. This will be brought into the closure in the below
    init = cls.__init__

    def initialize_properties(obj, *args, **kwargs):
        # Set default values. If we don't do this, properties that depend on
        # other might fail because the others rely on being set by a default
        # value
        for name, prop in own_properties.items():
            # Only assign our own properties, so we don't overwrite what's been
            # set by the base class
            if hasattr(obj, name):
                raise PropertyError(
                    "Property {} already assigned in {}".format(
                        name,
                        type(obj).__name__))
            if not prop.indirected and prop.default is not None:
                setattr(obj, name, prop.default)
        # Now call vanilla __init__, which can initialize members
        init(obj, *args, **kwargs)
        # Assert that all properties have been set
        for name, prop in properties.items():
            try:
                getattr(obj, name)
            except AttributeError:
                if not prop.unmapped:
                    raise PropertyError(
                        "Property {} is unassigned in __init__ for {}".format(
                            name, cls.__name__))
        # Assert that there are no fields in the object not captured by
        # properties, unless they are prefixed with "_"
        for name, prop in obj.__dict__.items():
            if name not in properties and not name.startswith("_"):
                raise PropertyError(
                    "{} : Variable {} is neither a Property nor "
                    "an internal variable (prefixed with \"_\")".format(
                        str(type(obj)), name))

    # Replace the __init__ method
    cls.__init__ = initialize_properties

    return cls


def indirect_property(cls, f, prop, override):

    # Make a copy of the original property, but override its getter and setter
    prop_name = prop.attr_name
    prop_indirect = copy.copy(prop)
    prop_indirect.indirected = True

    # Because this is a separate function, prop_name is caught in the closure
    def indirect_getter(obj):
        return getattr(f(obj), prop_name)

    def indirect_setter(obj, val):
        return setattr(f(obj), prop_name, val)

    prop_indirect.getter = indirect_getter
    prop_indirect.setter = indirect_setter

    # Add the property to the class
    if not override and hasattr(cls, prop_name):
        raise TypeError(
            "Property \"{}\" already exists in class \"{}\"".format(
                prop_name, cls.__name__))
    setattr(cls, prop_name, prop_indirect)


def indirect_properties(indirect_class, indirect_function, override=False):
    """ A decorator for objects that provides indirect properties defined
        in another class.
    """

    def indirection(cls):
        # For every property in the class we are indirecting to, create an
        # indirection property in this class
        for prop in indirect_class.__properties__.values():
            indirect_property(cls, indirect_function, prop, override)
        return make_properties(cls)

    return indirection


class OrderedDictProperty(Property):
    """ Property type for ordered dicts
    """

    @staticmethod
    def to_json(d):

        # The ordered dict is more of a list than a dict.
        retlist = [{k: v} for k, v in d.items()]
        return json.dumps(retlist, default=Property.json_dumper)

    @staticmethod
    def from_json(s, sdfg=None):

        obj = json.loads(s, object_hook=Property.json_loader)
        # This is expected to be a list (a JSON dict does not guarantee an order,
        # although it would often be lexicographically ordered by key name)

        import collections
        ret = collections.OrderedDict()
        for x in obj:
            ret[list(x.keys())[0]] = list(x.values())[0]

        return ret


class ListProperty(Property):
    """ Property type for lists.
    """

    def __set__(self, obj, val):
        if isinstance(val, str):
            val = list(val)
        elif isinstance(val, tuple):
            val = list(val)
        super(ListProperty, self).__set__(obj, val)

    @staticmethod
    def to_string(l):
        return str(l.dtype(l))

    @staticmethod
    def to_json(l):

        # The json_dumper will try to find the correct serialization in `toJSON`
        # and fallback to str() if that method does not exist
        return json.dumps(l, default=Property.json_dumper)

    @staticmethod
    def from_string(s):
        return list(s)

    @staticmethod
    def from_json(s, sdfg=None):
        # TODO: Typechecks (casts) to a predefined type
        return json.loads(s, object_hook=Property.json_loader)


###############################################################################
# Custom properties
###############################################################################


class SDFGReferenceProperty(Property):
    @staticmethod
    def to_json(obj):
        if obj is None: return 'null'

        return json.dumps(obj.toJSON())  # Make a string of a JSON

    @staticmethod
    def from_json(s, context=None):
        if s == "null": return None

        # Parse the string of the JSON back into an SDFG object
        return dace.SDFG.fromJSON_object(json.loads(json.loads(s)))


# TODO: does not currently work because of how enums work
class OrderProperty(Property):
    """ Custom property class that handles the mapping between the order
        property and the actual class fields (range and parameters). """

    # This is implemented in the context of dace.nodes.Map, but could in
    # principle be reused for other objects, assuming they set the internal
    # fields "_range" and "_params".

    def __get__(self, obj, objtype=None):
        # Copy to avoid changes in the list at callee to be reflected in
        # the map directly
        return list(obj._params)

    def __set__(self, obj, val):
        """ Update both params and ranges based on the new order. """
        # Make this more lenient to the input by comparing strings, and
        # using the new order to shuffle the original lists
        param_strings = list(map(str, obj._params))
        update_strings = list(map(str, val))
        if len(update_strings) != len(param_strings):
            raise ValueError(
                "Wrong length of new order: {} (found {}, expected {})".format(
                    str(val), len(update_strings), len(param_strings)))
        # The below will throw a ValueError if a parameter doesn't exist
        # We assume that no parameter will be present twice...
        indices = [param_strings.index(x) for x in update_strings]
        obj._params = [obj._params[i] for i in indices]
        obj._range.reorder(indices)

    @staticmethod
    def to_string(val):
        return "({})".format(", ".join(map(str, val)))

    @staticmethod
    def from_string(s):
        """Create a list of symbols from a list of strings."""
        return [sp.Symbol(i) for i in re.sub("[\(\)\[\]]", "", s).split(",")]

    @staticmethod
    def enum(obj):
        """Implement enum to populate e.g. dropdown."""
        return list(itertools.permutations(obj))


class RangeProperty(Property):
    """ Custom Property type for `dace.graph.subset.Range` members. """

    def __set__(self, obj, value):
        if isinstance(value, list):
            value = dace.subsets.Range(value)
        super(RangeProperty, self).__set__(obj, value)

    @property
    def dtype(self):
        return sbs.Range

    @staticmethod
    def to_string(obj):
        return sbs.Range.ndslice_to_string(obj)

    @staticmethod
    def from_string(s):
        return sbs.Range.from_string(s)

    @staticmethod
    def to_json(obj):
        if obj is None:
            return "null"
        # to_string is not enough - it does not preserve all information

        return obj.toJSON()

    @staticmethod
    def from_json(s, sdfg=None):
        from dace.subsets import Range

        if s == "null": return None

        return Range.fromJSON(s)


class DebugInfoProperty(Property):
    """ Custom Property type for DebugInfo members. """

    @property
    def dtype(self):
        return DebugInfo

    @property
    def allow_none(self):
        return True

    @staticmethod
    def to_string(di):
        if isinstance(di, DebugInfo):
            r = "file:" + str(di.filename) + " "
            r += "from line: " + str(di.start_line) + " col: " + str(
                di.start_column) + " "
            r += "to line: " + str(di.end_line) + " col: " + str(di.end_column)
            return r
        else:
            return "None"

    @staticmethod
    def from_string(s):

        if s is None:
            return None

        f = None
        sl = 0
        el = 0
        sc = 0
        ec = 0
        info_available = False
        di = None

        m = re.search("file: (\w+)", s)
        if m is not None:
            info_available = True
            f = sl = m.group(1)
        m = re.search("from line: (\d+)", s)
        if m is not None:
            sl = m.group(1)
            el = sl
            info_available = True
        m = re.search("to line: (\d+)", s)
        if m is not None:
            el = m.group(1)
            info_available = True
        m = re.search("from col: (\d+)", s)
        if m is not None:
            sc = m.group(1)
            ec = sc
            info_available = True
        m = re.search("to col: (\d+)", s)
        if m is not None:
            ec = m.group(1)
            info_available = True
        if info_available:
            di = DebugInfo(f, sl, sc, el, ec)
        return di

    @staticmethod
    def to_json(s):
        if not isinstance(s, DebugInfo):
            return json.dumps(None)
        nval = {
            "filename": s.filename,
            "start_line": s.start_line,
            "end_line": s.end_line,
            "start_col": s.start_column,
            "end_col": s.end_column
        }
        return json.dumps(nval)

    @staticmethod
    def from_json(s, sdfg=None):
        s = json.loads(s)
        if s is None: return None

        return DebugInfo(s['start_line'], s['start_col'], s['end_line'],
                         s['end_col'], s['filename'])


class ParamsProperty(Property):
    """ Property for list of parameters, such as parameters for a Map. """

    @property
    def dtype(self):
        return list

    @staticmethod
    def to_string(l):
        return "[{}]".format(", ".join(map(str, l)))

    @staticmethod
    def from_string(s):
        return [
            sp.Symbol(m.group(0))
            for m in re.finditer("[a-zA-Z_][a-zA-Z0-9_]*", s)
        ]

    @staticmethod
    def to_json(l):
        return json.dumps(l, default=Property.json_dumper)

    @staticmethod
    def from_json(l, sdfg=None):
        return json.loads(
            l, object_hook=lambda x: Property.json_loader(l, sdfg))


class SetProperty(Property):
    """Property for a set of elements of one type, e.g., connectors. """

    def __init__(
            self,
            element_type,
            getter=None,
            setter=None,
            default=None,
            from_string=None,
            to_string=None,
            from_json=None,
            to_json=None,
            unmapped=False,  # Don't enforce 1:1 mapping with a member variable
            allow_none=False,
            desc="",
            **kwargs):
        if to_json is None:
            to_json = self.to_json
        super(SetProperty, self).__init__(
            getter=getter,
            setter=setter,
            dtype=set,
            default=default,
            from_string=from_string,
            to_string=to_string,
            from_json=from_json,
            to_json=to_json,
            enum=None,
            unmapped=unmapped,
            allow_none=allow_none,
            desc=desc,
            **kwargs)
        self._element_type = element_type

    @property
    def dtype(self):
        return set

    @staticmethod
    def to_string(l):
        return str(l)

    @staticmethod
    def from_string(s):
        return [eval(i) for i in re.sub("[\{\}\(\)\[\]]", "", s).split(",")]

    @staticmethod
    def to_json(l):
        import json
        return json.dumps(list(sorted(l)))

    @staticmethod
    def from_json(l, sdfg=None):
        import json
        return set(json.loads(l))

    def __get__(self, obj, objtype=None):
        # Copy to avoid changes in the set at callee to be reflected in
        # the node directly
        return set(super(SetProperty, self).__get__(obj, objtype))

    def __set__(self, obj, val):
        # Check for uniqueness
        if len(val) != len(set(val)):
            dups = set([x for x in val if val.count(x) > 1])
            raise ValueError('Duplicates found in set: ' + str(dups))
        # Cast to element type
        try:
            new_set = set(self._element_type(elem) for elem in val)
        except (TypeError, ValueError):
            raise ValueError('Some elements could not be converted to %s' %
                             (str(self._element_type)))

        super(SetProperty, self).__set__(obj, new_set)


class LambdaProperty(Property):
    """ Custom Property type that accepts a lambda function, with conversions
        to and from strings. """

    @property
    def dtype(self):
        return str

    @staticmethod
    def from_string(s):
        return ast.parse(s).body[0].value

    @staticmethod
    def to_string(obj):
        if obj is None:
            return 'lambda: None'
        if isinstance(obj, str):
            return obj
        return unparse(obj)

    @staticmethod
    def to_json(obj):
        if obj is None: return 'null'
        return json.dumps(LambdaProperty.to_string(obj))

    @staticmethod
    def from_json(s, sdfg=None):
        if s == 'null': return None
        return LambdaProperty.from_string(json.loads(s))

    def __set__(self, obj, val):
        if val is not None:
            if isinstance(val, str):
                self.from_string(val)  # Check that from_string doesn't fail
            elif isinstance(val, ast.Lambda):
                val = self.to_string(val)  # Store as string internally
            else:
                raise TypeError(
                    "Lambda property must be either string or ast.Lambda")
        super(LambdaProperty, self).__set__(obj, val)


class SubgraphProperty(Property):
    """ Property class that provides read-only (loading from json value is disabled)
        access to a dict value. Intended for Transformation.subgraph.
    """

    def __set__(self, obj, val):
        if val is not None:
            super(SubgraphProperty, self).__set__(obj, val)

    @staticmethod
    def to_json(obj):
        return json.dumps(str(obj))

    @staticmethod
    def from_json(s, sdfg=None):
        return None


class CodeBlock(list):
    """ Helper class that represents AST code blocks for `CodeProperty`, 
        implemented as a list with an extra _as_string property. The object
        also stores the original string, allowing us to preserve comments and
        formatting from user input.
    """

    def __init__(self, *args, **kwargs):
        self._as_string = ""
        super().__init__(*args, **kwargs)

    @property
    def as_string(self):
        return self._as_string

    @as_string.setter
    def as_string(self, string):
        self._as_string = string


class CodeProperty(Property):
    """ Custom Property type that accepts code in various languages. """

    @property
    def dtype(self):
        return None

    @staticmethod
    def get_language(object_with_properties, prop_name):
        tmp = getattr(object_with_properties, "_" + prop_name)
        try:
            # To stay compatible, return the code only. The language has to be obtained differently
            tmp = tmp['language']
        except:
            pass
        return tmp

    @staticmethod
    def to_json(obj):
        lang = dace.types.Language.Python
        if obj is None:
            return json.dumps(obj)
        if isinstance(obj, str):
            return json.dumps(obj)

        if isinstance(obj, dict):
            lang = obj['language']
        else:
            lang = "Python"  # If not specified, we just don't want the validators go haywire
        ret = {'string_data': CodeProperty.to_string(obj), 'language': lang}
        return json.dumps(ret)

    @staticmethod
    def from_json(l, sdfg=None):
        tmp = json.loads(l)

        if tmp is None:
            return None

        try:
            lang = tmp['language']
        except:
            lang = None

        if lang == "NoCode":
            return None

        if lang is None:
            lang = dace.types.Language.Python
        elif lang.endswith("Python"):
            lang = dace.types.Language.Python
        elif lang.endswith("CPP"):
            lang = dace.types.Language.CPP

        try:
            cdata = tmp['string_data']
        except:
            print("UNRECOGNIZED CODE JSON: " + str(tmp))
            cdata = ""

        return CodeProperty.from_string(cdata, lang)

    @staticmethod
    def from_string(string, language=None):
        if language is None:
            raise TypeError("Must pass language as second argument to "
                            "from_string method of CodeProperty")
        if language == dace.types.Language.Python:
            block = CodeBlock(ast.parse(string).body)
            block.as_string = string
            return {'code_or_block': block, 'language': language}
        else:
            return {'code_or_block': string, 'language': language}

    @staticmethod
    def to_string(obj):
        if isinstance(obj, dict):
            # The object has annotated language in this case; ignore the language for this operation
            obj = obj['code_or_block']
        if isinstance(obj, str):
            return obj
        # Grab the originally parsed string if any
        if obj._as_string is not None and obj._as_string != "":
            return obj._as_string
        # It's probably good enough to assume that there is an original string
        # if the language was not Python, so we just throw the string to the
        # astunparser.
        return unparse(obj)

    def __get__(self, obj, val):
        tmp = super(CodeProperty, self).__get__(obj, val)
        try:
            # To stay compatible, return the code only. The language has to be obtained differently
            tmp = tmp['code_or_block']
        except (KeyError, TypeError):
            pass
        return tmp

    def __set__(self, obj, val):

        if val is None:
            # Keep as None. The "allow_none" check in the superclass
            # ensures that this is legal
            super(CodeProperty, self).__set__(obj, None)
            return
        elif isinstance(val, str):
            try:
                language = getattr(obj, "_" + self.attr_name)['language']
            except:
                language = dace.types.Language.Python
            if language is not None:
                # Store original string
                val = self.from_string(val, language)['code_or_block']
        else:
            try:
                language = val['language']
                val = val['code_or_block']
            except:
                # Default to Python
                language = dace.types.Language.Python
            try:
                if language is not dace.types.Language.Python and not isinstance(
                        val, str):
                    raise TypeError(
                        "Only strings accepted for other "
                        "languages than Python, got {t} ({s}).".format(
                            t=type(val).__name__, s=str(val)))
            except AttributeError:
                # Don't check language if it has not been set yet. We will
                # assume it's Python AST, since it wasn't a string
                pass
            if isinstance(val, str):
                val = self.from_string(val, language)['code_or_block']
            elif isinstance(val, (ast.FunctionDef, ast.With)):
                # TODO: the original parsing should have already stripped this
                val = CodeBlock(val.body)
            elif isinstance(val, ast.AST):
                val = CodeBlock([val])
            else:
                try:
                    iter(val)
                except TypeError:
                    raise TypeError(
                        "CodeProperty expected an iterable of expressions, "
                        " got {}".format(type(val).__name__))
                for e in val:
                    if not isinstance(e, ast.AST):
                        raise TypeError(
                            "Found type {} in list of AST expressions: "
                            "expected ast.AST".format(type(e).__name__))
        super(CodeProperty, self).__set__(obj, {
            'code_or_block': val,
            'language': language
        })


class SubsetProperty(Property):
    """ Custom Property type that accepts any form of subset, and enables
    parsing strings into multiple types of subsets. """

    @property
    def dtype(self):
        return None

    @property
    def allow_none(self):
        return True

    def __set__(self, obj, val):
        if (val is not None and not isinstance(val, sbs.Range)
                and not isinstance(val, sbs.Indices)):
            try:
                val = self.from_string(val)
            except SyntaxError:
                raise TypeError(
                    "Subset property must be either Range or Indices: got {}".
                    format(type(val).__name__))
        super(SubsetProperty, self).__set__(obj, val)

    @staticmethod
    def from_string(s):
        if s is None or s == 'None' or len(s) == 0:
            return None
        ranges = sbs.Range.from_string(s)
        if ranges:
            return ranges
        else:
            return sbs.Indices.from_string(s)

    @staticmethod
    def to_string(val):
        if isinstance(val, sbs.Range):
            return sbs.Range.ndslice_to_string(val)
        elif isinstance(val, sbs.Indices):
            return sbs.Indices.__str__(val)
        elif val is None:
            return 'None'
        raise TypeError

    @staticmethod
    def to_json(val):
        if val is None:
            return 'null'
        try:
            return val.toJSON()
        except:
            return json.dumps(SubsetProperty.to_string(val))

    @staticmethod
    def from_json(val, sdfg=None):
        if val == 'null':
            return None
        obj = json.loads(val, object_hook=Property.json_loader)
        return obj


class SymbolicProperty(Property):
    """ Custom Property type that accepts integers or Sympy expressions. """

    @property
    def dtype(self):
        return None

    def __set__(self, obj, val):
        if (not isinstance(val, sp.expr.Expr) and not isinstance(val, int)
                and not isinstance(val, str)):
            raise TypeError(
                "Property {} must an int or symbolic expression".format(
                    self.attr_name))
        super(SymbolicProperty, self).__set__(obj, val)

    @staticmethod
    def from_string(s):
        return pystr_to_symbolic(s)


class DataProperty(Property):
    """ Custom Property type that represents a link to a data descriptor.
        Needs the SDFG to be passed as an argument to `from_string` and
        `enum`. """

    def __init__(self, desc='', default=None, **kwargs):
        # Data can be None when no data is flowing, e.g., on a memlet with a
        # map that has no external inputs
        return super().__init__(
            dtype=str, allow_none=True, desc=desc, default=default, **kwargs)

    def typestring(self):
        return "DataProperty"

    @staticmethod
    def enum(sdfg=None):
        if sdfg is None:
            raise TypeError("Must pass SDFG as second argument to "
                            "enum method of ArrayProperty")
        return list(sdfg.arrays.keys())

    @staticmethod
    def from_string(s, sdfg=None):
        if sdfg is None:
            raise TypeError("Must pass SDFG as second argument to "
                            "from_string method of ArrayProperty")
        if s not in sdfg.arrays:
            raise ValueError("No data found in SDFG with name: {}".format(s))
        return s

    @staticmethod
    def to_string(obj):
        return str(obj)

    @staticmethod
    def to_json(obj):
        if obj is None:
            return "null"
        return json.dumps(str(obj))

    @staticmethod
    def from_json(s, context=None):
        sdfg = context['sdfg']
        s = json.loads(s)
        if sdfg is None:
            raise TypeError("Must pass SDFG as second argument")
        if s not in sdfg.arrays:
            if s is None:
                # This is fine
                #return "null" # Every SDFG has a 'null' element
                return None
            raise ValueError("No data found in SDFG with name: {}".format(s))
        return s


class ReferenceProperty(Property):
    """ Custom Property type that represents a link to another SDFG object.
        Needs the SDFG to be passed as an argument to `from_string`."""

    @staticmethod
    def from_string(s, sdfg=None):
        if sdfg is None:
            raise TypeError("Must pass SDFG as second argument to "
                            "from_string method of ReferenceProperty")
        for node in sdfg.states():
            if node.label == s:
                return node
        for node, _ in sdfg.all_nodes_recursive():
            if node.label == s:
                return node
        raise ValueError("No node found in SDFG with name: {}".format(s))

    @staticmethod
    def to_string(obj):
        return obj.label


class ShapeProperty(Property):
    """ Custom Property type that defines a shape. """

    @property
    def dtype(self):
        return tuple

    @staticmethod
    def from_string(s):
        if s[0] == "(" and s[-1] == ")":
            s = s[1:-1]
        return tuple([
            dace.symbolic.pystr_to_symbolic(m.group(0))
            for m in re.finditer("[^,;:]+", s)
        ])

    @staticmethod
    def to_string(obj):
        return ", ".join(map(str, obj))

    @staticmethod
    def to_json(obj):
        if obj is None:
            return json.dumps(obj)
        return json.dumps([*map(str, obj)])

    @staticmethod
    def from_json(s, sdfg=None):
        d = json.loads(s)
        if d is None:
            return None
        return tuple([dace.symbolic.pystr_to_symbolic(m) for m in d])

    def __set__(self, obj, val):
        if isinstance(val, list):
            val = tuple(val)
        super(ShapeProperty, self).__set__(obj, val)


class TypeProperty(Property):
    """ Custom Property type that finds a type according to the input string. 
    """

    @property
    def dtype(self):
        return type

    # TODO: this does not work both ways! If converted to a string we lose the
    # location information.
    @staticmethod
    def from_string(s):
        dtype = pydoc.locate(s)
        if dtype is None:
            raise ValueError("No type \"{}\" found.".format(s))
        if not isinstance(dtype, type):
            raise ValueError("Object \"{}\" is not a type.".format(dtype))
        return dtype


class TypeClassProperty(Property):
    """ Custom property type for memory as defined in dace.types,
        e.g. `dace.float32`. """

    @property
    def dtype(self):
        return dace.types.typeclass

    @staticmethod
    def from_string(s):
        dtype = pydoc.locate("dace.types.{}".format(s))
        if dtype is None or not isinstance(dtype, dace.types.typeclass):
            raise ValueError("Not a valid data type: {}".format(s))
        return dtype

    @staticmethod
    def to_string(obj):
        return obj.to_string()
