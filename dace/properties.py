# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import ast
from collections import OrderedDict
import copy
import warnings
from dace.frontend.python.astutils import unparse, TaskletFreeSymbolVisitor
import json
import pydoc
import re
import sympy as sp
import numpy as np
import dace.subsets as sbs
import dace
import dace.serialize
from dace.symbolic import pystr_to_symbolic
from dace.dtypes import DebugInfo, typeclass
from numbers import Integral, Number
from typing import List, Set, Type, Union, TypeVar, Generic

T = TypeVar('T')

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
            raise ValueError("You cannot pass sdfg=None when editing a ReferenceProperty!")
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


class Property(Generic[T]):
    """ Class implementing properties of DaCe objects that conform to strong
    typing, and allow conversion to and from strings to be edited. """

    def __init__(
            self,
            getter=None,
            setter=None,
            dtype: Type[T] = None,
            default=None,
            from_string=None,
            to_string=None,
            from_json=None,
            to_json=None,
            meta_to_json=None,
            choices=None,  # Values must be present in this enum
            unmapped=False,  # Don't enforce 1:1 mapping with a member variable
            allow_none=False,
            indirected=False,  # This property belongs to a different class
            category='General',
            desc="",
            optional=False,
            optional_condition=lambda _: True):

        self._getter = getter
        self._setter = setter
        self._dtype = dtype
        self._default = default
        self._optional = optional
        self._optional_condition = optional_condition

        if allow_none is False and default is None:
            try:
                self._default = dtype()
            except TypeError:
                if hasattr(self, 'dtype'):
                    try:
                        self._default = self.dtype()
                    except TypeError:
                        raise TypeError('Default not properly defined for property')
                else:
                    raise TypeError('Default not properly defined for property')

        if choices is not None:
            for choice in choices:
                if dtype is None:
                    dtype = type(choice)
                if not isinstance(choice, dtype):
                    raise TypeError("All choices must be an instance of dtype")

        if from_string is not None:
            self._from_string = from_string
        elif choices is not None:
            self._from_string = lambda s: choices[s]
        else:
            self._from_string = self.dtype

        if to_string is not None:
            self._to_string = to_string
        elif choices is not None:
            self._to_string = lambda val: val.__name__
        else:
            self._to_string = str

        if from_json is None:
            if self._from_string is not None:

                def fs(obj, *args, **kwargs):
                    if isinstance(obj, str):
                        # The serializer does not know about this property, so if
                        # we can convert using our to_string method, do that here
                        return self._from_string(obj)
                    # Otherwise ship off to the serializer, telling it which type
                    # it's dealing with as a sanity check
                    return dace.serialize.from_json(obj, *args, known_type=dtype, **kwargs)

                self._from_json = fs
            else:
                self._from_json = lambda *args, **kwargs: dace.serialize.from_json(*args, known_type=dtype, **kwargs)
        else:
            self._from_json = from_json
            if self.from_json != from_json:
                self.from_json = from_json

        if to_json is None:
            self._to_json = dace.serialize.to_json
        else:
            self._to_json = to_json
            if self.to_json != to_json:
                self.to_json = to_json

        if meta_to_json is None:

            def tmp_func(self):
                typestr = self.typestring()

                _default = self.to_json(self.default)

                mdict = {
                    "metatype": typestr,
                    "desc": self.desc,
                    "category": self.category,
                    "default": _default,
                }
                if self.indirected:
                    mdict['indirected'] = True
                return mdict

            self._meta_to_json = tmp_func
        else:
            self._meta_to_json = meta_to_json

        self._choices = choices
        self._unmapped = unmapped
        self._allow_none = allow_none
        self._indirected = indirected
        self._desc = desc
        self._category = category
        if desc is not None and len(desc) > 0:
            self.__doc__ = desc
        elif self.dtype is not None:
            self.__doc__ = "Object property of type %s" % self.dtype.__name__
        else:
            self.__doc__ = "Object property of type %s" % type(self).__name__

    def __get__(self, obj, objtype=None) -> T:
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
            raise ValueError("None not allowed for property {} in class {}".format(self.attr_name, type(obj).__name__))

        # Accept all DaCe/numpy typeclasses as Python native types
        if isinstance(val, np.number):
            val = val.item()

        # Edge cases for integer and float types
        if isinstance(val, int) and self.dtype == float:
            val = float(val)
        if isinstance(val, float) and self.dtype == int and val == int(val):
            val = int(val)

        # Check if type matches before setting
        if (self.dtype is not None and not isinstance(val, self.dtype) and not (val is None and self.allow_none)):
            if isinstance(val, str):
                raise TypeError("Received str for property {} of type {}. Use "
                                "dace.properties.set_property_from_string or the "
                                "from_string method of the property.".format(self.attr_name, self.dtype))
            raise TypeError("Invalid type \"{}\" for property {}: expected {}".format(
                type(val).__name__, self.attr_name, self.dtype.__name__))
        # If the value has not yet been set, we cannot pass it to the enum
        # function. Fail silently if this happens
        if self.choices is not None \
                and isinstance(self.choices,(list, tuple, set)) \
                and (val is not None or not self.allow_none):
            if val not in self.choices:
                raise ValueError("Value {} not present in choices: {}".format(val, self.choices))
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

    @property
    def optional(self):
        return self._optional

    @property
    def optional_condition(self):
        return self._optional_condition

    def typestring(self):
        typestr = ""
        try:
            typestr = self.dtype.__name__
        except:
            # Try again, it might be an enum
            try:
                typestr = self.choices.__name__
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
    def choices(self):
        return self._choices

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
    def get_property_element(object_with_properties, name):
        # Get the property object (as it may have more options than available by exposing the value)
        ps = dict(object_with_properties.__properties__)
        for tmp, _v in ps.items():
            pname = tmp
            if pname == name:
                return _v
        return None


###############################################################################
# Decorator for objects with properties
###############################################################################


def _property_generator(instance):
    for name, prop in type(instance).__properties__.items():
        if hasattr(instance, "_" + name):
            yield prop, getattr(instance, "_" + name)
        else:
            yield prop, getattr(instance, name)


def make_properties(cls):
    """ A decorator for objects that adds support and checks for strongly-typed
        properties (which use the Property class).
    """

    # Extract all Property members of the class
    properties = OrderedDict([(name, prop) for name, prop in cls.__dict__.items() if isinstance(prop, Property)])
    # Set the property name to its field name in the class
    for name, prop in properties.items():
        prop.attr_name = name
        prop.owner = cls
    # Grab properties from baseclass(es)
    own_properties = copy.copy(properties)
    for base in cls.__bases__:
        if hasattr(base, "__properties__"):
            duplicates = base.__properties__.keys() & own_properties.keys()
            if len(duplicates) != 0:
                raise AttributeError("Duplicate properties in class {} deriving from {}: {}".format(
                    cls.__name__, base.__name__, duplicates))
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
                raise PropertyError("Property {} already assigned in {}".format(name, type(obj).__name__))
            if not prop.indirected:
                if prop.allow_none or prop.default is not None:
                    setattr(obj, name, prop.default)
        # Now call vanilla __init__, which can initialize members
        init(obj, *args, **kwargs)
        # Assert that all properties have been set
        for name, prop in properties.items():
            try:
                getattr(obj, name)
            except AttributeError:
                if not prop.unmapped:
                    raise PropertyError("Property {} is unassigned in __init__ for {}".format(name, cls.__name__))
        # Assert that there are no fields in the object not captured by properties, unless they are prefixed with "_"
        for name, prop in obj.__dict__.items():
            if (name not in properties and not name.startswith("_") and name not in dir(type(obj))):
                raise PropertyError("{} : Variable {} is neither a Property nor "
                                    "an internal variable (prefixed with \"_\")".format(str(type(obj)), name))

    # Replace the __init__ method
    cls.__init__ = initialize_properties

    # Register our type with the serialization framework
    cls = dace.serialize.serializable(cls)

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
        raise TypeError("Property \"{}\" already exists in class \"{}\"".format(prop_name, cls.__name__))
    setattr(cls, prop_name, prop_indirect)


def indirect_properties(indirect_class, indirect_function, override=False):
    """ A decorator for objects that provides indirect properties defined
        in another class.
    """

    def indirection(cls):
        # For every property in the class we are indirecting to, create an
        # indirection property in this class
        inherited_props = {}
        for base_cls in cls.__bases__:
            if hasattr(base_cls, "__properties__"):
                inherited_props.update(base_cls.__properties__)
        for name, prop in indirect_class.__properties__.items():
            if (name in inherited_props and type(inherited_props[name]) == type(prop)):
                # Base class could already have indirected properties
                continue
            indirect_property(cls, indirect_function, prop, override)
        return make_properties(cls)

    return indirection


class OrderedDictProperty(Property):
    """ Property type for ordered dicts
    """

    def to_json(self, d):

        # The ordered dict is more of a list than a dict.
        retlist = [{k: v} for k, v in d.items()]
        return retlist

    @staticmethod
    def from_json(obj, sdfg=None):

        # This is expected to be a list (a JSON dict does not guarantee an order,
        # although it would often be lexicographically ordered by key name)

        ret = OrderedDict()
        for x in obj:
            ret[list(x.keys())[0]] = list(x.values())[0]

        return ret


class ListProperty(Property[List[T]]):
    """ Property type for lists.
    """

    def __init__(self, element_type: T, *args, **kwargs):
        """
        Create a List property with a uniform element type.

        :param element_type: The type of each element in the list, or a function
                             that converts an element to the wanted type (e.g.,
                             `dace.symbolic.pystr_to_symbolic` for symbolic
                             expressions)
        :param args: Other arguments (inherited from Property).
        :param kwargs: Other keyword arguments (inherited from Property).
        """

        kwargs['dtype'] = list
        super().__init__(*args, **kwargs)
        self.element_type = element_type

    def __set__(self, obj, val):
        if isinstance(val, str):
            val = list(map(self.element_type, list(val)))
        elif isinstance(val, tuple):
            val = list(map(self.element_type, val))
        super(ListProperty, self).__set__(obj, val)

    @staticmethod
    def to_string(l):
        return str(l)

    def to_json(self, l):
        if l is None:
            return None
        # If element knows how to convert itself, let it
        if hasattr(self.element_type, "to_json"):
            return [elem.to_json() for elem in l]
        # If elements are one of the JSON basic types, use directly
        if self.element_type in (int, float, list, tuple, dict):
            return list(map(self.element_type, l))
        # Otherwise, convert to strings
        return list(map(str, l))

    def from_string(self, s):
        if s.startswith('[') and s.endswith(']'):
            return [self.element_type(d.strip()) for d in s[1:-1].split(',')]
        else:
            return list(s)

    def from_json(self, data, sdfg=None):
        if data is None:
            return data
        if not isinstance(data, list):
            raise TypeError('ListProperty expects a list input, got %s' % data)
        # If element knows how to convert itself, let it
        if hasattr(self.element_type, "from_json"):
            return [self.element_type.from_json(elem) for elem in data]
        # Type-checks (casts) to the element type
        return list(map(self.element_type, data))


class TransformationHistProperty(Property):
    """ Property type for transformation histories.
    """

    def __init__(self, *args, **kwargs):
        """
        Create a List property with element type TransformationBase.

        :param args: Other arguments (inherited from Property).
        :param kwargs: Other keyword arguments (inherited from Property).
        """

        kwargs['dtype'] = list
        super().__init__(*args, **kwargs)

    def __set__(self, obj, val):
        super(TransformationHistProperty, self).__set__(obj, val)

    def to_json(self, hist):
        if hist is None:
            return None
        return [elem.to_json() if elem is not None else None for elem in hist]

    def from_json(self, data, sdfg=None):
        if data is None:
            return data
        if not isinstance(data, list):
            raise TypeError('TransformationHistProperty expects a list input, got %s' % data)
        return [dace.serialize.from_json(elem) for elem in data]


class DictProperty(Property):
    """ Property type for dictionaries. """

    def __init__(self, key_type, value_type, *args, **kwargs):
        """
        Create a dictionary property with uniform key/value types.

        The type of each element in the dictionary can be given as a type class,
        or as a function that converts an element to the wanted type (e.g.,
        `dace.symbolic.pystr_to_symbolic` for symbolic expressions).
        
        :param key_type: The type of the keys in the dictionary.
        :param value_type: The type of the values in the dictionary.
        :param args: Other arguments (inherited from Property).
        :param kwargs: Other keyword arguments (inherited from Property).
        """
        kwargs['dtype'] = dict
        if not isinstance(key_type, type) and not callable(key_type):
            raise TypeError("Expected type or callable, got: {}".format(key_type))
        if not isinstance(value_type, type) and not callable(value_type):
            raise TypeError("Expected type or callable, got: {}".format(value_type))
        super().__init__(*args, **kwargs)
        self.key_type = key_type
        self.value_type = value_type

        # Check whether a key/value is an instance of its type, if the given
        # type is a Python type, or call self.*_type if it's a callable.
        if isinstance(key_type, type):
            self.is_key = lambda k: isinstance(k, self.key_type)
        else:
            self.is_key = lambda k: False
        if isinstance(value_type, type):
            self.is_value = lambda v: isinstance(v, self.value_type)
        else:
            self.is_value = lambda v: False

    def __set__(self, obj, val):
        if isinstance(val, str):
            val = ast.literal_eval(val)
        elif isinstance(val, (tuple, list)):
            val = {k[0]: k[1] for k in val}
        elif isinstance(val, dict):
            val = {(k if self.is_key(k) else self.key_type(k)): (v if self.is_value(v) else self.value_type(v))
                   for k, v in val.items()}
        super(DictProperty, self).__set__(obj, val)

    @staticmethod
    def to_string(d):
        return str(d)

    def to_json(self, d):
        if d is None:
            return None
        saved_dictionary = d

        # If key knows how to convert itself, let it
        if hasattr(self.key_type, "to_json"):
            saved_dictionary = {k.to_json(): v for k, v in saved_dictionary.items()}
        # Otherwise, if the keys are not a native JSON type, convert to strings
        elif self.key_type not in (int, float, list, tuple, dict, str):
            saved_dictionary = {str(k): v for k, v in saved_dictionary.items()}

        # Same as above, but for values
        if hasattr(self.value_type, "to_json"):
            saved_dictionary = {k: v.to_json() for k, v in saved_dictionary.items()}
        elif self.value_type not in (int, float, list, tuple, dict, str):
            saved_dictionary = {k: str(v) for k, v in saved_dictionary.items()}
        else:
            saved_dictionary = {k: self.value_type(v) for k, v in saved_dictionary.items()}

        # Sort by key before saving
        return {k: v for k, v in sorted(saved_dictionary.items())} if None not in saved_dictionary else saved_dictionary

    @staticmethod
    def from_string(s):
        return dict(s)

    def from_json(self, data, sdfg=None):
        if data is None:
            return data
        if not isinstance(data, dict):
            raise TypeError('DictProperty expects a dictionary input, got '
                            '%s' % data)
        # If element knows how to convert itself, let it
        key_json = hasattr(self.key_type, "from_json")
        value_json = hasattr(self.value_type, "from_json")

        return {
            self.key_type.from_json(k, sdfg) if key_json else self.key_type(k):
            self.value_type.from_json(v, sdfg) if value_json else self.value_type(v)
            for k, v in data.items()
        }


###############################################################################
# Custom properties
###############################################################################


class EnumProperty(Property):

    def __init__(self, dtype, *args, **kwargs):
        kwargs['dtype'] = dtype
        super().__init__(*args, **kwargs)

        def f(s, *args, **kwargs):
            if s is None:
                return None
            if isinstance(s, dtype):
                return s
            try:
                self._undefined_val = None
                return dtype[s]
            except KeyError:
                self._undefined_val = s
                return dtype['Undefined']

        self._choices = dtype
        self._from_json = f
        self._from_string = f

        self._undefined_val = None

        def g(obj):
            if self._undefined_val is None:
                return dace.serialize.to_json(obj)
            else:
                return self._undefined_val

        self._to_json = g
        self._to_string = g


class SDFGReferenceProperty(Property):

    def to_json(self, obj):
        if obj is None:
            return None
        return obj.to_json()  # Store nested JSON

    def from_json(self, obj, context=None):
        if obj is None:
            return None

        # Backwards compatibility
        if isinstance(obj, str):
            obj = json.loads(obj)

        # Parse the JSON back into an SDFG object
        return dace.SDFG.from_json(obj, context)


class OptionalSDFGReferenceProperty(SDFGReferenceProperty):
    """
    An SDFG reference property that defaults to None if cannot be deserialized.
    """

    def from_json(self, obj, context=None):
        try:
            return super().from_json(obj, context)
        except TypeError as ex:
            warnings.warn(f'Could not deserialize optional SDFG ({type(ex).__name__}), defaulting to None: {str(ex)}')
            return None


class RangeProperty(Property):
    """ Custom Property type for `dace.subsets.Range` members. """

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


class DebugInfoProperty(Property):
    """ Custom Property type for DebugInfo members. """

    def __init__(self, **kwargs):
        if 'default' not in kwargs:
            kwargs['default'] = DebugInfo(0, 0, 0, 0)
        super().__init__(dtype=DebugInfo, **kwargs)

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
            r += "from line: " + str(di.start_line) + " col: " + str(di.start_column) + " "
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

        m = re.search(r"file: (\w+)", s)
        if m is not None:
            info_available = True
            f = sl = m.group(1)
        m = re.search(r"from line: (\d+)", s)
        if m is not None:
            sl = m.group(1)
            el = sl
            info_available = True
        m = re.search(r"to line: (\d+)", s)
        if m is not None:
            el = m.group(1)
            info_available = True
        m = re.search(r"from col: (\d+)", s)
        if m is not None:
            sc = m.group(1)
            ec = sc
            info_available = True
        m = re.search(r"to col: (\d+)", s)
        if m is not None:
            ec = m.group(1)
            info_available = True
        if info_available:
            di = DebugInfo(f, sl, sc, el, ec)
        return di


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
        super(SetProperty, self).__init__(getter=getter,
                                          setter=setter,
                                          dtype=set,
                                          default=default,
                                          from_string=from_string,
                                          to_string=to_string,
                                          from_json=from_json,
                                          to_json=to_json,
                                          choices=None,
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
        return [eval(i) for i in re.sub(r"[\{\}\(\)\[\]]", "", s).split(",")]

    def to_json(self, l):
        if l is None:
            return None
        return list(sorted(l))

    def from_json(self, l, sdfg=None):
        if l is None:
            return None
        return set(l)

    def __get__(self, obj, objtype=None):
        val = super(SetProperty, self).__get__(obj, objtype)
        if val is None:
            return val
        
        # Copy to avoid changes in the set at callee to be reflected in
        # the node directly
        return set(val)

    def __set__(self, obj, val):
        if val is None:
            return super(SetProperty, self).__set__(obj, val)
        
        # Check for uniqueness
        if len(val) != len(set(val)):
            dups = set([x for x in val if val.count(x) > 1])
            raise ValueError('Duplicates found in set: ' + str(dups))
        # Cast to element type
        try:
            new_set = set(self._element_type(elem) for elem in val)
        except (TypeError, ValueError):
            raise ValueError('Some elements could not be converted to %s' % (str(self._element_type)))

        super(SetProperty, self).__set__(obj, new_set)


class LambdaProperty(Property):
    """ Custom Property type that accepts a lambda function, with conversions
        to and from strings. """

    @property
    def dtype(self):
        return None

    @staticmethod
    def from_string(s):
        return ast.parse(s).body[0].value

    @staticmethod
    def to_string(obj):
        if obj is None:
            return 'lambda: None'
        if isinstance(obj, str):
            return unparse(ast.parse(obj))
        return unparse(obj)

    def to_json(self, obj):
        if obj is None: return None
        return LambdaProperty.to_string(obj)

    def from_json(self, s, sdfg=None):
        if s == None: return None
        return LambdaProperty.from_string(s)

    def __set__(self, obj, val):
        if val is not None:
            if isinstance(val, str):
                self.from_string(val)  # Check that from_string doesn't fail
            elif isinstance(val, ast.Lambda):
                val = self.to_string(val)  # Store as string internally
            else:
                raise TypeError("Lambda property must be either string or ast.Lambda")
        super(LambdaProperty, self).__set__(obj, val)


class CodeBlock(object):
    """ Helper class that represents code blocks with language.
        Used in `CodeProperty`, implemented as a list of AST statements if
        language is Python, or a string otherwise.
    """

    def __init__(self,
                 code: Union[str, List[ast.AST], 'CodeBlock'],
                 language: dace.dtypes.Language = dace.dtypes.Language.Python):
        if isinstance(code, CodeBlock):
            self.code = code.code
            self.language = code.language
            return

        self.language = language

        # Convert to the right type
        if language == dace.dtypes.Language.Python and isinstance(code, str):
            self.code = ast.parse(code).body
        elif (not isinstance(code, str) and language != dace.dtypes.Language.Python):
            raise TypeError('Only strings are supported for languages other '
                            'than Python')
        else:
            self.code = code

    def get_free_symbols(self, defined_syms: Set[str] = None) -> Set[str]:
        """
        Returns the set of free symbol names in this code block, excluding
        the given symbol names.
        """
        defined_syms = defined_syms or set()

        # Search AST for undefined symbols
        if self.language == dace.dtypes.Language.Python:
            visitor = TaskletFreeSymbolVisitor(defined_syms)
            if self.code:
                for stmt in self.code:
                    visitor.visit(stmt)
            return visitor.free_symbols

        return set()

    @property
    def as_string(self) -> str:
        if isinstance(self.code, str) or self.code is None:
            return self.code
        return unparse(self.code)

    @as_string.setter
    def as_string(self, code):
        if self.language == dace.dtypes.Language.Python:
            self.code = ast.parse(code).body
        else:
            self.code = code

    def to_json(self):
        # Two roundtrips to avoid issues in AST parsing/unparsing of negative
        # numbers, i.e., "(-1)" becomes "(- 1)"
        if self.language == dace.dtypes.Language.Python and self.code is not None:
            code = unparse(ast.parse(self.as_string))
        else:
            code = self.as_string

        ret = {'string_data': code, 'language': self.language.name}
        return ret

    @staticmethod
    def from_json(tmp, sdfg=None):
        if tmp is None:
            return None
        if isinstance(tmp, CodeBlock):
            return tmp

        try:
            lang = tmp['language']
        except:
            lang = None

        if lang == "NoCode":
            return None

        if lang is None:
            lang = dace.dtypes.Language.Python
        elif lang.endswith("Python"):
            lang = dace.dtypes.Language.Python
        elif lang.endswith("CPP"):
            lang = dace.dtypes.Language.CPP
        elif lang.endswith("sv") or lang.endswith("systemverilog"):
            lang = dace.dtypes.Language.SystemVerilog
        elif lang.endswith("MLIR"):
            lang = dace.dtypes.Language.MLIR

        try:
            cdata = tmp['string_data']
        except:
            print("UNRECOGNIZED CODE JSON: " + str(tmp))
            cdata = ""

        return CodeBlock(cdata, lang)


class CodeProperty(Property):
    """ Custom Property type that accepts code in various languages. """

    @property
    def dtype(self):
        return CodeBlock

    def to_json(self, obj):
        if obj is None:
            return None

        # Two roundtrips to avoid issues in AST parsing/unparsing of negative
        # numbers, i.e., "(-1)" becomes "(- 1)"
        if obj.language == dace.dtypes.Language.Python and obj.code is not None:
            code = unparse(ast.parse(obj.as_string))
        else:
            code = obj.as_string

        ret = {'string_data': code, 'language': obj.language.name}
        return ret

    def from_json(self, tmp, sdfg=None):

        if tmp is None:
            return None
        if isinstance(tmp, CodeBlock):
            return tmp

        try:
            lang = tmp['language']
        except:
            lang = None

        if lang == "NoCode":
            return None

        if lang is None:
            lang = dace.dtypes.Language.Python
        elif lang.endswith("Python"):
            lang = dace.dtypes.Language.Python
        elif lang.endswith("CPP"):
            lang = dace.dtypes.Language.CPP
        elif lang.endswith("sv") or lang.endswith("SystemVerilog"):
            lang = dace.dtypes.Language.SystemVerilog
        elif lang.endswith("MLIR"):
            lang = dace.dtypes.Language.MLIR

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
        return CodeBlock(string, language)

    @staticmethod
    def to_string(obj):
        if isinstance(obj, str):
            return obj
        return obj.as_string


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
        if isinstance(val, str):
            val = self.from_string(val)
        if (val is not None and not isinstance(val, sbs.Range) and not isinstance(val, sbs.Indices)):
            raise TypeError("Subset property must be either Range or Indices: got {}".format(type(val).__name__))
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

    def to_json(self, val):
        if val is None:
            return None
        try:
            return val.to_json()
        except AttributeError:
            return SubsetProperty.to_string(val)

    def from_json(self, val, sdfg=None):
        return dace.serialize.from_json(val)


class SymbolicProperty(Property):
    """ Custom Property type that accepts integers or Sympy expressions. """

    @property
    def dtype(self):
        return None

    def __set__(self, obj, val):
        if (val is not None and not isinstance(val, (sp.Expr, Number, np.bool_, str))):
            raise TypeError(f"Property {self.attr_name} must be a literal "
                            f"or symbolic expression, got: {type(val)}")
        if isinstance(val, (Number, str)):
            val = SymbolicProperty.from_string(str(val))

        super(SymbolicProperty, self).__set__(obj, val)

    @staticmethod
    def from_string(s):
        return pystr_to_symbolic(s, simplify=False)

    @staticmethod
    def to_string(obj):
        # Go through sympy once to reorder factors
        return str(pystr_to_symbolic(str(obj), simplify=False))


class DataProperty(Property):
    """ Custom Property type that represents a link to a data descriptor.
        Needs the SDFG to be passed as an argument to `from_string` and
        `choices`. """

    def __init__(self, desc='', default=None, **kwargs):
        # Data can be None when no data is flowing, e.g., on a memlet with a
        # map that has no external inputs
        return super().__init__(dtype=str, allow_none=True, desc=desc, default=default, **kwargs)

    def typestring(self):
        return "DataProperty"

    @staticmethod
    def choices(sdfg=None):
        if sdfg is None:
            raise TypeError("Must pass SDFG as second argument to "
                            "choices method of ArrayProperty")
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

    def to_json(self, obj):
        if obj is None:
            return None
        return str(obj)

    def from_json(self, s, context=None):
        if isinstance(context, dace.SDFG):
            sdfg = context
        else:
            sdfg = context['sdfg']
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
        return tuple([dace.symbolic.pystr_to_symbolic(m.group(0)) for m in re.finditer("[^,;:]+", s)])

    @staticmethod
    def to_string(obj):
        return ", ".join(map(str, obj))

    def to_json(self, obj):
        if obj is None:
            return None
        return list(map(str, obj))

    def from_json(self, d, sdfg=None):
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

    @staticmethod
    def from_string(s):
        dtype = pydoc.locate(s)
        if dtype is None:
            raise ValueError("No type \"{}\" found.".format(s))
        if not isinstance(dtype, type):
            raise ValueError("Object \"{}\" is not a type.".format(dtype))
        return dtype

    @staticmethod
    def from_json(obj, context=None):
        if obj is None:
            return None
        if isinstance(obj, str):
            return TypeProperty.from_string(obj)
        else:
            raise TypeError("Cannot parse type from: {}".format(obj))


class TypeClassProperty(Property):
    """ Custom property type for memory as defined in dace.types,
        e.g. `dace.float32`. """

    def __get__(self, obj, objtype=None) -> typeclass:
        return super().__get__(obj, objtype)

    @property
    def dtype(self):
        return typeclass

    @staticmethod
    def from_string(s):
        dtype = pydoc.locate("dace.dtypes.{}".format(s))
        if dtype is None or not isinstance(dtype, dace.dtypes.typeclass):
            raise ValueError("Not a valid data type: {}".format(s))
        return dtype

    @staticmethod
    def to_string(obj):
        return obj.to_string()

    def to_json(self, obj):
        if obj is None:
            return None
        return obj.dtype.to_json()

    @staticmethod
    def from_json(obj, context=None):
        if obj is None:
            return None
        elif isinstance(obj, str):
            return TypeClassProperty.from_string(obj)
        elif isinstance(obj, dict):
            # Let the deserializer handle this
            return dace.serialize.from_json(obj)
        else:
            raise TypeError("Cannot parse type from: {}".format(obj))


class NestedDataClassProperty(Property):
    """ Custom property type for nested data. """

    def __get__(self, obj, objtype=None) -> 'Data':
        return super().__get__(obj, objtype)

    @property
    def dtype(self):
        from dace import data as dt
        return dt.Data

    @staticmethod
    def from_string(s):
        from dace import data as dt
        dtype = getattr(dt, s, None)
        if dtype is None or not isinstance(dtype, dt.Data):
            raise ValueError("Not a valid data type: {}".format(s))
        return dtype

    @staticmethod
    def to_string(obj):
        return obj.to_string()

    def to_json(self, obj):
        if obj is None:
            return None
        return obj.to_json()

    @staticmethod
    def from_json(obj, context=None):
        if obj is None:
            return None
        elif isinstance(obj, str):
            return NestedDataClassProperty.from_string(obj)
        elif isinstance(obj, dict):
            # Let the deserializer handle this
            return dace.serialize.from_json(obj)
        else:
            raise TypeError("Cannot parse type from: {}".format(obj))


class LibraryImplementationProperty(Property):
    """
    Property for choosing an implementation type for a library node. On the
    Python side it is a standard property, but can expand into a combo-box in the editor.
    """

    def typestring(self):
        return "LibraryImplementationProperty"


class DataclassProperty(Property):
    """
    Property that stores pydantic models or dataclasses.
    """

    @staticmethod
    def to_string(obj):
        return str(obj)

    @staticmethod
    def from_string(s):
        raise TypeError('Dataclasses cannot be loaded from a string, only JSON')

    def to_json(self, obj):
        if obj is None:
            return None
        return obj.dict()

    def from_json(self, d, sdfg=None):
        if d is None:
            return None
        return self.dtype.parse_obj(d)
