import ast
import astunparse
import enum
from collections import OrderedDict
import copy
from dace.frontend.python.astutils import unparse
import itertools
import json
import pydoc
import re
import sympy as sp
import numpy as np
import dace.subsets as sbs
import dace
import dace.serialize
from dace.symbolic import pystr_to_symbolic
from dace.dtypes import DebugInfo

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
            choices=None,  # Values must be present in this enum
            unmapped=False,  # Don't enforce 1:1 mapping with a member variable
            allow_none=False,
            indirected=False,  # This property belongs to a different class
            category='General',
            desc=""):

        self._getter = getter
        self._setter = setter
        self._dtype = dtype
        self._default = default

        if allow_none is False and default is None:
            try:
                self._default = dtype()
            except TypeError:
                if hasattr(self, 'dtype'):
                    try:
                        self._default = self.dtype()
                    except TypeError:
                        raise TypeError(
                            'Default not properly defined for property')
                else:
                    raise TypeError(
                        'Default not properly defined for property')

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

            def f(obj, *args, **kwargs):
                if isinstance(obj, str) and self._from_string is not None:
                    # The serializer does not know about this property, so if
                    # we can convert using our to_string method, do that here
                    return self._from_string(obj)
                # Otherwise ship off to the serializer, telling it which type
                # it's dealing with as a sanity check
                return dace.serialize.from_json(
                    obj, *args, known_type=dtype, **kwargs)

            self._from_json = f
        else:
            self._from_json = from_json

        if to_json is None:
            self._to_json = dace.serialize.to_json
        else:
            self._to_json = to_json

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
        if self.choices is not None and isinstance(self.choices,
                                                   (list, tuple, set)):
            if val not in self.choices:
                raise ValueError("Value {} not present in choices: {}".format(
                    val, self.choices))
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


class ListProperty(Property):
    """ Property type for lists.
    """

    def __init__(self, element_type, *args, **kwargs):
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
            return l
        # Otherwise, convert to strings
        return list(map(str, l))

    @staticmethod
    def from_string(s):
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


###############################################################################
# Custom properties
###############################################################################


class SDFGReferenceProperty(Property):
    def to_json(self, obj):
        if obj is None:
            return None
        return dace.serialize.dumps(obj.to_json())  # Make a string of a JSON

    def from_json(self, obj, context=None):
        if obj is None:
            return None

        # Parse the string of the JSON back into an SDFG object
        # Need to use regular json.loads instead of dace.serialize.dumps
        return dace.SDFG.from_json(json.loads(obj), context)


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

    def to_json(self, l):
        return l

    def from_json(self, l, sdfg=None):
        return l


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
        return [eval(i) for i in re.sub("[\{\}\(\)\[\]]", "", s).split(",")]

    def to_json(self, l):
        return list(sorted(l))

    def from_json(self, l, sdfg=None):
        return set(l)

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

    def to_json(self, obj):
        return str(obj)

    def from_json(self, s, sdfg=None):
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

    def to_json(self, obj):
        lang = dace.dtypes.Language.Python
        if obj is None:
            return None

        if isinstance(obj, dict):
            lang = obj['language']

        ret = {
            'string_data': CodeProperty.to_string(obj),
            'language': lang.name
        }
        return ret

    def from_json(self, tmp, sdfg=None):

        if tmp is None:
            return None

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
        if language == dace.dtypes.Language.Python:
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
        if hasattr(obj, "_as_string") and obj._as_string:
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
                language = dace.dtypes.Language.Python
            if language is not None:
                # Store original string
                val = self.from_string(val, language)['code_or_block']
        else:
            try:
                language = val['language']
                val = val['code_or_block']
            except:
                # Default to Python
                language = dace.dtypes.Language.Python
            try:
                if language is not dace.dtypes.Language.Python and not isinstance(
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
                # The original parsing should have already stripped this,
                # but it's still good to handle this case.
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
        if isinstance(val, str):
            val = self.from_string(val)
        if (val is not None and not isinstance(val, sbs.Range)
                and not isinstance(val, sbs.Indices)):
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
        if (not isinstance(val, sp.expr.Expr) and not isinstance(val, int)
                and not isinstance(val, str)):
            raise TypeError(
                "Property {} must an int or symbolic expression".format(
                    self.attr_name))
        if isinstance(val, (int, float, str, complex)):
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
        return super().__init__(
            dtype=str, allow_none=True, desc=desc, default=default, **kwargs)

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
        return tuple([
            dace.symbolic.pystr_to_symbolic(m.group(0))
            for m in re.finditer("[^,;:]+", s)
        ])

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

    @property
    def dtype(self):
        return dace.dtypes.typeclass

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
