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

###############################################################################
# External interface to guarantee correct usage
###############################################################################


def set_property_from_string(prop, obj, string, sdfg=None):
    """ Interface function that guarantees that a property will always be
    correctly set, if possible, by accepting all possible input arguments to
    from_string. """

    # If the property is a string (property name), obtain it from the object
    if isinstance(prop, str):
        prop = type(obj).__properties__[prop]

    if isinstance(prop, CodeProperty):
        val = prop.from_string(string, obj.language)
    elif isinstance(prop, (ReferenceProperty, DataProperty)):
        if sdfg is None:
            raise ValueError(
                "You cannot pass sdfg=None when editing a ReferenceProperty!")
        val = prop.from_string(string, sdfg)
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
            enum=None,  # Values must be present in this enum
            unmapped=False,  # Don't enforce 1:1 mapping with a member variable
            allow_none=False,
            indirected=False,  # This property belongs to a different class
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
        self._enum = enum
        self._unmapped = unmapped
        self._allow_none = allow_none
        self._indirected = indirected
        self._desc = desc

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

    # Property-ception >:-)

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


###############################################################################
# Custom properties
###############################################################################


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
            unmapped=False,  # Don't enforce 1:1 mapping with a member variable
            allow_none=False,
            desc=""):
        super(SetProperty, self).__init__(
            getter=getter,
            setter=setter,
            dtype=set,
            default=default,
            from_string=from_string,
            to_string=to_string,
            enum=None,
            unmapped=unmapped,
            allow_none=allow_none,
            desc=desc)
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
    def from_string(string, language=None):
        if language is None:
            raise TypeError("Must pass language as second argument to "
                            "from_string method of CodeProperty")
        if language == dace.types.Language.Python:
            block = CodeBlock(ast.parse(string).body)
            block.as_string = string
            return block
        else:
            # Do nothing for now
            return string

    @staticmethod
    def to_string(obj):
        if isinstance(obj, str):
            return obj
        # Grab the originally parsed string if any
        if obj._as_string is not None and obj._as_string != "":
            return obj._as_string
        # It's probably good enough to assume that there is an original string
        # if the language was not Python, so we just throw the string to the
        # astunparser.
        return unparse(obj)

    def __set__(self, obj, val):
        # Check if the class has a language property
        if not hasattr(type(obj), "language"):
            raise AttributeError(
                "Class \"{}\" with a CodeProperty field must also "
                "have a \"language\" attribute.".format(type(obj).__name__))
        # Check if the object has a language attribute
        try:
            language = obj.language
        except AttributeError:
            # Language exists as an attribute, but has not yet been set. Accept
            # this, because __dict__ is not guaranteed to be in the order that
            # the attributes are defined in.
            language = None
        if val is None:
            # Keep as None. The "allow_none" check in the superclass
            # ensures that this is legal
            pass
        elif isinstance(val, str):
            if language is not None:
                # Store original string
                val = self.from_string(val, language)
        else:
            try:
                if language is not dace.types.Language.Python:
                    raise TypeError("Only strings accepted for other "
                                    "languages than Python.")
            except AttributeError:
                # Don't check language if it has not been set yet. We will
                # assume it's Python AST, since it wasn't a string
                pass
            if isinstance(val, (ast.FunctionDef, ast.With)):
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
        super(CodeProperty, self).__set__(obj, val)


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

    def __init__(self, desc='', default=None):
        # Data can be None when no data is flowing, e.g., on a memlet with a
        # map that has no external inputs
        return super().__init__(
            dtype=str, allow_none=True, desc=desc, default=default)

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
