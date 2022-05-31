# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains class decorators to ease creating classes and enumerations whose
    subclasses and values can be registered externally. """

from aenum import AutoNumberEnum, Enum, EnumType, extend_enum
from typing import Dict, Type


def make_registry(cls: Type):
    """
    Decorator that turns a class into a user-extensible class with three
    class methods: ``register``, ``unregister``, and ``extensions``.

    The first method accepts one class parameter and registers it into the
    extensions, the second method removes the class parameter from the
    registry, and the third method returns a list of currently-registered
    extensions.
    """
    def _register(cls: Type, subclass: Type, kwargs: Dict):
        cls._registry_[subclass] = kwargs

    def _unregister(cls: Type, subclass: Type):
        del cls._registry_[subclass]

    cls._registry_ = {}
    cls.register = lambda subclass, **kwargs: _register(cls, subclass, kwargs)
    cls.unregister = lambda subclass: _unregister(cls, subclass)
    cls.extensions = lambda: cls._registry_

    return cls


def autoregister(cls: Type, **kwargs):
    """
    Decorator for subclasses of user-extensible classes (see ``make_registry``)
    that automatically registers the subclass with the superclass registry upon
    creation.
    """
    registered = False
    for base in cls.__bases__:
        if hasattr(base, '_registry_') and hasattr(base, 'register'):
            base.register(cls, **kwargs)
            registered = True
    if not registered:
        raise TypeError('Class does not extend registry classes')
    return cls


def autoregister_params(**params):
    """
    Decorator for subclasses of user-extensible classes (see ``make_registry``)
    that automatically registers the subclass with the superclass registry upon
    creation. Uses the arguments given to register the value of the subclass.
    """
    return lambda cls: autoregister(cls, **params)


def undefined_safe_enum(cls: Type):
    """
    Decorator that adds a value ``Undefined`` to an enumeration.
    """
    if not issubclass(cls, Enum):
        raise TypeError("Only aenum.Enum subclasses may be used with undefined values")
    extend_enum(cls, 'Undefined')
    return cls


def extensible_enum(cls: Type):
    """
    Decorator that adds a function called ``register`` to an enumeration,
    extending its values. Note that new values cannot be unregistered.

    New entries can be registered either with a single, string argument for
    a new name (a value will be auto-assigned), or with additional arguments
    for the value.
    """
    if not issubclass(cls, Enum):
        raise TypeError("Only aenum.Enum subclasses may be made extensible")

    def _extend_enum(cls: Type, name: str, *value):
        extend_enum(cls, name, *value)

    cls.register = lambda name, *args: _extend_enum(cls, name, *args)
    return cls


class EnumElement:

    def __str__(self):
        return type(self).__name__

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return type(self).__name__ == type(other).__name__
    
    def to_json(self):
        from dace.serialize import to_json
        attr = {attr: to_json(value) for attr, value in self.__dict__.items()}
        ret = {'type': str(self), 'attributes': attr}
        return ret


class EMC(EnumType):

    def __init__(cls, *args, **kwds):
        super().__init__(*args, **kwds)
        elems = {
            k: v
            for k, v in object.__getattribute__(cls, '__dict__').items()
            if isinstance(v, type) and issubclass(v, EnumElement)
        }
        cls._classmembers_ = elems

    def __getattribute__(cls, name):
        try:
            clsmembers = object.__getattribute__(cls, '_classmembers_')
            if name in clsmembers:
                return object.__getattribute__(cls, name)()
        except AttributeError:
            pass
        return super().__getattribute__(name)

    def __instancecheck__(cls, instance):
        if isinstance(instance, str):
            return False
        try:
            object.__getattribute__(cls, str(instance))
        except AttributeError:
            return False
        return True
    
    def __getitem__(cls, name):
        item = cls.__class__.__getattribute__(cls, name)
        if isinstance(item, type):
            return item()
        return item


class AttributedEnum(AutoNumberEnum, metaclass=EMC):
    pass
