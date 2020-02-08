""" Contains class decorators to ease creating classes and enumerations whose
    subclasses and values can be registered externally. """

from aenum import Enum, extend_enum
from typing import Type


def make_registry(cls: Type):
    """
    Decorator that turns a class into a user-extensible class with three
    class methods: ``register``, ``unregister``, and ``extensions``.

    The first method accepts one class parameter and registers it into the
    extensions, the second method removes the class parameter from the
    registry, and the third method returns a list of currently-registered
    extensions.
    """
    cls._registry_ = set()
    cls.register = lambda subclass: cls._registry_.add(subclass)
    cls.unregister = lambda subclass: cls._registry_.remove(subclass)
    cls.extensions = lambda: list(cls._registry_)

    return cls


def autoregister(cls: Type):
    """
    Decorator for subclasses of user-extensible classes (see ``make_registry``)
    that automatically registers the subclass with the superclass registry upon
    creation.
    """
    registered = False
    for base in cls.__bases__:
        if hasattr(base, '_registry_') and hasattr(base, 'register'):
            base.register(cls)
            registered = True
    if not registered:
        raise TypeError('Class does not extend registry classes')
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
    cls.register = lambda name, *args: _extend_enum(cls, name, *args)
    return cls


#############
# Helper functions


def _extend_enum(cls: Type, name: str, *value):
    extend_enum(cls, name, *value)
