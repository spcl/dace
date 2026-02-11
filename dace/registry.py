# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains class decorators to ease creating classes and enumerations whose
    subclasses and values can be registered externally. """

from dace import attr_enum
from enum import Enum
from typing import Dict, Type, TypeVar, TYPE_CHECKING

T = TypeVar('T')
E = TypeVar('E', bound=attr_enum.ExtensibleAttributeEnum)


def make_registry(cls: Type[T]) -> Type[T]:
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


def undefined_safe_enum(cls: type[E]) -> type[E]:
    """
    Decorator that adds a value ``Undefined`` to an enumeration.
    """
    if not issubclass(cls, attr_enum.ExtensibleAttributeEnum):
        raise TypeError("Only ExtensibleAttributeEnum subclasses may be used with undefined values")
    cls.register('Undefined')
    return cls
