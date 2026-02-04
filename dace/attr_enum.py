# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dataclasses import is_dataclass
from enum import Enum, EnumMeta
from typing import Any


class _ExtensibleAttributeEnumMeta(EnumMeta):
    """Metaclass that converts nested dataclasses into enum members."""

    def __new__(mcs, name, bases, namespace, **kwargs):
        # Find and remove dataclasses from namespace before enum creation
        dataclass_members = {}
        for key in list(namespace.keys()):
            if not key.startswith('_'):
                value = namespace[key]
                if isinstance(value, type) and is_dataclass(value):
                    dataclass_members[key] = value

        # Remove dataclasses from namespace so EnumMeta doesn't see them
        for key in dataclass_members:
            del namespace[key]
            # Also remove from EnumDict's member tracking if present
            if hasattr(namespace, '_member_names') and key in namespace._member_names:
                del namespace._member_names[key]

        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Add dataclasses as template members after enum creation
        for member_name, dc in dataclass_members.items():
            obj = object.__new__(cls)
            obj._name_ = member_name
            obj._value_ = dc
            obj._dataclass_type = dc
            obj._is_template = True

            cls._member_map_[member_name] = obj
            cls._member_names_.append(member_name)
            cls._value2member_map_[dc] = obj
            type.__setattr__(cls, member_name, obj)

        return cls


class ExtensibleAttributeEnum(Enum, metaclass=_ExtensibleAttributeEnumMeta):
    """
    An extensible Enum base class that also supports dataclass templates as members.

    This class extends the standard library's Enum to allow members that act as
    factories for dataclass instances. Each template member can be called to create
    registered instances that compare equal to their template.

    Supported Member Types
    ----------------------
    - **Dataclass templates**: Frozen dataclass types that act as factories
    - **Auto values**: Using ``auto()`` for automatic integer values
    - **Constant values**: Any hashable value (strings, integers, etc.)

    Template Syntax
    ---------------
    There are two ways to define dataclass template members:

    **1. Inline class definition:**

        >>> class Vehicle(ExtensibleAttributeEnum):
        ...     @dataclass(frozen=True)
        ...     class CAR:
        ...         vin: str
        ...         make: str
        ...
        ...     UNKNOWN = auto()
        ...     BICYCLE = "bike"

    **2. External dataclass assignment:**

        >>> @dataclass(frozen=True)
        ... class CarData:
        ...     vin: str
        ...     make: str
        ...
        >>> class Vehicle(ExtensibleAttributeEnum):
        ...     CAR = CarData
        ...     UNKNOWN = auto()
        ...     BICYCLE = "bike"

    Creating Instances
    ------------------
    Template members are callable and create registered instances:

        >>> car = Vehicle.CAR(vin="123", make="ABC")
        >>> car.vin
        '123'
        >>> car.make
        'ABC'

    Instances with identical arguments return the same object:

        >>> car1 = Vehicle.CAR(vin="123", make="ABC")
        >>> car2 = Vehicle.CAR(vin="123", make="ABC")
        >>> car1 is car2
        True

    Equality Behavior
    -----------------
    Equality behavior depends on what is being compared:

    - A template equals any instance created from it
    - Two instances are equal or unequal based on their dataclass equality
    - Instances of different templates are not equal

    Examples:

        >>> # Template equals its instances
        >>> car = Vehicle.CAR(vin="123", make="ABC")
        >>> Vehicle.CAR == car
        True
        >>> car2 = Vehicle.CAR(vin="123", make="ABC")
        >>> car == car2
        True
        >>> car3 = Vehicle.CAR(vin="456", make="DEF")
        >>> car == car3
        False

    This equality behavior enables natural use in match/case statements:

        >>> def process(v: Vehicle):
        ...     match v:
        ...         case Vehicle.CAR:
        ...             print(f"Car: {v.make}")
        ...         case Vehicle.TRUCK:
        ...             print(f"Truck: {v.vin}")
        ...
        >>> process(Vehicle.CAR(vin="123", make="ABC"))
        Car: ABC

    Runtime Member Addition
    -----------------------
    New members can be added at runtime using class methods:

        >>> Vehicle.register_template("BOAT", BoatData)
        >>> Vehicle.register("SKATEBOARD", "skateboard")

    Notes
    -----
    - Dataclasses used as templates should be frozen (``frozen=True``) to ensure
      hashability and immutability.
    - Non-template members (auto, constants) cannot be called and will raise
      ``TypeError`` if invoked.
    - Hashing is based on member name, so templates and their instances hash
      identically.
    """

    def _generate_next_value_(name, start, count, last_values):
        return count + 1

    def __new__(cls, value: Any):
        obj = object.__new__(cls)
        obj._value_ = value

        if isinstance(value, type) and is_dataclass(value):
            obj._dataclass_type = value
            obj._is_template = True
        else:
            obj._dataclass_type = None
            obj._is_template = False

        return obj

    def __call__(self, **kwargs):
        if not self._is_template:
            raise TypeError(f"{self._name_} is not a template and cannot be called")

        cls = self.__class__
        data = self._dataclass_type(**kwargs)

        if data in cls._value2member_map_:
            return cls._value2member_map_[data]

        obj = object.__new__(cls)
        obj._name_ = self._name_
        obj._value_ = data
        obj._dataclass_type = self._dataclass_type
        obj._is_template = False

        cls._value2member_map_[data] = obj
        return obj

    def __getattr__(self, name: str):
        if name.startswith('_'):
            raise AttributeError(name)
        if self._dataclass_type is not None and not isinstance(self._value_, type):
            return getattr(self._value_, name)
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self is other:
            return True
        # Different member names means not equal
        if self._name_ != other._name_:
            return False
        # Template equals any instance of the same name
        if self._is_template or other._is_template:
            return True
        # Two instances compare by their dataclass values
        return self._value_ == other._value_

    def __hash__(self):
        return hash(self._name_)

    def __repr__(self):
        if self._is_template:
            return f"<{self.__class__.__name__}.{self._name_} (template)>"
        return f"<{self.__class__.__name__}.{self._name_}: {self._value_}>"

    @classmethod
    def register(cls, name: str, value: Any = None):
        """
        Add a new member at runtime.

        :param name: The name for the new member.
        :param value: The value for the member. If a dataclass type, it becomes a template.
                      If None is given, behaves like ``auto()``.
        :return: The newly created member.
        :raises ValueError: If a member with the given name already exists.


        Examples
        --------
        >>> Vehicle.register("SCOOTER", "scooter")
        >>> Vehicle.SCOOTER.value
        'scooter'
        """
        if name in cls._member_map_:
            raise ValueError(f"Member '{name}' already exists")

        obj = object.__new__(cls)
        obj._name_ = name
        if value is None:
            value = cls._generate_next_value_(name, 1, len(cls._member_names_), [])
        obj._value_ = value

        if isinstance(value, type) and is_dataclass(value):
            obj._dataclass_type = value
            obj._is_template = True
        else:
            obj._dataclass_type = None
            obj._is_template = False

        cls._member_map_[name] = obj
        cls._member_names_.append(name)
        cls._value2member_map_[value] = obj
        type.__setattr__(cls, name, obj)

        return obj

    @classmethod
    def register_template(cls, name: str, dataclass_type: type):
        """
        Add a new dataclass template member at runtime.

        :param name: The name for the new template member.
        :param dataclass_type: A dataclass type to use as the template.
        :return: The newly created template member.
        :raises TypeError: If dataclass_type is not a dataclass.
        :raises ValueError: If a member with the given name already exists.


        Examples
        --------
        >>> @dataclass(frozen=True)
        ... class BoatData:
        ...     hull_id: str
        ...     length_ft: float
        ...
        >>> Vehicle.register_template("BOAT", BoatData)
        >>> boat = Vehicle.BOAT(hull_id="HUL001", length_ft=25.5)
        >>> boat.length_ft
        25.5
        """
        if not is_dataclass(dataclass_type):
            raise TypeError(f"{dataclass_type} is not a dataclass")
        return cls.register(name, dataclass_type)

    @classmethod
    def from_json(cls, json_obj: dict[str, Any], context=None):
        """
        Deserialize an ExtensibleAttributeEnum member from a JSON object.
        """
        type_name = json_obj['type']
        member_name = type_name.split('.')[-1]
        member = cls[member_name]

        if member._is_template:
            # Try to create instance from dataclass fields
            from dace import serialize
            data_fields = {
                k: serialize.from_json(v, context, known_type=member._dataclass_type.__dataclass_fields__[k].type)
                for k, v in json_obj.items() if k != 'type'
            }
            if data_fields:
                return member(**data_fields)
            return member  # Uninstantiated template
        else:
            return member
