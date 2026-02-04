# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests the extensible attributed enum class.
"""
from dataclasses import dataclass
from enum import auto, Enum

from dace.attr_enum import ExtensibleAttributeEnum
from dace import serialize


@dataclass(frozen=True)
class ShipData:
    imo_number: str
    tonnage: float


@serialize.serializable
class Vehicle(ExtensibleAttributeEnum):
    # Inline dataclass templates
    @dataclass(frozen=True)
    class CAR:
        vin: str
        make: str
        horsepower: int

    @dataclass(frozen=True)
    class TRUCK:
        vin: str
        payload_capacity: float

    @dataclass(frozen=True)
    class BOAT:
        hull_id: str
        length_ft: float

    UNKNOWN = auto()  # Auto-assigned value
    BICYCLE = "bike"  # Normal value
    SHIP = ShipData  # Non-inline dataclass template


def test_attrenum_creation():
    car = Vehicle.CAR(vin="123", make="Toyota", horsepower=200)

    assert repr(car) == "<Vehicle.CAR: Vehicle.CAR(vin='123', make='Toyota', horsepower=200)>"
    assert repr(Vehicle.CAR) == "<Vehicle.CAR (template)>"

    # Checking attributes
    assert car.make == "Toyota"
    assert car.horsepower == 200
    assert Vehicle.UNKNOWN.value == 1
    assert Vehicle.BICYCLE.value == "bike"

    # Checking isinstance
    assert isinstance(Vehicle.CAR, Vehicle)
    assert isinstance(car, Vehicle)


def test_attrenum_dynamic_creation():
    assert Vehicle["CAR"] == Vehicle.CAR
    assert Vehicle(Vehicle.UNKNOWN.value) == Vehicle.UNKNOWN
    assert Vehicle("bike") == Vehicle.BICYCLE


def test_attrenum_listing():
    names = [v.name for v in Vehicle]
    assert names == ['UNKNOWN', 'BICYCLE', 'CAR', 'TRUCK', 'BOAT', 'SHIP']


def test_attrenum_equality():
    car = Vehicle.CAR(vin="123", make="Toyota", horsepower=200)
    truck = Vehicle.TRUCK(vin="456", payload_capacity=2000.0)
    car2 = Vehicle.CAR(vin="789", make="Toyota", horsepower=200)
    car3 = Vehicle.CAR(vin="123", make="Toyota", horsepower=200)

    # Template-instance equality
    assert Vehicle.CAR == car

    # Instance-instance equality
    assert car != car2
    assert car2 != truck

    # Template-template inequality
    assert Vehicle.CAR != Vehicle.TRUCK

    # The same checks for `is`
    assert car is car
    assert Vehicle.CAR is Vehicle.CAR
    assert car is car3
    # and `is not`
    assert Vehicle.CAR is not car
    assert car is not car2
    assert car2 is not truck
    assert Vehicle.CAR is not Vehicle.TRUCK


def test_attrenum_match():

    def describe(v: Vehicle):
        match v:
            case Vehicle.CAR:
                return "Car"
            case Vehicle.TRUCK:
                return "Truck"
            case Vehicle.BOAT:
                return "Boat"
            case Vehicle.UNKNOWN:
                return "Unknown vehicle"
            case _:
                raise ValueError("Unrecognized vehicle")

    car = Vehicle.CAR(vin="123", make="Toyota", horsepower=200)

    assert describe(car) == "Car"
    assert describe(Vehicle.TRUCK) == "Truck"


def test_attrenum_add_member():
    # Add a new member at runtime
    Vehicle.register("SCOOTER", "scooter")
    scooter = Vehicle.SCOOTER
    assert repr(scooter) == "<Vehicle.SCOOTER: scooter>"
    assert scooter.value == "scooter"
    assert scooter in Vehicle

    @dataclass(frozen=True)
    class MotorbikeData:
        frame_id: str
        gear_count: int

    Vehicle.register_template("BIKE", MotorbikeData)

    motorbike = Vehicle.BIKE(frame_id="FRM123", gear_count=21)
    assert motorbike.frame_id == "FRM123"
    assert motorbike == Vehicle.BIKE


def test_serialization():
    # Serialize instance
    car = Vehicle.CAR(vin="123", make="Toyota", horsepower=200)
    car_json = serialize.to_json(car)
    assert car_json == {'type': 'Vehicle.CAR', 'vin': '123', 'make': 'Toyota', 'horsepower': 200}
    car_restored = serialize.from_json(car_json)
    assert car_restored == car

    # Serialize template
    truck_json = serialize.to_json(Vehicle.TRUCK)
    assert truck_json == {'type': 'Vehicle.TRUCK'}
    truck_restored = serialize.from_json(truck_json)
    assert truck_restored == Vehicle.TRUCK


def test_serialization_nestedclass():

    # Standard enumeration
    @serialize.serializable
    class IsPlane(Enum):
        FALSE = 0
        TRUE = 1

    # Extensible attributed enum
    @serialize.serializable
    class PlaneModelType(ExtensibleAttributeEnum):
        MODEL_ONE = auto()
        MODEL_TWO = auto()

    # Nested dataclass
    @serialize.serializable
    @dataclass(frozen=True)
    class PlaneDetails:
        range_km: int
        capacity: int
        is_plane: IsPlane = IsPlane.FALSE

    @dataclass(frozen=True)
    class PlaneData:
        model: PlaneModelType
        details: PlaneDetails
        is_plane: IsPlane = IsPlane.TRUE

    Vehicle.register_template("PLANE", PlaneData)
    plane = Vehicle.PLANE(model=PlaneModelType.MODEL_ONE, details=PlaneDetails(range_km=5000, capacity=180))
    plane_json = serialize.to_json(plane)
    assert plane_json == {
        'type': 'Vehicle.PLANE',
        'model': 'MODEL_ONE',
        'is_plane': 'TRUE',
        'details': {
            'type': 'PlaneDetails',
            'range_km': 5000,
            'capacity': 180,
            'is_plane': 'FALSE',
        }
    }
    plane_restored = serialize.from_json(plane_json)
    assert plane_restored == plane


if __name__ == "__main__":
    test_attrenum_creation()
    test_attrenum_dynamic_creation()
    test_attrenum_listing()
    test_attrenum_equality()
    test_attrenum_match()
    test_attrenum_add_member()
    test_serialization()
    test_serialization_nestedclass()
