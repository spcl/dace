from typing import List, Dict, Optional
from numbers import Number
import numpy as np
import json


class Measurement:
    unit: str
    name: str
    data: List[Number]
    kernel_name: Optional[str]

    def __init__(self, name: str, unit: str, data: Optional[List[Number]] = None, kernel_name: Optional[str] = None):
        self.name = name
        self.unit = unit
        self.data = [] if data is None else data
        self.kernel_name = kernel_name

    def add_value(self, value: Number):
        self.data.append(value)

    def min(self):
        return min(self.data)

    def max(self):
        return max(self.data)

    def average(self):
        return np.average(self.data)

    def median(self):
        return np.median(self.data)

    def amount(self):
        return len(self.data)

    @staticmethod
    def to_json(measurement: 'Measurement') -> Dict:
        return {
                "__Measurement__": True,
                "unit": measurement.unit,
                "name": measurement.name,
                "data": measurement.data,
                "kernel_name": measurement.kernel_name
                }

    @staticmethod
    def from_json(dict: Dict) -> 'Measurement':
        if '__Measurement__' in dict:
            return Measurement(dict['name'], dict['unit'], data=dict['data'], kernel_name=dict['kernel_name'])


class ProgramMeasurement:
    measurements: Dict[str, Measurement]
    program: str
    parameters: Dict[str, Number]

    def __init__(self, program: str, parameters: Dict, measurements: Dict = None):
        self.program = program
        self.parameters = parameters
        self.measurements = {} if measurements is None else measurements

    def add_measurement(self, name: str, unit: str, **kwargs):
        self.measurements[name] = Measurement(name, unit, **kwargs)

    def add_value(self, name: str, value: Number):
        """
        Adds a value to a measurement identified by the name

        :param name: The name of the measurement
        :type name: str
        :param value: The value to add
        :type value: Number
        """
        self.measurements[name].add_value(value)

    @staticmethod
    def to_json(measurement: 'ProgramMeasurement') -> Dict:
        msr_dict = {}
        for msr in measurement.measurements.values():
            msr_dict[msr.name] = Measurement.to_json(msr)
        return {
                "__ProgramMeasurement__": True,
                "measurements": msr_dict,
                "program": measurement.program,
                "parameters": measurement.parameters,
                }

    @staticmethod
    def from_json(dict: Dict) -> 'ProgramMeasurement':
        if '__ProgramMeasurement__' in dict:
            return ProgramMeasurement(dict['program'], dict['parameters'],
                                      measurements=dict['measurements'])
        elif '__Measurement__' in dict:
            return Measurement.from_json(dict)
        else:
            return dict


