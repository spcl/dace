""" Collection of classes representing measurement data"""
from typing import List, Dict, Optional
from numbers import Number
import numpy as np
from subprocess import run
from os import path
from datetime import datetime


class Measurement:
    """
    Represents a measurement. A measurement is always measuring the same and using the same unit but can have several
    measurements for different repetitions
    """
    unit: str
    name: str
    data: List[Number]
    kernel_name: Optional[str]

    def __init__(self, name: str, unit: str, data: Optional[List[Number]] = None, kernel_name: Optional[str] = None):
        """
        Constructs the class

        :param name: The name of the measurement
        :type name: str
        :param unit: The unit of the measurement
        :type unit: str
        :param data: The data itself, defaults to None
        :type data: Optional[List[Number]], optional
        :param kernel_name: The name of the kernel measured if applicable, defaults to None
        :type kernel_name: Optional[str], optional
        """
        self.name = name
        self.unit = unit
        self.data = [] if data is None else data
        self.kernel_name = kernel_name

    def add_value(self, value: Number):
        """
        Adds a value to the measurement. The user must ensure that it is of the same unit and measures the same thing.

        :param value: The value/measurement to add
        :type value: Number
        """
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
    """
    Represents all measurements made with the same program. It consists of a list of measurements, the program name and
    the parameters used to achieve these measurements.
    """
    measurements: Dict[str, Measurement]
    program: str
    parameters: Dict[str, Number]

    def __init__(self, program: str, parameters: Dict, measurements: Dict = None):
        """
        Constructs the class.

        :param program: The name of the program
        :type program: str
        :param parameters: The parameters used
        :type parameters: Dict
        :param measurements: The measurements. Key is the measurement name, value the Measurement object,
                             defaults to None
        :type measurements: Dict, optional
        """
        self.program = program
        self.parameters = parameters
        self.measurements = {} if measurements is None else measurements

    def add_measurement(self, name: str, unit: str, **kwargs):
        """
        Adds a new measurement, specifing what it measures (the name) and unit. Takes also additional optional arguments
        to be passed to the Measurement constructor (e.g. for the kernel name)

        :param name: The name of the new measurement
        :type name: str
        :param unit: The unit of the new measurement
        :type unit: str
        """
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


class MeasurementRun:
    """
    Represents a measurement run which consits of a set of programs which are run with their data, the description of
    the run, the time it was run and the git hash.
    """
    description: str
    data: List[ProgramMeasurement]
    git_hash: str
    date: datetime

    def __init__(self,
                 description: str,
                 data: List[ProgramMeasurement] = [],
                 git_hash: str = '',
                 date: datetime = datetime.now()):
        """
        Constructs the class

        :param description: The description of this measurement run
        :type description: str
        :param data: The data itself as a list of ProgramMeasurements, defaults to []
        :type data: List[ProgramMeasurement], optional
        :param git_hash: The short git hash, if not given, will be read automatically, defaults to ''
        :type git_hash: str, optional
        :param date: The time the measurements were run, defaults to datetime.now()
        :type date: datetime, optional
        """
        self.description = description
        self.data = data
        self.git_hash = git_hash
        if self.git_hash == '':
            hash_output = run(['git', 'rev-parse', '--short', 'HEAD'],
                              cwd=path.split(path.dirname(__file__))[0],
                              capture_output=True)
            self.git_hash = hash_output.stdout.decode('UTF-8')
        self.date = date

    def add_program_data(self, data: ProgramMeasurement):
        self.data.append(data)

    @staticmethod
    def to_json(run: 'MeasurementRun') -> Dict:
        data_list = []
        for data in run.data:
            data_list.append(ProgramMeasurement.to_json(data))
        return {
                '__MeasurementRun__': True,
                'description': run.description,
                'git_hash': run.git_hash,
                'data': data_list,
                'date': run.date.isoformat()
                }

    @staticmethod
    def from_json(dict: Dict) -> 'MeasurementRun':
        if '__MeasurementRun__' in dict:
            return MeasurementRun(dict['description'], data=dict['data'], git_hash=dict['git_hash'],
                                  date=datetime.fromisoformat(dict['date']))
        if '__ProgramMeasurement__' in dict:
            return ProgramMeasurement(dict['program'], dict['parameters'],
                                      measurements=dict['measurements'])
        elif '__Measurement__' in dict:
            return Measurement.from_json(dict)
        else:
            return dict
