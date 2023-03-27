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
        print(f"Create Measurement with name: {name}, unit: {unit}")
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

    def __repr__(self) -> str:
        return f"Measurement of {self.name} [{self.unit}] kernel {self.kernel_name} and {len(self.data)} data points"

    def __eq__(self, other: 'Measurement') -> bool:
        return \
            self.unit == other.unit and \
            self.name == other.name and \
            self.data == other.data and \
            self.kernel_name == other.kernel_name


class ProgramMeasurement:
    """
    Represents all measurements made with the same program. It consists of a list of measurements, the program name and
    the parameters used to achieve these measurements.
    """
    measurements: Dict[str, List[Measurement]]
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
        if name not in self.measurements:
            self.measurements[name] = []
        self.measurements[name].append(Measurement(name, unit, **kwargs))

    def get_measurement(self, name: str, kernel: str = None) -> Optional[Measurement]:
        """
        Returns the measurement given the name and optional the kernel name

        :param name: The name of the measurement
        :type name: str
        :param kernel: The kernel name, if needed to distinguish mutiple measurment, defaults to None
        :type kernel: str, optional
        :return: Found measurement or None if there is none
        :rtype: Optional[Measurement]
        """
        if name in self.measurements:
            for measurement in self.measurements[name]:
                if kernel is None or measurement.kernel_name == kernel:
                    return measurement
        return None

    def add_value(self, name: str, value: Number):
        """
        Adds a value to a measurement identified by the name. Adds if to the first one with the same name

        :param name: The name of the measurement
        :type name: str
        :param value: The value to add
        :type value: Number
        """
        self.measurements[name][0].add_value(value)

    @staticmethod
    def to_json(measurement: 'ProgramMeasurement') -> Dict:
        msr_dict = {}
        for measurements in measurement.measurements.values():
            msr_dict[measurements[0].name] = []
            for msr in measurements:
                msr_dict[msr.name].append(Measurement.to_json(msr))
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

    def __eq__(self, other: 'ProgramMeasurement'):
        return \
            self.measurements == other.measurements and \
            self.program == other.program and \
            self.parameters == other.parameters


class MeasurementRun:
    """
    Represents a measurement run which consits of a set of programs which are run with their data, the description of
    the run, the time it was run and the git hash.
    """
    description: str
    data: List[ProgramMeasurement]
    git_hash: str
    date: datetime
    node: str

    def __init__(self,
                 description: str,
                 data: List[ProgramMeasurement] = None,
                 git_hash: str = '',
                 date: datetime = datetime.now(),
                 node: str = ''):
        """
        Constructs the class

        :param description: The description of this measurement run
        :type description: str
        :param data: The data itself as a list of ProgramMeasurements, defaults to None
        :type data: List[ProgramMeasurement], optional
        :param git_hash: The short git hash, if not given, will be read automatically, defaults to ''
        :type git_hash: str, optional
        :param date: The time the measurements were run, defaults to datetime.now()
        :type date: datetime, optional
        :param node: The name of the node the measurements were take on, if not set, will get it automatically
        :type str:
        """
        self.description = description
        self.data = data if data is not None else []
        self.git_hash = git_hash
        if self.git_hash == '':
            hash_output = run(['git', 'rev-parse', '--short', 'HEAD'],
                              cwd=path.split(path.dirname(__file__))[0],
                              capture_output=True)
            self.git_hash = hash_output.stdout.decode('UTF-8')
        self.date = date
        self.node = node
        if self.node == '':
            node_output = run(['uname', '-a'], capture_output=True)
            self.node = node_output.stdout.decode('UTF-8').split(' ')[1].split('.')[0]

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
                'date': run.date.isoformat(),
                'node': run.node
                }

    @staticmethod
    def from_json(dict: Dict) -> 'MeasurementRun':
        if '__MeasurementRun__' in dict:
            return MeasurementRun(dict['description'], data=dict['data'], git_hash=dict['git_hash'],
                                  date=datetime.fromisoformat(dict['date']), node=dict['node'])
        if '__ProgramMeasurement__' in dict:
            return ProgramMeasurement(dict['program'], dict['parameters'],
                                      measurements=dict['measurements'])
        elif '__Measurement__' in dict:
            return Measurement.from_json(dict)
        else:
            return dict

    def __eq__(self, other: 'MeasurementRun') -> bool:
        return \
            self.description == other.description and \
            self.data == other.data and \
            self.git_hash == other.git_hash and \
            self.date == other.date and \
            self.node == other.node
