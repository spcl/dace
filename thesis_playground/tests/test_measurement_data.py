import json

from measurements.data import Measurement, ProgramMeasurement, MeasurementRun
from execute.parameters import ParametersProvider


class TestMeasurement:

    def test_stats(self):
        measurement = Measurement("Test", "s")
        measurement.add_value(1.0)
        measurement.add_value(2.0)
        measurement.add_value(3.0)
        measurement.add_value(3.0)
        measurement.add_value(3.0)
        assert measurement.min() == 1.0
        assert measurement.max() == 3.0
        assert measurement.average() == 2.4
        assert measurement.median() == 3.0

    def test_equality(self):
        measurement1 = Measurement("Test1", "s", kernel_name="foo")
        measurement2 = Measurement("Test1", "ms")
        measurement3 = Measurement("Test1", "s", kernel_name="foo")
        assert measurement1 == measurement3
        assert measurement1 != measurement2
        assert measurement3 != measurement2
        measurement1.add_value(2.0)
        assert measurement1 != measurement3
        measurement3.add_value(2.0)
        assert measurement1 == measurement3
        measurement1.add_value(2.5)
        measurement3.add_value(2.3)
        assert measurement1 != measurement3

    def test_serde(self):
        measurement = Measurement("Test", "s")
        measurement.add_value(1.0)
        measurement.add_value(2.0)
        measurement.add_value(3.0)
        parsed_measurement = json.loads(
                json.dumps(measurement, default=Measurement.to_json),
                object_hook=Measurement.from_json)
        assert measurement == parsed_measurement


class TestProgramMeasurement:

    def test_stats(self):
        program_measurement = ProgramMeasurement("test_program", {})
        program_measurement.add_measurement("m1", "s", kernel_name="kernel1")
        program_measurement.add_measurement("m1", "s", kernel_name="kernel2")
        program_measurement.add_measurement("m2", "s")
        program_measurement.get_measurement("m1", kernel="kernel1").add_value(2.0)
        program_measurement.get_measurement("m1", kernel="kernel2").add_value(3.0)
        program_measurement.get_measurement("m2").add_value(4.0)
        assert program_measurement.get_measurement("m1", kernel="kernel1").min() == 2.0
        assert program_measurement.get_measurement("m1", kernel="kernel2").min() == 3.0
        assert program_measurement.get_measurement("m2").min() == 4.0

    def test_equality(self):
        measurement1 = ProgramMeasurement("Test1", {'foo': 1})
        measurement2 = ProgramMeasurement("Test1", {})
        measurement3 = ProgramMeasurement("Test1", {'foo': 1})
        assert measurement1 == measurement3
        assert measurement1 != measurement2
        assert measurement3 != measurement2

    def test_serde(self):
        params = ParametersProvider('test_program', update={'foo': 2})
        measurement = ProgramMeasurement("Test", params)
        measurement.add_measurement("m1", "s", data=[1.0, 2.0], kernel_name="kernel1")
        parsed_measurement = json.loads(
                json.dumps(measurement, default=ProgramMeasurement.to_json),
                object_hook=ProgramMeasurement.from_json)
        print(measurement.parameters)
        print(parsed_measurement.parameters)
        assert measurement == parsed_measurement


class TestMeasurementRun:

    def test_equality(self):
        run1 = MeasurementRun("run 1")
        run2 = MeasurementRun("run 1", git_hash='foo')
        run3 = MeasurementRun("run 1")

        assert run1 == run3
        assert run1 != run2
        assert run3 != run2

        run1.add_program_data(ProgramMeasurement("m1", ParametersProvider('p1'),
                                                 measurements={'time': [
                                                    Measurement("time", "s", data=[1.0, 2.0], kernel_name="kernel1")
                                                    ]}))
        assert run1 != run3
        run3.add_program_data(ProgramMeasurement("m1", ParametersProvider('p1'),
                                                 measurements={'time': [
                                                    Measurement("time", "s", data=[1.0, 2.0], kernel_name="kernel1")
                                                    ]}))
        assert run1 == run3

    def test_serde(self):
        run = MeasurementRun("run 1")
        run.add_program_data(ProgramMeasurement("m1", ParametersProvider('p1'),
                                                measurements={'time': [
                                                    Measurement("time", "s", data=[1.0, 2.0], kernel_name="kernel1")
                                                    ]}))
        parsed_run = json.loads(
                json.dumps(run, default=MeasurementRun.to_json),
                object_hook=MeasurementRun.from_json)
        assert run == parsed_run
