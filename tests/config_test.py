# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.config import Config, set_temporary, temporary_config


def test_set_temporary():
    path = ["compiler", "build_type"]
    current_value = Config.get(*path)
    with set_temporary(*path, value="I'm not a build type"):
        assert Config.get(*path) == "I'm not a build type"
    assert Config.get(*path) == current_value


def test_temporary_config():
    path = ["compiler", "build_type"]
    current_value = Config.get(*path)
    with temporary_config():
        Config.set(*path, value="I'm not a build type")
        assert Config.get(*path) == "I'm not a build type"
    assert Config.get(*path) == current_value


def test_temporary_config_exception():
    path = ["compiler", "build_type"]
    initial_value = Config.get(*path)
    new_value = initial_value + "_non_existing"
    assert initial_value != new_value

    try:
        with temporary_config():
            Config.set(*path, value=new_value)
            assert Config.get(*path) == new_value
            raise ValueError()
    except ValueError:
        assert Config.get(*path) == initial_value

    except:
        # Unknown exception type was raised.
        raise

    else:
        raise RuntimeError("No exception was raised.")


def test_set_temporary_exception():
    path = ["compiler", "build_type"]
    initial_value = Config.get(*path)
    new_value = initial_value + "_non_existing"
    assert initial_value != new_value

    try:
        with set_temporary(*path, value=new_value):
            assert Config.get(*path) == new_value
            raise ValueError()
    except ValueError:
        assert Config.get(*path) == initial_value

    except:
        # Unknown exception type was raised.
        raise

    else:
        raise RuntimeError("No exception was raised.")


if __name__ == '__main__':
    test_set_temporary()
    test_temporary_config()
    test_temporary_config_exception()
    test_set_temporary_exception()
