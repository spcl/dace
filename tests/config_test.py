# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.config import Config, set_temporary, temporary_config
import threading
import time


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


def test_config_isolation_multi_thread():

    thread_one_has_started_working = threading.Event()
    thread_two_has_started_working = threading.Event()
    thread_one_has_stoped_working = threading.Event()
    CONFIG_KEY = "compiler.cuda.backend"

    def thread1():
        # Because child processes do not inherent the configuration state of their
        #  parent they are set to the default.
        initial_value = Config.get(CONFIG_KEY)
        assert initial_value != "master"

        with temporary_config():
            Config.set(CONFIG_KEY, value="thread1")
            thread_one_has_started_working.set()
            assert Config.get(CONFIG_KEY) == "thread1"
            assert thread_two_has_started_working.wait(timeout=10)
            assert Config.get(CONFIG_KEY) == "thread1"

        assert initial_value == Config.get(CONFIG_KEY)
        thread_one_has_stoped_working.set()

    def thread2():
        assert thread_one_has_started_working.wait(timeout=10)
        initial_value = Config.get(CONFIG_KEY)
        assert initial_value not in ["master", "thread1"]

        with temporary_config():
            Config.set(CONFIG_KEY, value="thread2")
            assert Config.get(CONFIG_KEY) == "thread2"
            thread_two_has_started_working.set()

            # The restore event of the first thread should not have affected the
            #  configuration of the second thread.
            assert thread_one_has_stoped_working.wait(timeout=10)
            assert Config.get(CONFIG_KEY) == "thread2"

    with temporary_config():
        Config.set(CONFIG_KEY, value="master")
        threads = [threading.Thread(target=thread1), threading.Thread(target=thread2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert Config.get(CONFIG_KEY) == "master"


if __name__ == '__main__':
    test_set_temporary()
    test_temporary_config()
    test_temporary_config_exception()
    test_set_temporary_exception()
    test_config_isolation_multi_thread()
