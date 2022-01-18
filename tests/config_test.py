# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.config import set_temporary, Config


def test_set_temporary():
    path = ["compiler", "build_type"]
    current_value = Config.get(*path)
    with set_temporary(*path, value="I'm not a build type"):
        assert Config.get(*path) == "I'm not a build type"
    assert Config.get(*path) == current_value


if __name__ == '__main__':
    test_set_temporary()
