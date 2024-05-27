# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from os.path import basename
from dace.sdfg.state import LoopRegion
from tests.loop_region_code_location_file import *

def test_for_body_range():
    sdfg = function_for.to_sdfg()
    cfgs = sdfg.reset_cfg_list()
    assert len(cfgs) == 2
    loop_region : LoopRegion = cfgs[1]
    assert loop_region.body_debuginfo.start_line == 11
    assert loop_region.body_debuginfo.end_line == 12
    assert basename(loop_region.body_debuginfo.filename) == "loop_region_code_location_file.py"

def test_while_body_range():
    sdfg = function_while.to_sdfg()
    cfgs = sdfg.reset_cfg_list()
    assert len(cfgs) == 2
    loop_region : LoopRegion = cfgs[1]
    assert loop_region.body_debuginfo.start_line == 20
    assert loop_region.body_debuginfo.end_line == 22
    assert basename(loop_region.body_debuginfo.filename) == "loop_region_code_location_file.py"

def test_while_condition_range():
    sdfg = function_while_multiline_condition.to_sdfg()
    cfgs = sdfg.reset_cfg_list()
    assert len(cfgs) == 2
    loop_region : LoopRegion = cfgs[1]
    assert loop_region.condition_debuginfo.start_line == 31
    assert loop_region.condition_debuginfo.end_line == 33
    assert basename(loop_region.body_debuginfo.filename) == "loop_region_code_location_file.py"

if __name__ == '__main__':
    test_for_body_range()
    test_while_body_range()
    test_while_condition_range()
