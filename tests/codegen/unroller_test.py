# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import unittest


@dace.program
def Copy(output: dace.int32[5], input: dace.int32[5]):
    @dace.map
    def mytasklet(i: _[0:5]):
        inp << input[i]
        out >> output[i]

        out = inp


class UnrollerTest(unittest.TestCase):
    def test_unroller(self):
        sdfg = Copy.to_sdfg()

        # Transform map to unrolled
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.sdfg.nodes.MapEntry):
                    node.schedule = dace.ScheduleType.Unrolled

        input = np.ones([5], dtype=np.int32)
        output = np.zeros([5], dtype=np.int32)
        sdfg(output=output, input=input)

        self.assertTrue((output == input).all())


if __name__ == '__main__':
    unittest.main()
