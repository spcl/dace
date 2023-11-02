# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import unittest
import dace


# Try to detect invalid names in SDFG
class NameValidationTests(unittest.TestCase):
    # SDFG label
    def test_sdfg_name1(self):
        try:
            sdfg = dace.SDFG(' ')
            sdfg.validate()
            self.fail('Failed to detect invalid SDFG')
        except dace.sdfg.InvalidSDFGError as ex:
            print('Exception caught:', ex)

    def test_sdfg_name2(self):
        try:
            sdfg = dace.SDFG('3sat')
            sdfg.validate()
            self.fail('Failed to detect invalid SDFG')
        except dace.sdfg.InvalidSDFGError as ex:
            print('Exception caught:', ex)

    # State
    def test_state_duplication(self):
        try:
            sdfg = dace.SDFG('ok')
            s1 = sdfg.add_state('also_ok')
            s2 = sdfg.add_state('also_ok')
            s2.label = 'also_ok'
            sdfg.add_edge(s1, s2, dace.InterstateEdge())
            sdfg.validate()
            self.fail('Failed to detect duplicate state')
        except dace.sdfg.InvalidSDFGError as ex:
            print('Exception caught:', ex)

    def test_state_name1(self):
        try:
            sdfg = dace.SDFG('ok')
            sdfg.add_state('not ok')
            sdfg.validate()
            self.fail('Failed to detect invalid state')
        except dace.sdfg.InvalidSDFGError as ex:
            print('Exception caught:', ex)

    def test_state_name2(self):
        try:
            sdfg = dace.SDFG('ok')
            sdfg.add_state('$5')
            sdfg.validate()
            self.fail('Failed to detect invalid state')
        except dace.sdfg.InvalidSDFGError as ex:
            print('Exception caught:', ex)

    # Array
    def test_array(self):
        try:
            sdfg = dace.SDFG('ok')
            state = sdfg.add_state('also_ok')
            _8 = state.add_array('8', [1], dace.float32)
            t = state.add_tasklet('tasklet', {'a'}, {}, 'print(a)')
            state.add_edge(_8, None, t, 'a', dace.Memlet.from_array(_8.data, _8.desc(sdfg)))
            sdfg.validate()
            self.fail('Failed to detect invalid array name')
        except (dace.sdfg.InvalidSDFGError, NameError) as ex:
            print('Exception caught:', ex)

    # Tasklet
    def test_tasklet(self):
        try:
            sdfg = dace.SDFG('ok')
            state = sdfg.add_state('also_ok')
            A = state.add_array('A', [1], dace.float32)
            B = state.add_array('B', [1], dace.float32)
            t = state.add_tasklet(' tasklet', {'a'}, {'b'}, 'b = a')
            state.add_edge(A, None, t, 'a', dace.Memlet.from_array(A.data, A.desc(sdfg)))
            state.add_edge(t, 'b', B, None, dace.Memlet.from_array(B.data, B.desc(sdfg)))
            sdfg.validate()
            self.fail('Failed to detect invalid tasklet name')
        except dace.sdfg.InvalidSDFGNodeError as ex:
            print('Exception caught:', ex)

    # Connector
    def test_connector(self):
        try:
            sdfg = dace.SDFG('ok')
            state = sdfg.add_state('also_ok')
            A = state.add_array('A', [1], dace.float32)
            B = state.add_array('B', [1], dace.float32)
            t = state.add_tasklet('tasklet', {'$a'}, {' b'}, '')
            state.add_edge(A, None, t, '$a', dace.Memlet.from_array(A.data, A.desc(sdfg)))
            state.add_edge(t, ' b', B, None, dace.Memlet.from_array(B.data, B.desc(sdfg)))
            sdfg.validate()
            self.fail('Failed to detect invalid connectors')
        except dace.sdfg.InvalidSDFGError as ex:
            print('Exception caught:', ex)

    # Interstate edge
    def test_interstate_edge(self):
        try:
            sdfg = dace.SDFG('ok')
            state = sdfg.add_state('also_ok', is_start_state=True)
            A = state.add_array('A', [1], dace.float32)
            B = state.add_array('B', [1], dace.float32)
            t = state.add_tasklet('tasklet', {'a'}, {'b'}, 'b = a')
            state.add_edge(A, None, t, 'a', dace.Memlet.from_array(A.data, A.desc(sdfg)))
            state.add_edge(t, 'b', B, None, dace.Memlet.from_array(B.data, B.desc(sdfg)))
            sdfg.add_edge(state, state, dace.InterstateEdge(assignments={'%5': '1'}))
            sdfg.validate()
            self.fail('Failed to detect invalid interstate edge')
        except dace.sdfg.InvalidSDFGInterstateEdgeError as ex:
            print('Exception caught:', ex)


if __name__ == '__main__':
    unittest.main()
