import dace
from dace.memlet import Memlet
import numpy as np

from utils.general import use_cache
from execute.my_auto_opt import change_strides

NBLOCKS = dace.symbol('NBLOCKS')
KLEV = dace.symbol('KLEV')
# NBLOCKS = 7
# KLEV = 3


@dace.program
def kernel(inp: dace.float64[NBLOCKS, KLEV], out: dace.float64[NBLOCKS, KLEV]):
    for jn in dace.map[0:NBLOCKS]:
        for jk in range(KLEV):
            out[jn, jk] = inp[jn, jk]


def test_via_py_kernel():
    symbols = {'NBLOCKS': 7, 'KLEV': 3}
    inp = np.arange(symbols['NBLOCKS']*symbols['KLEV'], dtype=np.float64).reshape((symbols['NBLOCKS'], symbols['KLEV'])).copy()
    out = np.zeros_like(inp)
    print(inp)

    use_cache(dacecache_folder='kernel')
    sdfg = kernel.to_sdfg()
    sdfg = change_strides(sdfg, ('NBLOCKS', ), symbols)
    sdfg(inp=inp, out=out, **symbols)
    print(out)


def test_code_gen():
    sdfg = dace.SDFG('test_code_gen_graph')
    NBLOCKS = 7
    KLEV = 3
    inp = sdfg.add_array('inp', [NBLOCKS, KLEV], dace.float64, strides=[1, NBLOCKS])
    out = sdfg.add_array('out', [NBLOCKS, KLEV], dace.float64, strides=[1, NBLOCKS])
    state = sdfg.add_state('state', is_start_state=True)
    state.add_mapped_tasklet(
            name='map',
            map_ranges={'jn': f"0:{NBLOCKS}", 'jk': f"0:{KLEV}"},
            code='_in=_out',
            inputs={'_in': Memlet(data='inp', subset="jn, jk")},
            outputs={'_out': Memlet(data='out', subset="jk, jk")},
            external_edges=True
            )
    sdfg.save('graph.sdfg')
    for code_object in sdfg.generate_code():
        # print(code_object.name, code_object.language)
        if code_object.name == 'test_code_gen_graph' and code_object.language == 'cpp':
            print(code_object.clean_code)


def main():
    # test_via_py_kernel()
    test_code_gen()


if __name__ == '__main__':
    main()
