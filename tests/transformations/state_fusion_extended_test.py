from dace import SDFG, InterstateEdge,Memlet
from dace import dtypes
from dace.transformation.interstate import StateFusionExtended


def test_extended_fusion():
    """
    Test the extended state fusion transformation.
    It should fuse the two states into one and add a dependency between the two uses of tmp.
    """
    sdfg=SDFG('extended_state_fusion_test')
    sdfg.add_array('A', [20, 20], dtypes.float64)
    sdfg.add_array('B', [20, 20], dtypes.float64)
    sdfg.add_array('C', [20, 20], dtypes.float64)
    sdfg.add_array('D', [20, 20], dtypes.float64)
    sdfg.add_array('E', [20, 20], dtypes.float64)
    sdfg.add_array('F', [20, 20], dtypes.float64)

    sdfg.add_scalar('tmp', dtypes.float64)

    strt = sdfg.add_state("start")
    mid = sdfg.add_state("middle")

    sdfg.add_edge(strt, mid, InterstateEdge())

    acc_a = strt.add_read('A')
    acc_b = strt.add_read('B')
    acc_c = strt.add_write('C')
    acc_tmp = strt.add_access('tmp')

    acc2_d = mid.add_read('D')
    acc2_e = mid.add_read('E')
    acc2_f = mid.add_write('F')
    acc2_tmp = mid.add_access('tmp')

    t1 = strt.add_tasklet('t1', {'a', 'b'}, {
        'c',
    }, 'c[1,1] = a[1,1] + b[1,1]')
    t2 = strt.add_tasklet('t2', {}, {
        'tmpa',
    }, 'tmpa=4')

    t3 = mid.add_tasklet('t3', {'d', 'e'}, {
        'f',
    }, 'f[1,1] = e[1,1] + d[1,1]')
    t4 = mid.add_tasklet('t4', {}, {
        'tmpa',
    }, 'tmpa=7')

    strt.add_edge(acc_a, None, t1, 'a', Memlet.simple('A', '1,1'))
    strt.add_edge(acc_b, None, t1, 'b', Memlet.simple('B', '1,1'))
    strt.add_edge(t1, 'c', acc_c, None, Memlet.simple('C', '1,1'))
    strt.add_edge(t2, 'tmpa', acc_tmp, None, Memlet.simple('tmp', '0'))

    mid.add_edge(acc2_d, None, t3, 'd', Memlet.simple('D', '1,1'))
    mid.add_edge(acc2_e, None, t3, 'e', Memlet.simple('E', '1,1'))
    mid.add_edge(t3, 'f', acc2_f, None, Memlet.simple('F', '1,1'))
    mid.add_edge(t4, 'tmpa', acc2_tmp, None, Memlet.simple('tmp', '0'))
    sdfg.simplify()
    sdfg.apply_transformations_repeated(StateFusionExtended)
    assert sdfg.number_of_nodes()==1


if __name__ == '__main__':
    test_extended_fusion()
