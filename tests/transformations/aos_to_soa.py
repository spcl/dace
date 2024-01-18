import dace
import numpy as np


def test_distance():

    N = dace.symbol('N')
    float32 = dace.data.Scalar(dace.float32)
    Point = dace.data.Structure(dict(x=float32, y=float32), name='Point')
    PointView = dace.data.StructureView([('x', float32), ('y', float32)], name='Point', transient=True)

    sdfg = dace.SDFG('distance')
    sdfg.add_datadesc('A', Point[N])
    sdfg.add_datadesc('B', dace.float32[N])
    sdfg.add_datadesc('vPoint', PointView)

    state = sdfg.add_state()
    A = state.add_access('A')
    B = state.add_access('B')
    me, mx = state.add_map('distance', dict(i='0:N'))
    vA = state.add_access('vPoint')
    t = state.add_tasklet('distance', {'__x', '__y'}, {'__out'}, '__out = sqrt((__x * __x) + (__y * __y))')
    state.add_memlet_path(A, me, vA, dst_conn='views', memlet=dace.Memlet(data='A', subset='i'))
    state.add_edge(vA, None, t, '__x', memlet=dace.Memlet.from_array('vPoint.x', Point.members['x']))
    state.add_edge(vA, None, t, '__y', memlet=dace.Memlet.from_array('vPoint.y', Point.members['y']))
    state.add_memlet_path(t, mx, B, memlet=dace.Memlet(data='B', subset='i'), src_conn='__out')

    sdfg.view()

    struct_array = sdfg.arrays['A']
    shape = struct_array.shape
    new_names = dict()
    for name, desc in struct_array.stype.members.items():
        nname, _ = sdfg.add_array(name, shape, desc.dtype,
                       storage=struct_array.storage,
                       transient=struct_array.transient,
                       find_new_name=True)
        new_names[name] = nname
    
    for anode in state.data_nodes():
        if anode.data == 'A':
            for edge in state.out_edges(anode):
                memlet_path = state.memlet_path(edge)
                dst = memlet_path[-1].dst
                if isinstance(dst, dace.nodes.AccessNode) and isinstance(dst.desc(sdfg), dace.data.StructureView):
                    for name, desc in struct_array.stype.members.items():
                        nname = new_names[name]
                        ndesc = sdfg.arrays[nname]
                        shape = memlet_path[-1].data.subset.size_exact()
                        nview, _ = sdfg.add_view(nname, shape, ndesc.dtype, storage=ndesc.storage, strides=ndesc.strides, find_new_name=True)
                        nsrc = state.add_access(nname)
                        ndst = state.add_access(nview)
                        state.add_memlet_path(nsrc, me, ndst, dst_conn='views', memlet=dace.Memlet(data=nname, subset='i'))
                        for e2 in state.out_edges(dst):
                            dataname = e2.data.data
                            member = dataname.split('.')[1]
                            if member == name:
                                e2.data = dace.Memlet.from_array(nview, ndesc)
                                state.add_edge(ndst, None, e2.dst, e2.dst_conn, memlet=e2.data)
                                state.remove_edge(e2)
                    state.remove_node(dst)
                    for e2 in memlet_path:
                        if e2 in state.edges():
                            state.remove_edge(e2)
            state.remove_node(anode)
    
    sdfg.view()




if __name__ == "__main__":
    test_distance()
