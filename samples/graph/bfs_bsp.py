# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import argparse
import networkx as nx
import scipy
import scipy.io
import numpy as np

import dace as dp
from collections import OrderedDict

# Push-based, vertex-centric, data-driven, bulk-synchronous Breadth First Search

V = dp.symbol('V')
E = dp.symbol('E')
vtype = dp.uint32
INFINITY = np.iinfo(vtype.type).max

storage = dp.StorageType.Default

sdfg = dp.SDFG('bfs_bsp')
sdfg.add_symbol('srcnode', vtype)

istate = sdfg.add_state('init')
dstate = sdfg.add_state('doit')
estate = sdfg.add_state('doit2')
rst0 = sdfg.add_state('reset0')
rst1 = sdfg.add_state('reset1')

sdfg.add_edge(istate, rst1, dp.sdfg.InterstateEdge(assignments={'d': '1'}))
sdfg.add_edge(rst1, dstate, dp.sdfg.InterstateEdge())
sdfg.add_edge(dstate, rst0, dp.sdfg.InterstateEdge('fsz1 > 0', {'d': 'd+1'}))
sdfg.add_edge(rst0, estate, dp.sdfg.InterstateEdge())
sdfg.add_edge(estate, rst1, dp.sdfg.InterstateEdge('fsz0 > 0', {'d': 'd+1'}))

######################
# Initialization state

frontiersize1 = istate.add_scalar('fsz0', dtype=dp.uint32, storage=storage, transient=True)
frontier = istate.add_transient('frontier', [V], dtype=vtype, storage=storage)
depth = istate.add_array('depth', [V], dtype=vtype, storage=storage)

itask = istate.add_tasklet('init_depth', set(), {'d', 'f', 'ofsz'}, """
d = 0
f = srcnode
ofsz = 1
""")
istate.add_edge(itask, 'd', depth, None, dp.Memlet.simple(depth, 'srcnode'))
istate.add_edge(itask, 'f', frontier, None, dp.Memlet.simple(frontier, '0'))
istate.add_edge(itask, 'ofsz', frontiersize1, None, dp.Memlet.from_array('fsz0', frontiersize1.desc(sdfg)))

######################
# Reset states


def create_reset_state(state, frontier_index):
    frontiersize = state.add_scalar('fsz%d' % frontier_index, dtype=dp.uint32, storage=storage, transient=True)
    rtask = state.add_tasklet('reset_%d' % frontier_index, set(), {'ofsz'}, 'ofsz = 0')
    state.add_edge(rtask, 'ofsz', frontiersize, None,
                   dp.Memlet.from_array('fsz%d' % frontier_index, frontiersize.desc(sdfg)))


create_reset_state(rst0, 0)
create_reset_state(rst1, 1)

######################
# Computation state


def create_computation_state(dstate, in_frontier, out_frontier, frontier_index):
    frontiersize = dstate.add_scalar('fsz%d' % frontier_index, dtype=dp.uint32, storage=storage, transient=True)
    depth = dstate.add_array('depth', [V], dtype=vtype, storage=storage)
    out_frontiersize = dstate.add_scalar('fsz%d' % (1 - frontier_index),
                                         dtype=dp.uint32,
                                         storage=storage,
                                         transient=True)
    out_depth = dstate.add_array('depth', [V], dtype=vtype, storage=storage)
    G_row = dstate.add_array('G_row', [V + 1], dtype=vtype, storage=storage)
    G_col = dstate.add_array('G_col', [E], dtype=vtype, storage=storage)
    out_frontier_stream = dstate.add_stream('out_stream%d' % frontier_index, vtype, 1, transient=True)

    me, mx = dstate.add_map('frontiermap', dict(f='0:fsz%d' % frontier_index))
    me.in_connectors = {'IN_F': None, 'IN_R': None, 'IN_C': None, 'IN_D': None, 'fsz%d' % frontier_index: None}
    me.out_connectors = {'OUT_F': None, 'OUT_R': None, 'OUT_C': None, 'OUT_D': None}
    mx.in_connectors = {'IN_F': None, 'IN_FSZ': None, 'IN_D': None}
    mx.out_connectors = {'OUT_F': None, 'OUT_FSZ': None, 'OUT_D': None}

    # Map inputs
    dstate.add_edge(frontiersize, None, me, 'fsz%d' % frontier_index,
                    dp.Memlet.from_array('fsz%d' % frontier_index, frontiersize.desc(sdfg)))
    dstate.add_edge(in_frontier, None, me, 'IN_F', dp.Memlet.from_array(in_frontier.data, in_frontier.desc(sdfg)))
    dstate.add_edge(G_row, None, me, 'IN_R', dp.Memlet.from_array('G_row', G_row.desc(sdfg)))
    dstate.add_edge(G_col, None, me, 'IN_C', dp.Memlet.from_array('G_col', G_col.desc(sdfg)))
    dstate.add_edge(depth, None, me, 'IN_D', dp.Memlet.from_array('depth', depth.desc(sdfg)))

    # Map contents
    rowb = dstate.add_scalar('rowb%d' % frontier_index, vtype, storage, transient=True)
    rowe = dstate.add_scalar('rowe%d' % frontier_index, vtype, storage, transient=True)
    indirection = dstate.add_tasklet('indirection', {'in_f', 'in_row'}, {'ob', 'oe'},
                                     'ob = in_row[in_f]; oe = in_row[in_f + 1]')
    dstate.add_edge(me, 'OUT_F', indirection, 'in_f', dp.Memlet.simple(in_frontier, 'f'))
    dstate.add_edge(me, 'OUT_R', indirection, 'in_row',
                    dp.Memlet.simple('G_row', dp.subsets.Range.from_array(G_row.desc(sdfg)), num_accesses=2))
    dstate.add_edge(indirection, 'ob', rowb, None, dp.Memlet.from_array('rowb%d' % frontier_index, rowb.desc(sdfg)))
    dstate.add_edge(indirection, 'oe', rowe, None, dp.Memlet.from_array('rowe%d' % frontier_index, rowe.desc(sdfg)))

    # Internal neighbor map inputs
    nme, nmx = dstate.add_map('neighbormap', dict(nid='rowb{f}:rowe{f}'.format(f=frontier_index)))
    nme.in_connectors = {'IN_C': None, 'IN_D': None, 'rowb%d' % frontier_index: None, 'rowe%d' % frontier_index: None}
    nme.out_connectors = {'OUT_C': None, 'OUT_D': None}
    nmx.in_connectors = {'IN_D': None, 'IN_FSZ': None, 'IN_F': None}
    nmx.out_connectors = {'OUT_D': None, 'OUT_FSZ': None, 'OUT_F': None}

    dstate.add_edge(rowb, None, nme, 'rowb%d' % frontier_index,
                    dp.Memlet.from_array('rowb%d' % frontier_index, rowb.desc(sdfg)))
    dstate.add_edge(rowe, None, nme, 'rowe%d' % frontier_index,
                    dp.Memlet.from_array('rowe%d' % frontier_index, rowe.desc(sdfg)))

    dstate.add_edge(me, 'OUT_C', nme, 'IN_C', dp.Memlet.from_array('G_col', G_col.desc(sdfg)))
    dstate.add_edge(me, 'OUT_D', nme, 'IN_D', dp.Memlet.from_array('depth', depth.desc(sdfg)))

    # Internal neighbor map contents
    ctask = dstate.add_tasklet(
        'update_and_push', {'in_col', 'in_depth'}, {'out_depth', 'out_fsz', 'out_frontier'}, """
node = in_col[nid]
dep = in_depth[node]
if d < dep:
    out_depth[node] = d
    out_frontier = node
    out_fsz = 1
""")

    # Internal inputs
    dstate.add_edge(nme, 'OUT_C', ctask, 'in_col',
                    dp.Memlet.simple('G_col', dp.subsets.Range.from_array(G_col.desc(sdfg)), num_accesses=-1))
    dstate.add_edge(nme, 'OUT_D', ctask, 'in_depth',
                    dp.Memlet.simple('depth', dp.subsets.Range.from_array(depth.desc(sdfg)), num_accesses=-1))

    # Internal outputs
    dstate.add_edge(ctask, 'out_depth', nmx, 'IN_D',
                    dp.Memlet.simple('depth', dp.subsets.Range.from_array(depth.desc(sdfg)), num_accesses=-1))
    dstate.add_edge(
        ctask, 'out_fsz', nmx, 'IN_FSZ',
        dp.Memlet.simple(out_frontiersize, dp.subsets.Indices([0]), num_accesses=-1, wcr_str='lambda a,b: a+b'))
    dstate.add_edge(
        ctask, 'out_frontier', nmx, 'IN_F',
        dp.Memlet.simple(out_frontier_stream,
                         dp.subsets.Range.from_array(out_frontier_stream.desc(sdfg)),
                         num_accesses=-1))

    # Internal neighbor map outputs
    dstate.add_edge(nmx, 'OUT_D', mx, 'IN_D',
                    dp.Memlet.simple(depth, dp.subsets.Range.from_array(depth.desc(sdfg)), num_accesses=-1))
    dstate.add_edge(
        nmx, 'OUT_FSZ', mx, 'IN_FSZ',
        dp.Memlet.simple(out_frontiersize, dp.subsets.Indices([0]), num_accesses=-1, wcr_str='lambda a,b: a+b'))
    dstate.add_edge(nmx, 'OUT_F', mx, 'IN_F',
                    dp.Memlet.from_array(out_frontier_stream.data, out_frontier_stream.desc(sdfg)))

    # Map outputs
    dstate.add_edge(mx, 'OUT_D', out_depth, None,
                    dp.Memlet.simple(depth, dp.subsets.Range.from_array(depth.desc(sdfg)), num_accesses=-1))
    dstate.add_edge(
        mx, 'OUT_FSZ', out_frontiersize, None,
        dp.Memlet.simple(out_frontiersize, dp.subsets.Indices([0]), num_accesses=-1, wcr_str='lambda a,b: a+b'))
    dstate.add_edge(mx, 'OUT_F', out_frontier_stream, None,
                    dp.Memlet.from_array(out_frontier_stream.data, out_frontier_stream.desc(sdfg)))

    # Stream->Array interface
    dstate.add_nedge(out_frontier_stream, out_frontier, dp.Memlet.from_array(out_frontier.data,
                                                                             out_frontier.desc(sdfg)))


frontier = dstate.add_transient('frontier', [V], dtype=vtype, storage=storage)
frontier2 = dstate.add_transient('frontier2', [V], dtype=vtype, storage=storage)
create_computation_state(dstate, frontier, frontier2, 0)

frontier = estate.add_transient('frontier', [V], dtype=vtype, storage=storage)
frontier2 = estate.add_transient('frontier2', [V], dtype=vtype, storage=storage)
create_computation_state(estate, frontier2, frontier, 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("edges", type=int, nargs="?", default=64)
    parser.add_argument("vertices", type=int, nargs="?", default=64)
    parser.add_argument("-seed", type=int, nargs="?", default=None)
    parser.add_argument("-source", type=int, nargs="?", default=0)
    parser.add_argument("-loadmtx", type=str, nargs="?", default=None)
    parser.add_argument("-loadgr", type=str, nargs="?", default=None)
    parser.add_argument("-outfile", type=str, nargs="?", default=None)
    args = vars(parser.parse_args())

    E.set(args['edges'])
    V.set(args['vertices'])
    srcnode = args['source']
    outfile = args['outfile']

    regression = False
    if args['loadgr'] is not None:
        from support import readgr
        V, E, G_row, G_col = readgr.read_grfile(args['loadgr'])
    elif args['loadmtx'] is not None:
        M = scipy.io.mmread(args['loadmtx']).tocsr()
        E.set(M.nnz)
        V.set(M.shape[0])
        G_row = dp.ndarray([V + 1], dtype=vtype)
        G_col = dp.ndarray([E], dtype=vtype)
        G_row[:] = M.indptr
        G_col[:] = M.indices
    else:
        # Generate a random graph
        graph = nx.gnm_random_graph(V.get(), E.get(), seed=args['seed'])
        E.set(E.get() * 2)

        # Extract adjacency matrix
        M = nx.to_scipy_sparse_matrix(graph, dtype=vtype.type).tocsr()
        assert M.nnz == E.get()

        G_row = dp.ndarray([V + 1], dtype=vtype)
        G_col = dp.ndarray([E], dtype=vtype)
        G_row[:] = M.indptr
        G_col[:] = M.indices

        # Regression
        result = nx.shortest_path(graph, source=srcnode)
        result = [len(result[v]) - 1 if v in result else INFINITY for v in range(V.get())]
        regression = True

    print('Data loaded')
    print('Breadth-First Search E=%d, V=%d' % (E.get(), V.get()))

    # Allocate output arrays
    depth = dp.ndarray([V], vtype)
    depth[:] = vtype(INFINITY)

    sdfg(G_row=G_row, G_col=G_col, depth=depth, srcnode=srcnode, E=E, V=V)

    if regression:
        print('Comparing results...')
        diff = np.linalg.norm(depth - result) / V.get()
        print("Difference:", diff)
        exit(1 if diff >= 1e-5 else 0)

    if args['outfile'] is not None:
        print('Saving results...')
        output = np.ndarray([V.get(), 2], vtype.type)
        output[:, 0] = np.arange(0, V.get())
        output[:, 1] = depth[:]
        np.savetxt(outfile, output, fmt='%d')
        print('Results written to', outfile)
    exit(0)
