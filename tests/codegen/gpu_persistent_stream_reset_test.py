# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A stream reset inside a persistent kernel has to be ordered against the pushes of that state.

``reset()`` rewinds the queue head for the whole grid and every block runs it. The barrier that
ends the previous state releases the blocks, it does not keep them in step, so a block still on
its way into the state can rewind the head after another block has already pushed. The pushed
slots are then handed out a second time, while the consumer's count -- a separate atomic -- still
counts every push, so it reads slots that were never written. On the first level of a search those
slots are whatever ``Malloc`` returned, and the index taken from them faults the kernel.

These assert on emitted code, so they need a GPU for neither compilation nor a run.
"""
import re

import dace
from dace.config import set_temporary
from dace.sdfg.graph import SubgraphView
from dace.transformation.subgraph import GPUPersistentKernel

N = dace.symbol('N')


def push_and_drain() -> dace.SDFG:
    """Persistent kernel that fills a stream in one state and reads the array behind it in the next."""
    sdfg = dace.SDFG('persistent_stream_reset')
    sdfg.add_array('src', [N], dace.int32)
    sdfg.add_array('dst', [N], dace.int32)
    sdfg.add_transient('buf', [N], dace.int32, may_alias=True)
    sdfg.add_stream('q', dace.int32, transient=True, buffer_size=N)

    produce = sdfg.add_state('produce', is_start_block=True)
    consume = sdfg.add_state('consume')
    sdfg.add_edge(produce, consume, dace.InterstateEdge())

    src = produce.add_read('src')
    queue = produce.add_access('q')
    buf = produce.add_write('buf')
    entry, exit_ = produce.add_map('select', dict(i='0:N'))
    keep = produce.add_tasklet('keep', {'v'}, {'out'}, 'if v > 0:\n  out = i')
    produce.add_memlet_path(src, entry, keep, dst_conn='v', memlet=dace.Memlet.simple('src', 'i'))
    produce.add_memlet_path(keep, exit_, queue, src_conn='out', memlet=dace.Memlet.simple('q', '0', num_accesses=-1))
    produce.add_memlet_path(queue, buf, memlet=dace.Memlet.simple('buf', '0'))

    kept = consume.add_read('buf')
    dst = consume.add_write('dst')
    entry, exit_ = consume.add_map('copy', dict(i='0:N'))
    copy = consume.add_tasklet('cp', {'v'}, {'out'}, 'out = v')
    consume.add_memlet_path(kept, entry, copy, dst_conn='v', memlet=dace.Memlet.simple('buf', 'i'))
    consume.add_memlet_path(copy, exit_, dst, src_conn='out', memlet=dace.Memlet.simple('dst', 'i'))

    sdfg.fill_scope_connectors()
    sdfg.validate()

    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    kernel_states = set(sdfg.nodes()) - {sdfg.start_state, sdfg.sink_nodes()[0]}
    transform = GPUPersistentKernel()
    transform.setup_match(SubgraphView(sdfg, kernel_states))
    transform.kernel_prefix = 'search'
    transform.apply(sdfg)
    sdfg.validate()
    return sdfg


def test_a_stream_reset_is_ordered_against_the_pushes_of_its_state():
    # A persistent kernel over a Stream is legacy-codegen territory: the experimental generator
    # cannot allocate Stream descriptors, so apply_gpu_transformations refuses the program and
    # ``push_and_drain`` would hand back an untransformed two-state SDFG with no kernel in it.
    with set_temporary('compiler', 'cuda', 'implementation', value='legacy'):
        code = '\n'.join(obj.clean_code for obj in push_and_drain().generate_code())

    resets = list(re.finditer(r'(\w+)\.reset\(\);', code))
    assert resets, 'no stream reset was emitted, so this test would pass vacuously'

    for reset in resets:
        after = code[reset.end():]
        push = after.find(f'{reset.group(1)}.push(')
        assert push != -1, f'stream {reset.group(1)} is reset but never pushed to'
        barrier = after.find('__gbar.Sync();')
        assert barrier != -1 and barrier < push, (
            f'stream {reset.group(1)} is pushed to without a grid barrier after its reset: a block '
            'that enters the state late rewinds the queue head under a block that already pushed')


if __name__ == '__main__':
    test_a_stream_reset_is_ordered_against_the_pushes_of_its_state()
