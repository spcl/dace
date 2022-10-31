import click
import dace
import numpy as np
from dace.transformation.dataflow import StreamingMemory, Vectorization
from dace.transformation.interstate import FPGATransformState
from dace.transformation.subgraph import TemporalVectorization

N = dace.symbol('N')

def vadd(x: dace.float32[N], y: dace.float32[N]):
    return x + y

@click.command()
@click.argument("size_n", type=int)
@click.argument("veclen", type=int)
@click.option("--double_pumped/--no-double_pumped",
              default=False,
              help="Whether to double pump the compute kernel")
def cli(size_n, veclen, double_pumped):
    # Generate the test data and expected results
    N.set(size_n)
    x = np.random.rand(N.get()).astype(np.float32)
    y = np.random.rand(N.get()).astype(np.float32)
    result = np.zeros(N.get(), dtype=np.float32)
    expected = x + y
    
    # Generate the initial SDFG
    sdfg = dace.program(vadd).to_sdfg()
    
    # Remove underscores as Xilinx does not like them
    for dn in sdfg.nodes()[0].data_nodes():
        if '__' in dn.data:
            new_name = dn.data.replace('__', '')
            sdfg.replace(dn.data, new_name)

    # Apply vectorization transformation
    ambles = size_n % veclen != 0
    map_entry = [n for n, _ in sdfg.all_nodes_recursive() 
        if isinstance(n, dace.nodes.MapEntry)][0]
    applied = sdfg.apply_transformations(Vectorization, {
        'vector_len': veclen,
        'preamble': ambles, 'postamble': ambles,
        'propagate_parent': True, 'strided_map': False,
        'map_entry': map_entry
    })
    assert(applied == 1)
    
    # Transform to an FPGA implementation
    applied = sdfg.apply_transformations(FPGATransformState)
    assert(applied == 1)
    
    # Apply streaming memory transformation
    applied = sdfg.apply_transformations_repeated(StreamingMemory, {
        'storage': dace.StorageType.FPGA_Local,
        'buffer_size': 1
    })
    assert (applied == 3)
    
    # Apply temporal vectorization transformation
    sgs = dace.sdfg.concurrent_subgraphs(sdfg.states()[0])
    sf = TemporalVectorization()
    cba = [TemporalVectorization.can_be_applied(sf, sdfg, sg) for sg in sgs]
    assert (sum(cba) == 1)
    [TemporalVectorization.apply_to(sdfg, sg) for i, sg in enumerate(sgs) if cba[i]]
    
    sdfg.specialize({'N': N.get()})
    sdfg(x=x, y=y, __return=result)

    allclose = np.allclose(expected, result)
    assert(allclose)

if __name__ == '__main__':
    cli()
