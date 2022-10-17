import sys
sys.path.insert(0, '/home/carljohnsen/git/dace')

import click
import dace
import numpy as np
from dace.transformation.dataflow import StreamingMemory, Vectorization
from dace.transformation.interstate import FPGATransformState
from dace.transformation.subgraph import TemporalVectorization

N = dace.symbol('N')

@dace.program
def vadd(x: dace.float32[N], y: dace.float32[N]):
    return x + y

@click.command()
@click.argument("size_n", type=int)
@click.argument("veclen", type=int)
@click.option("--double_pumped/--no-double_pumped",
              default=False,
              help="Whether to double pump the compute kernel")
def cli(size_n, veclen, double_pumped):
    N.set(size_n)

    A = np.random.rand(N.get()).astype(np.float32)
    B = np.random.rand(N.get()).astype(np.float32)
    expected = A + B
    
    sdfg = vadd.to_sdfg()
    ambles = size_n%veclen!=0
    me = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)][0]
    #Vectorization.apply_to(sdfg, dict(vector_len=veclen, preamble=ambles, postamble=ambles))
    #sdfg.apply_transformations_repeated(Vectorization, dict(vector_len=veclen))
    applied = sdfg.apply_transformations(Vectorization, dict(vector_len=veclen, preamble=ambles, postamble=ambles, map_entry=me))#, propagate_parent=True))
    print (applied)
    sdfg.save('aoeu.sdfg')

    C = vadd(x=A, y=B)

    print (np.sum(np.abs(expected-C)))

if __name__ == '__main__':
    cli()
