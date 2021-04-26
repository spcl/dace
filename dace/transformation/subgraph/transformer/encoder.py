import dace   
import numpy as np
import os

import dace.sdfg.nodes as nodes 
import dace.libraries as lib
import dace.dtypes as dtypes 

from dace.transformation.subgraph import ReduceExpansion, MultiExpansion, SubgraphFusion
from dace.transformation.subgraph.composite import CompositeFusion
from dace.transformation.estimator import GreedyEnumerator, ConnectedEnumerator, BruteForceEnumerator
from dace.transformation.auto_optimize import auto_optimize

from dace.transformation.interstate import InlineSDFG
from dace.sdfg.graph import SubgraphView

from dace.codegen import compiler
from dace.transformation.helpers import nest_state_subgraph


#import substation
#import substation.transformer as transformer


def expand_reductions(sdfg):
    # expands a raw encoder sdfg using 
    # transformations suitable for encoding 

    graph = sdfg.nodes()[0]

    print("PREPROCESSING...")
    print("EXPANDING NODES...")
    def process(sdfg, graph):
        for node in graph.nodes():

            if isinstance(node, nodes.NestedSDFG):
                # search in nested
                for g in node.sdfg.nodes():
                    process(node.sdfg, g)

            elif isinstance(node, lib.standard.nodes.Reduce):
                # expand reduction 
                #print(f"REDUCE: {node}")
                t = ReduceExpansion(sdfg.sdfg_id, sdfg.nodes().index(graph),
                                    {ReduceExpansion._reduce: graph.nodes().index(node)},
                                    0)
                t.apply(sdfg)

            elif isinstance(node, lib.blas.nodes.Gemm):
                #print("GEMM")
                pass

            elif isinstance(node, lib.blas.nodes.BatchedMatMul):
                #print("BMM")
                pass
            
            elif isinstance(node, lib.blas.nodes.MatMul):
                pass 
                '''
                print(f"MM: {node.label}")
                print(type(node))
                label = node.label
                handle_prior = next(iter(graph.in_edges(node))).src
                impl = node.expand(sdfg, graph)
                node_post = graph.out_edges(handle_prior)[0].dst
                impl = node_post.expand(sdfg, graph)
                nsdfg = graph.out_edges(handle_prior)[0].dst

                # case 1
                if label == 'einsum_gemm':
                    # case 1.1: Two maps
                    # apply vanilla NestOut to nested sdfg 
                    nsdfg.sdfg.apply_transformations(NestOut)                
                
                elif label == '_MatMult_':
                    pass #FORNOW
                '''

            elif isinstance(node, lib.blas.nodes.Transpose):
                #print("TRP")
                pass #FORNOW

            
            elif isinstance(node, dace.sdfg.nodes.LibraryNode):
                raise RuntimeError(f"Library Node {node} not covered")
    
    process(sdfg, graph)

def pre_transformations(sdfg, gpu = False):
    graph = sdfg.nodes()[0]
    print("Applying strict trafos...")
    sdfg.apply_strict_transformations()
 
    print("FORNOW: Register -> Default")
    for node in graph.nodes():
        if isinstance(node, nodes.NestedSDFG) and 'einsum' not in node.label:
            nsdfg = node.sdfg 
            ngraph = nsdfg.nodes()[0]
            print(f"Searching Nested SDFG {node}")
            for nnode in ngraph.nodes():
                if isinstance(nnode, lib.blas.nodes.MatMul):
                    '''
                    print(f"Found node {nnode}")
                    node_to_delete = ngraph.out_edges(nnode)[0].dst 
                    print("node_to_delete=", node_to_delete)
                    dace.sdfg.utils.change_edge_src(ngraph, node_to_delete, nnode)
                    ngraph.remove_node(node_to_delete)
                    ngraph.out_edges(nnode)[0].src_conn = '_c'
                    print("****", ngraph.out_edges(nnode)[0].data)
                    '''
                    '''
                    for iedge in ngraph.in_edges(nnode):
                        if isinstance(iedge.src, nodes.AccessNode):
                            if gpu:
                                nsdfg.data(iedge.src.data).storage = dtypes.StorageType.GPU_Global
                            else:
                                nsdfg.data(iedge.src.data).storage = dtypes.StorageType.Default
                    '''
            


    print("Validate...")
    sdfg.validate()
    print("Done.")
    

def get_encoder_cpu():
    # returns a raw encoder sdfg
    sdfg = dace.sdfg.SDFG.from_file('../../estimator/programs/encoder.sdfg')
    return sdfg  

def get_encoder_gpu():
    sdfg = dace.sdfg.SDFG.from_file('../../estimator/programs/encoder.sdfg')
    sdfg.apply_gpu_transformations()
    return sdfg 

def get_encoder_debug():
    sdfg = dace.sdfg.SDFG.from_file('../../estimator/programs/encoder_debug.sdfg')
    return sdfg 


def get_args():
    kwargs = {}
    B = 8; SM = 128; P = 64; H = 16; emb = 128; N=P*H
    kwargs.update({'B':np.int32(B), 'SM': np.int32(SM), 'N':np.int32(N), 'P':np.int32(P), 'H':np.int32(H), 'emb':np.int32(emb)})
    
    kwargs['attn_wk'] = np.random.rand(P,H,N).astype(np.float32)
    kwargs['x'] = np.random.rand(B,SM,N).astype(np.float32)
    kwargs['attn_wv'] = np.random.rand(P,H,N).astype(np.float32)
    kwargs['attn_scale'] = np.float32(1.0)
    kwargs['attn_wq'] = np.random.rand(P,H,N).astype(np.float32)
    kwargs['attn_wo'] = np.random.rand(P,H,N).astype(np.float32)
    kwargs['norm1_bias'] = np.random.rand(N).astype(np.float32)
    kwargs['norm1_scale'] = np.random.rand(N).astype(np.float32)
    kwargs['linear1_w'] = np.random.rand(emb,N).astype(np.float32)
    kwargs['linear1_b'] = np.random.rand(emb).astype(np.float32)
    kwargs['linear2_b'] = np.random.rand(N).astype(np.float32)
    kwargs['linear2_w'] = np.random.rand(N,emb).astype(np.float32)
    kwargs['norm2_bias'] = np.random.rand(N).astype(np.float32)
    kwargs['norm2_scale'] = np.random.rand(N).astype(np.float32)
    
    kwargs['attn_dropout'] = np.ones((B,SM,N), dtype = np.float32)
    kwargs['linear1_dropout'] = np.ones((B,SM,emb), dtype = np.float32)
    kwargs['ff_dropout'] = np.ones((B,SM,N), dtype = np.float32)
    '''
    q, k, v have shape (batch, sequence length, embedding).
    wq, wk, wv have shape (heads, proj size, embedding).
    wo has shape (embedding, embedding).
    in_b is a bias for each linear projection for each head,
    shape (3, heads, proj size).
    out_b is a bias for wo, shape (embedding,).
    scale is a scalar.
    mask has shape (sequence length, sequence length).
    '''

    return kwargs

def get_args_numpy(args):
    kwargs = {} 

    # fetch required arguments from args 
    num_heads = args['H']
    proj_size = args['P']
    embed_size = args['N']

    kwargs['attn_in_b'] = np.zeros((3, num_heads, proj_size), dtype = np.float32)
    kwargs['attn_out_b'] = np.zeros((embed_size,), dtype = np.float32)
    
    kwargs['attn_dropout_p'] = 0
    kwargs['linear1_dropout_p'] = 0
    kwargs['ff_dropout_p'] = 0
    
    kwargs['x'] = args['x']
    kwargs['attn_wk'] = np.transpose(args['attn_wk'], (1,0,2))
    kwargs['attn_wv'] = np.transpose(args['attn_wv'], (1,0,2))
    kwargs['attn_wq'] = np.transpose(args['attn_wq'], (1,0,2))
    # this configuration works lol
    kwargs['attn_wo'] = np.transpose(args['attn_wo'], (2,1,0))
    kwargs['attn_wo'] = np.reshape(kwargs['attn_wo'], (args['N'], args['N']))
    # try me 


    kwargs['attn_scale'] = args['attn_scale']
    kwargs['norm1_bias'] = args['norm1_bias']
    kwargs['norm1_scale'] = args['norm1_scale']
    kwargs['linear1_w'] = args['linear1_w']
    kwargs['linear1_b'] = args['linear1_b']
    kwargs['linear2_b'] = args['linear2_b']
    kwargs['linear2_w'] = args['linear2_w']
    kwargs['norm2_bias'] = args['norm2_bias']
    kwargs['norm2_scale'] = args['norm2_scale']

    
    
    return kwargs 

    '''
    def encoder(x, attn_wq, attn_wk, attn_wv, attn_wo,
            attn_in_b, attn_out_b, attn_scale,
            norm1_scale, norm1_bias, norm2_scale, norm2_bias,
            linear1_w, linear1_b, linear2_w, linear2_b,
            attn_dropout_p, linear1_dropout_p, ff_dropout_p,
            activation='gelu'):
    '''

def run_encoder(sdfg, kwargs):
    sdfg.save('input.sdfg')
    result = sdfg(**kwargs)
    return result 

def run_encoder_numpy(kwargs, return_all_args = False):
    result_vec = transformer.encoder(**kwargs)
    # normed2, ......
    print(np.linalg.norm(result_vec[0]))
    if return_all_args:
        return result_vec
    else:
        return result_vec[0]

    
def test_transformation():
    ''' tests pre - tranformations in gpu and cpu '''
    sdfg = get_encoder_cpu()
    kwargs = get_args()

    result1 = sdfg(**kwargs)

    expand_reductions(sdfg)
    pre_transformations(sdfg)
    result2 = sdfg(**kwargs)

    print(np.linalg.norm(result1))
    print(np.linalg.norm(result2))



def assign_reduce(sdfg, implementation):
    ''' assigns reduction implementation to all reduce nodes '''
    graph = sdfg.nodes()[0]
    for node in graph.nodes():
        if isinstance(node, lib.standard.nodes.Reduce):
            print(f"Assigned reduce node {node}")
            node.implementation = implementation




def run(run_baseline_cpu = True,
        run_baseline_gpu = True, 
        run_baseline_numpy = True,
        debug = False,

        run_expanded_cpu = True,
        run_expanded_gpu = True,

        run_cached = False):
        
    results = {}


    kwargs_sdfg = get_args()
    kwargs_numpy = get_args_numpy(kwargs_sdfg)

    sdfg_cpu = get_encoder_cpu()

    graph = sdfg_cpu.nodes()[0]
    
        
    sdfg_cpu.apply_transformations_repeated(ReduceExpansion)
    ## ok 
    sg = SubgraphView(graph, graph.nodes())
    print(MultiExpansion.can_be_applied(sdfg_cpu, sg))
    me = MultiExpansion(sg)
    me.apply(sdfg_cpu)
    sdfg_cpu.save('intermediate.sdfg')

    sdfg_gpu = get_encoder_gpu() 
    
    
    if run_baseline_numpy:
        ### numpy reference
        result_np = run_encoder_numpy(kwargs_numpy)
        results['numpy_reference'] = result_np 
        
    if run_baseline_cpu:
        ### vanilla sdfg 
        assign_reduce(sdfg_cpu, 'pure')
        result_bcpu = run_encoder(sdfg_cpu, kwargs_sdfg)[0]
        results['baseline_cpu'] = result_bcpu

    if run_baseline_gpu:
        sdfg_gpu.save('gpu_sdfg.sdfg')
        assign_reduce(sdfg_gpu, 'pure')
        result_bgpu = run_encoder(sdfg_gpu, kwargs_sdfg)[0]
        results['baseline_gpu_pure'] = result_bgpu

        assign_reduce(sdfg_gpu, 'CUDA (device)')
        result_bgpu = run_encoder(sdfg_gpu, kwargs_sdfg)[0]
        results['baseline_gpu_device'] = result_bgpu

    if run_expanded_cpu:
        ### preprocessed sdfg 
        expand_reductions(sdfg_cpu)
        sdfg_cpu.validate()
        result2_cpu = run_encoder(sdfg_cpu, kwargs_sdfg)
        results['preprocessed_cpu'] = result2_cpu

    if run_expanded_gpu:
        expand_reductions(sdfg_gpu)
        sdfg_gpu.validate()
        assign_reduce(sdfg_gpu, 'CUDA (block allreduce)')
        result2_gpu = run_encoder(sdfg_gpu, kwargs_sdfg)
        results['preprocessed_gpu'] = result2_gpu 

    if run_cached:
        ### cached -- pass sdfg_cpu as a reference
        result_cached = run_cached(sdfg_cpu, kwargs_sdfg)
        results['cached'] = result_cached 

    for (result_name, result_array) in results.items():
        #try:
        print(np.linalg.norm(result_array), " -> ", result_name)
        #except ValueError
        #    print(np.linalg.norm(result_array), " -> ", result_name)


    if debug:
        ### run a numpy comparision test 
        sdfg_debug = get_encoder_debug()
        sdfg_debug.apply_gpu_transformations() 

        def print_result(name, sdfg_result, numpy_result, is_list = True):
            print("--------")
            print(name)
            print("sdfg\t", np.linalg.norm(sdfg_result))
            print("nupy\t", np.linalg.norm(numpy_result[0] if is_list else numpy_result))

            sdfg_squeezed = np.squeeze(sdfg_result)
            numpy_squeezed = np.squeeze(numpy_result[0] if is_list else numpy_result)

            print("shapes =", sdfg_result.shape, numpy_squeezed.shape)
            if sdfg_squeezed.shape == numpy_squeezed.shape:
                print(np.allclose(sdfg_squeezed, numpy_squeezed, rtol = 1e-4, atol = 1e-3))


        result_np = run_encoder_numpy(kwargs_numpy, return_all_args = True)

        assign_reduce(sdfg_debug, 'pure')


        '''
        #(normed2, attn, normed1, qq, kk, vv, attn_resid) = run_encoder(sdfg_debug, kwargs_sdfg)
        # 0       1      2       3   5   6   7   -      8       4        -1          10         11
        #                                  after scaling, after softmax, after einsum, after wo -> attn

        '''
        (normed2, attn, normed1, qq, kk, vv, attn_resid, mean1, std1) = run_encoder(sdfg_debug, kwargs_sdfg)
        print_result("normed2", normed2, result_np[0])
        print_result("attn", attn, result_np[1])
        print_result("normed1", normed1, result_np[2])
        #print_result("ff", ff, result_np[3])
        print_result("kk", kk, result_np[6])
        print_result("qq", qq, result_np[5])
        print_result("vv", vv, result_np[7])
        #print_result("alpha", alpha, result_np[8])
        #print_result("gamma", gamma, result_np[4])
        print_result("attn_resid", attn_resid, result_np[-1], is_list = False)
        print_result("mean1", mean1, result_np[10], is_list = False)
        print_result("std1", std1, result_np[11], is_list = False)
        
'''   
run(run_baseline_cpu = True, 
    run_baseline_gpu = False,
    run_expanded_cpu = False,
    run_expanded_gpu = False,
    run_baseline_numpy = True,
    run_cached = False,
    debug = False)
'''

sdfg = get_encoder_cpu()
#sdfg = dace.sdfg.SDFG.from_file('../../estimator/programs/hdiff32.sdfg')
sdfg.apply_strict_transformations()
graph = sdfg.nodes()[0]
kwargs_sdfg = get_args()

# apply Reduce expansion 
'''
for node in graph.nodes():
    if isinstance(node, lib.standard.nodes.Reduce):
        ReduceExpansion.apply_to(sdfg, _reduce = node)
'''
enum_connected_prune = ConnectedEnumerator(sdfg, graph)
enum_connected_noprune = ConnectedEnumerator(sdfg, graph)
enum_connected_noprune.prune = False 
enum_greedy = GreedyEnumerator(sdfg, graph)
#enum_brute_force = BruteForceEnumerator(sdfg, graph)

def print_stats(enum, name):
    print(name)
    for maps in enum:
        print(maps)

    print(f"------ {name} ------")
    print("Yields:", enum._no_yields)
    print("CF Evaluations:", enum._no_condition_function_evaluations)


r1 = sdfg(**kwargs_sdfg) 
auto_optimize(sdfg, device = dace.dtypes.DeviceType.CPU) 
r2 = sdfg(**kwargs_sdfg) 
print(np.linalg.norm(r1))
print(np.linalg.norm(r2))
