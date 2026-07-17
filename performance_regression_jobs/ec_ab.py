"""A/B: does InsertExplicitCopies (implicit copy edges -> explicit CopyLibraryNode) change runtime?
Baseline = the readable-perf pipeline (simplify+loop2map+mapfusion+len1->scalar); explicit = same +
InsertExplicitCopies. Correctness vs numpy, median of --reps timed reps. Legacy codegen."""
import sys, os, statistics, argparse
sys.path.insert(0,'/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs')
sys.path.insert(0,'/capstor/scratch/cscs/ybudanaz/aarch64/dace/cpu_codegen_perf_jobs')
import dace, numpy as np, npbench_polybench_perf as npp, run_readable_perf as rr, engine
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies

ap=argparse.ArgumentParser(); ap.add_argument('--reps',type=int,default=25)
ap.add_argument('--kernels',default='gemm,heat_3d,doitgen,seidel_2d,k2mm,gemver'); a=ap.parse_args()

def variant(K, info, ec):
    program,arrays,params=npp.build_program_and_data(K,info,info['parameters'][npp.PRESET])
    dace.Config.set('compiler','cpu','implementation', value='default')
    sdfg=rr.pipelined_sdfg(program, f'{K}_{"ec" if ec else "base"}', 'cpu')
    ncopies=(InsertExplicitCopies().apply_pass(sdfg, {}) or 0) if ec else 0
    return sdfg, arrays, params, ncopies

print(f"{'kernel':12s} {'copies':>6s} {'base_ms':>10s} {'explicit_ms':>11s} {'exp/base':>9s} {'correct(b/e)':>13s}",flush=True)
for K in a.kernels.split(','):
    info=npp.load_bench_info(K); res={}; corr={}; nc=0
    for tag,ec in [('base',False),('explicit',True)]:
        try:
            sdfg,arrays,params,ncopies=variant(K,info,ec); nc=ncopies or nc
            got=npp._run_dace(sdfg, info, arrays, params); ref=npp._run_numpy(info, arrays, params)
            corr[tag]=npp._compare(ref,got)
            s2,ar2,pa2,_=variant(K,info,ec); kw=npp._dace_call_kwargs(s2,ar2,pa2)
            ts=engine.time_sdfg(s2, kw, a.reps, warmup=1); res[tag]=statistics.median(ts)
        except Exception as e:
            corr[tag]=f'ERR'; res[tag]=None; print(f"  {K} {tag} ERR {str(e)[:60]}",flush=True)
    b,e=res.get('base'),res.get('explicit')
    r=f"{e/b:.3f}x" if b and e else "  -  "
    print(f"{K:12s} {nc:6d} {(b if b else 0):10.4f} {(e if e else 0):11.4f} {r:>9s}  {str(corr.get('base'))}/{str(corr.get('explicit'))}",flush=True)
print("EC_AB_DONE",flush=True)
