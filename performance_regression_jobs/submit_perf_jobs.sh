#!/bin/bash
# Submit all 8 perf jobs = {canon_vs, vector_vs} x {npbench, polybench, tsvc2, tsvc2_5}.
#
# Each job gets a DISTINCT results dir (results/<experiment>/, so every corpus
# lands in results/<experiment>/<corpus>/) and a unique job-name + output/error
# file (experiment + corpus + %j), so nothing collides across the 8 jobs and
# plotting stays clean. EXPERIMENT / CORPUS / RESULTS_DIR are passed to the one
# generic slurm_perf.sh via --export. Kernels self-distribute across each job's
# 4 ranks.
#
#   ./submit_perf_jobs.sh                 # clang++ for DaCe codegen (default)
#   CXX=g++ ./submit_perf_jobs.sh         # pin a different DaCe codegen compiler
set -eu
cd "$(dirname "$(readlink -f "$0")")"

CXX="${CXX:-clang++}"
EXPERIMENTS=(canon_vs vector_vs)
CORPORA=(npbench polybench tsvc2 tsvc2_5)

ids=()
for exp in "${EXPERIMENTS[@]}"; do
    results_dir="results/${exp}"
    for corpus in "${CORPORA[@]}"; do
        jid=$(sbatch --parsable \
            --job-name="dace-perf-${exp}-${corpus}" \
            --output="perf_${exp}_${corpus}_%j.out" \
            --error="perf_${exp}_${corpus}_%j.err" \
            --export=ALL,EXPERIMENT="${exp}",CORPUS="${corpus}",CXX="${CXX}",RESULTS_DIR="${results_dir}" \
            slurm_perf.sh)
        echo "submitted ${exp}/${corpus} (cpu): job ${jid} -> ${results_dir}/${corpus}/"
        ids+=("${jid}")
    done
done

# GPU canon_vs: the dace lanes on the GH200, one job per corpus (--gres=gpu:4, one GPU per rank). ONLY
# canon_vs -- vector_vs is a CPU auto-vectorizer axis with no GPU meaning. Same results dir as the cpu
# canon_vs (results/canon_vs/), so the '-gpu' preset tag lands the gpu rows beside the cpu ones in one
# summary.csv (device column = gpu). Skipped unless RUN_GPU=1, since it needs GPU allocation.
if [ "${RUN_GPU:-1}" = "1" ]; then
    for corpus in "${CORPORA[@]}"; do
        jid=$(sbatch --parsable \
            --job-name="dace-perf-canon_vs-gpu-${corpus}" \
            --output="perf_canon_vs_gpu_${corpus}_%j.out" \
            --error="perf_canon_vs_gpu_${corpus}_%j.err" \
            --gres=gpu:4 \
            --export=ALL,EXPERIMENT="canon_vs",CORPUS="${corpus}",DEVICE="gpu",CXX="${CXX}",RESULTS_DIR="results/canon_vs" \
            slurm_perf.sh)
        echo "submitted canon_vs/${corpus} (gpu): job ${jid} -> results/canon_vs/${corpus}/ [device=gpu]"
        ids+=("${jid}")
    done
fi

echo
echo "submitted ${#ids[@]} jobs: ${ids[*]}"
