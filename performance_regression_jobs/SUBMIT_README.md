# Submitting all 5 jobs

1. Find this node's CPU count: `nproc --all`
2. In each `submit_*.sh`, replace `__CPUS__` with that number (1 rank per CPU,
   `--cpus-per-task=1`). All 5 scripts run on 1 node, 8-hour time limit.
3. Submit everything at once:

   ```
   ./submit_all.sh
   ```

   Each job is submitted independently via `sbatch`, so all 4 run in parallel
   once the scheduler has room. Check status with `squeue --me`.

Results land under `results/<corpus>/` (default `--results-dir`); each
script's `--tables-only` pass at the end writes `correctness.md`/`speedup.md`.
