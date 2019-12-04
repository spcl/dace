from .sql_to_json import *

import sqlite3

if __name__ == '__main__':
    mr = MergeRuns()
    mr.compensate = True
    mr.mergev2("slow_pd.db")
    exit()

    cons = Conserver()
    cons.conserveAll("perfdata.db", "test.can")
    exit()

    mr = MergeRuns()
    mr.compensate = True
    mr.merge("perfdata.db")

    conn = sqlite3.Connection("perfdata.db")
    c = conn.cursor()

    c.execute("ATTACH DATABASE ':memory:' AS scratch_default;")

    va = VectorizationAnalysis()
    va.query_values(c, 5, 40)

    exit()

    ta = ThreadAnalysis()

    ma = MemoryAnalysis()
    #ma.query_values(c, 1, 16)
    ma.query_values(c, 5, 31)
    exit()

    cpa = CriticalPathAnalysis()
    cpa.query_values(c, 1, 0)
    exit()

    ta.query_values(c, 1)
    ta.query_values(c, 1, RunID=4)
    ta.query_values(c, 1, 16)

    mf = MetaFetcher()
    state_ids = mf.getAllSectionStateIds(
        c, "# ;export OMP_NUM_THREADS=4; Running in multirun config")

    mf.getSuperSectionCount(
        c, "# ;export OMP_NUM_THREADS=4; Running in multirun config")

    mf.getAllSectionNodeIds(c, state_ids[0])

    conn.close()
