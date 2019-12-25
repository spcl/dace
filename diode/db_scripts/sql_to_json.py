# This file provides a wrapper that aggregates, extracts and "prettifies" analysis results coming from the SQL-Backend.
import statistics
import sqlite3
import json
import re


class ThreadAnalysis:
    """ This class provides results equivalent to the JS "ThreadAnalysis" class """

    query_string = '''
SELECT
    rs.runID as `runid`, es.threadID as `thread_id`, vs.papicode as `code`, SUM(vs.Value) as `tot_val`, ss.SuperSectionID as `ssid`
FROM
    (
        SELECT
            *
        FROM {source}`Sections` AS sec
        WHERE
            sec.nodeID = ?
            %s
    ) sec
    INNER JOIN {source}`Entries` AS es ON es.SectionID = sec.SectionID
    INNER JOIN {source}`SuperSections` AS ss ON ss.SuperSectionID = sec.ssid
    INNER JOIN
    (
        SELECT
            *
        FROM
            {source}`{source_table}` AS vs
        WHERE
            -- Select PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_L2_TCM, PAPI_L3_TCM only!
            vs.papicode = -2147483641 OR
            vs.papicode = -2147483640 OR
            vs.papicode = -2147483598 OR
            vs.papicode = -2147483589
    ) vs ON vs.entryID = es.EntryID
    INNER JOIN (
        SELECT
            *
        FROM
            {source}`Runs` AS rs
        %s
    ) rs ON rs.runID = ss.runid
GROUP BY
    rs.runID, es.threadID, vs.papicode, ss.SuperSectionID
ORDER BY
    rs.runID, es.threadID
'''

    def __init__(self,
                 cache_across_runs=False,
                 use_merged_values=False,
                 perfdata_path="perfdata.db"):
        """ If cache_across_runs is set to true, a larger preselection (only based on the runid) is made first, THEN the supersections are filtered. """
        self.verbose = False
        self.source_table = "Values"
        if use_merged_values:
            self.source_table = "MergedValues"
        if cache_across_runs:
            self.cache_db = sqlite3.connect(":memory:")
            self.cache_db_created = False
            self.cache_for_nodeid = None
            self.cache_for_runid = None
        else:
            self.cache_db = None
        self.perfdata_path = perfdata_path

    def __del__(self):
        # We want to make sure that the temporary db is closed (if it was used)
        if self.cache_db is not None:
            self.cache_db.close()

    def print(self, x):
        if self.verbose:
            return print(x)

    def query_values(self,
                     c: sqlite3.Cursor,
                     section_id,
                     supersection_id=None,
                     RunID=None,
                     papiMode="default"):

        if RunID is None:
            runsel = ""
        else:
            runsel = "WHERE rs.runid = %d" % int(RunID)

        if papiMode is not None:
            if runsel != "":
                runsel = runsel + ' AND rs.papiMode="%s"' % papiMode
            else:
                runsel = 'WHERE rs.papiMode="%s"' % papiMode

        if supersection_id is None:
            ss_sel = ""
        else:
            ss_sel = "AND sec.ssid = %d" % int(supersection_id)

        if self.cache_db is not None:
            # Use caching to reduce the very long waiting times
            pass
            # Do not select SuperSections yet.
            cache_q = ThreadAnalysis.query_string.format(
                source="`source`.", source_table=self.source_table) % ("",
                                                                       runsel)
            if not self.cache_db_created or self.cache_for_nodeid != section_id or self.cache_for_runid != RunID:

                db_path = self.perfdata_path

                self.cache_db.execute("ATTACH DATABASE '" + db_path +
                                      "' as 'source';")
                self.cache_db.execute("DROP TABLE IF EXISTS `cache_table`;")
                self.cache_db.execute(
                    "CREATE TABLE `cache_table` AS " + cache_q + ";",
                    (section_id, ))
                self.cache_db.execute("DETACH DATABASE 'source';")
                self.cache_db.execute(
                    "CREATE INDEX `cache_index` ON `cache_table`(ssid);")
                self.cache_db_created = True
                self.cache_for_nodeid = section_id
                self.cache_for_runid = RunID

            cache_c = self.cache_db.cursor()
            if supersection_id is None:
                # Nothing to do
                cache_c.execute("SELECT * FROM `cache_table`;")
            else:
                cache_c.execute("SELECT * FROM `cache_table` WHERE ssid = ?;",
                                (int(supersection_id), ))
            fa = cache_c.fetchall()

        else:
            # Continue as normal
            q = ThreadAnalysis.query_string.format(
                source="", source_table=self.source_table) % (ss_sel, runsel)
            c.execute(q + ";", (section_id, ))

            fa = c.fetchall()

        tot_cyc = []
        l2_tcm = []
        l3_tcm = []
        balance_stdev = 0
        for x in fa:
            # Unwrap
            runid, tid, papicode, papivalsum, ssid = x
            if str(papicode) == '-2147483589':
                # PAPI_TOT_CYC; we need to take a look at that.
                #print("Adding r %d, t %d to tot_cyc" % (runid, tid))
                tot_cyc.append(papivalsum)

            if str(papicode) == '-2147483640':
                # PAPI_L3_TCM; we need to take a look at that.
                #print(str(x))
                l3_tcm.append(papivalsum)

            if str(papicode) == '-2147483641':
                # PAPI_L2_TCM; we need to take a look at that.
                #print(str(x))
                l2_tcm.append(papivalsum)

        try:
            balance_stdev = statistics.stdev(tot_cyc) / statistics.median(
                tot_cyc)
        except:
            balance_stdev = 0
        if len(tot_cyc) == 0:
            # It bugged (or an analysis was requested on a section ID that does not have TOT_CYC values)
            print("Empty query for %s:%s:%s" %
                  (str(section_id), str(supersection_id), str(RunID)))
            return None

        max_c = max(tot_cyc)
        min_c = min(tot_cyc)
        self.print("min_c: " + str(min_c) + "; max_c " + str(max_c))
        max_dev = map(lambda x: abs(x - max_c), tot_cyc)
        min_dev = map(lambda x: abs(x - min_c), tot_cyc)
        max_diff = max([*max_dev, *min_dev])
        balance_max = max_diff / statistics.median(tot_cyc)

        self.print("tot_cyc " + str(tot_cyc))
        self.print("l2_tcm " + str(l2_tcm))
        self.print("l3_tcm " + str(l3_tcm))
        self.print("balance_stdev: " + str(balance_stdev))
        self.print("balance_max: " + str(balance_max))

        d = {}
        d['cycles_per_thread'] = tot_cyc
        d['l2_tcm'] = l2_tcm
        d['l3_tcm'] = l3_tcm
        d['balance_max'] = balance_max
        d['balance_stdev'] = balance_stdev
        d['critical_path'] = max_c

        #return (d, jsonstring)
        return d


class MemoryAnalysis:
    """ This class provides results equivalent to the JS "ThreadAnalysis" class """

    sub_query_string = '''
SELECT
    rs.runID as `runid`, es.threadID as `thread_id`, vs.papicode as `code`, SUM(vs.Value) as `tot_val`, ss.SuperSectionID as `ssid`, sec.datasize as `datasize`, sec.input_datasize AS `input_datasize`
FROM
    (
        SELECT
            *
        FROM `Sections` AS sec
        WHERE
            sec.nodeID = ?
            %s
    ) sec
    INNER JOIN `Entries` AS es ON es.SectionID = sec.SectionID
    INNER JOIN `SuperSections` AS ss ON ss.SuperSectionID = sec.ssid
    INNER JOIN
    (
        SELECT
            *
        FROM
            `{source_table}` AS vs
        WHERE
            -- Select PAPI_TOT_CYC, PAPI_L2_TCM, PAPI_L3_TCM only!
            vs.papicode = -2147483641 OR    --L2
            vs.papicode = -2147483640 OR    --L3
            vs.papicode = -2147483589       --CYC
    ) vs ON vs.entryID = es.EntryID
    INNER JOIN (
        SELECT
            *
        FROM
            `Runs` AS rs
        %s
    ) rs ON rs.runID = ss.runid
GROUP BY
    rs.runID, es.threadID, vs.papicode, ss.SuperSectionID
    
'''

    create_temporary_table_string = '''
        CREATE TABLE `scratch_default`.memory_analysis_query AS
        %s
        
'''

    query_string = '''
        SELECT
            l2_sum,
            l3_sum,
            SUM(CYC),
            MAX(CYC),
            (datasize * 1.0 / MAX(CYC)) AS expected_bandwidth,
            datasize AS datasize,
            input_datasize AS input_datasize
            
        FROM
            (SELECT SUM(sq.tot_val) as l2_sum FROM memory_analysis_query AS sq WHERE sq.code = -2147483641) L2_pre_q,
            (SELECT SUM(sq.tot_val) as l3_sum FROM memory_analysis_query AS sq WHERE sq.code = -2147483640) L3_pre_q,
            (SELECT sq.tot_val as cyc, sq.datasize as datasize, sq.input_datasize as input_datasize FROM memory_analysis_query AS sq WHERE code = -2147483589) cyc_pre_q
        
'''
    query_arrays = '''
        SELECT  
            c AS cyc_arr,
            l2 AS l2_arr,
            l3 AS l3_arr,
            (l2 * 1.0 / c) as l3_bandwidth_cl_per_c,
            (l3 * 1.0 / c) as mem_bandwidth_cl_per_c
            
            
        FROM
            (SELECT tot_val as l2, ssid, thread_id, runid FROM memory_analysis_query AS sq WHERE sq.code = -2147483641) L2_pre_q
            NATURAL JOIN
            (SELECT tot_val as l3, ssid, thread_id, runid FROM memory_analysis_query AS sq WHERE sq.code = -2147483640) L3_pre_q
            NATURAL JOIN 
            (SELECT tot_val as c, ssid, thread_id, runid FROM memory_analysis_query AS sq WHERE code = -2147483589) cyc_pre_q
        
'''

    def __init__(self, shared_input_db=None, from_merged=False):
        """ cache_db is a database containing the result of memory_analysis_query, except not filtered to a given supersection. """
        self.verbose = False
        self.cache_db = shared_input_db
        self.source_table = "Values"
        if from_merged:
            self.source_table = "MergedValues"

    def print(self, x):
        if self.verbose:
            return print(x)

    def query_values(self,
                     c: sqlite3.Cursor,
                     section_id,
                     supersection_id=None,
                     RunID=None,
                     target_mem_bw=20.0,
                     papiMode="default"):

        section_id = int(section_id)

        if section_id == -1:
            section_id = 0x0FFFFFFFF  # Convert to unsigned

        if supersection_id is not None:
            supersection_id = int(supersection_id)
        if RunID is not None:
            RunID = int(RunID)

        if RunID is None:
            runsel = ""
        else:
            runsel = "WHERE rs.runid = %d" % int(RunID)

        if papiMode is not None:
            if runsel != "":
                runsel = runsel + ' AND rs.papiMode="%s"' % papiMode
            else:
                runsel = 'WHERE rs.papiMode="%s"' % papiMode

        if supersection_id is None:
            ss_sel = ""
        else:
            ss_sel = "AND sec.ssid = %d" % int(supersection_id)

        # Setup the real query
        q = MemoryAnalysis.sub_query_string.format(
            source_table=self.source_table) % (ss_sel, runsel)

        # Create the temporary table
        if self.cache_db is None:
            q = MemoryAnalysis.create_temporary_table_string % q
            c.execute(q, (section_id, ))
        else:
            # Select the correct supersection Select and switch cursors (databases)
            self.print("Memory Analysis used cache")
            q = MemoryAnalysis.create_temporary_table_string % (
                "SELECT * FROM `cache_mem_query` WHERE ssid = ?")
            cache_db_cursor = self.cache_db.cursor()
            cache_db_cursor.execute(q, (supersection_id, ))
            c = cache_db_cursor

        c.execute("SELECT COUNT(*) FROM memory_analysis_query ;")
        fa = c.fetchall()
        count, = fa[0]
        if count == 0:
            # This is an invalid combination of SuperSection / nodeID.
            return None

        # Run the actual query (now operating on the temporary table)
        q = MemoryAnalysis.query_string
        c.execute(q + ";")

        fa = c.fetchall()

        if len(fa) > 1:
            self.print("Longer result")
            self.print(str(fa))
            return
        assert len(fa) == 1

        self.print(
            "Result of 'section_id' %s, 'supersection_id' %s, Run_id %s ('%s')"
            % (str(section_id), str(supersection_id), str(RunID), runsel))
        self.print(str(fa[0]))

        l2_sum, l3_sum, tot_cyc, crit_path, expected_bandwidth, datasize, input_datasize = fa[
            0]
        self.print("l2_sum:    " + str(l2_sum))
        self.print("l3_sum:    " + str(l3_sum))
        self.print("tot_cyc:   " + str(tot_cyc))
        self.print("crit_path: " + str(crit_path))
        self.print("expected_bandwidth: " + str(expected_bandwidth))
        self.print("datasize: " + str(datasize))
        self.print("input_datasize: " + str(input_datasize))

        # For the cycles, we just make it easier.
        c.execute(self.query_arrays)
        fa_arr = c.fetchall()
        self.print("fa_arr: " + str(len(fa_arr)) + ": " + str(fa_arr))

        cyc = []
        l2_misses = []
        l3_misses = []
        l3_bw_cl_per_c = []
        mem_bw_cl_per_c = []

        self.print("codesel " + str(fa_arr[0]))
        for x in fa_arr:
            y, l2, l3, l3_bw, mem_bw = x
            cyc.append(y)
            l2_misses.append(l2)
            l3_misses.append(l3)
            l3_bw_cl_per_c.append(l3_bw)
            mem_bw_cl_per_c.append(mem_bw)

        self.print("cyc:             " + str(cyc))
        self.print("l2_misses:       " + str(l2_misses))
        self.print("l3_misses:       " + str(l3_misses))
        self.print("l3_bw_cl_per_c:  " + str(l3_bw_cl_per_c))
        self.print("mem_bw_cl_per_c: " + str(mem_bw_cl_per_c))

        # Delete the temporary table again
        c.execute("DROP TABLE memory_analysis_query;")

        d = {}
        d["critical_path_cyc"] = crit_path
        d["mem_bandwidth"] = mem_bw_cl_per_c
        d["l3_bandwidth"] = l3_bw_cl_per_c
        d["TOT_CYC"] = cyc
        d["L3_TCM"] = l3_misses
        d["L2_TCM"] = l2_misses
        d["Memory_Target_Bandwidth"] = target_mem_bw
        d["expected_bandwidth"] = expected_bandwidth
        d["datasize"] = datasize
        d["input_datasize"] = input_datasize

        cacheline_size = 64
        d["bytes_from_l3"] = list(
            map(lambda x: x * cacheline_size, d["L2_TCM"]))
        d["bytes_from_mem"] = list(
            map(lambda x: x * cacheline_size, d["L3_TCM"]))

        #return (d, "data: " + json.dumps(d))
        return d


class MetaFetcher:
    """ Class dedicated to simpler aggregate functions, e.g., getting the number
        of runs, selecting supersections of a given run. """

    def __init__(self, data_source="fresh", programID=None):
        self.verbose = False  # Disable outputs by default
        self.data_source = data_source
        self.programID = programID
        if self.data_source != "fresh":
            assert programID is not None

    def print(self, x):
        if self.verbose:
            return print(x)

    def getRunOptionsFromMeta(self, c: sqlite3.Cursor, meta_str):
        if meta_str != "meta:most_cores":
            raise ValueError("Unimplemented meta string")
        if self.data_source == "fresh":
            return self.fresh_getRunOptionsFromMeta(c, meta_str)
        else:
            return self.canned_getRunOptionsFromMeta(c, meta_str)

    @staticmethod
    def getMaxOptionsStrings(fa):
        max_str = ""
        max_val = 0
        for t in fa:
            o, = t
            match = re.findall(r"OMP_NUM_THREADS=(?P<tn>\d+);", o)
            threadnum = int(match[0])
            if threadnum > max_val:
                max_str = o
                max_val = threadnum
        return str(max_str)

    def fresh_getRunOptionsFromMeta(self, c: sqlite3.Cursor, meta_str):

        c.execute("SELECT options FROM Runs;")
        fa = c.fetchall()
        max_str = MetaFetcher.getMaxOptionsStrings(fa)
        return {"options": max_str}

    def canned_getRunOptionsFromMeta(self, c: sqlite3.Cursor, meta_str):
        c.execute(
            "SELECT runoptions FROM AnalysisResults WHERE forProgramID = ?",
            (self.programID, ))
        fa = c.fetchall()
        max_str = MetaFetcher.getMaxOptionsStrings(fa)
        return {"options": max_str}

    def getRepetitionCount(self, c: sqlite3.Cursor):
        if self.data_source == "fresh":
            return self.fresh_getRepetitionCount(c)
        else:
            return self.canned_getRepetitionCount(c)

    def canned_getRepetitionCount(self, c: sqlite3.Cursor):
        c.execute(
            """
SELECT
    repetitions
FROM
    SDFGs
WHERE
    forProgramID = ?
""", (self.programID, ))

        try:
            val = c.fetchall()[0]
        except:
            return 0
        rep, = val
        return rep

    def fresh_getRepetitionCount(self, c: sqlite3.Cursor):
        # Repetitions are hard to estimate. Every program must have a toplevel
        # supersection that is emitted exactly once every invocation
        query = """
SELECT
    MIN(sscount)
FROM
(
    SELECT COUNT(ss.SuperSectionID) AS sscount FROM SuperSections as ss GROUP BY ss.runID, ss.nodeID
) sub
"""
        c.execute(query)
        try:
            val = c.fetchall()[0]
        except:
            return 0
        rep, = val
        return rep

    def getSuperSectionCount(self,
                             c: sqlite3.Cursor,
                             runoptions,
                             papimode="default"):

        if self.data_source == "fresh":
            return self.fresh_getSuperSectionCount(c, runoptions, papimode)
        else:
            return self.canned_getSuperSectionCount(c, runoptions, papimode)

    def canned_getSuperSectionCount(self,
                                    c: sqlite3.Cursor,
                                    runoptions,
                                    papimode="default"):
        if papimode != "default":
            raise ValueError("Cannot load arbitrary sections from canned data")

        # Now we know that c as a cursor of the can
        c.execute(
            """
SELECT
    COUNT(*)
FROM
(
    SELECT DISTINCT
        forSuperSection
    FROM
        AnalysisResults AS ar
    WHERE
        ar.forProgramID = ?
        AND ar.runoptions = ?
)

;        """, (
                self.programID,
                runoptions,
            ))

        f = c.fetchall()

        # Extract the count
        x, = f[0]

        self.print("SuperSection count: " + str(x))

        return int(x)

    def fresh_getSuperSectionCount(self,
                                   c: sqlite3.Cursor,
                                   runoptions,
                                   papimode="default"):

        query = """
SELECT
    COUNT(*)
FROM
    (
        SELECT
            *
        FROM
            `Runs` AS rs
        WHERE
            rs.options = ?
            AND rs.papiMode = ?
        ) rs
    INNER JOIN
    `SuperSections` AS ss
    ON rs.runID = ss.runid
;
        """
        c.execute(query, (runoptions, papimode))
        f = c.fetchall()

        # Extract the count
        x, = f[0]

        self.print("SuperSection count: " + str(x))

        return int(x)

    def getSuperSectionDBIDS(self,
                             c: sqlite3.Cursor,
                             runoptions,
                             papimode="default"):
        if self.data_source == "fresh":
            return self.fresh_getSuperSectionDBIDS(c, runoptions, papimode)
        else:
            return self.canned_getSuperSectionDBIDS(c, runoptions, papimode)

    def canned_getSuperSectionDBIDS(self,
                                    c: sqlite3.Cursor,
                                    runoptions,
                                    papimode="default"):
        if papimode != "default":
            raise ValueError("Cannot load arbitrary sections from canned data")

        # Now we know that c as a cursor of the can
        c.execute(
            """
SELECT DISTINCT
    forSuperSection
FROM
    AnalysisResults AS ar
WHERE
    ar.forProgramID = ?
    AND ar.runoptions = ?
;        """, (
                self.programID,
                runoptions,
            ))
        f = c.fetchall()

        # Get all supersections to an array
        ret = []
        for x in f:
            y, = x
            ret.append(y)

        self.print("SuperSection ids: " + str(ret))

        return ret

    def fresh_getSuperSectionDBIDS(self,
                                   c: sqlite3.Cursor,
                                   runoptions,
                                   papimode="default"):

        query = """
SELECT
    ss.SuperSectionID
FROM
    (
        SELECT
            *
        FROM
            `Runs` AS rs
        WHERE
            rs.options = ?
            AND rs.papiMode = ?
        ) rs
    INNER JOIN
    `SuperSections` AS ss
    ON rs.runID = ss.runid
;
        """
        c.execute(query, (runoptions, papimode))
        f = c.fetchall()

        # Get all supersections to an array
        ret = []
        for x in f:
            y, = x
            ret.append(y)

        self.print("SuperSection ids: " + str(ret))

        return ret

    def getAllSectionStateIds(self,
                              c: sqlite3.Cursor,
                              runoptions,
                              papimode="default"):
        if self.data_source == "fresh":
            return self.fresh_getAllSectionStateIds(c, runoptions, papimode)
        else:
            return self.canned_getAllSectionStateIds(c, runoptions, papimode)

    def canned_getAllSectionStateIds(self,
                                     c: sqlite3.Cursor,
                                     runoptions,
                                     papimode="default"):
        if papimode != "default":
            raise ValueError("Cannot load arbitrary sections from canned data")

        query = """
SELECT
    DISTINCT ((forUnifiedID >> 16) & 0xFFFF)
FROM
    AnalysisResults
WHERE
    forProgramID = ?
    AND runoptions = ?
;
        """
        c.execute(query, (
            self.programID,
            runoptions,
        ))
        f = c.fetchall()

        self.print("All Section state IDs: ")

        ret = []
        for x in f:
            y, = x
            ret.append(int(y))

        self.print(str(ret))
        return ret

    def fresh_getAllSectionStateIds(self,
                                    c: sqlite3.Cursor,
                                    runoptions,
                                    papimode="default"):

        query = """
SELECT
    DISTINCT ((sec.nodeID >> 16) & 0xFFFF)
FROM
    (
        SELECT
            *
        FROM
            `Runs` AS rs
        WHERE
            rs.options = ?
            AND rs.papiMode = ?
        ) rs
    INNER JOIN
        `SuperSections` AS ss
    ON rs.runID = ss.runid
    INNER JOIN `Sections` as sec ON sec.ssid = ss.SuperSectionID
;
        """
        c.execute(query, (runoptions, papimode))
        f = c.fetchall()

        self.print("All Section state IDs: ")

        ret = []
        for x in f:
            y, = x
            ret.append(int(y))

        self.print(str(ret))
        return ret

    def getAllSectionNodeIds(self, c: sqlite3.Cursor, stateid):
        if self.data_source == "fresh":
            return self.fresh_getAllSectionNodeIds(c, stateid)
        else:
            return self.canned_getAllSectionNodeIds(c, stateid)

    def fresh_getAllSectionNodeIds(self, c: sqlite3.Cursor, stateid):

        stateid = int(stateid)
        query = """
SELECT
    DISTINCT (sec.nodeID)
FROM
    (
    SELECT
        (sec.nodeID & 0xFFFF) as nodeID,
        ((sec.nodeID >> 16) & 0xFFFF) as stateID
    FROM 
        `Sections` AS sec
    ) sec
WHERE
    sec.stateID = ?
;
        """
        c.execute(query, (stateid, ))
        f = c.fetchall()

        self.print("All Section node IDs for state " + str(stateid))

        ret = []
        for x in f:
            y, = x
            ret.append(int(y))

        self.print(str(ret))
        return ret

    def canned_getAllSectionNodeIds(self, c: sqlite3.Cursor, stateid):

        query = """
SELECT
    DISTINCT (ar.nodeID)
FROM
    (
    SELECT
        (ar.forUnifiedID & 0xFFFF) as nodeID,
        ((ar.forUnifiedID >> 16) & 0xFFFF) as stateID
    FROM 
        `AnalysisResults` AS ar
    WHERE
        ar.forProgramID = ?
    ) ar
WHERE
    ar.stateID = ?
;
        """
        c.execute(query, (
            self.programID,
            stateid,
        ))
        f = c.fetchall()

        self.print("All Section node IDs: ")

        ret = []
        for x in f:
            y, = x
            ret.append(int(y))

        self.print(str(ret))
        return ret

    def sections(self, c: sqlite3.Cursor, ssid):
        if self.data_source == "fresh":
            return self.fresh_sections(c, ssid)
        else:
            return self.canned_sections(c, ssid)

    def canned_sections(self, c: sqlite3.Cursor, ssid):

        ssid = int(ssid)
        query = """
SELECT
    DISTINCT forSection, forUnifiedID
FROM
    `AnalysisResults` AS ar
WHERE
    ar.forProgramID = ?
    AND ar.forSuperSection = ?
;
"""
        c.execute(query, (
            self.programID,
            ssid,
        ))
        fa = c.fetchall()
        ret = []
        for x in fa:
            sec, unified_id = x
            d = {}
            d["sectionID"] = sec
            d["unified_id"] = unified_id
            ret.append(d)
        return ret

    def fresh_sections(self, c: sqlite3.Cursor, ssid):

        ssid = int(ssid)
        query = """
SELECT
    SectionID, nodeID
FROM
    `Sections` AS sec
WHERE
    sec.ssid = ?
;
"""
        c.execute(query, (ssid, ))
        fa = c.fetchall()
        ret = []
        for x in fa:
            sec, unified_id = x
            d = {}
            d["sectionID"] = sec
            d["unified_id"] = unified_id
            ret.append(d)
        return ret

    def containsSection(self, c: sqlite3.Cursor, ssid, unified_id):
        if self.data_source == "fresh":
            return self.fresh_containsSection(c, ssid, unified_id)
        else:
            return self.canned_containsSection(c, ssid, unified_id)

    def canned_containsSection(self, c: sqlite3.Cursor, ssid, unified_id):

        ssid = int(ssid)
        unified_id = int(unified_id)
        query = """
SELECT
    COUNT(*)
FROM
    `AnalysisResults` AS ar
WHERE
    ar.forProgramID = ? AND
    ar.forSuperSection = ? AND
    ar.forUnifiedID = ?
;
        """
        c.execute(query, (
            self.programID,
            ssid,
            unified_id,
        ))
        f = c.fetchall()

        ret, = f[0]
        self.print("ret: " + str(ret))
        return ret

    def fresh_containsSection(self, c: sqlite3.Cursor, ssid, unified_id):

        ssid = int(ssid)
        unified_id = int(unified_id)
        query = """
SELECT
    COUNT(*)
FROM
    `Sections` AS sec
WHERE
    sec.ssid = ? AND
    sec.nodeID = ?
;
        """
        c.execute(query, (
            ssid,
            unified_id,
        ))
        f = c.fetchall()

        ret, = f[0]
        self.print("ret: " + str(ret))
        return ret

    def SimpleQuery(self, c: sqlite3.Cursor, params):
        """ Allows executing arbitrary queries. """
        params = tuple(params)
        query, params = params

        c.execute(query, params)
        fa = c.fetchall()

        return map(tuple, fa)

    def toSectionValid(self, c: sqlite3.Cursor, ssid, unified_id):
        if self.data_source == "fresh":
            return self.fresh_toSectionValid(c, ssid, unified_id)
        else:
            return self.canned_toSectionValid(c, ssid, unified_id)

    def canned_toSectionValid(self, c: sqlite3.Cursor, ssid, unified_id):

        ssid = int(ssid)
        unified_id = int(unified_id)
        query = """
SELECT
    COUNT(*)
FROM
    `AnalysisResults` AS ar
WHERE
    ar.forProgramID = ? AND
    ar.forSuperSection = ? AND
    ar.forUnifiedID = ?
;
        """
        c.execute(query, (
            self.programID,
            ssid,
            unified_id,
        ))
        f = c.fetchall()

        ret, = f[0]
        self.print("ret: " + str(ret))
        return ret

    def fresh_toSectionValid(self, c: sqlite3.Cursor, ssid, unified_id):

        ssid = int(ssid)
        unified_id = int(unified_id)
        query = """
SELECT
    COUNT(*)
FROM
        `Sections` AS sec
    INNER JOIN
        `SuperSections` AS ss
    ON ss.SuperSectionID = sec.ssid
WHERE
    sec.ssid = ? AND
    sec.nodeID = ?
;
        """
        c.execute(query, (
            ssid,
            unified_id,
        ))
        f = c.fetchall()

        ret, = f[0]
        self.print("ret: " + str(ret))
        return ret


class CriticalPathAnalysis:
    """ Implements the CriticalPathAnalysis in SQL """

    def __init__(self, use_merged_values=False, perfdata_path='perfdata.db'):
        self.use_merged_values = use_merged_values
        self.verbose = False
        self.perfdata_path = perfdata_path

    def print(self, x):
        if self.verbose:
            return print(x)

    def query_values(self,
                     c: sqlite3.Cursor,
                     section_entry_node,
                     stateid,
                     papimode="default"):
        # This analysis operates on the entire data set

        if section_entry_node is None:
            unified_id = 0x0FFFFFFFF  # Force unsigned
        else:
            unified_id = (stateid << 16) | (section_entry_node)

        # Process:
        # 1. Of all runs, get all repetitions.
        # 2. Project for every run and repetition to the entry node and stateid
        #    given in the parameters
        # 3. Get a thread analysis of every run of the leftover result
        # 4. Get the corresponding 2-dimensional values for critical paths,
        #    cycles per thread.

        # Most of the necessary work is already done in the ThreadAnalysis class

        # Get all affected values
        c.execute(
            """
SELECT
    DISTINCT sec.ssid, ss.runid
FROM
    (
        SELECT
            *
        FROM
            `Sections` AS sec
        WHERE
            sec.nodeID = ?
    ) sec
    INNER JOIN
        `SuperSections` AS ss
    ON ss.SuperSectionID = sec.ssid
    INNER JOIN
        (
            SELECT
                *
            FROM
                `Runs` AS rs
            WHERE
                rs.papiMode = ?
        ) rs
    ON ss.runid = rs.runid
ORDER BY
    sec.ssid ASC
;
        """, (unified_id, papimode))

        # Get a list of pairs to feed into the critical path analysis later
        ssid_run_pairs = c.fetchall()
        self.print("ssid-run-pairs: " + str(ssid_run_pairs))

        # Get the repetition count - we need it to know which supersections
        # belong together because they were emitted in a loop
        mf = MetaFetcher()
        repcount = mf.getRepetitionCount(c)

        ta = ThreadAnalysis(
            cache_across_runs=True,
            use_merged_values=self.use_merged_values,
            perfdata_path=self.perfdata_path)

        pair_list = []
        critical_paths = {}
        cycles_per_thread = {}

        curr_in_rep = 0  # Not the current repetition, but the current
        # supersection INSIDE a given repetition!
        max_count_per_rep = 0

        ssid_count = 0
        for x in ssid_run_pairs:
            t_ssid, t_runid = x

            t_ssid = int(t_ssid)
            t_runid = int(t_runid)

            self.print("Analyzing run %d, ss %d, unified_id %d" %
                       (t_runid, t_ssid, unified_id))

            # Get dict
            d = ta.query_values(c, unified_id, t_ssid, RunID=t_runid)
            if d is None:
                print(
                    "[FAIL] ThreadAnalysis did not return a valid result for parameters ("
                    + str(unified_id) + ", " + str(t_ssid) + ", " +
                    str(t_runid) + ")")

            # With these analysis values, continue on
            pair_list.append((t_ssid, t_runid, d))

            if not (t_runid in critical_paths):
                self.print("First time runid %d" % int(t_runid))
                critical_paths[t_runid] = []
                cycles_per_thread[t_runid] = []
                curr_in_rep = 0
                ssid_count = len(
                    [x[0] for x in ssid_run_pairs if x[1] == t_runid])

                max_count_per_rep = int(ssid_count / repcount)
                self.print("ssid_count: " + str(ssid_count))
                self.print("repcount: " + str(repcount))
                self.print("Max count per rep: " + str(max_count_per_rep))

            if curr_in_rep == max_count_per_rep:
                curr_in_rep = 0

            if curr_in_rep == 0:
                critical_paths[t_runid].append(d["critical_path"])
                cycles_per_thread[t_runid].append(d["cycles_per_thread"])
            else:
                # This is a scalar, so it should be fine
                critical_paths[t_runid][-1] += d["critical_path"]
                # This is a list.
                cycles_per_thread[t_runid][-1] = list(
                    map(
                        lambda x: x[0] + x[1],
                        zip(cycles_per_thread[t_runid][-1],
                            d["cycles_per_thread"])))
            curr_in_rep = curr_in_rep + 1

        # Map runids to max_thread_num
        tmp = critical_paths
        critical_paths = {}
        for rid, v in tmp.items():
            c.execute(
                """
SELECT
    options
FROM
    `Runs`
WHERE
    runid = ?
""", (rid, ))
            fa = c.fetchall()
            optstring, = fa[0]
            match = re.findall(r"OMP_NUM_THREADS=(?P<tn>\d+);", optstring)
            threadnum = int(match[0])
            critical_paths[threadnum] = v

        self.print("critical_paths: " + str(critical_paths))
        T1 = critical_paths[1]
        speedup = {}
        efficiency = {}
        self.print("Analyzing CPA")

        self.print("critical_paths: " + str(critical_paths))
        self.print("cpt: " + str(cycles_per_thread))
        self.print("T1: " + str(T1))

        for r, cpt in critical_paths.items():
            speedup[r] = ([T1[j] / x for j, x in enumerate(cpt)])

        self.print(str(speedup))

        for r, s in speedup.items():
            efficiency[r] = [x / r for x in s]
        self.print(str(efficiency))

        data = {}
        speedup_d = [{"thread_num": k, "value": v} for k, v in speedup.items()]
        efficiency_d = [{
            "thread_num": k,
            "value": v
        } for k, v in efficiency.items()]
        critical_paths_d = [{
            "thread_num": k,
            "value": v
        } for k, v in critical_paths.items()]
        data["speedup"] = speedup_d
        data["efficiency"] = efficiency_d
        data["critical_paths"] = critical_paths_d

        self.print(str(speedup_d))
        self.print(str(efficiency_d))

        return data


class MergeRuns:
    """ Class providing functionality to merge counter values of different runs.
        Intended to be used before every other analysis. """

    def __init__(self):
        from dace.codegen.instrumentation.papi import PAPISettings
        self.verbose = False
        self.compensate = PAPISettings.perf_compensate_overhead()

    def print(self, x):
        if self.verbose:
            return print(x)

    def mergev2(self, db_path):
        from dace.codegen.instrumentation.papi import PAPISettings
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        # Default is roughly 2MB of cache size. This sets it to just under 1GiB of RAM.
        c.execute("PRAGMA cache_size=200000")

        c.execute("ATTACH DATABASE ':memory:' AS scratch_default;")

        # Create index

        c.execute(
            "CREATE INDEX IF NOT EXISTS `SuperSection_index` ON SuperSections(SuperSectionID, runid, nodeID);"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS `Section_index` ON Sections(SectionID, ssid, nodeID);"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS `Entries_index` ON Entries(EntryID, SectionID, threadID);"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS `Values_index` ON `Values`(entryID, papiCode);"
        )

        # =============

        # Create a place to put all the merged data (since it's not
        # technically safe to merge, we don't modify the Values-table)
        c.execute("DROP TABLE IF EXISTS `MergedValues`;")
        c.execute("""
        CREATE TABLE `MergedValues`
             (ValueID INTEGER PRIMARY KEY,
             PapiCode INTEGER,
             Value INTEGER,
             entryID INTEGER,
             oldEntryID INTEGER,
             FOREIGN KEY(entryID) REFERENCES Entries(EntryID));
        """)
        PAPISettings.merging_print("Copying values")
        c.execute("""
        INSERT INTO `MergedValues`
        SELECT
            vs.*, vs.entryID -- Duplicate the entryID, as we will change one later
        FROM
            `Values` AS vs
        ;
        """)
        PAPISettings.merging_print("Creating index")
        # Create an Index on the old entry ID for faster loopups
        c.execute(
            "CREATE INDEX IF NOT EXISTS `oldEntryID_index` ON `MergedValues`(oldEntryID);"
        )

        update_list = []
        PAPISettings.merging_print("Running query")

        # Order properly first, then merge by rowids.
        query = """
SELECT
    *
FROM
        Runs AS rs 
    INNER JOIN
        `SuperSections` AS ss
    ON ss.runid = rs.runid
    INNER JOIN
        `Sections` AS sec
    ON sec.ssid = ss.SuperSectionID
    INNER JOIN
        `Entries` AS es
    ON es.SectionID = sec.SectionID
WHERE
    rs.papiMode = '{mode}'
ORDER BY
    ss.SuperSectionID, sec.nodeID, es.iteration, es.threadID, es.`order`
        """

        PAPISettings.merging_print("Loading default table")
        # Get a default table up.
        c.execute("CREATE TEMPORARY TABLE `default_sel` AS " +
                  query.format(mode="default"))
        PAPISettings.merging_print("Creating index")
        c.execute(
            "CREATE INDEX `ds_ind` ON `default_sel`(SuperSectionID, entryID, threadID, iteration);"
        )

        # Then, do a similar ordering for all other modes
        c.execute("SELECT DISTINCT papiMode FROM Runs;")
        modes = c.fetchall()
        for x in modes:
            modestr, = x
            if modestr == "default":
                continue
            PAPISettings.merging_print("Generating table for " + modestr)
            table_str = "{s}_sel".format(s=modestr)
            c.execute("CREATE TEMPORARY TABLE `{tablename}` AS ".format(
                tablename=table_str) + query.format(mode=modestr))
            c.execute(
                "CREATE INDEX `{t}_ind` ON `default_sel`(SuperSectionID, entryID, threadID, iteration);"
                .format(t=table_str))
            # Cool, now we just match by rowid
            PAPISettings.merging_print("Creating temporary table...")
            esel_query = """
SELECT
    DISTINCT ds.entryID as entryid, ms.entryID as oldentryid
FROM `default_sel` AS ds INNER JOIN `{t}` AS ms ON ds.rowid = ms.rowid
;
"""
            c.execute("CREATE TEMPORARY TABLE `sel` AS " +
                      esel_query.format(t=table_str))
            PAPISettings.merging_print("Creating index on selection")
            c.execute(
                "CREATE INDEX `sel_index` ON `sel`(entryID, oldentryID);")

            update_command = """
UPDATE
    `MergedValues` AS mv
SET
    entryID = (SELECT sel.entryID FROM `sel` WHERE sel.oldentryID = mv.oldEntryID LIMIT 1)
--WHERE
--    oldEntryID IN (SELECT sel.oldentryID FROM `sel`)
;
""".format(t=table_str)

            c.execute("SELECT entryID, oldEntryID FROM `sel`;")
            update_list.extend(c.fetchall())

            PAPISettings.merging_print("Dropping table")
            c.execute("DROP TABLE `sel`;")
            c.execute("DROP TABLE `{t}`;".format(t=table_str))
            c.execute("DROP INDEX `{t}_ind`;".format(t=table_str))

        PAPISettings.merging_print("Updating %d values" % len(update_list))
        c.executemany(
            """
UPDATE
    `MergedValues`
SET
    entryID = ?
WHERE
    -- entryID = X 
    oldEntryID = ?
;
        """, update_list)

        if self.compensate:
            self.compensate_func(c)

        PAPISettings.merging_print("Committing...")

        conn.commit()

        PAPISettings.merging_print("Cleaning")
        conn.execute("VACUUM;")
        conn.execute("PRAGMA optimize;")
        conn.commit()
        conn.close()

        PAPISettings.merging_print("Done")

    def check_1(self, c: sqlite3.Cursor):
        pass

    def compensate_func(self, c: sqlite3.Cursor):
        # Compensate
        q = """
UPDATE
`MergedValues`
SET
`value` = `value` - (
    SELECT
        CAST(comp.value AS INTEGER) AS compval
    FROM
        `Overheads` AS comp
    WHERE
        comp.papiCode = `MergedValues`.`papiCode`
)
WHERE EXISTS (
    SELECT
        CAST(comp.value AS INTEGER) AS compval
    FROM
        `Overheads` AS comp
    WHERE
        comp.papiCode = `MergedValues`.`papiCode`
)
"""
        c.execute(q)

    def merge(self, db_path):
        pass

        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        c.execute("ATTACH DATABASE ':memory:' AS scratch_default;")

        # Create index

        c.execute(
            "CREATE INDEX IF NOT EXISTS `SuperSection_index` ON SuperSections(SuperSectionID, runid, nodeID);"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS `Section_index` ON Sections(SectionID, ssid, nodeID);"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS `Entries_index` ON Entries(EntryID, SectionID, threadID);"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS `Values_index` ON `Values`(entryID, papiCode);"
        )

        # =============

        # Create a place to put all the merged data
        c.execute("DROP TABLE IF EXISTS `MergedValues`;")
        c.execute("""
        CREATE TABLE `MergedValues`
             (ValueID INTEGER PRIMARY KEY,
             PapiCode INTEGER,
             Value INTEGER,
             entryID INTEGER,
             oldEntryID INTEGER,
             FOREIGN KEY(entryID) REFERENCES Entries(EntryID));
        """)
        c.execute("""
        INSERT INTO `MergedValues`
        SELECT
            vs.*, vs.entryID -- Duplicate the entryID, as we will change one later
        FROM
            `Values` AS vs
        ;
        """)
        # Create an Index on the old entry ID for faster loopups
        c.execute(
            "CREATE INDEX IF NOT EXISTS `oldEntryID_index` ON `MergedValues`(oldEntryID);"
        )

        # TODO: Performance is abysmal (probably because it is executed
        # sequentially)

        print("Running query")
        query = """
SELECT
    DISTINCT
    rs.runid AS runid,
    rs.options AS options,
    rs.papiMode AS mode,
    sec.`order` AS secorder,
    es.`order` AS esorder,
    es.entryID AS entryID,
    sec.nodeID AS nodeID,
    es.threadID AS threadID,
    es.iteration AS iteration
FROM
        `Runs` AS rs
    INNER JOIN
        `SuperSections` AS ss
    ON ss.runid = rs.runid
    INNER JOIN
        `Sections` AS sec
    ON sec.ssid = ss.SuperSectionID
    INNER JOIN
        `Entries` AS es
    ON es.SectionID = sec.SectionID
WHERE
    rs.papiMode <> 'default'
"""

        # Store the result of the query, as we intend to do some more operations on that
        c.execute("CREATE TABLE `scratch_default`.temp_presel AS " + query)

        # Create an index on the later-used columns (there are no more inserts)
        c.execute(
            "CREATE INDEX `scratch_default`.presel_ind ON `temp_presel`(runID, nodeID, threadID, iteration, entryID)"
        )

        # Select everything from that table for analysis.
        c.execute("SELECT * FROM `scratch_default`.temp_presel;")
        fa = c.fetchall()

        # Cache values such that the same information does not have to be refetched
        cache_tuple = None
        def_fa = []

        debug_print = False

        # We setup another cached (temp) table for performance reasons
        defsel_query = """
    SELECT
        sec.`order` AS sec_order, es.`order` AS es_order, es.entryID AS entryID,
        rs.options AS options, sec.nodeID AS nodeID, es.threadID AS threadID, es.iteration AS iteration
    FROM 
            `Runs` AS rs
        INNER JOIN
            `SuperSections` AS ss
        ON ss.runid = rs.runid
        INNER JOIN
            `Sections` AS sec
        ON sec.ssid = ss.SuperSectionID
        INNER JOIN
            `Entries` AS es
        ON es.SectionID = sec.SectionID
    WHERE
        rs.papiMode = 'default'
    """

        # Create the table
        c.execute("CREATE TABLE `scratch_default`.temp_defsel AS " +
                  defsel_query)

        # Create an index for fast lookups
        c.execute(
            "CREATE INDEX `scratch_default`.ds_ind ON temp_defsel(options, nodeID, threadID, iteration)"
        )

        print("Selecting every element from %d elements" % len(fa))

        report_step = int(len(fa) / 100)

        previous_length = None

        update_list = [
        ]  # Holds the list of tuples used to update the value linking (using executemany to increase performance)
        for i, x in enumerate(fa):
            if report_step != 0 and i % report_step == 0:
                print("%d percent done" % (int(float(i) * 100.0 / len(fa))))
            if debug_print:
                print(str(x))

            # For each row of the result, find the corresponding default
            runid, options, mode, secorder, esorder, entryID, nodeID, threadID, iteration = x

            defsel_tuple = (options, nodeID, threadID, iteration)

            if defsel_tuple != cache_tuple:
                # Create a sub-selection because this might be repeated often
                # Limit recreation to a minimum
                if cache_tuple is None or (defsel_tuple[0] != cache_tuple[0] or
                                           defsel_tuple[1] != cache_tuple[1]):
                    c.execute(
                        "DROP TABLE IF EXISTS `scratch_default`.`subsel_1`;")
                    subsel_query = "SELECT * FROM `scratch_default`.`temp_presel` WHERE runid=? AND nodeID=?"
                    c.execute(
                        "CREATE TABLE `scratch_default`.`subsel_1` AS " +
                        subsel_query + ";", (
                            runid,
                            nodeID,
                        ))
                    c.execute(
                        "CREATE INDEX `scratch_default`.`subsel_index` ON `subsel_1`(threadID, iteration, entryID);"
                    )

                c.execute(
                    """
    SELECT
        td.sec_order, td.es_order, td.entryID
    FROM 
        `scratch_default`.`temp_defsel` AS td
    WHERE
        td.options = ?
        AND td.nodeID = ?
        AND td.threadID = ?
        AND td.iteration = ?
    --ORDER BY
    --    td.sec_order, td.es_order, td.entryID
    """, defsel_tuple)

                cache_tuple = defsel_tuple
                def_fa = c.fetchall()
            else:
                self.print("Used cache!")

            length_mismatched = False

            if previous_length is not None:
                if previous_length != len(def_fa):
                    print("Length mismatch %d vs %d" % (previous_length,
                                                        len(def_fa)))
                    length_mismatched = True

                    index = None
                    print("def_fa: %s" % str(def_fa))
                    previous_length = len(def_fa)
            else:
                previous_length = len(def_fa)
            if debug_print:
                print("Got %d default candidates" % (len(def_fa)))
            # There most likely WILL be multiple values for the same node
            # Since the multiple values MUST be ordered however (by iteration)
            # this can be matched by the order-fields.

            # Select all elements that match the nodeid, threadid and iteration and runid (old query)
            counter_query = """
SELECT
    COUNT(*)
FROM
    `scratch_default`.`temp_presel` AS tp
WHERE
    tp.runID = ?
    AND tp.nodeID = ?
    AND tp.threadID = ?
    AND tp.iteration = ?
    AND tp.entryID < ?
"""
            lean_counter_query = """
SELECT
    COUNT(*)
FROM
    `scratch_default`.`subsel_1` AS tp
WHERE
    tp.threadID = ?
    AND tp.iteration = ?
    AND tp.entryID < ?
"""
            c.execute(lean_counter_query, (threadID, iteration, entryID))
            index, = c.fetchall()[0]

            try:
                if length_mismatched:
                    raise ValueError(
                        "Length mismatched, go try the non-thread-fixed mode")
                def_val = def_fa[index]
            except:
                c.execute(
                    """
SELECT
    td.sec_order, td.es_order, td.entryID
FROM 
        `scratch_default`.`temp_defsel` AS td
WHERE
    td.options = ?
    AND td.nodeID = ?
    AND td.iteration = ?
--ORDER BY
--    td.sec_order, td.es_order, td.entryID
""", (options, nodeID, iteration))

                def_fa = c.fetchall()

                reduced_counter_query = """
SELECT
    COUNT(*)
FROM
    `scratch_default`.`temp_presel` AS tp
WHERE
    tp.runID = ?
    AND tp.nodeID = ?
    AND tp.iteration = ?
    AND tp.entryID < ?
"""
                c.execute(reduced_counter_query,
                          (runid, nodeID, iteration, entryID))
                tmp = c.fetchall()
                index, = tmp[0]
                try:
                    def_val = def_fa[index]
                except Exception as e:
                    print("def_fa:")
                    for x in def_fa:
                        print(x)
                    print("index: " + str(index))
                    raise e

            def_sec_order, def_es_order, def_es_id = def_val

            if debug_print:
                print("Matching index found: " + str(index) +
                      ", meaning %s => %s" % (str(entryID), str(def_es_id)))
            update_list.append((def_es_id, entryID))

        # TODO: Think about how to flush results every x entries to keep memory
        # usage reasonable.

        print("Updating %d values" % len(update_list))
        c.executemany(
            """
UPDATE
    `MergedValues`
SET
    entryID = ?
WHERE
    -- entryID = X
    oldEntryID = ?
;
        """, update_list)

        # After inserting, create an INDEX for faster lookups later
        c.execute("CREATE INDEX `mergedvalues_index` ON MergedValues(entryID)")

        # Apply an overhead compensation on the values (if requested)
        if self.compensate:
            pass

            self.compensate_func(c)

        c.execute("DROP TABLE `scratch_default`.temp_defsel")
        c.execute("DROP TABLE `scratch_default`.temp_presel")

        # Post-update checks
        print("Running checks")
        # Checking code
        c.execute("""
SELECT
    *
FROM
        `MergedValues` AS mv
    INNER JOIN
        `Entries` AS es
    ON es.entryID = mv.entryID
    INNER JOIN
        `Sections` AS sec
    ON es.SectionID = sec.SectionID
    INNER JOIN
        `SuperSections` AS ss
    ON sec.ssid = ss.SuperSectionID
    INNER JOIN
        `Runs` AS rs
    ON rs.runid = ss.runID
WHERE
    rs.papiMode <> "default"
""")

        fa_check1 = c.fetchall()
        if len(fa_check1) > 0:
            print("%d values are not relating to default" % len(fa_check1))
            for x in fa_check1:
                print(str(x))
                i = input()
                if i == 'c':
                    break
        assert len(fa_check1) == 0  # Please oh please don't have any leftovers

        print("Committing...")

        conn.commit()

        print("Cleaning")
        conn.execute("VACUUM;")

        conn.commit()

        conn.execute("PRAGMA optimize;")
        conn.close()

        print("Done")


class VectorizationAnalysis:
    """ Provides analysis of vectorization/flop counters. This requires a 
        MergedValues-table """

    query = """
SELECT
    *
FROM
    (
        SELECT *
        FROM `Sections` AS sec
        WHERE
            sec.nodeID = ?
            %s
    ) sec
    INNER JOIN
        `Entries` AS es
    ON es.sectionID = sec.SectionID
    INNER JOIN
        `MergedValues` AS mv
    ON mv.entryID = es.entryID

"""

    thread_query = """
SELECT
    tms.threadID, tms.papiCode, SUM(tms.value)
FROM
    `temp_merge_sel` AS tms
GROUP BY
    tms.threadID, tms.papiCode

"""

    def __init__(self,
                 shared_input_db=None,
                 critical_path_analysis=None,
                 perfdata_path="perfdata.db"):
        """ Initializes this analysis class.
            When a shared input database is specified, joins are not executed,
            but instead taken from the input database from table `temp_merge_sel`. 
            This table contains the result of VectorizationAnalysis.query.
            It can be reused for multiple queries.
            critical_path_analysis can be supplied to avoid recalculation.
        """
        self.shared_input_db = shared_input_db
        self.critical_path_analysis = critical_path_analysis
        self.verbose = False
        self.perfdata_path = perfdata_path

    def print(self, x):
        if self.verbose:
            return print(x)

    def query_values(self, c: sqlite3.Cursor, section_id,
                     supersection_id=None):

        self.print("Running vectorization analysis")

        if self.critical_path_analysis is None:
            cpa = CriticalPathAnalysis(perfdata_path=self.perfdata_path)
            cpa_data = cpa.query_values(
                c, section_id, 0
            )  # stateid=0 is fine here because section_id is the unified id.
        else:
            cpa_data = self.critical_path_analysis(section_id)

        if self.shared_input_db is None:
            ss_sel = ""
            if supersection_id is not None:
                ss_sel = "AND sec.ssid = %s" % str(supersection_id)
            q = VectorizationAnalysis.query % ss_sel

            c.execute(
                "CREATE TABLE `scratch_default`.`temp_merge_sel` AS " + q,
                (section_id, ))
            c.execute(
                "CREATE INDEX IF NOT EXISTS `scratch_default`.`tms_ind` ON `temp_merge_sel`(sectionID, threadID, papiCode);"
            )

            # Select the sums of everything, as (threadID, papiCode, sum) (due to pre-filtering, only of a single section)

            c.execute(VectorizationAnalysis.thread_query)
            fa = c.fetchall()
        else:
            # Use the cache, which is faster
            cache_c = self.shared_input_db.cursor()
            # Try the full cache first
            try:
                cache_c.execute("SELECT * FROM `thread_grouped`;")
                fa = cache_c.fetchall()
            except:
                # If not available, do the grouping yourself
                cache_c.execute(VectorizationAnalysis.thread_query)
                fa = cache_c.fetchall()

        sp_ops = []
        dp_ops = []

        sp_flops = []
        dp_flops = []
        sp_ops_scalar_single = []
        sp_ops_128b_single = []
        sp_ops_256b_single = []
        sp_ops_512b_single = []

        sp_ops_scalar_double = []
        sp_ops_128b_double = []
        sp_ops_256b_double = []
        sp_ops_512b_double = []

        # TODO: Allow a fallback using the more generic counters...

        tot_cycles = []
        for x in fa:
            threadID, papiCode, val = x

            if papiCode == -2147483589:
                tot_cycles.append(val)
            elif papiCode == -2147483544:
                dp_ops.append(val)
            elif papiCode == 1073741857:
                sp_ops_scalar_double.append(val)
            elif papiCode == 1073741858:
                sp_ops_128b_double.append(val)
            elif papiCode == 1073741859:
                sp_ops_256b_double.append(val)
            elif papiCode == 1073741860:
                sp_ops_512b_double.append(val)
            elif papiCode == -2147483545:
                sp_ops.append(val)
            elif papiCode == 1073741861:
                sp_ops_scalar_single.append(val)
            elif papiCode == 1073741862:
                sp_ops_128b_single.append(val)
            elif papiCode == 1073741863:
                sp_ops_256b_single.append(val)
            elif papiCode == 1073741864:
                sp_ops_512b_single.append(val)

        if len(sp_ops) == 0:
            sp_ops = sum(sp_ops_scalar_single) + sum(sp_ops_128b_single) + sum(
                sp_ops_256b_single) + sum(sp_ops_512b_single)
        if len(dp_ops) == 0:
            dp_ops = sum(sp_ops_scalar_double) + sum(sp_ops_128b_double) + sum(
                sp_ops_256b_double) + sum(sp_ops_512b_double)

        # Multiply to get FLOP counts (instead of instruction counts)
        def flop_each_thread(x):
            # We have in x: [[scalar_t1, scalar_t2, ...], [packed1_t1, packed1_t2, ...], ...]
            # We want to add these inner arrays together.
            zipped = zip(*x)
            # Now we can map each element of zipped to its sum to get flop per thread
            summed = map(sum, zipped)  # lambda x: sum(list(x))
            return list(summed)

        sp_flops = flop_each_thread([
            map(lambda x: x * 1, sp_ops_scalar_single),
            map(lambda x: x * 4, sp_ops_128b_single),
            map(lambda x: x * 8, sp_ops_256b_single),
            map(lambda x: x * 16, sp_ops_512b_single)
        ])
        dp_flops = flop_each_thread([
            map(lambda x: x * 1, sp_ops_scalar_double),
            map(lambda x: x * 2, sp_ops_128b_double),
            map(lambda x: x * 4, sp_ops_256b_double),
            map(lambda x: x * 8, sp_ops_512b_double)
        ])

        data = {}
        data["sp_flops"] = sp_flops
        data["dp_flops"] = dp_flops

        data["sp_flops_all"] = sum(sp_flops)
        data["dp_flops_all"] = sum(dp_flops)

        data["sp_flops_per_cycle"] = list(
            map(lambda x: float(x[0]) / x[1], zip(data["sp_flops"],
                                                  tot_cycles)))
        data["dp_flops_per_cycle"] = list(
            map(lambda x: float(x[0]) / x[1], zip(data["dp_flops"],
                                                  tot_cycles)))

        self.print("cpa_data: " + str(cpa_data["critical_paths"]))
        max_thread_num_crit_path = []
        max_thread_num = 0
        for x in cpa_data["critical_paths"]:
            if max_thread_num < int(x["thread_num"]):
                max_thread_num = int(x["thread_num"])
                max_thread_num_crit_path = x["value"]
        self.print("max_thread_num_crit_path" + str(max_thread_num_crit_path))
        crit_path = statistics.median(max_thread_num_crit_path)
        self.print("crit_path: " + str(crit_path))

        data["sp_flops_per_cycle_parallel"] = sum(data["sp_flops"]) / crit_path
        data["dp_flops_per_cycle_parallel"] = sum(data["dp_flops"]) / crit_path

        data["sp_ops"] = sp_ops
        data["dp_ops"] = dp_ops

        data["sp_ops_scalar"] = sp_ops_scalar_single
        data["sp_ops_128b"] = sp_ops_128b_single
        data["sp_ops_256b"] = sp_ops_256b_single
        data["sp_ops_512b"] = sp_ops_512b_single

        data["dp_ops_scalar"] = sp_ops_scalar_double
        data["dp_ops_128b"] = sp_ops_128b_double
        data["dp_ops_256b"] = sp_ops_256b_double
        data["dp_ops_512b"] = sp_ops_512b_double

        if self.shared_input_db is None:
            c.execute("DROP TABLE `scratch_default`.`temp_merge_sel`")

        self.print(str(data))

        return data


class CacheOpAnalysis:
    """ Provides analysis of cache operation counters. This requires a MergedValues-table """

    query = """
SELECT
    *
FROM
    (
        SELECT *
        FROM `Sections` AS sec
        WHERE
            sec.nodeID = ?
            %s
    ) sec
    INNER JOIN
        `Entries` AS es
    ON es.sectionID = sec.SectionID
    INNER JOIN
        `MergedValues` AS mv
    ON mv.entryID = es.entryID

"""

    thread_query = """
SELECT
    tms.threadID, tms.papiCode, SUM(tms.value)
FROM
    `temp_merge_sel` AS tms
GROUP BY
    tms.threadID, tms.papiCode

"""

    def __init__(self, shared_input_db=None):
        """ Initializes this analysis class.
        When a shared input database is specified, joins are not executed,
        but instead taken from the input database from table `temp_merge_sel`. 
        This table contains the result of VectorizationAnalysis.query.
        It can be reused for multiple queries.
        """
        self.shared_input_db = shared_input_db
        self.verbose = False

    def print(self, x):
        if self.verbose:
            return print(x)

    def query_values(self, c: sqlite3.Cursor, section_id,
                     supersection_id=None):

        self.print("Running cacheop analysis")

        #cpa = CriticalPathAnalysis()
        #cpa_data = cpa.query_values(c, section_id, 0) # stateid=0 is fine here because section_id is the unified id.

        if self.shared_input_db is None:
            ss_sel = ""
            if supersection_id is not None:
                ss_sel = "AND sec.ssid = %s" % str(supersection_id)
            q = CacheOpAnalysis.query % ss_sel

            c.execute(
                "CREATE TABLE `scratch_default`.`temp_merge_sel` AS " + q,
                (section_id, ))
            c.execute(
                "CREATE INDEX IF NOT EXISTS `scratch_default`.`tms_ind` ON `temp_merge_sel`(sectionID, threadID, papiCode);"
            )

            # Select the sums of everything, as (threadID, papiCode, sum) (due to pre-filtering, only of a single section)

            c.execute(CacheOpAnalysis.thread_query)
            fa = c.fetchall()
        else:
            # Use the cache, which is faster
            cache_c = self.shared_input_db.cursor()
            cache_c.execute(CacheOpAnalysis.thread_query)
            fa = cache_c.fetchall()

        cache_snoop = []
        cache_shared_to_exclusive = []
        cache_clean_to_exclusive = []
        cache_line_intervention = []

        tot_cycles = []
        for x in fa:
            threadID, papiCode, val = x

            if papiCode == -2147483589:
                tot_cycles.append(val)
            elif papiCode == -2147483639:
                cache_snoop.append(val)
            elif papiCode == -2147483638:
                cache_shared_to_exclusive.append(val)
            elif papiCode == -2147483637:
                cache_clean_to_exclusive.append(val)
            elif papiCode == -2147483635:
                cache_line_intervention.append(val)

        if self.shared_input_db is None:
            c.execute(
                "DROP TABLE IF EXISTS `scratch_default`.`temp_merge_sel`;")
        data = {}

        data["tot_cyc"] = tot_cycles

        data["cache_snoop"] = cache_snoop
        data["cache_shr2ex"] = cache_shared_to_exclusive
        data["cache_cln2ex"] = cache_clean_to_exclusive
        data["cache_intervention"] = cache_line_intervention

        self.print(str(data))

        return data


class MemoryOpAnalysis:
    """ Provides analysis of cache operation counters. This requires a MergedValues-table """

    query = """
SELECT
    *
FROM
    (
        SELECT *
        FROM `Sections` AS sec
        WHERE
            sec.nodeID = ?
            %s
    ) sec
    INNER JOIN
        `Entries` AS es
    ON es.sectionID = sec.SectionID
    INNER JOIN
        `MergedValues` AS mv
    ON mv.entryID = es.entryID

"""

    thread_query = """
SELECT
    tms.threadID, tms.papiCode, SUM(tms.value)
FROM
    `temp_merge_sel` AS tms
GROUP BY
    tms.threadID, tms.papiCode

"""

    def __init__(self, shared_input_db=None):
        """ Initializes this analysis class.
            When a shared input database is specified, joins are not executed,
            but instead taken from the input database from table `temp_merge_sel`. 
            This table contains the result of VectorizationAnalysis.query.
            It can be reused for multiple queries.
        """
        self.shared_input_db = shared_input_db

        self.verbose = False

    def print(self, x):
        if self.verbose:
            return print(x)

    def query_values(self, c: sqlite3.Cursor, section_id,
                     supersection_id=None):

        self.print("Running cacheop analysis")

        if self.shared_input_db is None:
            ss_sel = ""
            if supersection_id is not None:
                ss_sel = "AND sec.ssid = %s" % str(supersection_id)
            q = MemoryOpAnalysis.query % ss_sel

            c.execute(
                "CREATE TABLE `scratch_default`.`temp_merge_sel` AS " + q,
                (section_id, ))
            c.execute(
                "CREATE INDEX IF NOT EXISTS `scratch_default`.`tms_ind` ON `temp_merge_sel`(sectionID, threadID, papiCode);"
            )

            # Select the sums of everything, as (threadID, papiCode, sum) (due to pre-filtering, only of a single section)

            c.execute(MemoryOpAnalysis.thread_query)
            fa = c.fetchall()
        else:
            # Use the cache, which is faster
            cache_c = self.shared_input_db.cursor()
            cache_c.execute(MemoryOpAnalysis.thread_query)
            fa = cache_c.fetchall()

        write_stall = []
        load_ins = []
        store_ins = []

        tot_cycles = []
        for x in fa:
            threadID, papiCode, val = x

            if papiCode == -2147483589:
                tot_cycles.append(val)
            elif papiCode == -2147483612:  # PAPI_MEM_WCY: Write cycle stalls
                write_stall.append(val)
            elif papiCode == -2147483595:
                load_ins.append(val)
            elif papiCode == -2147483594:
                store_ins.append(val)

        if self.shared_input_db is None:
            c.execute("DROP TABLE IF EXISTS `temp_merge_sel`;")
        data = {}

        data["tot_cyc"] = tot_cycles

        data["write_stall"] = write_stall

        data["percent_write_stall"] = list(
            map(lambda x: (x[0] * 100.0) / x[1],
                zip(data["write_stall"], data["tot_cyc"])))

        data["load_ins"] = load_ins
        data["store_ins"] = store_ins

        self.print(str(data))

        return data


class Conserver:
    def __init__(self):
        pass

    def conserveAll(self,
                    db_path,
                    out_path,
                    sdfg_json,
                    repetitions,
                    clear_existing=False):

        from dace.codegen.instrumentation.papi import PAPISettings
        conn = sqlite3.connect(db_path)

        c = conn.cursor()

        # Attach a default scratch space to every connection such that we can write temporary dbs concurrently
        c.execute("ATTACH DATABASE ':memory:' AS scratch_default;")

        # Select all sections that are available
        c.execute("SELECT DISTINCT nodeID FROM `Sections`;")
        unified_ids = list(c.fetchall())

        # Create a new Database to just hold the analysis results
        outdb = sqlite3.connect(out_path)
        outdbc = outdb.cursor()

        if clear_existing:
            # Drop any existing table if requested
            outdbc.execute("DROP TABLE IF EXISTS AnalysisResults;")
            outdbc.execute("DROP TABLE IF EXISTS SDFGs;")

        outdbc.execute("""
CREATE TABLE IF NOT EXISTS AnalysisResults
(
    forProgramID INTEGER,
    AnalysisName VARCHAR(256),
    runoptions TEXT,
    forUnifiedID INTEGER,
    forSuperSection INTEGER,
    forSection INTEGER,
    json TEXT

);
""")

        # Create the table for the SDFGs
        outdbc.execute("""
CREATE TABLE IF NOT EXISTS SDFGs
(
    forProgramID INTEGER PRIMARY KEY,
    optimizations TEXT, -- Information about what optimizations have been applied
    repetitions INTEGER, -- Count of repetitions when running this program.
    json TEXT -- The JSON of the given SDFG
);
""")

        # Write the SDFG to the database
        outdbc.execute(
            """
INSERT INTO `SDFGs`
(optimizations, repetitions, json)    
VALUES
(
    ?, ?, ?
);                
""", (
                "NotImplemented",
                repetitions,
                sdfg_json,
            ))

        # Get the affected to store references
        forProgramID = outdbc.lastrowid

        # create a cached subselection for this unified id.
        cache_conn = sqlite3.connect(":memory:")
        cache_conn.execute("ATTACH DATABASE ':memory:' AS scratch_default;")

        # Provide a function to extract critical path analysis data
        def get_cpa(nodeid):
            outdbc.execute(
                "SELECT json FROM AnalysisResults WHERE AnalysisName='CriticalPathAnalysis' AND forUnifiedID=? AND forProgramID=?",
                (nodeid, forProgramID))
            res = outdbc.fetchall()
            assert len(res) == 1
            extract, = res[0]
            return json.loads(extract)

        thread_analysis = ThreadAnalysis(
            cache_across_runs=True,
            use_merged_values=True,
            perfdata_path=db_path)
        critical_path_analysis = CriticalPathAnalysis(perfdata_path=db_path)
        memory_analysis = MemoryAnalysis(
            shared_input_db=cache_conn, from_merged=True)
        vectorization_analysis = VectorizationAnalysis(
            shared_input_db=cache_conn,
            critical_path_analysis=get_cpa,
            perfdata_path=db_path)
        memory_op_analysis = MemoryOpAnalysis(shared_input_db=cache_conn)
        cache_op_analysis = CacheOpAnalysis(shared_input_db=cache_conn)

        analyses = [
            ("ThreadAnalysis", thread_analysis,
             lambda unified_id, ssid, sid: [unified_id, ssid]),
            ("CriticalPathAnalysis", critical_path_analysis,
             lambda unified_id, ssid, sid:
             [int(unified_id) & 0xFFFF, (int(unified_id) >> 16) & 0xFFFF]),
            ("MemoryAnalysis", memory_analysis,
             lambda unified_id, ssid, sid: [unified_id, ssid]),
            ("VectorizationAnalysis", vectorization_analysis,
             lambda unified_id, ssid, sid: [unified_id, ssid]),
            ("MemoryOpAnalysis", memory_op_analysis,
             lambda unified_id, ssid, sid: [unified_id, ssid]),
            ("CacheOpAnalysis", cache_op_analysis,
             lambda unified_id, ssid, sid: [unified_id, ssid])
        ]

        PAPISettings.canning_print("unified_ids: " + str(unified_ids))

        # Generate all analyses for this
        for x in unified_ids:
            unified_id, = x

            unified_id = int(unified_id)

            cache_conn.execute("DROP TABLE IF EXISTS `filtered_to_nodeid`;")
            cache_conn.execute("ATTACH DATABASE " + "'" + db_path + "'" +
                               " AS 'base';")
            cache_conn.execute(
                "CREATE TABLE `filtered_to_nodeid` AS " +
                VectorizationAnalysis.query % ("") + ";", (unified_id, ))
            cache_conn.execute("DETACH DATABASE 'base';")
            cache_conn.execute(
                "CREATE INDEX IF NOT EXISTS `f2n_ind` ON `filtered_to_nodeid`(ssid, threadID);"
            )

            cache_conn.execute("DROP TABLE IF EXISTS `cache_mem_query`;")
            cache_conn.execute("ATTACH DATABASE " + "'" + db_path + "'" +
                               " AS 'base';")
            cache_conn.execute(
                "CREATE TABLE `cache_mem_query` AS " +
                MemoryAnalysis.sub_query_string.format(
                    source_table="MergedValues") % ("", "") + ";",
                (unified_id, ))
            cache_conn.execute("DETACH DATABASE 'base';")
            cache_conn.execute(
                "CREATE INDEX IF NOT EXISTS `maq_ind` ON `cache_mem_query`(ssid);"
            )

            c.execute(
                """
SELECT DISTINCT
    options, ssid, sec.SectionID
FROM
    `MergedValues` AS mv INNER JOIN
    `Entries` AS es ON mv.entryID = es.entryID INNER JOIN
    `Sections` AS sec ON sec.SectionID = es.SectionID INNER JOIN
    `SuperSections` AS ss ON sec.ssid = ss.SuperSectionID
    NATURAL JOIN `Runs`
WHERE
    sec.nodeID = ?
    AND papimode = 'default'
ORDER BY
    ssid, sec.SectionID
""", (unified_id, ))
            section_ids = c.fetchall()

            prev_analysis_key = {}
            for i, a in enumerate(analyses):
                prev_analysis_key[i] = tuple()

            cache_previous_supersection = None
            # Loop over all sections
            for y in section_ids:
                runoptions, supersection_id, section_id = y
                supersection_id = int(supersection_id)
                section_id = int(section_id)

                PAPISettings.canning_print(
                    "Now treating (%d, %d, %d, %d)" %
                    (forProgramID, unified_id, supersection_id, section_id))

                # For caching: Select the current supersection from the cache (if not already done)
                if cache_previous_supersection != supersection_id:
                    cache_previous_supersection = supersection_id

                    cache_conn.execute(
                        "DROP TABLE IF EXISTS `temp_merge_sel`;")
                    cache_conn.execute(
                        "CREATE TABLE `temp_merge_sel` AS SELECT * FROM `filtered_to_nodeid` WHERE ssid = ?;",
                        (supersection_id, ))
                    cache_conn.execute(
                        "CREATE INDEX IF NOT EXISTS `tms_ind` ON `temp_merge_sel`(sectionID, threadID, papiCode);"
                    )
                    cache_conn.execute(
                        "DROP TABLE IF EXISTS `thread_grouped`;")
                    cache_conn.execute("CREATE TABLE `thread_grouped` AS " +
                                       VectorizationAnalysis.thread_query +
                                       ";")

                # First, thread analysis
                for analysis_i, elem in enumerate(analyses):
                    name, instance, argfunc = elem

                    args = argfunc(unified_id, supersection_id, section_id)
                    if (prev_analysis_key[analysis_i] == args):
                        # If the analysis is invariant to the current key change, do not generate redundant information
                        PAPISettings.canning_print("\tSkipping analysis " +
                                                   name)
                        continue
                    else:
                        prev_analysis_key[analysis_i] = args

                    PAPISettings.canning_print("Running analysis " + name)
                    analysis_result = instance.query_values(c, *args)
                    json_data = json.dumps(analysis_result)

                    outdbc.execute(
                        """
INSERT INTO `AnalysisResults`
(forProgramID, AnalysisName, runoptions, forUnifiedID, forSuperSection, forSection, json)    
VALUES
(
    ?, ?, ?, ?, ?, ?, ?
);                
""", (forProgramID, name, runoptions, unified_id, supersection_id, section_id,
                    json_data))

        cache_conn.close()

        conn.close()
        outdb.commit()
        outdb.close()
