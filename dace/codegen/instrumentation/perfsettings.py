from dace.graph.nodes import EntryNode, ExitNode, MapEntry, MapExit, Tasklet
from dace.graph.graph import SubgraphView
from dace.memlet import Memlet
from dace.data import Array

from dace.config import Config

from dace.types import ScheduleType

import re

import sympy as sp
import os
import ast
import sqlite3

# Helper function to get the module path
if __name__ == "__main__":
    print("path: " + os.path.dirname(__file__))


class PerfSettings(object):

    _unique_counter = 0

    _perf_enable_instrumentation = True
    perf_enable_override_config = True

    default_papi_counters = [
        "PAPI_TOT_INS", "PAPI_TOT_CYC", "PAPI_L2_TCM", "PAPI_L3_TCM"
    ]

    @staticmethod
    def get_config(*key_hierarchy, config=None):
        if config == None:
            return Config.get(*key_hierarchy)
        else:
            return config.get(*key_hierarchy)

    @staticmethod
    def get_bool_config(*key_hierarchy, config=None):
        if config == None:
            return Config.get_bool(*key_hierarchy)
        else:
            return config.get_bool(*key_hierarchy)

    @staticmethod
    def perf_merging_debug_output():
        return False

    @staticmethod
    def merging_print(dbg_str):
        if PerfSettings.perf_merging_debug_output():
            print(dbg_str)

    @staticmethod
    def perf_canning_debug_output():
        return False

    @staticmethod
    def canning_print(dbg_str):
        if PerfSettings.perf_canning_debug_output():
            print(dbg_str)

    @staticmethod
    def perf_transcriptor_debug_output():
        return False

    @staticmethod
    def transcriptor_print(dbg_str):
        if PerfSettings.perf_transcriptor_debug_output():
            print(dbg_str)

    @staticmethod
    def perf_compensate_overhead(config=None):
        return PerfSettings.get_config(
            "instrumentation", "enable_overhead_compensation", config=config)

    @staticmethod
    def perf_get_thread_nums(config=None):
        return ast.literal_eval(
            PerfSettings.get_config(
                "instrumentation", "thread_nums", config=config))

    @staticmethod
    def perf_use_multimode(config=None):
        return PerfSettings.get_config(
            "instrumentation", "multimode_run", config=config)

    @staticmethod
    def perf_use_sql(config=None):
        return PerfSettings.get_config(
            "instrumentation", "sql_backend_enable", config=config)

    @staticmethod
    def get_unique_number():
        ret = PerfSettings._unique_counter
        PerfSettings._unique_counter = PerfSettings._unique_counter + 1
        return ret

    @staticmethod
    def perf_multirun_num(config=None):
        """ Amount of iterations with different PAPI configurations to run. (1 means no multirun) """
        if not PerfSettings.perf_enable_instrumentation(config):
            return 1
        return 1 + len(PerfSettings.perf_get_thread_nums(config))

    @staticmethod
    def perf_multirun_options(config=None):
        """ Specifies the options for "multirunning": running the same program
            multiple times with different performance counters. """
        ret = []

        if PerfSettings.perf_multirun_num(config) == 1:
            return ret  # Don't specify these options by default

        #for i in range(0, 4):
        #    ret.append(("omp_num_threads", i + 1))
        o = PerfSettings.perf_get_thread_nums(config)
        for x in o:
            ret.append(("omp_num_threads", x))

        ret.append(
            ("cleanrun", 0)
        )  # Add a clean run (no instrumentation) to compare performance. This should be done without OMP_NUM_THREADS

        return ret

    @staticmethod
    def perf_default_papi_counters(config=None):
        mode = PerfSettings.get_config(
            "instrumentation", "papi_mode", config=config)
        assert mode != None

        if mode == "default":  # Most general operations (Cache misses and cycle times)
            return eval(
                PerfSettings.get_config(
                    "instrumentation", "default_papi_counters", config=config))
        elif mode == "vectorize":  # Vector operations (to check if a section was vectorized or not)
            return eval(
                PerfSettings.get_config(
                    "instrumentation", "vec_papi_counters", config=config))
        elif mode == "memop":  # Static memory operations (store/load counts)
            return eval(
                PerfSettings.get_config(
                    "instrumentation", "mem_papi_counters", config=config))
        elif mode == "cacheop":  # Cache operations (PAPI_CA_*)
            return eval(
                PerfSettings.get_config(
                    "instrumentation", "cache_papi_counters", config=config))
        else:
            # Use a fallback for this one
            return eval(
                Config.get("instrumentation",
                           str(mode) + "_papi_counters"))

        return eval(
            PerfSettings.get_config(
                "instrumentation", "default_papi_counters", config=config))

    @staticmethod
    def perf_enable_timing(config=None):
        return PerfSettings.get_config(
            "instrumentation", "timeit", config=config)

    @staticmethod
    def perf_enable_instrumentation(config=None):
        return PerfSettings.get_bool_config(
            "instrumentation", "enable_papi", config=config)

    @staticmethod
    def perf_enable_instrumentation_for(sdfg, node=None):
        return PerfSettings.perf_enable_instrumentation(
        ) and not sdfg.has_instrumented_parent()

    @staticmethod
    def perf_enable_overhead_collection():
        return True  # TODO: Make config dependent (this is too expensive now because it's executed for every single run)

    @staticmethod
    def perf_current_mode(config=None):
        return PerfSettings.get_config(
            "instrumentation", "papi_mode", config=Config)

    @staticmethod
    def perf_supersection_emission_debug():
        return False

    @staticmethod
    def perf_enable_counter_sanity_check(config=None):
        return PerfSettings.get_bool_config(
            "instrumentation",
            "enable_papi_counter_sanity_check",
            config=config)

    @staticmethod
    def perf_print_instrumentation_output():
        return False

    @staticmethod
    def perf_enable_vectorization_analysis(config=None):
        return PerfSettings.get_bool_config(
            "instrumentation", "enable_vectorization_analysis", config=config)

    @staticmethod
    def perf_max_scope_depth(config=None):
        # This variable selects the maximum depth inside a scope. For example,
        # "map { map {}}" with max_scope_depth 0 will result in
        # "map { profile(map{}) }", while max_scope_depth >= 1 result in
        # "map { map { profile() }}"
        return PerfSettings.get_config(
            "instrumentation", "max_scope_depth", config=config)

    perf_debug_profile_innermost = False  # innermost = False implies outermost
    perf_debug_annotate_scopes = True
    perf_debug_annotate_memlets = False
    perf_debug_hard_error = False  # If set to true, untreated cases cause program abort.

    #TODO: There should be a variable per MAP-Element that overrides the scope depth
    perf_tasklets = False

    perf_whitelist_schedules = [
        ScheduleType.Default, ScheduleType.CPU_Multicore,
        ScheduleType.Sequential
    ]


class PerfUtils(object):
    @staticmethod
    def unified_id(node_id, state_id):
        if node_id > 0x0FFFF:
            raise ValueError("Nodeid is too larget to fit in 16 bits!")
        if state_id > 0x0FFFF:
            raise ValueError("Stateid is too large to fit in 16 bits!")
        return (int(state_id) << 16) | int(node_id)

    @staticmethod
    def get_run_options(executor, iteration):
        optdict = {}
        omp_thread_num = None
        if (PerfSettings.perf_multirun_num(config=executor._config) != 1):
            opt, val = PerfSettings.perf_multirun_options(
                config=executor._config)[iteration]
            if opt == "omp_num_threads":
                omp_thread_num = val
                if executor.running_async:
                    # Add information about what is being run
                    executor.async_host.notify("Running option threads=" +
                                               str(omp_thread_num))
            if opt == "cleanrun":
                if executor.running_async:
                    # Add information about what is being run
                    executor.async_host.notify("Running baseline")

                optdict = PerfUtils.get_cleanrun_options()

        return (optdict, omp_thread_num)

    @staticmethod
    def get_roofline_data(data_source):
        import json
        with sqlite3.connect(data_source) as conn:
            c = conn.cursor()

            c.execute("SELECT forProgramID FROM `SDFGs`;")
            ids = c.fetchall()

            print("ids are " + str(ids))

            max_thread_num = max(PerfSettings.perf_get_thread_nums())

            query = """
SELECT
forUnifiedID, COUNT(forUnifiedID)
FROM
`AnalysisResults`
WHERE
forProgramID = ?
AND AnalysisName = 'VectorizationAnalysis'
AND runoptions LIKE ?
GROUP BY
forUnifiedID
;
"""

            global_min = (1 << 63)

            # Get the values for each program
            for x in ids:
                pid, = x
                print("for pid: " + str(pid))
                c.execute(
                    query,
                    (pid, '%OMP_NUM_THREADS={thread_num}%'.format(
                        thread_num=max_thread_num)))
                d = c.fetchall()

                local_min = (1 << 63)  # Local minimum
                for y in d:
                    uid, count = y

                    local_min = min(local_min, count)

                global_min = min(global_min, local_min)

            # Now we have the minimum in global_min, representing the repetition count

            # Now we can try to get the FLOP/C and B/C from the measurements and then get the median of all repetitions
            vec_query = """
SELECT
    json
FROM
    `AnalysisResults` AS ar
WHERE
    ar.AnalysisName = '{analysis_name}'
    AND ar.forProgramID = :pid
    AND ar.forUnifiedID = :uid
    AND ar.runoptions LIKE '%OMP_NUM_THREADS={thread_num}%'
ORDER BY
    ar.forSuperSection ASC
LIMIT
    :repcount
OFFSET
    :offset
;
"""
            retvals = []
            for x in ids:
                # Loop over all programs (since we want to collect values for all programs to send to the graph)
                pid, = x

                print("running analyses for pid " + str(pid))

                c.execute(
                    "SELECT DISTINCT forUnifiedID FROM AnalysisResults WHERE forProgramID = ?",
                    (pid, ))
                uids = c.fetchall()

                offset = 0  # Start with an offset of 0
                sp_op_sums = None
                dp_op_sums = None

                proc_mem_sums = None
                in_mem_sums = None
                bytes_from_mem_sums = None

                total_critical_path = None
                for y in uids:
                    uid, = y

                    c.execute(
                        """SELECT COUNT(*) FROM AnalysisResults
                    WHERE forProgramID = ?
                    AND forUnifiedId = ?
                    AND AnalysisName='VectorizationAnalysis'
                    AND runoptions LIKE '%OMP_NUM_THREADS={thread_num}%';"""
                        .format(thread_num=max_thread_num), (pid, uid))

                    nodecount, = c.fetchall()[0]

                    # Get the critical path as well
                    c.execute(
                        """
SELECT
    json
FROM
    AnalysisResults
WHERE
    AnalysisName = 'CriticalPathAnalysis'
    AND forProgramID = :pid
    AND forUnifiedID = :uid
LIMIT
    1 -- CriticalPathAnalysis gives 1 result per unified ID.
;
                    """.format(thread_num=max_thread_num), {
                            "pid": pid,
                            "uid": uid
                        })
                    tmpfetch = c.fetchall()
                    if len(tmpfetch) == 0:
                        print("Error will occur shortly, ran for " +
                                str(pid) + ", " + str(uid))
                    cpajson, = tmpfetch[0]

                    cpa = json.loads(cpajson)
                    crit_path = cpa["critical_paths"]
                    sel_crit_path = None
                    for cp in crit_path:
                        if cp["thread_num"] == max_thread_num:
                            sel_crit_path = cp["value"]
                            break

                    assert sel_crit_path != None

                    def list_accum(inout, array_to_add):
                        if inout == None:
                            return array_to_add
                        else:
                            return list(
                                map(lambda x: x[0] + x[1],
                                    zip(inout, array_to_add)))

                    total_critical_path = list_accum(
                        total_critical_path, sel_crit_path)

                    # The count of values per repetition (as there could be more than one)
                    count_per_rep = nodecount / global_min

                    sp_ops_reps = []
                    dp_ops_reps = []

                    proc_mem_reps = []
                    in_mem_reps = []
                    bytes_from_mem_reps = []

                    offset = 0

                    while int(offset) < int(nodecount):
                        sp_op_sum = 0
                        dp_op_sum = 0

                        proc_bytes_sum = 0
                        input_bytes_sum = 0
                        bytes_from_mem_sum = 0

                        c.execute(
                            vec_query.format(
                                analysis_name="VectorizationAnalysis",
                                thread_num=max_thread_num), {
                                    "pid": pid,
                                    "uid": uid,
                                    "repcount": count_per_rep,
                                    "offset": offset
                                })
                        data = c.fetchall()

                        c.execute(
                            vec_query.format(
                                analysis_name="MemoryAnalysis",
                                thread_num=max_thread_num), {
                                    "pid": pid,
                                    "uid": uid,
                                    "repcount": count_per_rep,
                                    "offset": offset
                                })
                        memdata = c.fetchall()

                        assert len(memdata) == len(data)

                        for md in memdata:
                            js, = md
                            j = json.loads(js)
                            total_proc_data = j["datasize"]
                            input_data = j["input_datasize"]
                            bytes_from_mem = j["bytes_from_mem"]

                            proc_bytes_sum += total_proc_data
                            input_bytes_sum += input_data
                            bytes_from_mem_sum += sum(bytes_from_mem)

                        for d in data:
                            js, = d
                            j = json.loads(js)
                            # Pitfall: sp_flops is the amount of floating point operations executed, while sp_ops is the amount of instructions executed!
                            sp_ops = j['sp_flops_all']
                            dp_ops = j['dp_flops_all']

                            sp_op_sum += sp_ops
                            dp_op_sum += dp_ops

                        sp_ops_reps.append(sp_op_sum)
                        dp_ops_reps.append(dp_op_sum)

                        proc_mem_reps.append(proc_bytes_sum)
                        in_mem_reps.append(input_bytes_sum)
                        bytes_from_mem_reps.append(bytes_from_mem_sum)

                        offset += int(count_per_rep)

                    # When everything has been summed up, we can push it
                    sp_op_sums = list_accum(sp_op_sums, sp_ops_reps)
                    dp_op_sums = list_accum(dp_op_sums, dp_ops_reps)

                    proc_mem_sums = list_accum(proc_mem_sums,
                                                proc_mem_reps)
                    in_mem_sums = list_accum(in_mem_sums, in_mem_reps)
                    bytes_from_mem_sums = list_accum(
                        bytes_from_mem_sums, bytes_from_mem_reps)

                print("Finalizing for pid " + str(pid))
                # Now we can select the median
                import statistics

                # First, zip the values together
                zipped = zip(total_critical_path, sp_op_sums,
                                dp_op_sums, proc_mem_sums, in_mem_sums,
                                bytes_from_mem_sums)

                zipped = list(zipped)

                # Create the FLOP/C and Byte/C numbers, respectively for all subtypes
                sp_flop_per_cyc_arr = map(lambda x: x[1] / x[0],
                                            zipped)
                dp_flop_per_cyc_arr = map(lambda x: x[2] / x[0],
                                            zipped)

                proc_mem_per_cyc_arr = map(lambda x: x[3] / x[0],
                                            zipped)
                in_mem_per_cyc_arr = map(lambda x: x[4] / x[0], zipped)
                bytes_from_mem_per_cyc_arr = map(
                    lambda x: x[5] / x[0], zipped)

                medcyc = statistics.median(total_critical_path)
                medindex = total_critical_path.index(medcyc)

                # Select the values from medindex
                sp_flop_per_cyc = list(sp_flop_per_cyc_arr)[medindex]
                dp_flop_per_cyc = list(dp_flop_per_cyc_arr)[medindex]

                proc_mem_per_cyc = list(proc_mem_per_cyc_arr)[medindex]
                in_mem_per_cyc = list(in_mem_per_cyc_arr)[medindex]
                bytes_from_mem_per_cyc = list(
                    bytes_from_mem_per_cyc_arr)[medindex]

                d = {}
                d["ProgramID"] = pid
                d["SP_FLOP_C"] = sp_flop_per_cyc
                d["DP_FLOP_C"] = dp_flop_per_cyc
                d["FLOP_C"] = sp_flop_per_cyc + dp_flop_per_cyc

                d["PROC_B_C"] = proc_mem_per_cyc
                d["INPUT_B_C"] = in_mem_per_cyc
                d["MEM_B_C"] = bytes_from_mem_per_cyc

                # Add the return values
                retvals.append(d)

            retdict = {"msg_type": "roofline-data", "data": retvals}
            return retdict

    @staticmethod
    def get_cleanrun_options():
        optdict = {}
        optdict[
            "DACE_instrumentation_timeit"] = "1"  # Actually, should be "true"
        optdict["DACE_instrumentation_enable_papi"] = "0"  # Disable PAPI
        optdict["DACE_compiler_use_cache"] = "0"  # Force recompilation
        return optdict

    @staticmethod
    def retrieve_instrumentation_results(executor, remote_workdir):
        if PerfSettings.perf_enable_instrumentation(config=executor._config):
            if executor.running_async:
                # Add information about what is being run
                executor.async_host.notify("Analyzing performance data")
            try:
                executor.copy_file_from_remote(
                    remote_workdir + "/instrumentation_results.txt", ".")
                executor.remote_delete_file(remote_workdir +
                                            "/instrumentation_results.txt")
                content = ""
                readall = False
                with open("instrumentation_results.txt") as ir:

                    if readall:
                        content = ir.read()

                    if readall and PerfSettings.perf_print_instrumentation_output(
                    ):
                        print(
                            "vvvvvvvvvvvvv Instrumentation Results vvvvvvvvvvvvvv"
                        )
                        print(content)
                        print(
                            "^^^^^^^^^^^^^ Instrumentation Results ^^^^^^^^^^^^^^"
                        )

                    if readall:
                        PerfUtils.print_instrumentation_output(
                            content, config=executor._config)
                    else:
                        PerfUtils.print_instrumentation_output(
                            ir, config=executor._config)

                os.remove("instrumentation_results.txt")
            except FileNotFoundError:
                print("[Warning] Could not transmit instrumentation results")

            if executor.running_async:
                # Add information about what is being run
                executor.async_host.notify("Done Analyzing performance data")

    @staticmethod
    def retrieve_vectorization_report(executor, code_objects, remote_dace_dir):
        if PerfSettings.perf_enable_vectorization_analysis():
            if executor.running_async:
                executor.async_host.notify("Running vectorization check")

            executor.copy_file_from_remote(
                remote_dace_dir + "/build/vecreport.txt", ".")
            with open("vecreport.txt") as r:
                content = r.read()
                print("Vecreport:")
                print(content)

                # Now analyze this...
                for code_object in code_objects:
                    code_object.perf_meta_info.analyze(content)
            os.remove("vecreport.txt")

            if executor.running_async:
                executor.async_host.notify("vectorization check done")

    @staticmethod
    def check_performance_counters(executor):
        if PerfSettings.perf_enable_instrumentation(
        ) and PerfSettings.perf_enable_counter_sanity_check():
            if executor.running_async:
                executor.async_host.notify("Checking remote PAPI Counters")
            PerfPAPIInfoStatic.info.load_info()

            papi_counters_valid = PerfPAPIInfoStatic.info.check_counters(
                [PerfSettings.perf_default_papi_counters()])
            if (not papi_counters_valid):
                # The program might fail
                print(
                    "Stopped execution because counter settings do not meet requirements"
                )

                if executor.running_async:
                    executor.async_host.notify(
                        "An error occurred when reading remote PAPI counters")

                    executor.async_host.counter_issue()
                else:
                    return

            if executor.running_async:
                executor.async_host.notify(
                    "Done checking remote PAPI Counters")

    @staticmethod
    def dump_counter_information(executor):
        if PerfSettings.perf_enable_instrumentation():
            if executor.running_async:
                executor.async_host.notify("Reading remote PAPI Counters")
            print("Counters:\n" + str(PerfUtils.read_available_perfcounters()))
            if executor.running_async:
                executor.async_host.notify("Done reading remote PAPI Counters")

    @staticmethod
    def gather_remote_metrics(config=None):
        """ Returns a dictionary of metrics collected by instrumentation. """

        # Run the tools/membench file on remote.
        remote_workdir = PerfSettings.get_config(
            "execution", "general", "workdir", config=config)
        from diode.remote_execution import Executor
        from string import Template
        import subprocess
        executor = Executor(None, True, None)
        executor.setConfig(config)

        remote_filepath = remote_workdir + "/" + "membench.cpp"

        executor.copy_file_to_remote("tools/membench.cpp", remote_filepath)

        libs = PerfSettings.get_config(
            "compiler", "cpu", "libs", config=config).split(" ")

        libflags = map(lambda x: "-l" + x, libs)

        libflagstring = "".join(libflags)

        path_resolve_command = "python3 -m dace.codegen.instrumentation.perfsettings"
        # Get the library path
        s = Template(
            PerfSettings.get_config(
                "execution", "general", "execcmd", config=config))
        cmd = s.substitute(
            host=PerfSettings.get_config(
                "execution", "general", "host", config=config),
            command=path_resolve_command)

        p = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True)

        stdout, _ = p.communicate(timeout=60)

        remote_dace_path = re.search(r"path: (?P<dace_path>.*)", str(stdout))
        if remote_dace_path:
            remote_dace_path = remote_dace_path['dace_path']
        print("Remote dace path: %s" % remote_dace_path)

        # Now create the include path from that
        include_path = "\"" + remote_dace_path + "/" + "runtime/include" + "\""

        print("remote_workdir: " + remote_workdir)
        compile_and_run_command = "cd " + remote_workdir + " && " + " pwd && " + PerfSettings.get_config(
            "compiler", "cpu", "executable", config=config
        ) + " " + PerfSettings.get_config(
            "compiler", "cpu", "args", config=config
        ) + " " + "-fopenmp" + " " + PerfSettings.get_config(
            "compiler", "cpu", "additional_args", config=config
        ) + " -I" + include_path + " " + "membench.cpp -o membench" + " " + libflagstring + " && " + "./membench"

        # Wrap that into a custom shell because ssh will not keep context.
        # The HEREDOC is needed because we already use " and ' inside the command.
        compile_and_run_command = "<< EOF\nsh -c '" + compile_and_run_command + "'" + "\nEOF"

        print("Compile command is " + compile_and_run_command)

        # run this command
        s = Template(
            PerfSettings.get_config(
                "execution", "general", "execcmd", config=config))
        cmd = s.substitute(
            host=PerfSettings.get_config(
                "execution", "general", "host", config=config),
            command=compile_and_run_command)

        p2 = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True)

        stdout2, _ = p2.communicate(timeout=60)

        #print("stdout2: " + str(stdout2))

        bytes_per_cycle = re.search(r"result: (?P<bytes_per_cycle>.*?$)",
                                    str(stdout2))
        if bytes_per_cycle:
            bytes_per_cycle = bytes_per_cycle['bytes_per_cycle']
        print("Bytes per cycle: %s" % bytes_per_cycle)

        executor.remote_delete_file(remote_workdir + "/membench.cpp")
        executor.remote_delete_file(remote_workdir + "/membench")

        return bytes_per_cycle

    @staticmethod
    def reduce_iteration_count(begin, end, step, retparams: dict):

        from dace.symbolic import symbols_in_sympy_expr, SymExpr

        # There are different rules when expanding depending on where the expand should happen

        if isinstance(begin, int):
            start_syms = []
        else:
            start_syms = symbols_in_sympy_expr(begin)

        if isinstance(end, int):
            end_syms = []
        else:
            end_syms = symbols_in_sympy_expr(end)

        if isinstance(step, int):
            step_syms = []
        else:
            step_syms = symbols_in_sympy_expr(step)

        def intersection(lista, listb):
            return [x for x in lista if x in listb]

        start_dyn_syms = intersection(start_syms, retparams.keys())
        end_dyn_syms = intersection(end_syms, retparams.keys())
        step_dyn_syms = intersection(step_syms, retparams.keys())

        def replace_func(element, dyn_syms, retparams):
            # Resolve all symbols using the retparams-dict

            for x in dyn_syms:
                target = sp.functions.Min(
                    retparams[x] * (retparams[x] - 1) / 2, 0)
                bstr = str(element)
                element = sp.sympify(bstr, sp.abc._clash)
                element = element.subs(
                    x, target)  # Add the classic sum formula; going upwards

                # To not have hidden elements that get added again later, we also replace the values in the other itvars...
                for k, v in retparams.items():
                    newv = sp.sympify(str(v), sp.abc._clash)

                    itsyms = symbols_in_sympy_expr(newv)
                    tarsyms = symbols_in_sympy_expr(target)
                    if x in map(str, tarsyms):
                        continue
                    # assert not x in itsyms # We never want to have the replaced symbol in its own expression. This can happen when applying 2 SMs

                    tmp = newv.subs(x, target)
                    if tmp != v:
                        retparams[k] = tmp

            return element

        if len(start_dyn_syms) > 0:
            pass
            begin = replace_func(begin, start_dyn_syms, retparams)

        if len(end_dyn_syms) > 0:
            pass
            end = replace_func(end, end_dyn_syms, retparams)

        if len(step_dyn_syms) > 0:
            pass
            print("Dynamic step symbols %s!" % str(step))
            raise NotImplementedError

        return (begin, end, step)

    @staticmethod
    def get_iteration_count(mapEntry: MapEntry, vars: dict):
        """ Get the number of iterations for this map, allowing other variables as bounds. """
        from dace.symbolic import symbols_in_sympy_expr, SymExpr

        _map = mapEntry.map
        _it = _map.params

        retparams = dict()
        for k, v in vars.items():
            retparams[k] = v

        #print("Params: " + str(_it))
        for i, r in enumerate(_map.range):
            begin, end, step = r

            end = end + 1  # end is inclusive, but we want it exclusive

            if isinstance(begin, SymExpr):
                begin = begin.expr
            if isinstance(end, SymExpr):
                end = end.expr
            if isinstance(step, SymExpr):
                step = step.expr

            begin, end, step = PerfUtils.reduce_iteration_count(
                begin, end, step, retparams)
            num = (end - begin) / step  # The count of iterations
            retparams[_it[i]] = num

        return retparams

    @staticmethod
    def all_maps(mapEntry: MapEntry, dfg: SubgraphView):
        children = [
            x for x in dfg.scope_dict(True)[mapEntry]
            if isinstance(x, EntryNode)
        ]

        sub = []
        for x in children:
            sub.extend(PerfUtils.all_maps(x, dfg))

        children.extend(sub)
        #children.extend([PerfUtils.all_maps(x, dfg) for x in children])
        return children

    @staticmethod
    def perf_get_supersection_start_string(node, sdfg, dfg, unified_id):
        from dace import types
        if node.map.schedule == types.ScheduleType.CPU_Multicore:
            # We have to find out if we should mark a section start here or later.
            children = PerfUtils.all_maps(node, dfg)
            #print("children: " + str(children))
            for x in children:
                if PerfUtils.map_depth(
                        x) > PerfSettings.perf_max_scope_depth():
                    break  # We have our relevant nodes.
                if x.map.schedule == types.ScheduleType.CPU_Multicore:
                    # Nested SuperSections are not supported
                    # We have to mark the outermost section,
                    # which also means that we have to somehow tell the lower nodes
                    # to not mark the section start.
                    x.map._can_be_supersection_start = False
                elif x.map.schedule == types.ScheduleType.Sequential:
                    x.map._can_be_supersection_start = False
                else:
                    # Any other type (FPGA, GPU) - not supported by PAPI. TODO: support
                    x.map._can_be_supersection_start = False

            if PerfSettings.perf_enable_instrumentation_for(
                    sdfg, node
            ) and PerfUtils.map_depth(
                    node
            ) <= PerfSettings.perf_max_scope_depth(
            ) and node.map._can_be_supersection_start and not dfg.is_parallel(
            ):
                return "__perf_store.markSuperSectionStart(%d);\n" % unified_id
            elif PerfSettings.perf_supersection_emission_debug():
                reasons = []
                if not node.map._can_be_supersection_start:
                    reasons.append("CANNOT_BE_SS")
                if dfg.is_parallel():
                    reasons.append("CONTAINER_IS_PARALLEL")
                if PerfUtils.map_depth(
                        node) > PerfSettings.perf_max_scope_depth():
                    reasons.append("EXCEED_MAX_DEPTH")
                if not PerfSettings.perf_enable_instrumentation_for(
                        sdfg, node):
                    reasons.append("MISC")

                return "// SuperSection start not emitted. Reasons: " + ",".join(
                    reasons) + "\n"
            # dedent end
        elif PerfSettings.perf_enable_instrumentation_for(sdfg, node) and (
                PerfUtils.map_depth(
                    node) == PerfSettings.perf_max_scope_depth() or
            (PerfUtils.is_deepest_node(node, dfg))
                and PerfUtils.map_depth(node) <=
                PerfSettings.perf_max_scope_depth()
        ) and node.map._can_be_supersection_start and not dfg.is_parallel():
            # Also put this here if it is _REALLY_ safe to do so. (Basically, even if the schedule is sequential, we can serialize to keep buffer usage low)
            return "__perf_store.markSuperSectionStart(%d);\n" % unified_id

        # Otherwise, do nothing (empty string)
        return ""

    @staticmethod
    def map_depth(mapEntry: MapEntry):
        # Returns the depth of this entry node.
        # For now, the depth is stored inside the MapEntry node.
        return mapEntry._map_depth

    @staticmethod
    def set_map_depth(mapEntry: MapEntry, DFG: SubgraphView):
        from dace.graph.nodes import Reduce, AccessNode, NestedSDFG

        # Set the depth for the mapEntry

        # We do not use mapEntry for now, but it might be required for different implementations

        # Get the sorted graph
        dfg_sorted = DFG.topological_sort()
        depth = 0
        following_nodes_invalid = False  # Set to True when a fencing map is encountered
        invalid_scope = -1
        invalid_index = PerfSettings.perf_max_scope_depth() + 1
        # Iterate and get the depth for every node, breaking when the specified node has been found
        for e in dfg_sorted:
            # Set the depth for every node on the way
            if isinstance(e, EntryNode):
                if not following_nodes_invalid and not e.map.schedule in PerfSettings.perf_whitelist_schedules:
                    print(
                        "Cannot instrument node %s, as it is running on a GPU (schedule %s)"
                        % (str(mapEntry), e.map.schedule))
                    following_nodes_invalid = True  # Invalidate all following maps
                    invalid_scope = depth + 1  # Mark this depth as invalid. Once the depth drops below this threshold, the invalid-mark will be removed
                if following_nodes_invalid and depth:
                    e._map_depth = invalid_index  # Set an invalid index (this will never be instrumented)
                else:
                    e._map_depth = max(e._map_depth, depth)

                # The consume node has no concept of fencing, yet is also executed in parallel. Therefore, we needed to be defensive here.
                try:
                    if e.fence_instrumentation:
                        following_nodes_invalid = True  # After a fence there must not be any instrumentation happening
                except:
                    pass

                depth += 1
            elif isinstance(e, ExitNode):
                depth -= 1
                if depth < invalid_scope:
                    invalid_scope = -1
                    following_nodes_invalid = False
            elif isinstance(e, NestedSDFG):
                e.sdfg.set_instrumented_parent()
                #depth += 1 # Not sure if we should add a depth here

                pass
            else:
                if isinstance(e, Reduce):
                    pass
                elif isinstance(e, AccessNode):
                    pass
                elif isinstance(e, Tasklet):
                    pass
                else:
                    print("Error-Type: " + type(e).__name__)
                    assert False

    @staticmethod
    def is_deepest_node(check: MapEntry, DFG: SubgraphView):
        nodes = DFG.nodes()
        checkdepth = PerfUtils.map_depth(check)
        return all(
            not isinstance(x, MapEntry) or PerfUtils.map_depth(x) <= checkdepth
            for x in nodes)

    @staticmethod
    def instrument_entry(mapEntry: MapEntry, DFG: SubgraphView):
        # Skip consume entries please
        from dace.graph.nodes import ConsumeEntry
        if isinstance(mapEntry, ConsumeEntry):
            return False
        depth = PerfUtils.map_depth(mapEntry)
        cond1 = PerfSettings.perf_enable_instrumentation(
        ) and depth <= PerfSettings.perf_max_scope_depth() and (
            PerfUtils.is_deepest_node(mapEntry, DFG)
            or depth == PerfSettings.perf_max_scope_depth())
        cond2 = mapEntry.map.schedule in PerfSettings.perf_whitelist_schedules
        cond3 = True
        try:
            cond3 = not mapEntry.fence_instrumentation
        except:
            # Some nodes might not have this fencing, but then it's safe to ignore
            cond3 = True
        if not cond2:
            print("Cannot instrument node %s, as it is running on a GPU" %
                  str(mapEntry))
        return cond1 and cond2 and cond3

    @staticmethod
    def has_surrounding_perfcounters(node, DFG: SubgraphView):
        """ Returns true if there is a possibility that this node is part of a
            section that is profiled. """
        parent = DFG.scope_dict()[node]

        if isinstance(parent, MapEntry):
            if not parent.map.schedule in PerfSettings.perf_whitelist_schedules:
                return False
            if parent.map._has_papi_counters or PerfUtils.map_depth(
                    parent) > PerfSettings.perf_max_scope_depth():
                return True

        if PerfSettings.perf_max_scope_depth() < 0:
            return True

        return False

    @staticmethod
    def get_memlet_byte_size(sdfg, memlet: Memlet):
        pass
        memdata = sdfg.arrays[memlet.data]
        # For now, deal with arrays only
        if isinstance(memdata, Array):
            elems = [str(memdata.dtype.bytes)]
            # The following for-loop is not relevant here, it just describes the shape of the source...
            #for x in memdata.shape:
            #    elems.append(str(x))
            try:
                if (memlet.num_accesses >= 0):
                    elems.append(
                        str(memlet.num_accesses)
                    )  # num_accesses seems to be the amount of accesses per tasklet execution
                else:
                    print(
                        "Refusing to add negative accesses (%d) in get_memlet_byte_size!"
                        % memlet.num_accesses)
            except:
                print("Unsupported memlet.num_accesses type, %s (%s)" % (str(
                    type(memlet.num_accesses)), str(memlet.num_accesses)))

            return "(" + "*".join(elems) + ")"

        else:
            print("Untreated data type: ", type(memdata).__name__)
            if PerfSettings.perf_debug_hard_error:
                assert False
            else:
                return "0"

    @staticmethod
    def get_out_memlet_costs(sdfg, state_id, node, dfg):
        from dace.graph import nodes
        from dace.sdfg import ScopeSubgraphView, SDFG, scope_contains_scope
        scope_dict = sdfg.nodes()[state_id].scope_dict()

        out_costs = 0
        for edge in dfg.out_edges(node):
            _, uconn, v, _, memlet = edge
            dst_node = dfg.memlet_path(edge)[-1].dst

            # Target is neither a data nor a tasklet node
            if (isinstance(node, nodes.AccessNode)
                    and (not isinstance(dst_node, nodes.AccessNode)
                         and not isinstance(dst_node, nodes.CodeNode))):
                continue

            # Skip array->code (will be handled as a tasklet input)
            if isinstance(node, nodes.AccessNode) and isinstance(
                    v, nodes.CodeNode):
                continue

            # code->code (e.g., tasklet to tasklet)
            if isinstance(v, nodes.CodeNode):
                shared_data_name = 's%d_n%d%s_n%d%s' % (
                    state_id, dfg.node_id(edge.src), edge.src_conn,
                    dfg.node_id(edge.dst), edge.dst_conn)
                #result.write('__%s = %s;' % (shared_data_name, edge.src_conn),
                #            sdfg, state_id, [edge.src, edge.dst])
                # TODO: Check how to deal with this...
                #raise NotImplementedError
                continue

            # If the memlet is not pointing to a data node (e.g. tasklet), then
            # the tasklet will take care of the copy
            if not isinstance(dst_node, nodes.AccessNode):
                continue
            # If the memlet is pointing into an array in an inner scope, then the
            # inner scope (i.e., the output array) must handle it
            if (scope_dict[node] != scope_dict[dst_node]
                    and scope_contains_scope(scope_dict, node, dst_node)):
                continue

            # Array to tasklet (path longer than 1, handled at tasklet entry)
            if node == dst_node:
                continue

            # Tasklet -> array
            if isinstance(node, nodes.CodeNode):
                if not uconn:
                    print("This would normally raise a syntax error!")
                    return 0  # We don't error-out because the error will be raised later

                try:
                    positive_accesses = bool(memlet.num_accesses >= 0)
                except TypeError:
                    positive_accesses = False

                if memlet.subset.data_dims() == 0 and positive_accesses:

                    if memlet.wcr is not None:
                        # write_and_resolve
                        # We have to assume that every reduction costs 3 accesses of the same size
                        out_costs += 3 * sp.sympify(
                            PerfUtils.get_memlet_byte_size(sdfg, memlet),
                            sp.abc._clash)
                    else:
                        #'%s.write(%s);\n'
                        # This standard operation is already counted
                        out_costs += sp.sympify(
                            PerfUtils.get_memlet_byte_size(sdfg, memlet),
                            sp.abc._clash)
            # Dispatch array-to-array outgoing copies here
            elif isinstance(node, nodes.AccessNode):
                pass
        return out_costs

    @staticmethod
    def get_tasklet_byte_accesses(tasklet: Tasklet, dfg: SubgraphView, sdfg,
                                  state_id):
        """ Get the amount of bytes processed by `tasklet`. The formula is 
            sum(inedges * size) + sum(outedges * size) """
        in_accum = []
        out_accum = []
        in_edges = dfg.in_edges(tasklet)
        out_edges = dfg.out_edges(tasklet)

        for ie in in_edges:
            # type ie.data == Memlet
            # type ie.data.data == Data
            in_accum.append(PerfUtils.get_memlet_byte_size(sdfg, ie.data))

        out_accum.append(
            str(PerfUtils.get_out_memlet_costs(sdfg, state_id, tasklet, dfg)))

        # Merge (kept split to be able to change the behavior easily)
        full = in_accum
        full.extend(out_accum)

        return "(" + "+".join(full) + ")"

    @staticmethod
    def get_map_exit_byte_accesses(mapexit: MapExit, dfg: SubgraphView, sdfg,
                                   state_id):
        """ Get the amount of bytes processed by mapexit. The formula is 
            sum(inedges * size) + sum(outedges * size) """
        in_accum = []
        out_accum = []
        in_edges = dfg.in_edges(mapexit)
        out_edges = dfg.out_edges(mapexit)

        out_connectors = mapexit.out_connectors

        for ie in in_edges:
            # type ie.data == Memlet
            # type ie.data.data == Data
            in_accum.append(PerfUtils.get_memlet_byte_size(sdfg, ie.data))

        for oe in out_edges:
            out_accum.append(PerfUtils.get_memlet_byte_size(sdfg, oe.data))

        # Merge (kept split to be able to change the behavior easily)
        full = in_accum
        full.extend(out_accum)

        return "(" + "+".join(full) + ")"

    @staticmethod
    def get_parents(outermost_node, node, sdfg, state_id):

        parent = None
        # Because dfg is only a subgraph view, it does not contain the entry
        # node for a given entry. This O(n) solution is suboptimal
        for state in sdfg.nodes():
            s_d = state.scope_dict(node_to_children=False)
            try:
                scope = s_d[node]
            except KeyError as e:
                continue

            if (scope != None):
                parent = scope
                break
        if (parent == None):
            return []
        if (parent == outermost_node):
            return [parent]

        return PerfUtils.get_parents(outermost_node, parent, sdfg,
                                     state_id) + [parent]

    @staticmethod
    def get_memory_input_size(node, sdfg, dfg, state_id, sym2cpp):
        from dace.graph import nodes
        curr_state = sdfg.nodes()[state_id]

        data_inputs = []
        for edge in curr_state.in_edges(node):
            # Accumulate over range size and get the amount of data accessed
            num_accesses = edge.data.num_accesses

            # TODO: It might be better to just take the source object size

            try:
                bytes_per_element = sdfg.arrays[edge.data.data].dtype.bytes
                #bytes_per_element = edge.data.data.dtype.bytes
            except:
                print("Failed to get bytes_per_element")
                bytes_per_element = 0
            data_inputs.append(bytes_per_element * num_accesses)

        from functools import reduce
        import operator
        input_size = reduce(operator.add, data_inputs, 0)

        import sympy as sp
        import dace.symbolic as symbolic

        input_size = sp.sympify(input_size, sp.abc._clash)

        used_symbols = symbolic.symbols_in_sympy_expr(input_size)
        defined_symbols = sdfg.symbols_defined_at(node)
        undefined_symbols = [
            x for x in used_symbols if not (x in defined_symbols)
        ]
        if len(undefined_symbols) > 0:
            # We cannot statically determine the size at this point
            print("Failed to determine size because of undefined symbols (\"" +
                  str(undefined_symbols) + "\") in \"" + str(input_size) +
                  "\", falling back to 0")
            input_size = 0

        input_size = sym2cpp(input_size)

        return input_size

    @staticmethod
    def accumulate_byte_movements_v2(outermost_node, node, dfg: SubgraphView,
                                     sdfg, state_id):

        itvars = dict()  # initialize an empty dict

        # First, get a list of children
        if isinstance(node, MapEntry):
            children = dfg.scope_dict(node_to_children=True)[node]
        else:
            children = []
        assert not (node in children)

        # If there still are children, descend recursively (dfs is fine here)
        if len(children) > 0:
            size = 0
            for x in children:
                size = size + PerfUtils.accumulate_byte_movements_v2(
                    outermost_node, x, dfg, sdfg, state_id)

            return size
        else:
            if isinstance(node, MapExit):
                return 0  # We can ignore this.

            # If we reached the deepest node, get all parents
            parent_list = PerfUtils.get_parents(outermost_node, node, sdfg,
                                                state_id)
            #print("Parents are " + str(parent_list))
            if isinstance(node, MapEntry):
                map_list = parent_list + [node]
            else:
                #print("node is of type " + type(node).__name__)
                map_list = parent_list

            # From all iterations, get the iteration count, replacing inner
            # iteration variables with the next outer variables.
            for x in map_list:
                itvars = PerfUtils.get_iteration_count(x, itvars)

            #print("itvars: " + str(itvars))

            itcount = 1
            for x in itvars.values():
                itcount = itcount * x
            #print("Probable itcount: " + str(itcount))

            #print("constants: " + str(sdfg.constants))

            if isinstance(node, MapEntry):
                raise ValueError(
                    "Unexpected node"
                )  # A map entry should never be the innermost node
            elif isinstance(node, MapExit):
                return 0  # We can ignore this.
            elif isinstance(node, Tasklet):
                return itcount * sp.sympify(
                    PerfUtils.get_tasklet_byte_accesses(
                        node, dfg, sdfg, state_id))
            else:
                if PerfSettings.perf_debug_hard_error:
                    raise NotImplementedError
                else:
                    return 0

    @staticmethod
    def accumulate_byte_movements(node, dfg: SubgraphView, sym2cpp, sdfg,
                                  state_id):
        """ Loops over all sub-iterations and calculates the number of bytes 
            moved (logically). """

        # The coefficient consists of multipliers (i.e. maps) and bytes (i.e.
        # memlet/tasklet movements)
        coeff_this_node = ""

        if isinstance(node, MapEntry):
            # get the iteration count for this entry
            coeff_this_node = '*'.join([
                '((%s - %s) / %s)' % (sym2cpp(re + 1), sym2cpp(rb),
                                      sym2cpp(rs))
                for rb, re, rs in node.map.range
            ])

            # Create a list to contain all suboperations (for this scope)
            subops = [coeff_this_node]

            for edge in dfg.edges():
                source = dfg.scope_dict()[edge.src]
                destination = dfg.scope_dict()[edge.dst]
                if source == node and edge.dst != node:
                    subops.append(
                        PerfUtils.accumulate_byte_movements(
                            edge.dst, dfg, sym2cpp, sdfg, state_id))
                if destination == node and edge.src != node:
                    subops.append(
                        PerfUtils.accumulate_byte_movements(
                            edge.src, dfg, sym2cpp, sdfg, state_id))

            # We can just simplify that directly
            if any(x == "0" for x in subops):
                return "0"
            coeff_this_node = ' * '.join([x for x in subops if x != ""])
            return coeff_this_node
        elif isinstance(node, MapExit):
            # Ignore this type, we already dealt with it when we processed
            # MapEntry
            return ""
        elif isinstance(node, Tasklet):
            # Exact data movement costs depend on the tasklet code
            return PerfUtils.get_tasklet_byte_accesses(node, dfg, sdfg,
                                                       state_id)

        else:
            if PerfSettings.perf_debug_hard_error:
                raise NotImplementedError
            else:
                return "0"

    class ParseStates:
        CONTROL = 0
        VALUES = 1
        SECTION_SIZE = 2

    class Entry:
        def __init__(self):
            pass
            self.values = {}
            self.nodeid = 0
            self.coreid = 0
            self.iteration = 0
            self.flags = 0

        def is_valid(self):
            return len(self.values) != 0

        def add(self, counter, value):
            self.values[counter] = value

        def get(self, name: str):
            try:
                return self.values[name]
            except:
                return None

        def toJSON(self):
            return '{{ "node": "{node}",\n"thread": "{thread}",\n"iteration": "{iteration}",\n"flags": {flags},\n"values": [{values}]\n}}\n'.format(
                node=str(self.nodeid),
                thread=str(self.coreid),
                iteration=str(self.iteration),
                flags=str(self.flags),
                values=", ".join([
                    '{{ "{code}": {value} }}'.format(
                        code=str(code), value=str(value))
                    for code, value in self.values.items()
                ]))

        def toSQL(self, c: sqlite3.Cursor, section_id, order):
            # section_id must be set by the calling section and represent
            # its DB key, order is an integer that must be monotonically
            # increasing
            c.execute(
                '''INSERT INTO `Entries`
                         (SectionID, `order`, threadID, iteration, flags)
                         VALUES
                         (?, ?, ?, ?, ?);
            ''', (section_id, order, self.coreid, self.iteration, self.flags))
            order += 1

            entry_id = c.lastrowid

            # Now, the values have to be inserted, based on the last entry id
            def mapper(x):
                k, v = x
                return (int(k), int(v), entry_id)

            vals = list(map(mapper, self.values.items()))
            # Then execute this in bulk
            c.executemany(
                '''INSERT INTO `Values`
                    (PapiCode, Value, entryID)
                    VALUES
                    (?, ?, ?)
                ''', vals)

            return order  # Do not commit every entry (needs write-back)

        def toCSVsubstring(self, delim=','):
            return delim.join([
                self.nodeid, self.coreid, self.iteration,
                *self.values.values()
            ])  # * == ... in other languages

    class Section:
        def __init__(self, nodeid=0, threadid=0):
            pass
            self.entries = []
            self.nodeid = nodeid
            self.datasize = 0
            self.input_datasize = 0
            self.bytes_moved = 0
            self.was_collapsed = False
            self.threadid = threadid

        def is_complete(self):
            """ Checks if all iterations are in this section. This might not 
                always be the case, e.g. in filtered sections. """
            itlist = [int(x.iteration) for x in self.entries]
            sortitlist = sorted(itlist)
            for i, e in enumerate(sortitlist):
                if (i != int(e)):
                    print("list: %s\n" % sortitlist)
                    return False
            return True

        def is_valid(self):
            return len(self.entries) != 0

        def add(self, e):
            self.entries.append(e)

        def addSection(self, sec):
            """ Merges another section into this section. """
            assert self.nodeid == sec.nodeid

            # We allow collapsing at most once.
            if self.was_collapsed:
                return
            if sec.was_collapsed:
                return
            # Add all entries
            for x in sec.entries:
                self.add(x)

            # merge meta
            #self.datasize += sec.datasize
            self.bytes_moved += sec.bytes_moved
            self.was_collapsed = True
            sec.was_collapsed = True

        def select_event(self, event: str):
            """ Selects all values of 'event' in correct order from all 
                entries. """
            return [
                int(x.get(event)) for x in self.entries if x.get(event) != None
            ]

        def select_thread(self, thread: int):
            """ Returns a section that only contains entries of `self` that 
                were obtained in the given thread. """
            ret = PerfUtils.Section(self.nodeid)

            for x in self.entries:
                if int(x.coreid) == int(thread):
                    ret.entries.append(x)

            return ret

        def select_node(self, node: int):
            """ Returns a section that only contains entries of `self` that 
                were obtained for the given node """
            ret = PerfUtils.Section(self.nodeid)

            for x in self.entries:
                if int(x.nodeid) == int(node):
                    ret.entries.append(x)

            return ret

        def filter(self, predicate):
            """ Returns a section that only contains entries `e` for which 
                `predicate(e)` returns true"""
            ret = PerfUtils.Section(self.nodeid)

            for x in self.entries:
                if predicate(x):
                    ret.entries.append(x)

            return ret

        def get_max_thread_num(self):
            """ Returns the maximal thread number in at most O(n) 
                complexity. """
            max = 0
            for x in self.entries:
                if int(x.coreid) > max:
                    max = int(x.coreid)
            return max

        def toCSVsubstring(self, prepend="", delim=',', linedelim='\n'):
            ret = ""
            for x in self.entries:
                ret += delim.join([
                    prepend, "node" + self.nodeid, self.threadid,
                    x.toCSVsubstring(delim)
                ]) + linedelim
            return ret

        def toJSON(self):
            return '{{ "entry_node": {entry_node}, "static_movement": {datasize}, "input_size": {input_size}, "entry_core": {core}, "entries": ['.format(
                entry_node=self.nodeid,
                datasize=self.datasize,
                input_size=self.input_datasize,
                core=self.threadid) + ", ".join(
                    [x.toJSON() for x in self.entries]) + "]}"

        def toSQL(self, conn: sqlite3.Connection, c: sqlite3.Cursor,
                  supersection_id, order):
            # section_id must be set by the calling section and represent its
            # DB key, order is an integer that must be monotonically increasing
            c.execute(
                '''INSERT INTO `Sections`
                         (ssid, `order`, nodeID, datasize, input_datasize)
                         VALUES
                         (?, ?, ?, ?, ?);
            ''', (supersection_id, order, self.nodeid, self.datasize,
                  self.input_datasize))
            order += 1

            section_id = c.lastrowid

            # Add all entries
            entry_order = 0
            for x in self.entries:
                x.toSQL(c, section_id, entry_order)
                entry_order += 1

            # Don't flush here yet - sqlite is slow on write

            return order

    class SuperSection:
        """ Contains multiple Sections. 
            @see Section
        """

        def __init__(self, supernode=0):
            self.sections = {}
            self.supernode = supernode

        def is_valid(self):
            return len(self.sections.values()) > 0 or self.supernode == -1

        def addSection(self, section):
            if int(section.threadid) in self.sections:
                self.sections[int(section.threadid)].append(section)
            else:
                self.sections[int(section.threadid)] = [section]

        def addEntry(self, entry):

            if not entry.is_valid():
                # ignore invalid entries
                return

            # We have 2 cases - either:
            # (a) the section starts outside of a parallel block:
            #   Every entry needs to be assigned to this block. There will only
            #   be one block with threadid == 0 in this case.
            # or (b) the section starts in a parallel block:
            #   Entries can be assigned by thread_id.
            if int(entry.coreid) in self.sections:
                # Assign by thread id
                try:
                    self.sections[int(entry.coreid)][-1].add(entry)
                except:
                    print("Sections has keys " + str(self.sections.keys()))
                    raise
            else:
                # Ideally, we can only add nodes to a section if they have the
                # same core id. However, in nested omp constructs, the
                # lower-level sections are usually just run on core 0.
                # So if a section starts on core 1, its entries might still
                # report core 0.
                try:
                    self.sections[0][-1].add(entry)
                except Exception as e:
                    print("error, contained sections:")
                    print(str(self.sections))
                    print(str(self.sections.values()))

                    mitigated = False
                    # Find the section that matches by nodeid...
                    for x in self.sections.values():
                        # Find the correct section and append to that
                        # (start with oldest entry)
                        for y in reversed(x):
                            if y.nodeid == entry.nodeid:
                                y.add(entry)
                                print(
                                    "Warning: Mitigation successful, but you should probably enable OMP_NESTED"
                                )
                                mitigated = True
                                break

                    if not mitigated:  # Only complain if we could not mitigate
                        raise e

        def getSections(self):
            l = []
            for x in self.sections.values():
                l.extend(x)
            return [x for x in l]

        def toCSVstring(self, delim=',', linedelim='\n'):
            """ Create a CSV string from the data. """

            # Squashes everything into a row, duplicating data.
            ret = ""
            for x in self.sections.values():
                for y in x:
                    ret += y.toCSVsubstring("supernode" + str(self.supernode),
                                            delim, linedelim)
            ret += "ENDSUPERSECTION" + linedelim
            return ret

        def toJSON(self):
            return '{{ "hint": "supersection", "supernode": {supernode},\n "sections": [{sections}] }}'.format(
                supernode=self.supernode,
                sections=",\n".join([x.toJSON() for x in self.getSections()]))

        def toSQL(self, conn: sqlite3.Connection, c: sqlite3.Cursor, run_id,
                  order):
            """ Create a supersection entry for a given run_id """
            c.execute(
                '''
            INSERT INTO `SuperSections`
                (runid, `order`, nodeID)
            VALUES
                (?, ?, ?);
            ''', (run_id, order, self.supernode))

            ssid = c.lastrowid
            order += 1

            section_order = 0

            for x in self.getSections():
                x.toSQL(conn, c, ssid, section_order)
                section_order += 1

            return order

    @staticmethod
    def perf_counter_store_string(counterlist: [str]):
        """ Creates a performance counter typename string. """
        return "PAPIValueStore<" + ", ".join(counterlist) + ">"

    @staticmethod
    def perf_counter_string_from_string_list(counterlist: [str]):
        """ Creates a performance counter typename string. """
        if isinstance(counterlist, str):
            print("Wrong format")
            counterlist = eval(counterlist)
        return "PAPIPerfLowLevel<" + ", ".join(counterlist) + ">"

    @staticmethod
    def perf_counter_string(node, config=None):
        """
        Creates a performance counter typename string.
        """

        mode = PerfSettings.get_config(
            "instrumentation", "papi_mode", config=config)
        if mode == "default":  # Only allow overriding in default mode
            try:
                assert isinstance(node.papi_counters, list)
                return PerfUtils.perf_counter_string_from_string_list(
                    node.papi_counters)
            except Exception as e:
                pass

        return PerfUtils.perf_counter_string_from_string_list(
            PerfSettings.perf_default_papi_counters())

    @staticmethod
    def perf_counter_start_measurement_string(node,
                                              unified_id,
                                              iteration,
                                              core_str="PAPI_thread_id()"):
        pcs = PerfUtils.perf_counter_string(node)
        return (
            'dace_perf::{counter_str} __perf_{id};\n' +
            'auto& __vs_{id} = __perf_store.getNewValueSet(__perf_{id}, {id}, {core}, {it});\n'
            + '__perf_{id}.enterCritical();\n').format(
                counter_str=pcs, id=unified_id, it=iteration, core=core_str)

    @staticmethod
    def perf_counter_end_measurement_string(unified_id):
        return '__perf_{id}.leaveCritical(__vs_{id});\n'.format(id=unified_id)

    @staticmethod
    def perf_section_start_string(unified_id,
                                  size,
                                  in_size,
                                  core_str="PAPI_thread_id()"):
        return (
            "__perf_store.markSectionStart(%d, (long long)%s, (long long)%s, %s);\n"
            % (unified_id, str(size), str(in_size), core_str))

    @staticmethod
    def perf_supersection_start_string(unified_id):
        return ("__perf_store.markSuperSectionStart(%d);\n" % (unified_id))

    @staticmethod
    def read_available_perfcounters(config=None):
        from string import Template
        import subprocess

        papi_avail_str = "papi_avail -a"
        s = Template(
            PerfSettings.get_config(
                "execution", "general", "execcmd", config=config))
        cmd = s.substitute(
            host=PerfSettings.get_config(
                "execution", "general", "host", config=config),
            command=papi_avail_str)
        p = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True)

        stdout, _ = p.communicate(timeout=60)

        counter_num = re.search(
            r"Number Hardware Counters[\s.]*:\s(?P<num_cntr>[0-9]+)",
            str(stdout))
        if counter_num:
            counter_num = int(counter_num['num_cntr'])
        print("Hardware counters: %s" % counter_num)

        print("PAPI preset events:")
        # Find non-derived events first
        non_derived = re.findall(
            r"(?P<event_name>PAPI_[0-9A-Z_]+)\s+0x[0-9a-zA-Z]+\s+No",
            str(stdout))
        print("Non-Derived: ", non_derived)

        # Now all derived events
        derived = re.findall(
            r"(?P<event_name>PAPI_[0-9A-Z_]+)\s+0x[0-9a-zA-Z]+\s+Yes",
            str(stdout))
        print("Derived: ", derived)

        return (non_derived, derived, counter_num)

    @staticmethod
    def collapse_sections(sections: list):
        """ Combine sections with the same ID into one single section. """

        seen = []  # Nodeids that were already collapsed
        collapsed = [
        ]  # The return value, consisting of all collapsed sections

        # Add all elements that were already collapsed
        collapsed = [x for x in sections if x.was_collapsed]

        print("%d sections were already collapsed" % len(collapsed))

        for _ in sections:
            preselection = [
                x for x in sections
                if not (x.nodeid, x.threadid) in seen and not x.was_collapsed
            ]
            if preselection == []:
                break
            target = preselection[0]
            seen.append((target.nodeid, target.threadid))
            selection = [
                x for x in sections
                if x.nodeid == target.nodeid and x.threadid == target.threadid
                and x != target and not x.was_collapsed
            ]
            for y in selection:
                target.addSection(y)
            collapsed.append(target)

            target.was_collapsed = True  # If selection is []

            assert target.was_collapsed

        # Debug
        removed_nodes = [x for x in sections if not (x in collapsed)]
        print("Removed nodes: " + str([x.toJSON() for x in removed_nodes]))
        print(
            "Reduced from %d sections to %d" % (len(sections), len(collapsed)))
        return collapsed

    @staticmethod
    def print_instrumentation_output(data: str, config=None):
        import json
        PerfSettings.transcriptor_print("print_instrumentation_output start")
        # Regex for Section start + bytes: # Section start \(node (?P<section_start_node>[0-9]+)\)\nbytes: (?P<section_start_bytes>[0-9]+)
        # Regex for general entries: # entry \((?P<entry_node>[0-9]+), (?P<entry_thread>[0-9]+), (?P<entry_iteration>[0-9]+), (?P<entry_flags>[0-9]+)\)\n((?P<value_key>[0-9-]+): (?P<value_val>[0-9-]+)\n)*

        if config == None:
            raise Exception("config is forbidden to be None for testing")
        else:
            print("Output file: " +
                  config.get("instrumentation", "sql_database_file"))
        print_values = False

        multirun_results = []
        multirun_supersections = []
        current_multirun_line = ""
        sections = []

        supersections = []
        current_supersection = PerfUtils.SuperSection()
        current_section = PerfUtils.Section()
        current_entry = PerfUtils.Entry()

        execution_times = [
        ]  # List of execution times, where the last one is a clean execution (no instrumentation other than a simple timer)

        state = PerfUtils.ParseStates.CONTROL
        if isinstance(data, str):
            lines = data.split('\n')
            is_string_input = True
        else:
            lines = data
            is_string_input = False

        overhead_values = False  # By default, we are not looking for overhead values, but real measurements
        overhead_values_array = []

        line_num = 0
        for line in lines:
            line_num = line_num + 1
            if not is_string_input:
                line = line[:-1]  # Remove trailing newline

            if "multirun" in line:
                # Multirun result

                try:
                    current_supersection.addEntry(current_entry)
                except Exception as e:
                    print("Error occurred in line " + str(line_num) + "!")
                    raise e

                # Reset variables
                current_section = PerfUtils.Section()
                current_entry = PerfUtils.Entry()

                sections.extend(current_supersection.getSections())
                supersections.append(current_supersection)

                current_supersection = PerfUtils.SuperSection()

                if current_multirun_line != "" and sections != []:
                    multirun_results.append((current_multirun_line.replace(
                        "\n", ""), sections))
                if current_multirun_line != "" and supersections != []:
                    multirun_supersections.append(
                        (current_multirun_line.replace("\n", ""),
                         supersections))

                current_multirun_line = line
                sections = []
                supersections = []
                continue
            if len(line) == 0:
                continue
            if line[0] == '#':
                state = PerfUtils.ParseStates.CONTROL
                overhead_values = False  # Reset the overhead flag
            if state == PerfUtils.ParseStates.CONTROL:
                # First try: Entry
                match = re.search(
                    r"# entry \((?P<entry_node>[0-9]+), (?P<entry_thread>[0-9]+), (?P<entry_iteration>[0-9]+), (?P<entry_flags>[0-9]+)\)",
                    line)
                if match:
                    d = match.groupdict()

                    try:
                        current_supersection.addEntry(current_entry)
                    except Exception as e:
                        print("Error occurred in line " + str(line_num) + "!")
                        raise e

                    current_entry = PerfUtils.Entry()

                    current_entry.nodeid = d['entry_node']
                    current_entry.coreid = d['entry_thread']
                    current_entry.iteration = d['entry_iteration']
                    current_entry.flags = d['entry_flags']
                    state = PerfUtils.ParseStates.VALUES
                    continue

                # Next try: Section header
                match = re.search(
                    r"# Section start \(node (?P<section_start_node>[0-9]+), core (?P<section_start_core>[0-9]+)\)",
                    line)
                if match:
                    d = match.groupdict()

                    try:
                        current_supersection.addEntry(current_entry)
                    except Exception as e:
                        print("Error occurred in line " + str(line_num) + "!")
                        raise e

                    current_entry = PerfUtils.Entry()
                    current_section = PerfUtils.Section(
                        d['section_start_node'], d['section_start_core'])
                    current_supersection.addSection(current_section)
                    state = PerfUtils.ParseStates.SECTION_SIZE
                    continue
                # Next try: Supersection header
                match = re.search(
                    r"# Supersection start \(node (?P<section_start_node>[0-9]+)\)",
                    line)
                if match:
                    d = match.groupdict()

                    supersection_node_id = d['section_start_node']

                    try:
                        current_supersection.addEntry(current_entry)
                    except Exception as e:
                        print("Error occurred in line " + str(line_num) + "!")
                        raise e
                    current_entry = PerfUtils.Entry()

                    if (current_section.is_valid()):
                        #sections.append(current_section)
                        pass

                    sections.extend(current_supersection.getSections())

                    supersections.append(current_supersection)
                    current_supersection = PerfUtils.SuperSection(
                        d['section_start_node'])

                    current_section = PerfUtils.Section()  # Clear the record

                    state = PerfUtils.ParseStates.CONTROL
                    continue
                # Next try: Section data moved
                match = re.search(r"# moved_bytes: (?P<moved_bytes>[0-9]+)",
                                  line)
                if match:
                    d = match.groupdict()
                    current_section.bytes_moved = d['moved_bytes']
                    continue
                # Next try: Section data moved
                match = re.search(r"# contention: (?P<contention>[0-9]+)",
                                  line)
                if match:
                    d = match.groupdict()
                    if int(d['contention']) != 0:
                        print(
                            "Contention: {cont}".format(cont=d['contention']))
                    continue

                # Next try: Timer
                match = re.search(
                    r"# Timer recorded (?P<time>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?) secs",
                    line)
                if match:
                    d = match.groupdict()
                    time = float(d['time'])
                    PerfSettings.transcriptor_print("Time taken is " +
                                                    str(time))
                    execution_times.append(time)
                    continue

                # Next try: Overhead
                match = re.search(r"# Overhead comp", line)
                if match:
                    # We have to switch to overhead value mode. To keep it simple, we just set a flag
                    overhead_values = True
                    state = PerfUtils.ParseStates.VALUES
                    continue

                # Next try: Entry (anonymous)
                # (Should not happen)
                print("Error, unexpected: anonymous entry %s" % line)
                print(str(match))
            elif state == PerfUtils.ParseStates.VALUES:
                match = re.search(r"(?P<counter>[0-9-]+): (?P<value>[0-9-]+)",
                                  line)
                if match:
                    d = match.groupdict()
                    if overhead_values:
                        overhead_values_array.append((d['counter'],
                                                      d['value']))
                    else:
                        current_entry.add(d['counter'], d['value'])
                else:
                    print("Failed to match expected values! " + str(line))
                continue
            elif state == PerfUtils.ParseStates.SECTION_SIZE:
                match = re.search(r"^bytes: (?P<bytes>[0-9-]+)", line)
                if match:
                    d = match.groupdict()
                    current_section.datasize = d['bytes']
                else:
                    match = re.search(r"^input_bytes: (?P<ibytes>[0-9-]+)",
                                      line)
                    if match:
                        d = match.groupdict()
                        current_section.input_datasize = d['ibytes']
                        continue
                    else:
                        pass
                continue

        try:
            current_supersection.addEntry(current_entry)
        except Exception as e:
            print("Error occurred in line " + str(line_num) + "!")
            raise e

        sections.extend(current_supersection.getSections())
        supersections.append(current_supersection)
        multirun_results.append((current_multirun_line, sections))
        multirun_supersections.append((current_multirun_line, supersections))

        PerfSettings.transcriptor_print("Execution times: " +
                                        str(len(execution_times)) + ": " +
                                        str(execution_times))
        PerfSettings.transcriptor_print("Multirun length: " +
                                        str(len(multirun_results)))

        reps = int(
            PerfSettings.get_config(
                "execution", "general", "repetitions", config=config))
        clean_times = execution_times[-reps:]
        inst_times = execution_times[-int(2 * reps):][:reps]

        PerfSettings.transcriptor_print("clean_times: " + str(clean_times))
        PerfSettings.transcriptor_print("inst_times: " + str(inst_times))
        percent_diff = []
        for i in range(0, reps):
            percent_diff.append(
                ((inst_times[i] / clean_times[i]) - 1.0) * 100.)

        PerfSettings.transcriptor_print("percent_diff: " + str(percent_diff))

        # Reduce overhead numbers (In general, the safest way is minimum)
        overhead_dict = {}
        for k, v in overhead_values_array:
            if not k in overhead_dict:
                overhead_dict[k] = int(v)
            overhead_dict[k] = min(int(v), overhead_dict[k])

        PerfSettings.transcriptor_print("Overhead numbers: " +
                                        str(overhead_dict))

        # Gimme a JSON string for the overhead numbers...
        overhead_number_string = ""
        if len(overhead_values_array) > 0:
            overhead_number_string = ', "overhead_numbers": ' + str(
                overhead_dict)
            overhead_number_string = overhead_number_string.replace("'", '"')

        modestr = str(
            PerfSettings.get_config(
                "instrumentation", "papi_mode", config=config))
        for o, s in multirun_results:
            try:
                PerfSettings.transcriptor_print("\tSection size: " +
                                                str(len(s)))
                PerfSettings.transcriptor_print("\t\tSection size: " +
                                                str(s[0].datasize))
            except:
                pass

        if PerfSettings.perf_use_sql():
            import sqlite3
            dbpath = PerfSettings.get_config(
                "instrumentation", "sql_database_file", config=config)
            conn = sqlite3.Connection(dbpath)
            c = conn.cursor()

            c.execute(''' SELECT * FROM `Programs` LIMIT 1''')
            f = c.fetchall()
            program_id = 1
            if len(f) == 0:
                # No program yet
                c.execute(
                    '''INSERT INTO `Programs`
                    (programHash)
                    VALUES
                    (?);
        ''', ("MyProgram", ))
                program_id = c.lastrowid  # Set the correct id
            else:
                pass  # Otherwise just reuse the old program ID

            conn.commit()

            for o, r_supersections in multirun_supersections:

                # Create an entry for the run
                c.execute(
                    '''
                INSERT INTO `Runs`
                    (program, papimode, options)
                VALUES
                    (?, ?, ?);
    ''', (program_id, modestr, o))

                run_id = c.lastrowid
                order = 0

                # Associate and write the supersections
                for x in r_supersections:
                    if x.is_valid():
                        order = x.toSQL(conn, c, run_id, order)

            # Set the overhead numbers
            for x in overhead_dict.items():
                pc, val = x
                c.execute(
                    '''
INSERT INTO `Overheads`
    (programID, papiCode, value)
VALUES
    (?, ?, ?)
;
''', (program_id, pc, float(val)))

            # Writeback and close
            conn.commit()
            conn.execute("PRAGMA optimize;")
            conn.close()
        # endif PerfSettings.perf_use_sql()

        try:
            totstr = '{ "type": "PerfInfo", "payload": [' + ", ".join(
                [
                    '{"runopts": "%s", "data": [%s]}' % (o, ", ".join(
                        [x.toJSON() for x in r_supersections if x.is_valid()]))
                    for o, r_supersections in multirun_supersections
                ]
            ) + '], "overhead_percentage": %s, "mode": "%s", "default_depth": %d %s}' % (
                str(percent_diff), modestr,
                PerfSettings.perf_max_scope_depth(), overhead_number_string)

            if False: # Disable debug json and csv by default
                with open("perf_%s.json" % modestr, "w") as out:
                    out.write(totstr)

                # Debug CSV output
                for idx, v in enumerate(multirun_supersections):
                    o, r_supersections = v
                    with open("perf%d.csv" % idx, "w") as out:
                        for x in r_supersections:
                            out.write(x.toCSVstring())

        except:
            import traceback
            print("[Error] Failed to jsonify")
            print(traceback.format_exc())

        # Check if this runs
        try:
            for s in sections:
                json.loads(s.toJSON())
        except:
            print("[Error] JSON contains syntax errors!")

        PerfSettings.transcriptor_print(
            "\t===>Print instrumentation output end")

        if print_values:
            print("==== ANALYSIS ====")
            print("Got %d sections" % len(sections))
            for i, section in enumerate(sections):
                print("Section %d (node %s)" % (i, section.nodeid))
                print("static memory movement (estimation): %s" % str(
                    section.datasize))
                print("runtime memory movement (measured):  %s" % str(
                    section.bytes_moved))

                max_thread_num = section.get_max_thread_num()
                print("max_thread_num: %d" % max_thread_num)
                tot_cyc = list()
                tot_l3_miss = list()
                tot_l2_miss = list()
                for t in range(0, max_thread_num + 1):
                    ts = section.select_thread(t)
                    tc = ts.select_event('-2147483589')
                    # print("tc: %s\nsum(tc): %s" % (str(tc), str(sum(tc))))
                    tot_cyc.append(sum(tc))

                    tl3 = ts.select_event('-2147483640')
                    tot_l3_miss.append(sum(tl3))

                    tl2 = ts.select_event('-2147483641')
                    tot_l2_miss.append(sum(tl2))

                # Now we can get the balance
                for i, t in enumerate(tot_cyc):
                    print("Thread %d took %d cycles" % (i, t))
                from statistics import stdev, mean
                if len(tot_cyc) > 1 and mean(tot_cyc) != 0:

                    print("stdev: %d" % stdev(tot_cyc))
                    print("Balance: %f" %
                          (float(stdev(tot_cyc)) / float(mean(tot_cyc))))

                for i, t in enumerate(tot_l3_miss):
                    print("Thread %d had %d L3 misses" % (i, t))
                sum_l3 = sum(tot_l3_miss)
                print(
                    "%d bytes (presumably) accessed\n%d L3 misses over all threads\n%d bytes loaded from memory"
                    % (int(section.datasize), int(sum_l3), int(sum_l3) * 64))

                for i, t in enumerate(tot_l2_miss):
                    print("Thread %d had %d L2 misses" % (i, t))
                sum_l2 = sum(tot_l2_miss)
                print(
                    "%d bytes (presumably) accessed\n%d L2 misses over all threads\n%d bytes loaded from L3"
                    % (int(section.datasize), int(sum_l2), int(sum_l2) * 64))


class PAPIUtil:
    @staticmethod
    def fallback_dict(available_events):
        """
        Defines potential fallbacks for unavailable PAPI (preset) events
        """
        d = dict()
        #TCM => DCM
        d['PAPI_L1_TCM'] = [
            x for x in ['PAPI_L1_DCM'] if x in available_events
        ]
        d['PAPI_L2_TCM'] = [
            x for x in ['PAPI_L2_DCM'] if x in available_events
        ]
        d['PAPI_L3_TCM'] = [
            x for x in ['PAPI_L3_DCM'] if x in available_events
        ]
        #DCM => TCM
        d['PAPI_L1_DCM'] = [
            x for x in ['PAPI_L1_TCM'] if x in available_events
        ]
        d['PAPI_L2_DCM'] = [
            x for x in ['PAPI_L2_TCM'] if x in available_events
        ]
        d['PAPI_L3_DCM'] = [
            x for x in ['PAPI_L3_TCM'] if x in available_events
        ]

        return d

    @staticmethod
    def get_fallback(event, available_events):
        """
        Returns a string identifying the most appropriate fallback for 'event',
        or None if no such fallback exists.
        """
        fbd = PAPIUtil.fallback_dict(available_events)
        fb = fbd[event]
        if (len(fb) == 0):
            return None
        else:
            return fb[0]


class PerfMetaInfo:
    """ Class dedicated to keep meta information about the generated code, in 
        particular line numbers. """

    def __init__(self):
        self.nodes = dict()  # Maps nodes to their strings
        self.lines = dict()  # Maps nodes to their line number

    def add_node(self, node, string):
        self.nodes[node] = string

    def has_node(self, node):
        return node in self.nodes.keys()

    def resolve(self, codestr: str):
        """ Maps all entries in self.node to line numbers """
        index = 0
        line = 1
        print("self.nodes: %s\ntype: %s" % (self.nodes, type(self.nodes)))
        for key, value in self.nodes.items():
            pos = codestr.find(value, index)
            if pos == -1:
                # We will not accept this. This should only ever occur if some
                # part of the program pretty-prints code.
                assert False
            sublines = codestr.count('\n', index, pos)
            line += sublines
            index = pos
            # We store the current line back to self.lines
            self.lines[key] = line

    def analyze(self, vectorizer_output: str):
        """ Checks if a certain operation or a segment within a region of an 
            operation was vectorized. """
        # We only match calls originating from ./src/cpu/*, but it might still
        # include some of the instrumentation. Consider running this on
        # non-instrumented code instead
        data = re.findall(
            r".*?src/cpu/(?P<file>[^:]*):(?P<line>[\d]*):(?P<col>[\d]*): (?P<msg>[^\n]*)",
            vectorizer_output)

        print("data is:\n%s" % data)

        print("Node information is\n%s\n" % self.nodes)
        print("Line information is\n%s\n" % self.lines)

        ret = dict(
        )  # We return a dict of node -> [(file, line, col, Message)]

        first = True
        tmp = (None, None)
        for key, value in self.lines.items():
            # We now find for each key the value of their respective start
            # (exception: MapExit, where the end counts)
            # Then, we associate the message to that key
            if not first:
                prevkey, prevval = tmp
                for file, line, col, message in data:
                    if int(prevval) <= int(line) and int(line) < int(value):
                        # Valid entry
                        if not (prevkey in ret.keys()):
                            ret[prevkey] = list()
                        ret[prevkey].append((file, line, col, message))
            else:
                first = False

            tmp = (key, value)

        # For the last entry:
        prevkey, prevval = tmp
        if prevkey != None:
            for file, line, col, message in data:
                if int(prevval) <= int(line):
                    # Valid entry
                    if not (prevkey in ret.keys()):
                        ret[prevkey] = list()
                    ret[prevkey].append((file, line, col, message))

        print("ret:\n%s" % ret)

        return ret


class PerfMetaInfoStatic:
    info = PerfMetaInfo()


class PerfPAPIInfo:
    """ Class used to keep information about the remote, most notably the 
        allowed configurations. """

    def __init__(self):
        self.num_hw_counters = -1
        self.preset_cost = dict()  # event: str -> num_counters: int
        self.cached_host = ""
        self.memspeed = 20.0  # B/c

    def set_memspeed(self, speed):
        self.memspeed = speed

    def load_info(self, config=None):
        """ Load information about the counters from remote. """
        from string import Template
        import subprocess

        print("Loading counter info from remote...")

        if self.cached_host == PerfSettings.get_config(
                "execution", "general", "host", config=config):
            return  # Do not run this every time, just the first time
        else:
            # else reset
            self.num_hw_counters = -1
            self.preset_cost = dict()

        non_derived, derived, num_ctrs = PerfUtils.read_available_perfcounters(
        )
        self.num_hw_counters = num_ctrs

        # Having these events, the non_derived (by definition) use 1 counter
        for x in non_derived:
            self.preset_cost[x] = 1

        # For the others, we have to request some more information.
        # NOTE: This could be moved into a shell script and run on remote
        # if issuing many commands is too slow
        for index, x in enumerate(derived):
            print("%d/%d Elements...\r" % (index + 1, len(derived)), end='')
            papi_avail_str = 'papi_avail -e %s | grep --color=never "Number of Native Events"' % x
            s = Template(
                PerfSettings.get_config(
                    "execution", "general", "execcmd", config=config))
            cmd = s.substitute(
                host=PerfSettings.get_config(
                    "execution", "general", "host", config=config),
                command=papi_avail_str)
            p = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True)

            stdout, _ = p.communicate(timeout=60)

            counter_num_grp = re.search(
                r"Number of Native Events:\s*(?P<num>\d+)", str(stdout))
            if counter_num_grp != None:
                self.preset_cost[x] = int(counter_num_grp['num'])
            else:
                print("\nError: Expected to find a number here...")

        self.cached_host = PerfSettings.get_config(
            "execution", "general", "host", config=config)
        print("\nDone")

    def check_counters(self, counter_lists: list):
        """ Checks if the specified counter groups can be used. """
        assert self.cached_host != ""

        counter_lists_set = list()

        for x in counter_lists:
            if not x in counter_lists_set:
                counter_lists_set.append(x)
        for counter_list in counter_lists_set:
            sum_counters = 0
            for c in counter_list:
                try:
                    int_val = int(c, 16)
                    # Integer values are not checked - they get a pass (which might be wrong)
                    sum_counters += 1
                    continue
                except:
                    pass
                try:
                    sum_counters += self.preset_cost[c]
                except:
                    # This should only happen with Native Events
                    print(
                        "check_counters failed with reason: Unknown/unsupported event code specified: %s"
                        % c)
                    return False
            if sum_counters > self.num_hw_counters:
                print(
                    "check_counters failed with reason: Not enough hardware counters to support specified events: "
                    + str(counter_lists))
                return False
        return True


class PerfPAPIInfoStatic:
    info = PerfPAPIInfo()
