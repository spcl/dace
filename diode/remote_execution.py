import os
import sys
import stat
import dace
import pickle
import tempfile
import traceback
import subprocess
import dace.types
from string import Template
from dace.codegen.compiler import generate_program_folder
from dace.config import Config
from dace.codegen.instrumentation.perfsettings import PerfSettings, PerfUtils, PerfMetaInfo, PerfMetaInfoStatic, PerfPAPIInfoStatic


class Executor:
    """ Remote DaCe program execution management class for DIODE. """

    def __init__(self, perfplot, headless, sdfg_renderer, async_host=None):
        self.counter = 0
        self.perfplot = perfplot
        self.headless = headless
        self.rendered_graph = sdfg_renderer

        self.running_async = async_host != None
        self.async_host = async_host

    def run(self, dace_state, fail_on_nonzero=False):
        dace_progname = dace_state.get_sdfg().name
        code_objects = dace_state.get_generated_code()

        # Figure out whether we should use MPI for launching
        use_mpi = False
        for code_object in code_objects:
            if code_object.target.target_name == 'mpi':
                use_mpi = True
                break

        # Check validity of at least the default for now
        if PerfSettings.perf_enable_instrumentation(
        ) and PerfSettings.perf_enable_counter_sanity_check():
            if self.running_async:
                # Add information about what is being run
                self.async_host.notify("Reading remote PAPI Counters")
            PerfPAPIInfoStatic.info.load_info()
            # TODO: Should iterate over all nodes to find counter overrides
            papi_counters_valid = PerfPAPIInfoStatic.info.check_counters(
                [PerfSettings.perf_default_papi_counters()])
            if (not papi_counters_valid):
                print("Stopped execution. Counter settings do not meet "
                      "requirements")
                if self.running_async:
                    # Add information about what is being run
                    self.async_host.notify(
                        "An error occurred when reading remote PAPI counters")
                return

            if self.running_async:
                # Add information about what is being run
                self.async_host.notify("Done reading remote PAPI Counters")

        remote_workdir = Config.get("execution", "general", "workdir")
        remote_dace_dir = remote_workdir + "/.dacecache/%s/" % dace_progname
        self.show_output("Executing DaCe program " + dace_progname + " on " + \
                Config.get("execution", "general", "host") + "\n")

        if PerfSettings.perf_enable_instrumentation():
            if self.running_async:
                # Add information about what is being run
                self.async_host.notify("Checking remote PAPI Counters")
            PerfUtils.read_available_perfcounters()
            if self.running_async:
                # Add information about what is being run
                self.async_host.notify("Done checking remote PAPI Counters")

        try:
            if self.running_async:
                # Add information about what is being run
                self.async_host.notify("Generating remote workspace")
            tmpfolder = tempfile.mkdtemp()
            generate_program_folder(code_objects, tmpfolder)
            self.create_remote_directory(remote_dace_dir)
            self.copy_folder_to_remote(tmpfolder, remote_dace_dir)

            if self.running_async:
                # Add information about what is being run
                self.async_host.notify("Compiling...")
            # call compile.py on the remote node in the copied folder
            self.remote_compile(remote_dace_dir, dace_progname)

            if self.running_async:
                # Add information about what is being run
                self.async_host.notify("Done compiling")

            # copy the input file and the fatso (with the right name)
            # to remote_dace_dir
            so_name = "lib" + dace_progname + "." + Config.get(
                'compiler', 'library_extension')
            self.copy_file_from_remote(remote_dace_dir + "/build/" + so_name,
                                       tmpfolder + "/" + so_name)
            self.copy_file_to_remote(tmpfolder + "/" + so_name,
                                     remote_dace_dir)

            dace_file = dace_state.get_dace_tmpfile()
            if dace_file is None:
                raise ValueError("Dace file is None!")

            # copy the SDFG
            try:
                local_sdfg = tmpfolder + "/sdfg.out"
                sdfg = dace_state.get_sdfg()
                with open(local_sdfg, 'wb') as f:
                    pickle.dump(sdfg, f, pickle.HIGHEST_PROTOCOL)
                    remote_sdfg = remote_workdir + "/sdfg.out"
                    self.copy_file_to_remote(local_sdfg, remote_sdfg)
            except:
                print(
                    "Could NOT pickle the SDFG! This is bad for Matlab/Tensorflow generated SDFGs!"
                )

            remote_dace_file = remote_workdir + "/" + os.path.basename(
                dace_file)
            self.copy_file_to_remote(dace_file, remote_dace_file)

            if self.running_async:
                # Add information about what is being run
                self.async_host.notify("All files copied to remote")

            # We got the file there, now we can run with different
            # configurations.
            for iteration in range(0, PerfSettings.perf_multirun_num()):
                omp_thread_num = None
                if (PerfSettings.perf_multirun_num() != 1):
                    opt, val = PerfSettings.perf_multirun_options()[iteration]
                    if (opt == "omp_num_threads"):
                        omp_thread_num = val

                if self.running_async:
                    # Add information about what is being run
                    self.async_host.notify("Running option threads=" +
                                           str(omp_thread_num))

                self.remote_exec_dace(
                    remote_workdir,
                    remote_dace_file,
                    use_mpi,
                    fail_on_nonzero,
                    omp_num_threads=omp_thread_num)

                if self.running_async:
                    # Add information about what is being run
                    self.async_host.notify("Done option threads=" +
                                           str(omp_thread_num))

            self.show_output("Execution Terminated\n")

            try:
                self.copy_file_from_remote(remote_workdir + "/results.log",
                                           ".")
            except:
                pass

            # Copy back the vectorization results
            if PerfSettings.perf_enable_vectorization_analysis():
                if self.running_async:
                    self.async_host.notify("Running vectorization check")

                self.copy_file_from_remote(
                    remote_dace_dir + "/build/vecreport.txt", ".")
                with open("vecreport.txt") as r:
                    content = r.read()
                    print("Vecreport:")
                    print(content)

                    # Now analyze this...
                    for code_object in code_objects:
                        code_object.perf_meta_info.analyze(content)
                os.remove("vecreport.txt")

                if self.running_async:
                    self.async_host.notify("vectorization check done")

            # Copy back the instrumentation results
            if PerfSettings.perf_enable_instrumentation():
                if self.running_async:
                    # Add information about what is being run
                    self.async_host.notify("Analyzing performance data")
                try:
                    self.copy_file_from_remote(
                        remote_workdir + "/instrumentation_results.txt", ".")
                    self.remote_delete_file(remote_workdir +
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
                            PerfUtils.print_instrumentation_output(content)
                        else:
                            PerfUtils.print_instrumentation_output(ir)

                    os.remove("instrumentation_results.txt")
                except FileNotFoundError:
                    print(
                        "[Warning] Could not transmit instrumentation results")

                if self.running_async:
                    # Add information about what is being run
                    self.async_host.notify("Done Analyzing performance data")

            if self.running_async:
                # Add information about what is being run
                self.async_host.notify("Cleaning up")

            try:
                self.remote_delete_file(remote_workdir + "/results.log")
            except:
                pass

            self.remote_delete_file(remote_dace_file)
            self.remote_delete_dir(remote_dace_dir)

            try:
                res = self.update_performance_plot("results.log",
                                                   str(self.counter))
                os.remove("results.log")
            except FileNotFoundError:
                print("WARNING: results.log could not be read")

            if self.running_async:
                # Add information about what is being run
                self.async_host.notify("Done cleaning")

            # Also, update the performance data.
            self.rendered_graph.set_memspeed_target()
            self.rendered_graph.render_performance_data()
        except Exception as e:
            print("\n\n\n")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("Running the program failed:")
            traceback.print_exc()
            print(
                "Inspect above output for more information about executed command sequence."
            )
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            if self.headless:
                sys.exit(1)

        if self.running_async:
            self.async_host.notify("All done")
        self.counter += 1

    def update_performance_plot(self, resfile, name):
        # Each result.log will give us many runs of one size and optimization.
        # We ignore everything in the result log except the timing
        times = self.perfplot.parse_result_log(resfile)
        self.perfplot.add_run(name, times)
        self.perfplot.render()
        t = sorted([float(s) for s in times])
        print(t)
        return t[int(len(t) / 2)]

    def show_output(self, outstr):
        """ Displays output of any ongoing compilation or computation. """
        if isinstance(outstr, str):
            print(outstr, end="", flush=True)
            return
        sys.stdout.buffer.write(outstr)

    def remote_delete_file(self, delfile):
        s = Template(Config.get("execution", "general", "execcmd"))
        cmd = s.substitute(
            host=Config.get("execution", "general", "host"),
            command="rm " + delfile)
        self.exec_cmd_and_show_output(cmd)

    def remote_delete_dir(self, deldir):
        s = Template(Config.get("execution", "general", "execcmd"))
        cmd = s.substitute(
            host=Config.get("execution", "general", "host"),
            command="rm -r " + deldir)
        self.exec_cmd_and_show_output(cmd)

    def delete_local_folder(self, path):
        os.removedirs(path)

    def remote_exec_dace(self,
                         remote_workdir,
                         dace_file,
                         use_mpi=True,
                         fail_on_nonzero=False,
                         omp_num_threads=None):
        run = "${command} "
        if use_mpi == True:
            run = Config.get("execution", "mpi", "mpiexec")
            nprocs = Config.get("execution", "mpi", "num_procs")
        else:
            nprocs = 1
        repetitions = Config.get("execution", "general", "repetitions")

        omp_num_threads_str = ""
        omp_num_threads_unset_str = ""
        perf_instrumentation_result_marker = ""
        if (omp_num_threads != None):
            omp_num_threads_str = "export OMP_NUM_THREADS=" + str(
                omp_num_threads) + "\n"
            omp_num_threads_unset_str = "unset OMP_NUM_THREADS\n"
            perf_instrumentation_result_marker = "echo '# ;%s; Running in multirun config' >> %s/instrumentation_results.txt\n" % (
                omp_num_threads_str.replace("\n", ""), remote_workdir)

        # Create a startscript which exports necessary env-vars
        start_sh = "set -x\n" + \
                   "export DACE_compiler_use_cache=1\n" + \
                   "export DACE_optimizer_interface=''\n" + \
                   "export DACE_profiling=1\n" + \
                   "export DACE_treps=" + str(repetitions) +"\n" + \
                   omp_num_threads_str + \
                   "cd " + remote_workdir + "\n" + \
                   perf_instrumentation_result_marker
        s = Template(run + " ")
        cmd = s.substitute(command="python3 " + dace_file, num_procs=nprocs)
        start_sh += cmd + "\n"
        start_sh += "export RETVAL=$?\n"
        start_sh += (
            "unset DACE_compiler_use_cache\n" +
            "unset DACE_optimizer_interface\n" + "unset DACE_treps\n" +
            "unset DACE_profiling\n" + omp_num_threads_unset_str +
            # TODO: separate program error and system error
            "exit $RETVAL\n")
        tempdir = tempfile.mkdtemp()
        startsh_file = os.path.join(tempdir, "start.sh")
        fh = open(startsh_file, "w")
        fh.write(start_sh)
        fh.close()
        st = os.stat(startsh_file)
        os.chmod(startsh_file, st.st_mode | stat.S_IEXEC)

        workdir = Config.get("execution", "general", "workdir")

        self.copy_file_to_remote(
            startsh_file,
            Config.get("execution", "general", "workdir") + "/start.sh")

        s = Template(Config.get("execution", "general", "execcmd"))
        cmd = s.substitute(
            host=Config.get("execution", "general", "host"),
            command=workdir + "/start.sh")
        self.exec_cmd_and_show_output(cmd, fail_on_nonzero)

        self.remote_delete_file(workdir + "/start.sh")

    def remote_compile(self, rem_path, dace_progname):
        compile_cmd = "python3 -m dace.codegen.compiler " + str(
            rem_path) + " " + dace_progname
        s = Template(Config.get("execution", "general", "execcmd"))
        cmd = s.substitute(
            host=Config.get("execution", "general", "host"),
            command=compile_cmd)
        self.exec_cmd_and_show_output(cmd)

    def create_remote_directory(self, path):
        """ Creates a path on a remote node.

            @note: We use `mkdir -p` for now, which is not portable.
        """
        mkdircmd = "mkdir -p " + path
        s = Template(Config.get("execution", "general", "execcmd"))
        cmd = s.substitute(
            host=Config.get("execution", "general", "host"), command=mkdircmd)
        self.exec_cmd_and_show_output(cmd)

    def copy_file_to_remote(self, src, dst):
        s = Template(Config.get("execution", "general", "copycmd_l2r"))
        cmd = s.substitute(
            host=Config.get("execution", "general", "host"),
            srcfile=src,
            dstfile=dst)
        self.exec_cmd_and_show_output(cmd)

    def copy_folder_to_remote(self, src, dst):
        for root, subdirs, files in os.walk(src):
            for filename in files:
                file_path = os.path.join(root, filename)
                self.copy_file_to_remote(file_path, dst + "/" + filename)
            for subdir in subdirs:
                self.create_remote_directory(dst + "/" + str(subdir))
                self.copy_folder_to_remote(src + "/" + str(subdir),
                                           dst + "/" + str(subdir))
            return

    def copy_file_from_remote(self, src, dst):
        s = Template(Config.get("execution", "general", "copycmd_r2l"))
        cmd = s.substitute(
            host=Config.get("execution", "general", "host"),
            srcfile=src,
            dstfile=dst)
        self.exec_cmd_and_show_output(cmd)

    def exec_cmd_and_show_output(self, cmd, fail_on_nonzero=True):
        self.show_output(cmd + "\n")
        p = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        while True:
            out = p.stdout.read(1)
            if out == '' or out == b'':
                break
            if out != '' and out != b'':
                self.show_output(out)
        stdout, _ = p.communicate(timeout=60)
        self.show_output(stdout)
        if p.returncode != 0 and fail_on_nonzero:
            print("The command " + cmd + " failed (retcode " +\
                    str(p.returncode) + ")!\n")
            if self.headless:
                os._exit(p.returncode)
            else:
                raise ValueError("The command " + cmd + " failed (retcode " + \
                         str(p.returncode) + ")!")


import threading, queue


class AsyncExecutor:
    """ Asynchronous remote execution. """

    def __init__(self, perfplot, headless, sdfg_renderer, diode):

        self.executor = Executor(perfplot, headless, sdfg_renderer, self)
        self.to_thread_message_queue = queue.Queue(128)
        self.from_thread_message_queue = queue.Queue(128)
        self.diode = diode
        self.running_thread = None

    def notify(self, message):

        import time

        print("Got message " + str(message))

        def deferred():

            status_text = self.diode.builder.get_object("run_status_text")
            status_progress_bar = self.diode.builder.get_object("run_status")
            status_text.set_text(message)

        from gi.repository import GObject
        GObject.idle_add(deferred)

        if (message == "All done"):
            self.to_thread_message_queue.put("quit")

        time.sleep(0.001)  # Equivalent of `sched_yield()` for Python

    def run_async(self, dace_state, fail_on_nonzero=False):
        if self.running_thread != None and self.running_thread.is_alive():
            print("Cannot start another thread!")
            return

        def task():
            self.run()

        self.running_thread = threading.Thread(target=task)
        self.running_thread.start()
        self.to_thread_message_queue.put(("run", dace_state, fail_on_nonzero))

    def callMethod(self, obj, name, *args):
        return getattr(obj, name)(*args)

    def run(self):
        while True:
            # Read a message (blocking)
            msg = self.to_thread_message_queue.get()
            if (msg == "quit"):
                break

            # Unwrap and call
            ret = self.callMethod(self.executor, *msg)

            # Put the return value (including the complete command)
            self.from_thread_message_queue.put(("retval", ret, *msg))

    def join(self, timeout=None):
        pass
