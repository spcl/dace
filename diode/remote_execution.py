import os
import sys
import stat
import tempfile
import traceback
import subprocess
from string import Template
from dace.codegen.compiler import generate_program_folder
from dace.config import Config
from dace.codegen.instrumentation.papi import PAPISettings, PAPIUtils


class Executor:
    """ Remote DaCe program execution management class for DIODE. """

    def __init__(self, perfplot, headless, sdfg_renderer, async_host=None):
        self.counter = 0
        self.perfplot = perfplot
        self.headless = headless
        self.exit_on_error = self.headless
        self.rendered_graphs = sdfg_renderer

        self.running_async = async_host is not None
        self.async_host = async_host

        self._config = None

        self.output_generator = None

    def setExitOnError(self, do_exit):
        self.exit_on_error = do_exit

    def setConfig(self, config):
        self._config = config

    def config_get(self, *key_hierarchy):
        if self._config is None:
            return Config.get(*key_hierarchy)
        else:
            return self._config.get(*key_hierarchy)

    def run(self, dace_state, fail_on_nonzero=False):
        dace_progname = dace_state.get_sdfg().name
        code_objects = dace_state.get_generated_code()

        # Figure out whether we should use MPI for launching
        use_mpi = False
        for code_object in code_objects:
            if code_object.target.target_name == 'mpi':
                use_mpi = True
                break

        # Check counter validity
        PAPIUtils.check_performance_counters(self)

        remote_workdir = self.config_get("execution", "general", "workdir")
        remote_dace_dir = remote_workdir + "/.dacecache/%s/" % dace_progname
        self.show_output("Executing DaCe program " + dace_progname + " on " + \
                self.config_get("execution", "general", "host") + "\n")

        try:
            if self.running_async:
                # Add information about what is being run
                self.async_host.notify("Generating remote workspace")
            tmpfolder = tempfile.mkdtemp()
            generate_program_folder(
                dace_state.get_sdfg(),
                code_objects,
                tmpfolder,
                config=self._config)
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

            # copy the input file and the .so file (with the right name)
            # to remote_dace_dir
            so_name = "lib" + dace_progname + "." + self.config_get(
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
                sdfg.save(local_sdfg)
                remote_sdfg = remote_workdir + "/sdfg.out"
                self.copy_file_to_remote(local_sdfg, remote_sdfg)
            except:
                print("Could NOT save the SDFG")

            remote_dace_file = remote_workdir + "/" + os.path.basename(
                dace_file)
            self.copy_file_to_remote(dace_file, remote_dace_file)

            if self.running_async:
                # Add information about what is being run
                self.async_host.notify("All files copied to remote")

            papi = PAPIUtils.is_papi_used(sdfg)


            # We got the file there, now we can run with different
            # configurations.
            if papi:
                multirun_num = PAPISettings.perf_multirun_num(config=self._config)
                for iteration in range(multirun_num):
                    optdict, omp_thread_num = PAPIUtils.get_run_options(
                        self, iteration)

                    self.remote_exec_dace(
                        remote_workdir,
                        remote_dace_file,
                        use_mpi,
                        fail_on_nonzero,
                        omp_num_threads=omp_thread_num,
                        additional_options_dict=optdict)

                    if self.running_async:
                        # Add information about what is being run
                        self.async_host.notify("Done option threads=" +
                                               str(omp_thread_num))
            else:
                self.remote_exec_dace(
                    remote_workdir,
                    remote_dace_file,
                    use_mpi,
                    fail_on_nonzero)
                
            self.show_output("Execution Terminated\n")

            try:
                self.copy_file_from_remote(remote_workdir + "/results.log",
                                           ".")
            except:
                pass

            if papi:
                # Copy back the vectorization results
                PAPIUtils.retrieve_vectorization_report(self, code_objects,
                                                        remote_dace_dir)

                # Copy back the instrumentation results
                PAPIUtils.retrieve_instrumentation_results(self, remote_workdir)

            if self.running_async:
                # Add information about what is being run
                self.async_host.notify("Cleaning up")

            try:
                self.remote_delete_file(remote_workdir + "/results.log")
            except:
                print(
                    "WARNING: results.log could not be transmitted (probably not created)"
                )

            self.remote_delete_file(remote_dace_file)
            self.remote_delete_dir(remote_dace_dir)

            def deferred():
                try:
                    res = self.update_performance_plot("results.log",
                                                       str(self.counter))
                    os.remove("results.log")
                except FileNotFoundError:
                    print("WARNING: results.log could not be read")

            if not self.headless or self.perfplot is None:
                if self.running_async and not self.headless:
                    self.async_host.run_sync(deferred)
                else:
                    deferred()

            if self.running_async:
                # Add information about what is being run
                self.async_host.notify("Done cleaning")

            # Update the performance data.
            if self.rendered_graphs is not None:
                self.rendered_graphs.set_memspeed_target()
                self.rendered_graphs.render_performance_data(
                    self.config_get("instrumentation", "papi_mode"))
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

        # If no perfplot is set, write it to the output as text with a prefix
        if self.perfplot is None:
            import re
            with open(resfile) as f:
                data = f.read()
            p = re.compile('\s(\d+\.\d+)$', re.MULTILINE)
            times = p.findall(data)
            self.show_output("\n~#~#" + str(times))
        else:
            times = self.perfplot.parse_result_log(resfile)
            self.perfplot.add_run(name, times)
            self.perfplot.render()
        t = sorted([float(s) for s in times])
        print(t)
        return t[int(len(t) / 2)]

    def show_output(self, outstr):
        """ Displays output of any ongoing compilation or computation. """

        if self.output_generator is not None:
            # Pipe the output
            self.output_generator(outstr)
            return

        if isinstance(outstr, str):
            print(outstr, end="", flush=True)
            return
        sys.stdout.buffer.write(outstr)

    def remote_delete_file(self, delfile):
        s = Template(self.config_get("execution", "general", "execcmd"))
        cmd = s.substitute(
            host=self.config_get("execution", "general", "host"),
            command="rm " + delfile)
        self.exec_cmd_and_show_output(cmd)

    def remote_delete_dir(self, deldir):
        s = Template(self.config_get("execution", "general", "execcmd"))
        cmd = s.substitute(
            host=self.config_get("execution", "general", "host"),
            command="rm -r " + deldir)
        self.exec_cmd_and_show_output(cmd)

    def delete_local_folder(self, path):
        os.removedirs(path)

    def remote_exec_dace(self,
                         remote_workdir,
                         dace_file,
                         use_mpi=True,
                         fail_on_nonzero=False,
                         omp_num_threads=None,
                         additional_options_dict={}):
        run = "${command} "
        if use_mpi == True:
            run = self.config_get("execution", "mpi", "mpiexec")
            nprocs = self.config_get("execution", "mpi", "num_procs")
        else:
            nprocs = 1
        repetitions = self.config_get("execution", "general", "repetitions")

        omp_num_threads_str = ""
        omp_num_threads_unset_str = ""
        perf_instrumentation_result_marker = ""
        if (omp_num_threads is not None):
            omp_num_threads_str = "export OMP_NUM_THREADS=" + str(
                omp_num_threads) + "\n"
            omp_num_threads_unset_str = "unset OMP_NUM_THREADS\n"
            perf_instrumentation_result_marker = "echo '# ;%s; Running in multirun config' >> %s/instrumentation_results.txt\n" % (
                omp_num_threads_str.replace("\n", ""), remote_workdir)

        # Create string from all misc options
        miscoptstring = ""
        miscoptresetstring = ""
        for optkey, optval in additional_options_dict.items():
            miscoptstring += "export " + str(optkey) + "=" + str(optval) + "\n"
            miscoptresetstring += "unset " + str(optkey) + "\n"

        # Create a startscript which exports necessary env-vars

        start_sh = "set -x\n" + \
                   "export DACE_compiler_use_cache=1\n" + \
                   "export DACE_optimizer_interface=''\n" + \
                   "export DACE_profiling=1\n" + \
                   "export DACE_treps=" + str(repetitions) +"\n" + \
                   miscoptstring + \
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
            miscoptresetstring +
            # TODO: separate program error and system error
            "exit $RETVAL\n")
        tempdir = tempfile.mkdtemp()
        startsh_file = os.path.join(tempdir, "start.sh")
        fh = open(startsh_file, "w")
        fh.write(start_sh)
        fh.close()
        st = os.stat(startsh_file)
        os.chmod(startsh_file, st.st_mode | stat.S_IEXEC)

        workdir = self.config_get("execution", "general", "workdir")

        self.copy_file_to_remote(
            startsh_file,
            self.config_get("execution", "general", "workdir") + "/start.sh")

        s = Template(self.config_get("execution", "general", "execcmd"))
        cmd = s.substitute(
            host=self.config_get("execution", "general", "host"),
            command=workdir + "/start.sh")
        self.exec_cmd_and_show_output(cmd, fail_on_nonzero)

        self.remote_delete_file(workdir + "/start.sh")

    def remote_compile(self, rem_path, dace_progname):
        compile_cmd = "python3 -m dace.codegen.compiler " + str(
            rem_path) + " " + dace_progname
        s = Template(self.config_get("execution", "general", "execcmd"))
        cmd = s.substitute(
            host=self.config_get("execution", "general", "host"),
            command=compile_cmd)
        self.exec_cmd_and_show_output(cmd)

    def create_remote_directory(self, path):
        """ Creates a path on a remote node.

            @note: We use `mkdir -p` for now, which is not portable.
        """
        mkdircmd = "mkdir -p " + path
        s = Template(self.config_get("execution", "general", "execcmd"))
        cmd = s.substitute(
            host=self.config_get("execution", "general", "host"),
            command=mkdircmd)
        self.exec_cmd_and_show_output(cmd)

    def copy_file_to_remote(self, src, dst):
        s = Template(self.config_get("execution", "general", "copycmd_l2r"))
        cmd = s.substitute(
            host=self.config_get("execution", "general", "host"),
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
        s = Template(self.config_get("execution", "general", "copycmd_r2l"))
        cmd = s.substitute(
            host=self.config_get("execution", "general", "host"),
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
            if self.headless and self.exit_on_error:
                os._exit(p.returncode)
            else:
                raise ValueError("The command " + cmd + " failed (retcode " + \
                         str(p.returncode) + ")!")


import threading, queue


class AsyncExecutor:
    """ Asynchronous remote execution. """

    def __init__(self, perfplot, headless, sdfg_renderer, diode):

        self.executor = Executor(perfplot, headless, sdfg_renderer, self)
        self.executor.setExitOnError(False)
        self.to_thread_message_queue = queue.Queue(128)
        self.from_thread_message_queue = queue.Queue(128)
        self.diode = diode
        self.running_thread = None
        self.autoquit = True  # This determines if a "quit"-message stops the thread

        self.sync_run_lock = threading.Lock()

    def counter_issue(self):
        self.diode.onCounterIssue()

    def run_sync(self, func):

        # Synchronize using a lock
        def deferred():
            with self.sync_run_lock:
                func()
            return False

        from gi.repository import GObject
        GObject.idle_add(deferred)

    def notify(self, message):

        if self.diode is None:
            return

        import time

        print("Got message " + str(message))

        def deferred():

            status_text = self.diode.builder.get_object("run_status_text")
            status_progress_bar = self.diode.builder.get_object("run_status")
            status_text.set_text(message)
            return False

        from gi.repository import GObject
        GObject.idle_add(deferred)

        if (message == "All done"):
            self.to_thread_message_queue.put("quit")

        time.sleep(0.001)  # Equivalent of `sched_yield()` for Python

    def run_async(self, dace_state, fail_on_nonzero=False):
        if self.running_thread is not None and self.running_thread.is_alive():
            print("Cannot start another thread!")
            return

        def task():
            self.run()

        self.running_thread = threading.Thread(target=task)
        self.running_thread.start()

        self.append_run_async(dace_state, fail_on_nonzero=False)

    def append_run_async(self, dace_state, fail_on_nonzero=False):
        self.to_thread_message_queue.put(("run", dace_state, fail_on_nonzero))

    def add_async_task(self, task):
        self.to_thread_message_queue.put(("execute_task", self, task))

    def execute_task(self, task):
        return task()

    def callMethod(self, obj, name, *args):
        # Shortcut for executing a simple task
        if name == "execute_task":
            _, subargs = args

            return self.execute_task(subargs)
        return getattr(obj, name)(*args)

    def run(self):
        while True:
            # Read a message (blocking)
            msg = self.to_thread_message_queue.get()
            if msg == "quit":
                if self.to_thread_message_queue.empty() and self.autoquit:
                    print("Quitting async execution")
                    break
                else:
                    # There still is some queued work.
                    continue
            if msg == "forcequit":
                break

            # Unwrap and call
            ret = self.callMethod(self.executor, *msg)

            # Put the return value (including the complete command)
            self.from_thread_message_queue.put(("retval", ret, *msg))

    def join(self, timeout=None):
        pass
