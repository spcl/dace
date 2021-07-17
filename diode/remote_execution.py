# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import multiprocessing
import os
import sys
import stat
import tempfile
import traceback
import subprocess
import runpy
from typing import List, Callable, Any, AnyStr
from string import Template
from dace.sdfg import SDFG
from dace.codegen.compiler import generate_program_folder, configure_and_compile
from dace.codegen.codegen import CodeObject
from dace.config import Config


def _task(obj):
    obj.run()


class FunctionStreamWrapper(object):
    """ Class that wraps around a function with a stream-like API (write). """
    def __init__(self, *funcs: Callable[[AnyStr], Any]):
        self.funcs = funcs

    def write(self, *args, **kwargs):
        for func in self.funcs:
            func(' '.join(args), **kwargs)

    def flush(self):
        pass


def _output_feeder(terminal: multiprocessing.Queue, output: AnyStr):
    if isinstance(output, str):
        # It's already in a usable format
        pass
    else:
        try:
            output = output.decode('utf-8')
        except UnicodeDecodeError:
            # Try again escaping
            output = output.decode('unicode_escape')
    terminal.put(output)


class Executor(object):
    """ DaCe program execution management class for DIODE. """
    def __init__(self, remote, async_host=None):
        self.counter = 0
        self.remote = remote
        self.exit_on_error = True

        self.running_async = async_host is not None
        self.async_host = async_host

        self._config = None

        self.output_queue = None

    def set_exit_on_error(self, do_exit):
        self.exit_on_error = do_exit

    def set_config(self, config):
        self._config = config

    def config_get(self, *key_hierarchy):
        if self._config is None:
            return Config.get(*key_hierarchy)
        else:
            return self._config.get(*key_hierarchy)

    @staticmethod
    def _use_mpi(code_objects: List[CodeObject]):
        # Figure out whether we should use MPI for launching
        for code_object in code_objects:
            if code_object.target.target_name == 'mpi':
                return True
        return False

    def run(self, dace_state, fail_on_nonzero=False):
        sdfg = dace_state.get_sdfg()

        if self.remote:
            self.show_output("Executing DaCe program " + sdfg.name + " on " +
                             self.config_get("execution", "general", "host") +
                             "\n")
            self.run_remote(sdfg, dace_state, fail_on_nonzero)
        else:
            self.show_output("Executing DaCe program " + sdfg.name +
                             " locally\n")
            self.run_local(sdfg, dace_state.get_dace_tmpfile())

    def run_local(self, sdfg: SDFG, driver_file: str):
        workdir = sdfg.build_folder
        if Config.get_bool('diode', 'general', 'library_autoexpand'):
            sdfg.expand_library_nodes()
        code_objects = sdfg.generate_code()
        use_mpi = Executor._use_mpi(code_objects)
        # TODO: Implement (instead of pyrun, use mpirun/mpiexec)
        if use_mpi:
            raise NotImplementedError('Running MPI locally unimplemented')

        # Pipe stdout/stderr back to client output
        stdout = sys.stdout
        stderr = sys.stderr
        sys.stdout = FunctionStreamWrapper(self.show_output, stdout.write)
        sys.stderr = FunctionStreamWrapper(self.show_output, stderr.write)

        # Compile SDFG
        generate_program_folder(sdfg, code_objects, workdir, self._config)
        configure_and_compile(workdir, sdfg.name)

        self.show_output("Running script\n")

        # Run driver script with the compiled SDFG(s) as the default
        old_usecache = Config.get_bool('compiler', 'use_cache')
        Config.set('compiler', 'use_cache', value=True)
        try:
            runpy.run_path(driver_file, run_name='__main__')
        # Catching all exceptions, including SystemExit
        except (Exception, SystemExit) as ex:
            # Corner case: If exited with error code 0, it is a success
            if isinstance(ex, SystemExit):
                # If the exit code is nonzero, "raise" will not trigger a
                # printout on the server
                if ex.code != 0:
                    traceback.print_exc()
                    raise
            else:
                raise

        self.show_output("Execution Terminated\n")

        # Revert configuration and output redirection
        Config.set('compiler', 'use_cache', value=old_usecache)
        sys.stdout = stdout
        sys.stderr = stderr

    def run_remote(self, sdfg: SDFG, dace_state, fail_on_nonzero: bool):
        dace_progname = sdfg.name
        code_objects = sdfg.generate_code()
        use_mpi = Executor._use_mpi(code_objects)
        remote_workdir = self.config_get("execution", "general", "workdir")
        remote_base_path = self.config_get('default_build_folder')
        remote_dace_dir = os.path.join(remote_workdir, remote_base_path,
                                       dace_progname)

        try:
            tmpfolder = tempfile.mkdtemp()
            generate_program_folder(sdfg,
                                    code_objects,
                                    tmpfolder,
                                    config=self._config)
            self.create_remote_directory(remote_dace_dir)
            self.copy_folder_to_remote(tmpfolder, remote_dace_dir)

            # call compile.py on the remote node in the copied folder
            self.remote_compile(remote_dace_dir, dace_progname)

            # copy the input file and the .so file (with the right name)
            # to remote_dace_dir
            so_name = "lib" + dace_progname + "." + self.config_get(
                'compiler', 'library_extension')
            self.copy_file_from_remote(
                os.path.join(remote_dace_dir, 'build', so_name),
                os.path.join(tmpfolder, so_name))
            self.copy_file_to_remote(os.path.join(tmpfolder, so_name),
                                     remote_dace_dir)

            dace_file = dace_state.get_dace_tmpfile()
            if dace_file is None:
                raise ValueError("Dace file is None!")

            remote_dace_file = os.path.join(remote_workdir,
                                            os.path.basename(dace_file))
            self.copy_file_to_remote(dace_file, remote_dace_file)

            self.remote_exec_dace(remote_workdir,
                                  remote_dace_file,
                                  use_mpi,
                                  fail_on_nonzero,
                                  repetitions=dace_state.repetitions)

            self.show_output("Execution Terminated\n")

            try:
                self.copy_file_from_remote(remote_workdir + "/results.log",
                                           ".")
            except RuntimeError:
                pass

            # Copy back the instrumentation and vectorization results
            try:
                self.copy_folder_from_remote(
                    os.path.join(remote_dace_dir, 'perf'), ".")
            except RuntimeError:
                pass

            try:
                self.remote_delete_file(remote_workdir + "/results.log")
            except RuntimeError:
                pass

            self.remote_delete_file(remote_dace_file)
            self.remote_delete_dir(remote_dace_dir)
        except:  # Running a custom script (the driver file), which can raise
            # any exception
            self.show_output(traceback.format_exc())
            raise

        self.counter += 1

    def show_output(self, outstr):
        """ Displays output of any ongoing compilation or computation. """

        if self.output_queue is not None:
            # Pipe the output
            _output_feeder(self.output_queue, outstr)
            return

        if isinstance(outstr, str):
            print(outstr, end="", flush=True)
            return
        sys.stdout.buffer.write(outstr)

    def remote_delete_file(self, delfile):
        s = Template(self.config_get("execution", "general", "execcmd"))
        cmd = s.substitute(host=self.config_get("execution", "general",
                                                "host"),
                           command="rm " + delfile)
        self.exec_cmd_and_show_output(cmd)

    def remote_delete_dir(self, deldir):
        s = Template(self.config_get("execution", "general", "execcmd"))
        cmd = s.substitute(host=self.config_get("execution", "general",
                                                "host"),
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
                         additional_options_dict=None,
                         repetitions=None):
        additional_options_dict = additional_options_dict or {}
        run = "${command} "
        if use_mpi == True:
            run = self.config_get("execution", "mpi", "mpiexec")
            nprocs = self.config_get("execution", "mpi", "num_procs")
        else:
            nprocs = 1

        repetitions = (repetitions or self.config_get("execution", "general",
                                                      "repetitions"))

        omp_num_threads_str = ""
        omp_num_threads_unset_str = ""
        perf_instrumentation_result_marker = ""
        if omp_num_threads is not None:
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
        cmd = s.substitute(host=self.config_get("execution", "general",
                                                "host"),
                           command=workdir + "/start.sh")
        self.exec_cmd_and_show_output(cmd, fail_on_nonzero)

        self.remote_delete_file(workdir + "/start.sh")

    def remote_compile(self, rem_path, dace_progname):
        compile_cmd = "python3 -m dace.codegen.compiler " + str(
            rem_path) + " " + dace_progname
        s = Template(self.config_get("execution", "general", "execcmd"))
        cmd = s.substitute(host=self.config_get("execution", "general",
                                                "host"),
                           command=compile_cmd)
        self.exec_cmd_and_show_output(cmd)

    def create_remote_directory(self, path):
        """ Creates a path on a remote node.

            @note: We use `mkdir -p` for now, which is not portable.
        """
        mkdircmd = "mkdir -p " + path
        s = Template(self.config_get("execution", "general", "execcmd"))
        cmd = s.substitute(host=self.config_get("execution", "general",
                                                "host"),
                           command=mkdircmd)
        self.exec_cmd_and_show_output(cmd)

    def copy_file_to_remote(self, src, dst):
        s = Template(self.config_get("execution", "general", "copycmd_l2r"))
        cmd = s.substitute(host=self.config_get("execution", "general",
                                                "host"),
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

    def copy_folder_from_remote(self, src: str, dst: str):
        s = Template(self.config_get("execution", "general", "copycmd_r2l"))
        cmd = s.substitute(host=self.config_get("execution", "general",
                                                "host"),
                           srcfile="-r " + src,
                           dstfile=dst)
        self.exec_cmd_and_show_output(cmd)

    def copy_file_from_remote(self, src, dst):
        s = Template(self.config_get("execution", "general", "copycmd_r2l"))
        cmd = s.substitute(host=self.config_get("execution", "general",
                                                "host"),
                           srcfile=src,
                           dstfile=dst)
        self.exec_cmd_and_show_output(cmd)

    def exec_cmd_and_show_output(self, cmd, fail_on_nonzero=True):
        self.show_output(cmd + "\n")
        p = subprocess.Popen(cmd,
                             shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)

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
            if self.exit_on_error:
                os._exit(p.returncode)
            else:
                raise RuntimeError("The command " + cmd + " failed (retcode " + \
                         str(p.returncode) + ")!")


class AsyncExecutor:
    """ Asynchronous remote execution. """
    def __init__(self, remote):
        self.executor = Executor(remote)
        self.executor.set_exit_on_error(False)
        self.to_proc_message_queue = multiprocessing.Queue(128)
        self.running_proc = None

        # This determines if a "quit"-message stops the subprocess
        self.autoquit = True

        self.sync_run_lock = multiprocessing.Lock()

    def run_sync(self, func):

        # Synchronize using a lock
        def deferred():
            with self.sync_run_lock:
                func()
            return False

        deferred()

    def run_async(self, dace_state, fail_on_nonzero=False):
        if self.running_proc is not None and self.running_proc.is_alive():
            print("Cannot start another sub-process!")
            return

        # Use multiple processes to handle crashing processes
        self.running_proc = multiprocessing.Process(target=_task,
                                                    args=(self, ))
        self.running_proc.start()

        self.append_run_async(dace_state, fail_on_nonzero=False)

    def append_run_async(self, dace_state, fail_on_nonzero=False):
        self.to_proc_message_queue.put(
            ("run", (dace_state.dace_code, dace_state.dace_filename,
                     dace_state.source_code, dace_state.sdfg.to_json(),
                     dace_state.remote), fail_on_nonzero))

    def add_async_task(self, task):
        self.to_proc_message_queue.put(("execute_task", self, task))

    def execute_task(self, task):
        return task()

    def callMethod(self, obj, name, *args):
        # Shortcut for executing a simple task
        if name == "execute_task":
            _, subargs = args

            return self.execute_task(subargs)
        elif name == "run":
            # Convert arguments back to dace_state, deserializing the SDFG
            from diode.DaceState import DaceState
            dace_state = DaceState(args[0][0], args[0][1], args[0][2],
                                   SDFG.from_json(args[0][3]), args[0][4])
            args = (dace_state, *args[1:])

        return getattr(obj, name)(*args)

    def run(self):
        while True:
            # Read a message (blocking)
            msg = self.to_proc_message_queue.get()
            if msg == "quit":
                if self.to_proc_message_queue.empty() and self.autoquit:
                    print("Quitting async execution")
                    break
                else:
                    # There still is some queued work.
                    continue
            if msg == "forcequit":
                break

            # Unwrap and call
            self.callMethod(self.executor, *msg)

    def join(self, timeout=None):
        pass
