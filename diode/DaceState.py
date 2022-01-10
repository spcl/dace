# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" State of DaCe program in DIODE. """
import os
import re
import sys
import copy
import dace
import tempfile
import traceback
from six import StringIO

from dace import dtypes
from dace.transformation import optimizer
from dace.sdfg import SDFG
from dace.frontend.python import parser


class DaceState:
    """ This class abstracts the DaCe implementation from the GUI.
        It accepts a string of DaCe code and compiles it, giving access to
        the SDFG and the generated code, as well as the matching
        transformations.
    
        DaCe requires the code to be in a file (for code inspection), but 
        while the user types in the GUI we do not have the data available in a 
        file. Thus we create a temporary directory and save it there. However,
        the user might check for the filename in the code, thus we provide the
        original file name in argv[0].
    """

    # TODO: rewrite this class to use in-memory code.

    def __init__(self, dace_code, fake_fname, source_code=None, sdfg=None, remote=False):

        self.compiled = False
        self.dace_tmpfile = None
        self.dace_filename = os.path.basename(fake_fname)
        self.sdfg = sdfg  # This is the toplevel one
        self.sdfgs = []  # This is a collection of all the SDFGs
        self.generated_code = []
        self.generated_code_files = None
        self.matching_patterns = []
        self.dace_code = dace_code
        self.source_code = source_code
        self.remote = remote
        self.repetitions = None
        self.errors = []  # Any errors that arise from compilation are placed here to show
        # them once the sdfg is rendered

        self.has_multiple_eligible_sdfgs = False

        if self.sdfg is not None:
            self.compiled = True
            self.sdfgs = [('deserialized', self.sdfg)]

        tempdir = tempfile.mkdtemp()
        self.dace_tmpfile = os.path.join(tempdir, self.dace_filename)
        fh = open(self.dace_tmpfile, "wb")
        fh.write(self.dace_code.encode('utf-8'))
        fh.close()

        # Create SDFG unless we already have one
        if self.sdfg is None and self.dace_code != "":
            saved_argv = sys.argv
            sys.argv = [self.dace_filename]
            gen_module = {}
            code = compile(self.dace_code, self.dace_tmpfile, 'exec')
            try:
                exec(code, gen_module)
            except Exception as ex:
                self.errors.append(ex)

            # Find dace programs
            self.sdfgs = [(name, obj.parse()) for name, obj in gen_module.items()
                          if isinstance(obj, parser.DaceProgram)]
            self.sdfgs += [(name, obj) for name, obj in gen_module.items() if isinstance(obj, SDFG)]
            try:
                self.sdfg = self.sdfgs[0][1]
            except IndexError:
                if len(self.errors) > 0:
                    raise self.errors[-1]
                if len(self.sdfgs) == 0:
                    raise ValueError('No SDFGs found in file. SDFGs are only '
                                     'recognized when @dace.programs or SDFG '
                                     'objects are found in the global scope')
                raise
            if len(self.sdfg) > 1:
                self.has_multiple_eligible_sdfgs = True

    def get_arg_initializers(self):
        sdfg = self.sdfg
        if sdfg is None:
            raise ValueError("Need an SDFG to produce initializers")
        data = set()
        for state in sdfg.nodes():
            data.update(set((n.data, n.desc(sdfg)) for n in state.nodes() if isinstance(n, dace.sdfg.nodes.AccessNode)))

        sym_args = list(sdfg.symbols.keys())
        data_args = [d for d in data if not d[1].transient]

        initializer = ""
        for d in data_args:
            initializer += str(d[0]) + " = dace.ndarray([" + \
            ", ".join([str(x) for x in list(d[1].shape)]) + "], " + \
            "dtype=dace." + d[1].dtype.to_string() + ")\n"

        return initializer

    def get_call_args(self):
        sdfg = self.sdfg
        if sdfg is None:
            raise ValueError("Need an SDFG to produce call arguments")
        data = set()
        for state in sdfg.nodes():
            data.update(set((n.data, n.desc(sdfg)) for n in state.nodes() if isinstance(n, dace.sdfg.nodes.AccessNode)))

        sym_args = list(sdfg.symbols.keys())
        data_args = [d for d in data if not d[1].transient]

        call_args = []
        for d in data_args:
            call_args.append(d[0])

        return call_args

    def compile(self):
        try:
            if dace.Config.get_bool('diode', 'general', 'library_autoexpand'):
                self.sdfg.expand_library_nodes()

            self.sdfg.validate()
            code = self.sdfg.generate_code()
            self.generated_code = code
            self.compiled = True
        # TODO: SDFG validation errors should be treated separately
        #except dace.sdfg.InvalidSDFGError:
        except:
            exstr = StringIO()
            formatted_lines = traceback.format_exc().splitlines()
            exstr.write("Compilation failed:\n%s\n\n" % formatted_lines[-1])
            traceback.print_exc(file=exstr)
            self.generated_code = exstr.getvalue()
            print("Codegen failed!\n" + str(self.generated_code))

    def get_dace_generated_files(self):
        """ Writes the generated code to a temporary file and returns the file
            name. Compiles the code if not already compiled. """
        tempdir = tempfile.mkdtemp()
        self.generated_code_files = []

        for codeobj in self.generated_code:
            name = codeobj.name
            extension = codeobj.language
            gencodefile = os.path.join(tempdir, '%s.%s' % (name, extension))

            with open(gencodefile, "w") as fh:
                # Clear location indicators from code
                clean_code = codeobj.clean_code
                fh.write(clean_code)

            self.generated_code_files.append(gencodefile)

        return self.generated_code_files

    def get_dace_tmpfile(self):
        """ Returns the current temporary path to the generated code files. """
        return self.dace_tmpfile

    def get_dace_fake_fname(self):
        """ Returns the original filename of the DaCe program, i.e., the name 
            of the file he stored to, before performing modifications in the 
            editor """
        return self.dace_filename

    def set_is_compiled(self, state):
        self.compiled = state

    def get_dace_code(self):
        return self.dace_code

    def get_generated_code(self):
        if self.compiled == False:
            self.compile()
        return self.generated_code

    def get_sdfg(self):
        if self.compiled == False:
            self.compile()
        return self.sdfgs[0][1]

    def set_sdfg(self, sdfg, name="Main SDFG"):
        self.sdfgs = [(name, sdfg)]
        self.sdfg = sdfg
        self.compiled = False
        if self.compiled == False:
            self.compile()

    def get_sdfgs(self):
        """ Returns the current set of SDFGs in the workspace.
            @rtype: Tuples of (name, SDFG).
        """
        if self.compiled == False:
            self.compile()
        return self.sdfgs
